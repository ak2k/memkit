"""Unit tests for the recall hook. No search subprocess is ever spawned.

Run: `pytest tests/test_memory_prompt_recall.py -q`

The hook is importable as a package module AND runnable as a loose file, and
both matter: the harness invokes the file, everything else imports the module.
Importing only defines constants/functions (main() is __main__-guarded).

Retrieval is one stage — SQLite FTS5, in this process — and it is driven for
real over tmp corpora: index built, queried, mutated, corrupted. There is no
subprocess and no stdout contract to pin. (The semantic stage this file once
tested through a captured embedding sample is gone, and the external index it
called could not have been driven the way FTS5 is here in any case.)

What the suite covers, roughly in the order the hook does it:

  - engine: index build and refresh, query construction, compound splitting,
    corrupted and missing databases, and the fail-open path around each.
  - floors and ranking: the evidence thresholds that decide eligibility, the
    common-word gate, and MAX_HITS truncation on top of them.
  - gates: the prompt classes that never reach retrieval at all, and — as
    load-bearing as any of them — the near misses that must NOT be gated,
    a relayed teammate message above all.
  - delivery integrity: the ordering of deliver, spend and record under a
    SIGTERM mask. These spawn the hook as a real subprocess with real signals
    and real closed pipes, because the property is about what survives a kill
    and an in-process fake can assert an ordering the shipped path does not
    have.
  - CLI and log shape: the record fields the analyzers read.
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from memkit import memory_prompt_recall as hook

# The hook AS A FILE — what the harness runs, and what the subprocess cases
# below exercise. Package import and file execution are two different entry
# points into the same source, and the delivery-integrity cases only mean
# anything against the second.
HOOK = hook.__file__

# Session ids below are LITERALS on purpose, and short ones. A downstream
# soak-log analyzer separates real records from harness ones by the SHAPE of
# the session field, so a fixture id that could pass for a real one would
# quietly land in somebody's measured numbers. The pin for that is
# test_no_fixture_session_id_can_pass_for_a_real_one at the end of this file.


# Where a fixture corpus lives under a redirected HOME. Two stores, because
# the interleave and the per-store gating only have anything to say with more
# than one; nothing here knows or cares that a real deployment has two as well.
PROJECT_DIR = "store/project"
PERSONAL_DIR = "store/personal"


def _write_config(home: Path, *, gate_root: str | None = None) -> Path:
    """Write a fixture config under `home` and return its path.

    Roots are `~`-rooted and expanded when the config is READ, which is the
    whole reason the subprocess cases can redirect an entire corpus with
    HOME=tmp_path and nothing else. Pre-resolving them here would work in this
    process and silently score the developer's real stores in the child.
    """
    config = {
        "schema": hook.SCHEMA,
        "roots": {"home": {"kind": "path", "path": "~"}},
        "stores": [
            {
                "id": "project",
                "role": "project",
                "dir": PROJECT_DIR,
                "live_root": "home",
                **({"cwd_gate": {"root": gate_root}} if gate_root else {}),
            },
            {
                "id": "personal",
                "role": "personal",
                "dir": PERSONAL_DIR,
                "live_root": "home",
            },
        ],
        "search_cli": "memory-recall --search",
    }
    path = home / "memkit.json"
    path.write_text(json.dumps(config))
    return path


def _env(tmp_path: Path, *, stores: bool = True) -> dict:
    """Environment for a spawned hook: a redirected HOME and a config that
    points at it. Both are needed — without the config the hook is inert by
    design and every injection case below would pass vacuously.

    `stores=False` writes the config and NOT the directories, which is how a
    case reaches `gate:nodirs` on purpose: configured stores that are not on
    disk are dropped, so the hook gets past every prompt-shaped gate and then
    finds nothing to search. That is the outcome that proves a prompt was not
    gated for its shape.
    """
    if stores:
        for rel in (PROJECT_DIR, PERSONAL_DIR):
            (tmp_path / rel / "search").mkdir(parents=True, exist_ok=True)
    return dict(
        os.environ,
        HOME=str(tmp_path),
        MEMKIT_CONFIG=str(_write_config(tmp_path)),
    )


# --- _interleave -------------------------------------------------------------


def test_interleave_round_robin_rank_order() -> None:
    assert hook._interleave([["a1", "a2"], ["b1", "b2"]]) == ["a1", "b1", "a2", "b2"]


def test_interleave_first_list_wins_ties() -> None:
    # Caller passes most-specific dir first; its rank-1 must lead.
    assert hook._interleave([["x"], ["y"]])[0] == "x"


def test_interleave_dedups_and_handles_uneven_and_empty() -> None:
    assert hook._interleave([["a", "b"], ["b", "c", "d"]]) == ["a", "b", "c", "d"]
    assert hook._interleave([[], ["only"]]) == ["only"]
    assert hook._interleave([]) == []


# --- _excluded ---------------------------------------------------------------


def test_excluded_ledgers_subindexes_archive() -> None:
    assert hook._excluded("/m/MEMORY.md")
    assert hook._excluded("/m/SEARCH.md")
    assert hook._excluded("/m/search/domain/INDEX.md")
    assert hook._excluded("/m/archive/old_gotcha.md")
    assert not hook._excluded("/m/search/unionfs_perms.md")


def test_excluded_hot_tier() -> None:
    # Hot memories are already in context via MEMORY.md; a pointer to one
    # spends the injection budget on something the model has read.
    assert hook._excluded("/m/hot/taskdb.md")
    assert not hook._excluded("/m/search/domain/ledgersvc_extraction.md")


# --- _search_root: tiered store vs not ---------------------------------------


def test_search_root_prefers_the_search_subtree(tmp_path: Path) -> None:
    store = tmp_path / "memories"
    (store / "search" / "domain").mkdir(parents=True)
    (store / "hot").mkdir()
    # Rooting the search at search/ is what keeps hot files out of the
    # fixed 10-chunk candidate window, not just out of the output.
    assert hook._search_root(str(store)) == str(store / "search")


def test_search_root_falls_back_to_an_unmigrated_store(tmp_path: Path) -> None:
    store = tmp_path / "memories"
    store.mkdir()
    assert hook._search_root(str(store)) == str(store)


# --- FTS5 lexical stage (a real index, over a real tmp corpus) ---------------


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    """A search-tier store with the index redirected into tmp_path.

    _state_dir is where _fts_db puts the DB, so stubbing it is what keeps a
    test run from touching the operator's live ~/.cache/memory-recall index.
    """
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path / "state"))
    root = tmp_path / "memories" / "search"
    root.mkdir(parents=True)
    return root


def _memo(root: Path, relpath: str, body: str) -> str:
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nname: {p.stem}\ntype: reference\n---\n\n{body}\n")
    return str(p)


def _identity(root: Path) -> list[tuple]:
    """Every indexed chunk's (rowid, path, mtime_ns, ctime_ns, size).

    rowid is in here on purpose: unchanged identity triples alone would not
    prove a steady-state run skipped the work, because a delete+reinsert
    writes the SAME triple back. FTS5 hands out fresh rowids, so a stable
    rowid is the evidence that no row was rewritten.
    """
    con = hook._fts_connect(hook._fts_db(str(root)))
    try:
        return con.execute(
            "SELECT rowid, path, mtime_ns, ctime_ns, size FROM chunks ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()


def test_fts_ranks_the_full_overlap_first_and_floors_the_weak_tail(
    corpus: Path,
) -> None:
    strong = _memo(
        corpus,
        "unionfs_perms.md",
        "# unionfs FUSE permissions\n\nunionfs mounts with default_permissions,"
        " so a media service needs the media group as its PRIMARY group or its"
        " writes are denied.",
    )
    _memo(
        corpus,
        "unrelated_zfs.md",
        "# zfs pools\n\n"
        + "zfs datasets, snapshots and scrubs on the pool.\n" * 20
        + "\nThe media library lives on this pool.",
    )
    hits = hook._fts_dir("unionfs media group denied", str(corpus))
    # The one-common-term match scores ~0 against the full-overlap chunk, so
    # FLOOR_LEX drops it rather than letting it ride along as a second hit.
    assert hits == [strong]


def test_fts_floor_is_relative_to_the_best_chunk(corpus: Path, monkeypatch) -> None:
    # Named so that rank order and lexicographic order DISAGREE: with a.md/b.md
    # an implementation that sorted by path would satisfy the assertion below
    # without ranking anything.
    strong = _memo(
        corpus, "z_meshnet.md", "# meshnet\n\nmeshnet mesh across nodes"
    )
    weak = _memo(corpus, "a_nodes.md", "# nodes\n\nnodes inventory")
    monkeypatch.setattr(hook, "FLOOR_LEX", 0.0)
    # Ordered: the caller injects the first MAX_HITS of this list, so a stage
    # that returned the right files in the wrong order would quietly inject
    # the wrong ones.
    assert hook._fts_dir("meshnet nodes", str(corpus)) == [strong, weak]
    # At a floor of 1.0 only chunks tying the query's best rank survive —
    # the normalization is against this query's best hit, not an absolute.
    monkeypatch.setattr(hook, "FLOOR_LEX", 1.0)
    assert hook._fts_dir("meshnet nodes", str(corpus)) == [strong]


def test_the_index_scores_every_hit_it_returns(corpus: Path, monkeypatch) -> None:
    """The logged score has to be the number the floor was compared against.

    It is only ever read next to FLOOR_LEX — "was that pointer a close call"
    — so a score on some other scale, or missing for a hit that was returned,
    makes the whole key misleading rather than merely incomplete.
    """
    strong = _memo(
        corpus, "z_meshnet.md", "# meshnet\n\nmeshnet mesh across nodes"
    )
    weak = _memo(corpus, "a_nodes.md", "# nodes\n\nnodes inventory")
    # The channel accumulates across the dirs of ONE retrieval and is emptied
    # at the next retrieval's entry, so recall() owns the clear and a test
    # calling _fts_dir directly has to stand in for it.
    hook._LEX_SCORES.clear()
    monkeypatch.setattr(hook, "FLOOR_LEX", 0.0)
    hits = hook._fts_dir("meshnet nodes", str(corpus))

    assert hits == [strong, weak]
    assert set(hook._LEX_SCORES) == {strong, weak}
    # Top-normalized: the best chunk in the dir is 1.0 by construction, and
    # rank order and score order are the same order.
    assert hook._LEX_SCORES[strong] == 1.0
    assert 0.0 < hook._LEX_SCORES[weak] < 1.0
    assert hook._scores(hits) == [1.0, round(hook._LEX_SCORES[weak], 3)]
    # A path the index never scored reads as 0.0 rather than raising: the log
    # is written on a path that must not be able to fail.
    assert hook._scores(["/gone.md"]) == [0.0]

    # Floored hits leave no score behind — the key means "returned", and a
    # score for something that was dropped would be read as a pointer.
    hook._LEX_SCORES.clear()
    monkeypatch.setattr(hook, "FLOOR_LEX", 1.0)
    assert hook._fts_dir("meshnet nodes", str(corpus)) == [strong]
    assert weak not in hook._LEX_SCORES


def test_fts_window_holds_candidate_limit_files_not_chunks(corpus: Path) -> None:
    """One chatty file's sections must not consume the whole candidate window.

    The contract is inherited, not invented here: the retired semantic stage
    counted files too (ck's --limit does), and the window kept that meaning
    when FTS5 replaced it. Losing it is invisible in a small eval — retrieval
    still answers, just from a pool of three or four files instead of ten, and
    the files it drops are the ones the relevance floor would have ranked next.
    """
    chatty = _memo(
        corpus,
        "zfs_notes.md",
        "\n".join(f"## note {i}\n\nzfs scrub cadence\n" for i in range(12)),
    )
    others = [
        _memo(corpus, f"host_{i}.md", f"# host {i}\n\nzfs scrub cadence on host {i}")
        for i in range(12)
    ]
    hits = hook._fts_dir("zfs scrub cadence", str(corpus))
    # The chatty file's 12 short all-term sections outrank every other file, so
    # a window that counted chunks would return it and almost nothing else.
    assert hits[0] == chatty
    assert len(hits) == hook.CANDIDATE_LIMIT
    assert len(set(hits)) == len(hits)
    assert len(set(hits) & set(others)) == hook.CANDIDATE_LIMIT - 1


def test_fts_never_returns_ledgers_subindexes_hot_or_archive(corpus: Path) -> None:
    wanted = _memo(corpus, "netguard.md", "# netguard\n\nnetguard bouncer decisions")
    for rel in ("MEMORY.md", "SEARCH.md", "domain/INDEX.md"):
        _memo(corpus, rel, "# ledger\n\nnetguard bouncer decisions")
    for rel in ("hot/taskdb.md", "archive/old.md"):
        _memo(corpus, rel, "# excluded tier\n\nnetguard bouncer decisions")
    assert hook._fts_dir("netguard bouncer decisions", str(corpus)) == [wanted]


def test_fts_reindexes_a_changed_file_and_sweeps_a_deleted_one(corpus: Path) -> None:
    memo = Path(_memo(corpus, "topic.md", "# topic\n\nvaultwarden backup schedule"))
    assert hook._fts_dir("vaultwarden backup", str(corpus)) == [str(memo)]

    memo.write_text("---\nname: topic\n---\n\n# topic\n\nphotoprism indexing pass\n")
    assert hook._fts_dir("photoprism indexing", str(corpus)) == [str(memo)]
    assert hook._fts_dir("vaultwarden backup", str(corpus)) == []

    memo.unlink()
    # The sweep is the only GC this index has: without it a deleted memory
    # stays answerable forever, and the hook would point at a missing file.
    assert hook._fts_dir("photoprism indexing", str(corpus)) == []


def test_fts_reindexes_a_rewrite_that_keeps_the_byte_count(corpus: Path) -> None:
    memo = Path(_memo(corpus, "topic.md", "# topic\n\nvaultwarden backup schedule"))
    size = memo.stat().st_size
    assert hook._fts_dir("vaultwarden backup", str(corpus)) == [str(memo)]

    # Padded back to the original size so `size` cannot be what fires: an
    # edit that happens to preserve the byte count (a typo fix, a reworded
    # line) is caught by the mtime leg of the identity triple alone.
    body = "---\nname: topic\ntype: reference\n---\n\n# topic\n\nphotoprism pass"
    memo.write_text(body.ljust(size))
    assert memo.stat().st_size == size, "the size leg must not be what fires"
    assert hook._fts_dir("photoprism pass", str(corpus)) == [str(memo)]
    assert hook._fts_dir("vaultwarden backup", str(corpus)) == []


needs_permissions = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads mode-000 files, so nothing is unreadable"
)


def _count_identity_reads(monkeypatch) -> dict:
    """Count _fts_identity calls: one means the pre-lock comparison alone,
    i.e. the sync converged and never opened a write transaction."""
    counter = {"n": 0}
    real = hook._fts_identity

    def counted(con):
        counter["n"] += 1
        return real(con)

    monkeypatch.setattr(hook, "_fts_identity", counted)
    return counter


@needs_permissions
def test_fts_keeps_an_unreadable_files_rows(corpus: Path) -> None:
    readable = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    locked = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    locked.write_text(locked.read_text() + "\nborgmatic drives the pruning\n")
    locked.chmod(0o000)
    try:
        # Dropping its rows before the read would leave the index empty of it
        # and fail identically on the rebuild, so the whole dir would stop
        # answering until someone chmodded the file back.
        hits = hook._fts_dir("restic pruning", str(corpus))
        # ... and what survives is the OLD content: the edit made while the
        # file was unreadable is not in the index, so an assertion on the old
        # terms alone could not tell sparing apart from re-indexing.
        assert hook._fts_dir("borgmatic", str(corpus)) == []
    finally:
        locked.chmod(0o644)
    assert readable in hits
    assert str(locked) in hits  # answerable from the rows it already had
    assert hook._fts_dir("borgmatic", str(corpus)) == [str(locked)]


@needs_permissions
def test_fts_sweep_spares_an_unreadable_file_when_another_one_changes(
    corpus: Path,
) -> None:
    changing = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    locked = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    # Its identity has to move, or it is simply an up-to-date file nobody
    # needs to read and sparing is not what protects it.
    locked.write_text(locked.read_text() + "\nborgmatic drives the pruning\n")
    locked.chmod(0o000)
    # The unreadable file alone converges without a transaction, so the sweep
    # only ever sees it when something ELSE takes the write lock.
    changing.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")
    try:
        hits = hook._fts_dir("restic pruning", str(corpus))
        assert hook._fts_dir("borgmatic", str(corpus)) == []  # spared, not read
    finally:
        locked.chmod(0o644)
    # A file held out of the walk's conclusions because it could not be read
    # is not a file that was deleted, and the sweep cannot tell them apart on
    # its own — it only knows the path is not in `disk`.
    assert sorted(hits) == sorted([str(changing), str(locked)])


def test_fts_sweep_spares_a_file_whose_stat_fails_when_another_changes(
    corpus: Path, monkeypatch
) -> None:
    changing = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    opaque = _memo(corpus, "b.md", "# b\n\nrestic repository pruning")
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    _failing_stat(monkeypatch, {opaque})
    changing.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")
    hook._fts_dir("restic pruning", str(corpus))
    # The write lock is taken for the file that changed, and the sweep runs
    # inside it over everything the walk did not put in `disk` — which is
    # where a file it could not stat would be destroyed if sparing did not
    # reach the sweep as well as the comparison.
    assert {row[1] for row in _identity(corpus)} == {str(changing), opaque}


def test_fts_replans_under_the_lock_when_another_writer_got_there_first(
    corpus: Path, monkeypatch
) -> None:
    ahead = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    behind = Path(_memo(corpus, "b.md", "# b\n\nzrepl snapshot replication"))
    hook._fts_dir("restic pruning", str(corpus))
    ahead.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")
    behind.write_text("---\nname: b\n---\n\n# b\n\nzrepl replication tuning\n")

    real = hook._fts_identity
    calls = {"n": 0}
    racer_rows: list[int] = []

    def racing(con):
        snapshot = real(con)
        calls["n"] += 1
        if calls["n"] == 1:
            # Between this session's plan and its lock, another session takes
            # the lock and re-indexes one of the two changed files. The
            # snapshot just handed back is now stale by construction.
            other = hook._fts_connect(hook._fts_db(str(corpus)))
            try:
                st = ahead.stat()
                other.execute("DELETE FROM chunks WHERE path = ?", (str(ahead),))
                other.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    (
                        str(ahead),
                        st.st_mtime_ns,
                        st.st_ctime_ns,
                        st.st_size,
                        ahead.read_text(),
                    ),
                )
                other.commit()
            finally:
                other.close()
            racer_rows.extend(r[0] for r in _identity(corpus) if r[1] == str(ahead))
        return snapshot

    monkeypatch.setattr(hook, "_fts_identity", racing)
    assert hook._fts_dir("zrepl replication", str(corpus)) == [str(behind)]
    assert calls["n"] == 2, "the in-lock snapshot was never taken"
    # Planning from the pre-lock snapshot makes every waiter redo work the
    # winner already committed, which is how one changed memory turns into an
    # N-session re-index queue that outgrows busy_timeout.
    assert racer_rows
    assert [r[0] for r in _identity(corpus) if r[1] == str(ahead)] == racer_rows


@needs_permissions
def test_fts_converges_while_a_file_stays_unreadable(corpus: Path, monkeypatch) -> None:
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    locked = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    hook._fts_dir("restic pruning", str(corpus))
    locked.write_text(locked.read_text() + "\nmore restic pruning here\n")
    locked.chmod(0o000)
    try:
        hook._fts_dir("restic pruning", str(corpus))  # notices, spares it
        reads = _count_identity_reads(monkeypatch)
        hook._fts_dir("restic pruning", str(corpus))
        # A spared file's identity can never match, so leaving it in the
        # comparison would take the write lock on every prompt from here on —
        # for as long as the mode persists, against every session at once.
        assert reads["n"] == 1
    finally:
        locked.chmod(0o644)


@needs_permissions
def test_fts_partial_walk_spares_an_unreadable_subtree(
    corpus: Path, monkeypatch
) -> None:
    top = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    sub = _memo(corpus, "domain/b.md", "# b\n\nrestic pruning over there")
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    (corpus / "domain").chmod(0o000)
    try:
        assert hook._fts_dir("restic pruning", str(corpus)) == [top]
        # Unseen is not deleted. Sweeping on a walk that could not read the
        # subtree would drop every memory in it, and the rebuild would not
        # bring them back either.
        assert sub in {row[1] for row in _identity(corpus)}
        reads = _count_identity_reads(monkeypatch)
        hook._fts_dir("restic pruning", str(corpus))
        assert reads["n"] == 1, "an unreadable subtree must still converge"
    finally:
        (corpus / "domain").chmod(0o755)
    assert sorted(hook._fts_dir("restic pruning", str(corpus))) == sorted([top, sub])


@needs_permissions
def test_fts_cold_index_over_an_unreadable_corpus_errors_without_rebuilding(
    corpus: Path, monkeypatch
) -> None:
    locked = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    locked.chmod(0o000)
    calls: list[str] = []
    real = hook._fts_sync

    def counted(con, root):
        calls.append(root)
        return real(con, root)

    monkeypatch.setattr(hook, "_fts_sync", counted)
    try:
        # A cold index that could read nothing must not commit empty and then
        # read back as a healthy corpus with nothing to say — that is a silent
        # loss of the whole lexical stage, and [] would hide it from errs_lex.
        with pytest.raises(OSError):
            hook._fts_dir("restic pruning", str(corpus))
    finally:
        locked.chmod(0o644)
    # An OSError is not index damage, so it must not spend a rebuild: only
    # sqlite errors route to unlink-and-retry.
    assert len(calls) == 1
    assert hook._fts_dir("restic pruning", str(corpus)) == [str(locked)]


@needs_permissions
def test_fts_partial_walk_converges_after_a_deletion_it_cannot_sweep(
    corpus: Path, monkeypatch
) -> None:
    doomed = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    sub = _memo(corpus, "domain/b.md", "# b\n\nrestic pruning over there")
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    (corpus / "domain").chmod(0o000)
    doomed.unlink()
    try:
        hook._fts_dir("restic pruning", str(corpus))
        reads = _count_identity_reads(monkeypatch)
        hook._fts_dir("restic pruning", str(corpus))
        hook._fts_dir("restic pruning", str(corpus))
        # An incomplete walk never sweeps, so the deleted file's rows cannot
        # go anywhere — and if they stayed in the comparison, the mismatch
        # they cause is unresolvable: every prompt would take the write lock
        # for as long as the subtree stayed dark.
        assert reads["n"] == 2, "one identity read per invocation, no lock"
    finally:
        (corpus / "domain").chmod(0o755)
    # The first COMPLETE walk is what settles it: absence is finally evidence
    # of deletion, and the subtree comes back with its rows intact.
    assert hook._fts_dir("restic pruning", str(corpus)) == [sub]
    assert {row[1] for row in _identity(corpus)} == {sub}


@needs_permissions
def test_fts_partial_walk_does_not_sweep_a_racing_writers_rows(
    corpus: Path, monkeypatch
) -> None:
    changing = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    _memo(corpus, "domain/b.md", "# b\n\nrestic pruning over there")
    hook._fts_dir("restic pruning", str(corpus))
    (corpus / "domain").chmod(0o000)
    changing.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")
    late = str(corpus / "domain" / "c.md")

    real = hook._fts_identity
    calls = {"n": 0}

    def racing(con):
        snapshot = real(con)
        calls["n"] += 1
        if calls["n"] == 1:
            other = hook._fts_connect(hook._fts_db(str(corpus)))
            try:
                other.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    (late, 1, 1, 1, "restic pruning newly written"),
                )
                other.commit()
            finally:
                other.close()
        return snapshot

    monkeypatch.setattr(hook, "_fts_identity", racing)
    try:
        # The changed file takes the write lock, and the snapshot the sweep is
        # exempted from was read before the racing insert existed.
        hook._fts_dir("restic pruning", str(corpus))
    finally:
        (corpus / "domain").chmod(0o755)
    assert calls["n"] == 2, "the in-lock snapshot was never taken"
    # Sparing is computed from the PRE-lock snapshot, so it cannot cover a row
    # that did not exist when that snapshot was taken. What saves it is that the
    # sweep candidates are drawn from the same pre-lock snapshot, so this row was
    # never a candidate.
    assert late in {row[1] for row in _identity(corpus)}


def test_fts_complete_walk_does_not_sweep_a_row_written_after_the_walk(
    corpus: Path, monkeypatch
) -> None:
    """A COMPLETE walk is what enables the sweep, so it is where a stale walk
    does the damage: this session's walk ran before the file existed at all."""
    changing = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    hook._fts_dir("restic pruning", str(corpus))
    changing.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")
    late = str(corpus / "c.md")

    real = hook._fts_identity
    calls = {"n": 0}

    def racing(con):
        snapshot = real(con)
        calls["n"] += 1
        if calls["n"] == 1:
            # Between this session's identity read and its lock, another
            # session creates and indexes a file this walk never saw.
            other = hook._fts_connect(hook._fts_db(str(corpus)))
            try:
                other.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    (late, 1, 1, 1, "restic pruning newly written"),
                )
                other.commit()
            finally:
                other.close()
        return snapshot

    monkeypatch.setattr(hook, "_fts_identity", racing)
    hook._fts_dir("restic pruning", str(corpus))
    assert calls["n"] == 2, "the in-lock snapshot was never taken"
    # "My walk never saw it" says nothing about a file that came into being
    # after the walk, so it must not be a sweep candidate at all.
    assert late in {row[1] for row in _identity(corpus)}


def test_fts_sweep_spares_a_candidate_another_session_reindexed(
    corpus: Path, monkeypatch
) -> None:
    """The other half: the row WAS in this session's snapshot, and a racing
    session rewrote it under a fresher view than this walk had."""
    changing = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    gone = Path(_memo(corpus, "c.md", "# c\n\nrestic pruning elsewhere"))
    hook._fts_dir("restic pruning", str(corpus))
    gone.unlink()  # this walk sees it deleted, so it becomes a sweep candidate
    changing.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")

    real = hook._fts_identity
    calls = {"n": 0}

    def racing(con):
        snapshot = real(con)
        calls["n"] += 1
        if calls["n"] == 1:
            # It came back, and another session indexed the file that is now
            # there. This session's "it is gone" predates all of that.
            _memo(corpus, "c.md", "# c\n\nrestic pruning restored")
            other = hook._fts_connect(hook._fts_db(str(corpus)))
            try:
                other.execute(
                    "UPDATE chunks SET mtime_ns = 99 WHERE path = ?", (str(gone),)
                )
                other.commit()
            finally:
                other.close()
        return snapshot

    monkeypatch.setattr(hook, "_fts_identity", racing)
    hook._fts_dir("restic pruning", str(corpus))
    assert calls["n"] == 2, "the in-lock snapshot was never taken"
    assert str(gone) in {row[1] for row in _identity(corpus)}


def test_fts_reindexes_an_equal_size_rewrite_with_the_mtime_put_back(
    corpus: Path,
) -> None:
    """mtime and size are both forgeable; ctime is what makes them add up."""
    memo = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    assert hook._fts_dir("restic pruning", str(corpus)) == [str(memo)]
    before = memo.stat()

    replaced = memo.read_text().replace(
        "restic repository pruning", "zrepl send bandwidth zone"
    )
    assert len(replaced) == len(memo.read_text()), "the rewrite must be equal-sized"
    memo.write_text(replaced)
    os.utime(memo, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert memo.stat().st_mtime_ns == before.st_mtime_ns
    assert memo.stat().st_size == before.st_size

    assert hook._fts_dir("zrepl bandwidth", str(corpus)) == [str(memo)]
    assert hook._fts_dir("restic pruning", str(corpus)) == []


def test_fts_rebuilds_an_index_built_on_the_previous_schema(corpus: Path) -> None:
    """The upgrade path is the damage path, and this is the test that says so:
    if a missing column landed anywhere else, the old index would answer from
    stale rows forever instead of being rebuilt once."""
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    db = hook._fts_db(str(corpus))
    con = sqlite3.connect(db)
    try:
        con.execute(
            "CREATE VIRTUAL TABLE chunks USING"
            " fts5(path UNINDEXED, mtime_ns UNINDEXED, size UNINDEXED, text)"
        )
        con.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            ("/gone/stale.md", 1, 1, "restic pruning from the old schema"),
        )
        con.commit()
    finally:
        con.close()

    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
    assert {row[1] for row in _identity(corpus)} == {memo}


def test_fts_converges_when_a_file_persistently_fails_to_open(
    corpus: Path, monkeypatch
) -> None:
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    broken = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2
    # Its identity has to move, or an index that already holds it converges on
    # the triple alone and never tries the read this test is about.
    broken.write_text(broken.read_text() + "\nmore restic pruning here\n")

    real_open = open

    def failing(path, *args, **kwargs):
        if str(path) == str(broken):
            raise OSError(5, "Input/output error", str(broken))
        return real_open(path, *args, **kwargs)

    # EIO, a stale NFS handle, a sandbox that grants metadata and withholds
    # content: the permission bits say readable and the open disagrees, which
    # is why the scan has to probe by opening.
    monkeypatch.setattr(hook, "open", failing, raising=False)
    assert str(broken) in hook._fts_dir("restic pruning", str(corpus))
    reads = _count_identity_reads(monkeypatch)
    hook._fts_dir("restic pruning", str(corpus))
    hook._fts_dir("restic pruning", str(corpus))
    # A failure only the mid-sync handler catches is rediscovered from inside
    # the transaction every time, so it would take the write lock on every
    # prompt for as long as the condition lasted.
    assert reads["n"] == 2


def _failing_stat(monkeypatch, targets: set[str], errno: int = 5) -> None:
    """Make os.stat fail for `targets` and behave normally for everything else.

    Delegating rather than blanket-failing matters: os.stat is patched
    globally for the duration, and pathlib and os.path.exists route through
    it too.
    """
    real_stat = os.stat

    def failing(path, *args, **kwargs):
        if str(path) in targets:
            raise OSError(errno, "Input/output error", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hook.os, "stat", failing)


def test_fts_keeps_rows_of_a_file_whose_stat_fails(corpus: Path, monkeypatch) -> None:
    keep = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    opaque = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    _failing_stat(monkeypatch, {str(opaque)})
    hook._fts_dir("restic pruning", str(corpus))
    # A stat that fails for any reason other than ENOENT has not established
    # that the file is gone, and the walk is otherwise complete — so without
    # sparing, the sweep reads that silence as a deletion and destroys the
    # rows of a file sitting right there. (It is absent from the hits either
    # way while the mode lasts: the search filters on os.path.exists, which
    # is the same failing stat. What must survive is the index.)
    assert {row[1] for row in _identity(corpus)} == {keep, str(opaque)}
    reads = _count_identity_reads(monkeypatch)
    hook._fts_dir("restic pruning", str(corpus))
    assert reads["n"] == 1, "a spared file must not re-take the write lock"


def test_fts_cold_build_spares_a_file_whose_stat_fails(
    corpus: Path, monkeypatch
) -> None:
    readable = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    opaque = _memo(corpus, "b.md", "# b\n\nrestic repository pruning")
    _failing_stat(monkeypatch, {opaque})
    # Nothing is lost here — there was nothing indexed to lose — but the file
    # must not be quietly counted as absent either: the readable corpus indexes
    # and answers, and the one that could not be stat'd is simply not in it.
    assert hook._fts_dir("restic pruning", str(corpus)) == [readable]
    assert {row[1] for row in _identity(corpus)} == {readable}


def test_fts_cold_build_that_can_stat_nothing_raises(corpus: Path, monkeypatch) -> None:
    paths = {
        _memo(corpus, "a.md", "# a\n\nrestic repository pruning"),
        _memo(corpus, "b.md", "# b\n\nrestic repository pruning"),
    }
    _failing_stat(monkeypatch, paths)
    # An index that holds nothing because it could examine nothing must not
    # commit and then read back as a healthy corpus with nothing to say.
    with pytest.raises(OSError):
        hook._fts_dir("restic pruning", str(corpus))


def test_fts_converges_when_a_file_opens_but_never_reads(
    corpus: Path, monkeypatch
) -> None:
    keep = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    broken = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2
    broken.write_text(broken.read_text() + "\nborgmatic drives the pruning\n")

    real_open = open

    class Unreadable:
        """Opens fine, fails on the read — EIO on a bad block, an ESTALE
        handle. The probe cannot detect this; only the read can."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            raise OSError(5, "Input/output error", str(broken))

        def close(self):
            pass

    def flaky(path, *args, **kwargs):
        if str(path) == str(broken):
            return Unreadable()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(hook, "open", flaky, raising=False)
    hits = hook._fts_dir("restic pruning", str(corpus))
    assert sorted(hits) == sorted([keep, str(broken)])
    # Spared, not re-indexed: the edit made while it was unreadable is absent,
    # which is what tells sparing apart from a successful read.
    assert hook._fts_dir("borgmatic", str(corpus)) == []
    reads = _count_identity_reads(monkeypatch)
    hook._fts_dir("restic pruning", str(corpus))
    hook._fts_dir("restic pruning", str(corpus))
    # Classifying this only from inside the transaction rediscovers it every
    # prompt: the identity can never match, so the lock is taken forever.
    assert reads["n"] == 2


def test_fts_reads_a_changed_files_contents_once_per_sync(
    corpus: Path, monkeypatch
) -> None:
    changing = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    hook._fts_dir("restic pruning", str(corpus))
    changing.write_text("---\nname: a\n---\n\n# a\n\nrestic pruning schedule\n")

    real_open = open
    opens = {"n": 0}

    def counted(path, *args, **kwargs):
        if str(path) == str(changing):
            opens["n"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(hook, "open", counted, raising=False)
    assert hook._fts_dir("restic pruning schedule", str(corpus)) == [str(changing)]
    # Exactly one: the read whose contents get indexed. Reading again under the
    # lock would widen the window in which the file can change out from under
    # the identity it is about to be indexed under, and a second pass to
    # predict whether the read will work only predicts what the read reports.
    assert opens["n"] == 1


def test_fts_reads_under_the_lock_for_a_file_a_racer_made_stale(
    corpus: Path, monkeypatch
) -> None:
    settled = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    changing = Path(_memo(corpus, "b.md", "# b\n\nzrepl snapshot replication"))
    hook._fts_dir("restic pruning", str(corpus))
    changing.write_text("---\nname: b\n---\n\n# b\n\nzrepl replication tuning\n")

    real = hook._fts_identity
    calls = {"n": 0}

    def racing(con):
        snapshot = real(con)
        calls["n"] += 1
        if calls["n"] == 1:
            # Another session writes a.md's rows under a different identity
            # after this session has already decided a.md is up to date. Only
            # the in-lock snapshot can see it, so nothing was staged for it.
            other = hook._fts_connect(hook._fts_db(str(corpus)))
            try:
                other.execute(
                    "UPDATE chunks SET mtime_ns = 1, size = 1 WHERE path = ?",
                    (str(settled),),
                )
                other.commit()
            finally:
                other.close()
        return snapshot

    monkeypatch.setattr(hook, "_fts_identity", racing)
    hook._fts_dir("zrepl replication", str(corpus))
    assert calls["n"] == 2, "the in-lock snapshot was never taken"
    # Staging pre-lock is an optimisation over the reads it can predict, not a
    # replacement for reading: a path that only goes stale under the lock has
    # no staged content, and dropping it would leave a wrong identity indexed.
    st = settled.stat()
    assert {(r[2], r[3], r[4]) for r in _identity(corpus) if r[1] == str(settled)} == {
        (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    }


def test_fts_keeps_rows_when_the_under_lock_read_fails(
    corpus: Path, monkeypatch
) -> None:
    settled = Path(_memo(corpus, "a.md", "# a\n\nrestic repository pruning"))
    changing = Path(_memo(corpus, "b.md", "# b\n\nzrepl snapshot replication"))
    hook._fts_dir("restic pruning", str(corpus))
    changing.write_text("---\nname: b\n---\n\n# b\n\nzrepl replication tuning\n")

    real_open = open
    opens = {"n": 0}

    def flaky(path, *args, **kwargs):
        if str(path) == str(settled):
            opens["n"] += 1
            raise PermissionError(13, "Permission denied", str(settled))
        return real_open(path, *args, **kwargs)

    real = hook._fts_identity
    calls = {"n": 0}

    def racing(con):
        snapshot = real(con)
        calls["n"] += 1
        if calls["n"] == 1:
            other = hook._fts_connect(hook._fts_db(str(corpus)))
            try:
                other.execute(
                    "UPDATE chunks SET mtime_ns = 1, size = 1 WHERE path = ?",
                    (str(settled),),
                )
                other.commit()
            finally:
                other.close()
        return snapshot

    monkeypatch.setattr(hook, "open", flaky, raising=False)
    monkeypatch.setattr(hook, "_fts_identity", racing)
    hook._fts_dir("zrepl replication", str(corpus))
    assert calls["n"] == 2, "the in-lock snapshot was never taken"
    # a.md matches the pre-lock snapshot, so nothing reads it until the
    # backstop does. Without this the stub is inert and the test is vacuous.
    assert opens["n"] == 1, "the under-lock read never happened"
    # The backstop reads before it deletes. Deleting first would empty the
    # index of a file that is about to be readable again, and the rebuild
    # would fail the same way — so a moment's unreadability costs the memory.
    assert str(settled) in {row[1] for row in _identity(corpus)}
    assert hook._fts_dir("zrepl replication", str(corpus)) == [str(changing)]


def test_fts_steady_state_reads_the_identity_once(corpus: Path, monkeypatch) -> None:
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    hook._fts_dir("restic pruning", str(corpus))
    reads = _count_identity_reads(monkeypatch)
    hook._fts_dir("restic pruning", str(corpus))
    # The second read only happens inside the transaction, so one read is the
    # evidence that an unchanged corpus never asks for the write lock.
    assert reads["n"] == 1


def test_fts_does_not_unlink_an_index_another_writer_holds(corpus: Path) -> None:
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
    db = Path(hook._fts_db(str(corpus)))
    inode = db.stat().st_ino

    blocker = hook._fts_connect(str(db))
    blocker.execute("BEGIN IMMEDIATE")
    try:
        # Steady state wants no lock at all, so contention is invisible here.
        assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
        # With a sync actually pending, the write lock is unreachable. Losing
        # that race is contention, not corruption: answer from the index as
        # it stands rather than deleting a live writer's file.
        _memo(corpus, "b.md", "# b\n\nrestic pruning elsewhere")
        assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
    finally:
        blocker.rollback()
        blocker.close()
    assert db.stat().st_ino == inode, "the index was replaced under the writer"
    # Once the lock is gone the skipped sync simply happens next invocation.
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2


def _busy_error(db: Path) -> sqlite3.OperationalError:
    """A genuine SQLITE_BUSY exception; sqlite_errorcode is set by the C
    layer, so a hand-built OperationalError would not exercise the real
    branch."""
    holder = sqlite3.connect(db)
    holder.execute("BEGIN IMMEDIATE")
    other = sqlite3.connect(db)
    other.execute("PRAGMA busy_timeout=0")
    try:
        other.execute("BEGIN IMMEDIATE")
        raise AssertionError("expected the write lock to be held")
    except sqlite3.OperationalError as exc:
        return exc
    finally:
        other.close()
        holder.rollback()
        holder.close()


def test_fts_dir_reports_contention_rather_than_deleting_the_index(
    corpus: Path, monkeypatch
) -> None:
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
    db = Path(hook._fts_db(str(corpus)))
    busy = _busy_error(db)

    def contended(path: str):
        raise busy

    monkeypatch.setattr(hook, "_fts_connect", contended)

    # Contention reaching this layer (from the connect, not the sync) is
    # still not damage: raising is what makes _stage count errs_lex, where
    # [] would read as "this dir had nothing to say" — and the healthy index
    # has to survive, since unlinking it would strand a live writer.
    with pytest.raises(sqlite3.OperationalError):
        hook._fts_dir("restic pruning", str(corpus))
    assert db.exists()


def test_fts_dir_raises_when_the_rebuild_fails_too(corpus: Path, monkeypatch) -> None:
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")

    def broken(db: str):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(hook, "_fts_connect", broken)
    # Raising, not returning [], is what lets _stage tell "this dir is broken"
    # apart from "this dir had nothing" — errs_lex is the only place a dead
    # lexical stage is visible.
    with pytest.raises(sqlite3.DatabaseError):
        hook._fts_dir("restic pruning", str(corpus))


def test_fts_rebuilds_once_on_a_non_busy_sqlite_error(
    corpus: Path, monkeypatch
) -> None:
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
    db = Path(hook._fts_db(str(corpus)))
    inode = db.stat().st_ino
    real = hook._fts_sync
    calls: list[str] = []

    def once(con, root):
        calls.append(root)
        if len(calls) == 1:
            raise sqlite3.OperationalError("no such column: text")
        return real(con, root)

    monkeypatch.setattr(hook, "_fts_sync", once)
    # Schema drift from an older hook version is damage, not contention: the
    # index is thrown away and rebuilt once, and the stage still answers on
    # the same invocation rather than costing a prompt its pointers.
    #
    # Hold the doomed file open across all of it. st_ino only identifies a
    # file while the freed number cannot be handed straight back, and ext4
    # reallocates eagerly: the rebuilt index came back wearing the SAME inode
    # on GitHub's linux runners, reddening this assertion on a hook that had
    # done exactly what it should. (APFS never reuses, which is why every
    # darwin run — including the nix checks — passed.) An open descriptor
    # keeps the unlinked inode allocated, so the replacement is forced to get
    # a different one and the proxy becomes sound rather than lucky. The
    # sibling test above needs no such pin: its `blocker` connection is
    # already holding the file open for the same window.
    with db.open("rb"):
        assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
        assert len(calls) == 2, "the rebuild must be a single retry, not a loop"
        assert db.stat().st_ino != inode, "the damaged index was not replaced"


def test_fts_contention_on_an_empty_index_is_an_error_not_no_hits(
    corpus: Path,
) -> None:
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    db = hook._fts_db(str(corpus))
    blocker = hook._fts_connect(db)  # creates the index, holding nothing
    blocker.execute("BEGIN IMMEDIATE")
    try:
        # Losing the write-lock race is only survivable when there is an index
        # to fall back on. Answering [] from one that holds no rows would tell
        # the caller this corpus has nothing to say, on a corpus that does.
        with pytest.raises(sqlite3.OperationalError):
            hook._fts_dir("restic pruning", str(corpus))
    finally:
        blocker.rollback()
        blocker.close()


def test_fts_does_not_point_at_a_file_deleted_since_the_last_sweep(
    corpus: Path,
) -> None:
    keep = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    gone = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning"))
    assert len(hook._fts_dir("restic pruning", str(corpus))) == 2

    gone.unlink()
    blocker = hook._fts_connect(hook._fts_db(str(corpus)))
    blocker.execute("BEGIN IMMEDIATE")
    try:
        # The sweep cannot run while another session holds the write lock, so
        # the index is legitimately a sweep behind — and every hit becomes a
        # pointer the user is told to go read.
        assert hook._fts_dir("restic pruning", str(corpus)) == [keep]
    finally:
        blocker.rollback()
        blocker.close()


def test_fts_busy_falls_back_to_the_message_when_there_is_no_errorcode() -> None:
    # A hand-built OperationalError has no sqlite_errorcode at all (the C
    # layer sets it), which is also how a pre-3.11 interpreter looks — so this
    # branch is reachable, not dead code.
    assert not hasattr(sqlite3.OperationalError("x"), "sqlite_errorcode")
    assert hook._fts_busy(sqlite3.OperationalError("database is locked"))
    assert not hook._fts_busy(sqlite3.OperationalError("no such table: chunks"))
    # Class first: a corrupt DB reports DatabaseError, never contention.
    assert not hook._fts_busy(sqlite3.DatabaseError("database is locked"))


def test_fts_busy_masks_extended_result_codes() -> None:
    exc = sqlite3.OperationalError("database is locked")
    # SQLITE_BUSY_TIMEOUT: what a SETLK build reports contention as. Compared
    # unmasked it reads as an unknown code, i.e. damage — and the recovery for
    # damage is deleting an index another session is actively writing.
    exc.sqlite_errorcode = 773
    assert hook._fts_busy(exc)
    exc.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
    assert not hook._fts_busy(exc)


def test_fts_steady_state_rewrites_nothing(corpus: Path) -> None:
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    _memo(corpus, "b.md", "# b\n\nzrepl snapshot replication")
    hook._fts_dir("restic pruning", str(corpus))
    before = _identity(corpus)
    assert before, "index built nothing"
    hook._fts_dir("zrepl replication", str(corpus))
    assert _identity(corpus) == before


def test_fts_self_heals_a_corrupt_index(corpus: Path) -> None:
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    db = Path(hook._fts_db(str(corpus)))
    db.write_bytes(b"this is not a database" * 100)
    # One unlink-and-rebuild is the whole recovery story, so a corrupt cache
    # costs a rebuild, never an answer.
    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]
    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]


def test_fts_missing_or_empty_root_and_empty_query(corpus: Path, tmp_path) -> None:
    assert hook._fts_dir("restic pruning", str(tmp_path / "absent")) == []
    assert hook._fts_dir("restic pruning", str(corpus)) == []  # exists, no files
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    assert hook._fts_dir("", str(corpus)) == []


def test_recall_isolates_a_failing_lex_dir(monkeypatch) -> None:
    # One corpus failing must not discard the other's results, and must not
    # read as "searched, found nothing" — errs_lex is the only place that
    # difference shows up, and a mistyped key leaves every other test green
    # while the soak log lies.
    monkeypatch.setattr(hook, "_search_dirs", lambda: ["/project", "/personal"])

    def fts(query: str, d: str) -> list[str]:
        if d == "/project":
            raise sqlite3.DatabaseError("index would not rebuild")
        return [f"{d}/search/lex.md"]

    monkeypatch.setattr(hook, "_fts_dir", fts)
    rec: dict = {}
    hits = hook.recall("unionfs media group denied writes", stats=rec)

    assert rec["errs_lex"] == 1
    assert rec["lex_hits"] == 1  # the healthy dir still contributes
    assert hits == ["/personal/search/lex.md"]


@needs_permissions
def test_recall_records_files_spared_and_dirs_the_walk_could_not_enter(
    corpus: Path, monkeypatch
) -> None:
    """A stage answering from PART of the corpus is otherwise invisible.

    errs_lex stays 0 through all of this, correctly — the stage did not fail,
    it just had less to search — so in the soak log a corpus half of which
    cannot be read is indistinguishable from one with nothing to say. These
    keys are the difference.
    """
    good = _memo(corpus, "a.md", "# a\n\nrestic repository pruning policy")
    locked = Path(_memo(corpus, "b.md", "# b\n\nrestic repository pruning policy"))
    locked.chmod(0o000)
    _memo(corpus, "domain/c.md", "# c\n\nrestic repository pruning policy")
    (corpus / "domain").chmod(0o000)
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])
    rec: dict = {}
    try:
        assert hook.recall("restic repository pruning policy", stats=rec) == [good]
    finally:
        locked.chmod(0o644)
        (corpus / "domain").chmod(0o755)
    assert rec["errs_lex"] == 0
    assert rec["lex_spared"] == 1  # the mode-000 file, held out of the sweep
    assert rec["lex_unwalked"] == 1  # the mode-000 subtree


def test_recall_records_a_sync_skipped_by_contention(corpus: Path, monkeypatch) -> None:
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])
    clean: dict = {}
    assert hook.recall("restic repository pruning", stats=clean) == [memo]
    # Absent rather than zero on a healthy run: a key that appears on every
    # line of the soak log is a key nobody greps for.
    assert not {"lex_spared", "lex_unwalked", "lex_busy_skip"} & clean.keys()

    busy = _busy_error(Path(hook._fts_db(str(corpus))))

    def contended(con, root):
        raise busy

    monkeypatch.setattr(hook, "_fts_sync", contended)
    rec: dict = {}
    assert hook.recall("restic repository pruning", stats=rec) == [memo]
    # Deliberately not an error — the index answered, one edit stale at worst
    # — so errs_lex is not where this shows up, and the swallow was invisible
    # until this key existed.
    assert rec["errs_lex"] == 0
    assert rec["lex_busy_skip"] == 1


def test_recall_logs_the_built_query_for_the_shadow_harness(
    corpus: Path, monkeypatch
) -> None:
    """The offline shadow harness replays this field through an embedder to
    decide whether the deleted semantic stage ever comes back, so the field is
    that harness's entire corpus and a gap in it is unrecoverable after the
    fact. It has been deleted once already — as collateral of the LEX_THIN
    branch that used to gate it — and no test noticed.
    """
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])

    prompt = "what is the restic repository pruning policy"
    rec: dict = {}
    hook.recall(prompt, stats=rec)
    # The BUILT query, not the raw prompt: what sem must be replayed against is
    # the terms the lexical stage actually searched for.
    assert rec["query"] == hook.build_query(prompt)
    assert "what" not in rec["query"]

    # Capped, so one pathological prompt cannot bloat every future log line.
    long = " ".join(f"qq{chr(97 + i // 26)}{chr(97 + i % 26)}xx" for i in range(60))
    full = hook.build_query(long)
    # build_query answers None on a prompt with nothing content-bearing in it,
    # which 60 invented nouns are not.
    assert full is not None
    assert len(full) > 160, "the cap is not exercised — this test proves nothing"
    wide: dict = {}
    hook.recall(long, stats=wide)
    assert wide["query"] == full[:160]


def test_recall_records_an_index_it_had_to_rebuild(corpus: Path, monkeypatch) -> None:
    """Self-healing is silent by design, which is the problem.

    A corrupt index costs an unlink and a cold rebuild and then answers
    correctly, so errs_lex stays 0 and the caller cannot tell. A cache being
    destroyed and rebuilt on EVERY prompt therefore reads exactly like a
    healthy one — same answers, ~40x the latency — until this key exists.
    """
    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    Path(hook._fts_db(str(corpus))).write_bytes(b"this is not a database" * 100)
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])

    rec: dict = {}
    assert hook.recall("restic repository pruning", stats=rec) == [memo]
    assert rec["errs_lex"] == 0
    assert rec["lex_rebuilds"] == 1

    # The cold FIRST build of an index is not damage and must not be counted,
    # or every fresh machine looks like a corruption incident.
    clean: dict = {}
    assert hook.recall("restic repository pruning", stats=clean) == [memo]
    assert "lex_rebuilds" not in clean


def test_lex_hits_name_the_section_that_matched(corpus: Path, monkeypatch) -> None:
    """Which PART of a 400-line memory matched is what makes the pointer
    actionable — otherwise the agent is told to read the file and has to
    re-run the search in its head to find the paragraph."""
    # FOUR chunks match the query and only one is the answer, so a label taken
    # from the first row, the last row, or either lexicographic extreme names
    # a different section — the file is built that way on purpose.
    memo = _memo(
        corpus,
        "taskdb.md",
        "# Taskdb\n\nthe ledgerdb-backed task tracker"
        "\n\n## Alpha ordering\n\nledgerdb sorts ids lexically, " + "padding " * 20 + ""
        "\n\n## Delta compaction\n\nledgerdb gc reclaims the table files"
        "\n\n## Zulu identity\n\nexternal_ref names a ledgerdb row, " + "padding " * 20,
    )
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])
    assert hook.recall("ledgerdb compaction reclaims", stats={}) == [memo]

    # The label comes from the best-RANKED chunk, which rests on sqlite's
    # bare-column rule under a lone min() — a rule that, if it did not hold
    # for fts5, would hand back an arbitrary chunk of the right file and read
    # as plausible every time.
    assert hook._LEX_SECTIONS[memo] == "Delta compaction"
    assert hook._pointer_line(memo, ["ledgerdb", "compaction"], 3).endswith(
        " [section: Delta compaction]"
    )


def test_no_section_for_a_frontmatter_hit(corpus: Path, monkeypatch) -> None:
    """A pointer can legitimately have no section, which must render as
    silence rather than an empty tag."""
    memo = _memo(corpus, "restic_pruning_policy.md", "# Elsewhere\n\nnothing to see")
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])
    # The match is in the preamble — frontmatter and the description line —
    # which is the file's own summary, not a place inside the document.
    assert hook.recall("restic pruning policy", stats={}) == [memo]
    assert memo not in hook._LEX_SECTIONS
    assert "[section:" not in hook._pointer_line(memo, ["restic", "pruning"], 3)


def test_section_label_is_the_heading_text_capped(corpus: Path) -> None:
    assert hook._section_label("## Delta compaction\n\nbody") == "Delta compaction"
    assert hook._section_label("---\ndescription: d\n---") == ""
    assert hook._section_label("plain prose with no heading") == ""
    long = hook._section_label("### " + "w" * 200)
    assert len(long) == 60 and long.endswith("...")


# --- term evidence comes from the index, not from a regex -------------------


def test_evidence_counts_identifier_internal_terms_the_way_fts5_does(
    corpus: Path, monkeypatch
) -> None:
    """The shipped defect, from the outside.

    unicode61 splits on `_`, so FTS5 matches the query term `reply` inside
    `LATEST_REPLY` and the file ranks. The `\\b` regex that used to count the
    evidence does not split there, so it found nothing, and the hit reached
    the floor claiming zero matched terms — where the exemption written for
    the semantic stage waved it through unjudged and the pointer line told the
    agent it was a semantic guess. One regex, a precision hole and a false
    provenance claim.
    """
    memo = _memo(
        corpus, "helpdesk_ticket_fields.md", "## Fields\n\nthe LATEST_REPLY column"
    )
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])
    assert hook.recall("latest reply column", stats={}) == [memo]
    assert hook._LEX_MATCHED[memo] == ["latest", "reply", "column"]

    # Counted, therefore judgeable, therefore honestly labelled.
    matched, total, _ = hook._relevance(["latest", "reply", "column"], memo)
    assert matched == ["latest", "reply", "column"]
    assert "[matches 3/3 prompt terms" in hook._pointer_line(memo, matched, total)


def test_evidence_counts_the_file_while_the_section_names_the_chunk(
    corpus: Path, monkeypatch
) -> None:
    """The two halves of a pointer answer two different questions, on purpose.

    `[section: ...]` names the chunk that ranked, and is exactly true.
    `[matches n/m]` counts the whole FILE, and so can credit a hit with terms
    from sections the ranker never looked at — 23% of advertised matched terms,
    by audit. That imprecision is bought deliberately: MIN_MATCHED_TERMS was
    calibrated against file-wide counts, and scoping the count to the chunk
    shrinks every total by a term or two without moving the floor to match. Over
    200 replayed prompts that took the hook from 18 silent prompts to 32 — newly
    silent on 14 the shipped code served, including both hand-classified genuine
    losses, where it returned nothing at all. Re-deriving the floor for
    chunk-scoped counts is an open calibration project; until it lands, the
    honest pin is the behavior this asserts, not the tidier one.
    """
    memo = _memo(
        corpus,
        "borg_repo_layout.md",
        "## Repo layout\n\nborg repo layout\n\n## Unrelated\n\nzermatt chalet",
    )
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(corpus)])
    assert hook.recall("borg repo layout zermatt", stats={}) == [memo]
    # BM25 gives the win to the section holding the rare term, so the chunk that
    # ranked is the one-word `Unrelated` — while the other three query terms,
    # which the count does credit, live in the section that lost.
    assert hook._LEX_SECTIONS[memo] == "Unrelated"
    terms = ["borg", "repo", "layout", "zermatt"]
    assert hook._LEX_MATCHED[memo] == terms
    assert hook._relevance(terms, memo)[0] == terms
    assert "[matches 4/4 prompt terms" in hook._pointer_line(memo, terms, 4)


def test_a_candidate_the_index_reported_no_term_for_is_rejected(
    corpus: Path,
) -> None:
    """The index is the only witness now, so silence from it is a `no`.

    Zero matched terms used to be an EXEMPTION, on the theory that it meant a
    semantic hit whose evidence arithmetic could not judge; 47 of the 50
    pointers that ever took it came from prompts where the semantic stage
    never ran, so what it actually waved through was identifier-shaped
    lexical hits the old regex could not count. With that stage gone, a
    candidate whose own index reports no matched term is a contradiction
    between claim and evidence — unreachable while _record_matched is the
    only writer, which is the point: this fails a future divergence, not a
    case seen today.
    """
    memo = _memo(corpus, "restic_retention.md", "## Policy\n\nrestic keeps\n")
    hook._LEX_MATCHED.clear()
    matched, total, mtype = hook._relevance(["restic", "prune"], memo)
    assert (matched, total, mtype) == ([], 2, "reference")
    assert hook._passes_floor([], 4, "reference") is False
    # Nothing changes once there IS evidence to judge.
    assert hook._passes_floor(["unionfs"], 4, "reference") is True
    assert hook._passes_floor(["the", "a"], 9, "reference") is False


def test_md_sections_split_on_headings_and_keep_the_preamble() -> None:
    # The preamble is a chunk of its own because it holds the frontmatter
    # `description:` — the line memories are written to be found by.
    text = "---\ndescription: d\n---\n# One\nbody one\n## Two\nbody two"
    assert hook._md_sections(text) == [
        "---\ndescription: d\n---",
        "# One\nbody one",
        "## Two\nbody two",
    ]
    assert hook._md_sections("no headings here") == ["no headings here"]


def test_fts_db_path_is_per_corpus_root(corpus: Path, tmp_path) -> None:
    # Distinct DBs per root are what make the sweep safe when an invocation
    # searches one store and not the other.
    assert hook._fts_db(str(corpus)) != hook._fts_db(str(tmp_path))
    assert Path(hook._fts_db(str(corpus))).parent == tmp_path / "state"


def test_index_records_which_corpus_it_holds(corpus: Path) -> None:
    """sha256 is one-way, so nothing else in the cache dir says what
    `fts5-9d0e2c1b4a77.db` is an index OF — and "why was this memory not
    recalled" starts by looking at the index that should have held it."""
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    hook._fts_dir("restic pruning", str(corpus))
    sidecar = Path(hook._fts_db(str(corpus)).removesuffix(".db") + ".root")
    assert sidecar.read_text() == str(corpus) + "\n"

    # Advisory only: the engine never reads it, so losing it costs answers
    # nothing. Anything that made a query depend on it would turn a debugging
    # aid into a second thing that can break retrieval.
    sidecar.unlink()
    assert hook._fts_dir("restic pruning", str(corpus)) == [str(corpus / "a.md")]


def test_the_index_records_what_its_last_build_found(corpus: Path) -> None:
    """Never indexed and indexed-over-an-empty-corpus are the same silence.

    Both answer nothing and neither leaves a pointer, so the only way to tell
    them apart was to open the index — which syncs it, which rebuilds whatever
    the walk finds stale. A diagnostic that repairs the state it is measuring
    cannot report on it, so the answer is recorded at build time instead.
    """
    build = Path(hook._fts_db(str(corpus)).removesuffix(".db") + ".build")
    assert not build.exists(), "an unbuilt index must leave no record"

    hook._fts_dir("restic pruning", str(corpus))
    assert json.loads(build.read_text())["files"] == 0

    memo = _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    hook._fts_dir("restic pruning", str(corpus))
    record = json.loads(build.read_text())
    assert record["outcome"] == "ok" and record["files"] == 1
    assert record["ts"] > 0

    # Advisory, exactly like `.root`: the engine never reads it, so losing it
    # costs answers nothing.
    build.unlink()
    assert hook._fts_dir("restic pruning", str(corpus)) == [memo]


def test_a_rebuilt_index_says_so_where_the_next_run_can_read_it(
    corpus: Path,
) -> None:
    """An index that self-heals leaves no other trace on disk, so a cache being
    destroyed and rebuilt on every prompt reads exactly like a healthy one.
    `lex_rebuilds` says so in the soak log; this says so beside the index, for
    a reader who has the machine and not the log."""
    _memo(corpus, "a.md", "# a\n\nrestic repository pruning")
    db = hook._fts_db(str(corpus))
    hook._fts_dir("restic pruning", str(corpus))
    assert json.loads(Path(db.removesuffix(".db") + ".build").read_text())[
        "outcome"
    ] == "ok"

    Path(db).write_bytes(b"this is not a database" * 100)
    hook._fts_dir("restic pruning", str(corpus))
    assert json.loads(Path(db.removesuffix(".db") + ".build").read_text())[
        "outcome"
    ] == "rebuilt"


# --- _description ------------------------------------------------------------


def test_description_prefers_frontmatter_and_caps_length(tmp_path: Path) -> None:
    f = tmp_path / "m.md"
    f.write_text("---\ndescription: Short hook line\n---\n# Heading\n")
    assert hook._description(str(f)) == "Short hook line"
    f.write_text(f"---\ndescription: {'x' * 300}\n---\n")
    out = hook._description(str(f))
    assert len(out) == 160 and out.endswith("...")


def test_description_falls_back_to_heading_then_empty(tmp_path: Path) -> None:
    f = tmp_path / "m.md"
    f.write_text("# Just a heading\n\nbody\n")
    assert hook._description(str(f)) == "Just a heading"
    f.write_text("no structure at all\n")
    assert hook._description(str(f)) == ""


# --- _relevance + _passes_floor (type-aware relevance floor) -----------------


def test_relevance_reads_type_from_the_file_and_terms_from_the_index(
    tmp_path: Path,
) -> None:
    # Two sources, deliberately: `type:` is a property of the file, the
    # matched terms are a property of the chunk that ranked, and only the
    # index knows the second one.
    f = tmp_path / "m.md"
    f.write_text("---\ntype: feedback\n---\nabout blobsync and nodepool mounts\n")
    hook._LEX_MATCHED[str(f)] = ["nodepool", "blobsync"]
    try:
        matched, total, mtype = hook._relevance(["blobsync", "nodepool", "zzz"], str(f))
    finally:
        hook._LEX_MATCHED.clear()
    assert mtype == "feedback"
    assert matched == ["blobsync", "nodepool"]  # query order, not index order
    assert total == 3


def test_passes_floor_feedback_bars() -> None:
    # feedback: needs >=2 matched AND ratio >= 0.12 (distinctive terms used
    # so the Zipf branch doesn't mask the feedback bars under test)
    assert not hook._passes_floor(["blobsync"], 3, "feedback")  # 1 term
    assert not hook._passes_floor(["blobsync", "syncbox"], 50, "feedback")  # ratio
    assert hook._passes_floor(["blobsync", "syncbox"], 8, "feedback")  # 0.25 ok


def test_passes_floor_zipf_common_word_coincidence() -> None:
    # The 2026-07-19 extension: all-common-English matches are coincidence
    # for EVERY type unless >=3 terms matched. Requires common-words.txt.
    assert hook._common_words(), "common-words.txt missing or empty"
    # reference hit matching only conversational filler -> floored
    assert not hook._passes_floor(["see"], 2, "reference")
    assert not hook._passes_floor(["see", "possible"], 3, "project")
    # one distinctive term rescues it
    assert hook._passes_floor(["nodepool"], 3, "reference")
    assert hook._passes_floor(["see", "unionfs"], 3, "reference")
    # >=3 common matches = real overlap ("media write permission denied")
    assert hook._passes_floor(["media", "write", "permission"], 8, "reference")


def test_all_common_hits_need_a_share_of_the_prompt() -> None:
    # The 2026-08-12 addition. Three common words are evidence when they are
    # the prompt's subject (3/8) and coincidence when they are debris in a
    # long one (3/40) — English frequency cannot tell those apart, because
    # "media"/"write" are as common as "yet"/"project" (measured).
    assert hook._passes_floor(["media", "write", "permission"], 8, "reference")
    assert not hook._passes_floor(["media", "write", "permission"], 40, "reference")
    # A distinctive term is still enough on its own, at any ratio.
    assert hook._passes_floor(["unionfs"], 40, "reference")


def test_function_words_never_reach_the_query() -> None:
    # `yet/use/project` injected a memory on its own; the first two are
    # function words and are now stripped before ck ever sees the prompt.
    q = hook.build_query("can we use the unionfs mount yet like via project")
    assert q is not None
    assert "use" not in q.split() and "yet" not in q.split()
    assert "like" not in q.split() and "via" not in q.split()
    assert "unionfs" in q.split()


def test_common_words_golden_contract() -> None:
    # Lock the generator's contract: conversational filler in, identifiers
    # out. If a regen ever flips these, the floor's semantics changed.
    common = hook._common_words()
    for w in ("see", "fix", "sure", "check", "possible", "improvements"):
        assert w in common, f"{w} should be common"
    for w in ("nodepool", "syncbox", "codefmt", "postgres", "unionfs", "blobsync"):
        assert w not in common, f"{w} must stay distinctive"


def test_common_words_fails_open(monkeypatch, tmp_path) -> None:
    # Missing wordlist -> empty set -> Zipf branch never floors (pre-Zipf
    # behavior), and single-common-term hits pass again.
    monkeypatch.setattr(hook, "_COMMON", None)
    monkeypatch.setattr(hook, "COMMON_WORDS_FILE", str(tmp_path / "absent.txt"))
    assert hook._common_words() == frozenset()
    assert hook._passes_floor(["see"], 2, "reference")
    monkeypatch.setattr(hook, "_COMMON", None)  # un-cache for other tests


def test_common_words_ships_armed() -> None:
    """The Zipf floor's guard, asserted as its own subject.

    An untracked or renamed wordlist is invisible to the flake's git filter,
    so the sealed derivation would ship a hook whose _common_words() fails
    open to an empty set: the floor stops flooring, silently, while every
    test that names the floor's SEMANTICS still passes — each of those
    asserts what the gate lets through. This has happened once, to a
    different wordlist, past a green local suite.

    Two floor tests do fail on an empty list today, but incidentally, and
    that coverage would leave with them if either were rewritten. Neither
    catches a TRUNCATED list at all.
    """
    assert os.path.exists(hook.COMMON_WORDS_FILE), hook.COMMON_WORDS_FILE
    assert len(hook._common_words()) > 10_000


# --- _display_path -----------------------------------------------------------


def test_display_path_is_home_relative_everywhere(tmp_path: Path, monkeypatch) -> None:
    # ~-relative regardless of cwd (repo subdir, worktree, or outside).
    #
    # _cwd_in_root is lru_cached and reads os.getcwd(), which is sound in the
    # hook (one cwd per process) and a trap here: a value computed under this
    # chdir would outlive the monkeypatch undo and decide which stores every
    # later test searches. Clear on the way in AND out — in because an earlier
    # test may have cached a real answer, out because this one may have cached
    # tmp_path's.
    hook._cwd_in_root.cache_clear()
    monkeypatch.chdir(tmp_path)
    p = hook.os.path.expanduser(f"~/{PROJECT_DIR}/search/foo.md")
    assert hook._display_path(p) == f"~/{PROJECT_DIR}/search/foo.md"
    assert hook._display_path("/etc/hosts") == "/etc/hosts"
    hook._cwd_in_root.cache_clear()


def test_the_cwd_gate_is_cached_and_must_be_cleared_across_a_chdir(
    tmp_path: Path, monkeypatch
) -> None:
    # Pins the contract the docstring states, so a future caller that chdirs
    # finds it asserted rather than has to infer it.
    #
    # A tmpdir stands in for the gating root rather than a real checkout: a
    # test that needs the developer's own tree to exist passes on the laptop
    # and fails in a sealed build sandbox. The prefix branch is string work, so
    # a tmp dir exercises the same code an actual repo would.
    inside, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    inside.mkdir()
    elsewhere.mkdir()
    hook._cwd_in_root.cache_clear()
    monkeypatch.chdir(inside)
    assert hook._cwd_in_root(str(inside)) is True
    monkeypatch.chdir(elsewhere)
    assert hook._cwd_in_root(str(inside)) is True  # stale: the cached answer
    hook._cwd_in_root.cache_clear()
    assert hook._cwd_in_root(str(inside)) is False
    hook._cwd_in_root.cache_clear()


# --- _session_state_path -----------------------------------------------------


def test_session_state_path_sanitizes_traversal() -> None:
    p = hook._session_state_path("../../etc/passwd")
    assert "/etc/passwd" not in p
    assert Path(p).name == "______etc_passwd.json"
    assert "/" not in Path(p).name.replace(".json", "")


# --- end-to-end gates (subprocess: gates fire before any search) -------------


def _run(payload: dict) -> str:
    out = subprocess.run(
        ["python3", HOOK],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0  # fail-open contract: always exit 0
    return out.stdout


def test_gates_short_slash_stopword_and_malformed() -> None:
    assert _run({"session_id": "t", "prompt": "quick fix"}) == ""  # <3 words
    assert _run({"session_id": "t", "prompt": "/compact do it now please ok"}) == ""
    assert (
        _run({"session_id": "t", "prompt": "can you please help me with this and that"})
        == ""
    )


def test_malformed_stdin_fails_open() -> None:
    out = subprocess.run(
        ["python3", HOOK], input="not json", capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0 and out.stdout == ""


def _injecting_repo(tmp_path: Path) -> Path:
    """A HOME whose corpus makes INJECT_PROMPT produce a pointer.

    Pair it with _env(tmp_path), which writes the config that names these
    directories: with no config the hook has no stores and every case below
    would pass by injecting nothing, which is the wrong kind of green.
    """
    for rel in (PROJECT_DIR, PERSONAL_DIR):
        (tmp_path / rel / "search").mkdir(parents=True, exist_ok=True)
    (tmp_path / PROJECT_DIR / "search" / "unionfs_perms.md").write_text(
        "---\nname: unionfs_perms\n"
        "description: unionfs mount permissions and the media group\n"
        "type: reference\n---\n\n"
        "unionfs mount permissions: FUSE default_permissions ignores the\n"
        "supplementary groups, so the media group has to be primary.\n"
    )
    return tmp_path


INJECT_PROMPT = "unionfs mount permissions"


def test_a_reader_that_closed_early_still_fails_open(tmp_path) -> None:
    # The harness is entitled to stop reading, and a hook that answers by
    # exiting 120 has blamed the prompt for the reader's decision.
    #
    # Has to be a real process with a real pipe: the failure is CPython's
    # shutdown flush of a block-buffered stdout, which runs after the module's
    # own exception handling is over, so nothing in-process can see it.
    # Reproduced at rc=120 on main before the fix.
    repo = _injecting_repo(tmp_path)
    env = _env(tmp_path)
    prompt = INJECT_PROMPT

    def drive(session: str, stdout, close_after: tuple[int, ...] = ()):
        proc = subprocess.Popen(
            ["python3", HOOK],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(repo),
        )
        for fd in close_after:
            os.close(fd)
        assert proc.stdin is not None and proc.stderr is not None
        try:
            proc.stdin.write(json.dumps({"session_id": session, "prompt": prompt}))
            proc.stdin.close()
        except BrokenPipeError:  # pragma: no cover - child already gone
            pass
        err = proc.stderr.read()
        out = proc.stdout.read() if proc.stdout else ""
        proc.wait(timeout=60)
        return proc.returncode, out, err

    # First, with a reader: proves this fixture actually makes the hook PRINT.
    # Without it the closed-reader run below would pass on a hook that never
    # wrote anything, which is the failure mode this test exists to catch.
    rc, out, err = drive("t2", subprocess.PIPE)
    assert rc == 0 and out.strip(), f"fixture printed nothing: rc={rc}, stderr={err}"

    # Now the same thing into a pipe whose reader is already gone. A different
    # session id because the first run recorded these paths as injected, and
    # the dedup would otherwise leave this run with nothing to print.
    #
    # Both parent copies close while the child still runs, and that ordering is
    # the test: while the parent holds the read end the child's write has a
    # reader, so closing in a finally after wait() reproduces nothing and the
    # assertion below passes against the unfixed hook. Measured — main is
    # rc=120 with this ordering and rc=0 with the other.
    r, w = os.pipe()
    rc, _, err = drive("t3", w, close_after=(w, r))
    assert rc == 0, f"fail-open broken: rc={rc}, stderr={err}"
    assert "BrokenPipeError" not in err


def test_pointers_the_reader_never_saw_are_not_spent(tmp_path) -> None:
    # Dedup means "already shown this session", and it is permanent: a path
    # recorded here can never be offered to this session again. Spending it for
    # a run whose output reached nobody retires the memory on the strength of a
    # delivery that did not happen. Measured on main as a written state file
    # and an `injected` record for a run that printed nothing.
    repo = _injecting_repo(tmp_path)
    r_fd, w_fd = os.pipe()
    # Closed before the child exists, so the pipe never has a reader and the
    # write cannot succeed. Closing it after the spawn would leave the outcome
    # to whichever side won: the pointer text is a few hundred bytes and would
    # fit in the buffer of a pipe still briefly open.
    os.close(r_fd)
    proc = subprocess.Popen(
        ["python3", HOOK],
        stdin=subprocess.PIPE,
        stdout=w_fd,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(tmp_path),
        cwd=str(repo),
    )
    os.close(w_fd)
    assert proc.stdin is not None
    proc.stdin.write(json.dumps({"session_id": "nobody", "prompt": INJECT_PROMPT}))
    proc.stdin.close()
    assert proc.wait(timeout=60) == 0

    rec = _last_record(tmp_path)
    assert rec["outcome"] == "output-lost", rec
    # It had something to say — otherwise this test would pass on a run that
    # simply found nothing, which is not the state being pinned.
    assert rec["injected"], rec
    assert not (tmp_path / ".cache/memory-recall/nobody.json").exists()


def _full_ledger(tmp_path: Path, session: str) -> Path:
    """A session ledger at POINTER_BUDGET whose entries are worth nothing, so
    anything the prompt retrieves outranks them and _replace must evict.

    Evidence 0.0 rather than None: a None sends the run through the `legacy`
    budget gate, which returns before delivery, and the eviction path would
    never be reached — the test would pass without exercising anything.
    """
    state = tmp_path / ".cache/memory-recall" / f"{session}.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    ledger = {f"/nowhere/filler{i:02d}.md": 0.0 for i in range(hook.POINTER_BUDGET)}
    state.write_text(json.dumps({"shown": sorted(ledger), "spent": ledger}))
    return state


def _drive_full_ledger(
    tmp_path: Path, repo: Path, session: str, *, reader_open: bool
) -> tuple:
    """Run the hook against a saturated ledger. Returns (record, ledger moved)."""
    state = _full_ledger(tmp_path, session)
    before = state.read_text()
    if reader_open:
        stdout, post_close = subprocess.PIPE, None
    else:
        r_fd, w_fd = os.pipe()
        os.close(r_fd)  # no reader ever, as in the test above
        stdout, post_close = w_fd, w_fd
    proc = subprocess.Popen(
        ["python3", HOOK],
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(tmp_path),
        cwd=str(repo),
    )
    if post_close is not None:
        os.close(post_close)
    assert proc.stdin is not None
    proc.stdin.write(json.dumps({"session_id": session, "prompt": INJECT_PROMPT}))
    proc.stdin.close()
    if reader_open and proc.stdout is not None:
        proc.stdout.read()
    assert proc.wait(timeout=60) == 0
    return _last_record(tmp_path), state.read_text() != before


def test_an_eviction_the_reader_never_saw_is_not_reported(tmp_path) -> None:
    # `evicted` names ledger entries this run displaced, and the displacement
    # lives in the state write — which is guarded by `delivered`. Built outside
    # that guard it reported evictions on a run that never wrote the ledger,
    # naming displacements no later session can observe and handing the
    # analyzers a budget-pressure signal from a prompt that spent nothing.
    #
    # The open-reader arm is not decoration: without it this passes on a hook
    # that never evicts at all, which is the same green for the opposite
    # reason. It has to be first — the closed arm must not be what saturates
    # the ledger — and on its own session, since dedup is per session.
    repo = _injecting_repo(tmp_path)
    shown, moved = _drive_full_ledger(tmp_path, repo, "eviction-seen", reader_open=True)
    assert shown["outcome"] == "injected", shown
    assert shown.get("evicted"), f"fixture evicted nothing: {shown}"
    assert moved, "the delivered run did not rewrite the ledger"

    lost, moved = _drive_full_ledger(tmp_path, repo, "eviction-lost", reader_open=False)
    assert lost["outcome"] == "output-lost", lost
    assert lost["injected"], lost  # it had pointers, so it had evictions to claim
    assert "evicted" not in lost, f"claimed an eviction it never persisted: {lost}"
    assert not moved, "the ledger moved for a run that reached nobody"


# --- query construction: compound splitting + word gate -----------------------

_SPLIT = hook.re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", hook.re.I)


def test_compound_split_regex_letter_digit_boundaries() -> None:
    assert _SPLIT.split("nodepool1") == ["nodepool", "1"]
    assert _SPLIT.split("rack31") == ["rack", "31"]
    assert _SPLIT.split("node25") == ["node", "25"]
    assert _SPLIT.split("plain") == ["plain"]
    # mixed boundaries both directions
    assert _SPLIT.split("13900x2") == ["13900", "x", "2"]


def test_four_word_question_passes_the_gate(tmp_path, monkeypatch) -> None:
    # A five-word question naming one host must reach the search stage — this
    # exact shape was wrongly gated at the old 6-word minimum. Run with
    # CONFIGURED stores that are not on disk: _search_dirs finds nothing and
    # the hook exits via gate:nodirs, which proves the shape gate PASSED
    # (a shape-gated prompt would exit before the dirs check). Soak-log
    # evidence distinguishes the two.
    env = _env(tmp_path, stores=False)
    out = subprocess.run(
        ["python3", HOOK],
        input=json.dumps({"session_id": "t", "prompt": "what's the fqn of nodepool1"}),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert out.returncode == 0
    log = tmp_path / ".cache" / "memory-recall" / "log.jsonl"
    assert log.is_file()
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:nodirs"  # NOT gate:shape


# --- soak log ------------------------------------------------------------------


# --- per-session pointer budget ----------------------------------------------


def _drive_main(monkeypatch, tmp_path, hits: list[str], session: str) -> dict:
    """Run main() in-process with the retrieval stage stubbed, and return the
    soak-log record it wrote. Subprocess-driving it would need a real corpus;
    the budget is about state, not retrieval."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(hook, "_search_dirs", lambda: ["/corpus"])

    def _recall(prompt, stats=None, dirs=None, deadline=None):
        # The real one clears the side channels on entry and repopulates
        # _LEX_MATCHED from the index. A stub that only cleared would hand
        # every hit an empty term list, which the floor now reads as a
        # candidate with no evidence and rejects — so the stub has to stand in
        # for the index too, tokenizing the way unicode61 does (identifiers
        # split on `_`) rather than the way a regex would.
        hook._LEX_MATCHED.clear()
        hook._LEX_SECTIONS.clear()
        hook._LEX_SCORES.clear()
        terms = (hook.build_query(prompt) or "").split()
        for i, path in enumerate(hits):
            tokens = set(re.split(r"[^0-9a-z]+", Path(path).read_text().lower()))
            hook._LEX_MATCHED[path] = [t for t in terms if t in tokens]
            # Descending and distinct, so a log key that pairs names with
            # scores by position is caught when it pairs them by anything
            # else. The values are the stub's, not the ranker's — what the
            # ranker actually puts here is
            # test_the_index_scores_every_hit_it_returns.
            hook._LEX_SCORES[path] = round(1.0 - i * 0.05, 3)
        return hits

    monkeypatch.setattr(hook, "recall", _recall)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(
            json.dumps({"session_id": session, "prompt": "the unionfs mount is stale"})
        ),
    )
    hook.main()
    log = tmp_path / ".cache" / "memory-recall" / "log.jsonl"
    return json.loads(log.read_text().splitlines()[-1])


def test_session_pointer_budget_stops_injection(monkeypatch, tmp_path, capsys) -> None:
    memos = []
    for name in ("unionfs_perms.md", "unionfs_mount_order.md"):
        p = tmp_path / name
        p.write_text(
            "---\ndescription: unionfs perms\ntype: reference\n---\nunionfs\n"
        )
        memos.append(str(p))
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)

    # One pointer short of the budget, two eligible hits: the remaining
    # allowance caps the prompt below MAX_HITS rather than overshooting.
    (state / "s1.json").write_text(
        json.dumps([f"/spent/{i}.md" for i in range(hook.POINTER_BUDGET - 1)])
    )
    rec = _drive_main(monkeypatch, tmp_path, memos, "s1")
    assert rec["outcome"] == "injected"
    assert len(rec["injected"]) == 1
    out = capsys.readouterr().out
    # One pointer, and the notice naming what the cap cost. The cap here is
    # the BUDGET, not MAX_HITS — both cut eligible hits, and a notice that
    # only understood one of them would silently under-report on long
    # sessions, which is exactly where the budget starts biting.
    assert out.count("\n- ") == 2
    assert "…1 further match — search: memory-recall --search" in out
    assert rec["truncated"] == 1

    # At the budget: gated, with no output at all.
    (state / "s2.json").write_text(
        json.dumps([f"/spent/{i}.md" for i in range(hook.POINTER_BUDGET)])
    )
    rec = _drive_main(monkeypatch, tmp_path, memos, "s2")
    assert rec["outcome"] == "gate:budget"
    assert rec["pointers"] == hook.POINTER_BUDGET
    assert capsys.readouterr().out == ""


def test_raising_the_cap_only_appends(monkeypatch, tmp_path, capsys) -> None:
    # MAX_HITS is a budget, not a verdict: _eligible judges every candidate
    # before the cap applies, so a larger cap must ADD a pointer without
    # reordering or dropping the ones the smaller cap already showed. Over the
    # labelled-pair oracle, 2 -> 3 moved 47 pairs from TRUNCATED to SHOWN and
    # left every other bucket identical; this is that same claim at the render
    # surface, which is the only place a reordering would reach a reader.
    memos = []
    for word in ("perms", "order", "cache", "policy"):
        p = tmp_path / f"unionfs_{word}.md"
        p.write_text(
            f"---\ndescription: unionfs {word}\ntype: reference\n---\n"
            "unionfs stale mount\n"
        )
        memos.append(str(p))
    (tmp_path / ".cache" / "memory-recall").mkdir(parents=True)

    def pointers(session: str) -> list[str]:
        _drive_main(monkeypatch, tmp_path, memos, session)
        return [
            ln
            for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("- ") and "further match" not in ln
        ]

    # Separate sessions, because the dedup ledger would otherwise spend the
    # first run's pointers and make the second run's list a different question.
    monkeypatch.setattr(hook, "MAX_HITS", 2)
    two = pointers("s1")
    monkeypatch.setattr(hook, "MAX_HITS", 3)
    three = pointers("s2")

    assert len(two) == 2
    assert len(three) == 3
    assert three[:2] == two


def _spend(state: Path, name: str, ledger: dict) -> Path:
    p = state / f"{name}.json"
    p.write_text(json.dumps({"shown": sorted(ledger), "spent": ledger}))
    return p


def test_the_budget_ledger_records_what_each_pointer_cost(
    monkeypatch, tmp_path
) -> None:
    """Replacement needs a price on every slot, so injection has to write one.

    Also the compatibility direction: a session that started under the old
    bare-list schema keeps its dedup set — those paths must not be re-injected
    just because the file's shape changed under it.
    """
    memos = _eligible_memos(tmp_path, 2)
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    (state / "s1.json").write_text(json.dumps(["/spent/old.md"]))

    rec = _drive_main(monkeypatch, tmp_path, memos, "s1")
    assert rec["outcome"] == "injected"
    written = json.loads((state / "s1.json").read_text())
    assert "/spent/old.md" in written["shown"]
    assert set(written["shown"]) == {"/spent/old.md", *memos}
    # The pre-ledger pointer has no recorded price and must not be given one.
    assert written["spent"]["/spent/old.md"] is None
    # Every one of the prompt's three content terms is in these memos.
    assert written["spent"][memos[0]] == pytest.approx(1.0)


def test_a_full_budget_is_a_bar_not_a_deadline(monkeypatch, tmp_path) -> None:
    """Past the budget, evidence buys a slot from the weakest pointer holding
    one — so the 31st prompt's strong match is no longer beaten by the 3rd
    prompt's incidental one purely for arriving later."""
    memos = _eligible_memos(tmp_path, 1)
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    ledger = {f"/spent/{i}.md": 0.5 for i in range(hook.POINTER_BUDGET)}
    ledger["/spent/weakest.md"] = 0.1
    del ledger[f"/spent/{hook.POINTER_BUDGET - 1}.md"]
    _spend(state, "s1", ledger)

    rec = _drive_main(monkeypatch, tmp_path, memos, "s1")
    assert rec["outcome"] == "injected"
    assert rec["evicted"] == ["weakest.md"]
    written = json.loads((state / "s1.json").read_text())
    assert len(written["spent"]) == hook.POINTER_BUDGET
    assert "/spent/weakest.md" not in written["spent"]
    # Evicted from the ledger, never from the dedup set: the budget must not
    # be able to buy the same pointer twice.
    assert "/spent/weakest.md" in written["shown"]


def test_a_tie_does_not_buy_a_slot(monkeypatch, tmp_path) -> None:
    """`strictly exceeds`, asserted as its own subject. Equal evidence must
    lose to the incumbent, or a full budget churns forever between hits of
    identical strength and the cap stops meaning anything."""
    memos = _eligible_memos(tmp_path, 1)
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    # A full budget priced at exactly what these memos score: all the terms.
    _spend(state, "s1", {f"/spent/{i}.md": 1.0 for i in range(hook.POINTER_BUDGET)})

    rec = _drive_main(monkeypatch, tmp_path, memos, "s1")
    assert rec["outcome"] == "gate:budget:weak"
    assert rec["best"] == pytest.approx(1.0)
    assert json.loads((state / "s1.json").read_text())["spent"] == {
        f"/spent/{i}.md": 1.0 for i in range(hook.POINTER_BUDGET)
    }


def test_an_unpriced_full_budget_is_decided_before_retrieval_runs(
    monkeypatch, tmp_path
) -> None:
    """The one budget answer that does not depend on what was found.

    A ledger of pointers with no recorded evidence has nothing for a new hit
    to beat, so the verdict is known before the lexical stage runs — and 86
    records a day were paying for a full retrieval whose result was then
    discarded. Scoring those pointers 0.0 instead of leaving them unpriced
    would have been the cheap alternative and the wrong one: it reads `no
    evidence` as `worthless`, letting the first hit of any strength evict
    them, which makes exactly the oldest sessions unbounded.
    """
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    (state / "s1.json").write_text(
        json.dumps([f"/spent/{i}.md" for i in range(hook.POINTER_BUDGET)])
    )
    called = []
    monkeypatch.setattr(hook, "_search_dirs", lambda: ["/corpus"])
    monkeypatch.setattr(
        hook,
        "recall",
        lambda *a, **k: called.append(1) or [],
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(
            json.dumps({"session_id": "s1", "prompt": "the unionfs is stale"})
        ),
    )
    hook.main()
    log = (state / "log.jsonl").read_text().splitlines()[-1]
    rec = json.loads(log)
    assert rec["outcome"] == "gate:budget"
    assert rec["ledger"] == "legacy"
    assert called == [], "retrieval ran for a verdict already known"


def _eligible_memos(tmp_path: Path, n: int) -> list[str]:
    memos = []
    for i in range(n):
        p = tmp_path / f"unionfs_{i}.md"
        p.write_text(
            "---\ndescription: unionfs perms\ntype: reference\n---\n"
            "unionfs mount stale\n"
        )
        memos.append(str(p))
    return memos


def test_truncation_notice_names_what_the_cap_cost(
    monkeypatch, tmp_path, capsys
) -> None:
    """MAX_HITS is a budget, not a verdict.

    Without this line the eligible memory past the cap is indistinguishable
    from a corpus that had nothing more to say, and the agent has no reason
    to look — which is the whole failure mode a small cap creates.
    """
    memos = _eligible_memos(tmp_path, hook.MAX_HITS + 1)
    rec = _drive_main(monkeypatch, tmp_path, memos, "t3")
    out = capsys.readouterr().out
    assert len(rec["injected"]) == hook.MAX_HITS
    assert rec["truncated"] == 1
    # Named and scored, not just counted. The rank-oracle join that put 20.1%
    # of misses in this bucket is a join against the filenames a session went
    # on to READ, and a count joins with nothing; the score then says whether
    # the thing the cap dropped was a near-miss or the tail.
    assert rec["truncated_files"] == [os.path.basename(memos[-1])]
    assert rec["truncated_scores"] == [round(1.0 - hook.MAX_HITS * 0.05, 3)]
    # Positional pairing, on the key that is read most: three names, three
    # scores, best first.
    assert len(rec["scores"]) == len(rec["injected"]) == len(rec["overlap"])
    assert rec["scores"] == sorted(rec["scores"], reverse=True)
    assert rec["scores"][0] == 1.0
    # The query goes in whole and sanitized: a notice whose command searches
    # for something other than what N was counted over is worse than none.
    assert "- …1 further match — search: " in out
    assert '--search "unionfs mount stale"' in out


def test_no_notice_when_nothing_was_cut(monkeypatch, tmp_path, capsys) -> None:
    memos = _eligible_memos(tmp_path, hook.MAX_HITS)
    rec = _drive_main(monkeypatch, tmp_path, memos, "t2")
    assert len(rec["injected"]) == hook.MAX_HITS
    # Additive key: absent, not 0, so `truncated` in the soak log always means
    # a prompt that lost a pointer.
    assert "truncated" not in rec
    assert "truncated_files" not in rec and "truncated_scores" not in rec
    assert "further match" not in capsys.readouterr().out


def _termed_memo(tmp_path: Path, name: str, body: str) -> str:
    """A memo whose matched terms — and so its _evidence share — are exactly
    the ones `body` contains.

    The frontmatter is deliberately empty of the prompt's vocabulary: the
    retrieval stub tokenizes the whole file, so a description mentioning
    unionfs the way _eligible_memos does would price every memo alike and
    leave nothing for the budget to choose between.
    """
    p = tmp_path / f"{name}.md"
    p.write_text(f"---\ndescription: a note\ntype: reference\n---\n{body}\n")
    return str(p)


def test_the_pointer_the_budget_refused_is_the_one_the_log_names(
    monkeypatch, tmp_path
) -> None:
    """`truncated_files` by identity, not by position.

    `picks` is a prefix of `eligible` only while the budget has room. Past it
    _replace ranks the offered window by EVIDENCE, which is not the rank order
    the window arrived in, and returns a subsequence with holes. A positional
    `eligible[len(picks):]` then slides by however many it dropped from the
    middle: the file the budget actually refused is missing from the record
    and a file that was INJECTED is named as cut, in the same record. The
    count stays right either way, which is how it survived — and these names
    are the report's most-cut table, i.e. the evidence the next MAX_HITS
    decision gets argued from.
    """
    window = [
        _termed_memo(tmp_path, "top", "unionfs mount stale"),  # 3/3 terms
        _termed_memo(tmp_path, "thin", "unionfs"),  # 1/3: outbid
        _termed_memo(tmp_path, "third", "unionfs mount stale"),  # 3/3
    ]
    tail = _termed_memo(tmp_path, "tail", "unionfs")  # eligible, past MAX_HITS
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    # Full, and priced BETWEEN the two evidence levels above, so the window's
    # first and third buy a slot and its second does not. That hole in the
    # middle of the offered window is the whole fixture.
    _spend(state, "s1", {f"/spent/{i}.md": 0.5 for i in range(hook.POINTER_BUDGET)})

    rec = _drive_main(monkeypatch, tmp_path, [*window, tail], "s1")

    assert rec["outcome"] == "injected", rec
    assert rec["injected"] == ["top.md", "third.md"], rec
    # Non-vacuous: this is the replacement branch, not the room branch.
    assert len(rec["evicted"]) == 2, rec
    assert rec["truncated"] == 2
    assert rec["truncated_files"] == ["thin.md", "tail.md"], rec


def test_no_file_is_both_injected_and_reported_as_cut(monkeypatch, tmp_path) -> None:
    """The invariant behind that fix, on both branches that build the record.

    `injected` and `truncated_files` are read as disjoint by everything
    downstream — a name in both is a record contradicting itself. Only the
    replacement branch ever broke it; asserting it on the room branch too is
    what keeps a rewrite of either from reintroducing it on the other.
    """
    memos = [
        _termed_memo(tmp_path, "a", "unionfs mount stale"),
        _termed_memo(tmp_path, "b", "unionfs"),
        _termed_memo(tmp_path, "c", "unionfs mount stale"),
        _termed_memo(tmp_path, "d", "unionfs"),
    ]
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    _spend(state, "s2", {f"/spent/{i}.md": 0.5 for i in range(hook.POINTER_BUDGET)})

    for session in ("s1", "s2"):  # s1 has room, s2 is at the budget
        rec = _drive_main(monkeypatch, tmp_path, memos, session)
        assert rec["outcome"] == "injected", rec
        assert rec["truncated"], f"{session} cut nothing, so it asserts nothing"
        assert set(rec["injected"]).isdisjoint(rec["truncated_files"]), rec


def test_an_under_budget_run_reports_no_eviction(monkeypatch, tmp_path) -> None:
    """`evicted` is additive, the way `truncated` is.

    A key present on every injecting record is a key nobody greps for, and it
    would turn "did the budget bite here" from a presence check into a
    question about list length — which is how the report's budget-pressure
    count would come to count every prompt in the log.
    """
    rec = _drive_main(monkeypatch, tmp_path, _eligible_memos(tmp_path, 1), "s1")
    assert rec["outcome"] == "injected"
    assert "evicted" not in rec, rec


def test_a_ledger_the_write_lost_says_so_and_claims_no_eviction(
    monkeypatch, tmp_path
) -> None:
    """The third arm of the eviction contract: delivered, but not persisted.

    `evicted` does not describe this run's output, it describes a mutation of
    the ledger, so it is only true once that mutation is on disk. Delivery is
    no evidence of that — the state write has its own failure, swallowed on
    purpose so a read-only cache dir cannot cost the user a prompt — and a run
    reporting evictions here would name displacements no later session can
    observe. `state` is the other half and the worse half: with the write
    lost, `shown` does not advance either, so the session re-offers the same
    pointers on every prompt and the budget stops bounding anything, which in
    the log is otherwise a run of ordinary injections.
    """
    memos = _eligible_memos(tmp_path, 1)
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    ledger = {f"/spent/{i}.md": 0.1 for i in range(hook.POINTER_BUDGET)}

    # Writable first, on its own session: without it this passes on a hook
    # that never evicts at all, which is the same green for the opposite
    # reason.
    _spend(state, "s1", ledger)
    ok = _drive_main(monkeypatch, tmp_path, memos, "s1")
    assert ok["evicted"], f"fixture evicted nothing: {ok}"

    kept = _spend(state, "s2", ledger)
    before = kept.read_text()
    # The soak log appends to an EXISTING file, which needs write permission
    # on the file and not on its directory; the state write creates a new temp
    # file beside the ledger, which needs the directory. A read-only directory
    # therefore fails exactly the write under test and still leaves the record
    # that has to report it.
    (state / "log.jsonl").touch()
    state.chmod(0o500)
    try:
        rec = _drive_main(monkeypatch, tmp_path, memos, "s2")
    finally:
        state.chmod(0o700)

    assert rec["outcome"] == "injected", rec
    assert rec["injected"], rec
    assert rec["state"] == "unwritten", rec
    assert "evicted" not in rec, f"claimed an eviction it never persisted: {rec}"
    assert kept.read_text() == before, "the ledger moved despite the failed write"


def test_a_write_that_dies_midway_leaves_the_previous_ledger_intact(
    monkeypatch, tmp_path
) -> None:
    """Written beside and renamed over, never truncated in place.

    `open(path, "w")` destroys the ledger before writing its replacement, so
    anything that stops the write in between — ENOSPC, a full volume, a
    SIGKILL after the harness's grace — leaves a valid prefix of invalid JSON.
    That is not a corrupt-file inconvenience: _load_session reads it as a
    fresh session, so the budget silently resets to zero and the session
    spends it again on paths it has already shown, with nothing in the log
    saying so. The ledger is also the one piece of state here that cannot be
    rebuilt; the FTS index regenerates from the corpus, this does not.
    """
    memos = _eligible_memos(tmp_path, 1)
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    ledger_path = _spend(state, "s1", {"/spent/old.md": 0.9})
    before = ledger_path.read_text()

    # The damage in question, produced directly, because without it this test
    # would pass on a hook whose failed write happens to be a no-op for some
    # unrelated reason. A prefix is valid utf-8 and invalid JSON, and loads as
    # a session that has spent nothing.
    torn = state / "torn.json"
    torn.write_text(before[: len(before) // 2])
    assert hook._load_session(str(torn)) == (set(), {})

    def _dump_then_fail(obj, f, *_a, **_kw) -> None:
        f.write(json.dumps(obj)[:20])  # a prefix reaches the disk, then ENOSPC
        raise OSError("No space left on device")

    # The hook's json module, so the patch reaches the state write and not the
    # json.dumps this file and the soak log use.
    monkeypatch.setattr(hook.json, "dump", _dump_then_fail)
    rec = _drive_main(monkeypatch, tmp_path, memos, "s1")

    assert rec["outcome"] == "injected", rec
    assert rec["state"] == "unwritten", rec
    assert ledger_path.read_text() == before
    assert hook._load_session(str(ledger_path))[1] == {"/spent/old.md": 0.9}
    assert not list(state.glob("*.tmp")), "the failed write left its temp file behind"


def test_a_crash_past_the_gates_lands_a_record_before_it_is_swallowed(
    monkeypatch, tmp_path
) -> None:
    """__main__ suppresses every exception to keep the hook fail-open, so a
    raise anywhere past the gates exits 0, silent, with nothing logged — and a
    prompt that leaves NO record is the one result the soak log cannot count.

    Reproduced with a hand-written ledger whose evidence values were strings:
    _replace's sort raised TypeError, and because the state file persists,
    every prompt of that session was dead for the rest of its life while the
    log showed nothing at all. The loader now defuses that particular shape
    (test below), so the raise is injected here rather than smuggled in
    through the ledger — the recorder is about any raise, not that one.
    """
    memos = _eligible_memos(tmp_path, 1)
    state = tmp_path / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    _spend(state, "s1", {f"/spent/{i}.md": 0.1 for i in range(hook.POINTER_BUDGET)})

    def _boom(*_a, **_kw):
        raise TypeError("'<' not supported between instances of 'str' and 'float'")

    monkeypatch.setattr(hook, "_replace", _boom)
    # Re-raised on purpose: fail-open is __main__'s job, and swallowing it
    # here would also swallow it in the --search CLI, which is not fail-open.
    with pytest.raises(TypeError):
        _drive_main(monkeypatch, tmp_path, memos, "s1")

    rec = _last_record(tmp_path)
    assert rec["outcome"] == "error", rec
    assert rec["err"] == "TypeError", rec


def test_a_ledger_with_unusable_prices_degrades_instead_of_dying(tmp_path) -> None:
    """Nothing here trusts the file's VALUES.

    This hook writes only floats, but the file sits under a guessable name,
    survives across sessions, and has already carried one other schema. A
    `spent` whose values are strings is what sent _replace's sort into a
    TypeError; loaded as "not comparable" — the same bucket the legacy
    schema's unpriced pointers use — it costs that session its replacement
    budget and nothing else. Booleans are called out because they are ints to
    isinstance and would otherwise price a pointer at 1.0 or 0.0.
    """
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            {
                "shown": ["/a.md", "/b.md", "/c.md", 7],
                "spent": {"/a.md": "0.9", "/b.md": 0.4, "/c.md": True, "/d.md": None},
            }
        )
    )
    shown, spent = hook._load_session(str(p))
    assert shown == {"/a.md", "/b.md", "/c.md"}
    assert spent == {"/a.md": None, "/b.md": 0.4, "/c.md": None, "/d.md": None}


def test_soak_log_written_for_gated_prompt(tmp_path) -> None:
    env = _env(tmp_path)
    subprocess.run(
        ["python3", HOOK],
        input=json.dumps({"session_id": "slog", "prompt": "hi"}),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    log = tmp_path / ".cache" / "memory-recall" / "log.jsonl"
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:shape"
    assert rec["words"] == 1
    assert "ms" in rec and "prompt_sha" in rec
    # never the prompt text itself
    assert "hi" not in log.read_text().replace('"hi"', "")


ENVELOPE_MARKERS = [
    "<task-notification id=7>agent finished the memory audit</task-notification>",
    "<teammate-message from=team-lead>please run the final battery</teammate-message>",
    "<system-reminder>the task list has not been used recently</system-reminder>",
    "<command-name>/memory-status</command-name>",
    "<local-command-stdout>ran 51 tests in 4.1s</local-command-stdout>",
    "<user-prompt-submit-hook>recall fired on the previous turn</user-prompt-submit-hook>",
    "<agent-progress>subagent is still building the index</agent-progress>",
    "<background-task-result>exit code 0 after 40 seconds</background-task-result>",
    "<hook-feedback>the pipefail guard rejected that command</hook-feedback>",
    "[SYSTEM NOTIFICATION] your background task has completed successfully",
    "<command-message>memory-status is running…</command-message>",
    "<bash-input>uv run tools/report.py --days 7</bash-input>",
    "<bash-stdout>122 passed in 3.88s</bash-stdout>",
    "<!-- Generated by ce-lite converter -->\nreviewer notes follow",
    "[Request interrupted by user]",
]


@pytest.mark.parametrize("prompt", ENVELOPE_MARKERS)
def test_every_harness_envelope_marker_gates(prompt) -> None:
    """Kills the mutant that empties the marker list.

    Each of these is scaffolding the harness addresses to the agent. The
    shipped hook injected on 100% of them at the full MAX_HITS, on the
    scaffolding's own vocabulary, which is why this is a gate and not a
    ranking change.

    Goes through _is_envelope, never _ENVELOPE directly: a test that applies
    the pattern itself brings its own anchoring and cannot see the hook lose
    its own."""
    assert hook._is_envelope(prompt.strip())


# Prompts a person actually typed, several of them ABOUT envelopes. None may
# gate: prefix-anchoring is the whole reason the false-positive rate is zero
# rather than merely small, so a prompt that mentions, quotes, or leads into an
# envelope is still a question and must be answered.
NEAR_MISS_HUMAN = [
    "why does the hook fire on <task-notification> blocks from the harness?",
    "can you filter out teammate-message envelopes before indexing them",
    "the [SYSTEM NOTIFICATION] prefix shows up in my soak log, is that expected",
    "here is what the harness sent me, can you explain it?\n"
    "<task-notification id=3>build finished</task-notification>",
    "explain this please\n<teammate-message from=x>do the thing</teammate-message>",
    "what does system-reminder mean in the transcript",
]


@pytest.mark.parametrize("prompt", NEAR_MISS_HUMAN)
def test_prompts_about_envelopes_are_not_gated(prompt) -> None:
    assert not hook._is_envelope(prompt.strip())


# The class anchoring alone cannot save: a question that OPENS with the tag it
# is asking about. Each of these starts where an envelope starts, and none of
# them is one — the tag does not own its line and nothing ever closes it.
OPENS_WITH_TAG_HUMAN = [
    "<system-reminder> blocks keep leaking into my prompts, how do I stop it",
    "<teammate-message> envelopes are being indexed, can we exclude them",
    "<task-notification id=3> — what is the id field for, and who sets it",
]


@pytest.mark.parametrize("prompt", OPENS_WITH_TAG_HUMAN)
def test_a_question_that_opens_with_a_tag_is_still_a_question(prompt) -> None:
    """Completeness is the half of the gate anchoring does not cover.

    The harness emits whole envelopes: the tag owns its first line, or it is
    closed somewhere in the body. A prompt that merely begins with the tag's
    characters and then keeps talking is a person, and gating it costs them
    retrieval on precisely the question where the answer is a memory about
    this hook."""
    assert not hook._is_envelope(prompt.strip())


def test_completeness_takes_either_form_and_neither_alone_is_the_rule() -> None:
    """Both arms, because a mutant that drops either one still passes the
    other's tests. The tag owning its line is how multi-line bodies arrive,
    so requiring the close tag unconditionally would un-gate all of them; the
    close tag is how one-liners arrive, so requiring the tag to own its line
    would un-gate those."""
    assert hook._is_envelope("<system-reminder>\nthe task list is unused")
    assert hook._is_envelope("<hook-feedback>rejected that command</hook-feedback>")
    # And the marker that is not a tag at all is untouched by either arm:
    # nothing to close, nothing to own a line.
    assert hook._is_envelope("[Request interrupted by user]")


def test_a_short_envelope_is_recorded_as_an_envelope_not_a_shape(
    tmp_path, monkeypatch
) -> None:
    """Under MIN_PROMPT_WORDS the shape gate would refuse it too, and which
    gate gets the credit is not cosmetic: `gate:shape` reads in the soak log
    as "a person typed something too short to search", which is the population
    every rate is taken over. The envelope gate has to come first in both
    directions — this is the short one, the long one is
    test_envelope_gate_beats_the_shape_gate_at_any_length."""
    prompt = "<bash-stdout>ok</bash-stdout>"
    assert len(prompt.split()) < hook.MIN_PROMPT_WORDS
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"session_id": "t", "prompt": prompt})),
    )
    hook.main()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:envelope"


GATE_CASES = [
    ("<bash-stdout>ok</bash-stdout>", "gate:envelope"),
    ("/deploy the fleet to every host", "gate:shape"),
    # Two content words: build_query answers this one, so a GATED rule written
    # as `build_query(...) is None` — what the inverted join used — calls it
    # searchable while production refuses it. The case that rule cannot see.
    ("deploy nixos", "gate:shape"),
    ("word " * 1000, "gate:shape"),
    ("the and of", "gate:stopwords"),
]


@pytest.mark.parametrize("prompt,expected", GATE_CASES)
def test_the_shared_gate_predicate_answers_what_main_logs(
    prompt, expected, tmp_path, monkeypatch
) -> None:
    """prompt_gate() is consumed by an analyzer that reports what production
    would have done, so its value has to BE what production does, not a
    plausible copy of it. Each case is driven both ways: through the predicate
    and through main(), and the two must name the same gate."""
    assert hook.prompt_gate(prompt.strip()) == expected
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    # A corpus root that exists and holds nothing. Only the stopwords case
    # needs it — that gate sits AFTER the nodirs check, so on a machine with
    # no memory stores main() answers gate:nodirs and the case reads as a
    # disagreement between the predicate and main() when it is really a
    # statement about the machine. It failed exactly that way in the Nix
    # sandbox while passing on the author's laptop. Nothing here reaches
    # retrieval, so an empty directory is enough to get past the check.
    monkeypatch.setattr(hook, "_search_dirs", lambda: [str(tmp_path)])
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"session_id": "t", "prompt": prompt})),
    )
    hook.main()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"] == expected


def test_the_two_word_prompt_is_one_the_stopword_gate_would_pass(tmp_path) -> None:
    """Non-vacuousness for the case above, asserted rather than asserted-about:
    if build_query ever starts refusing two-word prompts, that case stops
    distinguishing the shared predicate from the rule it replaced and this
    test says so instead of quietly passing."""
    assert hook.build_query("deploy nixos") is not None


def test_a_prompt_past_every_gate_leaves_the_predicate_with_nothing_to_say(
    tmp_path, monkeypatch
) -> None:
    """The other direction: None has to mean main() reached retrieval. Stubbing
    the corpora empty stops it at gate:nodirs, which is the outcome AFTER all
    three prompt gates — and is deliberately not part of the predicate, since
    it answers for the machine rather than for the prompt."""
    prompt = "the unionfs mount denies writes to the media library"
    assert hook.prompt_gate(prompt) is None
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    monkeypatch.setattr(hook, "_search_dirs", list)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"session_id": "t", "prompt": prompt})),
    )
    hook.main()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:nodirs"


def test_a_machineless_run_is_logged_as_such_even_when_the_prompt_is_stopwords(
    tmp_path, monkeypatch
) -> None:
    """Splitting the predicate's answer around the dirs check is load-bearing,
    and only this prompt shows it: it fails BOTH gate:stopwords and
    gate:nodirs. Relaying the predicate whole — before the dirs check — would
    log gate:stopwords and blame the operator's wording for a machine that had
    no corpora to search. main() is the only thing that can order these, since
    the predicate deliberately cannot see the corpora."""
    prompt = "the and of"
    assert hook.prompt_gate(prompt) == "gate:stopwords"
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    monkeypatch.setattr(hook, "_search_dirs", list)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"session_id": "t", "prompt": prompt})),
    )
    hook.main()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:nodirs"


def test_a_relayed_teammate_message_reaches_retrieval_through_main(tmp_path) -> None:
    """The predicate test above is not enough: the gate is only worth what
    main() does with it, and this shape is 1201 of the transcripts' prompts
    against zero bare <teammate-message> ones — the entire population the
    anchoring exists to protect. Driven against configured-but-absent stores,
    so the hook exits at gate:nodirs; reaching that outcome is itself the
    proof, since the envelope gate sits ahead of the dirs check."""
    relay = (
        "Another Claude session sent a message:\n"
        '<teammate-message teammate_id="team-lead">the sqlite fts5 vocab table '
        "needs the three-arg form</teammate-message>"
    )
    out = subprocess.run(
        ["python3", HOOK],
        input=json.dumps({"session_id": "t", "prompt": relay}),
        capture_output=True,
        text=True,
        timeout=30,
        env=_env(tmp_path, stores=False),
    )
    assert out.returncode == 0
    assert _last_record(tmp_path)["outcome"] == "gate:nodirs"


def test_a_relayed_teammate_message_is_not_gated() -> None:
    """The one envelope-shaped class that must keep retrieving.

    A relayed teammate message is agent-authored but substantive: unlike
    task-notification boilerplate it carries real technical vocabulary, and the
    receiving agent is a reader retrieval can serve. It is also the live shape —
    across the transcripts there are 1201 relay-prefixed prompts and zero bare
    <teammate-message> ones, so this is what the gate actually meets.

    What saves it is prefix-anchoring and nothing else: the relay lead precedes
    the tag, so the marker is present in the prompt but not at its start. An
    unanchored predicate would suppress all 1201."""
    relay = (
        "Another Claude session sent a message:\n"
        '<teammate-message teammate_id="team-lead">the sqlite fts5 vocab table '
        "needs the three-arg form, and the rebase should preserve the SIGTERM "
        "flag-after-write ordering</teammate-message>"
    )
    assert "teammate-message" in relay
    assert not hook._is_envelope(relay.strip())


def test_envelope_gate_records_its_own_outcome_and_skips_retrieval(
    tmp_path, monkeypatch
) -> None:
    """The outcome key is additive, and the gate must land ahead of the search.

    build_query is replaced with a bomb: reaching it at all is the failure this
    pins, because the point of gating rather than filtering is that the
    scaffolding vocabulary never reaches the index."""

    def _boom(_):
        raise AssertionError("retrieval ran on a harness envelope")

    monkeypatch.setattr(hook, "build_query", _boom)
    monkeypatch.setattr(hook, "recall", _boom)
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"session_id": "t", "prompt": ENVELOPE_MARKERS[0]})),
    )
    hook.main()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:envelope"
    assert "injected" not in rec


def test_envelope_gate_beats_the_shape_gate_at_any_length(
    tmp_path, monkeypatch
) -> None:
    """An envelope is an envelope at any length.

    A notification past the 4000-char shape limit would otherwise be recorded
    as gate:shape, which reads in the soak log as "a user pasted a blob" — the
    one thing the stratification exists to tell apart."""
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    long_env = (
        "<task-notification id=9>" + ("status ok. " * 500) + "</task-notification>"
    )
    assert len(long_env) > 4000
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"session_id": "t", "prompt": long_env})),
    )
    hook.main()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"] == "gate:envelope"


def test_envelope_gate_reads_the_stripped_prompt() -> None:
    """Deliberately broader than the offline stratifier, which anchors on the
    raw text. A harness that emits a leading newline is still a harness."""
    assert hook._is_envelope(
        "\n  <task-notification id=1>done</task-notification>".strip()
    )


def test_every_record_says_which_code_wrote_it(tmp_path, monkeypatch) -> None:
    """A soak log is only ever read as a comparison across time, and none of
    those comparisons is sound unless a record names its own behavior. The
    stamp has to move when the FILE moves, not when the tuning constants do:
    the sem stage's misreported fire rate was a code defect with every
    constant unchanged, and a constants hash would have called those records
    comparable to today's."""
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path))
    hook._soak_log({"outcome": "probe"})
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert len(rec["v"]) == 8 and all(c in "0123456789abcdef" for c in rec["v"])

    monkeypatch.setattr(hook, "_VERSION", None)
    monkeypatch.setattr(hook, "__file__", str(tmp_path / "other.py"))
    (tmp_path / "other.py").write_text("# a hook that behaves differently\n")
    hook._soak_log({"outcome": "probe"})
    after = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert after["v"] != rec["v"]


def test_the_log_names_what_the_floor_dropped(monkeypatch, tmp_path) -> None:
    """`why wasn't memory X recalled` has three answers — never retrieved,
    retrieved and floored, retrieved and capped — and a count tells them
    apart from none of the others. Naming the files answers it from the log
    instead of from a replay against a corpus that has since changed."""
    memos = _eligible_memos(tmp_path, 1)
    # Matches on `the`/`is` alone: retrieved, then dropped by the Zipf floor.
    junk = tmp_path / "unrelated_note.md"
    junk.write_text("---\ndescription: d\ntype: reference\n---\nthe mount is\n")

    rec = _drive_main(monkeypatch, tmp_path, [*memos, str(junk)], "s1")
    assert rec["outcome"] == "injected"
    assert rec["floored"] == 1
    assert rec["floored_files"] == ["unrelated_note.md"]
    # And how well it had ranked before the floor took it: floored-from-the-top
    # says the floor is overruling the ranker, floored-from-the-tail says they
    # agree. The stub's scores descend with rank, so this one is the runner-up.
    assert rec["floored_scores"] == [0.95]

    # Absent rather than empty when nothing was dropped: a key present on
    # every line is a key nobody greps for.
    rec = _drive_main(monkeypatch, tmp_path, memos, "s2")
    assert rec["floored"] == 0
    assert "floored_files" not in rec
    assert "floored_scores" not in rec


# --- --search CLI (subprocess: argv routing is only real from outside) --------


def _cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", HOOK, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env(tmp_path),
    )


def _last_record(tmp_path: Path) -> dict:
    log = tmp_path / ".cache" / "memory-recall" / "log.jsonl"
    return json.loads(log.read_text().splitlines()[-1])


def test_search_cli_prints_the_pointers_it_would_have_injected(tmp_path) -> None:
    corpus = tmp_path / "plans"
    corpus.mkdir()
    for name, desc in (
        ("botbumps.md", "the bot is the sole lockfile writer"),
        ("bumps.md", "How input bumps reach the fleet"),
    ):
        (corpus / name).write_text(
            f"---\ndescription: {desc}\ntype: reference\n---\n\n"
            "# Rollout\n\nautobump opens the flake lock bump\n"
        )
    out = _cli(tmp_path, "--search", "autobump flake lock", "--dir", str(corpus))

    assert out.returncode == 0
    # Same lines main() injects: home-relative path, the file's own
    # description, the term-overlap evidence, and the matching section.
    assert "~/plans/botbumps.md — the bot is the sole lockfile writer" in out.stdout
    assert "[matches 3/3 prompt terms: autobump, flake, lock]" in out.stdout
    assert "[section: Rollout]" in out.stdout

    rec = _last_record(tmp_path)
    # Observable in the soak log, distinctly, and without touching session
    # state: a CLI run must not spend a session's pointer budget.
    assert rec["outcome"] == "cli" and rec["session"] == "cli" and rec["shown"] == 2
    assert not list((tmp_path / ".cache" / "memory-recall").glob("*.json"))


def test_search_cli_uses_grep_exit_codes(tmp_path) -> None:
    # Nothing content-bearing survives the stopword strip, so no corpus is
    # opened at all: the one empty result that cannot be a failed dir in
    # disguise, and the only way to assert 1 without a real corpus.
    nothing = _cli(tmp_path, "--search", "the and of")
    assert nothing.returncode == 1 and nothing.stdout == ""

    # 2 is "the search itself failed", never "no such memory" — an agent
    # reading a crash as absence stops looking.
    bad = _cli(tmp_path, "--nope")
    assert bad.returncode == 2 and bad.stdout == ""
    assert "usage:" in bad.stderr

    # A --dir the caller NAMED and that is not there is the same class of
    # failure: silently searching nothing and calling it absence is how a
    # typo becomes "there is no memory about this".
    typo = _cli(
        tmp_path, "--search", "autobump flake lock", "--dir", str(tmp_path / "x")
    )
    assert typo.returncode == 2 and typo.stdout == ""
    assert "not a directory" in typo.stderr


def test_no_argv_still_reads_the_hook_payload_from_stdin(tmp_path) -> None:
    """The argv branch must not swallow the path the harness actually uses:
    UserPromptSubmit invokes this file with no arguments and a JSON payload."""
    out = subprocess.run(
        ["python3", HOOK],
        input=json.dumps({"session_id": "stdin", "prompt": "hi"}),
        capture_output=True,
        text=True,
        timeout=30,
        env=_env(tmp_path),
    )
    assert out.returncode == 0  # fail-open, unlike the CLI
    rec = _last_record(tmp_path)
    assert rec["outcome"] == "gate:shape" and rec["session"] != "cli"


# --- the time budget ---------------------------------------------------------
#
# The harness kills this hook at HARNESS_TIMEOUT and the soak log is written
# LAST, so an overrun erases its own evidence: the run that most needs a record
# is the run that leaves none. These tests pin the two halves of the answer —
# never start work that cannot finish in the budget, and write something even
# when killed anyway.


def _stub_dirs(monkeypatch, dirs: list[str]) -> list[str]:
    """Stub retrieval over `dirs`; return the list of dirs actually searched."""
    searched: list[str] = []

    def fake_fts(query, d):
        searched.append(d)
        return [f"{d}/a.md"]

    monkeypatch.setattr(hook, "_search_dirs", lambda: dirs)
    monkeypatch.setattr(hook, "_fts_dir", fake_fts)
    return searched


def test_the_budget_ends_before_the_harness_kills_the_hook() -> None:
    # Not a tautology: these numbers live in two files (the harness's own
    # settings entry carries the timeout) and drifted apart once already.
    assert hook.BUDGET_SECONDS < hook.HARNESS_TIMEOUT


def test_a_dir_past_the_deadline_is_skipped_not_started(monkeypatch) -> None:
    # Skipped, and SAID so: a corpus that was never searched must not read as
    # a corpus that was searched and failed (errs_lex), which is the
    # distinction the soak log is mined for. The deadline still has something
    # to bound with no subprocess in the picture — a cold or invalidated index
    # makes the first dir's sync re-chunk the whole corpus, and the second
    # would start its own with the budget already gone.
    searched = _stub_dirs(monkeypatch, ["/a", "/b"])
    rec: dict = {}
    hook.recall("unionfs mount permissions", stats=rec, deadline=time.monotonic() - 1)
    assert searched == []
    assert rec["skipped_lex"] == 2 and rec["errs_lex"] == 0


def test_no_deadline_means_no_clock(monkeypatch) -> None:
    # --search and the eval are not on a prompt's critical path.
    searched = _stub_dirs(monkeypatch, ["/a", "/b"])
    rec: dict = {}
    hook.recall("unionfs mount permissions", stats=rec)
    assert searched == ["/a", "/b"]
    assert "skipped_lex" not in rec


def test_sigterm_writes_the_record_the_harness_would_have_erased(tmp_path) -> None:
    # A FIFO named like a memory is what makes this deterministic: the corpus
    # walk opens every .md it finds, and a FIFO with no writer blocks that
    # open forever, so the hook is reliably inside the lexical stage when the
    # signal lands. (Before the semantic stage was deleted this parked on a
    # `ck` that slept.)
    for rel in (PROJECT_DIR, PERSONAL_DIR):
        (tmp_path / rel / "search").mkdir(parents=True, exist_ok=True)
    personal = tmp_path / PERSONAL_DIR / "search"
    os.mkfifo(personal / "blocks_the_walk.md")
    env = _env(tmp_path)
    with subprocess.Popen(
        ["python3", HOOK],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    ) as proc:
        assert proc.stdin is not None
        proc.stdin.write(
            json.dumps({"session_id": "kill", "prompt": "unionfs mount permissions"})
        )
        proc.stdin.close()
        time.sleep(1.0)  # into the walk, blocked on the FIFO
        proc.terminate()
        proc.wait(timeout=10)
    rec = _last_record(tmp_path)
    assert rec["outcome"] == "killed"
    assert "ms" in rec and rec["prompt_sha"]


def test_a_kill_before_the_payload_arrives_still_exits_clean() -> None:
    # The test above kills mid-search, which is the interesting window but not
    # the first one: the hook is blocked in json.load from its first statement
    # until the harness writes the payload, and until this fix that window had
    # no handler at all. Nothing is written to stdin here, so the read is still
    # blocked when the signal lands. Reproduced at rc=-15 on main.
    proc = subprocess.Popen(
        ["python3", HOOK],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.0)  # well past interpreter start, still waiting on stdin
    proc.terminate()
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"rc={proc.returncode}, stderr={err}"
    assert out == "" and err == ""


# Deliver the signal at one exact instant — the moment the record reaches the
# file — without a sleep to race or a source line number to drift: the patched
# _soak_log signals its own process on the way out. Has to be a subprocess,
# because the handler ends in os._exit and would take pytest with it.
_KILL_AS_THE_RECORD_LANDS = """
import importlib.util, json, os, signal, sys

spec = importlib.util.spec_from_file_location("hook", sys.argv[1])
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

fd = os.open(sys.argv[2], os.O_WRONLY | os.O_APPEND | os.O_CREAT)
fired = []


def soak(rec):
    os.write(fd, (json.dumps(rec) + "\\n").encode())
    if not fired:
        fired.append(1)
        os.kill(os.getpid(), signal.SIGTERM)


h._soak_log = soak
h.main()
"""


def test_a_kill_as_the_record_lands_leaves_exactly_one(tmp_path) -> None:
    # `logged` flips after the write returns, so a signal arriving in between
    # found a hook that believed it had recorded nothing and appended a second
    # `killed` record for the same prompt. Every rate the analyzers report is a
    # count over records, so one prompt with two records is one prompt counted
    # twice. Masking SIGTERM across the pair closes it. Measured on main as
    # ['gate:shape', 'killed'].
    records = tmp_path / "records.jsonl"
    proc = subprocess.run(
        ["python3", "-c", _KILL_AS_THE_RECORD_LANDS, HOOK, str(records)],
        input=json.dumps({"session_id": "sigwin", "prompt": "hi"}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    outcomes = [json.loads(x)["outcome"] for x in records.read_text().splitlines()]
    assert outcomes == ["gate:shape"], outcomes


def test_the_module_imports_under_python_39(tmp_path) -> None:
    """The harness picks the interpreter, and on a stock macOS that is
    /usr/bin/python3 (3.9). A module-scope `X | None` raises TypeError at
    IMPORT there — and this hook is fail-open, so the crash presents as a
    corpus with nothing to say. `from __future__ import annotations` is what
    keeps 3.10+ annotation syntax legal below it; nothing else in the test
    suite would notice it going away, since CI runs 3.13.
    """
    src = Path(HOOK).read_text()
    assert "from __future__ import annotations\n" in src
    py39 = Path("/usr/bin/python3")
    if not py39.exists():
        pytest.skip("no system python to check the real 3.9 path against")
    out = subprocess.run(
        [str(py39), HOOK],
        input=json.dumps({"session_id": "py39", "prompt": "hi"}),
        capture_output=True,
        text=True,
        timeout=30,
        env=_env(tmp_path),
    )
    assert out.returncode == 0, out.stderr
    assert "TypeError" not in out.stderr


# --- what a downstream consumer's inverse test pins from over here ------------
#
# Two invariants whose OTHER half lives in a consumer repo. Both halves are
# named in the extraction's assertion inventory; these are the memkit-side
# ones, and they are here rather than there because a memkit PR must redden
# memkit's own CI — a check that only fires after a downstream version bump is
# a check that fires too late to stop the bump.


def test_every_gated_marker_has_a_probe_that_the_gate_recognises() -> None:
    """The synthesizer is the shared artifact; this half asserts it is honest.

    A consumer's transcript stratifier has to classify every marker this hook
    gates on, and the two files cannot import each other's regexes. The claim
    had already inverted once: five markers were added to the hook and none to
    the stratifier, and an exhaustive replay then reported ~1900 false
    positives that belonged to the instrument rather than to the gate.

    Here: every synthesized probe must actually be gated. If it is not, the
    synthesizer has drifted from the pattern it derives from, and the
    consumer's half is comparing against strings this hook never sees.
    """
    probes = hook.envelope_probes()
    assert len(probes) >= 15, probes
    for probe in probes:
        assert hook._is_envelope(probe), probe


def test_the_probe_synthesizer_refuses_a_construct_it_was_not_taught(
    monkeypatch,
) -> None:
    # A silently-skipped alternative is an untested marker with a cleaner
    # looking suite, so the synthesizer raises rather than dropping it.
    monkeypatch.setattr(
        hook, "_ENVELOPE", re.compile(r"^(?:<(?:task-notification|thing\d+)\b)")
    )
    with pytest.raises(RuntimeError, match="unhandled regex construct"):
        hook.envelope_probes()


def test_no_fixture_session_id_can_pass_for_a_real_one() -> None:
    """No literal in this file may look like a real session id.

    Downstream, the soak-log analyzers separate real records from harness ones
    by the SHAPE of the session field, so a fixture id shaped like a real one
    lands in somebody's measured numbers as a prompt a human typed. The full
    check is an AST scan that lives with those analyzers; this is the cheap
    superset — a regex over the whole file, which catches strictly more than
    the scanner would and needs nothing from the consumer.

    A false positive here is a fixture string that happens to contain a
    hex-uuid-shaped run. The fix is to reword the fixture.
    """
    real = re.compile(r"[0-9a-f]{8}-[0-9a-f]{3}")
    for n, line in enumerate(Path(__file__).read_text().splitlines(), 1):
        assert not real.search(line), f"{__file__}:{n} — {line.strip()}"
