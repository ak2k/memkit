"""What the eval's exit code is allowed to mean.

Driven as a SUBPROCESS, because the claim is entirely about exit status and
`main()` reaches it through `sys.exit`. The eval is a CI gate, so its exit code
is its whole interface to the thing consuming it.

The property under test is one: every way of NOT gating is non-zero. A run that
gated everything and found nothing wrong, and a run that gated nothing at all,
must not be the same answer — the second is the commoner state on a store
anybody edits, and it read as green for most of this check's life.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from memkit import eval_memory_recall as ev
from memkit import memory_prompt_recall as hook

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A writable copy of the fixture corpus, config and snapshot.

    Copied rather than pointed at: these cases drift the corpus and re-baseline
    it, and the committed fixture is what every other gate in this repo scores
    against. The config resolves its roots `config_relative` from itself, so a
    copy is self-contained with nothing to redirect.
    """
    dst = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, dst)
    # copytree preserves mode, and the source is read-only under `nix flake
    # check` because the fixtures live in the store. The flake's own
    # fixture-eval check chmods after its `cp -r` for exactly this reason.
    for path in (dst, *dst.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    return dst


def _eval(
    corpus: Path,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "memkit.eval_memory_recall",
            "--config",
            str(corpus / "memkit.json"),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=cwd,
    )


def _drift(corpus: Path) -> None:
    """Edit a memory without moving any case's outcome.

    What a typical memory edit looks like: the corpus differs from the one the
    snapshot was written on, and every case still scores as recorded.
    """
    memo = corpus / "corpus" / "project" / "search" / "flange_torque.md"
    memo.write_text(memo.read_text() + "\nA sentence nobody searches for.\n")


SNAPSHOT = "eval-expectations.json"

# The rule a failing run prints, quoted in pieces: one half for a change that
# edits no memory, one for a change that does.
NO_REBASELINE = "edits no memory store and no case"
REVIEW_THEN_UPDATE = "review what moved, then --update-snapshot"


def _recorded(corpus: Path) -> dict:
    return json.loads((corpus / SNAPSHOT).read_text(encoding="utf-8"))


def _record(corpus: Path, state: dict) -> None:
    (corpus / SNAPSHOT).write_text(json.dumps(state), encoding="utf-8")


def _line(stdout: str, needle: str) -> str:
    return next(ln for ln in stdout.splitlines() if needle in ln)


def test_the_committed_fixture_corpus_gates_clean(corpus: Path) -> None:
    """The control. Without it every case below could pass because the eval is
    broken in some way that has nothing to do with the snapshot."""
    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "every gating case matched the snapshot" in out.stdout


def test_a_memory_edit_that_moves_no_outcome_passes_without_a_re_baseline(
    corpus: Path,
) -> None:
    """The common memory commit. Every case was compared against the corpus in
    front of it and every one held, so the run gated everything and found
    nothing wrong — a pass, and the snapshot is still true as written."""
    before = (corpus / SNAPSHOT).read_bytes()
    _drift(corpus)
    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no re-baseline needed" in out.stdout, out.stdout
    assert "[PASS" in out.stdout
    assert (corpus / SNAPSHOT).read_bytes() == before, "a gating run wrote the snapshot"


@pytest.mark.parametrize("edit_memory", [False, True], ids=["corpus-as-recorded", "memory-edited"])
def test_a_moved_outcome_fails_and_says_how_to_attribute_it(
    corpus: Path, edit_memory: bool
) -> None:
    """A recorded outcome that no longer holds fails the run whatever the
    corpus did, and the run cannot say whether the corpus or the retriever
    moved it — so it prints the rule that can, both halves of it.

    Every case in the slice moved, so the slice compared all of them and none
    held: a refusal for comparing nothing would be the wrong message here.
    """
    if edit_memory:
        _drift(corpus)
    state = _recorded(corpus)
    for row in state["cases"]["noinject"].values():
        row["status"] = "NOINJECT-FAIL"
    _record(corpus, state)

    out = _eval(corpus)
    assert out.returncode == 2, out.stdout + out.stderr
    row = _line(out.stdout, "thanks, that is exactly")
    assert "<- MOVED (snapshot says NOINJECT-FAIL)" in row, row
    assert "2 gating failure(s)" in out.stdout, out.stdout
    assert NO_REBASELINE in out.stdout, out.stdout
    assert "do not re-baseline" in out.stdout, out.stdout
    assert REVIEW_THEN_UPDATE in out.stdout, out.stdout
    assert "nothing was gated" not in out.stderr, out.stderr
    assert "no re-baseline needed" not in out.stdout


def test_an_unrecorded_case_gates_after_a_memory_edit(corpus: Path) -> None:
    """A case nobody baselined is a case asserting nothing until somebody
    happens to re-baseline, so adding one would be the way to add an ungated
    case. It gates whatever else the change edited."""
    _drift(corpus)
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    state["eval"]["cases"]["noinject"].append(
        {"prompt": "what time zone is the standup in"}
    )
    config.write_text(json.dumps(state))

    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    row = _line(out.stdout, "what time zone is the standup in")
    assert "<- NEW (no expectation recorded)" in row, row
    assert "1 gating failure(s)" in out.stdout, out.stdout


def test_a_failure_count_past_255_still_exits_non_zero(corpus: Path) -> None:
    """A process exit status is taken mod 256, so a count passed through as
    the status reads 256 failures as a pass. The count saturates at 255 and
    the summary line still carries the whole of it."""
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    state["eval"]["cases"]["noinject"].extend(
        {"prompt": f"what time zone is standup number {i} in"} for i in range(256)
    )
    config.write_text(json.dumps(state))

    out = _eval(corpus)
    assert out.returncode == 255, out.stdout[-2000:] + out.stderr
    assert "256 gating failure(s)" in out.stdout, out.stdout[-2000:]


def test_a_memory_moved_between_tiers_gates(corpus: Path) -> None:
    """Moving a memory from hot/ to search/ flips what every case about it
    asserts, from abstention to injection, with no byte of it edited. The
    snapshot's row then answers a different question, and leaving it ungated
    until somebody re-baselines is the inert gate this rule closes."""
    project = corpus / "corpus" / "project"
    (project / "hot" / "gasket_replacement.md").rename(
        project / "search" / "gasket_replacement.md"
    )
    out = _eval(corpus)
    assert out.returncode == 1, out.stdout + out.stderr
    row = _line(out.stdout, "gasket replacement interval")
    assert "<- DRIFT (snapshot says hot, now search)" in row, row
    assert "1 gating failure(s)" in out.stdout, out.stdout
    assert NO_REBASELINE in out.stdout, out.stdout


def test_a_case_pointed_at_another_memory_gates(corpus: Path) -> None:
    """The corpus untouched, the case retargeted. What is asserted changed,
    so the recorded outcome is about a different file and has to be
    re-baselined in the same change rather than left standing ungated."""
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    for case in state["eval"]["cases"]["suite"]:
        if case["file"] == "widget_calibration.md":
            case["file"] = "sprocket_alignment.md"
    config.write_text(json.dumps(state))

    out = _eval(corpus)
    assert out.returncode == 1, out.stdout + out.stderr
    row = _line(out.stdout, "recalibrate a widget")
    assert (
        "<- DRIFT (snapshot says widget_calibration.md, case now names "
        "sprocket_alignment.md)" in row
    ), row
    assert "1 gating failure(s)" in out.stdout, out.stdout


def test_a_gating_slice_that_ran_no_case_is_refused_after_a_memory_edit(
    corpus: Path,
) -> None:
    """Zero failures and zero cases print the same exit code, and an empty
    slice would buy a pass by having no expectations to fail. That holds in
    every run, a memory edit included."""
    _drift(corpus)
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    state["eval"]["cases"]["noinject"] = []
    config.write_text(json.dumps(state))

    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "nothing was gated: the noinject slice ran 0 cases" in out.stderr, (
        out.stderr
    )


def test_a_gating_slice_of_unrecorded_cases_fails_on_each_of_them(
    corpus: Path,
) -> None:
    """A slice whose every case is new ran every one of them and each one
    failed, so the run reports that many gating failures. It gated all of
    them, and a refusal saying it gated nothing would misstate the run."""
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    state["eval"]["cases"]["noinject"] = [
        {"prompt": "what time zone is the standup in"},
        {"prompt": "who is buying lunch on friday"},
        {"prompt": "remind me what the parking code is"},
    ]
    config.write_text(json.dumps(state))
    # A slice nobody has baselined, so no row of the cases it replaced is left
    # to fail beside these three.
    recorded = _recorded(corpus)
    recorded["cases"]["noinject"] = {}
    _record(corpus, recorded)

    out = _eval(corpus)
    assert out.returncode == 3, out.stdout + out.stderr
    assert "3 gating failure(s)" in out.stdout, out.stdout
    assert "nothing was gated" not in out.stderr, out.stderr


def test_a_moved_outcome_outside_the_gate_reports_without_failing(
    corpus: Path,
) -> None:
    """`vocab` is not a gating slice in the fixture config, so its rows report
    and never decide the exit code, whatever else the change edited. Something
    did move, so the pass does not say the snapshot needs no re-baseline."""
    _drift(corpus)
    state = _recorded(corpus)
    for row in state["cases"]["vocab"].values():
        row["status"] = "VOCAB-MISS"
    _record(corpus, state)

    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    row = _line(out.stdout, "the machine reads its old zero")
    assert "<- MOVED (snapshot says VOCAB-MISS; not gating)" in row, row
    assert "--update-snapshot accepts these" in out.stdout, out.stdout
    assert NO_REBASELINE not in out.stdout
    assert "no re-baseline needed" not in out.stdout, out.stdout


@pytest.mark.parametrize(
    "slice_,code", [("noinject", 1), ("vocab", 0)], ids=["gating", "report-only"]
)
def test_a_stale_row_gates_in_a_gating_slice_and_reports_outside_one(
    corpus: Path, slice_: str, code: int
) -> None:
    """A case deleted from the config leaves its row behind, and no case runs
    it. A gating slice's rows are the record of what its gate checks, so in
    one a row no case asks is a mismatch like any other: deleting the case
    fails the run until a re-baseline drops the row in a diff a reviewer sees.
    Outside the gate the row reports, and still wants that re-baseline."""
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    gone = state["eval"]["cases"][slice_].pop()["prompt"]
    config.write_text(json.dumps(state))

    out = _eval(corpus)
    assert out.returncode == code, out.stdout + out.stderr
    row = _line(out.stdout, gone[:40])
    assert f"in the snapshot's {slice_} slice, not in the suite" in row, row
    assert ("not gating" in row) is (code == 0), row
    assert f"{code} gating failure(s)" in out.stdout, out.stdout
    assert "no re-baseline needed" not in out.stdout, out.stdout
    if not code:
        assert "--update-snapshot accepts these" in out.stdout, out.stdout
        return
    assert REVIEW_THEN_UPDATE in out.stdout, out.stdout
    # The sanctioned path drops the row, and the gate is green after it.
    assert _eval(corpus, "--update-snapshot").returncode == 0
    assert gone not in _recorded(corpus)["cases"][slice_]
    assert _eval(corpus).returncode == 0


@pytest.mark.parametrize("slice_", ["suite", "noinject", "vocab"])
def test_a_prompt_repeated_in_a_slice_is_refused(corpus: Path, slice_: str) -> None:
    """The snapshot keys a slice's rows by prompt, so a second case with the
    same prompt overwrites the first one's row. A re-baseline then records one
    of the two, and every later run fails the other as drift, which no
    re-baseline can clear. One prompt per slice is the contract, refused at
    load like a brief named twice."""
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    cases = state["eval"]["cases"][slice_]
    twin = dict(cases[0])
    if "file" in twin:
        twin["file"] = "sprocket_alignment.md"
    cases.append(twin)
    config.write_text(json.dumps(state))

    for args in ((), ("--update-snapshot",)):
        out = _eval(corpus, *args)
        assert out.returncode == 1, out.stdout + out.stderr
        assert (
            f"the {slice_} slice names the same prompt twice: {twin['prompt']!r}"
            in out.stderr
        ), out.stderr


def test_update_snapshot_writes_the_outcomes_and_no_fingerprint(
    corpus: Path,
) -> None:
    """A re-baseline writes, and exits 0 — accepting what the run reported is
    the act, so a non-zero here would be indistinguishable from a refusal. It
    records outcomes only: nothing reads which corpus they were measured on."""
    _drift(corpus)
    state = _recorded(corpus)
    prompt = next(iter(state["cases"]["noinject"]))
    state["cases"]["noinject"][prompt]["status"] = "NOINJECT-FAIL"
    _record(corpus, state)

    out = _eval(corpus, "--update-snapshot")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "wrote" in out.stdout
    written = _recorded(corpus)
    assert set(written) == {"note", "cases"}, sorted(written)
    assert written["cases"]["noinject"][prompt]["status"] == "NOINJECT-OK"

    # And the gate is live again immediately, which is what makes the remedy a
    # remedy rather than a way to switch the check off.
    assert _eval(corpus).returncode == 0


@contextlib.contextmanager
def _an_index_that_cannot_answer(tmp_path: Path) -> Iterator[dict[str, str]]:
    """An environment whose state directory the run cannot write.

    `recall` returns no hits for such an index rather than raising, so every
    search case scores a MISS and every abstention and noinject case passes.
    """
    if os.geteuid() == 0:
        pytest.skip("root writes everything, so this cannot be staged")
    cache = tmp_path / "cache"
    state = cache / "memory-recall"
    state.mkdir(parents=True)
    state.chmod(0o500)
    try:
        yield {**os.environ, "XDG_CACHE_HOME": str(cache)}
    finally:
        state.chmod(0o700)


def _searches(corpus: Path) -> int:
    """How many cases a run over this config takes to retrieval. Every fixture
    case does, on the prompt path; a brief does when the task path's own gate
    lets it through."""
    data = json.loads((corpus / "memkit.json").read_text())
    calls = sum(len(cases) for cases in data["eval"]["cases"].values())
    if "long_briefs" in data["eval"]:
        briefs = ev.long_brief_set(corpus / data["eval"]["long_briefs"])
        calls += sum(
            hook.task_gate(case["brief"]) is None
            for half in ("served", "unserved")
            for case in briefs[half]
        )
    return calls


def _store_searches(corpus: Path) -> int:
    """How many store searches a run over this config makes: one per store for
    every case that reaches retrieval."""
    data = json.loads((corpus / "memkit.json").read_text())
    return _searches(corpus) * len(data["stores"])


def _shape(corpus: Path, shape: str) -> None:
    config = corpus / "memkit.json"
    data = json.loads(config.read_text())
    if shape == "no-long-briefs":
        del data["eval"]["long_briefs"]
        data["eval"]["gating_slices"] = ["noinject", "suite"]
    elif shape == "long-briefs-only":
        data["eval"]["cases"] = {}
        data["eval"]["gating_slices"] = ["longbrief"]
    config.write_text(json.dumps(data))


@pytest.mark.parametrize("shape", ["as-shipped", "no-long-briefs", "long-briefs-only"])
def test_a_re_baseline_from_an_index_that_cannot_answer_writes_nothing(
    corpus: Path, tmp_path: Path, shape: str
) -> None:
    """Written, a run on an index that cannot answer is a snapshot of the
    failure, and every later run that cannot write there either matches it.

    Without the long-brief slice no rate fails either, so nothing but this
    refusal stands between that run and an exit 0. With only that slice, the
    task path's searches are the only ones that could have failed. The count
    is of store searches, one per store per case, so a two-store fixture
    reports its searches rather than dozens of stores."""
    _shape(corpus, shape)
    before = (corpus / SNAPSHOT).read_bytes()
    with _an_index_that_cannot_answer(tmp_path) as env:
        out = _eval(corpus, "--update-snapshot", env=env)
    assert "search tier: 0/" in out.stdout, out.stdout
    assert out.returncode != 0, out.stdout
    assert (
        f"refusing to write a snapshot: {_store_searches(corpus)} store "
        "search(es) failed in this run" in out.stderr
    ), out.stderr
    assert "wrote" not in out.stdout, out.stdout
    assert (corpus / SNAPSHOT).read_bytes() == before, "the snapshot was rewritten"


def test_a_gating_run_on_an_index_that_cannot_answer_fails(
    corpus: Path, tmp_path: Path
) -> None:
    """The gating half of the refusal above. A snapshot that recorded what a
    dead index scores — every search case a MISS, every abstention and noinject
    case a pass — matches a dead index case for case, so the snapshot alone
    reads that run as a pass over every gating case."""
    _shape(corpus, "no-long-briefs")
    state = _recorded(corpus)
    del state["cases"]["longbrief"]
    for slice_ in ("suite", "vocab"):
        for row in state["cases"][slice_].values():
            row["status"] = {"PASS": "MISS", "VOCAB-FOUND": "VOCAB-MISS"}.get(
                row["status"], row["status"]
            )
    _record(corpus, state)
    with _an_index_that_cannot_answer(tmp_path) as env:
        out = _eval(corpus, env=env)
    assert "search tier: 0/" in out.stdout, out.stdout
    assert "0 gating failure(s)" in out.stdout, out.stdout
    assert out.returncode == 1, out.stdout + out.stderr
    assert (
        f"refusing to gate: {_store_searches(corpus)} store search(es) failed "
        "in this run" in out.stderr
    ), out.stderr
    assert "no re-baseline needed" not in out.stdout, out.stdout


@contextlib.contextmanager
def _an_index_another_process_is_writing(cache: Path) -> Iterator[None]:
    """Every index under `cache` held mid-write by another connection, as a
    concurrent session's hook holds it while its own sync runs. A sync that
    needs the lock waits out the busy timeout and is skipped, and the query
    still answers from the rows the index held before."""
    held = []
    try:
        for db in sorted((cache / "memory-recall").glob("fts5-*.db")):
            con = sqlite3.connect(db, isolation_level=None)
            held.append(con)
            con.execute("BEGIN IMMEDIATE")
        assert held, "no index was built to hold"
        yield
    finally:
        for con in held:
            con.close()


# The memory the flange cases target, rewritten so that none of them finds it.
UNRELATED = (
    "---\nname: office_plants\ndescription: The office plants are watered on "
    "Fridays.\ntype: reference\n---\n\nWater the office plants on Fridays.\n"
)


@pytest.mark.parametrize("args", [(), ("--update-snapshot",)], ids=["gate", "write"])
@pytest.mark.parametrize(
    "shape,needle,held_status",
    [
        ("no-long-briefs", "flange fastener tightening", "[PASS"),
        ("long-briefs-only", "vessel-reassembly", "[BRIEF-SERVED"),
    ],
    ids=["prompt-path", "task-path"],
)
def test_a_run_whose_sync_lost_the_lock_neither_gates_nor_writes(
    corpus: Path,
    tmp_path: Path,
    shape: str,
    needle: str,
    held_status: str,
    args: tuple,
) -> None:
    """A sync that loses the write lock to another process is skipped, and the
    query answers from the rows the index held before this change's memory
    edit. The outcomes then match the snapshot because the run measured the
    corpus as it was, so the run may neither pass the gate nor become the
    baseline."""
    _shape(corpus, shape)
    if shape == "no-long-briefs":
        state = _recorded(corpus)
        del state["cases"]["longbrief"]
        _record(corpus, state)
    before = (corpus / SNAPSHOT).read_bytes()
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    warm = _eval(corpus, env=env)
    assert warm.returncode == 0, warm.stdout + warm.stderr
    (corpus / "corpus" / "project" / "search" / "flange_torque.md").write_text(
        UNRELATED, encoding="utf-8"
    )
    with _an_index_another_process_is_writing(tmp_path / "cache"):
        out = _eval(corpus, *args, env=env)
    row = _line(out.stdout, needle)
    assert row.startswith(held_status), row
    assert out.returncode != 0, out.stdout + out.stderr
    assert (
        f"{_searches(corpus)} case(s) scored on an index whose sync left memory "
        "files out of it (lex_busy_skip)" in out.stderr
    ), out.stderr
    assert "wrote" not in out.stdout, out.stdout
    assert (corpus / SNAPSHOT).read_bytes() == before, "the snapshot was rewritten"

    # Non-vacuity: the rows the index held are what answered. A run that
    # syncs the edit moves the same case.
    synced = _eval(corpus, env=env)
    assert "scored on an index whose sync" not in synced.stderr, synced.stderr
    assert "<- MOVED" in _line(synced.stdout, needle), synced.stdout
    assert synced.returncode != 0, synced.stdout


# Memories each path must not deliver: one a noinject prompt asks about, one an
# unserved brief does.
ABOUT_A_NOINJECT_PROMPT = (
    "---\nname: linked_list_reversal\ndescription: Reverse a linked list in "
    "place with three pointers.\ntype: reference\n---\n\nTo reverse a linked "
    "list in place, walk the list once and flip each next pointer.\n"
)
ABOUT_AN_UNSERVED_BRIEF = (
    "---\nname: warehouse_slotting\ndescription: Re-slot a warehouse from the "
    "order history: fast movers near the pick faces, pickers walk less.\ntype: "
    "reference\n---\n\nSlot the warehouse from the order history, not the "
    "current layout. Fast movers go to the widest aisles; heavy and bulky items "
    "stay off the mezzanine; print the pick-face labels on the move plan's "
    "timeline; measure walking distance per order.\n"
)


@pytest.mark.parametrize("args", [(), ("--update-snapshot",)], ids=["gate", "write"])
@pytest.mark.parametrize(
    "shape,memory,needle,quiet,hide,counter",
    [
        (
            "no-long-briefs", ABOUT_A_NOINJECT_PROMPT, "reverse a linked list",
            "[NOINJECT-OK", "file", "lex_spared",
        ),
        (
            "long-briefs-only", ABOUT_AN_UNSERVED_BRIEF, "warehouse-slotting",
            "[BRIEF-QUIET", "file", "lex_spared",
        ),
        (
            "no-long-briefs", ABOUT_A_NOINJECT_PROMPT, "reverse a linked list",
            "[NOINJECT-OK", "dir", "lex_unwalked",
        ),
    ],
    ids=["prompt-path-unreadable", "task-path-unreadable", "prompt-path-unlisted"],
)
def test_a_run_whose_sync_could_not_read_a_memory_neither_gates_nor_writes(
    corpus: Path,
    tmp_path: Path,
    shape: str,
    memory: str,
    needle: str,
    quiet: str,
    hide: str,
    counter: str,
    args: tuple,
) -> None:
    """A memory the sync cannot read, or one in a directory it cannot list,
    keeps whatever rows the index already held for it, and a new one has none.
    The query answers without it, so a memory edit that should move a case
    matches the snapshot instead, and the run may neither pass the gate nor
    become the baseline. The refusal names the counter that fired."""
    if os.geteuid() == 0:
        pytest.skip("root reads everything, so this cannot be staged")
    _shape(corpus, shape)
    if shape == "no-long-briefs":
        state = _recorded(corpus)
        del state["cases"]["longbrief"]
        _record(corpus, state)
    before = (corpus / SNAPSHOT).read_bytes()
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    warm = _eval(corpus, env=env)
    assert warm.returncode == 0, warm.stdout + warm.stderr
    folder = corpus / "corpus" / "project" / "search" / "added"
    folder.mkdir()
    memo = folder / "added_memory.md"
    memo.write_text(memory, encoding="utf-8")
    hidden = memo if hide == "file" else folder
    hidden.chmod(0)
    try:
        out = _eval(corpus, *args, env=env)
    finally:
        hidden.chmod(0o700)
    assert _line(out.stdout, needle).startswith(quiet), out.stdout
    assert out.returncode != 0, out.stdout + out.stderr
    assert (
        f"{_searches(corpus)} case(s) scored on an index whose sync left memory "
        f"files out of it ({counter})" in out.stderr
    ), out.stderr
    assert "wrote" not in out.stdout, out.stdout
    assert (corpus / SNAPSHOT).read_bytes() == before, "the snapshot was rewritten"

    # Non-vacuity: once the sync can read it, the same memory moves the case.
    synced = _eval(corpus, env=env)
    assert "scored on an index whose sync" not in synced.stderr, synced.stderr
    assert "<- MOVED" in _line(synced.stdout, needle), synced.stdout
    assert synced.returncode != 0, synced.stdout


def test_a_memory_over_the_size_cap_neither_refuses_the_gate_nor_the_write(
    corpus: Path, tmp_path: Path
) -> None:
    """The index declines a file over its size cap on every run and holds no
    rows for it, so no rerun would read it and the run already measures the
    corpus the hook can see. The sync counts it as spared all the same, and
    that alone must not refuse the run."""
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    memo = corpus / "corpus" / "project" / "search" / "oversize.md"
    memo.write_bytes(b"flange " * (hook.INDEX_FILE_MAX_BYTES // 7 + 1))
    assert memo.stat().st_size > hook.INDEX_FILE_MAX_BYTES
    out = _eval(corpus, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "every gating case matched the snapshot" in out.stdout, out.stdout
    written = _eval(corpus, "--update-snapshot", env=env)
    assert written.returncode == 0, written.stdout + written.stderr
    assert "wrote" in written.stdout, written.stdout


@pytest.mark.parametrize(
    "rec,gaps",
    [
        ({}, ()),
        ({"errs_lex": 0, "lex_hits": 3, "lex_outside": 1, "lex_secret": 2}, ()),
        ({"lex_spared": 2, "lex_oversize": 2}, ()),
        ({"lex_spared": 3, "lex_oversize": 2}, ("lex_spared",)),
        ({"lex_busy_skip": 1}, ("lex_busy_skip",)),
        ({"lex_unwalked": 1}, ("lex_unwalked",)),
        ({"lex_deadline": 4, "lex_spared": 4}, ("lex_deadline", "lex_spared")),
        ({"lex_unswept": 5}, ("lex_unswept",)),
        ({"skipped_lex": 1}, ("skipped_lex",)),
    ],
    ids=[
        "clean", "deliberate-refusals", "oversize-only", "spared-past-oversize",
        "busy", "unwalked", "deadline", "unswept", "store-skipped",
    ],
)
def test_a_sync_gap_is_every_counter_that_leaves_a_memory_unindexed(
    rec: dict, gaps: tuple
) -> None:
    """The counters that cannot be staged cheaply through a whole run, read
    the way the run reads them: off one search's record. A refusal the index
    makes on every run — a link out of the store, a file over the cap, a
    credential the scan matched — leaves nothing stale and is not a gap."""
    assert ev.sync_gaps(rec) == gaps


def test_a_snapshot_that_still_carries_a_fingerprint_reads_as_before(
    corpus: Path,
) -> None:
    """Consumers' committed snapshots carry a `corpus` digest until their next
    re-baseline. It names a corpus that is not this one and is ignored: the
    run gates the outcomes it records, and they all hold."""
    state = _recorded(corpus)
    state = {"note": state["note"], "corpus": "0" * 64, **state}
    _record(corpus, state)

    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "every gating case matched the snapshot" in out.stdout


def test_no_snapshot_at_all_still_refuses(corpus: Path) -> None:
    """A run with nothing to gate against must not report having gated."""
    (corpus / SNAPSHOT).unlink()
    out = _eval(corpus)
    assert out.returncode != 0
    assert "--update-snapshot" in out.stderr


# --- the long-brief slice ----------------------------------------------------
#
# Two rates over sixteen paired briefs, and they gate differently from
# everything above: the snapshot records what happened and `--update-snapshot`
# accepts it, while these record what has to be true whatever happened.

BRIEFS = "long-briefs"


def _write_brief(path, text: str) -> None:
    """A brief file IS the brief, to the byte.

    The loader refuses leading or trailing whitespace rather than trimming it,
    because a gate that trims measures a brief the fixture does not contain —
    so a test that edits one writes the stripped text, the same as a fixture
    author does.
    """
    path.write_text(text.strip(), encoding="utf-8")


def _served(corpus: Path, name: str) -> Path:
    return corpus / BRIEFS / "served" / name


def _rates(stdout: str) -> str:
    return next(ln for ln in stdout.splitlines() if ln.startswith("long briefs:"))


def _unserve(corpus: Path, *names: str) -> None:
    """Rewrite served briefs into ones the corpus has nothing to say about.

    Each gets its own subject: a case is a distinct brief, so three copies of
    one filler text are refused as one case in three places — which is a
    different failure from the coverage collapse these cases are about.
    """
    for index, name in enumerate(names):
        _write_brief(
            _served(corpus, name),
            f"# Brief {index}\n\n"
            + f"Sort the mailroom trays for round {index} by postcode. " * 200,
        )


def test_the_slice_refuses_when_the_hook_process_delivers_nothing(
    corpus: Path, tmp_path: Path
) -> None:
    """Every stage this slice drives is one of the task path's own functions,
    and none of them is the REGISTERED ENTRY POINT.

    `main`'s event dispatch, the tool-name check, the ledger write, the signal
    handlers and the stdout delivery all sit between a correct `_task_block`
    and a subagent that actually receives it. A break in any of them leaves the
    real hook emitting no `updatedInput` while this slice, calling the helpers
    directly, goes on reporting served coverage.

    Driven by renaming the tool the hook answers for, which is one of the five
    stages the finding names and is what a harness rename actually looks like:
    the process records `task:notool` and emits nothing, while every helper
    this slice calls goes on working perfectly and reporting coverage.
    """
    broken = _copy_hook(tmp_path, 'TASK_TOOL = "Agent"', 'TASK_TOOL = "Renamed"')
    out = _eval(corpus, "--hook", str(broken))
    assert out.returncode != 0, out.stdout
    assert "the hook PROCESS delivered" in out.stderr, out.stderr

    # Non-vacuity: the same run against the shipped hook is clean, so the
    # refusal above is the break's and not the check's.
    ok = _eval(corpus)
    assert ok.returncode == 0, ok.stdout + ok.stderr


def _gate_out_the_project_store(corpus: Path) -> None:
    """Only the long-brief slice, with the project store gated to the fixture
    root, so a run standing outside that root is gated out of it."""
    _shape(corpus, "long-briefs-only")
    config = corpus / "memkit.json"
    data = json.loads(config.read_text())
    for store in data["stores"]:
        if store["id"] == "project":
            store["cwd_gate"] = {"root": "self"}
    config.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "brief,code",
    [("period-close-automation", 1), ("gearbox-acceptance", 0)],
    ids=["memory-in-a-searched-store", "memory-only-in-the-gated-store"],
)
def test_all_stores_gates_a_served_row_whose_memory_this_cwd_searches(
    corpus: Path, tmp_path: Path, brief: str, code: int
) -> None:
    """`--all-stores` reads a store this cwd is gated out of, and a brief row
    whose memory lives only there is a delivery production refuses from here,
    so it reports. A row whose memory is in a store this cwd searches is one
    production delivers, and one such row moving fails the run although the
    coverage rate stays inside its slack."""
    _gate_out_the_project_store(corpus)
    state = _recorded(corpus)
    for name, row in state["cases"][ev.LONG_BRIEF_SLICE].items():
        if brief in name:
            row["status"] = "BRIEF-MISS"
    _record(corpus, state)

    out = _eval(corpus, "--all-stores", cwd=tmp_path)
    assert "which this cwd is gated out of" in out.stdout, out.stdout
    row = _line(out.stdout, brief)
    assert "<- MOVED (snapshot says BRIEF-MISS" in row, row
    assert out.returncode == code, out.stdout + out.stderr
    assert f"{code} gating failure(s) in longbrief" in out.stdout, out.stdout


# A memory the warehouse-slotting brief is about, which no other brief is.
SLOTTING = (
    "---\nname: warehouse_slotting\ndescription: Re-slotting the north "
    "warehouse before peak season puts the fast movers in the widest aisles "
    "nearest the pack bench, argued from the picked-line history.\ntype: "
    "reference\n---\n\n# Warehouse slotting\n\nWalk the aisle widths with a "
    "tape before drawing a slotting plan; the racking drawing is out of "
    "date.\nFast movers go to the wide aisles near the pack bench. Pickers walk "
    "less when the catalogue's top fifteen percent sit together.\n"
)


@pytest.mark.parametrize(
    "store,code",
    [("personal", 1), ("project", 0)],
    ids=["memory-in-a-searched-store", "memory-only-in-the-gated-store"],
)
def test_all_stores_gates_a_leak_of_a_memory_this_cwd_searches(
    corpus: Path, tmp_path: Path, store: str, code: int
) -> None:
    """The leak half of the rule above. One new memory makes one quiet brief
    leak, which is 1/16 and under the injection ceiling, so the row is the
    only thing that can fail the run. It does when production would deliver
    that memory from this cwd."""
    _gate_out_the_project_store(corpus)
    memory = corpus / "corpus" / store / "search" / "warehouse_slotting.md"
    memory.write_text(SLOTTING, encoding="utf-8")

    out = _eval(corpus, "--all-stores", cwd=tmp_path)
    row = _line(out.stdout, "warehouse-slotting")
    assert "<- MOVED (snapshot says BRIEF-QUIET" in row, row
    assert "1/16 leaked" in _line(out.stdout, " leaked ("), out.stdout
    assert out.returncode == code, out.stdout + out.stderr
    assert f"{code} gating failure(s) in longbrief" in out.stdout, out.stdout


def test_the_fixture_note_states_the_counts_it_has(corpus: Path) -> None:
    """The file a reviewer reads to decide whether this gate discriminates.

    Its `note` and `note_thresholds` are the auditable claim — how many cases,
    of which classes, at what measured rates — and both had drifted: the served
    ratio said "7 of 8 served (0.875)" against a nine-entry `served` list and a
    live run printing 8/9, and the lookalike class was described as four when
    a later commit had grown it to eight. The gate was stronger than described
    and the description was wrong on both load-bearing numbers, which is the
    same defect class as a declared count that is not true of the bytes.

    Derived from the data and from a live run rather than restated, so the next
    case added here fails loudly instead of drifting silently the way this did.
    """
    index = json.loads(
        (corpus / BRIEFS / "index.json").read_text(encoding="utf-8")
    )
    served, unserved = index["served"], index["unserved"]
    lookalikes = [u for u in unserved if u.get("note")]
    plain = len(unserved) - len(lookalikes)
    assert f"{plain} briefs with no distinctive overlap" in index["note"], plain
    assert (
        f"and {len(lookalikes)} that carry one or two distinctive corpus tokens"
        in index["note"]
    ), len(lookalikes)
    held = [u for u in lookalikes if u["note"].startswith("held out")]
    assert (
        f"{len(lookalikes) - len(held)} written alongside the bar they "
        f"calibrate, and {len(held)} held out" in index["note"]
    ), (len(held), index["note"])

    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    rates = _rates(out.stdout)
    got = re.search(r"(\d+)/(\d+) served \(([\d.]+)", rates)
    assert got, rates
    assert int(got.group(2)) == len(served), (rates, len(served))
    stated = f"{got.group(1)} of {got.group(2)} served ({got.group(3)})"
    assert stated in index["note_thresholds"], (stated, rates)


def test_the_long_brief_slice_reports_both_rates_and_the_thresholds(
    corpus: Path,
) -> None:
    """The control, and the calibration restated as a measurement: the numbers
    the task gate's constants were set from are reproduced by the shipped
    tree, beside the thresholds they were set to."""
    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    line = _rates(out.stdout)
    assert "8/9 served (0.889, floor 0.750)" in line, line
    assert "0/16 leaked (0.000, ceiling 0.084)" in line, line
    # Per-case rows too, so a single outcome moving is visible in a diff even
    # though it is under the rate slack.
    assert "[BRIEF-SERVED]" in out.stdout
    assert "[BRIEF-QUIET ]" in out.stdout


def test_coverage_under_the_floor_fails_the_run(corpus: Path) -> None:
    """A task gate that stops serving briefs it was calibrated to serve is the
    unit's headline failure, and it is silent everywhere else: every brief
    still gets a spawn, the spawn still runs, and nothing anywhere says the
    pointers stopped arriving."""
    _unserve(corpus, "backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md")
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "long-brief coverage" in out.stderr
    assert "under the 0.750 floor" in out.stderr
    assert "5/9 served" in _rates(out.stdout)


def test_injection_over_the_ceiling_fails_the_run(corpus: Path) -> None:
    """The other half of the pair. A coverage floor on its own is met by a gate
    that serves every brief, so the ceiling is what stops the fix for the case
    above being "lower the bars until everything passes"."""
    leak = (
        "\n\nThe sprocket backlash after a gearbox rebuild traces to the shim "
        "stack rather than chain tension, and the flange fasteners want a "
        "crossing sequence over three passes.\n"
    )
    for name in ("accessibility-audit.md", "warehouse-slotting.md"):
        path = corpus / BRIEFS / "unserved" / name
        _write_brief(path, path.read_text() + leak)
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "long-brief injection" in out.stderr
    assert "over the 0.084 ceiling" in out.stderr
    assert "[BRIEF-LEAK  ]" in out.stdout


def test_a_rate_failure_is_not_accepted_by_a_re_baseline(corpus: Path) -> None:
    """`--update-snapshot` accepts what a run reported, which is right for the
    snapshot and wrong for a floor: a threshold a re-baseline can silence is
    not a threshold.

    The write still lands — the remedy for a moved corpus must not be blocked
    by this — so what is asserted is the exit status and the message, not the
    absence of a file.
    """
    _unserve(corpus, "backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md")
    snapshot = corpus / "eval-expectations.json"
    before = snapshot.read_text()
    out = _eval(corpus, "--update-snapshot")
    assert out.returncode != 0, out.stdout
    assert "under the 0.750 floor" in out.stderr
    assert "wrote" in out.stdout
    assert snapshot.read_text() != before, "the re-baseline itself must still land"


def test_a_rate_failure_answers_ahead_of_the_snapshot_after_a_memory_edit(
    corpus: Path,
) -> None:
    """A rate is not a snapshot comparison — it is an absolute measurement of
    the corpus in front of it, which no re-baseline may accept — so it answers
    first and on its own. Printed beside the snapshot's remedy, a coverage
    collapse invites the one command that cannot fix it."""
    _drift(corpus)
    _unserve(corpus, "backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md")
    out = _eval(corpus)
    assert out.returncode != 0
    assert "long-brief coverage" in out.stderr
    assert NO_REBASELINE not in out.stdout, out.stdout


def test_an_edited_brief_reads_as_a_new_case_rather_than_inheriting_one(
    corpus: Path,
) -> None:
    """A brief is a query, and its snapshot key carries a digest of its text —
    otherwise an edited brief silently keeps the outcome recorded for the text
    it used to have, which is the one kind of drift no case line can report."""
    first = _eval(corpus)
    assert first.returncode == 0, first.stdout + first.stderr
    # The control, and the half that fails under a filename-only key: the
    # committed snapshot's keys are the ones this run produces, so nothing is
    # unrecorded before anything is edited.
    assert "NEW (no expectation recorded" not in first.stdout, first.stdout

    path = _served(corpus, "rotor-swap-programme.md")
    _write_brief(path, path.read_text() + "\nOne more paragraph nobody asked for.")
    out = _eval(corpus)
    # The edited brief is unrecorded, and the row for its previous text is now
    # an expectation nothing iterates. Both name that brief and only that
    # brief — a key scheme that changed for everything would satisfy a bare
    # substring search on either word.
    new_rows = [ln for ln in out.stdout.splitlines() if "NEW (no expectation" in ln]
    stale = [ln for ln in out.stdout.splitlines() if "not in the suite" in ln]
    assert len(new_rows) == 1, new_rows
    assert len(stale) == 1, stale
    assert "rotor-swap-programme.md#" in new_rows[0]
    assert "rotor-swap-programme.md#" in stale[0]


def test_an_older_hook_with_no_task_path_skips_the_slice_rather_than_scoring_it(
    corpus: Path, tmp_path: Path
) -> None:
    """An A/B against a build from before the task path existed. Scoring it as
    "served nothing" would report the absence of a feature as a quality
    regression, which is the one comparison an A/B must not make.

    Only for an explicitly named `--hook` copy: the same absence in the
    SHIPPED hook is the feature having been deleted, and the case below is
    that half.
    """
    # The WHOLE directory, because the hook resolves common-words.txt beside
    # __file__ and `load_hook` refuses a lone .py for it. `copytree` preserves
    # mode and the source is read-only under `nix flake check`, so the helper
    # chmods — same reason the `corpus` fixture above does.
    src = _strip_task_path(tmp_path)
    out = _eval(corpus, "--hook", str(src))
    assert out.returncode == 0, out.stdout + out.stderr
    # The missing symbol by name: the probe covers nine of them plus a
    # keyword, so "no task path" would be true of one gap and misleading about
    # the other eight.
    assert "this hook has no task_gate — slice skipped" in out.stdout
    assert not re.search(r"long briefs: \d+/\d+ served", out.stdout), out.stdout


def _copy_hook(tmp_path: Path, old: str, new: str) -> Path:
    """A writable copy of the package with one substitution applied.

    The WHOLE directory, because the hook resolves `common-words.txt` beside
    `__file__` and `load_hook` refuses a lone `.py`. `copytree` preserves mode
    and the source is read-only under `nix flake check`, so the copy is chmodded
    — same reason the `corpus` fixture does.

    A copy rather than an edit in place: the shipped tree is read-only on that
    leg, and editing it would mutate the source other tests in the same session
    are running against.
    """
    root = tmp_path / "memkit"
    shutil.copytree(Path(__file__).resolve().parent.parent / "src" / "memkit", root)
    for path in (root, *root.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    src = root / "memory_prompt_recall.py"
    text = src.read_text()
    assert text.count(old) == 1, old
    src.write_text(text.replace(old, new))
    return src


def _strip_task_path(tmp_path: Path) -> Path:
    """A writable copy of the package with `task_gate` renamed away."""
    return _copy_hook(tmp_path, "def task_gate(", "def _no_task_gate(")


def test_the_shipped_hook_losing_its_task_path_is_a_failure_not_a_skip(
    corpus: Path, tmp_path: Path, monkeypatch
) -> None:
    """The skip above exists for an older build named with `--hook`. Applied to
    the hook this repo ships, the same branch turns a regression that deletes
    or renames the task path into a green run with the only gate over it
    silently not run.

    Driven by pointing the eval's own `STOCK_HOOK` at a stripped copy, which is
    what makes the copy the shipped hook as far as the run is concerned.
    """
    src = _strip_task_path(tmp_path)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, sys;"
            "from memkit import eval_memory_recall as ev;"
            f"ev.STOCK_HOOK = pathlib.Path({str(src)!r});"
            "sys.argv = ['memory-eval', '--config', "
            f"{str(corpus / 'memkit.json')!r}];"
            "ev.main()",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode != 0, out.stdout
    assert "has no `task_gate`" in out.stderr, out.stderr
    assert "rather than a --hook copy" in out.stderr, out.stderr


def test_a_config_with_no_long_briefs_key_says_the_slice_did_not_run(
    corpus: Path,
) -> None:
    """Every way of not having this gate was silent: a config predating the
    key, a typo in it, or a newer config read by an older memkit that drops
    what it does not know. A green run has to say which gates it ran.

    The config here has also stopped NAMING the slice among its gating ones,
    which is the difference between an adopter who never wrote paired briefs
    and one whose gate went missing — the second is the refusal beside this.
    """
    _gating(corpus, "suite", "noinject")
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    del state["eval"]["long_briefs"]
    config.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "eval.long_briefs is not configured — slice skipped" in out.stdout
    assert not re.search(r"long briefs: \d+/\d+ served", out.stdout), out.stdout


def _reindex(corpus: Path, **over) -> None:
    index = corpus / BRIEFS / "index.json"
    state = json.loads(index.read_text())
    state.update(over)
    index.write_text(json.dumps(state))


def test_thresholds_loose_enough_to_be_unfailable_are_refused(corpus: Path) -> None:
    """`min_served: 0.0` and `max_injected: 1.0` leave both comparisons unable
    to fail, and the run still prints two rates and exits 0 — which reads
    exactly like a gate that held. The file may be stricter than the code's
    bounds and never looser, so loosening is a diff somebody reads."""
    _reindex(corpus, min_served=0.0)
    out = _eval(corpus)
    assert out.returncode != 0
    assert "cannot fail" in out.stderr, out.stderr

    _reindex(corpus, min_served=0.75, max_injected=1.0)
    out = _eval(corpus)
    assert out.returncode != 0
    assert "cannot fail" in out.stderr, out.stderr

    # A finite-number check too: NaN compares false against everything, so a
    # rate of NaN is a gate that never fires and never says so.
    _reindex(corpus, max_injected=0.084, min_served=float("nan"))
    out = _eval(corpus)
    assert out.returncode != 0
    assert "not a finite rate" in out.stderr, out.stderr


def test_a_population_too_small_to_carry_a_rate_is_refused(corpus: Path) -> None:
    """Deleting the negative briefs removes the population that measures
    leakage; the arithmetic then reports zero leakage over nothing. Same for
    coverage. A rate needs a population, and this says so rather than
    dividing."""
    index = corpus / BRIEFS / "index.json"
    state = json.loads(index.read_text())
    kept = state["unserved"][:2]
    state["unserved"] = kept
    index.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode != 0
    assert "rate can be taken over" in out.stderr, out.stderr

    state["unserved"] = []
    index.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode != 0
    assert "rate over an empty population" in out.stderr, out.stderr


def test_the_slice_scores_what_reaches_the_subagent_not_what_ranked(
    corpus: Path, tmp_path: Path
) -> None:
    """The slice stopped at the relevance floor, so a brief whose emission the
    harness would refuse — a malformed `updatedInput`, or one over the write
    bound — scored as served. Retrieval is not delivery on this path.

    Driven by shrinking the write bound to a value every emission crosses: the
    ranker is untouched and every served brief still ranks its target first, so
    a slice that scored retrieval would report 7/8 unchanged.
    """
    before = _eval(corpus)
    assert "8/9 served" in _rates(before.stdout), before.stdout

    src = _copy_hook(tmp_path, "PIPE_BUFFER_BOUND = 16384", "PIPE_BUFFER_BOUND = 64")
    out = _eval(corpus, "--hook", str(src))
    assert "0/9 served" in _rates(out.stdout), out.stdout
    assert out.returncode != 0, out.stdout
    assert "long-brief coverage" in out.stderr


def _gating(corpus: Path, *slices: str) -> None:
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    state["eval"]["gating_slices"] = list(slices)
    config.write_text(json.dumps(state))


def test_one_leaked_brief_fails_the_run_even_under_the_rate_slack(
    corpus: Path,
) -> None:
    """The rate slack exists so a corpus can move by one case without a red
    CI; it is not a license for one new wrong injection.

    One leak in sixteen is 0.062, under the 0.084 ceiling, so the RATE holds,
    and only the per-case row can fail the run: in a gating slice it reads
    `<- MOVED` and reaches the exit code, so a single new injection into an
    autonomous subagent's instructions is a red run. The two controls do
    different jobs: the rate bounds systemic loosening, the snapshot bounds
    one case moving. A new memory is the edit, so every brief keeps its key and
    the leak is the only row that moves.
    """
    memory = corpus / "corpus" / "personal" / "search" / "warehouse_slotting.md"
    memory.write_text(SLOTTING, encoding="utf-8")
    out = _eval(corpus)
    assert out.returncode == 1, out.stdout + out.stderr
    # The RATE held — this is the case the rate cannot catch.
    assert "1/16 leaked (0.062, ceiling 0.084)" in _rates(out.stdout), out.stdout
    assert "long-brief injection" not in out.stderr, out.stderr
    leak = next(ln for ln in out.stdout.splitlines() if "[BRIEF-LEAK  ]" in ln)
    assert "<- MOVED (snapshot says BRIEF-QUIET)" in leak, leak
    assert re.search(r"1 gating failure\(s\) in [\w/]*longbrief", out.stdout), out.stdout


def test_the_shipped_config_gates_the_only_slice_over_subagent_delivery(
    corpus: Path,
) -> None:
    """The fixture config IS the release gate, so what it names is the
    contract. Asserted here rather than left to a reader of the JSON: the
    slice's per-case rows are advisory until the config says otherwise, and
    every other check in this file would stay green with the entry removed."""
    state = json.loads((corpus / "memkit.json").read_text())
    assert "longbrief" in state["eval"]["gating_slices"], state["eval"]
    assert state["eval"].get("long_briefs"), state["eval"]


def test_a_config_that_gates_a_slice_it_cannot_run_is_refused(corpus: Path) -> None:
    """The ungated state the skip line was papering over.

    A config naming `longbrief` among its gating slices has said it wants
    subagent delivery gated. Without `eval.long_briefs` there is nothing to
    run, and the old code printed one line and exited 0 — a green eval over a
    task path that could be completely broken. Asking for a gate that cannot
    run is a refusal, not a note.
    """
    config = corpus / "memkit.json"
    state = json.loads(config.read_text())
    del state["eval"]["long_briefs"]
    config.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "gating_slices names `longbrief`" in out.stderr, out.stderr


def test_duplicate_or_shared_cases_cannot_stand_in_for_a_population(
    corpus: Path,
) -> None:
    """The population floors count list ENTRIES, so twelve copies of one brief
    satisfied a bar written to mean twelve briefs — a rate re-measuring one
    passing case while the rest of the corpus regressed unobserved.

    Same for a brief in both halves, which is a case asserting two opposite
    outcomes and scoring whichever it is asked for.
    """
    index = corpus / BRIEFS / "index.json"
    state = json.loads(index.read_text())

    duped = dict(state)
    duped["unserved"] = [state["unserved"][0]] * len(state["unserved"])
    index.write_text(json.dumps(duped))
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "names the same brief twice" in out.stderr, out.stderr

    shared = dict(state)
    shared["unserved"] = [
        {"brief": state["served"][0]["brief"]}, *state["unserved"][1:]
    ]
    index.write_text(json.dumps(shared))
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "in both halves" in out.stderr, out.stderr


def test_a_case_pointing_outside_the_brief_directory_is_refused(
    corpus: Path,
) -> None:
    """`brief` is joined onto the fixture root, so an absolute path or a `..`
    walks out of it — and a case that reads a file from somewhere else is a
    gate measuring something nobody reviewing this directory can see."""
    index = corpus / BRIEFS / "index.json"
    state = json.loads(index.read_text())
    state["served"][0] = {"brief": "../../memkit.json", "file": "x.md"}
    index.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    # A REFUSAL, not a traceback: exit 1 is reserved for a gate failing, so a
    # crash and a real regression were the same signal to whatever runs this.
    assert "Traceback" not in out.stderr, out.stderr
    assert "outside" in out.stderr, out.stderr

    state["served"][0] = {"brief": 17, "file": "x.md"}
    index.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "Traceback" not in out.stderr, out.stderr
    assert "brief" in out.stderr, out.stderr


def test_a_hook_copy_missing_any_of_the_slice_is_skipped_not_crashed(
    corpus: Path, tmp_path: Path
) -> None:
    """The probe was one symbol wide and the surface below it grew to nine.

    A `--hook` copy with a task path but from before the floor helper — the
    immediately preceding commit of this branch qualifies — passed the probe
    and died mid-run with an uncaught AttributeError, after the suite slice had
    already printed its PASS lines. Exit 1 is reserved for a gate failing, so a
    crash and a real regression were the same signal to CI.
    """
    src = _copy_hook(tmp_path, "def _task_floor(", "def _no_task_floor(")
    out = _eval(corpus, "--hook", str(src))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "_task_floor" in out.stdout, out.stdout
    assert "slice skipped" in out.stdout, out.stdout
    assert "AttributeError" not in out.stderr, out.stderr


def test_a_hook_copy_missing_the_slices_keyword_is_skipped_too(
    corpus: Path, tmp_path: Path
) -> None:
    """The keyword, not only the name: a copy can carry `_pointer_line`
    without the `over_brief` argument this slice passes it, and the failure is
    the same uncaught TypeError one call later."""
    src = _copy_hook(tmp_path, "over_brief: bool = False", "over_long: bool = False")
    out = _eval(corpus, "--hook", str(src))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "over_brief" in out.stdout, out.stdout
    assert "slice skipped" in out.stdout, out.stdout


def test_the_slice_emits_through_the_hooks_own_writer(corpus: Path) -> None:
    """One spelling of "may these bytes be written".

    The slice used to rebuild the emission decision itself — `_task_payload`,
    then its own size test — so any divergence between the two was invisible
    to the only gate over subagent delivery, and the evaluator would go on
    extracting filenames from a string the hook would have refused to write.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "src" / "memkit" / "eval_memory_recall.py"
    ).read_text(encoding="utf-8")
    assert "_task_emission(" in source
    assert "_task_payload(" not in source
    assert "PIPE_BUFFER_BOUND" not in source
    # And one stage earlier, for the same reason: the block the emission is
    # judged against is built once, by the hook, cap and truncation notice
    # included.
    assert "_task_block(" in source
    assert "_task_framed(" not in source
    assert "TASK_MAX_HITS]" not in source


def test_an_index_that_cannot_answer_is_not_scored_as_a_retrieval_miss(
    corpus: Path, tmp_path: Path
) -> None:
    """The state the task path added `index-unavailable` FOR, and the one the
    gate could not see.

    Parallel spawns are the normal case on the task path, they share one
    sqlite index, and a contender that loses a cold build's write-lock race
    meets an index with no committed rows. `recall` suppresses that per dir
    and returns the other dirs' hits, so from the outside it is indistinguish-
    able from a corpus with nothing to say — which is why production splits the
    two by `errs_lex` and records `task:index-unavailable` rather than
    `task:nomatch`.

    The gate scored briefs serially and read only the hits, so every one of
    those cases would have counted as BRIEF-MISS: an infrastructure failure
    arriving as a coverage number, quietly, with the rate and the rows looking
    exactly like a task gate that stopped serving. It refuses now.

    The condition is injected rather than raced. A real lock race is the same
    fact arriving nondeterministically and slowly; what has to be gated is
    what the run DOES with the fact, and a hook copy whose lexical stage cannot
    answer produces it on every dir, every time.
    """
    src = _copy_hook(
        tmp_path,
        '    db = _fts_db(d)\n    _fts_note_root(db, d)',
        '    raise sqlite3.OperationalError("database is locked")\n'
        '    db = _fts_db(d)\n    _fts_note_root(db, d)',
    )
    out = _eval(corpus, "--hook", str(src))
    assert out.returncode != 0, out.stdout
    assert "could not answer" in out.stderr, out.stderr
    assert "not a retrieval miss" in out.stderr, out.stderr
    assert "[BRIEF-NOINDEX]" in out.stdout, out.stdout
    # And it is the refusal that fails the run, not the coverage rate dropping
    # out from under it — the point is that the number is not reported as a
    # measurement at all.
    assert "0/9 served" in _rates(out.stdout), out.stdout


def test_an_index_that_cannot_answer_is_not_scored_as_a_quiet_brief(
    corpus: Path, tmp_path: Path
) -> None:
    """The same fact the served half already refuses, on the half that
    certifies the injection ceiling.

    An index that could not answer injects nothing, and injecting nothing is
    exactly what a correctly quiet brief looks like — so on this half the
    unattributable result is the CLEAN one. Counted, it certifies the ceiling
    against a corpus that was never searched, which is this suite's only bound
    on what the task path says to an unattended subagent. The served half
    refuses on a MISS and this one has to refuse on a QUIET, which is why one
    fix did not cover both.

    Same injected condition as the served half's case, for the same reason: a
    hook copy whose lexical stage cannot answer produces it on every dir, every
    time, where a real lock race produces it slowly and at random.
    """
    unserved = len(
        json.loads((corpus / BRIEFS / "index.json").read_text())["unserved"]
    )
    assert unserved, "the fixture corpus must carry negative cases"
    src = _copy_hook(
        tmp_path,
        '    db = _fts_db(d)\n    _fts_note_root(db, d)',
        '    raise sqlite3.OperationalError("database is locked")\n'
        '    db = _fts_db(d)\n    _fts_note_root(db, d)',
    )
    out = _eval(corpus, "--hook", str(src))
    assert out.returncode != 0, out.stdout
    # Not one negative case reports a clean row it cannot account for.
    assert "[BRIEF-QUIET " not in out.stdout, out.stdout
    assert out.stdout.count("[BRIEF-NOINDEX]") == unserved + 9, out.stdout
    assert out.stderr.count("cannot say anything about leakage") == unserved, (
        out.stderr
    )
    # And a healthy index is unaffected: `unanswerable` is zero there, so the
    # refusal cannot be firing on the shape of the run rather than the fact.
    assert "[BRIEF-QUIET " in _eval(corpus).stdout


def test_a_neighbours_pointer_line_is_not_this_memory_being_delivered(
    corpus: Path, tmp_path: Path
) -> None:
    """The names are read back out of the emitted bytes so that a pick the
    block dropped scores as a miss. Tested by containment against the whole
    line, a name another memory's name merely EXTENDS is found on that
    neighbor's line and scores as delivered — the delivery gate satisfied by
    a pointer the subagent never received.

    Latent on the shipped fixtures, where no basename is a substring of
    another; one fixture named `balancing.md` beside `turbine_balancing.md` is
    all it takes, so it is driven with exactly that.
    """
    domain = corpus / "corpus" / "project" / "search" / "domain"
    twin = corpus / "corpus" / "project" / "search" / "balancing.md"
    twin.write_text(
        (domain / "turbine_balancing.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    index = corpus / BRIEFS / "index.json"
    state = json.loads(index.read_text())
    for case in state["served"]:
        if case["brief"] == "served/rotor-swap-programme.md":
            case["file"] = twin.name
    index.write_text(json.dumps(state, indent=2))
    # The stock hook baselines the edited corpus, so the divergence under test
    # is the copy's rather than the twin's.
    assert _eval(corpus, "--update-snapshot").returncode == 0

    src = _copy_hook(
        tmp_path,
        "        [_pointer_line(*pick, over_brief=True) for pick in picks], truncated",
        "        [_pointer_line(*pick, over_brief=True) for pick in picks[1:]], "
        "truncated",
    )
    out = _eval(corpus, "--hook", str(src))
    row = next(ln for ln in out.stdout.splitlines() if "rotor-swap-programme" in ln)
    assert "[BRIEF-MISS" in row, row
    # The quoted repr, so this assertion is not the containment test it is
    # about — a first draft of it read `twin.name not in row` and matched
    # inside `'turbine_balancing.md'`.
    assert f"'{twin.name}'" not in row.split("(got ")[1], row


def test_the_slice_spends_the_budget_the_gate_and_the_query_already_spent(
    corpus: Path, tmp_path: Path
) -> None:
    """Production stamps `t0` in `main` and hands it down, so the brief gate
    and the query builder are billed to the same budget the search then runs
    under. The slice started its clock at the search, which hands retrieval a
    budget production has already spent part of — and a gate with more budget
    than production reports pointers production abandons.

    Driven by making the query builder cost more than the whole budget, which
    is honest about what it proves: the structural divergence, not that the
    1.4-3.2 ms production actually spends there matters.
    """
    src = _copy_hook(tmp_path, "TASK_BUDGET_SECONDS = 7", "TASK_BUDGET_SECONDS = 0.25")
    text = src.read_text()
    marker = "def build_task_query(stripped: str) -> str | None:\n"
    assert text.count(marker) == 1, marker
    src.write_text(text.replace(marker, marker + "    time.sleep(0.5)\n"))

    out = _eval(corpus, "--hook", str(src))
    assert "0/9 served" in _rates(out.stdout), out.stdout
    assert out.returncode != 0, out.stdout


def test_the_slice_retrieves_under_the_deadline_production_passes(
    corpus: Path, tmp_path: Path
) -> None:
    """Production calls `recall` with `deadline=t0 + TASK_BUDGET_SECONDS`; the
    slice called it with no deadline at all, so `recall` ran on its default
    unlimited budget.

    On the tiny fixture corpus the two agree, which is exactly why it survived:
    the divergence only shows against a consumer's own store under `--repo` or
    `--all-stores`, where the gate can wait and report served pointers that
    production abandons. Driven by moving the budget the gate is supposed to
    honor — a hook copy whose `TASK_BUDGET_SECONDS` has already expired serves
    nothing, and a slice that passes no deadline cannot tell.
    """
    before = _eval(corpus)
    assert "8/9 served" in _rates(before.stdout), before.stdout

    src = _copy_hook(tmp_path, "TASK_BUDGET_SECONDS = 7", "TASK_BUDGET_SECONDS = -1")
    out = _eval(corpus, "--hook", str(src))
    assert "0/9 served" in _rates(out.stdout), out.stdout


def test_a_name_the_brief_already_contains_is_not_delivery(
    corpus: Path, tmp_path: Path
) -> None:
    """The slice read the names back out of the WHOLE updated prompt, which is
    the brief the slice itself supplied plus the block — so a brief that
    happens to name a corpus file scored as served whether or not the block
    carried anything.

    No shipped fixture brief names one today, which is what makes this latent
    rather than firing: a gate that can be satisfied by its own input is one
    edit to a fixture away from being satisfied by nothing at all. Driven with
    a hook copy whose block carries no pointer lines and a brief that names its
    own target file.
    """
    index = json.loads((corpus / BRIEFS / "index.json").read_text())
    case = index["served"][0]
    brief = corpus / BRIEFS / case["brief"]
    _write_brief(brief, brief.read_text() + f"\n\nSee also the note in {case['file']}.")
    # Anchored on the line that JOINS THE BODY IN, not on the preamble prose
    # in front of it: the prose is edited whenever the reader's rule changes,
    # and an anchor that moves with it turns a real regression into an
    # AssertionError about a string literal.
    src = _copy_hook(
        tmp_path,
        '        + "\\n".join(body)\n        # The last thing before the delimiter',
        '        + ""\n        # The last thing before the delimiter',
    )
    out = _eval(corpus, "--hook", str(src))
    assert "0/9 served" in _rates(out.stdout), out.stdout


def test_two_filenames_holding_one_brief_are_one_case(corpus: Path) -> None:
    """Uniqueness was checked on the resolved PATH while the comment above it
    states the invariant as "a CASE is a distinct brief".

    So two filenames holding the same text both counted toward the minimum
    population and both fed the rate denominators — one behavior repeated
    enough times to satisfy a bar written to mean that many briefs, which is
    the same defect the path check exists to prevent wearing a different
    filename.
    """
    index = corpus / BRIEFS / "index.json"
    state = json.loads(index.read_text())
    original = corpus / BRIEFS / state["unserved"][0]["brief"]
    twin = original.with_name("twin-" + original.name)
    twin.write_text(original.read_text())
    # Both listed, sixteen entries still: the population floor and the rate
    # denominators see two cases where there is one brief.
    state["unserved"] = [
        state["unserved"][0],
        {"brief": f"unserved/{twin.name}"},
        *state["unserved"][2:],
    ]
    index.write_text(json.dumps(state))
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "same brief under two names" in out.stderr, out.stderr


def test_a_gating_slice_nobody_has_is_refused_rather_than_crashed(
    corpus: Path,
) -> None:
    """README tells an adopter to add `longbrief` to `eval.gating_slices` by
    hand, and a typo there ended a fully green run with a `KeyError` traceback
    and exit 1 — which CI reads as the regression that did not happen.

    The module already wraps a malformed fixture index for this reason: exit 1
    is documented as a gate failing or a refusal, and a stack trace makes a
    configuration mistake look like a crash in the tool.
    """
    _gating(corpus, "suite", "noinject", "longbriefs")
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "Traceback" not in out.stderr, out.stderr
    assert "no such slice" in out.stderr, out.stderr
    assert "longbrief" in out.stderr, out.stderr


def test_the_floor_faces_negatives_it_was_not_calibrated_against() -> None:
    """The four incidental-token briefs that TASK_MIN_MATCHED was set from were
    written in the same commit as the number and the snapshot that scores them,
    so the slice's green on them says only that the bar reproduces its own
    calibration set. These four came afterwards and were scored once, as they
    stand. Pinned here so a later edit cannot quietly leave the bar with only
    its own fixtures to answer to.
    """
    index = json.loads((FIXTURES / BRIEFS / "index.json").read_text())
    held = [c for c in index["unserved"] if "held out" in c.get("note", "")]
    assert len(held) >= 4, [c.get("brief") for c in index["unserved"]]
    assert "note_holdout" in index
    for case in held:
        assert (FIXTURES / BRIEFS / case["brief"]).is_file(), case


def test_the_padded_tool_input_is_never_smaller_than_the_overhead_it_assumes(
) -> None:
    """`TASK_INPUT_ASSUMED_OVERHEAD` is the only thing keeping this gate's
    payload from being SMALLER than what production sends the Agent tool, and
    a gate whose payload is smaller than production's scores a brief `served`
    at a size production refuses with `task:oversize`.

    That invariant has drifted once already, silently: the pad used to land on
    `description`'s VALUE rather than on the whole serialized object, so it
    stopped being the stated assumption the moment the input grew a key. A
    person noticed. Nothing in the suite did — the constant, the `spare`
    variable and the padded object's size were referenced by no test at all.

    Asserted against the shape `_task_delivery` builds, and then against that
    shape PLUS a key, which is the exact class of change that caused the drift.
    `max(spare, 0)` used to absorb the second case in silence; it raises now,
    because a gate that has quietly stopped being conservative is worth more
    to know about than one more green run.
    """
    shape = {
        "prompt": "",
        "description": "score this brief",
        "subagent_type": "general-purpose",
    }
    weight = len(json.dumps(shape, ensure_ascii=False))
    assert weight <= ev.TASK_INPUT_ASSUMED_OVERHEAD, (
        weight,
        ev.TASK_INPUT_ASSUMED_OVERHEAD,
    )
    # The padded object REACHES the assumed overhead rather than merely fitting
    # under it — the pad exists to make the gate's payload no smaller than a
    # real one, so an assumption nothing grows into is not an assumption.
    padded = dict(shape, description=shape["description"] + "." * (
        ev.TASK_INPUT_ASSUMED_OVERHEAD - weight
    ))
    assert len(json.dumps(padded, ensure_ascii=False)) == (
        ev.TASK_INPUT_ASSUMED_OVERHEAD
    )
    # And a shape that has outgrown the constant fails LOUDLY. This is the
    # drift c8be3fd fixed, reproduced: one more key, and the pad silently
    # became a no-op that left the gate optimistic.
    with pytest.raises(ValueError, match="TASK_INPUT_ASSUMED_OVERHEAD"):
        ev._pad_to_overhead(dict(shape, model="x" * ev.TASK_INPUT_ASSUMED_OVERHEAD))


def test_a_description_that_mentions_a_file_does_not_prove_it_was_delivered(
) -> None:
    """The readback has to read the PATH field, not the whole line.

    Splitting a pointer line on whitespace and taking every token's basename
    makes any word of a surviving DESCRIPTION able to vouch for a pointer that
    was shed or never emitted — and descriptions in this corpus are file
    contents, so a memory that mentions its neighbor by name is ordinary
    rather than contrived. The gate then reports subagent coverage for a
    pointer the subagent did not receive, which is the one thing this slice
    exists to measure.
    """
    block = (
        "- /store/search/sprocket_alignment.md — supersedes "
        "flange_torque.md for the 2026 rebuild [matches 3 terms from this "
        "brief: sprocket, backlash, shim]"
    )
    assert ev._delivered_names(block) == {"sprocket_alignment.md"}
    # Non-vacuity: a line that really does carry the path still counts, and a
    # path rendered with a `~` or a relative prefix is still its basename.
    both = block + "\n- ~/store/search/flange_torque.md — star pattern, "
    both += "three passes [matches 2 terms from this brief: flange, torque]"
    assert ev._delivered_names(both) == {
        "sprocket_alignment.md",
        "flange_torque.md",
    }
    # And a line whose separator was consumed does not parse into a delivery.
    assert ev._delivered_names("- /store/search/eaten.md no separator here") == set()


def test_a_memory_with_no_description_still_reads_back_as_delivered(
    tmp_path,
) -> None:
    """`_pointer_line` renders the em-dash separator CONDITIONALLY, and the
    readback was anchored on it.

    `_description` returns "" for a memory with neither `description:`
    frontmatter nor a `# ` heading, and on OSError. Such a pointer line has no
    em-dash anywhere, so it did not parse and its name was dropped from the
    delivered set. The served half of the long-brief gate then undercounts,
    which fails loudly; the UNSERVED half's test is `ok = not shown`, so a
    pointer that really did reach an unattended subagent scored BRIEF-QUIET —
    a leak certified as clean, on the file's own account of the only bound
    this suite has on what the task path says to an unattended subagent.

    Latent on the shipped fixtures, where every memory carries a description,
    and live the moment the gate is pointed at a real store.
    """
    bare = tmp_path / "no_description.md"
    bare.write_text(
        "---\nname: no_description\ntype: reference\n---\n\nSome body about sprockets.\n"
    )
    described = tmp_path / "has_description.md"
    described.write_text(
        "---\nname: has_description\ndescription: about sprockets\n"
        "type: reference\n---\n\nSome body about sprockets.\n"
    )
    terms = ["sprocket", "backlash"]
    for path in (bare, described):
        hook._LEX_MATCHED[str(path)] = list(terms)
    assert hook._description(str(bare)) == ""
    line = hook._pointer_line(str(bare), terms, 5)
    assert "\u2014" not in line, line
    assert ev._delivered_names(line) == {"no_description.md"}, line
    # Non-vacuity: the described sibling parses through the other branch.
    other = hook._pointer_line(str(described), terms, 5)
    assert "\u2014" in other, other
    assert ev._delivered_names(other) == {"has_description.md"}, other


def test_the_eval_scores_a_corpus_holding_a_symlinked_memory(
    tmp_path, monkeypatch
) -> None:
    """The eval reads the hook's per-hit root the way the hook writes it.

    `_LEX_ROOT`'s value carries whether a repository chose the root, and the
    root is only ever CONSULTED for a candidate that is a link — so a reader
    holding the wrong shape scores every ordinary corpus and then aborts the
    whole run on the first store that commits one, which `_store_path`'s own
    rule deliberately admits.

    The corpus here holds one, which is what makes this fail on a reader that
    is out of step: no other fixture corpus in the tree does.
    """
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path / "state"))
    (tmp_path / "state").mkdir()
    root = tmp_path / "memories" / "search"
    root.mkdir(parents=True)
    real = root / "real.md"
    real.write_text(
        "---\nname: real\ndescription: sprocket notes\ntype: reference\n---\n\n"
        "sprocket backlash gearbox shim stack\n"
    )
    (root / "linked.md").symlink_to(real)

    prompt = "sprocket backlash gearbox shim stack"
    hits = hook.recall(prompt, dirs=[str(root)])
    assert {Path(h).name for h in hits} == {"real.md", "linked.md"}, hits
    passed, shown = ev.pointers(hook, prompt, hits)
    assert set(passed) == {"real.md", "linked.md"}, passed
    assert shown == passed[: hook.MAX_HITS], (shown, passed)


def test_the_eval_floors_a_repository_chosen_credential_the_way_the_hook_does(
    tmp_path, monkeypatch
) -> None:
    """The eval scores a candidate with the arguments the hook passes, or it is
    measuring a retriever no session meets.

    Whether a REPOSITORY chose the corpus is one of those arguments: it is what
    turns on the credential scan. Dropped, the eval scores a file the hook
    floors — and this gate's whole claim is that a case cannot pass on a hit
    production would refuse.

    The user's own corpus is the control: the same body, scored the same way it
    always was, because nothing about it is repository-chosen.
    """
    monkeypatch.setattr(hook, "_state_dir", lambda: str(tmp_path / "state"))
    (tmp_path / "state").mkdir()
    body = (
        "---\nname: shims\ndescription: sprocket notes\ntype: reference\n---\n\n"
        "sprocket backlash gearbox shim stack\n"
    )
    planted = body + "AKIA0123456789ABCDEF\n"
    prompt = "sprocket backlash gearbox shim stack"

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".memkit.json").write_text(
        json.dumps(
            {
                hook.PROJECT_SCHEMA_KEY: hook.PROJECT_SCHEMA,
                "store": {"id": "app", "dir": "docs/memories"},
            }
        )
    )
    repo_corpus = repo / "docs" / "memories" / "search"
    repo_corpus.mkdir(parents=True)
    (repo_corpus / "shims.md").write_text(planted)

    mine = tmp_path / "mine" / "search"
    mine.mkdir(parents=True)
    (mine / "shims.md").write_text(planted)

    def scored(corpus: Path) -> list[str]:
        hits = hook.recall(prompt, dirs=[str(corpus)])
        assert [Path(h).name for h in hits] == ["shims.md"], hits
        # What the hook's own injection path would do with the same hit.
        terms = list(dict.fromkeys((hook.build_query(prompt) or "").split()))
        mine_ = [
            Path(h).name
            for h in hits
            if hook._passes_floor(
                *hook._relevance(
                    terms, h, hook._lex_root(h), hook._lex_read_only(h)
                )
            )
        ]
        passed, _ = ev.pointers(hook, prompt, hits)
        assert passed == mine_, (passed, mine_)
        return passed

    assert scored(repo_corpus) == [], "the repository's planted file was scored"
    assert scored(mine) == ["shims.md"], "the user's own corpus moved"


def test_a_same_named_file_in_another_store_is_not_the_delivery() -> None:
    """The gate compared basenames, so a pointer to the wrong store's file
    satisfied a case whose target was never delivered.

    Two configured stores holding one name is ordinary — `beads.md` in a
    project store and in the personal one — and the difference is exactly what
    a retrieval gate exists to measure.
    """
    line = (
        "- ~/personal/search/beads.md \u2014 the other store's copy "
        "[matches 2 terms from this brief: bd, dolt]"
    )
    assert ev._delivered_paths(line) == {"~/personal/search/beads.md"}
    assert "~/project/search/beads.md" not in ev._delivered_paths(line)
    # The basename view still agrees with itself; it is simply not an identity.
    assert ev._delivered_names(line) == {"beads.md"}


def test_a_pointer_path_holding_a_bracket_is_not_cut_short() -> None:
    """The description-less anchor is a second pattern rather than an
    alternation: a non-greedy match stops at whichever branch appears
    EARLIEST, so `a [1].md` on a line that also carries a description would
    have parsed as `a`."""
    with_desc = "- /store/a [1].md \u2014 notes [matches 1/1 prompt terms: a]"
    assert ev._delivered_paths(with_desc) == {"/store/a [1].md"}
    without = "- /store/a [1].md [matches 1/1 prompt terms: a]"
    assert ev._delivered_paths(without) == {"/store/a [1].md"}


def test_the_task_surface_declares_every_hook_name_the_slice_reaches() -> None:
    """The gate over subagent delivery, held to the surface it actually uses.

    `TASK_SURFACE` is what `task_surface_gap` walks before the long-brief
    slice runs, and its own comment says why it is a list rather than one
    probe: a copy carrying `task_gate` and nothing else passed the probe and
    then died mid-run with an uncaught AttributeError, after the slice had
    printed its PASS lines. Written out by hand, it had already drifted —
    `_display_path` is reached at eval_memory_recall.py:908 and was not in it,
    so a rename of that one name reproduced the exact failure the constant
    exists to prevent, past a gap check reporting no gap.

    DERIVED, by walking what the guarded block reaches rather than by
    restating it. The roots are the two functions the block calls; everything
    they reach transitively is inside the gate, so a `hook.` name added
    anywhere under them joins this assertion by existing.
    """
    tree = ast.parse(Path(ev.__file__).read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    reached: set[str] = set()
    seen: set[str] = set()
    stack = ["task_delivery", "over_cap_faults"]
    assert set(stack) <= set(fns), sorted(set(stack) - set(fns))
    while stack:
        name = stack.pop()
        if name in seen or name not in fns:
            continue
        seen.add(name)
        for node in ast.walk(fns[name]):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "hook"
            ):
                reached.add(node.attr)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                stack.append(node.func.id)
    # Non-vacuity in both directions: the walk really did enter the delivery
    # pipeline, and it really did read attributes off the hook.
    assert "_task_delivery" in seen, sorted(seen)
    assert {"recall", "_task_block"} <= reached, sorted(reached)
    assert reached <= set(ev.TASK_SURFACE), sorted(reached - set(ev.TASK_SURFACE))

    # And every declared name is one the shipped hook has, or the gate refuses
    # the hook this repo ships over a name nothing reaches any more.
    absent = [n for n in ev.TASK_SURFACE if getattr(hook, n, None) is None]
    assert not absent, absent


def test_no_digest_in_the_eval_dies_on_a_lone_surrogate() -> None:
    """The rule the hook module already holds, applied to the module beside it.

    Same scan, IMPORTED rather than copied: it is the one that matches the
    rule instead of a shape, and the reason it exists is that a list of call
    sites and a shape-matching predicate each let the same defect back in. A
    second copy here would be the third way to do that.

    Zero exceptions in this file. The hook has one argued strict encode
    because there a raise IS the refusal; nothing in the eval refuses
    anything by dying, so a strict encode here is a crash and nothing else.
    """
    from test_memory_prompt_recall import _unhandled_encodes

    source = Path(ev.__file__).read_text(encoding="utf-8")
    assert _unhandled_encodes(source) == []
    # Non-vacuity: the scan still sees its subject in this file's own text.
    assert _unhandled_encodes(source + '\ndef f(t):\n    return t.encode("utf-8")\n')
