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

import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

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


def _eval(corpus: Path, *args: str) -> subprocess.CompletedProcess:
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
    )


def _drift(corpus: Path) -> None:
    """Move the corpus without moving any case's outcome.

    The point is the FINGERPRINT, not the retrieval: an edit that changed a
    result would also fail the old code, and this has to fail on a corpus that
    still scores identically — which is what a typical memory edit looks like.
    """
    memo = corpus / "corpus" / "project" / "search" / "flange_torque.md"
    memo.write_text(memo.read_text() + "\nA sentence nobody searches for.\n")


def test_the_committed_fixture_corpus_gates_clean(corpus: Path) -> None:
    """The control. Without it every case below could pass because the eval is
    broken in some way that has nothing to do with fingerprints."""
    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "matches the snapshot" in out.stdout


def test_a_moved_corpus_refuses_instead_of_reporting_green(corpus: Path) -> None:
    """The fix. Nothing measured under a moved corpus is attributable to the
    tool, so nothing may gate — and that is a refusal, not a pass."""
    _drift(corpus)
    out = _eval(corpus)

    assert out.returncode != 0, out.stdout
    # The remedy IS the message: one command, and the instruction to commit
    # what it writes in the same change.
    assert "--update-snapshot" in out.stderr
    assert "same change" in out.stderr
    # And the report above it is unchanged — the per-case lines and the
    # explanation of why none of them counted. Only the ending moved.
    assert "[PASS" in out.stdout
    assert "these stores are not the ones baselined" in out.stdout
    assert "the corpus moved, so nothing gates" in out.stdout


def test_update_snapshot_still_works_on_the_corpus_it_is_the_remedy_for(
    corpus: Path,
) -> None:
    """A re-baseline that refused on a moved corpus would name a fix and then
    reject it. It writes, and it exits 0 — accepting what the run reported is
    the act, so a non-zero here would be indistinguishable from a refusal."""
    snapshot = corpus / "eval-expectations.json"
    before = json.loads(snapshot.read_text())["corpus"]
    _drift(corpus)

    out = _eval(corpus, "--update-snapshot")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "wrote" in out.stdout
    after = json.loads(snapshot.read_text())["corpus"]
    assert after != before, "the fingerprint is what a re-baseline is for"

    # And the gate is live again immediately, which is what makes the remedy a
    # remedy rather than a way to switch the check off.
    assert _eval(corpus).returncode == 0


def test_no_snapshot_at_all_still_refuses(corpus: Path) -> None:
    """Unchanged by this fix, asserted because it is the same property: a run
    with nothing to gate against must not report having gated."""
    (corpus / "eval-expectations.json").unlink()
    out = _eval(corpus)
    assert out.returncode != 0
    assert "--update-snapshot" in out.stderr


def test_a_real_regression_still_fails_on_a_matched_corpus(corpus: Path) -> None:
    """The fix must not have made the gate coarser. With the fingerprint
    MATCHING, a case whose outcome moved still fails on its own terms — and
    with a different message, so the two reds stay distinguishable."""
    snapshot = corpus / "eval-expectations.json"
    state = json.loads(snapshot.read_text())
    # Move a recorded STATUS, which is the field the diff compares — leaving
    # the corpus alone, so the fingerprint still matches and the run is in the
    # regime where an outcome that moved is attributable to the tool.
    noinject = state["cases"]["noinject"]
    prompt = next(iter(noinject))
    noinject[prompt]["status"] = "NOINJECT-LEAK"
    snapshot.write_text(json.dumps(state))

    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "REGRESSION" in out.stdout
    assert "corpus moved" not in out.stderr


# --- the long-brief slice ----------------------------------------------------
#
# Two rates over sixteen paired briefs, and they gate differently from
# everything above: the snapshot records what happened and `--update-snapshot`
# accepts it, while these record what has to be true whatever happened.

BRIEFS = "long-briefs"


def _served(corpus: Path, name: str) -> Path:
    return corpus / BRIEFS / "served" / name


def _rates(stdout: str) -> str:
    return next(ln for ln in stdout.splitlines() if ln.startswith("long briefs:"))


def test_the_long_brief_slice_reports_both_rates_and_the_thresholds(
    corpus: Path,
) -> None:
    """The control, and the calibration restated as a measurement: the numbers
    the task gate's constants were set from are reproduced by the shipped
    tree, beside the thresholds they were set to."""
    out = _eval(corpus)
    assert out.returncode == 0, out.stdout + out.stderr
    line = _rates(out.stdout)
    assert "7/8 served (0.875, floor 0.750)" in line, line
    assert "0/12 leaked (0.000, ceiling 0.084)" in line, line
    # Per-case rows too, so a single outcome moving is visible in a diff even
    # though it is under the rate slack.
    assert "[BRIEF-SERVED]" in out.stdout
    assert "[BRIEF-QUIET ]" in out.stdout


def test_coverage_under_the_floor_fails_the_run(corpus: Path) -> None:
    """A task gate that stops serving briefs it was calibrated to serve is the
    unit's headline failure, and it is silent everywhere else: every brief
    still gets a spawn, the spawn still runs, and nothing anywhere says the
    pointers stopped arriving."""
    for name in ("backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md"):
        _served(corpus, name).write_text(
            "# Brief\n\n" + "Sort the mailroom trays by postcode. " * 200
        )
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    assert "long-brief coverage" in out.stderr
    assert "under the 0.750 floor" in out.stderr
    assert "4/8 served" in _rates(out.stdout)


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
        path.write_text(path.read_text() + leak)
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
    for name in ("backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md"):
        _served(corpus, name).write_text("# Brief\n\n" + "mailroom trays. " * 200)
    snapshot = corpus / "eval-expectations.json"
    before = snapshot.read_text()
    out = _eval(corpus, "--update-snapshot")
    assert out.returncode != 0, out.stdout
    assert "under the 0.750 floor" in out.stderr
    assert "wrote" in out.stdout
    assert snapshot.read_text() != before, "the re-baseline itself must still land"


def test_a_rate_failure_survives_a_moved_corpus_instead_of_becoming_drift(
    corpus: Path,
) -> None:
    """A moved corpus makes every SNAPSHOT comparison unattributable, and the
    run refuses on that. A rate is not a comparison — it is an absolute
    measurement of the corpus in front of it — so it still answers, and it has
    to answer first: filed as drift, a coverage collapse gets re-baselined away
    by the very command the refusal recommends."""
    _drift(corpus)
    for name in ("backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md"):
        _served(corpus, name).write_text("# Brief\n\n" + "mailroom trays. " * 200)
    out = _eval(corpus)
    assert out.returncode != 0
    assert "long-brief coverage" in out.stderr
    assert "corpus moved" not in out.stderr


def test_an_edited_brief_reads_as_a_new_case_rather_than_inheriting_one(
    corpus: Path,
) -> None:
    """The corpus fingerprint covers the stores and not this directory, because
    these are the queries rather than the corpus. So the brief's own digest is
    in its snapshot key — otherwise an edited brief silently keeps the outcome
    recorded for the text it used to have, which is the one kind of drift no
    case line can report."""
    first = _eval(corpus)
    assert first.returncode == 0, first.stdout + first.stderr
    # The control, and the half that fails under a filename-only key: the
    # committed snapshot's keys are the ones this run produces, so nothing is
    # unrecorded before anything is edited.
    assert "NEW (no expectation recorded" not in first.stdout, first.stdout

    path = _served(corpus, "rotor-swap-programme.md")
    path.write_text(path.read_text() + "\nOne more paragraph nobody asked for.\n")
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
    assert "no task path — slice skipped" in out.stdout
    assert not re.search(r"long briefs: \d+/\d+ served", out.stdout), out.stdout


def _strip_task_path(tmp_path: Path) -> Path:
    """A writable copy of the package with `task_gate` renamed away."""
    stripped = tmp_path / "memkit"
    shutil.copytree(Path(__file__).resolve().parent.parent / "src" / "memkit", stripped)
    for path in (stripped, *stripped.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    src = stripped / "memory_prompt_recall.py"
    src.write_text(src.read_text().replace("def task_gate(", "def _no_task_gate("))
    return src


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
    what it does not know. A green run has to say which gates it ran."""
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
    corpus: Path,
) -> None:
    """The slice stopped at the relevance floor, so a brief whose emission the
    harness would refuse — a malformed `updatedInput`, or one over the write
    bound — scored as served. Retrieval is not delivery on this path.

    Driven by shrinking the write bound to a value every emission crosses: the
    ranker is untouched and every served brief still ranks its target first, so
    a slice that scored retrieval would report 7/8 unchanged.
    """
    before = _eval(corpus)
    assert "7/8 served" in _rates(before.stdout), before.stdout

    hook_dir = Path(__file__).resolve().parent.parent / "src" / "memkit"
    src = hook_dir / "memory_prompt_recall.py"
    original = src.read_text()
    try:
        src.write_text(original.replace("PIPE_BUFFER_BOUND = 16384",
                                        "PIPE_BUFFER_BOUND = 64"))
        out = _eval(corpus)
    finally:
        src.write_text(original)
    assert "0/8 served" in _rates(out.stdout), out.stdout
    assert out.returncode != 0, out.stdout
    assert "long-brief coverage" in out.stderr
