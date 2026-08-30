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
import json
import re
import shutil
import stat
import subprocess
import sys
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


def test_a_rate_failure_survives_a_moved_corpus_instead_of_becoming_drift(
    corpus: Path,
) -> None:
    """A moved corpus makes every SNAPSHOT comparison unattributable, and the
    run refuses on that. A rate is not a comparison — it is an absolute
    measurement of the corpus in front of it — so it still answers, and it has
    to answer first: filed as drift, a coverage collapse gets re-baselined away
    by the very command the refusal recommends."""
    _drift(corpus)
    _unserve(corpus, "backlash-rig.md", "gearbox-acceptance.md", "vessel-reassembly.md")
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
    CI; it is not a licence for one new wrong injection.

    One leak in twelve is 0.083, under the 0.084 ceiling, so the RATE holds —
    and the per-case row said `<- REGRESSION ... not gating` and never reached
    the exit code. That made a single new injection into an autonomous
    subagent's instructions a green run. The two controls do different jobs:
    the rate bounds systemic loosening, the snapshot bounds one case moving.
    """
    path = corpus / BRIEFS / "unserved" / "accessibility-audit.md"
    _write_brief(
        path,
        path.read_text()
        + "\n\nThe sprocket backlash after a gearbox rebuild traces to the "
        "shim stack rather than chain tension, and the flange fasteners want "
        "a crossing sequence over three passes.",
    )
    out = _eval(corpus)
    assert out.returncode != 0, out.stdout
    # The RATE held — this is the case the rate cannot catch.
    assert "1/16 leaked (0.062, ceiling 0.084)" in _rates(out.stdout), out.stdout
    assert "long-brief injection" not in out.stderr, out.stderr
    leak = next(ln for ln in out.stdout.splitlines() if "[BRIEF-LEAK  ]" in ln)
    assert "not gating" not in leak, leak
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
    neighbour's line and scores as delivered — the delivery gate satisfied by
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
    # The stock hook baselines the moved corpus: the divergence under test is
    # the copy's, and a run against a fingerprint that moved gates nothing.
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
    honour — a hook copy whose `TASK_BUDGET_SECONDS` has already expired serves
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
    population and both fed the rate denominators — one behaviour repeated
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
    contents, so a memory that mentions its neighbour by name is ordinary
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


def test_the_corpus_fingerprint_survives_a_lone_surrogate(
    corpus: Path, monkeypatch, tmp_path: Path
) -> None:
    """The fingerprint is taken before the gate does anything, so a strict
    encode there ends the run with a traceback and no eval at all.

    Two sources, and neither is exotic. A store id comes out of `json.load`,
    which turns an escaped `\\udXXX` in the config into a lone surrogate. A
    relative path comes out of `rglob`, which turns a filename the filesystem
    holds as undecodable bytes into one. Both used to reach a bare
    `.encode()`, which raises `UnicodeEncodeError` on either.

    Accepted-open once, on the argument that the raise is visible on a CLI —
    but visible is not the same as diagnosable: `UnicodeEncodeError` out of a
    hashing loop names neither the store nor the file, and the run it ends is
    the one that would have told the adopter what their corpus scores.
    """
    blob = json.loads((corpus / "memkit.json").read_text(encoding="utf-8"))
    surrogate = json.loads('"\\ud800"')
    blob["stores"][0]["id"] += surrogate
    (corpus / "memkit.json").write_text(json.dumps(blob), encoding="utf-8")
    cfg = hook.load_config(str(corpus / "memkit.json"))
    first = ev.corpus_fingerprint(cfg, corpus)
    assert len(first) == 64, first

    # Non-vacuity: the digest still SEPARATES, which is the whole reason it is
    # taken over the raw id rather than over a sanitized one.
    blob["stores"][0]["id"] += "x"
    (corpus / "memkit.json").write_text(json.dumps(blob), encoding="utf-8")
    assert ev.corpus_fingerprint(
        hook.load_config(str(corpus / "memkit.json")), corpus
    ) != first

    # The filename half. APFS refuses to create a name that is not valid
    # UTF-8, so the two answers the filesystem would give are supplied rather
    # than staged — the subject is what this function does with a name it is
    # handed, and on a filesystem that allows one it is handed exactly this.
    monkeypatch.setattr(
        Path, "rglob", lambda self, pat: iter([Path(f"{self}/m\udcff.md")])
    )
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"body")
    assert len(ev.corpus_fingerprint(cfg, corpus)) == 64


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
