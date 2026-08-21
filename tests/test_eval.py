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
