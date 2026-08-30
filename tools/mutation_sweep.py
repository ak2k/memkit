"""Break the source on purpose and require the suite to notice.

A green suite is evidence only if a broken tree would have turned it red. That
is not something a coverage number answers: a line can be executed by every
test in the file and asserted on by none of them, and the difference is
invisible until the day the line changes and nothing complains. This walks a
corpus of small, deliberate defects — one load-bearing rule each — applies
them one at a time, and requires the paired tests to fail.

WHAT A VERDICT MEANS

    CAUGHT           the selection was green, the mutation made it red
    CAUGHT-NOTHING   the selection stayed green: the rule is unguarded
    ANCHOR           the `old` text is not in the file, or not once
    NOOP             `old` == `new`, so the probe proves nothing
    BASELINE         the selection was already red before the mutation
    SELECTION        pytest could not collect the named nodes
    SKIPPED          every paired test skipped, so the probe asked nothing
    REVERT           the file did not come back byte-identical

Only CAUGHT is a pass, and every other verdict exits non-zero. The three that
look like bookkeeping are the ones that matter most: each is a way a sweep can
report a number it did not earn.

ANCHOR is an error rather than a skip. A probe whose anchor has moved is a
probe that stopped testing anything, and a sweep that skips it counts down its
own corpus silently, run after run, until "512/512 caught" is a statement about
whatever is left.

BASELINE is checked because a mutation that turns a red selection redder is not
a catch. The check is per distinct selection and cached, so probes sharing a
selection pay for it once.

SELECTION is separated from CAUGHT because pytest exits non-zero for a bad node
id too. Read as a catch, a typo in a probe would count as proof — the most
comfortable possible way for a sweep to be wrong. Only exit code 1 is a catch.

SKIPPED is separated from CAUGHT-NOTHING because a skipped test is a green test
to an exit code. Cases here skip without `uv`, without a network or without a
checkout, and reporting those as an unguarded rule blames the tree for the
machine. Both are failures — neither is proof — but only one of them is about
the code.

THE STALE BYTECODE TRAP

A mutation of the same byte length as the text it replaces leaves a `.pyc`
whose validation record — source mtime in whole seconds, source size — still
matches. Revert inside the same second and the cache written from the MUTATED
source stays valid for the reverted file, so the next probe imports code that
is in no file on disk. The failure is silent and it moves: the sweep reports a
catch or a miss belonging to a different probe.

So no bytecode is written at all. Caches are purged before the first probe, and
every child runs under PYTHONDONTWRITEBYTECODE, which is the half that matters
— purging alone just resets the clock.

`--prove-pyc-defence` stages that window rather than waiting for it: it pins the
source mtime across a same-length mutation and its revert, which is the same
second spelled deterministically. What comes back is a RED verdict on a tree
byte-identical to the commit, and then the same tree green under the defence. A
demonstration that has to be raced for is one that proves nothing on the run
where it did not fire.

RUN

    .venv/bin/python tools/mutation_sweep.py                # the whole corpus
    .venv/bin/python tools/mutation_sweep.py --smoke        # the tagged subset
    .venv/bin/python tools/mutation_sweep.py --selftest     # falsify the sweep
    .venv/bin/python tools/mutation_sweep.py --prove-pyc-defence

The interpreter must be one that imports both pytest and the memkit under
audit. That is asserted before any probe runs: a sweep pointed at an installed
copy would mutate this tree and test another, and every verdict would be
CAUGHT-NOTHING for a reason that has nothing to do with the tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "tools" / "mutation_probes.json"

# The selftests below all drive one cheap, central case.
CHILD_ENV_TEST = (
    "tests/test_doctor.py"
    "::test_the_gate_hands_a_child_no_variable_that_names_code"
)

CAUGHT = "CAUGHT"


class ProbeError(Exception):
    """A probe that cannot be run as written."""


def _load_corpus(path: Path) -> list:
    raw = json.loads(path.read_text(encoding="utf-8"))
    probes = raw["probes"] if isinstance(raw, dict) else raw
    seen = set()
    for probe in probes:
        for field in ("name", "module", "file", "old", "new", "tests"):
            if field not in probe:
                raise ProbeError(f"{probe.get('name', '<unnamed>')}: no {field}")
        name = probe["name"]
        if name in seen:
            raise ProbeError(f"{name}: two probes share this name")
        seen.add(name)
        if not probe["tests"]:
            raise ProbeError(f"{name}: names no test")
    return list(probes)


def _validate(probe: dict) -> None:
    """Everything decidable without running anything."""
    if probe["old"] == probe["new"]:
        raise ProbeError("old and new are the same text, so nothing is mutated")
    if probe.get("same_length") and len(probe["old"]) != len(probe["new"]):
        raise ProbeError(
            "declared same_length, but the two texts differ in length "
            f"({len(probe['old'])} vs {len(probe['new'])})"
        )
    target = REPO / probe["file"]
    if not target.is_file():
        raise ProbeError(f"{probe['file']} is not a file in this tree")


def _anchor(text: str, probe: dict) -> None:
    want = probe.get("occurrences", 1)
    found = text.count(probe["old"])
    if found != want:
        raise ProbeError(
            f"the anchor appears {found} times in {probe['file']}, wanted {want}"
        )


def _purge_bytecode(root: Path) -> int:
    removed = 0
    for cache in root.rglob("__pycache__"):
        if ".venv" in cache.parts or ".git" in cache.parts:
            continue
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    return removed


def _pytest_env(*, write_bytecode: bool) -> dict:
    env = dict(os.environ)
    # The half of the defence that lasts. Purging clears what is there; this
    # stops the run under audit from writing a fresh cache that the NEXT probe
    # would inherit.
    if write_bytecode:
        env.pop("PYTHONDONTWRITEBYTECODE", None)
    else:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_tests(
    python: str, nodes: list, *, write_bytecode: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        # Deliberately NOT `-q`: it drops the one-line outcome summary, and a
        # run whose every case SKIPPED is then indistinguishable from one that
        # passed. That distinction is the difference between "this rule is
        # unguarded" and "this machine could not ask".
        [
            python,
            "-m",
            "pytest",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-x",
            *nodes,
        ],
        cwd=REPO,
        env=_pytest_env(write_bytecode=write_bytecode),
        capture_output=True,
        text=True,
    )


def _preflight(python: str) -> None:
    """Assert the interpreter under test imports THIS tree.

    A sweep whose pytest resolves memkit from site-packages edits one tree and
    exercises another. Every probe then reports CAUGHT-NOTHING, which reads as
    an unguarded codebase rather than as a misconfigured run.
    """
    probe = (
        "import memkit, pytest, sys; "
        "print(memkit.__file__); print(pytest.__version__)"
    )
    out = subprocess.run(  # noqa: S603
        [python, "-c", probe], capture_output=True, text=True, cwd=REPO
    )
    if out.returncode != 0:
        raise SystemExit(
            f"{python} cannot import memkit and pytest:\n{out.stderr.strip()}"
        )
    where = Path(out.stdout.splitlines()[0]).resolve()
    if REPO not in where.parents:
        raise SystemExit(
            f"{python} imports memkit from {where},\nwhich is outside {REPO}. "
            "Install the tree under audit as editable first."
        )
    print(f"interpreter : {python}")
    print(f"memkit      : {where}")


class Sweep:
    """One pass over a corpus, against one interpreter."""

    def __init__(self, python: str, *, verify_baseline: bool = True) -> None:
        self.python = python
        self.verify_baseline = verify_baseline
        self._green: dict = {}

    def _baseline_is_green(self, nodes: list) -> tuple:
        """Whether `nodes` pass unmutated. Cached across probes."""
        key = tuple(nodes)
        if key not in self._green:
            out = _run_tests(self.python, nodes)
            self._green[key] = (out.returncode, out)
        code, out = self._green[key]
        return code == 0, code, out

    def run_probe(self, probe: dict, *, write_bytecode: bool = False) -> tuple:
        """Apply, test, revert. Returns (verdict, detail)."""
        try:
            _validate(probe)
        except ProbeError as exc:
            return "NOOP" if "same text" in str(exc) else "ANCHOR", str(exc)

        target = REPO / probe["file"]
        original = target.read_bytes()
        try:
            text = original.decode("utf-8")
        except UnicodeDecodeError as exc:
            return "ANCHOR", f"{probe['file']} is not utf-8: {exc}"

        try:
            _anchor(text, probe)
        except ProbeError as exc:
            return "ANCHOR", str(exc)

        nodes = list(probe["tests"])
        if self.verify_baseline:
            green, code, out = self._baseline_is_green(nodes)
            if not green:
                if code in (4, 5):
                    return "SELECTION", _tail(out)
                return "BASELINE", _tail(out)

        mutated = text.replace(probe["old"], probe["new"])
        target.write_bytes(mutated.encode("utf-8"))
        restored = False
        try:
            out = _run_tests(self.python, nodes, write_bytecode=write_bytecode)
        finally:
            # The revert happens whatever went wrong, but its VERDICT is read
            # below rather than returned here: a return inside `finally` would
            # discard an exception from the run above, which is the one report
            # worth more than any verdict this function can produce.
            target.write_bytes(original)
            restored = target.read_bytes() == original
        if not restored:
            return "REVERT", f"{probe['file']} did not come back"

        if out.returncode == 1:
            return CAUGHT, _first_failure(out)
        if out.returncode == 0:
            if _all_skipped(out):
                return "SKIPPED", "every paired test skipped, so nothing was asked"
            return "CAUGHT-NOTHING", "the selection stayed green"
        if out.returncode in (4, 5):
            return "SELECTION", _tail(out)
        return "CAUGHT-NOTHING", f"pytest exited {out.returncode}: {_tail(out)}"


def _counts(out: subprocess.CompletedProcess) -> dict:
    """pytest's own tally, per outcome, from its summary line."""
    found = re.findall(
        r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)",
        out.stdout or "",
    )
    tally: dict = {}
    for number, outcome in found:
        tally[outcome] = tally.get(outcome, 0) + int(number)
    return tally


def _all_skipped(out: subprocess.CompletedProcess) -> bool:
    """Whether the selection ran nothing at all.

    A skipped test is a GREEN test to an exit code, and several cases here skip
    on a machine with no `uv`, no network or no checkout. Counting those as
    "the rule is unguarded" would blame the tree for the environment — and
    counting them as caught would be worse. They get their own verdict.
    """
    tally = _counts(out)
    return bool(tally.get("skipped")) and not tally.get("passed")


def _tail(out: subprocess.CompletedProcess, lines: int = 6) -> str:
    body = (out.stdout or "") + (out.stderr or "")
    return " / ".join(body.strip().splitlines()[-lines:])[:400]


def _first_failure(out: subprocess.CompletedProcess) -> str:
    for line in (out.stdout or "").splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            return line.split(" - ")[0].strip()[:160]
    return "red"


# --- the sweep's own falsification ------------------------------------------
#
# A harness that reports what it is asked to report is worth nothing, so the
# cases it must NOT wave through are exercised against the real tree before any
# real verdict is believed. Each expects a specific verdict; a run where one of
# these comes back CAUGHT is a run whose numbers mean nothing.

SELFTESTS = [
    (
        "a no-op probe proves nothing and is refused",
        {
            "name": "selftest-noop",
            "module": "selftest",
            "file": "src/memkit/_exec.py",
            "old": "CHILD_ENV_KEEP",
            "new": "CHILD_ENV_KEEP",
            "tests": [CHILD_ENV_TEST],
        },
        "NOOP",
    ),
    (
        "an anchor that is not in the file is an error, never a skip",
        {
            "name": "selftest-anchor",
            "module": "selftest",
            "file": "src/memkit/_exec.py",
            "old": "def a_function_this_module_does_not_have(",
            "new": "def something_else(",
            "tests": [CHILD_ENV_TEST],
        },
        "ANCHOR",
    ),
    (
        "a mutation no paired test notices is reported, not passed",
        {
            "name": "selftest-caught-nothing",
            "module": "selftest",
            "file": "src/memkit/_exec.py",
            "old": '"""The one place this package starts a process',
            "new": '"""THE ONE PLACE this package starts a process',
            "tests": [CHILD_ENV_TEST],
        },
        "CAUGHT-NOTHING",
    ),
    (
        "a node id pytest cannot collect is not read as a catch",
        {
            "name": "selftest-selection",
            "module": "selftest",
            "file": "src/memkit/_exec.py",
            "old": 'CHILD_ENV_KEEP = ("HOME"',
            "new": 'CHILD_ENV_KEEP = ("HOMER"',
            "tests": ["tests/test_doctor.py::test_no_such_test_exists_here"],
        },
        "SELECTION",
    ),
]


def selftest(python: str) -> int:
    print("=== the sweep, falsified ===\n")
    sweep = Sweep(python)
    bad = 0
    for why, probe, want in SELFTESTS:
        got, detail = sweep.run_probe(probe)
        ok = got == want
        bad += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] {why}")
        print(f"      wanted {want}, got {got} — {detail}\n")
    print("the sweep refuses what it must" if not bad else f"{bad} selftest(s) wrong")
    return 1 if bad else 0


def prove_pyc_defence(python: str, probes: list) -> int:
    """Stage the stale-bytecode window, then show the defence closing it.

    The window is narrow and it is not narrow enough to ignore: a `.pyc` is
    revalidated against the source's mtime IN WHOLE SECONDS and its size in
    bytes, so a same-length edit reverted inside the same second as the
    compile leaves a cache that is still considered current for a file whose
    contents it no longer describes.

    Waiting for that to happen by chance would prove nothing on the run where
    it did not, so the mtime is pinned here rather than raced for. Pinning it
    is the whole of the staging — the size already matches, which is what
    `same_length` on a probe is for. What follows is a REVERTED, byte-clean
    tree whose tests fail, which is the sweep reporting a verdict that belongs
    to no probe.

    Then the same sequence with the defence on. A defence nobody has watched
    fail is a line of code with an opinion about itself.
    """
    # A python source file, specifically: the trap is about bytecode, and a
    # same-length edit to Markdown or JSON would make this a demonstration
    # that cannot fail whatever the defence does.
    same = [
        p
        for p in probes
        if p.get("same_length")
        and p["file"].startswith("src/")
        and p["file"].endswith(".py")
    ]
    if not same:
        print("no python-source probe is tagged same_length; nothing to prove")
        return 1
    probe = same[0]
    nodes = list(probe["tests"])
    target = REPO / probe["file"]
    original = target.read_bytes()
    mutated = original.decode("utf-8").replace(probe["old"], probe["new"])

    print("=== the stale-bytecode window ===\n")
    print(f"probe : {probe['name']}")
    print(f"file  : {probe['file']}")
    print(f"edit  : {len(probe['old'])} bytes replaced by {len(probe['new'])}\n")

    def pin(mtime: float) -> None:
        os.utime(target, (mtime, mtime))

    stamp = target.stat().st_mtime
    try:
        # Compile the MUTATED source into a cache, with the mtime pinned to the
        # value the reverted file will carry.
        _purge_bytecode(REPO)
        target.write_bytes(mutated.encode("utf-8"))
        pin(stamp)
        before = _run_tests(python, nodes, write_bytecode=True)
        print(f"1. mutated source, cache written   -> pytest rc={before.returncode}")

        # Revert. Same bytes as the commit, same size, same second.
        target.write_bytes(original)
        pin(stamp)
        clean = target.read_bytes() == original
        print(f"2. reverted, byte-identical={clean}")

        # No purge, and bytecode writing still on: whatever the cache says now
        # is what the interpreter will believe about a file that is correct.
        trapped = _run_tests(python, nodes, write_bytecode=True)
        print(f"3. defence OFF, clean tree         -> pytest rc={trapped.returncode}")

        # And the same clean tree under the defence.
        _purge_bytecode(REPO)
        guarded = _run_tests(python, nodes)
        print(f"4. defence ON,  clean tree         -> pytest rc={guarded.returncode}\n")
    finally:
        target.write_bytes(original)
        pin(stamp)
        _purge_bytecode(REPO)

    if guarded.returncode != 0:
        print("the guarded run is red on a clean tree; the sweep is not trustworthy")
        return 1
    if trapped.returncode == 0:
        print(
            "the window did not open on this filesystem (its mtime granularity\n"
            "is finer than the cache record, or the cache was not consulted).\n"
            "The defence is on for every real run regardless; this is a\n"
            "demonstration that did not fire, not a defence that failed."
        )
        return 0
    print(
        "the window is real: step 3 is a RED verdict on a tree that is\n"
        "byte-identical to the commit, produced by bytecode compiled from a\n"
        "mutation no longer on disk. Step 4 is the same tree with the caches\n"
        "purged and PYTHONDONTWRITEBYTECODE set, and it is green.\n"
        "That verdict would have been charged to whichever probe ran next."
    )
    return 0


def _git_is_clean() -> tuple:
    """Whether every TRACKED file is as committed.

    Untracked files are excluded deliberately. The question here is whether a
    probe reverted, and a probe only ever rewrites a file that is already in
    the tree — so an untracked scratch file is not evidence about that, and
    counting one would refuse to run in the state this tool is written in.
    """
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    dirty = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return (not dirty), dirty


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Break the source on purpose and require the suite to notice."
    )
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("-k", "--filter", default="", help="substring of probe name")
    ap.add_argument("--module", default="", help="only probes for this module")
    ap.add_argument("--smoke", action="store_true", help="only smoke-tagged probes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prove-pyc-defence", action="store_true")
    ap.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the was-it-green-first check (faster, and worth less)",
    )
    args = ap.parse_args(argv)

    probes = _load_corpus(Path(args.corpus))
    if args.module:
        probes = [p for p in probes if p["module"] == args.module]
    if args.filter:
        probes = [p for p in probes if args.filter in p["name"]]
    if args.smoke:
        probes = [p for p in probes if p.get("smoke")]

    if args.list:
        by_module: dict = {}
        for probe in probes:
            by_module.setdefault(probe["module"], []).append(probe["name"])
        for module in sorted(by_module):
            print(f"{module} ({len(by_module[module])})")
            for name in by_module[module]:
                print(f"  {name}")
        print(f"\n{len(probes)} probes")
        return 0

    _preflight(args.python)
    if args.selftest:
        return selftest(args.python)
    if args.prove_pyc_defence:
        return prove_pyc_defence(args.python, probes)

    clean_before, dirty = _git_is_clean()
    if not clean_before:
        print("the tree is not clean; a revert failure would be unreadable:")
        for line in dirty:
            print(f"  {line}")
        return 1

    purged = _purge_bytecode(REPO)
    print(f"purged      : {purged} bytecode caches")
    print(f"probes      : {len(probes)}\n")

    sweep = Sweep(args.python, verify_baseline=not args.no_baseline)
    tally: dict = {}
    failures = []
    started = time.monotonic()
    for index, probe in enumerate(probes, 1):
        verdict, detail = sweep.run_probe(probe)
        tally[verdict] = tally.get(verdict, 0) + 1
        mark = "." if verdict == CAUGHT else "!"
        print(f"{mark} [{index:>3}/{len(probes)}] {verdict:<14} {probe['name']}")
        if verdict != CAUGHT:
            print(f"      {detail}")
            failures.append((probe["name"], verdict, detail))

    caught = tally.get(CAUGHT, 0)
    print(f"\nCAUGHT {caught}/{len(probes)}  in {time.monotonic() - started:.0f}s")
    for verdict in sorted(v for v in tally if v != CAUGHT):
        print(f"  {verdict}: {tally[verdict]}")

    clean_after, dirty = _git_is_clean()
    if not clean_after:
        print("\nthe tree did NOT come back clean:")
        for line in dirty:
            print(f"  {line}")
    else:
        print("the tree is byte-identical to where it started")

    if failures or not clean_after:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
