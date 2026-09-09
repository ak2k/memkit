"""What the built artifacts and the type-check configs have to carry.

Two things that are true of the repo rather than of any function in it, and
that go wrong silently: a licence obligation met by a build-backend default
nobody declared, and a second pyright config whose include list is a hand-kept
list of files.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "src" / "memkit"
HOOK = PKG / "memory_prompt_recall.py"


# --- the licence obligation, in the artifacts themselves ----------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, Path]:
    """One wheel and one sdist, built from this checkout.

    Built rather than asserted about, because the claim is about what an
    adopter RECEIVES. Every cheaper form of this test — the include list says
    NOTICE, the metadata declares it — is a claim about the recipe, and the
    recipe is exactly what was already right while the artifact was in doubt.

    Needs a build frontend and a network to resolve the backend, so it skips
    where there is neither: `nix flake check` runs this suite in a sandbox with
    no uv and no network. The plain-python CI leg has both, which is where this
    gates.
    """
    if shutil.which("uv") is None:
        pytest.skip("no uv to build with — the plain-python CI leg is where this runs")
    out = tmp_path_factory.mktemp("dist")
    built = subprocess.run(
        ["uv", "build", "--out-dir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stderr
    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, sorted(p.name for p in out.iterdir())
    return {"wheel": wheels[0], "sdist": sdists[0]}


def test_the_wheel_carries_notice_beside_the_licence(built) -> None:
    """Apache-2.0 §4(d) obligates redistributing NOTICE, and the wheel is a
    redistribution: `uvx --from git+...` clones this repo and builds ONE, so an
    obligation met only by the sdist is met by nothing an adopter of the only
    python channel M2 ships ever receives.
    """
    names = zipfile.ZipFile(built["wheel"]).namelist()
    licences = [n for n in names if "/licenses/" in n]
    assert any(n.endswith("/licenses/NOTICE") for n in licences), names
    # Beside the LICENSE, not instead of it: §4(d) is in addition to §4(a).
    assert any(n.endswith("/licenses/LICENSE") for n in licences), names


def test_the_sdist_carries_notice_beside_the_licence(built) -> None:
    with tarfile.open(built["sdist"]) as tar:
        names = tar.getnames()
    assert any(n.endswith("/NOTICE") for n in names), names
    assert any(n.endswith("/LICENSE") for n in names), names


def test_the_wheel_declares_every_console_script(built) -> None:
    """The entry points are what an adopter actually gets on PATH, and nothing
    was checking them. The suites reach the dispatcher through `-m`, the flake
    checks invoke the other three binaries by path, and a `[project.scripts]`
    key that named a module or callable that does not exist would install a
    console script that traceback on first use — past every gate here.
    """
    with zipfile.ZipFile(built["wheel"]) as z:
        entry_points = next(
            z.read(n).decode()
            for n in z.namelist()
            if n.endswith(".dist-info/entry_points.txt")
        )
    declared = dict(
        line.split("=", 1) for line in entry_points.splitlines() if "=" in line
    )
    scripts = {name.strip(): target.strip() for name, target in declared.items()}
    assert scripts == {
        "memkit": "memkit.cli:cli",
        "memory-recall": "memkit.memory_prompt_recall:cli",
        "memory-integrity": "memkit.memory_integrity:cli",
        "memory-eval": "memkit.eval_memory_recall:cli",
    }, entry_points
    # And that each target resolves — the half a text assertion cannot make.
    for target in scripts.values():
        module, _, attr = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attr)), target


def test_the_licence_files_are_declared_and_not_left_to_a_default(built) -> None:
    """Both files land today even with nothing declaring them — hatchling's
    default license-files glob picks up NOTICE* on its own. That is the state
    this pins against: an obligation resting on an undeclared backend default
    is one a backend bump drops with no diff to read, and the artifact tests
    above would then be the first thing to notice, on whichever PR happened to
    bump it. Declared metadata makes it somebody's edit instead.
    """
    with zipfile.ZipFile(built["wheel"]) as z:
        metadata = next(
            z.read(n).decode() for n in z.namelist() if n.endswith(".dist-info/METADATA")
        )
    declared = {
        line.split(":", 1)[1].strip()
        for line in metadata.splitlines()
        if line.startswith("License-File:")
    }
    assert declared == {"LICENSE", "NOTICE"}, metadata


# --- which pyright config a new file lands in --------------------------------


def _imports(path: Path) -> set[str]:
    """First-party module names `path` imports, absolute and relative.

    `from memkit import cli` names a MODULE, not an attribute, whenever
    `memkit/cli.py` exists — so the imported names are offered as
    `memkit.<name>` too and `_resolve` keeps whichever ones are files. Reading
    only `node.module` there would see `memkit` and miss the module actually
    being pulled in.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = f"memkit.{node.module}" if node.level and node.module else node.module
            if node.level and not node.module:
                base = "memkit"
            if base:
                found.add(base)
                found |= {f"{base}.{a.name}" for a in node.names if a.name != "*"}
    return {n for n in found if n == "memkit" or n.startswith("memkit.")}


def _module_file(name: str) -> Path | None:
    """The file `name` names, if it names a module at all — else None.

    A dotted tail may be a module (`memkit.a.b` -> `a/b.py`), a subpackage
    (`a/b/__init__.py`), or not a module at all: `from memkit import __version__`
    puts an ATTRIBUTE in the same syntactic position as a submodule, and
    nothing about the name says which it is.
    """
    parts = name.split(".")[1:]  # drop the leading `memkit`
    if not parts:
        return PKG / "__init__.py"
    stem = PKG.joinpath(*parts)
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _resolve(name: str) -> set[Path]:
    """Files that importing `name` executes: the module itself plus every
    package `__init__.py` on the way to it. Empty when `name` is not a module.

    Deliberately NOT seeded with `memkit/__init__.py`. Seeding it made every
    first-party name resolve to at least one file, which made the guard below
    unable to fire for any input at all — a typo'd import contributed nothing
    and passed, which is exactly the quiet shrink the guard is written to stop.
    """
    module = _module_file(name)
    if module is None:
        return set()
    parts = name.split(".")[1:]
    # Importing anything under the package runs the package's own __init__
    # first, and every intermediate one on the way down.
    files = {PKG / "__init__.py", module}
    for i in range(len(parts) - 1):
        init = PKG.joinpath(*parts[: i + 1]) / "__init__.py"
        if init.is_file():
            files.add(init)
    return {f for f in files if f.is_file()}


def _import_closure(start: Path) -> set[Path]:
    """Every file importing `start` executes, transitively, first-party only.

    Raises AssertionError on a first-party name that is neither a module nor a
    real attribute of its parent. That distinction cannot be made from the name
    — `from memkit import cli` and `from memkit import __version__` are the same
    syntax — so it is settled by importing the parent and asking it. Failing
    loudly is the point: a name that silently contributes nothing returns a
    SMALLER closure, and a smaller closure is what lets the equality assertion
    below agree with an include list that has a file missing.
    """
    seen = {start}
    queue = [start]
    while queue:
        for name in _imports(queue.pop()):
            resolved = _resolve(name)
            if not resolved:
                parent, _, attr = name.rpartition(".")
                assert _module_file(parent) is not None, (
                    f"{name!r} is not a memkit module and neither is {parent!r}"
                )
                assert hasattr(importlib.import_module(parent), attr), (
                    f"{name!r} is neither a memkit module nor an attribute of "
                    f"{parent!r} — a typo here would otherwise shrink the closure"
                )
                continue
            for module in resolved - seen:
                seen.add(module)
                queue.append(module)
    return seen


# The two entry points a 3.9 interpreter may execute. The hook, because the
# harness runs it with whatever `python3` the PATH resolves to. The dispatcher,
# because the plugin's `bin/memkit` runs it on that same interpreter: only
# checker-backed work routes to 3.12, and sending the whole dispatcher there
# would put `memkit doctor` out of reach on a stock-python mac, which is the
# machine that most needs to ask whether its install works.
ENTRY_POINTS_39 = (HOOK, PKG / "cli.py")


def _floor_39_closure() -> set[Path]:
    """Every file a 3.9 interpreter can reach from either entry point."""
    return set().union(*(_import_closure(entry) for entry in ENTRY_POINTS_39))


def test_the_39_config_covers_exactly_the_39_entry_points() -> None:
    """That config's `include` is a hand-written file list, so a module either
    entry point grows an import of is unchecked at 3.9 until somebody remembers
    to add it. What that costs is invisible: a file that raises on import is
    reported by the harness as nothing at all, which is also what a corpus with
    nothing to say looks like.

    The direction is the easy half to invert. A module that merely IMPORTS one
    of these does not belong here — nothing puts it in front of the 3.9
    interpreter.
    """
    config = json.loads((REPO / "pyrightconfig-hook39.json").read_text())
    listed = {(REPO / p).resolve() for p in config["include"]}
    assert listed == _floor_39_closure()


def test_the_dispatcher_is_in_the_39_floor_because_a_wrapper_runs_it_there() -> None:
    """The pin above is an equality against a closure, so it would stay green
    if `cli.py` were dropped from BOTH the config and the entry-point list in
    one edit. This is the half that says which entry points there are, and it
    is a claim about `bin/memkit`: that file execs `memkit.cli` with the same
    interpreter the hook wrapper resolves, and nothing else connects the two.
    """
    wrapper = (REPO / "bin" / "memkit").read_text(encoding="utf-8")
    assert "-m memkit.cli" in wrapper
    assert PKG / "cli.py" in set(ENTRY_POINTS_39)
    assert HOOK in _import_closure(PKG / "cli.py"), (
        "cli.py no longer imports the hook — check whether it still answers to "
        "the 3.9 floor before editing this"
    )


def test_the_closure_helper_sees_the_import_shapes_real_code_uses(tmp_path) -> None:
    """The pin above is only as good as this walk, and a walk that misses an
    import shape does not fail — it returns a SMALLER closure, which is
    precisely what makes an equality assertion agree with an include list that
    has a file missing. So the shapes are pinned directly.
    """
    source = tmp_path / "probe.py"
    source.write_text(
        "import memkit.cli\n"
        "from memkit import memory_prompt_recall\n"
        "from memkit.memory_prompt_recall import SCHEMA\n"
        "from . import eval_memory_recall\n"
        "import json\n"
    )
    found = _imports(source)
    assert "memkit.cli" in found
    # `from memkit import memory_prompt_recall` names a module, not an
    # attribute — reading only `node.module` would see `memkit` and miss it.
    assert "memkit.memory_prompt_recall" in found
    # Relative imports resolve against the package, not against nothing.
    assert "memkit.eval_memory_recall" in found
    assert not any(n.startswith("json") for n in found)

    # And the name -> file step: a module, plus every __init__.py executed on
    # the way to it. A name that is not a module resolves to nothing at all —
    # which is what makes the guard below able to fire.
    assert _resolve("memkit.cli") == {PKG / "__init__.py", PKG / "cli.py"}
    assert _resolve("memkit.memory_prompt_recall.SCHEMA") == set()


def test_the_closure_walk_fails_on_a_first_party_import_that_is_not_there(
    tmp_path,
) -> None:
    """The guard has to be able to FIRE, and the first version of it could not:
    the resolver seeded every name with `memkit/__init__.py`, so a typo'd
    import resolved to one real file, contributed nothing, and passed. The
    comment claimed a behaviour the code did not have.

    The hard half is that an attribute tail must stay legal —
    `from memkit import __version__` and `from memkit import cli` are the same
    syntax — so the distinction is settled by importing the parent and asking
    it, not by looking at the name.
    """
    for line in (
        "from memkit import definitely_absent\n",
        "import memkit.definitely_absent\n",
    ):
        probe = tmp_path / "probe.py"
        probe.write_text(line)
        with pytest.raises(AssertionError, match="definitely_absent"):
            _import_closure(probe)

    legal = tmp_path / "legal.py"
    legal.write_text(
        # An attribute of a module, and a module imported from its package:
        # neither may fire, and the second must still be walked into.
        "from memkit.memory_prompt_recall import SCHEMA\n"
        "from memkit import cli\n"
    )
    reached = _import_closure(legal)
    assert HOOK in reached and PKG / "cli.py" in reached


def test_the_package_config_covers_new_files_without_being_edited() -> None:
    """The other half of the convention, and the reason it is only ever one
    file that needs a hand: this config includes whole directories, so anything
    added under them is checked with no edit at all."""
    config = json.loads((REPO / "pyrightconfig.json").read_text())
    assert {"src", "tests", "tools"} <= set(config["include"])
    assert (REPO / "src" / "memkit" / "cli.py").is_file()


FLOOR_REQUIRED_ENV = "MEMKIT_FLOOR_REQUIRED"


def _floor_interpreter() -> str | None:
    """A real 3.9, or None.

    `uv python find` first, because `uv python install 3.9` provisions one in
    well under a second and that is what makes this affordable as a gate; a
    `python3.9` on PATH answers too, for a machine that has one already.
    """
    for probe in (["uv", "python", "find", "3.9"],):
        try:
            out = subprocess.run(probe, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return shutil.which("python3.9")


def test_the_hook_and_both_subcommands_run_on_a_real_39(tmp_path) -> None:
    """The floor, EXECUTED — which is what a static pass cannot do.

    pyright at 3.9 catches a PEP-604 annotation evaluated at runtime and a
    3.10+ call whose type it can see. It does not catch a module attribute
    that exists in the version it was told about and not in the one the harness
    runs: `sqlite3.SQLITE_BUSY` landed in 3.11, and a reference to it reachable
    on 3.9 is a hook that raises on a stock mac — reported by the harness as
    nothing at all, which is also what a corpus with nothing to say looks like.

    CI's own comment said no runner ships 3.9 any more. That was true and is
    not: `uv python install 3.9` provisions one in about a tenth of a second,
    so the floor can be run rather than argued about. Set
    `MEMKIT_FLOOR_REQUIRED=1` — CI does — and a missing interpreter fails this
    rather than skipping it, because a gate that quietly stops gating is the
    shape of the failure it exists to catch.
    """
    interpreter = _floor_interpreter()
    if interpreter is None:
        if os.environ.get(FLOOR_REQUIRED_ENV) == "1":
            raise AssertionError(
                f"{FLOOR_REQUIRED_ENV}=1 and no 3.9 interpreter was found — "
                "`uv python install 3.9` provisions one"
            )
        pytest.skip("no python3.9 available; MEMKIT_FLOOR_REQUIRED=1 makes this fail")
    assert interpreter is not None
    # A HOME OF ITS OWN, WITH SOMETHING TO LOSE IN IT. The floor script is not
    # a pytest module, so no fixture isolates it and the runner passes the
    # whole environment through — and it called `_sweep()` fifteen lines before
    # it redirected HOME, so running the suite unlinked from the developer's
    # real cache (28,000 files here) and rewrote its cursor. Seeding a
    # collectible file in the HOME handed over is what turns "it is hermetic"
    # into something this can observe.
    home = tmp_path / "home"
    state = home / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    victim = state / "aaaaaaaa-1111-4222-8333-aaaaaaaaaaaa.json"
    victim.write_text("{}", encoding="utf-8")
    stale = time.time() - 30 * 86400
    os.utime(victim, (stale, stale))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMKIT_")}
    env["HOME"] = str(home)
    env.pop("XDG_CACHE_HOME", None)
    out = subprocess.run(
        [interpreter, str(REPO / "tests" / "floor39.py")],
        capture_output=True, text=True, timeout=600,
        env=env,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "floor39: ok on 3.9" in out.stdout, out.stdout
    assert victim.is_file(), "the floor gate swept the HOME it was handed"


def test_the_floor_gate_fails_rather_than_skips_when_it_is_required(
    monkeypatch, tmp_path
):
    """A gate that quietly stops gating is the shape of the failure the gate
    exists to catch, so the skip has to be switchable off and the switch has to
    be tested — otherwise the one thing CI relies on is the one thing nobody
    has watched work."""
    monkeypatch.setattr(sys.modules[__name__], "_floor_interpreter", lambda: None)
    monkeypatch.setenv(FLOOR_REQUIRED_ENV, "1")
    # BaseException and then a type check, not `pytest.raises(AssertionError)`:
    # a `Skipped` raised inside a `raises(AssertionError)` block propagates and
    # marks this case skipped, which reads as green. The exact failure this
    # test exists to catch would therefore have been invisible to it.
    with pytest.raises(BaseException) as caught:  # noqa: B017, PT011
        test_the_hook_and_both_subcommands_run_on_a_real_39(tmp_path)
    assert caught.typename == "AssertionError", caught.typename
    assert "no 3.9 interpreter" in str(caught.value)

    monkeypatch.delenv(FLOOR_REQUIRED_ENV)
    with pytest.raises(BaseException) as caught:  # noqa: B017, PT011
        test_the_hook_and_both_subcommands_run_on_a_real_39(tmp_path)
    assert caught.typename == "Skipped", caught.typename


def test_the_wrapper_guards_exactly_the_files_it_will_import() -> None:
    """The third copy of the 3.9 closure, and the only one nothing pinned.

    `pyrightconfig-hook39.json`'s include list and `PAYLOAD` are both asserted
    equal to a computed closure. `bin/memkit`'s `for _need in ...` loop holds
    the same fact and was hand-edited three times this milestone with nothing
    checking it — and a module added later and forgotten there produces exactly
    the failure the loop's own comment says it exists to prevent: a raw
    traceback out of the import machinery where the sibling wrappers print one
    sentence, on the surface an adopter reaches when something is already
    wrong.
    """
    wrapper = (REPO / "bin" / "memkit").read_text(encoding="utf-8")
    listed = re.search(r"for _need in ([^;]+); do", wrapper)
    assert listed, wrapper
    guarded = {name.strip() for name in listed.group(1).split() if name.strip()}
    reachable = {str(path.relative_to(REPO)) for path in _import_closure(PKG / "cli.py")}
    assert guarded == reachable, (
        sorted(guarded - reachable), sorted(reachable - guarded)
    )


# --- which suite a new test file lands in ------------------------------------


def test_every_test_file_is_in_the_flake_suite_map() -> None:
    """`flake.nix` names each suite by hand, and the nix leg is where that map
    is read.

    The throw it already carries is the right behaviour in the wrong PLACE: a
    test file added with no entry breaks `nix flake check` at EVALUATION, so
    every other check in the flake stops running too and the whole nix leg
    reports one error about a file nobody was thinking about. This fails in the
    suite the author is already running, names the file, and says what to add.

    A regex rather than a nix evaluation, because the point is to answer on a
    machine with no nix on it. It fails LOUDLY when it cannot find the block: a
    test that quietly asserts about an empty set is the same defect one level
    up.
    """
    flake = (REPO / "flake.nix").read_text(encoding="utf-8")
    block = re.search(r"\n\s*suiteNames = \{\n(.*?)\n\s*\};\n", flake, re.S)
    assert block, (
        "flake.nix has no `suiteNames = { ... };` block this test can read. "
        "It was renamed or restructured — update this test with it, because an "
        "unparsed block here checks nothing."
    )
    mapped = dict(re.findall(r'"([^"]+\.py)"\s*=\s*"([^"]+)";', block.group(1)))
    assert mapped, block.group(1)
    on_disk = {
        path.name for path in (REPO / "tests").glob("test_*.py") if path.is_file()
    }
    assert on_disk, "no test files found — this test is measuring the wrong tree"
    missing = sorted(on_disk - set(mapped))
    assert not missing, (
        f"tests/{missing} has no entry in flake.nix's suiteNames, so "
        "`nix flake check` fails to EVALUATE and no check in the flake runs. "
        "Add a line naming the suite."
    )
    # A name for a file that is not there is a suite that never runs: the
    # rename half of the same fact.
    stale = sorted(set(mapped) - on_disk)
    assert not stale, f"suiteNames names {stale}, which is not in tests/"
    # And two files under one name is one derivation running one of them.
    assert len(set(mapped.values())) == len(mapped), sorted(mapped.items())


# --- every guard in the auto-memory closure, and the probe that pins it ------


_CLOSURE_MODULES = ("src/memkit/cli_doctor.py", "src/memkit/harness_memory.py")

# `_auto_memory_rows` is the row under audit. The other four are entry points
# it reaches through data rather than through a call this walk could resolve:
# the settings reader, the scope list, the harness's own enumeration, and the
# directory resolver.
_CLOSURE_SEEDS = (
    "_auto_memory_rows",
    "Settings.__init__",
    "settings_scopes",
    "inventory",
    "harness_dir",
)

_EXIT_STATEMENTS = (ast.Return, ast.Continue, ast.Raise)

# Identity, not line: `(file, function, kind, ordinal within function)`. Line
# numbers are informative only — one edit above a function shifts every guard
# below it, and a line-keyed table would go red on guards nobody touched.
_UNPROBED: dict[tuple[str, str, str, int], str] = {
    ("cli_doctor.py", "_managed_dir", "if->exit", 1): (
        "on darwin the managed settings directory is the Library one"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", 1): (
        "a settings file that is not there leaves the scope empty, not failed"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", 2): (
        "a settings file that cannot be opened is recorded FORBIDDEN"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", 3): (
        "any other OS error on that file is recorded UNREADABLE"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", 4): (
        "settings that do not parse are recorded UNPARSED"
    ),
    ("cli_doctor.py", "Settings.__init__", "if->exit", 2): (
        "settings whose top level is not an object are UNPARSED, not empty"
    ),
    ("cli_doctor.py", "_session_cwd", "except", 1): (
        "returns '' when the session's own directory will not resolve"
    ),
    ("cli_doctor.py", "Machine.config", "if->exit", 1): (
        "no resolved config path means there is no config to read"
    ),
    ("cli_doctor.py", "Machine.config", "except", 1): (
        "a ConfigError is kept as the config's error rather than raised"
    ),
    ("cli_doctor.py", "Machine.config", "except", 2): (
        "anything else out of the loader is kept as text, so doctor survives it"
    ),
    ("cli_doctor.py", "_store_relation", "if->exit", 2): (
        "the first store root holding the directory decides at versus over"
    ),
    ("cli_doctor.py", "_how_inside", "if->exit", 1): (
        "a directory under a pruned name is 'pruned' before anything else"
    ),
    ("cli_doctor.py", "_placed", "if->exit", 1): (
        "outside every store, nothing retrieves what lands there"
    ),
    ("cli_doctor.py", "_placed", "if->exit", 2): (
        "a directory at or over a corpus root is refused, with what it costs"
    ),
    ("cli_doctor.py", "_placed", "if->exit", 3): (
        "a directory under a pruned name is refused as never indexed"
    ),
    ("cli_doctor.py", "_placed", "if->exit", 4): (
        "a store with no search/ yet is refused, naming the root it would get"
    ),
    ("cli_doctor.py", "_nearest_store", "if->exit", 1): (
        "no config means no nearest store and a distance of zero"
    ),
    ("cli_doctor.py", "_odd_switch", "if->exit", 1): (
        "no scope, or a real bool, is not an odd value to remark on"
    ),
    ("cli_doctor.py", "_checkout_remedy", "if->exit", 1): (
        "the checkout scope gets the remedy that names what changing it costs"
    ),
    ("cli_doctor.py", "_env_switch_note", "if->exit", 1): (
        "no forced value means no note about the variable"
    ),
    ("cli_doctor.py", "_env_switch_note", "if->exit", 2): (
        "a forced-on value says no settings scope turns the feature off"
    ),
    ("cli_doctor.py", "_env_switch_remedy", "if->exit", 1): (
        "a forced-on value's remedy is to unset the variable, not edit settings"
    ),
    ("cli_doctor.py", "_adopter_owns", "if->exit", 1): (
        "the named scope's own flag answers; no other scope stands in for it"
    ),
    ("cli_doctor.py", "_declared_below", "if->exit", 1): (
        "a name outside SCOPE_ORDER has nothing below it"
    ),
    ("cli_doctor.py", "_declared_below", "if->exit", 2): (
        "a scope that is not on this machine is skipped, not counted"
    ),
    ("cli_doctor.py", "_default_memory_dir", "if->exit", 1): (
        "with no session directory the harness's own directory is underivable"
    ),
    ("cli_doctor.py", "_default_memory_dir", "except", 1): (
        "a key that will not compute is reported as unknown, never raised"
    ),
    ("cli_doctor.py", "_consolidation_recency", "if->exit", 1): (
        "no default directory means no recency to report"
    ),
    ("cli_doctor.py", "_left_behind", "if->exit", 1): (
        "nothing outside the configured directory is nothing left behind"
    ),
    ("cli_doctor.py", "_auto_memory_rows", "if->exit", 3): (
        "an off switch this checkout carries gets the checkout remedy"
    ),
    ("cli_doctor.py", "_auto_memory_rows", "if->exit", 4): (
        "scopes contradicting the off switch are disclosed, not passed over"
    ),
    ("harness_memory.py", "project_key", "if->exit", 2): (
        "a key over the cap is refused: the harness's suffix is unmeasured"
    ),
    ("harness_memory.py", "_project_path", "except", 1): (
        "a cwd that will not resolve is used as it was given"
    ),
    ("harness_memory.py", "_project_path", "if->exit", 1): (
        "no repository root leaves the resolved path as the project path"
    ),
    ("harness_memory.py", "_project_path", "if->exit", 2): (
        "no common ancestor leaves the resolved path as the project path"
    ),
    ("harness_memory.py", "_project_path", "if->exit", 3): (
        "a submodule's git directory makes its worktree root the project"
    ),
    ("harness_memory.py", "_project_path", "except", 2): (
        "an unknown root, or a failing resolve, leaves the resolved path"
    ),
    ("harness_memory.py", "switch", "if->exit", 1): (
        "the first scope in the harness's order that declares the key answers"
    ),
    ("harness_memory.py", "harness_dir", "if->exit", 1): (
        "a value that is not a non-empty string names no directory"
    ),
    ("harness_memory.py", "env_switch", "if->exit", 1): (
        "an unset or empty variable is no answer, not an off one"
    ),
    ("harness_memory.py", "env_switch", "if->exit", 2): (
        "a spelling the harness reads as off means the feature does not run"
    ),
    ("harness_memory.py", "configured_dir", "if->exit", 1): (
        "no scope declares the key, so no directory is configured"
    ),
    ("harness_memory.py", "configured_dir", "if->exit", 2): (
        "a declared value the resolver rejects configures no directory either"
    ),
}


def _qualified_functions(tree: ast.AST, module: str) -> dict:
    """`{qualified name: (module, node)}` for every function in one module."""
    found: dict[str, tuple[str, ast.AST]] = {}

    def descend(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = prefix + child.name
                found[qualified] = (module, child)
                descend(child, qualified + ".")
            elif isinstance(child, ast.ClassDef):
                descend(child, prefix + child.name + ".")
            else:
                descend(child, prefix)

    descend(tree, "")
    return found


def _auto_memory_closure(functions: dict) -> set:
    """The seeds plus every function they reach, calls resolved by simple name.

    By NAME, because the alternative is type inference: `machine.config()` and
    `_placed(...)` are the same kind of edge to this walk, and a walk that
    followed only the unambiguous ones would leave the guards behind the
    ambiguous ones unaccounted for. The cost is that a same-named function in
    the other module joins the closure — an over-approximation, which is the
    safe direction for a floor.
    """
    by_simple: dict[str, list[str]] = {}
    for qualified in functions:
        by_simple.setdefault(qualified.rsplit(".", 1)[-1], []).append(qualified)

    reached: set[str] = set()
    pending = [seed for seed in _CLOSURE_SEEDS if seed in functions]
    assert len(pending) == len(_CLOSURE_SEEDS), sorted(set(_CLOSURE_SEEDS) - set(functions))
    while pending:
        qualified = pending.pop()
        if qualified in reached:
            continue
        reached.add(qualified)
        for node in ast.walk(functions[qualified][1]):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            name = getattr(called, "id", None) or getattr(called, "attr", None)
            pending.extend(
                candidate
                for candidate in by_simple.get(name or "", ())
                if candidate not in reached
            )
    return reached


def _guards_owned_by(node: ast.AST, blocked: set) -> list:
    """Every guard under `node` that no nested closure function owns first."""
    guards = []
    pending = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        if id(child) in blocked:
            continue
        if isinstance(child, ast.ExceptHandler):
            guards.append(("except", child))
        elif isinstance(child, ast.If) and any(
            isinstance(statement, _EXIT_STATEMENTS) for statement in child.body
        ):
            guards.append(("if->exit", child))
        pending.extend(ast.iter_child_nodes(child))
    return guards


def _probe_spans(text: str, probes: list, module: str) -> list:
    """`(probe name, first line, last line)` for each probe anchored once here.

    ONCE is the rule the pairing needs: a probe whose `old` appears twice is
    mutated in both places, so it says nothing about which guard it covers.
    """
    spans = []
    for probe in probes:
        if probe["file"] != module or probe.get("occurrences", 1) != 1:
            continue
        if text.count(probe["old"]) != 1:
            continue
        start = text.index(probe["old"])
        first = text.count("\n", 0, start) + 1
        last = first + probe["old"].count("\n")
        if probe["old"].endswith("\n"):
            last -= 1
        spans.append((probe["name"], first, last))
    return spans


def _guard_table() -> list:
    """`(file, function, kind, ordinal, lineno, end_lineno, probe names)`."""
    functions: dict[str, tuple[str, ast.AST]] = {}
    text: dict[str, str] = {}
    for module in _CLOSURE_MODULES:
        text[module] = (REPO / module).read_text(encoding="utf-8")
        functions.update(_qualified_functions(ast.parse(text[module]), module))

    closure = _auto_memory_closure(functions)
    blocked = {id(functions[name][1]) for name in closure}
    probes = json.loads(
        (REPO / "tools" / "mutation_probes.json").read_text(encoding="utf-8")
    )["probes"]
    spans = {module: _probe_spans(text[module], probes, module) for module in text}

    rows = []
    for qualified in closure:
        module, node = functions[qualified]
        for kind, guard in _guards_owned_by(node, blocked - {id(node)}):
            rows.append((module, qualified, kind, guard.lineno, guard.end_lineno))
    rows.sort()

    ordinals: dict[tuple, int] = {}
    table = []
    for module, qualified, kind, lineno, end_lineno in rows:
        key = (module, qualified, kind)
        ordinals[key] = ordinals.get(key, 0) + 1
        covering = tuple(
            name
            for name, first, last in spans[module]
            if first <= end_lineno and last >= lineno
        )
        table.append(
            (
                Path(module).name,
                qualified,
                kind,
                ordinals[key],
                lineno,
                end_lineno,
                covering,
            )
        )
    table.sort(key=lambda row: (row[0], row[4]))
    return table


def _printed(table: list) -> str:
    lines = []
    for name, qualified, kind, ordinal, lineno, end_lineno, covering in table:
        lines.append(
            f"{name}\t{qualified}\t{kind}\t{ordinal}\t{lineno}-{end_lineno}\t"
            + (",".join(covering) if covering else "UNPROBED")
        )
    return "\n".join(lines)


def _refrozen(table: list) -> str:
    """`_UNPROBED` as it would have to read for this tree — paste-ready."""
    lines = ["_UNPROBED = {"]
    for name, qualified, kind, ordinal, _lineno, _end, covering in table:
        if covering:
            continue
        identity = (name, qualified, kind, ordinal)
        reason = _UNPROBED.get(identity, "WHAT THIS GUARD REFUSES OR RETURNS")
        lines.append(f'    {identity!r}: (\n        "{reason}"\n    ),')
    lines.append("}")
    return "\n".join(lines)


def test_every_guard_in_the_auto_memory_closure_has_a_probe() -> None:
    """The floor under `_auto_memory_rows`: no guard arrives without a probe.

    Every round of this milestone added guards and a green suite, and the
    sweep still reported every probe caught — because the probes were counted
    against each other and never against the code. Twenty-three of the guards
    one fix delta added had nothing anchored on them, so deleting any of them
    changed no test's answer.

    This is a ratchet in both directions, and the second one is the one that
    does the work. A guard added with no probe is a new UNPROBED identity, and
    the list never grows. A guard that GAINS a probe has to leave the list in
    the same commit, which is what keeps the count falling rather than a
    line-item that once counted 57 and now means nothing.

    Identity is `(file, function, kind, ordinal within function)`: the same
    guards keyed by line went red on forty entries the change never touched.
    """
    table = _guard_table()
    print(_printed(table))

    assert len(table) > 60, "the walk found almost no guards — it is broken"
    probed = [row for row in table if row[6]]
    assert probed, "no probe anchored on any guard — the corpus was not read"

    computed = {row[:4] for row in table if not row[6]}
    frozen = set(_UNPROBED)
    unpinned = sorted(computed - frozen)
    retired = sorted(frozen - computed)
    still_guards = {row[:4] for row in table}
    assert not unpinned and not retired, (
        f"guards with no probe that _UNPROBED does not list: {unpinned}\n"
        "each needs a probe in tools/mutation_probes.json, or an entry here "
        "saying what it refuses or returns.\n"
        f"listed here but no longer unprobed: "
        f"{[row for row in retired if row in still_guards]}\n"
        f"listed here but no longer a guard at all: "
        f"{[row for row in retired if row not in still_guards]}\n"
        "re-freeze in the commit that changed them:\n" + _refrozen(table)
    )


def test_every_probe_on_these_two_files_still_anchors() -> None:
    """A moved anchor is red here, not only in the sweep.

    `mutation_sweep.py` calls this ANCHOR and refuses to run the probe, but
    nothing in CI ran the sweep, so a probe whose `old` had drifted off the
    code it was written for cost nothing until someone ran it by hand. The
    pairing test above reads the same anchors: an anchor that no longer
    matches silently un-probes a guard, which is the failure this file exists
    to make loud.
    """
    probes = json.loads(
        (REPO / "tools" / "mutation_probes.json").read_text(encoding="utf-8")
    )["probes"]
    checked = 0
    for module in _CLOSURE_MODULES:
        text = (REPO / module).read_text(encoding="utf-8")
        for probe in probes:
            if probe["file"] != module:
                continue
            checked += 1
            wanted = probe.get("occurrences", 1)
            assert text.count(probe["old"]) == wanted, (
                f"{probe['name']}: its anchor appears "
                f"{text.count(probe['old'])} times in {module}, wanted "
                f"{wanted} — the sweep calls this ANCHOR and runs nothing"
            )
            assert probe["new"] != probe["old"], (
                f"{probe['name']}: old and new are the same text, so the "
                "probe mutates nothing"
            )
    assert checked == 98, checked
