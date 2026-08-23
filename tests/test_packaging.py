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
import shutil
import subprocess
import tarfile
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
