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


def _resolve(name: str) -> set[Path]:
    """Files that importing `name` executes: the module itself plus every
    package `__init__.py` on the way to it.

    A dotted tail may be a module (`memkit.a.b` -> `a/b.py`), a subpackage
    (`a/b/__init__.py`), or an attribute of a module that is already counted.
    All three shapes appear in ordinary code and the first version of this
    helper resolved none of them, so a subpackage the hook imported would have
    slipped the 3.9 pin in silence — the failure mode this pin exists for.
    """
    parts = name.split(".")[1:]  # drop the leading `memkit`
    files = {PKG / "__init__.py"}
    for i in range(len(parts)):
        stem = PKG.joinpath(*parts[: i + 1])
        if i < len(parts) - 1:
            files.add(stem / "__init__.py")
        else:
            files |= {stem.with_suffix(".py"), stem / "__init__.py"}
    return {f for f in files if f.is_file()}


def _hook_import_closure() -> set[Path]:
    """Every file the recall hook can reach at import time, transitively."""
    seen = {HOOK}
    queue = [HOOK]
    while queue:
        for name in _imports(queue.pop()):
            resolved = _resolve(name)
            # An unresolvable first-party name is the case that must never
            # pass quietly: silently contributing nothing shrinks the closure,
            # and a SMALLER closure is what makes this assertion agree with an
            # include list that is missing a file.
            assert resolved, f"first-party import {name!r} resolves to no file"
            for module in resolved - seen:
                seen.add(module)
                queue.append(module)
    return seen


def test_the_39_config_covers_exactly_the_hooks_import_path() -> None:
    """The hook's floor is 3.9 because the harness runs it with whatever
    `python3` the PATH resolves to, and that config's `include` is a
    hand-written file list — so a module the hook grows an import of is
    unchecked at 3.9 until somebody remembers to add it. What that costs is
    invisible: a hook that raises on import is reported by the harness as
    nothing at all, which is also what a corpus with nothing to say looks like.

    The direction is the easy half to invert. `cli.py` imports the hook, and
    that does not put `cli.py` on the hook's import path — nothing puts it in
    front of the harness's interpreter.
    """
    config = json.loads((REPO / "pyrightconfig-hook39.json").read_text())
    listed = {(REPO / p).resolve() for p in config["include"]}
    assert listed == _hook_import_closure()


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
    # the way to it. An attribute tail contributes no file of its own.
    assert _resolve("memkit.cli") == {PKG / "__init__.py", PKG / "cli.py"}
    assert _resolve("memkit.memory_prompt_recall.SCHEMA") == {PKG / "__init__.py"}


def test_the_package_config_covers_new_files_without_being_edited() -> None:
    """The other half of the convention, and the reason it is only ever one
    file that needs a hand: this config includes whole directories, so anything
    added under them is checked with no edit at all."""
    config = json.loads((REPO / "pyrightconfig.json").read_text())
    assert {"src", "tests", "tools"} <= set(config["include"])
    assert (REPO / "src" / "memkit" / "cli.py").is_file()
