"""What the built artifacts and the type-check configs have to carry.

Two things that are true of the repo rather than of any function in it, and
that go wrong silently: a licence obligation met by a build-backend default
nobody declared, and a second pyright config whose include list is a hand-kept
list of files.
"""

from __future__ import annotations

import ast
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
    """First-party module names `path` imports, absolute and relative."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.add(f"memkit.{node.module}" if node.module else "memkit")
            elif node.module:
                found.add(node.module)
    return {n for n in found if n == "memkit" or n.startswith("memkit.")}


def _hook_import_closure() -> set[Path]:
    """Every file the recall hook can reach at import time, transitively."""
    seen = {HOOK}
    queue = [HOOK]
    while queue:
        for name in _imports(queue.pop()):
            module = PKG / (name.split(".", 1)[1] + ".py" if "." in name else "__init__.py")
            if module.is_file() and module not in seen:
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


def test_the_package_config_covers_new_files_without_being_edited() -> None:
    """The other half of the convention, and the reason it is only ever one
    file that needs a hand: this config includes whole directories, so anything
    added under them is checked with no edit at all."""
    config = json.loads((REPO / "pyrightconfig.json").read_text())
    assert {"src", "tests", "tools"} <= set(config["include"])
    assert (REPO / "src" / "memkit" / "cli.py").is_file()
