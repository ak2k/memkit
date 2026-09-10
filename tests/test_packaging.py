"""What the built artifacts and the type-check configs have to carry.

Two things that are true of the repo rather than of any function in it, and
that go wrong silently: a licence obligation met by a build-backend default
nobody declared, and a second pyright config whose include list is a hand-kept
list of files.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
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
    # `XDG_CACHE_HOME` dropped for the same reason the floor case below drops
    # it: the build frontend keeps its own cache there, and a runner pointing
    # the variable somewhere it cannot write turns this into a build failure
    # that says nothing about the artifact. Dropped rather than redirected, so
    # the build still hits a warm cache.
    env = dict(os.environ)
    env.pop("XDG_CACHE_HOME", None)
    built = subprocess.run(
        ["uv", "build", "--out-dir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
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


def test_the_38_config_names_the_one_file_and_the_floor_it_checks_it_at() -> None:
    """A CI step whose whole value lives in a config file needs one assertion
    pinning that file, or it passes checking something else.

    Nothing here read `pyrightconfig-shape38.json`, so a `pythonVersion`
    edited to 3.12 or an `include` emptied left a green third pyright step
    that duplicated the first: the file it exists for is also covered by
    `pyrightconfig.json` at 3.12, so nothing else would turn red. The two
    configs above are pinned this way already; this one arrived without it.
    """
    config = json.loads((REPO / "pyrightconfig-shape38.json").read_text())
    assert config["include"] == ["tools/harness_shape.py"]
    assert config["pythonVersion"] == SHAPE_FLOOR_VERSION
    assert (REPO / "tools" / "harness_shape.py").is_file()


FLOOR_REQUIRED_ENV = "MEMKIT_FLOOR_REQUIRED"


# The two versions below this repository's own floor that something here has
# to run on. The hook is dispatched by whatever `python3` the harness resolves,
# which on a stock macOS is 3.9.6; `tools/harness_shape.py` is piped over ssh
# into somebody else's host, and the first one it went to ran 3.8.18.
FLOOR_VERSION = "3.9"
SHAPE_FLOOR_VERSION = "3.8"


def _floor_interpreter(version: str = FLOOR_VERSION) -> str | None:
    """A real interpreter of `version`, or None.

    `uv python find` first, because `uv python install 3.9` provisions one in
    well under a second and that is what makes this affordable as a gate; a
    `python3.9` on PATH answers too, for a machine that has one already.

    AND THE CHILD IS ASKED WHAT IT IS. A name is not a version: a pyenv, asdf,
    conda or Nix shim called `python3.8` is ordinary on a developer's machine
    and answers this gate with whatever it forwards to, so the whole floor
    case goes green having executed 3.12. An interpreter that will not say, or
    says something else, is no interpreter of this version.
    """
    for probe in (["uv", "python", "find", version],):
        try:
            out = subprocess.run(probe, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return _of_version(out.stdout.strip(), version)
    named = shutil.which(f"python{version}")
    return None if named is None else _of_version(named, version)


def _of_version(interpreter: str, version: str) -> str | None:
    """`interpreter` if it really is `version`, and None otherwise."""
    try:
        out = subprocess.run(
            [interpreter, "-c", 'import sys;print("%d.%d" % sys.version_info[:2])'],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return interpreter if out.stdout.strip() == version else None


def _require_floor_interpreter(version: str = FLOOR_VERSION) -> str:
    """A real interpreter of `version`, or a verdict — never a quiet pass.

    One implementation for every floor case, because this switch is what
    `test_the_floor_gate_fails_rather_than_skips_when_it_is_required` watches,
    and a second copy of it is a copy nothing watches.
    """
    interpreter = _floor_interpreter(version)
    if interpreter is None:
        if os.environ.get(FLOOR_REQUIRED_ENV) == "1":
            raise AssertionError(
                f"{FLOOR_REQUIRED_ENV}=1 and no {version} interpreter was found — "
                f"`uv python install {version}` provisions one"
            )
        pytest.skip(
            f"no python{version} available; MEMKIT_FLOOR_REQUIRED=1 makes this fail"
        )
    assert interpreter is not None
    return interpreter


def test_of_version_believes_the_answer_and_not_the_name(tmp_path) -> None:
    """The guard against a lying `python3.8`, checked without needing a liar.

    Every floor case rests on this one comparison, and on a machine whose
    `python3.8` really is 3.8 the whole gate stays green with the comparison
    deleted — the failure it exists for is invisible exactly where it is run.
    So ask it here instead, of an interpreter that is present and is not the
    version asked for: the running one answers for itself, refuses to answer
    for its predecessor, and a path that cannot be executed at all is no
    interpreter of any version rather than an exception out of the gate.
    """
    running = f"{sys.version_info[0]}.{sys.version_info[1]}"
    mismatch = f"{sys.version_info[0]}.{sys.version_info[1] - 1}"
    assert _of_version(sys.executable, running) == sys.executable
    assert _of_version(sys.executable, mismatch) is None
    unrunnable = tmp_path / f"python{mismatch}"
    unrunnable.write_text("", encoding="utf-8")
    assert _of_version(str(unrunnable), mismatch) is None


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
    interpreter = _require_floor_interpreter()
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
    monkeypatch.setattr(sys.modules[__name__], "_floor_interpreter", lambda *_: None)
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


@pytest.mark.parametrize(
    "version", [FLOOR_VERSION, SHAPE_FLOOR_VERSION], ids=["py39", "py38"]
)
def test_harness_shape_runs_on_a_real_floor_interpreter(tmp_path, version) -> None:
    """The capture tool at both floors, both ways it is actually invoked.

    `tools/harness_shape.py` is the one file here that runs on machines this
    project has no other claim on: it is piped over ssh into whatever `python3`
    a colleague's host resolves, which on a stock macOS is 3.9.6 and on an
    older Linux is older still — 3.8.18 on the first host it went to, which is
    why the file's own floor is a version below this repository's. Nothing else
    would notice a 3.10 idiom in it — the suite runs it under 3.12, and the
    failure lands as a syntax error in somebody else's terminal.

    3.8 IS EXECUTED HERE AND NOWHERE ELSE. `pyrightconfig-shape38.json` catches
    the typing half; typeshed no longer carries the 3.8 guards, so a 3.9-only
    stdlib call is invisible to it and visible to this.

    BY PATH AND ON STDIN, because those are two different executions: `python3
    -` gives the module no `__file__` and an `argv[0]` of `-`, so a tool that
    reads either one works from the repository and dies over the pipe.

    AND THE SAME BYTES AS 3.12, because a shape captured over ssh is compared
    against shapes captured here: an interpreter that runs the tool and answers
    a different document is a fixture nobody can reproduce.

    AND THE DESTINATION HALF, which no other gate here executes at all: `--out`
    is `os.open`, `os.mkdir` and `os.stat` called with `dir_fd=`, and which of
    those the floor's `os.supports_dir_fd` actually holds is stated by a
    comment in the tool rather than by a run. ONE ROUTE IS EXEMPT and named so
    rather than silently absent — `--raw` redirected at a file, which reaches
    `_stdout_destination`'s `F_GETPATH` and needs a redirect this harness would
    have to build around the interpreter it is testing.
    """
    interpreter = _require_floor_interpreter(version)
    memory = tmp_path / "config" / "projects" / "-h-u-git-app" / "memory"
    memory.mkdir(parents=True)
    (memory / "one.md").write_text(
        "---\nname: one\ndescription: a fact\n---\n\nbody\n", encoding="utf-8"
    )
    tool = REPO / "tools" / "harness_shape.py"
    args = ["--config-dir", str(tmp_path / "config")]
    by_path = subprocess.run(
        [interpreter, str(tool), *args],
        capture_output=True, text=True, timeout=600,
    )
    assert by_path.returncode == 0, by_path.stdout + by_path.stderr
    on_stdin = subprocess.run(
        [interpreter, "-", *args],
        input=tool.read_text(encoding="utf-8"),
        capture_output=True, text=True, timeout=600,
    )
    assert on_stdin.returncode == 0, on_stdin.stdout + on_stdin.stderr
    assert by_path.stdout == on_stdin.stdout, "two invocations, two answers"
    here = subprocess.run(
        [sys.executable, str(tool), *args],
        capture_output=True, text=True, timeout=600,
    )
    assert here.returncode == 0, here.stdout + here.stderr
    assert by_path.stdout == here.stdout, "two interpreters, two answers"
    shape = json.loads(by_path.stdout)
    assert shape["anonymised"] is True
    assert len(shape["memory_dirs"]) == 1, shape

    landing = tmp_path / "shape.json"
    wrote = subprocess.run(
        [interpreter, str(tool), *args, "--out", str(landing)],
        capture_output=True, text=True, timeout=600,
    )
    assert wrote.returncode == 0, wrote.stdout + wrote.stderr
    assert json.loads(landing.read_text(encoding="utf-8")) == shape
    # The exit code alone is not the assertion, on any of these: an exit 2 for
    # the wrong reason is the failure this whole file is written against, and
    # every refusal below has a sibling that answers 2 for something else.
    taken = subprocess.run(
        [interpreter, str(tool), *args, "--out", str(landing)],
        capture_output=True, text=True, timeout=600,
    )
    assert taken.returncode == 2, taken.stdout + taken.stderr
    assert "already at this name" in taken.stderr.splitlines()[0], taken.stderr
    below = tmp_path / "made" / "here" / "shape.json"
    made = subprocess.run(
        [interpreter, str(tool), *args, "--out", str(below)],
        capture_output=True, text=True, timeout=600,
    )
    assert made.returncode == 0, made.stdout + made.stderr
    assert json.loads(below.read_text(encoding="utf-8")) == shape
    checkout = tmp_path / "co"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(checkout), check=True, timeout=60)
    leak = checkout / "raw.json"
    refused = subprocess.run(
        [interpreter, str(tool), *args, "--raw", "--out", str(leak)],
        capture_output=True, text=True, timeout=600,
    )
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert "inside a git worktree" in refused.stderr.splitlines()[0], refused.stderr
    assert not leak.exists(), "refused and written anyway"


def _floor_step_selectors() -> list:
    """Every `-k` expression CI runs with the floor gate switched on."""
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text(
        encoding="utf-8"
    )
    steps = re.split(r"^      - (?=name:)", workflow, flags=re.MULTILINE)
    required = [step for step in steps if f'{FLOOR_REQUIRED_ENV}: "1"' in step]
    assert required, FLOOR_REQUIRED_ENV
    selectors = []
    for step in required:
        found = re.findall(r'-k "((?:[^"\\]|\\\s)*)"', step)
        assert found, step
        selectors.extend(" ".join(item.split()) for item in found)
    return selectors


def _names_this_file_defines() -> set:
    """The test names and parametrize ids a `-k` token could be naming."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.add(node.name)
        if isinstance(node, ast.keyword) and node.arg == "ids":
            names.update(
                item.value
                for item in getattr(node.value, "elts", [])
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return names


def test_the_floor_steps_select_the_tests_they_name() -> None:
    """A `-k` is a name written down twice, and only one copy is checked by
    anything.

    The 3.9 step used to select `a or (b and py39)`: pytest exits 5 for a
    selection that matches nothing but 0 whenever any half of an `or` still
    does, so renaming the capture-tool case would have taken it out of CI and
    left the step green on the hook case alone — the floor that is executed
    nowhere else, silently not executed. One clause per invocation is what
    turns a name that stopped matching red, and this is what says the names
    still match.
    """
    known = _names_this_file_defines()
    assert "test_harness_shape_runs_on_a_real_floor_interpreter" in known
    for selector in _floor_step_selectors():
        depth = 0
        for token in re.findall(r"[()]|[^\s()]+", selector):
            if token == "(":
                depth += 1
                continue
            if token == ")":
                depth -= 1
                continue
            if token == "or":
                assert depth > 0, (
                    f"{selector!r}: a top-level `or` keeps the step green on "
                    f"one clause alone; give each clause its own invocation"
                )
                continue
            if token in ("and", "not"):
                continue
            assert any(token in name for name in known), (selector, token)


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


# --- a judgment made and then dropped, over every path-judging module --------


# Every name here JUDGES — a path, a descriptor, a repository, the filesystem
# under one — and its answer IS the value. A call made for its side effect has
# asked the question and then acted on something else: the path re-derived
# from a string, the directory reopened by name, the descriptor taken twice.
# Each module was carrying its own copy of this lint over its own table, so a
# module that grew a fifth table grew no lint; one test over `(module, table)`
# is the whole class in one place.
#
# Each entry is `(module, guards, uncalled, floor)`. `uncalled` names the
# guards deliberately at zero call sites, so the non-vacuity half below can
# still demand that every OTHER name is reached. `floor` is the module's call
# count measured when the entry was written: the lint cannot silently empty,
# and it cannot silently shrink either.
_PATH_GUARD_MODULES = (
    (
        "src/memkit/cli_init.py",
        (
            (
                "_refuse_escape",
                "the ONE resolution it judged; a dropped call judged one path"
                " and wrote another",
            ),
        ),
        (),
        2,
    ),
    (
        "src/memkit/cli_doctor.py",
        (
            ("_within", "containment over two resolved paths, one spelling of one rule"),
            ("_pruned", "whether the indexing walk descends to it at all"),
            ("_folds", "whether the filesystem holding it reads two cases as one"),
            ("_present", "directory, not a directory, or this run could not look"),
            ("_placed", "whether anything retrieves what lands there, and how"),
            ("_store_relation", "which store the directory overlaps, and how"),
        ),
        (),
        13,
    ),
    (
        "src/memkit/memory_prompt_recall.py",
        (
            ("_named_dir_flags", "the --dir door; dropping (scan, mark) drops both"),
            ("_repo_root", "which checkout, and which spelling of it; None is an answer"),
            ("_project_store", "(store or None, reason) — the refusal sentence IS the value"),
            ("_store_path", "the containment decision, and the only one"),
            ("_store_live_dir", "the one predicate for 'is this store searchable'"),
            ("_inside", "containment over two resolved paths, one spelling of one rule"),
            ("_cwd_in_root", "whether the session cwd is inside a gated root"),
            ("_regular_fd", "the judged descriptor and its stat ARE the value"),
            ("_lex_root", "the root a path was filed under"),
            ("_lex_read_only", "the scan half, read through an untyped handle"),
            ("_lex_marked", "the mark half; the pointer's only provenance claim"),
            ("path_refusal", "why a path may not be acted on, or the empty string"),
            ("_trust_gate", "what an uninitialized install refuses with, or None"),
        ),
        # The module's public path judgment, in the set at zero call sites so
        # that the first call site someone adds arrives under the rule.
        ("path_refusal",),
        26,
    ),
    (
        "tools/harness_shape.py",
        (
            ("_regular_fd", "the descriptor IS the answer: an fstat judged what was opened"),
            ("_resolves_inside", "whether a link reaches bytes this capture is reading"),
            ("_outside", "whether a row target escapes the directory being walked"),
            ("_row_present", "whether the row names something in the directory"),
            ("_open_dir", "a descriptor for the level, refusing a link or a file at it"),
            ("_landing_dir", "where a write actually lands, which is not its name"),
            ("_occupant", "which way a destination name is already taken"),
            ("_git_says_worktree", "git's own answer about a directory"),
            ("_worktree_above", "whether a checkout stands above the descriptor"),
            ("_inside_worktree", "the same question asked about a directory name"),
            ("_stdout_is_a_file", "whether fd 1 is a file rather than a pipe"),
            ("_stdout_destination", "the path fd 1 writes to, or no answer"),
            ("_is_own_config_dir", "whether the tree named is this machine's own"),
            ("create", "_Landing: the descriptor of the destination it just made"),
            ("discard", "_Landing: whether what it created is gone from the name"),
            ("inside_worktree", "_Landing: the checkout question asked of the descriptor"),
        ),
        (),
        27,
    ),
)


def _defined_functions(tree: ast.AST) -> set:
    """Every `def` in one module, at module level or on a class.

    Not module level alone: three of the shape tool's guards are `_Landing`
    methods, and a table that could not name them would have to leave the
    class's own judgments out of the lint.
    """
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _discarded(target: ast.expr) -> bool:
    """Whether an assignment target means "the answer is not read".

    `_ = guard(...)` and `*_, = guard(...)` are spellings of standing alone
    with a comment's worth of ceremony and nothing else.
    """
    if isinstance(target, ast.Starred):
        return _discarded(target.value)
    if isinstance(target, ast.Name):
        return target.id == "_"
    if isinstance(target, (ast.Tuple, ast.List)):
        return bool(target.elts) and all(
            _discarded(element) for element in target.elts
        )
    return False


@pytest.mark.parametrize(
    "entry", _PATH_GUARD_MODULES, ids=[entry[0] for entry in _PATH_GUARD_MODULES]
)
def test_every_guard_call_uses_the_value_it_returns(entry) -> None:
    """A judgment nobody reads is a judgment that decided nothing.

    Each name in the module's table answers a question and the caller acts on
    the answer; a call standing alone as a statement has asked it and then
    gone on to act on something else. Every instance of that class in these
    four modules was a real defect, so the cheap catch for the next one is
    that no call to any of them may be an expression statement or be assigned
    to a name that means "not read".

    Two non-vacuity halves, because a lint that silently empties is worse than
    none: every name must still resolve to a `def` in its module, so a rename
    fails here rather than quietly clearing the set, and the module's call
    sites must still meet the floor measured beside it.

    The modules are read as TEXT and never imported. The hook has to be —
    importing it would run its module body in this process — and the other
    three follow the same rule so that the walk is the same walk.
    """
    module, guards, uncalled, floor = entry
    source = (REPO / module).read_text(encoding="utf-8")
    tree = ast.parse(source, module)
    named = frozenset(name for name, _reason in guards)

    missing = sorted(named - _defined_functions(tree))
    assert not missing, (
        f"{module}: {missing} is in the guard table and is not defined there — "
        "a rename that empties this lint is what this half catches"
    )

    def called(node) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Name) and node.func.id in named:
            return node.func.id
        if isinstance(node.func, ast.Attribute) and node.func.attr in named:
            return node.func.attr
        return None

    calls = [answer for answer in map(called, ast.walk(tree)) if answer is not None]
    assert len(calls) >= floor, f"{module}: {len(calls)} call sites, floor {floor}"
    unreached = sorted(named - set(calls) - set(uncalled))
    assert not unreached, (
        f"{module}: {unreached} is named and never called — the lint would "
        "pass over a module that judges nothing"
    )

    def throws_away(node) -> bool:
        """Whether the statement drops whatever its value expression answers."""
        if isinstance(node, ast.Expr):
            return True
        if isinstance(node, ast.Assign):
            return all(_discarded(target) for target in node.targets)
        if isinstance(node, ast.AnnAssign):
            return _discarded(node.target)
        return False

    dropped = []
    for node in ast.walk(tree):
        # The isinstance is the checker's, not the walk's: `throws_away`
        # answers about these three and nothing else, and a bare `ast.AST`
        # carries neither `.value` nor `.lineno`.
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        if not throws_away(node):
            continue
        answer = called(node.value)
        if answer is not None:
            dropped.append(f"{module}:{node.lineno}  {answer}")
    assert dropped == [], dropped


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

# Identity, not position: `(file, function, kind, digest of the guard's own
# source)`. Line numbers are informative only — one edit above a function
# shifts every guard below it — and so is the ordinal, which an insert ABOVE
# an existing guard in the same function renumbers all the way down, taking
# forty entries the change never touched red with it. The digest moves when
# and only when that guard's own text moves, which is the commit that owes it
# a probe or a fresh reason.
_UNPROBED: dict[tuple[str, str, str, str], str] = {
    ("cli_doctor.py", "_managed_dir", "if->exit", "2f9cea9c03bd"): (
        "on darwin the managed settings directory is the Library one"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", "49129f5bd440"): (
        "a settings file that is not there leaves the scope empty, not failed"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", "be52c1456571"): (
        "a settings file that cannot be opened is recorded FORBIDDEN"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", "9fb71c982f1d"): (
        "any other OS error on that file is recorded UNREADABLE"
    ),
    ("cli_doctor.py", "Settings.__init__", "except", "fbb7856eaf8e"): (
        "settings that do not parse are recorded UNPARSED"
    ),
    ("cli_doctor.py", "Settings.__init__", "if->exit", "935e50823400"): (
        "settings whose top level is not an object are UNPARSED, not empty"
    ),
    ("cli_doctor.py", "_session_cwd", "except", "7be5ed5a3324"): (
        "returns '' when the session's own directory will not resolve"
    ),
    ("cli_doctor.py", "Machine.config", "if->exit", "90414542b26a"): (
        "no resolved config path means there is no config to read"
    ),
    ("cli_doctor.py", "Machine.config", "except", "cbc63a02c4c4"): (
        "a ConfigError is kept as the config's error rather than raised"
    ),
    ("cli_doctor.py", "Machine.config", "except", "867f3b5a9eff"): (
        "anything else out of the loader is kept as text, so doctor survives it"
    ),
    ("cli_doctor.py", "_store_relation", "if->exit", "69d858ab3254"): (
        "the first store root holding the directory decides at versus over"
    ),
    ("cli_doctor.py", "_how_inside", "if->exit", "166cef1d0249"): (
        "a directory under a pruned name is 'pruned' before anything else"
    ),
    ("cli_doctor.py", "_placed", "if->exit", "c763ff8d20e0"): (
        "outside every store, nothing retrieves what lands there"
    ),
    ("cli_doctor.py", "_placed", "if->exit", "303991247d30"): (
        "a directory at or over a corpus root is refused, with what it costs"
    ),
    ("cli_doctor.py", "_odd_switch", "if->exit", "c2544feb2e26"): (
        "no scope, or a real bool, is not an odd value to remark on"
    ),
    ("cli_doctor.py", "_checkout_remedy", "if->exit", "3c3a1e15434a"): (
        "the checkout scope gets the remedy that names what changing it costs"
    ),
    ("cli_doctor.py", "_env_switch_note", "if->exit", "1004cb47f421"): (
        "no forced value means no note about the variable"
    ),
    ("cli_doctor.py", "_env_switch_note", "if->exit", "9e0b2ca707e3"): (
        "a forced-on value says no settings scope turns the feature off"
    ),
    ("cli_doctor.py", "_env_switch_remedy", "if->exit", "a34a62758f07"): (
        "a forced-on value's remedy is to unset the variable, not edit settings"
    ),
    ("cli_doctor.py", "_adopter_owns", "if->exit", "2357aa3e124a"): (
        "the named scope's own flag answers; no other scope stands in for it"
    ),
    ("cli_doctor.py", "_declared_below", "if->exit", "08d7fcc55715"): (
        "a name outside SCOPE_ORDER has nothing below it"
    ),
    ("cli_doctor.py", "_declared_below", "if->exit", "de75ac5d93fc"): (
        "a scope that is not on this machine is skipped, not counted"
    ),
    ("cli_doctor.py", "_default_memory_dir", "if->exit", "abb8fe235aee"): (
        "with no session directory the harness's own directory is underivable"
    ),
    ("cli_doctor.py", "_default_memory_dir", "except", "db9090f32353"): (
        "a key that will not compute is reported as unknown, never raised"
    ),
    ("cli_doctor.py", "_consolidation_recency", "if->exit", "5b67efc7d3cf"): (
        "no default directory means no recency to report"
    ),
    ("cli_doctor.py", "_left_behind", "if->exit", "835a1a226452"): (
        "nothing outside the configured directory is nothing left behind"
    ),
    ("cli_doctor.py", "_auto_memory_rows", "if->exit", "e17afb6c013a"): (
        "an off switch this checkout carries gets the checkout remedy"
    ),
    ("cli_doctor.py", "_auto_memory_rows", "if->exit", "01351b3db1ef"): (
        "scopes contradicting the off switch are disclosed, not passed over"
    ),
    ("harness_memory.py", "project_key", "if->exit", "ba392320016e"): (
        "a key over the cap is refused: the harness's suffix is unmeasured"
    ),
    ("harness_memory.py", "_project_path", "except", "eb2dc065a69c"): (
        "a cwd that will not resolve is used as it was given"
    ),
    ("harness_memory.py", "_project_path", "if->exit", "8d8e208844ba"): (
        "no repository root leaves the resolved path as the project path"
    ),
    ("harness_memory.py", "_project_path", "if->exit", "563054543608"): (
        "no common ancestor leaves the resolved path as the project path"
    ),
    ("harness_memory.py", "_project_path", "if->exit", "e35b6f5cf918"): (
        "a submodule's git directory makes its worktree root the project"
    ),
    ("harness_memory.py", "_project_path", "except", "8c1c7e4bc59c"): (
        "an unknown root, or a failing resolve, leaves the resolved path"
    ),
    ("harness_memory.py", "switch", "if->exit", "a146d101d634"): (
        "the first scope in the harness's order that declares the key answers"
    ),
    ("harness_memory.py", "harness_dir", "if->exit", "ed6c126d09cf"): (
        "a value that is not a non-empty string names no directory"
    ),
    ("harness_memory.py", "env_switch", "if->exit", "0798dbaa09bb"): (
        "an unset or empty variable is no answer, not an off one"
    ),
    ("harness_memory.py", "env_switch", "if->exit", "7e109f8354c0"): (
        "a spelling the harness reads as off means the feature does not run"
    ),
    ("harness_memory.py", "configured_dir", "if->exit", "db68486ef19a"): (
        "no scope declares the key, so no directory is configured"
    ),
    ("harness_memory.py", "configured_dir", "if->exit", "537b608d7fc3"): (
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


def _digest(text: str, lineno: int, end_lineno: int) -> str:
    """A guard's own source as twelve hex characters.

    The LEFT MARGIN IS STRIPPED, so a guard moved into or out of a `with`
    block — reindented, deciding exactly what it decided before — keeps its
    identity. Its own text is the whole of what is hashed: a guard is the same
    guard while what it tests and what it does about it are the same.
    """
    lines = text.splitlines(keepends=True)[lineno - 1 : end_lineno]
    body = textwrap.dedent("".join(lines))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def _identity(row) -> tuple:
    """`(file, function, kind, digest)` — what `_UNPROBED` is keyed on."""
    _name, qualified, kind, _ordinal, digest = row[:5]
    return (row[0], qualified, kind, digest)


def _guard_table() -> list:
    """`(file, function, kind, ordinal, digest, lineno, end_lineno, probes)`.

    The ordinal stays in the row because the width test below finds a guard by
    position after mutating the source; it is not what the freeze is keyed on.
    """
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
                _digest(text[module], lineno, end_lineno),
                lineno,
                end_lineno,
                covering,
            )
        )
    table.sort(key=lambda row: (row[0], row[5]))
    return table


# A guard longer than this is one no reader holds in their head, and the
# pairing below accepts a probe that lands anywhere inside it. Every guard in
# this closure but the two the auto-memory row's own branches open fits inside
# a quarter of it.
_WIDE_GUARD = 40


def _empties(
    text: str, module: str, qualified: str, kind: str, ordinal: int, probe: dict
) -> bool:
    """True when this probe leaves that guard's condition a false constant.

    Nothing reaches the body then, so one verdict answers for every line of
    it — which is exactly what a probe mutating one clause a hundred lines in
    cannot do.
    """
    if kind != "if->exit" or text.count(probe["old"]) != 1:
        return False
    try:
        tree = ast.parse(text.replace(probe["old"], probe["new"], 1))
    except SyntaxError:
        return False
    functions = _qualified_functions(tree, module)
    if qualified not in functions:
        return False
    blocked = {
        id(node)
        for name, (_module, node) in functions.items()
        if name != qualified
    }
    found = sorted(
        (
            guard
            for found_kind, guard in _guards_owned_by(
                functions[qualified][1], blocked
            )
            if found_kind == kind
        ),
        key=lambda guard: (guard.lineno, guard.end_lineno),
    )
    if len(found) < ordinal:
        return False
    condition = found[ordinal - 1].test
    return isinstance(condition, ast.Constant) and not condition.value


def _printed(table: list) -> str:
    lines = []
    for name, qualified, kind, _ordinal, digest, lineno, end_lineno, covering in table:
        lines.append(
            f"{name}\t{qualified}\t{kind}\t{digest}\t{lineno}-{end_lineno}\t"
            + (",".join(covering) if covering else "UNPROBED")
        )
    return "\n".join(lines)


def _refrozen(table: list) -> str:
    """`_UNPROBED` as it would have to read for this tree — paste-ready."""
    lines = ["_UNPROBED = {"]
    for row in table:
        if row[7]:
            continue
        identity = _identity(row)
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

    Identity is `(file, function, kind, digest of the guard's own source)`:
    keyed by line, and then by ordinal within the function, the same guards
    went red on forty entries the change never touched — a line moves when
    anything above it moves, and an ordinal moves when anything above it in
    the same function is INSERTED. A digest moves only with the guard.
    """
    table = _guard_table()
    print(_printed(table))

    assert len(table) > 60, "the walk found almost no guards — it is broken"
    probed = [row for row in table if row[7]]
    assert probed, "no probe anchored on any guard — the corpus was not read"

    computed = {_identity(row) for row in table if not row[7]}
    frozen = set(_UNPROBED)
    unpinned = sorted(computed - frozen)
    retired = sorted(frozen - computed)
    still_guards = {_identity(row) for row in table}
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


def test_a_guard_too_wide_for_one_probe_has_one_that_empties_it() -> None:
    """A mutation a hundred lines inside a body is not coverage of the body.

    The pairing above asks only that some probe land inside the guard, which
    is the whole answer for a guard of five lines and almost none of it for
    the two the auto-memory row's branches open: four probes sat inside the
    off-switch branch, and the branch could have gone whole with three of them
    still green.

    A probe that leaves the condition a false constant answers for every line
    at once, because nothing reaches any of them. Splitting the body closes
    this the other way, and either is the fix — the budget is a reading limit
    rather than a shape.
    """
    table = _guard_table()
    assert len(table) > 60, "the walk found almost no guards — it is broken"
    probes = json.loads(
        (REPO / "tools" / "mutation_probes.json").read_text(encoding="utf-8")
    )["probes"]
    by_name = {probe["name"]: probe for probe in probes}

    unspanned = []
    for module in _CLOSURE_MODULES:
        text = (REPO / module).read_text(encoding="utf-8")
        spans = _probe_spans(text, probes, module)
        for name, qualified, kind, ordinal, _digest, lineno, end_lineno, _cov in table:
            if name != Path(module).name:
                continue
            width = end_lineno - lineno + 1
            if width <= _WIDE_GUARD:
                continue
            if not any(
                _empties(text, module, qualified, kind, ordinal, by_name[probe])
                for probe, first, last in spans
                if first <= lineno <= last
            ):
                unspanned.append((name, qualified, kind, ordinal, width))
    assert not unspanned, (
        f"guards over {_WIDE_GUARD} lines that no probe empties: {unspanned}\n"
        "write one that leaves the condition false — the corpus spells that "
        "`if False:` — or split the body so no guard runs this wide."
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
    assert checked == 114, checked
