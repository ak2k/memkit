"""The plugin surface: manifests, registration, and the three wrappers.

What every case here has in common is that the thing it pins lives in two
files that no compiler, importer or type checker connects. A manifest key and
the environment variable a shell script reads. A `timeout` in a JSON
registration and a constant in the hook. A python floor in the checker and the
probe that avoids it. Each pair is one edit away from disagreeing, and every
disagreement is silent at runtime — a hook that no longer receives its config
still exits 0 and still prints nothing, which is also what a corpus with
nothing to say looks like.

The wrappers are driven as real processes against a SHIM interpreter: a script
on PATH called `python3` that records the environment it was handed instead of
running anything. That is the only way to see what the wrapper decided, since
what it decides is what it exports into the process it replaces itself with.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from memkit import memory_prompt_recall as hook

REPO = Path(__file__).resolve().parent.parent
PLUGIN_MANIFEST = REPO / ".claude-plugin" / "plugin.json"
MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
HOOKS_JSON = REPO / "hooks" / "hooks.json"
BIN = REPO / "bin"
COMMON_SH = BIN / "lib" / "common.sh"

# The payload an adopter receives. A plugin installed from a github source is a
# clone, so a file that is not tracked is a file that is not there — for
# everyone except the person who wrote it.
PAYLOAD = [
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
    "hooks/hooks.json",
    "bin/memkit",
    "bin/memkit-hook",
    "bin/memkit-recall",
    "bin/lib/common.sh",
    "src/memkit/memory_prompt_recall.py",
    "src/memkit/common-words.txt",
    "src/memkit/cli.py",
    "src/memkit/__init__.py",
    # The checker `bin/memkit` routes to when a local python meets the 3.12
    # floor: `MEMKIT_CHECKER_CMD` is `<python> -m memkit.memory_integrity`, run
    # against THIS tree, so leaving it out of the payload made that route name
    # a module the adopter does not have. Safe to add — its only first-party
    # import is `memkit.memory_prompt_recall`, already here.
    "src/memkit/memory_integrity.py",
]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _needs_checkout() -> None:
    """Skip only where a checkout genuinely cannot exist, and FAIL elsewhere.

    These cases read the index and the commit graph, so the packaged nix leg —
    which builds from a store copy with no `.git` — cannot run them. That leg
    sets `MEMKIT_NO_CHECKOUT`, and everywhere else a missing checkout is a
    broken environment rather than a licence to pass: the plain-python job is
    where these are the gate, and a skip there reports green under the same
    check name as a run.
    """
    if os.environ.get("MEMKIT_NO_CHECKOUT") == "1":
        pytest.skip("packaged build — no .git in the store copy, by construction")
    assert (REPO / ".git").exists(), (
        "no .git here, and this context did not declare itself packaged. These "
        "cases read the index and the commit graph; skipping them silently is "
        "what makes a green check name mean nothing."
    )
    assert shutil.which("git"), "git is not on PATH"


# --- the manifests ------------------------------------------------------------


def test_the_option_key_mangles_to_the_variable_the_wrapper_reads() -> None:
    """The one pin that makes config delivery work at all.

    The harness builds the variable by uppercasing the option key with
    non-alphanumerics replaced by `_` (read out of the 2.1.238 bundle, and
    confirmed end to end: `memkitConfig` arrived as
    `CLAUDE_PLUGIN_OPTION_MEMKITCONFIG`). Nothing connects the key in the
    manifest to the name in the shell script, and a rename on either side
    leaves a plugin that installs, loads, reports healthy, and serves nothing.

    The mangling is applied here rather than asserted as a literal, so a key
    with an underscore or a digit is still checked against the real rule.
    """
    options = _json(PLUGIN_MANIFEST)["userConfig"]
    assert list(options) == ["memkitConfig"], options
    key = next(iter(options))
    expected = "CLAUDE_PLUGIN_OPTION_" + re.sub(r"[^A-Za-z0-9_]", "_", key).upper()
    assert expected in COMMON_SH.read_text(encoding="utf-8"), expected


def test_the_option_key_is_one_the_harness_will_accept() -> None:
    """Keys are identifier-shaped; a hyphen or a dot is not a warning.

    Measured on 2.1.238: `userConfig.some.dotted-key: Invalid key in record`,
    and a plugin carrying one installs and then reports `failed to load` — a
    state visible only in `claude plugin list`, not at install time.
    """
    for key in _json(PLUGIN_MANIFEST)["userConfig"]:
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key), key


def test_the_option_is_required_and_says_what_it_is_for() -> None:
    """`required` is what makes a forgotten `--config` loud.

    It does not block the install (measured: a warning naming the option and
    the two ways to set it), which is the right severity — the plugin is inert
    without it, not broken. A silently optional one would leave an adopter with
    a plugin that installed cleanly and will never say anything.

    A `default` is declared for the interactive flow, and the wrapper must not
    depend on it: a declared default is NOT exported to hook processes when the
    option is unset (measured), so an install that skipped `--config` reaches
    the hook with nothing on either rung. That state is inert, which is why the
    warning is the right severity — and it is why the default is a suggestion
    to the person installing rather than an input to the wrapper.
    """
    option = _json(PLUGIN_MANIFEST)["userConfig"]["memkitConfig"]
    assert option["required"] is True
    assert option["type"] == "string"
    for field in ("title", "description"):
        assert option[field].strip(), field
    # The description is rendered by the harness during `/plugin install`, so
    # it is the first screen a cold adopter reads — and it named
    # `/memkit:init`, which this payload ships no `commands/` directory for and
    # `cli.py` lists in `_PENDING`. The adopter runs it, gets nothing, has no
    # config, and the plugin stays silently inert.
    #
    # So any command it names in the present tense must exist.
    from memkit.cli import _HANDLERS, _PENDING

    described = option["description"]
    named = {n for n in (*_PENDING, *_HANDLERS) if f"/memkit:{n}" in described}
    assert named <= set(_HANDLERS), sorted(named - set(_HANDLERS))
    assert "manual in this build" in described, described


def test_the_marketplace_entry_pins_a_commit_rather_than_a_branch() -> None:
    """An unpinned same-repo source means every commit to main becomes hook
    code in an adopter's next session, on a surface that runs before every
    prompt. The schema accepts `ref` and `sha`, and `sha` wins.

    The sha is checked for SHAPE here and for EXISTENCE below, because a
    plausible-looking placeholder is the failure this is guarding: a 40-hex
    string that names no commit fails at `git clone`, on the adopter's machine,
    after they have accepted a trust dialog.
    """
    entry = _json(MARKETPLACE)["plugins"][0]
    source = entry["source"]
    assert source["source"] == "github"
    assert source["repo"] == "ak2k/memkit"
    assert re.fullmatch(r"[0-9a-f]{40}", source["sha"]), source
    assert "ref" not in source, "a ref beside a sha is dead config — sha wins"


# The paragraph that tells an adopter the marketplace route is not live yet.
# It is load-bearing rather than decorative: at a pin whose commit carries no
# payload the install SUCCEEDS and registers nothing, so nothing an adopter is
# told to run distinguishes it from a correct install waiting for a config.
NOT_YET_INSTALLABLE = "**Not yet installable from this marketplace.**"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60
    )


def test_the_pinned_sha_is_a_commit_in_this_history() -> None:
    _needs_checkout()
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    assert _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, (
        f"{sha} is not a commit in this repository"
    )
    # Existence is not enough: a sha from an unrelated branch, or from a fork,
    # satisfies `cat-file` and is not what an adopter would be served.
    assert _git("merge-base", "--is-ancestor", sha, "HEAD").returncode == 0, (
        f"{sha} is not an ancestor of HEAD"
    )


def test_the_readme_and_the_pinned_payload_say_the_same_thing() -> None:
    """Both directions, because both have happened in one repo or another: a
    pin moved without the README edit, and a README edit without the pin.

    Read out of the object store rather than by checking anything out —
    `git cat-file -e <sha>:<path>` per payload entry is the same question the
    harness's clone answers, and it costs no worktree.

    Green today by naming the state we are actually in, and the release commit
    that moves the pin is the one whose own test forces the paragraph out.
    """
    _needs_checkout()
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    at_sha = {
        path: _git("cat-file", "-e", f"{sha}:{path}").returncode == 0
        for path in PAYLOAD
    }
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    if all(at_sha.values()):
        assert NOT_YET_INSTALLABLE not in readme, (
            "the pin now carries the whole payload — the README still says it "
            "does not"
        )
    else:
        assert NOT_YET_INSTALLABLE in readme, (
            "the pin carries no plugin payload and the README no longer says "
            f"so: {sorted(p for p, ok in at_sha.items() if not ok)}"
        )


def test_the_manifest_and_the_marketplace_entry_agree_on_the_version() -> None:
    """`claude plugin tag` refuses to tag a release when they disagree, which
    is late: by then the version in the entry is what adopters resolve."""
    entry = _json(MARKETPLACE)["plugins"][0]
    manifest = _json(PLUGIN_MANIFEST)
    assert entry["name"] == manifest["name"] == "memkit"
    assert entry["version"] == manifest["version"]


def test_the_manifests_carry_the_metadata_an_adopter_is_shown() -> None:
    """Name, description, version, author — what `/plugin` renders in a list
    and what a directory submission is judged on. Absent metadata installs
    perfectly and reads as an unattributed blob.

    `--strict` demands these too, but only through the invocation that points
    at the plugin manifest itself: pointed at the repo root the validator
    checks the MARKETPLACE, and raises schema errors from the plugin manifests
    it lists while passing over their metadata warnings. Measured on 2.1.238 by
    deleting `author` and watching the single-invocation CI step stay green.
    CI now runs both; this fails in the suite rather than three minutes into a
    workflow.
    """
    manifest = _json(PLUGIN_MANIFEST)
    for field in ("name", "description", "version", "author"):
        assert manifest.get(field), field
    assert _json(MARKETPLACE).get("description")
    assert isinstance(_json(MARKETPLACE)["owner"], dict), "owner must be an object"


def test_ci_validates_both_manifests_and_not_only_the_marketplace() -> None:
    """The step is what makes the assertion above a release gate rather than a
    unit test, and it was checking half of what its name claimed."""
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text()
    assert "claude plugin validate . --strict" in workflow
    assert "claude plugin validate .claude-plugin/plugin.json --strict" in workflow


def test_ci_runs_the_rigs_harness_tier_as_a_gate_rather_than_a_courtesy() -> None:
    """The one required context in which the HARNESS produces the config
    option, rather than a test setting it.

    Two halves, and the second is what makes it a gate. The `python` job has to
    install the pinned binary before the suite — which it does, for the CLI
    tier — and it has to declare the tier required, or a job whose install step
    quietly stopped working reports green while every scenario that watches the
    harness skips. A skipped scenario and a passing one look identical in a
    check name.
    """
    from rig import REQUIRED_ENV

    # The `python:` job's own text, not the whole file: a whole-file grep
    # cannot tell which job carries a line, cannot tell a live line from a
    # commented one, and would pass with the declaration moved to the `nix`
    # job — where `tests/rig` is not even collected.
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text()
    jobs = re.split(r"^  (?=\w[\w-]*:$)", workflow, flags=re.MULTILINE)
    python_job = [j for j in jobs if j.startswith("python:")]
    assert len(python_job) == 1, [j.split(":", 1)[0] for j in jobs]
    body = python_job[0]
    live = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    assert f'{REQUIRED_ENV}: "1"' in live, REQUIRED_ENV
    assert "npm install -g @anthropic-ai/claude-code@" in live
    assert "-m pytest" in live
    # And nowhere else, or the declaration can drift into a job that never
    # runs the rig while this stays green.
    elsewhere = workflow.replace(body, "")
    assert REQUIRED_ENV not in elsewhere, "declared outside the job that runs it"


def test_every_payload_file_is_tracked() -> None:
    """A github install is a clone. An untracked wrapper works perfectly on the
    machine it was written on and is missing for every adopter — and the
    failure it produces there is the wrapper's own "payload is incomplete"
    refusal, i.e. a plugin that installs and never speaks again.
    """
    _needs_checkout()
    out = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *PAYLOAD],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr


def test_a_git_gated_case_fails_rather_than_skips_where_a_checkout_is_expected(
    monkeypatch, tmp_path
) -> None:
    """The skip is the thing being guarded, not the tool.

    Three of these cases skipped whenever `.git` was absent, so the packaged
    leg and a broken plain-python job produced the same green under the same
    check name. Only a context that DECLARES itself packaged may skip.
    """
    monkeypatch.delenv("MEMKIT_NO_CHECKOUT", raising=False)
    monkeypatch.setattr("test_plugin_surface.REPO", tmp_path, raising=False)
    # A skip here would SKIP THIS CASE rather than fail it, which is the whole
    # defect wearing the test's own clothes — so it is caught by name.
    try:
        _needs_checkout()
    except pytest.skip.Exception as skipped:
        raise AssertionError(f"skipped where it must fail: {skipped}") from None
    except AssertionError as failed:
        assert "did not declare itself packaged" in str(failed), failed
    else:
        raise AssertionError("a missing checkout was accepted")

    # `Skipped` derives from BaseException, so it has to be named: catching
    # `Exception` here let the assertion skip the case it is asserting about.
    monkeypatch.setenv("MEMKIT_NO_CHECKOUT", "1")
    with pytest.raises(pytest.skip.Exception, match="packaged build"):
        _needs_checkout()


def test_the_packaged_leg_is_the_only_context_that_declares_itself_packaged() -> None:
    """The marker's one producer, pinned where the skip is read.

    A test that skips on an environment variable is a test anybody can turn
    off; what makes it honest is that exactly one build sets it, and that build
    is the one whose source really has no checkout.
    """
    flake = (REPO / "flake.nix").read_text(encoding="utf-8")
    assert flake.count('MEMKIT_NO_CHECKOUT = "1"') == 1, flake.count(
        'MEMKIT_NO_CHECKOUT = "1"'
    )
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text()
    assert "MEMKIT_NO_CHECKOUT" not in workflow


def test_the_payload_carries_every_file_its_own_entry_points_import() -> None:
    """PAYLOAD is hand-kept, and the failure it produces is the wrapper's own
    "the plugin payload is incomplete" refusal — a plugin that installs and
    never speaks again.

    A SUBSET assertion, not equality: the closure of the two 3.9 entry points
    does not reach `memory_integrity.py`, which the dispatcher routes to only
    when a 3.12 interpreter is available, so equality would make that entry
    red. What this catches is the direction that matters — a module a shipped
    entry point imports that nobody remembered to list.
    """
    import sys

    sys.path.insert(0, str(REPO / "tests"))
    from test_packaging import _floor_39_closure

    reachable = {
        str(path.relative_to(REPO)) for path in _floor_39_closure()
    }
    listed = set(PAYLOAD)
    assert reachable <= listed, sorted(reachable - listed)


def test_the_payload_root_carries_no_config_of_its_own() -> None:
    """The other direction of the tracking assertion above, and the one the
    wrappers' admission rule rests on.

    That test asks whether each payload entry is tracked; nothing asked what
    ELSE the payload carries. A `memkit.json` at the root — committed, or
    merely sitting in the checkout of whoever runs the rig — used to be a
    config rung, and a config decides both which directories the every-prompt
    hook reads and which binary it exec's.

    Against the REAL repository root rather than the `root` fixture, which is
    built from PAYLOAD and can never contain one. Both states are checked: the
    index, because a clone delivers that, and the working tree, because the rig
    stages from it.
    """
    _needs_checkout()
    out = subprocess.run(
        ["git", "ls-files", "--", ":(top)memkit.json"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert out.stdout.strip() == "", out.stdout
    assert not (REPO / "memkit.json").exists(), "a config at the payload root"


def test_the_wrappers_are_executable_in_the_index() -> None:
    """Mode 100755 in git, not merely on this filesystem. A clone restores the
    executable bit from the index, and a wrapper checked in as 644 is a hook
    the harness cannot run at all."""
    _needs_checkout()
    out = subprocess.run(
        ["git", "ls-files", "-s", "bin/"], cwd=REPO,
        capture_output=True, text=True, timeout=60,
    )
    modes = {
        line.split("\t")[1]: line.split()[0] for line in out.stdout.splitlines()
    }
    for wrapper in ("bin/memkit", "bin/memkit-hook", "bin/memkit-recall"):
        assert modes[wrapper] == "100755", (wrapper, modes[wrapper])
    # And the sourced library is NOT executable: it sits in bin/lib precisely
    # so that nothing on the agent's PATH can invoke it.
    assert modes["bin/lib/common.sh"] == "100644", modes["bin/lib/common.sh"]


# --- the registration ---------------------------------------------------------


def _entries() -> list[tuple[str, dict]]:
    """(event, handler) for every hook this plugin registers."""
    out = []
    for event, groups in _json(HOOKS_JSON)["hooks"].items():
        for group in groups:
            for handler in group["hooks"]:
                out.append((event, handler))
    return out


def test_the_registration_passes_zero_arguments_on_every_entry() -> None:
    """The hook file is dual-mode: no arguments reads a payload off stdin, ANY
    argument is the search CLI — where argparse answers an unrecognised flag
    with exit 2. On UserPromptSubmit, exit 2 does not merely fail: the turn is
    blocked and the user gets their prompt handed back unanswered (measured on
    2.1.238). So a stray argument in this file costs every prompt in every
    session, on every machine that installed the plugin.

    Enumerating EVERY entry rather than the one that exists today is the point.
    A later unit adds PreToolUse to this same file, and a pin that names one
    event would go on passing while the new entry carried the defect.
    """
    assert _entries(), "hooks.json registers nothing — this pin would be vacuous"
    for event, handler in _entries():
        assert handler.get("args", []) == [], (event, handler)
        # DECLARED rather than omitted, because `args: []` is what selects the
        # exec form: without the key the command is a shell string, and the
        # hook file's dual mode turns any stray word into the search CLI.
        assert "args" in handler, (event, "declare args: [] rather than omitting it")
        assert handler["type"] == "command"
        # And no space in the command itself — under a shell that would be two
        # arguments, which is the same defect arriving through the path.
        assert " " not in handler["command"], handler["command"]


def test_every_registered_timeout_matches_the_constant_it_is_paired_with() -> None:
    """`timeout` is the harness's kill, and the hook's own budget is set
    beneath it so that an overrun leaves a record instead of being killed
    mid-write. Nothing connects the number in this JSON to the constant in the
    module; before the plugin existed, the consumer's own suite carried this
    assertion because the registration lived in its settings file.

    Scoped per event, because a later unit's PreToolUse entry gets its OWN
    constant pair — sharing the module's single budget would put an internal
    deadline above the harness's kill point.
    """
    expected = {"UserPromptSubmit": hook.HARNESS_TIMEOUT}
    for event, handler in _entries():
        assert event in expected, f"{event} has no declared constant pair"
        assert handler["timeout"] == expected[event], (event, handler["timeout"])
    # Both halves of the relation, so this file cannot be edited into agreement
    # with a budget that no longer sits beneath it.
    assert hook.BUDGET_SECONDS < hook.HARNESS_TIMEOUT


def test_the_registration_runs_the_wrapper_and_not_the_hook_directly() -> None:
    """A registration naming the hook file would work — and would inherit
    whatever MEMKIT_CONFIG the launching context carried, which is ambient
    configuration arriving by the back door."""
    for _event, handler in _entries():
        assert handler["command"] == "${CLAUDE_PLUGIN_ROOT}/bin/memkit-hook", handler


def test_the_checker_floor_in_the_probe_matches_the_checkers_own_guard() -> None:
    """KTD13's probe exists to avoid the checker's version guard, so a floor
    that drifted would route an interpreter straight into the refusal it was
    picked to avoid. The number lives in two files by necessity: one of them
    cannot import the other."""
    guard = (REPO / "src" / "memkit" / "memory_integrity.py").read_text()
    match = re.search(r"sys\.version_info < \((\d+), (\d+)\)", guard)
    assert match, "the checker's version guard moved — this pin cannot see it"
    probe = COMMON_SH.read_text(encoding="utf-8")
    assert f"MEMKIT_CHECKER_FLOOR_MAJOR={match.group(1)}" in probe
    assert f"MEMKIT_CHECKER_FLOOR_MINOR={match.group(2)}" in probe


# --- the wrappers, as processes ----------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A plugin root whose files are the repo's, reached through symlinks.

    Symlinked FILES rather than directories, deliberately. The wrappers resolve
    their own tree with `cd "$(dirname $0)/.." && pwd -P`, and `pwd -P` walks
    through a symlinked *directory* component — so a `bin` symlink would send
    every wrapper back to the real repo and quietly test the wrong tree.
    """
    plugin = tmp_path / "plugin"
    for rel in PAYLOAD:
        dest = plugin / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(REPO / rel)
    return plugin


def _shim(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


# Records what the wrapper handed it, instead of being an interpreter. `-` as a
# default marks unset, which is a different fact from empty: the hook wrapper
# must UNSET MEMKIT_CONFIG when no rung answers, not blank it.
SHIM_BODY = """
{
  echo "argv=$*"
  echo "MEMKIT_CONFIG=${MEMKIT_CONFIG-<unset>}"
  echo "MEMKIT_PLUGIN=${MEMKIT_PLUGIN-<unset>}"
  echo "MEMKIT_CHECKER_ROUTE=${MEMKIT_CHECKER_ROUTE-<unset>}"
  echo "MEMKIT_CHECKER_CMD=${MEMKIT_CHECKER_CMD-<unset>}"
  echo "PYTHONPATH=${PYTHONPATH-<unset>}"
} > "$SHIM_OUT"
exit 0
"""


def _run(
    wrapper: Path, *args: str, env: dict, cwd: Path | None = None, shell_trace=False
) -> subprocess.CompletedProcess:
    argv = ["sh", "-x", str(wrapper)] if shell_trace else [str(wrapper)]
    return subprocess.run(
        [*argv, *args],
        capture_output=True, text=True, timeout=60, env=env,
        cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL,
    )


class Shim:
    """A PATH holding one fake `python3`, and a reader for what it saw.

    A class rather than attributes bolted onto a returned function: the shim's
    directory and its output file are things a case reaches for, and hanging
    them off a callable cost three `type: ignore`s in a file whose pyright pass
    is now a gate.

    Callable, so the call sites that build an environment read as they did.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "shimbin"
        self.out = tmp_path / "shim-out.txt"
        self._home = tmp_path / "home"
        _shim(self.dir, "python3", SHIM_BODY)

    def __call__(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": f"{self.dir}:/usr/bin:/bin",
            "HOME": str(self._home),
            "SHIM_OUT": str(self.out),
        }
        env.update(extra)
        return env

    def read(self) -> dict[str, str]:
        return dict(
            line.split("=", 1)
            for line in self.out.read_text().splitlines()
            if "=" in line
        )


@pytest.fixture
def shimmed(tmp_path: Path) -> Shim:
    return Shim(tmp_path)


def _config_file(path: Path, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"schema": hook.SCHEMA, "roots": {}, "stores": []}
    blob.update(extra)
    path.write_text(json.dumps(blob))
    return path


def test_rung_one_is_the_manifest_option(root, tmp_path, shimmed) -> None:
    config = _config_file(tmp_path / "opt.json")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    seen = shimmed.read()
    assert seen["MEMKIT_CONFIG"] == str(config)
    assert seen["MEMKIT_PLUGIN"] == "1"
    # Zero arguments to the hook: anything else is the search CLI.
    assert seen["argv"] == str(root / "src" / "memkit" / "memory_prompt_recall.py")


def test_rung_one_expands_a_tilde_the_way_a_person_types_it(
    root, tmp_path, shimmed
) -> None:
    """The option value is a string typed into an install command, not a word
    any shell expanded. `~/.cache/...` — which is what the manifest's own
    default says — arrives with a literal tilde, and every rung would miss it.
    """
    home = tmp_path / "home"
    config = _config_file(home / ".cache" / "memory-recall" / "memkit.json")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG="~/.cache/memory-recall/memkit.json")
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_rung_two_is_the_plugin_data_dir(root, tmp_path, shimmed) -> None:
    data = tmp_path / "plugindata"
    config = _config_file(data / "memkit.json")
    env = shimmed(CLAUDE_PLUGIN_DATA=str(data))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_a_config_inside_the_payload_is_not_a_rung(root, tmp_path, shimmed) -> None:
    """The admission rule, from the side that has no environment to check.

    A plugin install is a clone of a pinned commit, so a `memkit.json` at the
    payload root is a file the repository can ship — and a config names the
    directories the every-prompt hook reads and the binary it exec's. Both
    halves are asserted here because they fail differently: the config half
    poisons what is served, and the interpreter half decides what runs at all,
    before anything has parsed a byte of JSON.
    """
    _config_file(root / "memkit.json", interpreter=str(tmp_path / "evil"))
    env = shimmed()
    assert "CLAUDE_PLUGIN_ROOT" not in env and "CLAUDE_PLUGIN_DATA" not in env
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    seen = shimmed.read()
    assert seen["MEMKIT_CONFIG"] == "<unset>", seen["MEMKIT_CONFIG"]
    # The shim on PATH answered, i.e. the payload's file named no interpreter
    # either. A rung reading it would have made this run the other one.
    assert seen["argv"] == str(root / "src" / "memkit" / "memory_prompt_recall.py")


def test_the_rungs_are_tried_in_order(root, tmp_path, shimmed) -> None:
    option = _config_file(tmp_path / "one.json")
    data = tmp_path / "two"
    _config_file(data / "memkit.json")
    env = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(option), CLAUDE_PLUGIN_DATA=str(data)
    )
    _run(root / "bin" / "memkit-hook", env=env)
    assert shimmed.read()["MEMKIT_CONFIG"] == str(option)

    # Rung 1 naming a file that is not there yet is the NORMAL state between
    # install and init, and it must fall through rather than stopping.
    env = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(tmp_path / "absent.json"),
        CLAUDE_PLUGIN_DATA=str(data),
    )
    _run(root / "bin" / "memkit-hook", env=env)
    assert shimmed.read()["MEMKIT_CONFIG"] == str(data / "memkit.json")


@pytest.mark.parametrize(
    "wrapper,args",
    [
        ("memkit-hook", ()),
        ("memkit-recall", ("--search", "x")),
        ("memkit", ("doctor",)),
    ],
)
def test_every_wrapper_answers_the_config_question_identically(
    root, tmp_path, shimmed, wrapper, args
) -> None:
    """The config-delivery policy is hand-duplicated in all three wrappers and
    every rung and override case drove one of them.

    Both directions of the override, because the second is the one that
    matters: setting `MEMKIT_CONFIG` when a rung answered is obvious, and
    UNSETTING it when none did is what stops an adopter's other memkit
    installation handing this one a corpus nobody pointed it at. A wrapper that
    dropped the `unset` would serve the ambient config and look identical from
    outside.
    """
    config = _config_file(tmp_path / "shared.json")
    answered = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config),
        MEMKIT_CONFIG=str(tmp_path / "ambient.json"),
    )
    assert _run(root / "bin" / wrapper, *args, env=answered).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config), wrapper

    shimmed.out.unlink()
    ambient = shimmed(MEMKIT_CONFIG=str(_config_file(tmp_path / "ambient.json")))
    assert _run(root / "bin" / wrapper, *args, env=ambient).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>", wrapper


def test_no_rung_leaves_the_config_unset_rather_than_inherited(
    root, tmp_path, shimmed
) -> None:
    """A hard override in BOTH directions, and the unsetting half is the one
    that matters. An adopter with a nix or pip memkit on the same machine may
    have MEMKIT_CONFIG exported in the shell that launched the harness;
    inheriting it would make the plugin serve stores nobody pointed it at.
    """
    env = shimmed(MEMKIT_CONFIG=str(_config_file(tmp_path / "ambient.json")))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>"


def test_an_unset_data_dir_never_becomes_a_root_level_path(
    root, tmp_path, shimmed
) -> None:
    """`${CLAUDE_PLUGIN_DATA}/memkit.json` with the variable unset is
    `/memkit.json`, and a hook that reads every prompt must never stat a
    root-level path it did not mean to name.

    Read off `sh -x`, which traces every command the shell evaluated — the only
    portable way to see a test the wrapper *made* rather than a path it
    returned. The cwd-relative variant of the same bug is covered too: a
    `memkit.json` sitting in the directory the session happens to stand in must
    not be picked up either.
    """
    cwd = tmp_path / "somewhere"
    cwd.mkdir()
    _config_file(cwd / "memkit.json")
    for env in (shimmed(), shimmed(CLAUDE_PLUGIN_DATA="")):
        out = _run(root / "bin" / "memkit-hook", env=env, cwd=cwd, shell_trace=True)
        assert out.returncode == 0
        # `/memkit.json` as a WHOLE word: every legitimate candidate this
        # builds is prefixed by an absolute directory, so the bare root-level
        # path can only appear through the empty expansion.
        assert not re.search(r"(?<![\w/])/memkit\.json\b", out.stderr), out.stderr
        assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>"


def test_a_recorded_interpreter_wins_over_the_path(root, tmp_path, shimmed) -> None:
    """PATH probing alone hands the process that reads every prompt to whatever
    direnv/mise/venv shim the launching shell carried. The nix channel pins its
    interpreter absolutely; the plugin channel must not silently drop that.
    """
    recorded = _shim(tmp_path / "recorded", "python3", SHIM_BODY)
    marker = tmp_path / "recorded-ran.txt"
    recorded.write_text(
        f'#!/bin/sh\necho "$*" > "{marker}"\n' + SHIM_BODY
    )
    recorded.chmod(0o755)
    config = _config_file(tmp_path / "rec.json", interpreter=str(recorded))
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert marker.is_file(), "the PATH interpreter answered instead of the recorded one"


def test_an_unusable_recorded_interpreter_falls_back_to_the_path(
    root, tmp_path, shimmed
) -> None:
    """An interpreter recorded at init and gone by now — a venv deleted, a
    homebrew python upgraded out from under its path — must not take retrieval
    down with it."""
    config = _config_file(
        tmp_path / "gone.json", interpreter=str(tmp_path / "no" / "such" / "python3")
    )
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_a_relative_recorded_interpreter_is_not_a_path_into_the_session_dir(
    root, tmp_path, shimmed
) -> None:
    """`[ -x ]` and `exec` resolve a non-absolute value against different
    things, and only one of them is the wrapper's to choose.

    The test is against the wrapper's CWD, which under the harness is whatever
    directory the session stands in. `exec` on a value with a slash resolves
    against the same CWD and runs it; on a slashless word it searches PATH
    instead. So a recorded `./interp/python3` used to hand the process that
    reads every prompt to an executable sitting in the directory somebody
    happened to open — chosen by the session, not by the install.
    """
    session = tmp_path / "session"
    marker = tmp_path / "relative-ran.txt"
    _shim(session / "interp", "python3", f'echo ran > "{marker}"')
    config = _config_file(tmp_path / "rel.json", interpreter="./interp/python3")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    out = _run(root / "bin" / "memkit-hook", env=env, cwd=session)
    assert out.returncode == 0, out.stderr
    assert not marker.exists(), "a config named an interpreter inside the cwd"
    assert shimmed.read()["argv"] == str(
        root / "src" / "memkit" / "memory_prompt_recall.py"
    )


def test_a_recorded_interpreter_the_path_cannot_answer_still_exits_zero(
    root, tmp_path
) -> None:
    """The exit contract, at the config field rather than at an empty PATH.

    A slashless `"interpreter": "python3"` passed the executable test against
    a CWD that happened to hold a file of that name, and then `exec` — which
    searches PATH for a slashless word — found nothing and left the
    every-prompt hook exiting 127. On UserPromptSubmit that is a blocked turn,
    produced by a config field on a machine where the fallback would have
    worked had it been consulted.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("sed", "head"):
        found = shutil.which(name)
        assert found, name
        (tools / name).symlink_to(found)
    session = tmp_path / "session"
    _shim(session, "python3", "exit 0")
    config = _config_file(tmp_path / "bare.json", interpreter="python3")
    env = {
        "PATH": str(tools),
        "HOME": str(tmp_path / "home"),
        "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(config),
    }
    out = _run(root / "bin" / "memkit-hook", env=env, cwd=session)
    assert out.returncode == 0, (out.returncode, out.stderr)
    assert "no python3" in out.stderr


def test_a_directory_recorded_as_the_interpreter_is_not_exec_d(
    root, tmp_path, shimmed
) -> None:
    """`[ -x ]` is true of a DIRECTORY — the execute bit means "searchable" —
    so a value like `/opt/homebrew/opt/python@3.12/libexec/bin`, which is how
    the field gets written by hand from a PATH entry with the last segment
    dropped, passed the guard and skipped the PATH probe.

    `exec` then died 126, on every prompt of every session, with no fallback:
    the hook wrapper's whole contract is that every path exits 0, and 126 on
    UserPromptSubmit hands the user their prompt back unanswered. The status
    comes from `exec` rather than from a literal, so no scrape can see it —
    only a run can.
    """
    a_directory = tmp_path / "libexec-bin"
    a_directory.mkdir()
    config = _config_file(tmp_path / "dir.json", interpreter=str(a_directory))
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    out = _run(root / "bin" / "memkit-hook", env=env)
    assert out.returncode == 0, (out.returncode, out.stderr)
    # The PATH probe answered, which is the fallback the refusal promises.
    assert shimmed.read()["argv"] == str(
        root / "src" / "memkit" / "memory_prompt_recall.py"
    )
    # And every wrapper, because each one has its own exit vocabulary and 126
    # is in none of them.
    for wrapper, args in (
        ("memkit-recall", ("--search", "flange torque")),
        ("memkit", ("doctor",)),
    ):
        other = _run(root / "bin" / wrapper, *args, env=shimmed(
            CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)
        ))
        assert other.returncode != 126, (wrapper, other.returncode, other.stderr)


def test_a_relative_config_path_is_not_a_path_into_the_session_dir(
    root, tmp_path, shimmed
) -> None:
    """The same rule as the interpreter field, on the rung above it.

    An adopter who typed `--config memkitConfig=memkit.json` at install has
    every repository they later open handing the every-prompt hook its own
    `memkit.json` — which names both the store roots whose file contents are
    injected into the model's context and the absolute binary exec'd on every
    prompt. The manifest asks for an absolute path and nothing enforced it.
    """
    session = tmp_path / "someone-elses-repo"
    session.mkdir()
    marker = tmp_path / "session-config-used.txt"
    _shim(session, "python3", f'echo used > "{marker}"')
    _config_file(session / "memkit.json", interpreter=str(session / "python3"))
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG="memkit.json")
    out = _run(root / "bin" / "memkit-hook", env=env, cwd=session)
    assert out.returncode == 0, out.stderr
    assert not marker.exists(), "the session directory named the interpreter"
    assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>"


def test_a_relative_plugin_data_dir_is_not_a_path_into_the_session_dir(
    root, tmp_path, shimmed
) -> None:
    """The same rule as rung 1, on the rung beside it.

    `$CLAUDE_PLUGIN_DATA` is the harness's variable and the wrappers do not vet
    where it came from, which is recorded above the resolver. What they CAN do
    is refuse to resolve it against the session's directory: a relative value
    made the every-prompt hook read `<cwd>/<value>/memkit.json`, a config
    naming both the store roots whose contents are injected and the binary that
    is exec'd. The comment claiming "ABSOLUTE, on every rung" was written by
    the commit that guarded one of the two.
    """
    session = tmp_path / "someone-elses-repo"
    (session / "reldata").mkdir(parents=True)
    marker = tmp_path / "session-data-used.txt"
    _shim(session, "python3", f'echo used > "{marker}"')
    _config_file(
        session / "reldata" / "memkit.json", interpreter=str(session / "python3")
    )
    env = shimmed(CLAUDE_PLUGIN_DATA="reldata")
    out = _run(root / "bin" / "memkit-hook", env=env, cwd=session)
    assert out.returncode == 0, out.stderr
    assert not marker.exists(), "the session directory named the interpreter"
    assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>"

    # And a tilde is expanded rather than refused, exactly as rung 1 does it —
    # the value is typed by a person or written by a harness, not expanded by
    # any shell.
    home = tmp_path / "home"
    data = home / "plugindata"
    config = _config_file(data / "memkit.json")
    tilde = shimmed(CLAUDE_PLUGIN_DATA="~/plugindata")
    assert _run(root / "bin" / "memkit-hook", env=tilde).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_every_wrapper_declares_the_name_it_answers_to() -> None:
    """`MEMKIT_SELF` decides the binary named in every message the shared
    library emits, and the library's fallback means a wrapper that forgot to
    set it fails silently — naming the wrong binary beside its own exit code,
    which is the defect the variable was added to fix.

    Nothing in the library can enforce it: a hard failure there would sit on
    the every-prompt path for a diagnostic. So the enforcement is here.
    """
    for wrapper in ("memkit", "memkit-hook", "memkit-recall"):
        text = (BIN / wrapper).read_text()
        assert f"\nMEMKIT_SELF={wrapper}\n" in text, wrapper
        # Before the library is SOURCED — not merely before it is mentioned —
        # or the first message it emits on this wrapper's behalf carries the
        # fallback instead of the name.
        sourced = text.index('. "$MEMKIT_ROOT/bin/lib/common.sh"')
        assert text.index(f"MEMKIT_SELF={wrapper}") < sourced, wrapper


@pytest.mark.parametrize(
    "value",
    [
        "/proc/self/cwd/memkit.json",
        "//etc/memkit.json",
        "/./etc/memkit.json",
        "/tmp/../etc/memkit.json",
        "/dev/fd/3/memkit.json",
        "relative/memkit.json",
    ],
)
def test_a_config_rung_admits_only_what_the_interpreter_rule_admits(
    root, tmp_path, shimmed, value
) -> None:
    """One admission rule, applied to the whole class it names.

    The interpreter field refused non-canonical and process-relative paths and
    the config rungs tested only for a leading slash — the wider blast radius
    guarded more weakly, since a config decides which directories the
    every-prompt hook reads AND which binary it execs. On Linux, which is this
    repo's CI and the nix channel, `/proc/self/cwd/memkit.json` is absolute,
    passes a leading-slash test, and resolves through the running process.

    Driven with a file that really is there at the non-canonical spelling, so
    a refusal cannot be confused with the path not existing.
    """
    data = tmp_path / "data"
    data.mkdir()
    _config_file(data / "memkit.json")
    noncanonical = f"{tmp_path}/./data/memkit.json"
    for rung, env in (
        ("option", shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=noncanonical)),
        ("data dir", shimmed(CLAUDE_PLUGIN_DATA=f"{tmp_path}/./data")),
    ):
        out = _run(root / "bin" / "memkit-hook", env=env)
        assert out.returncode == 0, (rung, out.stderr)
        assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>", rung

    # The class itself, one spelling per run, through the option.
    out = _run(
        root / "bin" / "memkit-hook",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=value),
    )
    assert out.returncode == 0, out.stderr
    assert shimmed.read()["MEMKIT_CONFIG"] == "<unset>", value

    # And a canonical absolute path is still served, or this is "refuse
    # everything" wearing a rule's clothes.
    good = _config_file(tmp_path / "good" / "memkit.json")
    assert _run(
        root / "bin" / "memkit-hook",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(good)),
    ).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(good)


def test_a_process_relative_interpreter_is_refused(root, tmp_path, shimmed) -> None:
    """Absolute is not the same as fixed. `/proc/self/cwd/python3` is resolved
    by the kernel through the RUNNING process, so it names an executable in
    whatever directory the session stands in — the outcome the absoluteness
    rule exists to prevent, restored on Linux, which is what the nix channel
    and this repo's CI run.

    `/dev/fd/*` is the same class and is reachable on this platform, which is
    what makes the case runnable here rather than only reasoned about.
    """
    # Every respelling, because the guard is a literal prefix test and the
    # kernel normalises before it resolves — six of these walked past it while
    # naming exactly what it refuses.
    for value in (
        "/proc/self/cwd/python3",
        "//proc/self/cwd/python3",
        "/./proc/self/cwd/python3",
        "/tmp/../proc/self/cwd/python3",
        "/proc//self/cwd/python3",
        "/usr/../proc/self/cwd/python3",
        "/a/./b/../proc/self/cwd/python3",
        "/dev/fd/3/python3",
        "/dev//fd/3/python3",
        "/./dev/fd/3/python3",
        "/proc/self/cwd/.",
        "/proc/self/cwd/..",
    ):
        config = _config_file(tmp_path / "proc.json", interpreter=value)
        env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
        out = _run(root / "bin" / "memkit-hook", env=env)
        assert out.returncode == 0, out.stderr
        assert value in out.stderr, (value, out.stderr)
        assert (
            "session stands in" in out.stderr or "canonical" in out.stderr
        ), (value, out.stderr)
        assert shimmed.read()["argv"] == str(
            root / "src" / "memkit" / "memory_prompt_recall.py"
        )
    # And a canonical absolute path is still honoured, or the guard above is
    # just "refuse everything".
    honoured = _shim(tmp_path / "real", "python3", "exit 0")
    marker = tmp_path / "canonical-ran.txt"
    honoured.write_text(f'#!/bin/sh\necho ran > "{marker}"\n')
    honoured.chmod(0o755)
    config = _config_file(tmp_path / "ok.json", interpreter=str(honoured))
    assert _run(
        root / "bin" / "memkit-hook",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)),
    ).returncode == 0
    assert marker.is_file(), "a canonical absolute interpreter was refused"


def test_a_recorded_interpreter_expands_a_tilde_and_says_when_it_is_refused(
    root, tmp_path, shimmed
) -> None:
    """Two halves of one complaint: an adopter records an interpreter, keeps a
    working install, and runs every prompt under a python they did not choose.

    The tilde half is a trap of this file's own making — the config PATH one
    rung above is tilde-expanded, so the file teaches that `~` works — and the
    silence half is what made it undiagnosable: exit 0, nothing on stderr, and
    no surface in this build reports the resolved interpreter.
    """
    home = tmp_path / "home"
    recorded = _shim(home / "venv" / "bin", "python3", "exit 0")
    marker = tmp_path / "recorded-ran.txt"
    recorded.write_text(f'#!/bin/sh\necho ran > "{marker}"\n')
    recorded.chmod(0o755)
    config = _config_file(tmp_path / "tilde.json", interpreter="~/venv/bin/python3")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert marker.is_file(), "a tilde interpreter was refused, as if it were relative"

    # And a value that really is unusable is refused OUT LOUD.
    gone = _config_file(tmp_path / "gone.json", interpreter=str(tmp_path / "nope"))
    out = _run(
        root / "bin" / "memkit-hook",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(gone)),
    )
    assert out.returncode == 0
    assert "is not an executable file" in out.stderr, out.stderr
    assert "Falling back" in out.stderr, out.stderr


def test_every_shared_message_names_the_wrapper_that_is_running(
    root, tmp_path
) -> None:
    """The messages are shared between the three wrappers and the exit codes
    are not.

    `memkit-recall` exits 4 when nothing can run it. An agent that read
    `memkit:` on that line looks 4 up in the `memkit` table, where it means
    "the subcommand exists and is not in this build" — a wrong diagnosis
    produced by the name in the message rather than by the code.
    """
    empty = {"PATH": str(tmp_path / "nothing"), "HOME": str(tmp_path)}
    for wrapper, args in (
        ("memkit-hook", ()),
        ("memkit-recall", ("--search", "x")),
        ("memkit", ("doctor",)),
    ):
        out = _run(root / "bin" / wrapper, *args, env=empty)
        assert out.stderr.startswith(f"{wrapper}: "), (wrapper, out.stderr)


def test_no_interpreter_is_a_named_refusal_that_still_exits_zero(
    root, tmp_path
) -> None:
    """The exit contract, at the one failure that cannot be fixed by fixing the
    store. Exit 2 on UserPromptSubmit blocks the turn and hands the prompt back
    unanswered; any non-zero exit puts an error in front of the user on every
    prompt. So the refusal exits 0 and speaks on stderr, where doctor — which
    runs this wrapper directly — reads it.
    """
    env = {"PATH": str(tmp_path / "empty"), "HOME": str(tmp_path)}
    out = _run(root / "bin" / "memkit-hook", env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""
    assert "no python3" in out.stderr and "3.9" in out.stderr


def _code_only(line: str) -> str:
    r"""One shell line with its comment and its quoted spans removed.

    Both removals matter for finding a real `exit`. A trailing `# why` must not
    hide the statement in front of it, and the word inside
    `echo "exit 98 in a string"` must not be read as one. Quoted spans are
    blanked rather than dropped so column-shaped assertions stay honest.

    Only a `#` that starts a word is a comment: `"$#"` is an argument count and
    `${1#\~/}` is a parameter expansion, and both occur in these wrappers.
    """
    out: list[str] = []
    quote = None
    for char in line:
        if quote is not None:
            out.append(" " if char != quote else char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "#" and (not out or out[-1].isspace()):
            break
        else:
            out.append(char)
    return "".join(out)


EXIT_TOKEN = re.compile(r"(?<![\w-])exit(?![\w-])")


def _exit_statuses(text: str, where: str) -> set[int]:
    r"""Every status a shell file can `exit` with, DEFAULT-DENY.

    The rule is inverted from the obvious one: rather than matching the exit
    forms we expect and ignoring the rest, this finds every `exit` token in
    code and fails on any whose form it does not recognise. The previous shape
    — `^\s*exit (\d+)$` — required end-of-line immediately after the digits,
    so `exit 1  # why`, `cmd && exit 1` (the idiomatic POSIX form) and a bare
    `exit` were all invisible, and a paired "no computed exits" guard did not
    fire when the next character was a digit. Measured: an unreached
    `exit 1  # comment` planted in the hook wrapper left the whole suite green.

    Necessary and NOT sufficient, in both directions, which is why every case
    using this pairs it with real runs. A scrape cannot see `set -u` aborting
    on an unbound variable — non-zero, with no literal to find — and it cannot
    see the final `exec`, which hands the python side's own status through as
    the wrapper's. What it does see is the thing a runtime case cannot: a
    branch nobody remembered to reach.
    """
    statuses = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        code = _code_only(raw).rstrip()
        if not EXIT_TOKEN.search(code):
            continue
        at = f"{where}:{lineno}"
        assert len(EXIT_TOKEN.findall(code)) == 1, f"{at}: two exits on one line"
        # Whatever precedes it may be a `&&`/`||` chain or a `case` arm; what
        # FOLLOWS it must be a literal status and nothing else.
        match = re.search(EXIT_TOKEN.pattern + r"\s*(\S*)\s*(;;)?$", code)
        assert match, f"{at}: unrecognised exit form: {code.strip()!r}"
        assert match.group(1).isdigit(), (
            f"{at}: exit without a literal status — a bare `exit` propagates "
            f"$? and a computed one cannot be read here: {code.strip()!r}"
        )
        statuses.add(int(match.group(1)))
    assert statuses, f"{where} has no exit literals — this pin would be vacuous"
    return statuses


def _exit_literals(wrapper: str) -> set[int]:
    return _exit_statuses((REPO / "bin" / wrapper).read_text(), wrapper)


def test_the_sourced_library_can_end_no_wrapper() -> None:
    """`bin/lib/common.sh` is SOURCED, so an `exit` in it exits the wrapper —
    and every scrape read `bin/<wrapper>` only.

    Measured: a top-level `[ … ] && exit 1` planted in the library made
    `bin/memkit-hook` return 1 — the non-zero `UserPromptSubmit` exit the hook
    wrapper's whole contract is about, which blocks the turn — and the entire
    suite stayed green. The library is also where this round put new refusal
    paths, so it is exactly the file gaining reasons to want one.

    A NEGATIVE pin rather than `_exit_literals`, which asserts a non-empty set:
    the right number of exits here is none.
    """
    text = COMMON_SH.read_text(encoding="utf-8")
    found = [
        (n, line.strip())
        for n, line in enumerate(text.splitlines(), 1)
        if EXIT_TOKEN.search(_code_only(line))
    ]
    assert not found, found
    # `return` is how this file ends a function, and it must have some, or the
    # assertion above is about a file that stopped being a resolver.
    assert re.search(r"^\s*return \d+$", text, re.MULTILINE), "no returns either"


def test_the_exit_scrape_sees_the_forms_it_would_otherwise_miss() -> None:
    """The anti-vacuity control for the helper above, which is the static half
    of the hook's fail-open contract.

    Each line here is a form the previous regex could not see. A scrape that
    silently stops matching is a green test about nothing, and this one guards
    a property whose failure mode is an error in front of every prompt.
    """
    seen = _exit_statuses(
        "\n".join(
            (
                "exit 0",
                "exit 2  # trailing comment",
                "[ -n \"$x\" ] && exit 3",
                "foo || exit 4",
                "    exit 5 ;;",
                "# exit 99 in prose is not an exit",
                'echo "exit 98 in a string is not one either"',
            )
        ),
        "<control>",
    )
    assert seen == {0, 2, 3, 4, 5}, seen
    for hostile in ("exit", "exit $rc", "exit 1 || true", "exit 0; exit 1"):
        with pytest.raises(AssertionError):
            _exit_statuses(hostile, "<control>")


def _half_delivered(tmp_path: Path, wrapper: str, *, library: bool) -> Path:
    """A payload root missing the hook file, with or without the library.

    The distinction is the whole point: copying only the wrapper leaves control
    at the `common.sh` guard, so the branch that answers for an incomplete
    payload is never executed and a wrong exit code there survives every test
    that thinks it covers it.
    """
    root = tmp_path / f"{wrapper}-{'half' if library else 'bare'}"
    if root.exists():
        # Built once per shape and reused. Re-copying is not merely wasteful:
        # the source may be read-only — it is under /nix/store in the packaged
        # check — and `shutil.copy` carries the mode across, so the second
        # write to the same destination fails with EACCES.
        return root
    (root / "bin" / "lib").mkdir(parents=True)
    shutil.copy(REPO / "bin" / wrapper, root / "bin" / wrapper)
    if library:
        shutil.copy(COMMON_SH, root / "bin" / "lib" / "common.sh")
    return root


def test_the_hook_wrapper_never_exits_non_zero(root, tmp_path, shimmed) -> None:
    """Every reachable refusal, in one place, because the property is about the
    SET of them: a new branch that exits 1 is invisible until it is the branch
    an adopter is on, and by then it is a message in front of every prompt.
    """
    assert _exit_literals("memkit-hook") == {0}

    cases = [
        # no interpreter anywhere
        (root / "bin" / "memkit-hook", {"PATH": str(tmp_path / "nothing")}, ()),
        # cannot locate the tree at all: no library to source
        (_half_delivered(tmp_path, "memkit-hook", library=False) / "bin"
         / "memkit-hook", shimmed(), ()),
        # the library is there and the hook file is not — the branch the case
        # above cannot reach, because control leaves at the library guard
        (_half_delivered(tmp_path, "memkit-hook", library=True) / "bin"
         / "memkit-hook", shimmed(), ()),
        # arguments that should never arrive, and are ignored if they do
        (root / "bin" / "memkit-hook", shimmed(), ("--search", "x")),
    ]
    for wrapper, env, args in cases:
        out = _run(wrapper, *args, env={"HOME": str(tmp_path), **env})
        assert out.returncode == 0, (wrapper, args, out.returncode, out.stderr)
    # And the branches really are distinct, or two of the cases above are one
    # case run twice.
    messages = {
        _run(
            _half_delivered(tmp_path, "memkit-hook", library=lib) / "bin"
            / "memkit-hook",
            env={"HOME": str(tmp_path), **shimmed()},
        ).stderr.split("\n")[0]
        for lib in (False, True)
    }
    assert len(messages) == 2, messages


@pytest.mark.parametrize(
    "wrapper,args,allowed",
    [
        ("memkit-hook", (), {0}),
        ("memkit-recall", ("--search", "x"), {2, 4}),
        ("memkit", ("doctor",), {1}),
    ],
)
def test_a_payload_that_cannot_be_READ_is_refused_not_exec_d(
    tmp_path, shimmed, wrapper, args, allowed
) -> None:
    """`[ -f ]` admits a file this process cannot open, and the guard's whole
    job is to stop before something else fails on it.

    Measured on the hook wrapper: with the payload at mode 000 the guard passed
    and `exec` reached CPython, which exits **2** when it cannot open the
    script — the one status the file's own header says must never be returned,
    because on UserPromptSubmit it hands the user their prompt back unanswered.
    An unreadable `common.sh` is the same shape one line up: sourcing it is
    fatal on some `/bin/sh`.

    Each wrapper against ITS OWN documented set, because they do not share one:
    the hook may only ever exit 0, and the other two have codes that mean
    "could not start".
    """
    root = tmp_path / "payload"
    for rel in PAYLOAD:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dest)
        dest.chmod(0o755 if rel.startswith("bin/") else 0o644)

    for unreadable in ("src/memkit/memory_prompt_recall.py", "src/memkit/cli.py"):
        (root / unreadable).chmod(0o000)
    out = _run(root / "bin" / wrapper, *args, env=shimmed())
    assert out.returncode in allowed, (wrapper, out.returncode, out.stderr)
    assert "incomplete" in out.stderr, out.stderr

    # And the library one rung up, which is sourced rather than exec'd.
    (root / "bin" / "lib" / "common.sh").chmod(0o000)
    out = _run(root / "bin" / wrapper, *args, env=shimmed())
    assert out.returncode in allowed, (wrapper, out.returncode, out.stderr)
    assert "cannot locate the plugin tree" in out.stderr, out.stderr


def test_the_search_wrapper_says_it_could_not_start_rather_than_that_you_erred(
    root, tmp_path, shimmed
) -> None:
    """The opposite assertion to the hook wrapper's, and it is opposite on
    purpose: here there is no prompt to get out of the way of, and the caller
    is an agent choosing a next move from the code.

    2 already means "what you asked for is wrong" — an unparseable config, a
    `--dir` that is not there, arguments that make no sense — and all three of
    those send an agent to fix its own request. A machine with no python on it
    answers none of them, so it gets a code of its own; otherwise the one
    failure no query can survive is reported as the one class of failure a
    different query might.
    """
    assert _exit_literals("memkit-recall") == {hook.EXIT_ERROR, hook.EXIT_CANNOT_START}
    assert hook.EXIT_CANNOT_START not in (
        hook.EXIT_OK, hook.EXIT_NO_MATCH, hook.EXIT_ERROR, hook.EXIT_INERT
    )
    # A code an agent branches on and cannot look up is a code it has to guess
    # at, so the row is part of the change rather than a follow-up — and this
    # one has to warn about the collision, because `memkit`'s own table gives
    # the same number a different meaning.
    from memkit.cli import EXIT_NOT_IN_BUILD

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    row = next(
        (
            line for line in readme.splitlines()
            if line.startswith(f"| {hook.EXIT_CANNOT_START} |")
            and "memkit-recall" in line
        ),
        None,
    )
    assert row, "no README row for the search CLI's start-failure code"
    assert EXIT_NOT_IN_BUILD == hook.EXIT_CANNOT_START and "different meaning" in row

    empty = {"PATH": str(tmp_path / "nothing"), "HOME": str(tmp_path)}
    cannot_start = [
        # no interpreter anywhere
        (root / "bin" / "memkit-recall", empty, ("--search", "x")),
        # cannot locate the tree
        (_half_delivered(tmp_path, "memkit-recall", library=False) / "bin"
         / "memkit-recall", shimmed(), ("--search", "x")),
        # the library present, the hook file absent
        (_half_delivered(tmp_path, "memkit-recall", library=True) / "bin"
         / "memkit-recall", shimmed(), ("--search", "x")),
    ]
    for wrapper, env, args in cannot_start:
        out = _run(wrapper, *args, env={"HOME": str(tmp_path), **env})
        assert out.returncode == hook.EXIT_CANNOT_START, (
            wrapper, out.returncode, out.stderr
        )
    # And the one branch that really is a wrong invocation keeps saying so.
    bare = _run(root / "bin" / "memkit-recall", env=shimmed())
    assert bare.returncode == hook.EXIT_ERROR, bare.stderr


def test_arguments_to_the_hook_wrapper_are_ignored_not_forwarded(
    root, shimmed
) -> None:
    """The second of two independent guards on the same failure. hooks.json
    passing zero arguments is pinned above; this is what holds if that pin is
    ever edited away, because a forwarded `--search` turns the every-prompt
    hook into a CLI whose argparse exits 2 — a blocked turn, every turn.
    """
    env = shimmed()
    out = _run(root / "bin" / "memkit-hook", "--search", "anything", env=env)
    assert out.returncode == 0
    assert shimmed.read()["argv"] == str(
        root / "src" / "memkit" / "memory_prompt_recall.py"
    )
    assert "ignoring 2 argument" in out.stderr


def test_the_wrapper_resolves_its_tree_through_a_doubled_separator(
    root, tmp_path, shimmed
) -> None:
    """The harness expands `${CLAUDE_PLUGIN_ROOT}` WITH a trailing slash
    (measured on 2.1.238), so the registration hands the wrapper
    `<root>//bin/memkit-hook`. Everything the wrapper builds is derived from
    that string.

    Read off the hook path the wrapper hands its interpreter, because that is
    the derived value the whole tree is built from — equality rather than a
    "//" search, so a derivation that normalized the separator and landed in
    the wrong tree is caught too.
    """
    doubled = f"{root}//bin/memkit-hook"
    out = subprocess.run(
        [doubled], capture_output=True, text=True, timeout=60,
        env=shimmed(), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["argv"] == str(root / "src" / "memkit" / "memory_prompt_recall.py")


@pytest.mark.parametrize(
    "wrapper,args",
    [
        ("memkit-hook", ()),
        ("memkit-recall", ("--search", "flange torque")),
        ("memkit", ("doctor",)),
    ],
)
def test_a_wrapper_invoked_by_name_from_the_path_still_finds_its_tree(
    root, shimmed, wrapper, args
) -> None:
    """`bin/` is on the agent's PATH while the plugin is enabled, so a bare
    `memkit …` has to find its own tree with no directory in argv[0] to walk up
    from.

    All THREE wrappers, because the derivation is hand-duplicated in each and
    covering one covered one: measured, deleting the `command -v` branch from
    `bin/memkit-hook` and from `bin/memkit` left the whole suite green. It
    matters most for `bin/memkit`, which is the one an agent really does type
    bare, and where a broken derivation is an exit 1 with "cannot locate the
    plugin tree" rather than anything about memkit.

    Run through `sh <name>` from the wrapper's own directory, and that is the
    only way to produce the case rather than a convenience. MEASURED on this
    platform: a shebang script executed through a PATH lookup receives the
    RESOLVED path as `$0` — the kernel passes execve's pathname to the
    interpreter, not argv[0] — so `subprocess.run(["memkit-recall", …])`
    exercises the slashed branch and says nothing about this one. Handing the
    name to `sh` directly is what leaves `$0` bare.

    The tree the wrapper resolved is the whole assertion: a `command -v` that
    answered with a different install on the same PATH would run that install's
    files, which is the wrong-tree failure this derivation exists to avoid, and
    it is invisible in an exit code.
    """
    env = shimmed()
    env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
    out = subprocess.run(
        ["sh", wrapper, *args],
        capture_output=True, text=True, timeout=60, env=env,
        cwd=str(root / "bin"), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    if wrapper == "memkit":
        # The dispatcher runs the package rather than the hook file, so the
        # tree shows up as the PYTHONPATH it prepends.
        assert seen["argv"] == "-m memkit.cli doctor", seen["argv"]
        assert seen["PYTHONPATH"].split(":")[0] == str(root / "src")
    else:
        hook_file = root / "src" / "memkit" / "memory_prompt_recall.py"
        assert seen["argv"] == " ".join((str(hook_file), *args)), seen["argv"]


def test_the_search_wrapper_refuses_rather_than_blocking_on_stdin(root, shimmed):
    """No arguments is not "search for nothing" — it is the hook's payload
    mode, where the file blocks on stdin for a JSON payload that is never
    coming. An agent that ran it bare would hang until something killed it,
    with no output to explain why."""
    out = _run(root / "bin" / "memkit-recall", env=shimmed())
    assert out.returncode == 2
    assert "--search" in out.stderr
    assert not shimmed.out.exists(), "the interpreter should not have been reached"


# --- the dispatcher's checker route ------------------------------------------


def test_the_dispatcher_runs_the_package_from_this_tree(root, shimmed) -> None:
    """PREPENDED, not assigned — and the difference is only visible when there
    is something to prepend to.

    With no `PYTHONPATH` in the environment the two are the same string, so the
    assertion held for a wrapper that had stopped preserving what the caller
    set. An adopter with a pip-installed memkit of another version is exactly
    who that costs: their `PYTHONPATH` disappearing takes their own packages
    with it.
    """
    inherited = "/opt/an/adopters/own/packages"
    out = _run(root / "bin" / "memkit", "doctor", env=shimmed(PYTHONPATH=inherited))
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["argv"] == "-m memkit.cli doctor"
    assert seen["PYTHONPATH"] == f"{root / 'src'}:{inherited}"
    assert seen["MEMKIT_PLUGIN"] == "1"

    # And with nothing inherited it is just this tree, or the wrapper is
    # prepending to an empty string and leaving a stray separator.
    plain = _run(root / "bin" / "memkit", "doctor", env=shimmed())
    assert plain.returncode == 0, plain.stderr
    assert shimmed.read()["PYTHONPATH"] == str(root / "src")


def test_the_checker_route_is_python_when_one_meets_the_floor(
    root, tmp_path, shimmed
) -> None:
    env = shimmed()
    # A `python3` that claims 3.12 by exiting 0 for the floor probe, and
    # records for the shim contract otherwise.
    _shim(
        shimmed.dir, "python3",
        'case "$*" in *version_info*) exit 0 ;; esac\n' + SHIM_BODY,
    )
    out = _run(root / "bin" / "memkit", "doctor", env=env)
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["MEMKIT_CHECKER_ROUTE"] == "python"
    assert seen["MEMKIT_CHECKER_CMD"].endswith("-m memkit.memory_integrity")
    # THIS tree's checker, so the checker and the hook are one release.
    assert seen["MEMKIT_CHECKER_CMD"].startswith(str(shimmed.dir))


def test_the_checker_route_falls_back_to_uvx_below_the_floor(
    root, tmp_path, shimmed
) -> None:
    """The stock-python mac in the success criteria: 3.9.6 everywhere, and a
    checker that hard-refuses below 3.12. uvx provisions its own interpreter,
    which is what makes that machine able to run the checker at all."""
    env = shimmed()
    _shim(
        shimmed.dir, "python3",
        'case "$*" in *version_info*) exit 1 ;; esac\n' + SHIM_BODY,
    )
    _shim(shimmed.dir, "uvx", "exit 0")
    out = _run(root / "bin" / "memkit", "doctor", env=env)
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["MEMKIT_CHECKER_ROUTE"] == "uvx"
    assert "git+https://github.com/ak2k/memkit" in seen["MEMKIT_CHECKER_CMD"]


def test_no_checker_route_is_a_state_the_dispatcher_reports_rather_than_dies_on(
    root, tmp_path, shimmed
) -> None:
    """`none` must not be fatal HERE. Refusing at this level would take out
    `--help` and diagnosis — the two things an adopter in that state most needs
    — on behalf of a subcommand they did not run. The operation that cannot
    proceed without a checker is the one that refuses by name.
    """
    env = shimmed()
    _shim(
        shimmed.dir, "python3",
        'case "$*" in *version_info*) exit 1 ;; esac\n' + SHIM_BODY,
    )
    out = _run(root / "bin" / "memkit", "doctor", env=env)
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["MEMKIT_CHECKER_ROUTE"] == "none"
    assert seen["MEMKIT_CHECKER_CMD"] == ""


def test_the_dispatcher_refuses_by_name_when_nothing_can_run_it(
    root, tmp_path
) -> None:
    """Unlike the hook wrapper, this one exits non-zero: there is no prompt to
    get out of the way of, and a CLI that reports success while running nothing
    is the false green the whole observation surface exists to prevent. Not 2,
    which means "you invoked this wrongly" — an agent reading that retries with
    different arguments against a machine that cannot run memkit at all.
    """
    from memkit.cli import EXIT_NO_RUNTIME, EXIT_NOT_IN_BUILD, EXIT_USAGE

    out = _run(root / "bin" / "memkit", "doctor", env={"PATH": str(tmp_path / "none")})
    assert out.returncode == EXIT_NO_RUNTIME
    assert "no python3" in out.stderr

    # Every non-zero code this wrapper can produce, against the table an agent
    # reads. A shell script is the one place a new exit code can appear with
    # nothing to look it up in, and the two codes it must never borrow are
    # already spoken for by the dispatcher it fronts. Through the same
    # default-deny helper as the other two wrappers, so this copy cannot go on
    # believing a narrower regex.
    codes = _exit_literals("memkit") - {0}
    assert codes == {EXIT_NO_RUNTIME}, codes
    assert EXIT_NO_RUNTIME not in (EXIT_USAGE, EXIT_NOT_IN_BUILD)


# --- the commands this channel tells an agent to run -------------------------

# A token that could be a command an agent types. Path-shaped and
# dotted-filename tokens are dropped on purpose: `~/.cache/memory-recall/` is a
# directory this channel really does use and `memkit.json` is a file, and
# neither is something anybody runs.
# A token that could be a command an agent types. Every command any channel
# ships is either the bare word `memkit` or a hyphenated `mem…-…`, so requiring
# that shape drops the English words this text is full of (`memory`,
# `memories`) STRUCTURALLY rather than by naming them. Path-shaped and
# dotted-filename tokens are dropped too: `~/.cache/memory-recall/` is a
# directory this channel really uses and `memkit.json` is a file.
COMMANDISH = re.compile(r"(?<![\w./-])(memkit|mem[a-z0-9]+-[a-z0-9-]+)(?![\w./-])")

# The one hyphenated `mem…` token that is not a command, DERIVED rather than
# listed: it is the frame's XML tag, and the emitter is where that fact lives.
#
# Derived because a hand-kept exception list is the cheapest way to silence a
# real failure here — demonstrated: making the dispatcher advertise
# `memkit-init` turns the case below red, and one line added to a list turns it
# green with the bad advice still printed. There is nothing to add a line to
# now, and the equality below says so out loud.
NOT_A_COMMAND = {hook.FRAME_TAG}


def _corpus(tmp_path: Path, **extra) -> Path:
    """A store with more matching memories than one pointer block can carry, so
    the truncation notice — the one actionable line memkit emits — is rendered.
    """
    corpus = tmp_path / "store" / "search"
    corpus.mkdir(parents=True)
    for n in range(hook.MAX_HITS + 3):
        (corpus / f"flange_torque_{n}.md").write_text(
            f"---\ndescription: Flange fastener {n} tightens in a star pattern, "
            "in three passes, to the torque the table gives.\ntype: reference\n"
            f"---\n\n# Flange torque {n}\n\nThree passes, star pattern.\n"
        )
    return _config_file(
        tmp_path / "memkit.json",
        roots={"home": {"kind": "path", "path": str(tmp_path)}},
        stores=[{"id": "s", "role": "personal", "dir": "store", "live_root": "home"}],
        **extra,
    )


def _surfaces(
    root: Path, tmp_path: Path, config: Path | None, broken: Path
) -> dict[str, str]:
    """Every surface this channel renders, driven as the agent would reach it.

    Real processes through the real wrappers, with a real interpreter: what is
    under test is the name a plugin adopter is handed, and the wrappers are what
    make this channel a channel at all.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        # `None` is the state between install and init: the option names a
        # path that is not there yet, which is what an adopter's install
        # command produces before anything has written the file.
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config or tmp_path / "not-yet.json"),
    )
    env.pop("MEMKIT_CONFIG", None)
    recall = root / "bin" / "memkit-recall"
    dispatcher = root / "bin" / "memkit"
    query = "flange fastener tightening star pattern passes torque"

    def run(wrapper: Path, *args: str, **extra: str) -> str:
        out = subprocess.run(
            [str(wrapper), *args], capture_output=True, text=True, timeout=120,
            env={**env, **extra}, stdin=subprocess.DEVNULL, cwd=str(tmp_path),
        )
        return out.stdout + out.stderr

    # The truncation notice lives on the HOOK path, not the search CLI: the CLI
    # returns everything it found, and the notice exists because the block a
    # prompt gets is capped.
    hook_out = subprocess.run(
        [str(root / "bin" / "memkit-hook")],
        input=json.dumps({"session_id": "surfaces", "prompt": query}),
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path),
    )

    return {
        # The only line in an injected block that tells the agent to do
        # something.
        "truncation": hook_out.stdout + hook_out.stderr,
        "help": run(recall, "--help"),
        "usage error": run(recall, "--nope"),
        "debug-config": run(recall, "--debug-config"),
        "config error": run(
            recall, "--search", query,
            CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(broken),
        ),
        "inert": run(
            recall, "--search", query,
            CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(tmp_path / "absent.json"),
        ),
        "dispatcher help": run(dispatcher, "--help"),
        "dispatcher refusal": run(dispatcher, "doctor"),
    }


def test_every_command_this_channel_prints_is_one_it_ships(root, tmp_path) -> None:
    """The invariant, over the SET of surfaces rather than over the one that
    was wrong.

    A command memkit prints as a next step has to resolve on the caller's PATH
    and has to resolve to THIS install. A plugin install ships no
    `memory-recall`, so an agent following that advice got exit 127 — and on a
    machine that also has a pip or nix memkit it got the other install's
    stores, which is the collision the distinct name exists to prevent.

    Scraped rather than compared against a list of strings, so a surface that
    starts printing a command name later is covered without an edit here.
    """
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    shipped = {
        entry.name
        for entry in (root / "bin").iterdir()
        if entry.is_file() and os.access(entry, os.X_OK)
    }
    assert "memkit-recall" in shipped, shipped
    # The exception set is exactly the frame tag, and nothing may be added to
    # it by hand: an allowlist is the cheapest way to silence a real failure
    # here, and the case would then go on passing by no longer looking at the
    # one thing it is for. Anything else that needs excusing is a defect in the
    # scrape's SHAPE, which is a change somebody has to argue for.
    assert {hook.FRAME_TAG} == NOT_A_COMMAND, NOT_A_COMMAND
    assert NOT_A_COMMAND.isdisjoint(shipped | {hook.SEARCH_BINARY}), NOT_A_COMMAND

    # Three config states, because they reach the name through three different
    # routes and a fix can cover one without the others: a config that omits
    # `search_cli` takes the default applied in `Config.__init__`, one that
    # names it takes the field itself — this is the value the README's own
    # worked example produces — and NO CONFIG AT ALL takes the default applied
    # where the config would have been read.
    #
    # The third is the state the original defect was reported in — a freshly
    # installed plugin, before init — and leaving it out let a channel-aware
    # fix applied at one of the two application points pass: with the override
    # written `if _plugin_install() and cfg is not None`, an unconfigured
    # plugin goes back to advertising the binary it does not ship and the whole
    # suite stays green.
    # A config that raises something `load_config` does NOT convert to
    # ConfigError: `json.load` on a deeply nested document raises
    # RecursionError, and `_config()` catches only ConfigError. That is the
    # state the dispatcher's `except` fallback is reached in, and it was
    # handing plugin adopters the binary their channel does not ship.
    raising = tmp_path / "raising.json"
    raising.parent.mkdir(parents=True, exist_ok=True)
    raising.write_text("[" * 200_000 + "]" * 200_000)
    states = {
        "omitted": _corpus(tmp_path / "omitted"),
        "named": _corpus(tmp_path / "named", search_cli="memory-recall --search"),
        "absent": None,
        "raising outside ConfigError": raising,
    }
    for state, config in states.items():
        surfaces = _surfaces(root, tmp_path / state.split()[0], config, broken)
        named: set[str] = set()
        for surface, text in surfaces.items():
            found = set(COMMANDISH.findall(text))
            assert found <= shipped | NOT_A_COMMAND, (
                state, surface, sorted(found - shipped - NOT_A_COMMAND), text
            )
            named |= found
        # Anti-vacuity at the STATE rather than at each surface: with no
        # config the hook is inert by construction and `--debug-config` reports
        # routes rather than commands, so two surfaces legitimately name none —
        # but a state in which NOTHING names a command is a state this case is
        # not measuring.
        assert named & shipped, (state, sorted(named))
        # The truncation notice specifically: the line whose whole purpose is
        # to be run, and the one a config value used to be able to break. With
        # no config there is no corpus to truncate and no path to name, so what
        # that state pins instead is the dispatcher's fallback — which is where
        # a pre-init adopter actually meets a command name.
        if config is not None and config.name != "raising.json":
            assert "memkit-recall --config " in surfaces["truncation"], state
        else:
            # PRE-INIT, and the command must still name `--config`. A bare
            # `memkit-recall --search` answers `inert`, exit 3, in the shell
            # the dispatcher runs in — and the refusal beside it says exit 3
            # means "no config", which is the one conclusion the `--config`
            # interpolation exists to prevent. There is no path to fill in
            # yet, so it carries the placeholder the README uses.
            refusal = surfaces["dispatcher refusal"]
            assert "memkit-recall" in refusal, refusal
            if config is None:
                # The pre-init state specifically. A config that RAISES cannot
                # know a path to name — `_meanwhile`'s fallback is reached
                # precisely because resolving it failed — so the placeholder is
                # claimed only where there is a path the adopter could supply.
                assert "memkit-recall --config " in refusal, refusal
                assert "memkitConfig" in refusal, refusal
                bare = f"`{hook.PLUGIN_SEARCH_BINARY} --search"
                assert bare not in refusal, refusal


def test_the_advertised_command_runs_from_the_agents_bash_tool(
    root, tmp_path
) -> None:
    """The invariant's second clause, on the channel the command is typed into.

    MEASURED in a live session: a Bash-tool process gets the plugin's `bin/` on
    PATH and NONE of `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA` or
    `CLAUDE_PLUGIN_OPTION_*` — four plugin bin directories were on PATH and no
    plugin variable was set. Both surviving config rungs are plugin env, so a
    bare `memkit-recall --search` there resolves nothing and answers `inert`,
    telling the agent a serving installation is unconfigured. That is the one
    conclusion exit 3 exists to prevent.

    So the command is taken out of the block the hook injected and RUN, in that
    environment, rather than compared to a string. Quoting is part of the
    claim: a config path can contain a space, and a command an agent cannot
    paste is not a command.
    """
    # A space, because quoting is part of the claim — and an INVISIBLE
    # codepoint, because the emission pass strips those: a path quoted here
    # and rewritten there names a file that does not exist, which is a command
    # worse than none because it looks runnable.
    corpus = tmp_path / "spaced dir"
    config = _corpus(corpus)
    hidden = _corpus(tmp_path / "hidden\u200bdir")
    bare = _run(
        root / "bin" / "memkit-recall", "--debug-config",
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path / "home"),
            "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(hidden),
        },
    )
    assert bare.returncode == 0, bare.stderr
    advertised_for_hidden = [
        x for x in bare.stdout.splitlines() if x.startswith("search_cli:")
    ]
    assert advertised_for_hidden == ["search_cli: memkit-recall --search"], (
        advertised_for_hidden
    )
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config),
    )
    env.pop("MEMKIT_CONFIG", None)
    query = "flange fastener tightening star pattern passes torque"
    injected = subprocess.run(
        [str(root / "bin" / "memkit-hook")],
        input=json.dumps({"session_id": "bashtool", "prompt": query}),
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path),
    )
    assert injected.returncode == 0, injected.stderr
    # The notice line specifically, found by the reserved prefix that makes it
    # memkit's own — a retrieved description cannot start a line, which is what
    # the frame's carve-out rests on.
    advertised = [
        line.split("search: ", 1)[1]
        for line in injected.stdout.splitlines()
        if line.startswith(hook.NOTICE_PREFIX) and "search: " in line
    ]
    assert advertised, injected.stdout

    # The Bash tool's shape: the plugin's bin on PATH, no plugin environment.
    bash_tool = {
        "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
    }
    assert not [k for k in bash_tool if k.startswith("CLAUDE_PLUGIN_")]
    out = subprocess.run(
        shlex.split(advertised[0]), capture_output=True, text=True, timeout=120,
        env=bash_tool, cwd=str(tmp_path), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == hook.EXIT_OK, (out.returncode, out.stderr, advertised[0])
    assert "flange_torque_" in out.stdout, out.stdout

    # And EVERY backticked command the dispatcher's two surfaces hand out —
    # not the one that was fixed. These are the first things an agent probing a
    # fresh install touches, and each of them is a command it will paste.
    for args in (("doctor",), ("--help",)):
        surface = _run(
            root / "bin" / "memkit", *args, env={**env, "PATH": os.environ["PATH"]}
        )
        printed = re.findall(r"`([^`]+)`", surface.stdout + surface.stderr)
        commands = [c for c in printed if c.split()[0].startswith("memkit")]
        assert commands, (args, surface.stdout, surface.stderr)
        for command in commands:
            runnable = command.replace('"<terms>"', '"flange torque"')
            probed = subprocess.run(
                shlex.split(runnable), capture_output=True, text=True, timeout=120,
                env=bash_tool, cwd=str(tmp_path), stdin=subprocess.DEVNULL,
            )
            assert probed.returncode == hook.EXIT_OK, (
                args, runnable, probed.returncode, probed.stdout, probed.stderr
            )


def test_the_scrape_can_see_a_command_this_channel_does_not_ship(tmp_path) -> None:
    """The control for the case above, which would otherwise pass by finding
    nothing. Off the plugin channel the same surfaces name `memory-recall` —
    correctly, since pip and nix install that console script — and the scrape
    sees it."""
    out = subprocess.run(
        ["python3", hook.__file__, "--help"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert "memory-recall" in set(COMMANDISH.findall(out.stdout + out.stderr))


# --- what the inert message says a config can arrive by ----------------------

# One phrase per rung `memkit_resolve_config` really tries. The mapping is the
# only handwritten link in the chain: the rungs are scraped from the shell and
# the phrases are read out of the module, so the two ends cannot be edited into
# agreement through this table without someone editing this table too.
# One phrase per CANDIDATE PATH `memkit_resolve_config` really builds, keyed on
# the shell expression that builds it — and the scrape below additionally pins
# that `_candidate` is the function's only sink, so a rung that skipped the
# variable entirely cannot serve a config unseen.
#
# The mapping is the only handwritten link in the chain: the expressions are
# scraped from the shell and the phrases are read out of the module.
ROUTE_FOR_RUNG = {
    '$(memkit_expand_home "$CLAUDE_PLUGIN_OPTION_MEMKITCONFIG")':
        "the `memkitConfig` install option",
    '$(memkit_expand_home "$CLAUDE_PLUGIN_DATA")/memkit.json':
        "$CLAUDE_PLUGIN_DATA/memkit.json",
}


def _resolver_rungs() -> set[str]:
    """Every candidate path `memkit_resolve_config` tests, as written.

    `_candidate=` is the resolver's one shape for "a path this rung might
    serve": each rung assigns it and then `[ -f ]`s it. Anything assigned there
    and not in ROUTE_FOR_RUNG is an admission route nobody has classified, and
    the message that enumerates the routes is stale the moment one appears.
    """
    text = COMMON_SH.read_text(encoding="utf-8")
    match = re.search(r"^memkit_resolve_config\(\) \{$(.*?)^\}$", text, re.S | re.M)
    assert match, "memkit_resolve_config moved — this pin cannot see it"
    body = match.group(1)
    # Every value the function can PRINT is a route it serves, and the scrape
    # has to start there rather than at the assignments: a rung written
    # `if [ -f "$HOME/.memkit.json" ]; then printf '%s\\n' "$HOME/.memkit.json";
    # return 0; fi` assigns nothing and is served all the same. Measured — the
    # suite stayed green with exactly that rung in place.
    #
    # So `_candidate` must be the only sink, and then classifying the
    # assignments classifies the routes.
    # EQUALITY, not subset, which is what anchors this on a non-empty match:
    # the regex is keyed on one spelling, so a rung written with `echo` printed
    # nothing this could see and the subset held vacuously. A rung spelled
    # `if [ -f "$HOME/.memkit.json" ]; then echo "$HOME/.memkit.json"; fi` is a
    # live admission route reachable from any home directory, and it left the
    # whole file green.
    # Statements redirected to STDERR are not sinks: the resolver's answer is
    # what it writes to stdout, and a refusal message written the same way is
    # not an admission route. Removed before scraping, continuations included.
    to_stdout = re.sub(r"printf(?:\\\n|[^\n])*?>&2", "", body)
    printed = set(re.findall(r"""printf\s+'%s\\n'\s+(\S+)""", to_stdout))
    assert printed == {'"$_candidate"'}, sorted(printed)
    # And no OTHER way of writing to stdout, since the equality above only
    # constrains the spelling it can see.
    for other in ("echo ", "printf '%s'", "cat ", "tee ", ">&1"):
        assert other not in to_stdout, (other, to_stdout)
    # An assignment counts wherever the line puts it — a rung written
    # `… || _candidate=<expr>` is a rung.
    candidates = set(
        re.findall(r"^\s*(?:\|\||&&)?\s*_candidate=(\S.*?)\s*$", body, re.M)
    )
    # An empty assignment is the rejection arm of the absoluteness guard, not a
    # route: `_candidate=""` is how a non-absolute value is dropped.
    return {c for c in candidates if c not in ('""', "''")}


def test_the_inert_message_names_the_rungs_the_resolver_actually_tries() -> None:
    """The rungs live in POSIX sh and the sentence that describes them lives in
    Python, with nothing between them.

    A rung deleted there used to leave a confident sentence here — telling an
    agent to configure an install through a route the code no longer has, on
    the one surface whose whole job is to say why nothing is happening. Set
    equality in both directions, so a rung added is as red as a rung removed.
    """
    rungs = _resolver_rungs()
    assert rungs == set(ROUTE_FOR_RUNG), (sorted(rungs), sorted(ROUTE_FOR_RUNG))
    # The basename is part of the route, not decoration: the message tells an
    # adopter which FILE to create.
    assert "memkit.json" in "".join(rungs), rungs
    expected = {"--config PATH"} | {ROUTE_FOR_RUNG[rung] for rung in rungs}
    assert set(hook.PLUGIN_CONFIG_ROUTES) == expected, hook.PLUGIN_CONFIG_ROUTES


# Phrases that described the config rung this repo deleted. A tombstone rather
# than a derivation: the rung is gone from the shell, so nothing can scrape it
# out of the resolver, and the only way a document naming it goes red is a list
# of the words it was described with. Both documents that enumerate routes
# carried one of these after the code stopped honouring it.
RETIRED_ROUTE_PHRASES = (
    "beside the plugin",
    "one beside",
    "in the plugin's own directory",
    "memkit.json beside",
)

ROUTE_DOCS = ("README.md", "docs/ROLLOUT.md")


def test_the_soak_logs_growth_rule_is_published_where_a_consumer_reads_it() -> None:
    """The outcome vocabulary is now a cross-repo contract: another
    repository's analyzers compute injection rates from `log.jsonl`, and its
    suite asserts every outcome memkit can emit has been classified.

    So the rule a consumer codes against has to be in writing, not implied by
    the producer's behaviour — and the discriminator in particular, since
    without it the only way to exclude a non-prompt record from a per-prompt
    population is to learn each new outcome's NAME, which is the coupling the
    static enumeration exists to remove.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert '"concludes": false' in readme, "the discriminator is undocumented"
    assert "grows without a version bump" in readme
    assert "prompt_sha" in readme


def test_the_rollout_runbook_verifies_both_channels() -> None:
    """The per-host checks read `~/.claude/hooks`, a `/nix/store` symlink and a
    consumer checkout — none of which a plugin install has — so on a host that
    installed memkit as a plugin every one of them fails or passes vacuously.

    That is the silent-failure mode the runbook's own opening says it exists to
    prevent, and README sends every second-host adopter there with no channel
    caveat.
    """
    rollout = (REPO / "docs" / "ROLLOUT.md").read_text(encoding="utf-8")
    assert "## Per-host verify, plugin channel" in rollout
    # The commands that block was written from, each run against a real
    # install before it was written down.
    for command in (
        "claude plugin list",
        "claude plugin details memkit@memkit",
        "pluginConfigs",
        "--debug-config",
    ):
        assert command in rollout, command
    # And the nix block says which channel it is for, so a plugin adopter does
    # not work through four checks that cannot apply.
    assert "nix-channel checks" in rollout


def test_no_document_still_offers_the_config_route_the_code_dropped() -> None:
    """An operator who follows a runbook naming a deleted route drops a
    `memkit.json` into the payload root and gets a plugin that installs,
    reports enabled and serves nothing — with no error anywhere, which is the
    silent failure the runbook exists to prevent.

    Both documents that enumerate the routes are checked, because the round
    that deleted the rung rewrote one of them and missed the other, leaving the
    repo shipping two answers to "which paths will an every-prompt hook read".
    """
    for name in ROUTE_DOCS:
        text = (REPO / name).read_text(encoding="utf-8").lower()
        for phrase in RETIRED_ROUTE_PHRASES:
            assert phrase not in text, f"{name} still offers: {phrase!r}"
        # And each still names the rung that IS there, or the tombstone above
        # would pass on a document that stopped describing routes at all.
        assert "$claude_plugin_data" in text or "plugin's own data directory" in text, name


def test_both_channels_inert_messages_name_only_their_own_routes(
    root, tmp_path
) -> None:
    """Both branches, explicitly, because the suite runs under one of them.

    With `MEMKIT_PLUGIN` unset the plugin wording is the untested branch and
    vice versa, and the defect this replaced was exactly a message that was
    right for the channel the tests happened to run on: a plugin install told
    an agent to set `$MEMKIT_CONFIG`, which both wrappers strip before the hook
    sees it, so following the advice measurably changed nothing.
    """
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    through_plugin = subprocess.run(
        [str(root / "bin" / "memkit-recall"), "--search", "flange torque"],
        capture_output=True, text=True, timeout=120, env=env,
        stdin=subprocess.DEVNULL,
    )
    assert through_plugin.returncode == hook.EXIT_INERT, through_plugin.stderr
    assert hook.CONFIG_ENV not in through_plugin.stderr, through_plugin.stderr
    for route in hook.PLUGIN_CONFIG_ROUTES:
        assert route in through_plugin.stderr, route

    direct = subprocess.run(
        ["python3", hook.__file__, "--search", "flange torque"],
        capture_output=True, text=True, timeout=120, env=env,
        stdin=subprocess.DEVNULL,
    )
    assert direct.returncode == hook.EXIT_INERT, direct.stderr
    assert "CLAUDE_PLUGIN" not in direct.stderr, direct.stderr
    for route in hook.CONFIG_ROUTES:
        assert route in direct.stderr, route

    # And `--help`, which is the cheapest probe an agent runs and therefore the
    # first place it learns what to try. It told a plugin adopter the config
    # default was `$MEMKIT_CONFIG` and, on `--dir`, to unset it — a variable
    # both wrappers strip, so the first claim is false and following the second
    # changes nothing.
    plugin_help = _run(root / "bin" / "memkit-recall", "--help", env=env)
    assert plugin_help.returncode == 0, plugin_help.stderr
    assert hook.CONFIG_ENV not in plugin_help.stdout, plugin_help.stdout
    for route in hook.PLUGIN_CONFIG_ROUTES:
        assert route in plugin_help.stdout, route
    direct_help = subprocess.run(
        ["python3", hook.__file__, "--help"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert f"${hook.CONFIG_ENV}" in direct_help.stdout, direct_help.stdout


def test_the_help_epilog_carries_every_exit_code_this_binary_can_produce(
    root, tmp_path
) -> None:
    """The epilog's own comment says it is built from the constants so the help
    and the README cannot drift from what the code returns. It was complete
    before the start-failure code existed and stopped being complete when it
    landed — an agent meeting an undocumented 4 falls back to the nearest
    neighbour or to shell convention, and both readings are wrong in the unsafe
    direction.

    Over the CONSTANTS, so the next code added is covered without an edit here.
    """
    real = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    rendered = _run(root / "bin" / "memkit-recall", "--help", env=real).stdout
    # OVER the constants, which is what the epilog's own comment claims of
    # itself — a hand-written list of five is the drift it says it prevents,
    # and it is what let the code this round added go unlisted.
    codes = {
        value
        for name, value in vars(hook).items()
        if name.startswith("EXIT_") and isinstance(value, int)
    }
    assert len(codes) >= 5, codes
    listed = {int(m) for m in re.findall(r"^  (\d+)  ", rendered, re.MULTILINE)}
    assert listed == codes, (listed, codes)
    # And the collision with the dispatcher's table is stated on both sides,
    # in both directions: the two tables swap 1 and 4, and 1 is the dangerous
    # one — on this table it means "nothing matched", which tells an agent to
    # stop looking.
    from memkit.cli import EXIT_NO_RUNTIME, EXIT_NOT_IN_BUILD

    assert "dispatcher's table is its own" in rendered, rendered
    dispatcher = _run(root / "bin" / "memkit", "--help", env=real).stdout
    assert "swaps these two" in dispatcher, dispatcher
    assert EXIT_NO_RUNTIME == hook.EXIT_NO_MATCH
    assert EXIT_NOT_IN_BUILD == hook.EXIT_CANNOT_START


def test_debug_config_says_when_it_overrode_the_field_it_is_labelled_with(
    root, tmp_path
) -> None:
    """`--debug-config` is the command the README and the rollout runbook both
    name as *the* verification surface, and every line of it reports the file —
    except this one, whose label is the config key verbatim and whose value is
    not from the config. An operator cannot tell the two apart.
    """
    real = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    overridden = _corpus(tmp_path / "over", search_cli="memory-recall --search")
    out = _run(
        root / "bin" / "memkit-recall", "--debug-config",
        env={**real, "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(overridden)},
    )
    assert out.returncode == 0, out.stderr
    assert "! the config's own `search_cli` is not in effect" in out.stdout, out.stdout

    # And NOT where the config never declared the field: there is no value to
    # have been overridden, and the line asserts something false about "the
    # name it records". Measured byte-identical on two configs differing only
    # in whether the key is present.
    undeclared = _corpus(tmp_path / "undeclared")
    silent = _run(
        root / "bin" / "memkit-recall", "--debug-config",
        env={**real, "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(undeclared)},
    )
    assert silent.returncode == 0, silent.stderr
    assert "! the config's own" not in silent.stdout, silent.stdout

    # And no divergence line where there is no divergence, or the note is
    # decoration rather than a report. Off the plugin channel the field IS the
    # advertised command — which is also why the note can never be silent on
    # the plugin channel: the `--config <path>` prefix that makes the command
    # runnable in the agent's Bash tool is not something a config file can
    # carry, so the two always differ there.
    same = subprocess.run(
        ["python3", hook.__file__, "--debug-config"],
        capture_output=True, text=True, timeout=120,
        env={**real, "MEMKIT_CONFIG": str(overridden)},
    )
    assert same.returncode == 0, same.stderr
    assert "search_cli: memory-recall --search" in same.stdout, same.stdout
    assert "! the config's own" not in same.stdout, same.stdout


# --- the hook file the wrapper actually runs ---------------------------------


def test_the_wrapper_execs_the_byte_identical_hook(root, tmp_path) -> None:
    """`_VERSION` is a sha256 of the hook's own bytes and is stamped on every
    soak record, so it is what makes records comparable across install
    channels. A wrapper that copied, patched or wrapped the file — to bake a
    config in, say — would fork the log into halves that no analyzer can join,
    silently: the field would still be there, still eight hex characters.
    """
    corpus = tmp_path / "store" / "search"
    corpus.mkdir(parents=True)
    (corpus / "flange_torque.md").write_text(
        "---\ndescription: Flange fasteners tighten in a star pattern, in three "
        "passes.\ntype: reference\n---\n\n# Flange torque\n\nThree passes.\n"
    )
    config = _config_file(
        tmp_path / "memkit.json",
        roots={"home": {"kind": "path", "path": str(tmp_path)}},
        stores=[{"id": "s", "role": "personal", "dir": "store", "live_root": "home"}],
    )
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config),
    )
    env.pop("MEMKIT_CONFIG", None)
    payload = json.dumps(
        {"session_id": "wrapv", "prompt": "flange fastener tightening passes"}
    )

    through_wrapper = subprocess.run(
        [str(root / "bin" / "memkit-hook")], input=payload, capture_output=True,
        text=True, timeout=60, env=env,
    )
    assert through_wrapper.returncode == 0
    # A pointer, not an exit code: an inert hook and a wired one both exit 0
    # and print nothing on a prompt with no answer.
    assert "flange_torque.md" in through_wrapper.stdout, through_wrapper.stdout

    direct = subprocess.run(
        ["python3", hook.__file__], input=payload, capture_output=True, text=True,
        timeout=60, env=dict(env, MEMKIT_CONFIG=str(config), session="x"),
    )
    assert direct.returncode == 0

    records = [
        json.loads(line)
        for line in (tmp_path / ".cache" / "memory-recall" / "log.jsonl")
        .read_text()
        .splitlines()
    ]
    versions = {r["v"] for r in records}
    assert len(versions) == 1, versions
    assert versions != {"?"}, "the hook could not read itself — the pin is vacuous"


def test_the_plugin_marker_is_absent_without_the_wrapper(tmp_path) -> None:
    """R6's non-degradation premise, from the other side: nothing about a
    plugin install may reach a nix or pip one. The marker is exported by the
    wrapper and by nothing else, so a hook run any other way cannot take a
    plugin-only branch.
    """
    out = subprocess.run(
        ["python3", hook.__file__, "--debug-config"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert out.returncode == hook.EXIT_INERT
    assert "MEMKIT_PLUGIN" not in out.stdout
