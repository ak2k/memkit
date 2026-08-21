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
]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    option is unset (measured), so rung 3 rather than this string is what
    covers that case.
    """
    option = _json(PLUGIN_MANIFEST)["userConfig"]["memkitConfig"]
    assert option["required"] is True
    assert option["type"] == "string"
    for field in ("title", "description"):
        assert option[field].strip(), field
    assert "init" in option["description"]


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


def test_the_pinned_sha_is_a_commit_in_this_history() -> None:
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("no git checkout here — the plain-python CI leg is where this runs")
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    out = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, f"{sha} is not a commit in this repository"


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


def test_every_payload_file_is_tracked() -> None:
    """A github install is a clone. An untracked wrapper works perfectly on the
    machine it was written on and is missing for every adopter — and the
    failure it produces there is the wrapper's own "payload is incomplete"
    refusal, i.e. a plugin that installs and never speaks again.
    """
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("no git checkout here — the plain-python CI leg is where this runs")
    out = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *PAYLOAD],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr


def test_the_wrappers_are_executable_in_the_index() -> None:
    """Mode 100755 in git, not merely on this filesystem. A clone restores the
    executable bit from the index, and a wrapper checked in as 644 is a hook
    the harness cannot run at all."""
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("no git checkout here — the plain-python CI leg is where this runs")
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
        # Exec form, so the command is never re-parsed by a shell: a plugin
        # path containing a space is otherwise two arguments.
        assert "args" in handler, (event, "declare args: [] rather than omitting it")
        assert handler["type"] == "command"
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


@pytest.fixture
def shimmed(tmp_path: Path):
    """A PATH holding one fake `python3`, and a reader for what it saw."""
    shim_dir = tmp_path / "shimbin"
    out = tmp_path / "shim-out.txt"
    _shim(shim_dir, "python3", SHIM_BODY)

    def build(**extra) -> dict:
        env = {
            "PATH": f"{shim_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "SHIM_OUT": str(out),
        }
        env.update(extra)
        return env

    def read() -> dict:
        return dict(
            line.split("=", 1) for line in out.read_text().splitlines() if "=" in line
        )

    build.read = read  # type: ignore[attr-defined]
    build.dir = shim_dir  # type: ignore[attr-defined]
    build.out = out  # type: ignore[attr-defined]
    return build


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


def test_rung_three_needs_no_environment_at_all(root, shimmed) -> None:
    """The rung that does not share the other two's failure mode. Both of those
    are plugin env exports into the hook process — one mechanism wearing two
    hats — and a script can always find itself."""
    config = _config_file(root / "memkit.json")
    env = shimmed()
    assert "CLAUDE_PLUGIN_ROOT" not in env and "CLAUDE_PLUGIN_DATA" not in env
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_the_rungs_are_tried_in_order(root, tmp_path, shimmed) -> None:
    option = _config_file(tmp_path / "one.json")
    data = tmp_path / "two"
    _config_file(data / "memkit.json")
    _config_file(root / "memkit.json")
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
        # `/memkit.json` as a WHOLE word — rung 3 legitimately builds
        # `<root>/memkit.json`, and the root is an absolute path, so only the
        # empty-expansion bug can put the bare root-level path in the trace.
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


def test_the_hook_wrapper_never_exits_non_zero(root, tmp_path, shimmed) -> None:
    """Every reachable refusal, in one place, because the property is about the
    SET of them: a new branch that exits 1 is invisible until it is the branch
    an adopter is on, and by then it is a message in front of every prompt.
    """
    broken = tmp_path / "brokenroot"
    (broken / "bin").mkdir(parents=True)
    shutil.copy(REPO / "bin" / "memkit-hook", broken / "bin" / "memkit-hook")

    cases = [
        # no interpreter anywhere
        (root / "bin" / "memkit-hook", {"PATH": str(tmp_path / "nothing")}, ()),
        # an incomplete payload: no library, and no hook file
        (broken / "bin" / "memkit-hook", shimmed(), ()),
        # arguments that should never arrive, and are ignored if they do
        (root / "bin" / "memkit-hook", shimmed(), ("--search", "x")),
    ]
    for wrapper, env, args in cases:
        out = _run(wrapper, *args, env={"HOME": str(tmp_path), **env})
        assert out.returncode == 0, (wrapper, args, out.returncode, out.stderr)


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
    that string."""
    config = _config_file(root / "memkit.json")
    doubled = f"{root}//bin/memkit-hook"
    out = subprocess.run(
        [doubled], capture_output=True, text=True, timeout=60,
        env=shimmed(), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["MEMKIT_CONFIG"] == str(config)
    assert "//" not in seen["argv"], seen["argv"]


def test_a_wrapper_invoked_by_name_from_the_path_still_finds_its_tree(
    root, tmp_path, shimmed
) -> None:
    """`bin/` is on the agent's PATH while the plugin is enabled, so
    `memkit-recall --search …` arrives with argv[0] of `memkit-recall` and no
    directory to walk up from."""
    config = _config_file(root / "memkit.json")
    env = shimmed()
    env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
    out = subprocess.run(
        ["memkit-recall", "--search", "flange torque"],
        capture_output=True, text=True, timeout=60, env=env,
        stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["MEMKIT_CONFIG"] == str(config)
    assert seen["argv"].endswith("memory_prompt_recall.py --search flange torque")


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
    out = _run(root / "bin" / "memkit", "doctor", env=shimmed())
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    assert seen["argv"] == "-m memkit.cli doctor"
    assert seen["PYTHONPATH"].split(":")[0] == str(root / "src")
    assert seen["MEMKIT_PLUGIN"] == "1"


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
    # already spoken for by the dispatcher it fronts.
    codes = {int(m) for m in re.findall(r"^\s*exit (\d+)$",
                                        (REPO / "bin" / "memkit").read_text(),
                                        re.MULTILINE)} - {0}
    assert codes == {EXIT_NO_RUNTIME}, codes
    assert EXIT_NO_RUNTIME not in (EXIT_USAGE, EXIT_NOT_IN_BUILD)


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
