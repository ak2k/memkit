"""Scenarios that run the real `claude` binary against a scratch profile.

Everything here is a claim about a harness this repo does not own, so nothing
here can be settled by reading memkit's own files. Two tiers:

  CLI tier — needs the binary and nothing else. Validation, marketplace add,
    install, and reading back what got registered. Runs in CI, which installs
    a pinned Claude Code for exactly this.
  LIVE tier — needs a model to answer a prompt, which means the author's local
    proxy. `MEMKIT_RIG_LIVE=1` to opt in. A scenario that silently skips
    everywhere but one machine is not a gate, so it says which it is.

The live tier is where the U2 verification lives: an install that yields a
hook which FIRES, asserted on a pointer rather than on an exit code, because
an inert hook and a wired one both exit 0 and print nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rig import (
    REPO,
    Profile,
    cli_tier_reason,
    live_tier_reason,
    stage_plugin,
)

cli_tier = pytest.mark.skipif(
    cli_tier_reason() is not None, reason=cli_tier_reason() or ""
)
live_tier = pytest.mark.skipif(
    live_tier_reason() is not None, reason=live_tier_reason() or ""
)

# A prompt whose terms the fixture corpus answers, and the memory it must
# reach. Taken from the fixture eval's own suite, so the retrieval claim here
# and the one the eval gates on are about the same pair.
PROMPT = "sprocket backlash after the gearbox rebuild"
EXPECTED = "sprocket_alignment.md"


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> Path:
    """The working tree, staged as a marketplace that serves it in place.

    Module-scoped: the copy is the expensive part and no scenario mutates it.
    """
    return stage_plugin(tmp_path_factory.mktemp("staged") / "memkit", REPO)


@pytest.fixture
def profile(tmp_path) -> Profile:
    return Profile(tmp_path / "rig")


def _fixture_config(profile: Profile) -> Path:
    """The invented two-store corpus, inside the profile's own HOME.

    Copied rather than pointed at: the hook writes an index and a `.build`
    record beside nothing, but the eval snapshot and the corpus itself belong
    to the repo and a scenario must not be able to touch them.
    """
    fixtures = profile.home / "fixtures"
    shutil.copytree(REPO / "tests" / "fixtures", fixtures)
    return fixtures / "memkit.json"


def _soak(profile: Profile) -> list[dict]:
    log = profile.home / ".cache" / "memory-recall" / "log.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


# --- CLI tier -----------------------------------------------------------------


@cli_tier
def test_validate_strict_passes_on_the_real_tree_in_both_modes(
    profile: Profile,
) -> None:
    """The admission gate, on the tree as committed rather than on a staged
    copy.

    Both invocations, because the validator picks its mode from what it is
    pointed at: the repo root validates the MARKETPLACE (and the plugin
    manifests it lists, for schema errors only), while the plugin manifest's
    own path is the one that raises the metadata warnings `--strict` fails on.
    A step that ran only the first passed a manifest with `author` deleted.

    The mode is asserted, not assumed — the two commands differ by an argument,
    and a validator that quietly resolved both to the same manifest would make
    this pair look like coverage it is not.
    """
    marketplace = profile.claude(
        "plugin", "validate", str(REPO), "--strict", check=False
    )
    assert marketplace.returncode == 0, f"{marketplace.stdout}\n{marketplace.stderr}"
    assert "Validating marketplace manifest" in marketplace.stdout

    plugin = profile.claude(
        "plugin", "validate", str(REPO / ".claude-plugin" / "plugin.json"),
        "--strict", check=False,
    )
    assert plugin.returncode == 0, f"{plugin.stdout}\n{plugin.stderr}"
    assert "Validating plugin manifest" in plugin.stdout


@cli_tier
def test_the_strict_gate_can_fail(profile: Profile, tmp_path) -> None:
    """A gate nobody has watched fail is a gate nobody has watched.

    Two breakages, one per mode, because that is exactly the distinction the
    step above rests on: a metadata warning is invisible to the marketplace
    invocation and fatal to the plugin one.
    """
    manifest = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    del manifest["author"]
    unattributed = tmp_path / "plugin.json"
    unattributed.write_text(json.dumps(manifest))
    out = profile.claude(
        "plugin", "validate", str(unattributed), "--strict", check=False
    )
    assert out.returncode == 1, out.stdout
    assert "author" in out.stdout

    broken = tmp_path / "broken" / ".claude-plugin"
    broken.mkdir(parents=True)
    manifest = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    manifest["userConfig"] = {"dotted.key": manifest["userConfig"]["memkitConfig"]}
    (broken / "plugin.json").write_text(json.dumps(manifest))
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    market["plugins"][0]["source"] = "./"
    (broken / "marketplace.json").write_text(json.dumps(market))
    out = profile.claude(
        "plugin", "validate", str(broken.parent), "--strict", check=False
    )
    assert out.returncode == 1, out.stdout
    assert "dotted.key" in out.stdout


@cli_tier
def test_a_local_marketplace_add_and_install_lands_the_option(
    profile: Profile, staged: Path
) -> None:
    """The cold-adopter path, minus the model: add, install, and supply the
    config option in the same non-interactive command an agent would run.

    The option's arrival is what the whole config-delivery design turns on, so
    it is read back out of the harness's own settings rather than assumed from
    a zero exit.
    """
    profile.marketplace_add(staged)
    config = _fixture_config(profile)
    out = profile.install("memkit@memkit", config={"memkitConfig": str(config)})
    assert "Successfully installed" in out.stdout, out.stdout

    installed = profile.installed()
    assert [p["id"] for p in installed] == ["memkit@memkit"]
    assert installed[0]["enabled"] is True

    settings = json.loads((profile.config_dir / "settings.json").read_text())
    assert settings["pluginConfigs"]["memkit@memkit"]["options"] == {
        "memkitConfig": str(config)
    }


@cli_tier
def test_installing_without_the_option_is_loud_but_not_fatal(
    profile: Profile, staged: Path
) -> None:
    """An adopter who forgets `--config` gets a plugin that is inert, not one
    that is broken — and is told so at install time.

    Both halves are the product: `required: true` is what produces the warning,
    and a *fatal* required option would be worse, since the config it names is
    created by `/memkit:init` and does not exist yet at install.
    """
    profile.marketplace_add(staged)
    out = profile.claude("plugin", "install", "memkit@memkit", "--yes")
    assert "Successfully installed" in out.stdout
    assert "not yet set" in out.stdout and "--config" in out.stdout
    assert profile.installed()[0]["enabled"] is True


@cli_tier
def test_the_hook_the_harness_would_run_is_the_wrapper_and_it_answers(
    profile: Profile, staged: Path
) -> None:
    """The registration's command, resolved and executed exactly as the
    harness would — same `${CLAUDE_PLUGIN_ROOT}` expansion, same zero
    arguments — without needing a model.

    It is the cheap half of the live scenario below, and it is the half that
    still runs in CI. A pointer, never an exit code.
    """
    profile.marketplace_add(staged)
    config = _fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})

    registration = json.loads((staged / "hooks" / "hooks.json").read_text())
    command = registration["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    # The harness expands this with a trailing slash on the root; reproduce
    # that rather than a tidied version of it.
    resolved = command.replace("${CLAUDE_PLUGIN_ROOT}", f"{staged}/")

    out = subprocess.run(
        [resolved],
        input=json.dumps({"session_id": "rig-cli", "prompt": PROMPT}),
        capture_output=True,
        text=True,
        timeout=120,
        env=profile.env(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)),
    )
    assert out.returncode == 0, out.stderr
    assert EXPECTED in out.stdout, out.stdout


# --- LIVE tier ----------------------------------------------------------------


@live_tier
def test_an_installed_plugin_injects_a_pointer_into_a_real_turn(
    profile: Profile, staged: Path
) -> None:
    """The U2 verification: marketplace add, install, ask a question, and the
    pointers arrive.

    Asserted twice over, because the two assertions fail in different ways. The
    soak record is what the hook believes it did; the transcript is what the
    model actually received. A hook that emitted into a closed pipe would
    satisfy neither, and a hook that never ran would leave no record at all.
    """
    profile.marketplace_add(staged)
    config = _fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})
    project = profile.project("work")

    out = profile.claude(
        "-p",
        f"{PROMPT} — name the memory file you were given, if any",
        "--output-format",
        "json",
        cwd=str(project),
        timeout=300,
    )
    answer = json.loads(out.stdout)
    assert answer["is_error"] is False, answer

    injected = [r for r in _soak(profile) if r["outcome"] == "injected"]
    assert injected, [r["outcome"] for r in _soak(profile)]
    assert EXPECTED in injected[-1]["injected"]
    assert EXPECTED in answer["result"], answer["result"]


@live_tier
def test_an_uninitialised_install_refuses_and_leaves_a_record(
    profile: Profile, staged: Path
) -> None:
    """The trust gate, in the harness rather than in a subprocess test.

    The turn has to complete normally — this is the state a cold adopter is in
    between accepting the trust dialog and running init, and a hook that
    interfered with the prompt there would be the worst possible first
    impression. What is new is that the refusal is no longer indistinguishable
    from silence.
    """
    profile.marketplace_add(staged)
    profile.claude("plugin", "install", "memkit@memkit", "--yes")
    project = profile.project("work")

    out = profile.claude(
        "-p", "reply with exactly: PONG", "--output-format", "json",
        cwd=str(project), timeout=300,
    )
    answer = json.loads(out.stdout)
    assert answer["is_error"] is False and "PONG" in answer["result"]

    marker = profile.config_dir / "plugins" / "data" / "memkit-memkit" / "trust.json"
    assert marker.is_file(), "the gate refused without recording it"
    records = json.loads(marker.read_text())["records"]
    assert [r["outcome"] for r in records] == ["trust:unconfigured"]
    # And nothing in the shared state dir: an uninitialised install has not
    # been consented to and does not get to create it.
    assert not (profile.home / ".cache" / "memory-recall").exists()


@live_tier
def test_a_plugin_beside_a_settings_registration_is_detected_at_runtime(
    profile: Profile, staged: Path
) -> None:
    """R6 on a real machine: the author's own fleet will install this plugin
    alongside its nix wiring, and both hooks then serve every prompt.

    The settings entry here stands in for the nix one — a second copy of the
    hook at a second path, registered the way a non-plugin install registers,
    pointing at the SAME config. That pair is the one a version stamp cannot
    see: `_VERSION` is a hash of the hook's bytes and both copies are the same
    release.
    """
    profile.marketplace_add(staged)
    config = _fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})

    # The "other" installation, at its own path.
    other = profile.home / "other-install"
    other.mkdir()
    for name in ("memory_prompt_recall.py", "common-words.txt"):
        shutil.copy(REPO / "src" / "memkit" / name, other / name)
    settings_path = profile.config_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings.setdefault("hooks", {}).setdefault("UserPromptSubmit", []).append(
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{other / 'memory_prompt_recall.py'}",
                    "timeout": 15,
                }
            ]
        }
    )
    settings_path.write_text(json.dumps(settings, indent=2))
    (other / "memory_prompt_recall.py").chmod(0o755)

    project = profile.project("work")
    for prompt in (
        PROMPT,
        "flange fastener tightening sequence and passes",
        "reconcile the ledger before the period closes",
    ):
        profile.claude(
            "-p", prompt, "--output-format", "json",
            cwd=str(project), timeout=300,
            extra_env={"MEMKIT_CONFIG": str(config)},
        )

    duplicates = [r for r in _soak(profile) if r["outcome"] == "dup-registration"]
    assert duplicates, [r["outcome"] for r in _soak(profile)]
    assert duplicates[0]["other_config"] == "memkit.json"


@live_tier
def test_the_plugin_bin_is_on_the_agents_path_and_loses_every_name_collision(
    profile: Profile, staged: Path
) -> None:
    """Both halves of KTD12's premise, measured rather than assumed.

    That plugin `bin/` reaches the Bash tool's PATH is documented and is what
    makes `memkit-recall` a command an agent can run at all. Its PRECEDENCE is
    not documented, and the answer turns out to be the unhelpful one: the
    plugin's directory is appended, so any name already on the adopter's PATH
    wins.

    That is why the two names that matter are collision-proof by construction.
    `memkit-recall` searching the wrong stores would be a wrong answer wearing
    a right one's clothes — the failure this naming exists to prevent — and no
    other tool ships that name.

    `memkit` is the exception, and it collides with memkit's OWN console
    script: an adopter with a pip or nix install gets that one, not the
    plugin's. It is why the skills must invoke it as
    `${CLAUDE_PLUGIN_ROOT}/bin/memkit` rather than bare, and this case is here
    to fail if the precedence ever changes in either direction.
    """
    profile.marketplace_add(staged)
    config = _fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})
    project = profile.project("work")

    # A decoy of memkit's own console-script name, ahead of anything the
    # harness appends — the shape of the collision an adopter with a pip or
    # nix install already has.
    decoy_dir = profile.home / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "memkit").write_text("#!/bin/sh\necho decoy\n")
    (decoy_dir / "memkit").chmod(0o755)

    out = profile.claude(
        "-p",
        "Run exactly this and report its output verbatim, nothing else: "
        "command -v memkit-recall memkit",
        "--output-format", "json", "--allowedTools", "Bash",
        cwd=str(project), timeout=300,
        extra_env={"PATH": f"{decoy_dir}:{os.environ['PATH']}"},
    )
    answer = json.loads(out.stdout)["result"]
    assert f"{staged}/bin/memkit-recall" in answer, answer
    # The collision, resolved against the plugin — which is the fact the skill
    # contract is built on, not a preference.
    assert f"{decoy_dir}/memkit" in answer, answer


@live_tier
def test_the_pty_driver_reaches_a_session_the_headless_flag_cannot(
    profile: Profile, staged: Path
) -> None:
    """The instrument itself, exercised so it cannot rot unnoticed.

    `claude -p` sets `CLAUDE_CODE_ENTRYPOINT=sdk-cli` and a terminal session
    reports `cli`; anything gated on that difference — the harness's own
    memory consolidation, for one — is unreachable without this driver. Proven
    against the dump hook's env capture rather than against the model's reply,
    since the point is what the harness told the hook.
    """
    if shutil.which("uv") is None:
        pytest.skip("no uv — the pty driver resolves pexpect through it")
    profile.register_dump_hooks("UserPromptSubmit")
    project = profile.project("work")

    out = subprocess.run(
        ["uv", "run", "--script", str(Path(__file__).parent / "drive_interactive.py"),
         str(project), "say the single word ok", "180"],
        capture_output=True, text=True, timeout=420, env=profile.env(),
    )
    dumps = profile.dumps("UserPromptSubmit")
    assert dumps, f"the driver never reached a turn\n{out.stdout[-3000:]}"
    assert dumps[-1]["env"].get("CLAUDE_CODE_ENTRYPOINT") == "cli", (
        dumps[-1]["env"].get("CLAUDE_CODE_ENTRYPOINT")
    )
