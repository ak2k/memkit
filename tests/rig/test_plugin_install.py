"""Scenarios that run the real `claude` binary against a scratch profile.

Everything here is a claim about a harness this repo does not own, so nothing
here can be settled by reading memkit's own files. Two tiers:

  CLI tier — needs the binary and nothing else. Validation, marketplace add,
    install, and reading back what got registered. Runs in CI, which installs
    a pinned Claude Code for exactly this. It never dispatches a hook.
  HARNESS tier — needs the binary and a real turn, and no model. This is where
    the config-delivery claim is settled, because it is the only tier in which
    the harness rather than the test produces
    `CLAUDE_PLUGIN_OPTION_MEMKITCONFIG`. Runs in CI, and FAILS rather than
    skips there.
  LIVE tier — needs a model to answer a prompt, which means the author's local
    proxy. `MEMKIT_RIG_LIVE=1` to opt in. A scenario that silently skips
    everywhere but one machine is not a gate, so it says which it is.

The U2 verification is an install that yields a hook which FIRES, asserted on a
pointer rather than on an exit code, because an inert hook and a wired one both
exit 0 and print nothing.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rig import (
    NO_MODEL_ENV,
    REPO,
    Profile,
    assert_no_model_answered,
    assert_scratch_config_dir,
    cli_tier_reason,
    harness_tier_reason,
    live_tier_reason,
    require_claude,
    stage_plugin,
)
from rig import (
    RIG as RIG_DIR,
)

cli_tier = pytest.mark.skipif(
    cli_tier_reason() is not None, reason=cli_tier_reason() or ""
)
harness_tier = pytest.mark.skipif(
    harness_tier_reason() is not None, reason=harness_tier_reason() or ""
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


# --- the rig's own safety and its instruments ---------------------------------


def test_the_unreachable_turn_gate_refuses_a_result_it_cannot_read() -> None:
    """The gate has to fail on garbage, not pass on it.

    A turn that dispatches the hook and then dies with empty or non-JSON stdout
    used to satisfy it, because the absence of `modelUsage` was checked in a
    dict that had failed to parse — so a real endpoint, a proxy or a broken CLI
    all kept a required check green. Driven directly, because a live turn will
    not produce those on request.
    """
    assert_no_model_answered('{"is_error": true}', "")  # the ordinary case
    for stdout in ("", "   ", "not json at all", "<html>bot wall</html>", "[]", "null"):
        with pytest.raises(AssertionError):
            assert_no_model_answered(stdout, "stderr")
    with pytest.raises(AssertionError, match="a model answered"):
        assert_no_model_answered('{"modelUsage": {"claude": {"in": 1}}}', "")


# Values the guard must refuse, and the reason each one is here. `$HOME` is
# scratch in every caller, so a guard deriving the real home from it passes all
# of them — which is what the driver's own copy did.
NOT_SCRATCH = [
    Path(pwd.getpwuid(os.getuid()).pw_dir) / ".claude",
    Path(pwd.getpwuid(os.getuid()).pw_dir),
    Path(pwd.getpwuid(os.getuid()).pw_dir) / "projects" / "somewhere",
    Path("/"),
    Path("/etc"),
]


def test_the_scratch_guard_refuses_every_config_dir_that_is_not_scratch(
    tmp_path,
) -> None:
    """The rig's headline safety property, DRIVEN.

    `claude plugin install` writes wherever `CLAUDE_CONFIG_DIR` points and the
    author's own profile carries a live memkit registration, so this is what
    stands between a scenario and that profile. It is now a positive
    allowlist — the temp tree or a cache dir — because the two negative tests
    it replaces both derived the real home from `$HOME`, which every caller has
    already redirected into the scratch tree.
    """
    assert_scratch_config_dir(tmp_path / "rig" / "claude-config")
    Profile(tmp_path / "ok")._guard()  # the ordinary case stays silent

    for unsafe in NOT_SCRATCH:
        with pytest.raises(AssertionError):
            assert_scratch_config_dir(unsafe)
        hijacked = Profile(tmp_path / "rig2")
        hijacked.config_dir = unsafe
        with pytest.raises(AssertionError):
            hijacked._guard()


def test_the_pty_driver_refuses_through_the_same_guard(tmp_path) -> None:
    """The sibling that is a separate process, and could not share the parent's
    copy — which is how it ended up with a guard that could not refuse.

    RUN rather than grepped: the case this replaces read the driver's source
    and asserted two substrings were present, so it proved the guard was
    written and never that it refuses. Driven with `HOME` scratch and
    `CLAUDE_CONFIG_DIR` pointing at the real profile, which is the exact shape
    that walked past it.
    """
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    project = tmp_path / "project"
    project.mkdir()
    out = subprocess.run(
        [sys.executable, str(RIG_DIR / "drive_interactive.py"), str(project), "hi", "5"],
        capture_output=True, text=True, timeout=120,
        env={
            "PATH": os.environ["PATH"],
            # Scratch, exactly as `Profile.env()` sets it — this is what made
            # the previous guard blind.
            "HOME": str(tmp_path / "scratch-home"),
            "CLAUDE_CONFIG_DIR": str(real_home / ".claude"),
        },
    )
    assert out.returncode != 0, out.stdout
    assert "not a scratch config dir" in (out.stdout + out.stderr), out.stderr
    # And it never reached the harness: a spawn would say so.
    assert "claude" not in out.stdout.lower() or "not a scratch" in out.stderr


def test_hookdump_records_the_argv_and_env_it_exists_to_record(tmp_path) -> None:
    """The instrument, exercised on the path it is for.

    It is used by one scenario, which asserts one unrelated field, so it could
    stop recording argv or env entirely and nothing would notice until someone
    reached for it mid-investigation. Run directly here rather than through a
    turn: what is under test is the recorder, not the harness.
    """
    profile = Profile(tmp_path / "rig")
    subprocess.run(
        [sys.executable, str(RIG_DIR / "hookdump.py"), "UserPromptSubmit"],
        input=json.dumps({"session_id": "dump", "prompt": "hello"}),
        capture_output=True, text=True, timeout=60,
        env={**profile.env(), "MEMKIT_RIG_MARKER": "present"},
    )
    dumps = profile.dumps("UserPromptSubmit")
    assert len(dumps) == 1, len(dumps)
    record = dumps[0]
    # FIELD-SCOPED failure messages. A record deliberately holds
    # `dict(os.environ)` of the spawned process and the whole prompt, and this
    # case runs unconditionally in the `python` job — whose logs are public for
    # a public repo. Passing the record as the assertion message copies the
    # runner's ambient environment into them, which is the bound
    # `tests/rig/__init__.py` states these records must not cross.
    assert record["argv"] == ["UserPromptSubmit"], record["argv"]
    assert record["payload"]["prompt"] == "hello", record["payload"]
    assert record["env"]["MEMKIT_RIG_MARKER"] == "present", (
        record["env"].get("MEMKIT_RIG_MARKER")
    )
    assert record["cwd"], "no cwd recorded"
    assert record["raw_len"] > 0, record["raw_len"]


# --- the staged payload itself ------------------------------------------------


def test_the_staged_payload_is_what_the_channel_delivers(staged: Path) -> None:
    """The rig's own instrument, because every scenario below inherits it.

    A github install is a clone, so an untracked file is a file no adopter
    receives. When staging was a copy of the working tree it delivered more
    than the channel can — which makes an untracked wrapper pass here and be
    missing for everyone, and lets a developer's own `memkit.json` sit at the
    payload root of every rig install, where it used to decide what the hook
    read.

    A smoke check only: on a clean checkout it agrees with `git ls-files`
    whichever way the staging is implemented, so the case that actually gates
    the implementation is the one below.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO,
        capture_output=True, text=True, timeout=120, check=True,
    )
    tracked = {name for name in listed.stdout.split("\0") if name}
    present = {
        str(path.relative_to(staged))
        for path in staged.rglob("*")
        if path.is_file()
    }
    assert present <= tracked, sorted(present - tracked)
    # And the wrappers survived the copy as executables — a staged 644 wrapper
    # is a hook the harness cannot run, which would be a failure about the rig
    # rather than about the plugin.
    for wrapper in ("bin/memkit", "bin/memkit-hook", "bin/memkit-recall"):
        assert os.access(staged / wrapper, os.X_OK), wrapper


def test_staging_reads_the_index_rather_than_the_working_tree(tmp_path) -> None:
    """The gate the case above cannot be.

    `present <= tracked` compares the staged tree against the same
    `git ls-files` the stager used, so on a fresh clone — which is every CI run
    — it holds whichever way staging is implemented. Measured: reverting
    `stage_plugin` to the old `copytree` left that case green, and every rig
    scenario inherits the instrument.

    So this builds a repository whose working tree and index DISAGREE, which is
    the only state that can tell the two implementations apart, and asserts the
    untracked files are absent from the staged payload. The two it plants are
    the two that matter: a `memkit.json` at the payload root, which was a
    config rung until this branch deleted it, and a stray executable in `bin/`,
    which is on the agent's PATH.
    """
    scratch = tmp_path / "scratch-repo"
    (scratch / ".claude-plugin").mkdir(parents=True)
    (scratch / "bin").mkdir()
    (scratch / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"name": "memkit", "plugins": [{"name": "memkit", "source": "x"}]})
    )
    (scratch / "bin" / "memkit-hook").write_text("#!/bin/sh\nexit 0\n")
    (scratch / "bin" / "memkit-hook").chmod(0o755)
    def run(*a: str) -> None:
        subprocess.run(
            a, cwd=scratch, capture_output=True, text=True, timeout=120, check=True
        )

    run("git", "init", "-q")
    run("git", "-c", "user.email=rig@local", "-c", "user.name=rig", "add", "-A")

    # After the index is written, so the tree and the index disagree.
    (scratch / "memkit.json").write_text('{"schema": 1}')
    (scratch / "bin" / "stray").write_text("#!/bin/sh\necho stray\n")

    staged_out = stage_plugin(tmp_path / "out", scratch)
    present = {str(p.relative_to(staged_out)) for p in staged_out.rglob("*") if p.is_file()}
    assert "memkit.json" not in present, present
    assert "bin/stray" not in present, present
    assert "bin/memkit-hook" in present, present


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


# --- HARNESS tier -------------------------------------------------------------
#
# What separates this tier from the one above is who produces the environment.
# The CLI-tier case below resolves the registration's command itself and runs
# it with `CLAUDE_PLUGIN_OPTION_MEMKITCONFIG` set by the test — which makes the
# one claim it looks like it is testing unfalsifiable, because the test is the
# source of the variable. Here the harness exports it or nothing does.
#
# Measured, and it is why this tier can exist: a rename of the assumed prefix
# across bin/lib/common.sh and both test files leaves the whole suite green and
# the installed plugin recording `trust:unconfigured` on a real turn. The
# variable name is a fact about a build this repo does not own, pinned by
# renovate precisely because it can move.


# Well under the uncapped retry budget rather than just over it. The fast path
# measures around a second; the previous 180 s sat 4 s below the 184 s the full
# budget takes on the pinned harness, so if the retry ceilings ever stopped
# being honoured — the exact event this tier exists to catch — the difference
# between a named failure and a bare `TimeoutExpired` was four seconds.
NO_MODEL_TIMEOUT = 60


def _no_model(profile: Profile, prompt: str, *, cwd: Path) -> None:
    """One real turn whose model call cannot succeed.

    Nothing about the turn's own SUCCESS is asserted, deliberately: with no
    reachable model it legitimately fails, and gating on that would be gating
    on the half of the run this says nothing about. Hook dispatch precedes the
    model call, so the evidence is memkit's artifacts, written before the
    failure.

    What IS asserted is that no model answered. A resolver that wildcards every
    hostname, or a runner with a transparent proxy, would otherwise turn this
    into a billed turn that passes for the wrong reason.
    """
    try:
        out = profile.claude(
            "-p", prompt, "--output-format", "json",
            cwd=str(cwd), timeout=NO_MODEL_TIMEOUT, check=False,
            extra_env=dict(NO_MODEL_ENV),
        )
    except subprocess.TimeoutExpired:
        # `pytest.fail` raises, but pyright cannot know that from here, and a
        # `NoReturn` it cannot see is an "out is possibly unbound" below.
        raise AssertionError(
            f"the turn ran past {NO_MODEL_TIMEOUT}s with the model unreachable. "
            "The retry ceilings (CLAUDE_CODE_MAX_RETRIES, ANTHROPIC_MAX_RETRIES) "
            "are what bound it; a harness that stopped honouring them takes the "
            "full budget, measured at 184s on the pinned build."
        ) from None
    assert_no_model_answered(out.stdout, out.stderr)


@harness_tier
def test_the_harness_delivers_the_option_and_the_hook_serves_the_turn(
    profile: Profile, staged: Path
) -> None:
    """Config delivery, measured on an environment this repo did not build.

    The config sits at a path under the profile's own HOME, which is a path no
    rung but the install option can name — the other rung is
    `$CLAUDE_PLUGIN_DATA/memkit.json` — so a pointer to a memory in that store
    is proof that the option arrived, under the name the wrapper reads, from
    the harness.
    """
    require_claude()
    profile.marketplace_add(staged)
    config = _fixture_config(profile)
    profile.install("memkit@memkit", config={"memkitConfig": str(config)})

    # The test supplies no plugin variable of any kind; `Profile.env` strips
    # them. Without this the scenario could pass on its own environment.
    assert not [k for k in profile.env() if k.startswith("CLAUDE_PLUGIN_")]

    _no_model(profile, PROMPT, cwd=profile.project("work"))

    injected = [r for r in _soak(profile) if r["outcome"] == "injected"]
    assert injected, [r["outcome"] for r in _soak(profile)] or "no soak records"
    assert EXPECTED in injected[-1]["injected"], injected[-1]


@harness_tier
def test_without_the_option_the_same_turn_leaves_a_refusal_and_no_store(
    profile: Profile, staged: Path
) -> None:
    """The negative, without which the case above cannot tell rung 1 from some
    other rung answering — or from a hook that would have served that corpus
    whatever the harness did.

    Same install, same turn, one flag removed. The trust marker is the
    plugin-scoped record of the refusal, and the absent state directory is the
    other half: an install that has not been configured has not been consented
    to, and does not get to create it.
    """
    require_claude()
    profile.marketplace_add(staged)
    _fixture_config(profile)  # on disk, and nothing names it
    profile.claude("plugin", "install", "memkit@memkit", "--yes")

    _no_model(profile, PROMPT, cwd=profile.project("work"))

    marker = profile.config_dir / "plugins" / "data" / "memkit-memkit" / "trust.json"
    assert marker.is_file(), "the gate refused without recording it"
    assert [r["outcome"] for r in json.loads(marker.read_text())["records"]] == [
        "trust:unconfigured"
    ]
    assert not (profile.home / ".cache" / "memory-recall").exists()


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
