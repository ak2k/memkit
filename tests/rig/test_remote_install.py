"""The adopter path, over the real network: `marketplace add ak2k/memkit`,
then `install` of the sha that manifest pins, with no credential of any kind.

WHY THIS TIER EXISTS. Every other rig scenario stages the working tree as a
marketplace that serves itself in place — `stage_plugin` rewrites each entry's
`source` to `"./"` — which is the right thing to test on a pull request and is
precisely wrong for one question. The source it rewrites away is the TRANSPORT,
and the transport is a claim about a harness this repo does not own: what
`owner/repo` resolves to, whether a `github` source clones over SSH, whether
`marketplace add` falls back to HTTPS where `install` does not. Staging cannot
observe any of it, and nothing else in the repo ever fetched the real manifest.

That gap is not hypothetical. v0.1.0 shipped a marketplace entry whose source
was `{"source": "github", "repo": "ak2k/memkit"}`, which clones over SSH with
no HTTPS fallback: `marketplace add` succeeded (that fetch does fall back) and
`install` then died with `Permission denied (publickey)` on the machine of
every adopter without GitHub SSH keys. Every check in the repo was green.

WHAT ONLY THIS TIER OBSERVES, and no other tier can:

  - that MAIN's manifest — the file github serves, not the one in the working
    tree — is fetchable by a machine with no credential, and names a source
    that machine can then clone.
  - that the PINNED payload arrives over that transport. Staging installs the
    working tree; this installs the commit adopters actually receive, which is
    usually not the commit under test.
  - that the install REGISTERED something. `claude plugin details` is the only
    readback that distinguishes a real install from one that exited 0 and wired
    up nothing — a pin whose commit carries no `hooks/hooks.json` reports
    `Successfully installed` and `Hooks (0)`, measured, which is why exit 0 is
    asserted alongside the inventory and never instead of it.
  - that the hook the remote install registered SERVES A TURN, read off
    memkit's own artifacts rather than off the harness's word for it.

NO SKIPS WITH THE FLAG SET. `MEMKIT_RIG_REMOTE=1` opts the tier in;
`MEMKIT_RIG_REQUIRED=1` alongside it turns a missing `claude` into a failure.
A missing binary, a marketplace add that fails, an inventory this file cannot
parse — each is red with a message naming what it expected, because the whole
defect class here is silent and a skip is silent too.

The two `cli_tier` cases at the bottom are the ones that keep the assertions
above honest, and they need no network — so they run on every pull request
rather than once a day. One drives the `Hooks (n)` parser against an install
that really did register nothing; the other drives the historical SSH failure
and asserts it is still a failure, which is what makes the `url` source type in
`.claude-plugin/marketplace.json` a decision rather than a leftover.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from rig import (
    NO_MODEL_ENV,
    REMOTE_MARKETPLACE,
    REPO,
    Profile,
    assert_no_model_answered,
    cli_tier_reason,
    fixture_config,
    remote_tier_reason,
    require_claude,
    soak_records,
    stage_plugin,
)

cli_tier = pytest.mark.skipif(
    cli_tier_reason() is not None, reason=cli_tier_reason() or ""
)
remote_tier = pytest.mark.skipif(
    remote_tier_reason() is not None, reason=remote_tier_reason() or ""
)

# `plugin@marketplace`, and both halves happen to be `memkit`: the marketplace
# name comes from MAIN's manifest and the plugin name from the entry in it.
SPEC = "memkit@memkit"

# A clone from github, on a runner, with the ~1.2 MiB of tracked tree this
# repository is. Generous rather than tight on purpose — the failure this tier
# is for is a transport that CANNOT work, which fails in seconds, so a timeout
# here means the network and there is nothing to learn from making that verdict
# arrive sooner.
NETWORK_TIMEOUT = 300

# Matches the harness tier's budget for the same reason: well under the
# uncapped retry ceiling, so a harness that stopped honouring
# CLAUDE_CODE_MAX_RETRIES produces a named failure rather than a bare
# TimeoutExpired.
NO_MODEL_TIMEOUT = 60

# The memory a prompt from the fixture corpus must reach. The PROMPT is read
# out of the fixture eval's own suite below rather than written twice, so a
# corpus edit that renames the memory fails here by name instead of quietly
# measuring a retrieval nobody meant.
EXPECTED = "sprocket_alignment.md"


def _fixture_prompt(expected: str = EXPECTED) -> str:
    suite = json.loads(
        (REPO / "tests" / "fixtures" / "memkit.json").read_text(encoding="utf-8")
    )["eval"]["cases"]["suite"]
    for case in suite:
        if case.get("file") == expected:
            return case["prompt"]
    raise AssertionError(
        f"no case in the fixture eval's suite retrieves {expected} — the corpus "
        f"moved and this tier would otherwise measure nothing: "
        f"{[c.get('file') for c in suite]}"
    )


PROMPT = _fixture_prompt()


# --- reading the harness back -------------------------------------------------


def _hook_count(details: subprocess.CompletedProcess[str]) -> int:
    """The number `claude plugin details` reports in its component inventory.

    A PARSE FAILURE IS A FAILURE, never a zero and never a skip. Returning 0
    for output this cannot read would turn every future change to the
    inventory's format into a green run reporting that memkit registers no
    hooks — which is the exact shape of silence this tier exists to break.
    """
    if details.returncode != 0:
        raise AssertionError(
            f"`claude plugin details {SPEC}` exited {details.returncode} — an "
            "installed plugin the harness cannot describe is not installed\n"
            f"--- stdout ---\n{details.stdout}\n--- stderr ---\n{details.stderr}"
        )
    found = re.search(r"^\s*Hooks \((\d+)\)", details.stdout, re.MULTILINE)
    if found is None:
        raise AssertionError(
            "no `Hooks (n)` line in the component inventory, so nothing here "
            "can say what the install registered. The inventory's format is a "
            "fact about the pinned Claude Code and may have moved:\n"
            f"{details.stdout}"
        )
    return int(found.group(1))


def _install_path(profile: Profile) -> Path:
    """Where the harness put the payload it cloned, from its OWN readback.

    `plugin list --json` reports `installPath`; the on-disk layout underneath
    it is not a documented surface and is not walked here.
    """
    installed = profile.installed()
    entry = [p for p in installed if p["id"] == SPEC]
    assert entry, f"{SPEC} is not in `plugin list --json`: {installed}"
    path = entry[0].get("installPath")
    assert path, f"the harness reported no installPath for {SPEC}: {entry[0]}"
    return Path(path)


def _fetched_manifest(profile: Profile) -> dict:
    """MAIN's marketplace manifest, as the harness fetched it.

    Read from the profile rather than from the working tree, and that is the
    whole point: on any branch but main those two files differ, and the one an
    adopter is served is this one.
    """
    manifest = (
        profile.config_dir
        / "plugins"
        / "marketplaces"
        / "memkit"
        / ".claude-plugin"
        / "marketplace.json"
    )
    assert manifest.is_file(), (
        f"`marketplace add {REMOTE_MARKETPLACE}` left no manifest at {manifest} "
        "— either the fetch did not happen or the cache layout moved"
    )
    return json.loads(manifest.read_text(encoding="utf-8"))


def _network(
    what: str, run: Callable[[], subprocess.CompletedProcess[str]]
) -> subprocess.CompletedProcess[str]:
    """Run a step that reaches github, and name the dependency when it hangs.

    A bare `TimeoutExpired` from inside a scenario reads as a memkit failure.
    This tier is the one place where it usually is not.
    """
    try:
        return run()
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"{what} ran past {NETWORK_TIMEOUT}s. This tier clones from "
            "github.com anonymously; check that the runner has egress before "
            "reading this as a memkit regression."
        ) from None


@pytest.fixture
def profile(tmp_path) -> Profile:
    return Profile(tmp_path / "rig")


# --- REMOTE tier --------------------------------------------------------------


@remote_tier
def test_the_real_marketplace_add_and_install_register_a_hook(
    profile: Profile,
) -> None:
    """Steps one and two of the adopter path, with no credential on the box.

    THREE assertions and none of them is redundant. The fetched manifest is
    what says the transport an adopter is pointed at is one they can use — the
    v0.1.0 regression is visible here and nowhere else in this repo. Exit 0 is
    what says the clone over that transport worked. And the inventory is what
    says the install registered something: a pin whose commit carries no
    `hooks/hooks.json` produces `Successfully installed` and `Hooks (0)`, so
    the exit status alone is a gate that passes on the failure it is for.
    """
    require_claude()

    added = _network(
        f"`marketplace add {REMOTE_MARKETPLACE}`",
        lambda: profile.marketplace_add(REMOTE_MARKETPLACE, timeout=NETWORK_TIMEOUT),
    )
    assert "Successfully added marketplace" in added.stdout, added.stdout

    entry = _fetched_manifest(profile)["plugins"][0]
    source = entry["source"]
    # The transport, read off the file github served rather than off the
    # working tree. `url` + `https://` is the anonymous-clone claim; a `github`
    # source, or an `ssh://`/`git@` url, is the shipped regression.
    assert source.get("source") == "url", (
        "MAIN's manifest names a source type an adopter without SSH keys "
        "cannot clone from — this is the v0.1.0 regression",
        source,
    )
    assert str(source.get("url", "")).startswith("https://"), source
    assert re.fullmatch(r"[0-9a-f]{40}", str(source.get("sha", ""))), (
        "MAIN's manifest does not pin a commit, so every push to main becomes "
        "hook code in an adopter's next session",
        source,
    )

    config = fixture_config(profile)
    installed = _network(
        f"`plugin install {SPEC}`",
        lambda: profile.install(
            SPEC, config={"memkitConfig": str(config)}, timeout=NETWORK_TIMEOUT
        ),
    )
    assert "Successfully installed" in installed.stdout, installed.stdout

    # The payload minimum, in the tree the harness actually cloned. This is
    # what fails when the pin names a commit that carries no plugin — before
    # the inventory is consulted, and with a message that says which file.
    root = _install_path(profile)
    for name in ("hooks/hooks.json", "bin/memkit-hook", "bin/lib/common.sh",
                 "src/memkit/memory_prompt_recall.py"):
        assert (root / name).is_file(), (
            f"the pinned payload at {root} carries no {name} — the manifest "
            f"pins {source['sha']}, which is not a commit that can serve as a "
            "plugin"
        )
    # And the wrapper survived the clone executable: a 644 hook is one the
    # harness cannot run, and it fails as silence.
    assert os.access(root / "bin" / "memkit-hook", os.X_OK), root

    # The readback that exit 0 cannot give.
    details = profile.details(SPEC)
    assert _hook_count(details) == 1, details.stdout

    # And the option the install was given is what the harness recorded — the
    # config-delivery claim, on the remote payload rather than a staged one.
    settings = json.loads((profile.config_dir / "settings.json").read_text())
    assert settings["pluginConfigs"][SPEC]["options"] == {
        "memkitConfig": str(config)
    }


@remote_tier
def test_a_turn_on_the_remote_install_injects_from_a_store_in_the_scratch_home(
    profile: Profile,
) -> None:
    """Step three: one real turn, with the model route dead, on the plugin as
    an adopter receives it.

    The turn's own exit status is not asserted and means nothing here — with no
    reachable model it legitimately fails. Hook dispatch precedes the model
    call, so the evidence is memkit's own artifacts, written before the failure.

    The store is COPIED INTO THE SCRATCH HOME, which is what makes the soak
    record proof rather than coincidence: that path is one no rung but the
    install option can name, so a pointer to a memory in it says the option
    arrived, under the name the wrapper reads, from a harness that installed
    the payload over the network.
    """
    require_claude()
    _network(
        f"`marketplace add {REMOTE_MARKETPLACE}`",
        lambda: profile.marketplace_add(REMOTE_MARKETPLACE, timeout=NETWORK_TIMEOUT),
    )
    config = fixture_config(profile)
    _network(
        f"`plugin install {SPEC}`",
        lambda: profile.install(
            SPEC, config={"memkitConfig": str(config)}, timeout=NETWORK_TIMEOUT
        ),
    )
    details = profile.details(SPEC)
    assert _hook_count(details) == 1, (
        "nothing registered, so the turn below would prove nothing",
        details.stdout,
    )

    # The test supplies no plugin variable of any kind; `Profile.env` strips
    # them. Without this the scenario could pass on its own environment.
    assert not [k for k in profile.env() if k.startswith("CLAUDE_PLUGIN_")]

    try:
        turn = profile.claude(
            "-p", PROMPT, "--output-format", "json",
            cwd=str(profile.project("work")),
            timeout=NO_MODEL_TIMEOUT,
            check=False,
            extra_env=dict(NO_MODEL_ENV),
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"the turn ran past {NO_MODEL_TIMEOUT}s with the model unreachable. "
            "The retry ceilings (CLAUDE_CODE_MAX_RETRIES, ANTHROPIC_MAX_RETRIES) "
            "are what bound it; a harness that stopped honouring them takes the "
            "full budget, measured at 184s on the pinned build."
        ) from None
    # A runner with a transparent proxy, or a resolver that wildcards every
    # name, would otherwise turn this into a billed turn passing for the wrong
    # reason — and this tier is the one that runs with egress.
    assert_no_model_answered(turn.stdout, turn.stderr)

    records = soak_records(profile)
    assert records, (
        "the remotely installed hook wrote no soak record for a turn it was "
        f"dispatched on — nothing ran. `plugin details` reported one hook; "
        f"the log under {profile.home}/.cache/memory-recall is absent or empty"
    )
    injected = [r for r in records if r["outcome"] == "injected"]
    assert injected, [r["outcome"] for r in records]
    assert EXPECTED in injected[-1]["injected"], injected[-1]


# --- CLI tier: the two cases that keep the assertions above honest ------------


@cli_tier
def test_an_install_that_registers_nothing_still_reports_success(
    profile: Profile, tmp_path
) -> None:
    """The inventory assertion, watched failing.

    A gate nobody has watched fail is a gate nobody has watched, and this one
    guards the outcome that fooled everyone: `Successfully installed` from a
    payload that wired up nothing. Reproduced by staging the tree with
    `hooks/` removed, which is the state a pin at a commit before the plugin
    skeleton would deliver — offline, so it gates on every pull request rather
    than once a day.

    Both halves are asserted. If the exit status ever started reporting this,
    the remote tier's `Hooks (1)` would be belt-and-braces rather than the
    load-bearing assertion it currently is, and that is worth being told.
    """
    staged = stage_plugin(tmp_path / "staged" / "memkit", REPO)
    shutil.rmtree(staged / "hooks")

    profile.marketplace_add(staged)
    out = profile.install(SPEC, check=False)
    assert out.returncode == 0 and "Successfully installed" in out.stdout, (
        "an install that registers nothing now fails outright — good news, and "
        "the remote tier's inventory assertion is no longer the only thing "
        f"catching it\n{out.stdout}\n{out.stderr}"
    )
    details = profile.details(SPEC)
    assert _hook_count(details) == 0, details.stdout


@cli_tier
def test_the_source_type_this_marketplace_rejected_still_cannot_clone(
    profile: Profile, tmp_path
) -> None:
    """The historical bug, driven, so the `url` source type is a decision
    somebody can re-read rather than a leftover nobody dares touch.

    `{"source": "github"}` clones over SSH. `marketplace add` falls back to
    HTTPS, which is why the failure lands at INSTALL and why every check in the
    repo stayed green through v0.1.0. Reproduced against a scratch marketplace
    carrying that source with SSH stubbed dead, which is the state of every
    runner and of every adopter who has never pushed to github from this
    machine — the stub only makes it deterministic on a laptop that has.

    The failure message says what a RED here means: not that memkit broke, but
    that the harness grew an HTTPS fallback for this source type and the
    workaround in `.claude-plugin/marketplace.json` can be revisited.
    """
    false_binary = shutil.which("false")
    assert false_binary, "no `false` on PATH to stub SSH with"

    market = tmp_path / "ssh-marketplace" / ".claude-plugin"
    market.mkdir(parents=True)
    shipped = json.loads(
        (REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    shipped["name"] = "memkit-ssh"
    for entry in shipped["plugins"]:
        entry["source"] = {
            "source": "github",
            "repo": REMOTE_MARKETPLACE,
            "sha": entry["source"]["sha"],
        }
    (market / "marketplace.json").write_text(json.dumps(shipped, indent=2))

    profile.marketplace_add(market.parent)
    out = profile.install(
        "memkit@memkit-ssh", check=False, extra_env={"GIT_SSH_COMMAND": false_binary}
    )
    combined = out.stdout + out.stderr
    assert out.returncode != 0, (
        "a `github` source installed with SSH stubbed dead — the harness has "
        "gained an HTTPS fallback for this source type, so the `url` "
        "workaround in .claude-plugin/marketplace.json can be revisited\n"
        f"{combined}"
    )
    assert "clone" in combined.lower(), combined
    assert [p["id"] for p in profile.installed()] == [], profile.installed()


# --- the tier gates themselves ------------------------------------------------


def test_the_network_tier_is_opt_in_and_then_required(monkeypatch) -> None:
    """The two-gate contract, driven in both directions.

    It is the whole reason this file can hold network scenarios without a bare
    `pytest` reaching github, and the reason a CI job that has opted in cannot
    report green on scenarios that never ran. Both halves fail silently if they
    regress — an accidental network call looks like a slow test, and a skipped
    scenario looks exactly like a passing one in a check name.

    `live_tier_reason` is checked to the same contract because it did NOT hold
    to it until this branch: `live.yml` sets `MEMKIT_RIG_REQUIRED=1` and says
    it does so to turn an unstartable scenario into a failure, while the
    function returned the missing-binary skip before ever reading the variable.
    """
    import rig

    for reason, opt_in in (
        (rig.remote_tier_reason, rig.REMOTE_ENV),
        (rig.live_tier_reason, rig.LIVE_ENV),
    ):
        monkeypatch.delenv(opt_in, raising=False)
        monkeypatch.delenv(rig.REQUIRED_ENV, raising=False)
        assert opt_in in (reason() or ""), (opt_in, reason())

        # Opted in, binary missing, and nobody has called it a gate: a skip,
        # which is what a developer without the harness installed should get.
        monkeypatch.setenv(opt_in, "1")
        monkeypatch.setattr(rig.shutil, "which", lambda _name: None)
        assert "no `claude` on PATH" in (reason() or ""), reason()

        # Opted in AND declared required: no reason, so the scenario runs and
        # fails on `require_claude()` rather than skipping.
        monkeypatch.setenv(rig.REQUIRED_ENV, "1")
        assert reason() is None, reason()
        with pytest.raises(AssertionError, match=rig.REQUIRED_ENV):
            rig.require_claude()
        monkeypatch.undo()


def test_a_soak_log_that_cannot_be_read_fails_rather_than_reads_empty(
    tmp_path,
) -> None:
    """The evidence reader, driven on the garbage a real turn will not produce.

    Every no-model scenario in the rig rests on this list: an empty one means
    the hook never ran. A `json.loads` that raised out of a comprehension —
    which is what this replaced — reported a `ValueError` from a test helper
    for what is a failure of memkit's own artifact, and a version that swallowed
    it would report `no records` for a log full of them.
    """
    profile = Profile(tmp_path / "rig")
    log = profile.home / ".cache" / "memory-recall" / "log.jsonl"
    log.parent.mkdir(parents=True)

    assert soak_records(profile) == []  # absent is empty, and that is a state
    log.write_text('{"outcome": "injected", "injected": ["a.md"]}\n\n')
    assert [r["outcome"] for r in soak_records(profile)] == ["injected"]

    for garbage in ("not json", "[]", '{"no": "outcome"}', "null"):
        log.write_text(f'{{"outcome": "injected"}}\n{garbage}\n')
        with pytest.raises(AssertionError, match="line 2"):
            soak_records(profile)
