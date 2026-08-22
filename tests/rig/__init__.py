"""A scratch Claude Code profile, and the instruments to watch a hook inside it.

Everything memkit ships as a plugin is a claim about a harness this repo does
not own: that a manifest option reaches a hook's environment under a particular
name, that a registration passes no arguments, that an installed wrapper finds
its config and emits pointers. None of those can be settled by reading the
plugin's own files, and all of them fail SILENTLY — a hook that never fires and
a hook that fires and finds nothing are the same empty transcript. So the
scenarios drive the real `claude` binary against a profile that exists for the
length of one test.

Three instruments, matching the three things that go wrong:

  `Profile`      an isolated `CLAUDE_CONFIG_DIR` with the three prompts a fresh
                 profile stops on already answered. Without it every scenario
                 hangs on the theme picker.
  `hookdump.py`  registered as a hook, records argv, env, cwd and stdin per
                 invocation — the WHOLE environment, `ANTHROPIC_API_KEY`
                 included, and the whole prompt. Its records belong to the
                 scratch profile's tmp tree and must not be copied out of one. Registered at SETTINGS scope, which bounds what it
                 can settle: a settings-scope hook receives none of
                 `CLAUDE_PLUGIN_OPTION_*`, `CLAUDE_PLUGIN_ROOT` or
                 `CLAUDE_PLUGIN_DATA` (measured on 2.1.239), and the `argv` it
                 records is its own. What memkit's own registration received is
                 read off memkit's ARTIFACTS instead — the soak record and the
                 trust marker — which is the more honest instrument anyway,
                 since it measures what the product did rather than what a
                 probe saw.
  `drive_interactive.py`  a pty, for the runs that must not look headless.
                 `claude -p` sets `CLAUDE_CODE_ENTRYPOINT=sdk-cli` and some
                 harness behaviour keys on that, so a pty is the only way to
                 exercise the path a person is actually on.

**The safety property this module exists to hold: nothing here may touch the
real `~/.claude`.** Every entry point asserts its config dir is inside a tmp
tree before it runs a binary that writes. That assertion is not decoration —
`claude plugin install` writes to whatever `CLAUDE_CONFIG_DIR` names, and the
author's own profile carries a live memkit registration that a stray install
would sit beside.

Three tiers, and the split is what keeps CI honest:

  CLI tier     — needs only the `claude` binary. Marketplace add, install,
                 validate, and reading back what got registered. It never
                 dispatches a hook, so anything it claims about delivery it
                 claims about an environment this repo built.
  HARNESS tier — needs the binary and a real turn, and no model. Hook dispatch
                 precedes the model call, so a turn that cannot reach one still
                 runs install, option delivery, config resolution, retrieval
                 and injection — measured, and it is what makes the delivery
                 claim gateable at all. The turn's own exit status means
                 nothing here and is not asserted; the evidence is memkit's
                 artifacts, written before the call that fails.
  LIVE tier    — needs a model to answer a prompt, which means the author's
                 local proxy. Opt-in through `MEMKIT_RIG_LIVE=1`, because a
                 scenario that silently skips in CI and only ever runs on one
                 machine is a scenario nobody should read as a gate.

**WHAT A GREEN CI DOES NOT COVER**, recorded here because the next reader's
question is what the required checks mean. Two claims live only in the live
tier and are invisible to every merge: that a hook's stdout actually reaches
the turn's context (the harness tier reads memkit's artifacts, which say what
the hook believes it wrote), and that a duplicate registration is detected
across several real turns. A scheduled live run is the fix and it waits for the
first tagged release; until then, a harness bump is checked by a human running
`MEMKIT_RIG_LIVE=1 pytest tests/rig` and saying so in the PR, which is what
`renovate.json` dashboard-gates that bump for.

`MEMKIT_RIG_REQUIRED=1` (set by the `python` job) turns a missing `claude` into
a FAILURE rather than a skip for the harness tier. A gate that skips itself
when its dependency is missing is the shape of gate this tier exists to
replace: the whole defect it guards against — a plugin whose belief about the
harness is wrong but internally consistent — is silent, and so is a skip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RIG = Path(__file__).resolve().parent
REPO = RIG.parent.parent
HOOKDUMP = RIG / "hookdump.py"

# The live tier's dependency, named rather than probed: a base URL that is not
# Anthropic's, plus any key, is what keeps a scratch profile off first-party
# auth entirely. A fresh CLAUDE_CONFIG_DIR cannot see the real login in any
# case — macOS keychain credentials are stored under a service name hashed per
# config dir — so this is the only route a scenario has to a model.
LIVE_ENV = "MEMKIT_RIG_LIVE"
DEFAULT_PROXY = "http://127.0.0.1:18317"
# Declared by a caller that considers the harness tier a gate rather than a
# convenience — CI. See the tier note above for why a skip is the wrong answer
# there.
REQUIRED_ENV = "MEMKIT_RIG_REQUIRED"

# Routes the model call somewhere that cannot answer, without touching what
# runs before it. Bedrock rather than a bogus `ANTHROPIC_BASE_URL`, because a
# developer machine's keychain OAuth wins over both that and a bogus key
# (measured) — this is the only route that is unreachable on a laptop and on a
# credential-less runner alike.
#
# THE REGION IS WHAT MAKES IT UNREACHABLE, not the endpoint override. Measured
# on 2.1.239: with `AWS_REGION=us-east-1` the turn ends in 1.0s with
# `403 The security token included in the request is invalid` — an AWS API
# response, which a refused connection to 127.0.0.1:9 cannot produce — and
# removing `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` entirely changes nothing (1.01s,
# the same 403). The endpoint variable is ignored, so the "loopback discard"
# this used to claim was a real outbound HTTPS request to
# bedrock-runtime.us-east-1.amazonaws.com on every CI run — fast because AWS
# answered promptly, and on a runner whose egress is blackholed rather than
# refused it would have hung to the subprocess timeout for a reason that has
# nothing to do with memkit.
#
# A region that cannot resolve moves the failure into DNS, before anything
# leaves the machine: `bedrock-runtime.memkit-rig-nowhere-1.amazonaws.com` has
# no address (checked), and the turn ends in 0.54s with the hook already
# dispatched. The endpoint override is kept only so that a resolver which
# wildcards every name still has nowhere to connect.
NO_MODEL_ENV = {
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "AWS_REGION": "memkit-rig-nowhere-1",
    "AWS_ACCESS_KEY_ID": "AKIAINVALIDINVALID00",
    "AWS_SECRET_ACCESS_KEY": "invalid",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://127.0.0.1:9",
    # A profile carries a base URL for the live tier; this leaves no reachable
    # endpoint under that name either.
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    "CLAUDE_CODE_MAX_RETRIES": "0",
    "ANTHROPIC_MAX_RETRIES": "0",
}


def assert_no_model_answered(stdout: str, stderr: str) -> None:
    """Refuse a turn whose result cannot be read, then refuse one that a model
    answered.

    Two steps, and the ORDER is the point: asserting the absence of
    `modelUsage` in a dict that failed to parse is asserting the absence of a
    key in `{}`, which is true of every failure — an empty stdout, a crash
    before the first byte, a proxy's HTML. That kept a required CI gate green
    for reasons having nothing to do with memkit.

    A function rather than four lines inline so it can be driven with the
    garbage a live turn will not produce on request.
    """
    try:
        answer = json.loads(stdout)
    except ValueError:
        raise AssertionError(
            "the turn produced no JSON to check, so nothing here can say the "
            f"model was unreachable. stdout={stdout[:400]!r} "
            f"stderr={stderr[:400]!r}"
        ) from None
    if not isinstance(answer, dict):
        raise AssertionError(f"the turn's result is not an object: {answer!r}")
    if answer.get("modelUsage"):
        raise AssertionError(
            "a model answered a turn this scenario needs to be unreachable — "
            f"the route is not dead here: {answer.get('modelUsage')}"
        )


def claude_bin() -> str | None:
    return shutil.which("claude")


def cli_tier_reason() -> str | None:
    """Why the CLI tier cannot run here, or None when it can."""
    if claude_bin() is None:
        return "no `claude` on PATH — the CLI tier needs the real binary"
    return None


def harness_tier_reason() -> str | None:
    """Why the harness tier cannot run here, or None when it can.

    Never a reason when a caller has declared it required: the scenario then
    runs and fails on the missing binary, which is what a gate does.
    """
    if os.environ.get(REQUIRED_ENV) == "1":
        return None
    return cli_tier_reason()


def require_claude() -> str:
    """The binary, or a failure that says why a skip was not the answer."""
    binary = claude_bin()
    assert binary is not None, (
        f"no `claude` on PATH and {REQUIRED_ENV}=1 — this tier is a gate, so a "
        "missing harness is a failure rather than a skip"
    )
    return binary


def live_tier_reason() -> str | None:
    """Why the live tier cannot run here, or None when it can."""
    if (reason := cli_tier_reason()) is not None:
        return reason
    if os.environ.get(LIVE_ENV) != "1":
        return f"{LIVE_ENV}=1 not set — the live tier needs a model to answer"
    return None


class Profile:
    """One scratch `CLAUDE_CONFIG_DIR`, seeded past everything that blocks.

    A fresh profile stops on three things before it will run a single prompt,
    and each one looks like a hang rather than a prompt when stdin is a pipe:
    the onboarding theme picker, the trust dialog for the project directory,
    and the custom-API-key confirmation. All three are answered here, in the
    file the harness reads them from.

    `home` is redirected too, and not for tidiness. The hook resolves its
    derived state under `~/.cache/memory-recall` and its fixture corpus under
    `~`, so a scenario that left HOME alone would index the author's real
    corpus and write its soak records into the author's real log.
    """

    def __init__(self, root: Path, *, proxy: str | None = None) -> None:
        self.root = root
        self.config_dir = root / "claude-config"
        self.home = root / "home"
        self.projects = root / "projects"
        self.hooklog = root / "hooklog"
        self.proxy = proxy or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_PROXY
        self._api_key = "rig-local-proxy"
        for d in (self.config_dir, self.home, self.projects, self.hooklog):
            d.mkdir(parents=True, exist_ok=True)
        self._seed()

    def _seed(self) -> None:
        # `lastOnboardingVersion` is what the harness compares against its own
        # build to decide whether to re-run onboarding, so a number below the
        # running version re-opens the picker. It is deliberately absurd rather
        # than merely current: this profile never wants onboarding, and pinning
        # today's version would make the rig start hanging the week the harness
        # ships a new one.
        (self.config_dir / ".claude.json").write_text(
            json.dumps(
                {
                    "hasCompletedOnboarding": True,
                    "lastOnboardingVersion": "99.0.0",
                    "theme": "dark",
                    "customApiKeyResponses": {
                        "approved": [self._api_key, self._api_key[-20:]]
                    },
                    "projects": {
                        str(self.projects): {"hasTrustDialogAccepted": True},
                    },
                }
            ),
            encoding="utf-8",
        )

    def project(self, name: str) -> Path:
        """A trusted project directory. Trust is recorded per absolute path, so
        a directory created after `_seed` has to be added to the same file."""
        path = self.projects / name
        path.mkdir(parents=True, exist_ok=True)
        blob = json.loads((self.config_dir / ".claude.json").read_text())
        blob["projects"][str(path)] = {"hasTrustDialogAccepted": True}
        (self.config_dir / ".claude.json").write_text(json.dumps(blob), encoding="utf-8")
        return path

    def env(self, **extra: str) -> dict[str, str]:
        env = dict(
            os.environ,
            CLAUDE_CONFIG_DIR=str(self.config_dir),
            HOME=str(self.home),
            ANTHROPIC_BASE_URL=self.proxy,
            ANTHROPIC_API_KEY=self._api_key,
            # The pin cannot drift mid-scenario, and an autoupdate would also
            # rewrite the binary a scenario is making claims about.
            DISABLE_AUTOUPDATER="1",
            MEMKIT_RIG_HOOKLOG=str(self.hooklog),
        )
        # Dropped BEFORE `extra` is applied, not after. MEMKIT_CONFIG in the
        # ambient environment would reach a spawned hook and make a
        # config-delivery scenario pass without the delivery — but a scenario
        # that sets it deliberately, to stand a non-plugin registration up
        # beside the plugin, has to be able to. Popping last silently emptied
        # that scenario: both hooks fired, one of them found no config, and the
        # coexistence case it exists to prove could not occur.
        env.pop("MEMKIT_CONFIG", None)
        # And every plugin variable, for the sharper version of the same
        # reason. `CLAUDE_PLUGIN_OPTION_MEMKITCONFIG` is the whole claim the
        # config-delivery design rests on, and a scenario that inherits one
        # from the developer's shell measures its own environment rather than
        # the harness's. Popped before `extra` so a scenario that sets one
        # deliberately still can.
        for name in [k for k in env if k.startswith("CLAUDE_PLUGIN_")]:
            env.pop(name)
        env.update(extra)
        return env

    def _guard(self) -> None:
        """Refuse to run anything against a config dir that is not scratch.

        The one failure this rig must never have. `claude plugin install`
        writes wherever `CLAUDE_CONFIG_DIR` points, and the author's own
        profile has a live memkit registration in it.
        """
        resolved = self.config_dir.resolve()
        real = Path.home().resolve()
        assert resolved != (real / ".claude"), resolved
        assert real not in resolved.parents or ".cache" in resolved.parts, resolved

    def claude(
        self, *args: str, timeout: int = 120, check: bool = True, **kw
    ) -> subprocess.CompletedProcess[str]:
        """Run the real `claude` with this profile's environment."""
        self._guard()
        binary = claude_bin()
        assert binary is not None, "no `claude` on PATH"
        env = self.env(**kw.pop("extra_env", {}))
        out = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=kw.pop("cwd", str(self.projects)),
            stdin=subprocess.DEVNULL,
            **kw,
        )
        if check and out.returncode != 0:
            raise AssertionError(
                f"claude {' '.join(args)} exited {out.returncode}\n"
                f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}"
            )
        return out

    # --- plugin lifecycle ----------------------------------------------------

    def marketplace_add(self, path: Path) -> subprocess.CompletedProcess[str]:
        return self.claude("plugin", "marketplace", "add", str(path))

    def install(
        self, spec: str, *, config: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        args = ["plugin", "install", spec, "--yes"]
        for key, value in (config or {}).items():
            args += ["--config", f"{key}={value}"]
        return self.claude(*args)

    def installed(self) -> list[dict]:
        out = self.claude("plugin", "list", "--json")
        return json.loads(out.stdout)

    # --- instruments ---------------------------------------------------------

    def register_dump_hooks(self, *events: str) -> None:
        """Register `hookdump.py` on `events` in this profile's settings.

        Settings-scope rather than plugin-scope on purpose: this is the
        instrument, and an instrument registered the same way as the thing
        under test cannot tell them apart in the log.
        """
        settings_path = self.config_dir / "settings.json"
        settings = (
            json.loads(settings_path.read_text()) if settings_path.is_file() else {}
        )
        hooks = settings.setdefault("hooks", {})
        for event in events:
            hooks.setdefault(event, []).append(
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{sys.executable} {HOOKDUMP} {event}",
                            "timeout": 20,
                        }
                    ]
                }
            )
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def dumps(self, event: str | None = None) -> list[dict]:
        """Every hook invocation recorded so far, oldest first."""
        records = []
        for path in sorted(self.hooklog.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            if event is None or record.get("event") == event:
                records.append(record)
        return records



def stage_plugin(dest: Path, repo: Path = REPO) -> Path:
    """A copy of the plugin tree whose marketplace serves that copy in place.

    The shipped `marketplace.json` pins a released commit sha, which is a
    claim about what adopters receive and is exactly wrong for testing what is
    in the working tree: `claude plugin install` on it clones from GitHub over
    the network and installs a commit that is not this one. So the staged copy
    gets a same-directory source, and the shipped entry's own shape — that it
    is sha-pinned, and that the sha is a commit in this history — is pinned by
    a static test instead.

    TRACKED FILES ONLY, from `git ls-files`, because that is what the channel
    this rig stands in for delivers: a github install is a clone. A copy of the
    working tree stages files no adopter can receive — which makes an untracked
    wrapper pass here and be missing for everyone — and it stages files the
    payload must not carry at all, an adopter's own `memkit.json` at the root
    being the one that used to decide what the hook read.

    No fallback when git cannot answer. A staged tree assembled some other way
    is a tree whose relationship to the real channel is unknown, and a rig that
    quietly degrades to a more forgiving copy is the thing these scenarios
    exist to replace.

    Returns the staged root, which is both the marketplace and the plugin.
    """
    assert shutil.which("git") is not None, "staging a payload needs git"
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo, capture_output=True, text=True, timeout=120, check=True,
    )
    tracked = [name for name in listed.stdout.split("\0") if name]
    assert tracked, f"no tracked files under {repo}"
    dest.mkdir(parents=True)
    for name in tracked:
        source = repo / name
        # A tracked file the working tree no longer has (mid-rebase, a deleted
        # file not yet staged): skipping it silently would stage a payload that
        # is neither the index's nor the tree's.
        assert source.is_file(), f"tracked but missing from the working tree: {name}"
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # copy2, so the executable bit survives — a wrapper staged as 644 is a
        # hook the harness cannot run, which is a failure about this function
        # rather than about the plugin.
        shutil.copy2(source, target)
    manifest = dest / ".claude-plugin" / "marketplace.json"
    blob = json.loads(manifest.read_text())
    for entry in blob["plugins"]:
        entry["source"] = "./"
    manifest.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return dest
