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
                 invocation. It is how "zero arguments" and "the option
                 arrived" become assertions rather than readings of a manifest.
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

Two tiers, and the split is what keeps CI honest:

  CLI tier   — needs only the `claude` binary. Marketplace add, install,
               validate, and reading back what got registered. Runs wherever
               the binary is (CI installs a pinned one), skipped where it is
               not.
  LIVE tier  — needs a model to answer a prompt, which means the author's local
               proxy. Opt-in through `MEMKIT_RIG_LIVE=1`, because a scenario
               that silently skips in CI and only ever runs on one machine is a
               scenario nobody should read as a gate.
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


def claude_bin() -> str | None:
    return shutil.which("claude")


def cli_tier_reason() -> str | None:
    """Why the CLI tier cannot run here, or None when it can."""
    if claude_bin() is None:
        return "no `claude` on PATH — the CLI tier needs the real binary"
    return None


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
        env.update(extra)
        # MEMKIT_CONFIG in the ambient environment would reach a spawned hook
        # and make a config-delivery scenario pass without the delivery.
        env.pop("MEMKIT_CONFIG", None)
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

    def clear_dumps(self) -> None:
        for path in self.hooklog.glob("*.json"):
            path.unlink()


# Files and directories a staged copy leaves behind. `.git` is the big one and
# the reason this is a filter rather than a plain copy — but note what the
# exclusion implies: a github install delivers TRACKED files only, so a staged
# copy is more forgiving than the real channel. `test_plugin_surface.py` covers
# that gap statically by asserting every payload file is tracked.
STAGE_EXCLUDE = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def stage_plugin(dest: Path, repo: Path = REPO) -> Path:
    """A copy of the plugin tree whose marketplace serves that copy in place.

    The shipped `marketplace.json` pins a released commit sha, which is a
    claim about what adopters receive and is exactly wrong for testing what is
    in the working tree: `claude plugin install` on it clones from GitHub over
    the network and installs a commit that is not this one. So the staged copy
    gets a same-directory source, and the shipped entry's own shape — that it
    is sha-pinned, and that the sha is a commit in this history — is pinned by
    a static test instead.

    Returns the staged root, which is both the marketplace and the plugin.
    """
    shutil.copytree(
        repo,
        dest,
        ignore=lambda _d, names: {n for n in names if n in STAGE_EXCLUDE},
    )
    manifest = dest / ".claude-plugin" / "marketplace.json"
    blob = json.loads(manifest.read_text())
    for entry in blob["plugins"]:
        entry["source"] = "./"
    manifest.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return dest


def prompt(profile: Profile, text: str, *, cwd: Path, timeout: int = 180) -> str:
    """One headless turn, returning the model's text.

    `--output-format json` rather than the default, because the assertion a
    live scenario makes is usually about what the hook did, and a run that
    failed to reach a model at all has to be distinguishable from one that
    reached it and said nothing.
    """
    out = profile.claude(
        "-p",
        text,
        "--output-format",
        "json",
        cwd=str(cwd),
        timeout=timeout,
    )
    return out.stdout
