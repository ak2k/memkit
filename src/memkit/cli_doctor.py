"""`memkit doctor` — one envelope naming every state this install can be in.

The failure this exists to prevent is an agent proceeding confidently on a
false green. Every other diagnostic memkit has answers one question well and
goes quiet about the rest: `--debug-config` prints what resolved and stays
green over a corpus that retrieval cannot see, `--search` proves the store and
says nothing about the hook that serves prompts, and `claude plugin details`
reports a registered hook on a plugin that is switched off. An adopter holding
three green lights and no pointers has no next move, and both walkthroughs
this design was written from spent their time inventing one.

So the shape is a report of MANY checks with a closed status vocabulary rather
than one verdict, and the vocabulary is what an agent branches on:

    PASS                    earned, on evidence this run collected
    INFO                    a fact worth stating that blocks nothing
    ASSUMPTIONS-UNVERIFIED  a claim this build cannot check here
    UNKNOWN                 the check could not be answered at all
    FAIL                    retrieval is broken, or will be

ALL-GREEN IS ZERO `FAIL`, not zero non-PASS, and that is a decision rather than
a convenience. The harness version stamp mismatches for every adopter who is
not on the pinned build, and a criterion that counted it would be unreachable
for almost everybody — which makes the whole report unreadable, because the
one thing a reader takes from it is whether anything is wrong.

READ-ONLY, with one disclosed exception: `hook-path` executes the installed
wrapper, because a fixed-query retrieval proves the store and not the path that
serves pointers. What that run touches is its own derived state, and the
`state-dir` check says so. Read-only means no store write, no config write, no
settings write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable

from memkit.memory_prompt_recall import (
    CONFIG_ENV,
    CONFIG_ROUTES,
    GENERATED_CONFIG_NAME,
    INIT_JOURNAL_NAME,
    PLUGIN_CONFIG_ROUTES,
    PLUGIN_DATA_ENV,
    PLUGIN_ENV,
    SCHEMA,
    ConfigError,
    _display_path,
    _state_dir_candidate,
    load_config,
    sanitize,
)

# The envelope's own version, and NOT the config's `SCHEMA`. They are two
# different contracts with two different readers — a config this build cannot
# speak is a FAIL inside an envelope that parsed fine — and one number for both
# would make a config migration look like a doctor migration to every consumer.
ENVELOPE_SCHEMA = 1

# R4's closed set. Anything outside it is a status an agent has no branch for,
# which is the same as no answer.
PASS = "PASS"  # noqa: S105 - a check status, not a credential
INFO = "INFO"
UNVERIFIED = "ASSUMPTIONS-UNVERIFIED"
UNKNOWN = "UNKNOWN"
FAIL = "FAIL"
STATUSES = (PASS, INFO, UNVERIFIED, UNKNOWN, FAIL)

# Who may act on a remedy. An agent may act only on `agent` and only when the
# check is not terminal; a `user` remedy is relayed to the human and the agent
# stops. The split is not about difficulty — it is about consent: every remedy
# that changes the harness's own configuration, or that decides what an
# every-prompt hook reads, belongs to the person.
AGENT = "agent"
USER = "user"
ACTORS = (AGENT, USER)

# What the human column says for each status. Deliberately shorter than the
# machine word: the report is read in a terminal, in a column, by somebody
# scanning for the one line that is not OK.
LABELS = {
    PASS: "OK",
    INFO: "INFO",
    UNVERIFIED: "UNVERIFIED",
    UNKNOWN: "UNKNOWN",
    FAIL: "FAIL",
}
_LABEL_WIDTH = max(len(v) for v in LABELS.values())

# Every string in this envelope is bounded where it is BUILT, not where it is
# printed. Details quote adopter-controlled text — a config path, a memory's
# description, the tail of an error log — and the envelope is relayed into a
# model's context by the skill that runs it. A bound applied at render time
# would leave the `--json` consumer holding the unbounded copy.
#
# Bytes rather than characters, because that is what a context window and a
# pipe both measure, and a CJK detail is three times its own length.
#
# EVIDENCE GOES FIRST in every detail that carries any — the paths, the values,
# the counts — and prose after, because a bound cuts from the end. A detail
# whose two paths are the whole point and whose second one was truncated away
# is worse than a shorter message: it reads as complete.
DETAIL_MAX_BYTES = 600


def _bound(text: str) -> str:
    """One display string, sanitized and bounded, in that order.

    Sanitizing after bounding would let a truncation land inside an escape
    sequence and produce a string the sanitizer never saw whole.
    """
    text = sanitize(text)
    raw = text.encode("utf-8")
    if len(raw) <= DETAIL_MAX_BYTES:
        return text
    # `errors="ignore"` is what makes the cut safe on a multi-byte boundary:
    # the partial codepoint at the end is dropped rather than replaced, so the
    # result is text that was really in the original.
    return raw[: DETAIL_MAX_BYTES - 3].decode("utf-8", "ignore") + "..."


class Check:
    """One question, its answer, and what to do about it.

    SANITIZED AT CONSTRUCTION rather than at render, so there is no way to
    build a check whose detail reaches a reader unsanitized — the report and
    the JSON are two renderings of these objects and neither gets a second
    chance to apply it. Doctor's report is relayed verbatim into a model's
    context and read by a human, which makes it the third of memkit's
    model-facing surfaces alongside the prompt block and the task prompt.
    """

    __slots__ = ("id", "status", "detail", "remedy", "actor", "terminal")

    def __init__(
        self,
        id: str,
        status: str,
        detail: str,
        remedy: str = "",
        actor: str = AGENT,
        terminal: bool = False,
    ) -> None:
        assert status in STATUSES, status
        assert actor in ACTORS, actor
        self.id = id
        self.status = status
        self.detail = _bound(detail)
        self.remedy = _bound(remedy)
        self.actor = actor
        self.terminal = terminal

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
            "actor": self.actor,
            "terminal": self.terminal,
        }


# The checks this build runs, in the order the report prints them. An explicit
# tuple rather than the registry's insertion order, because the producers are
# spread over the file and a report whose order followed the source would
# reshuffle whenever a function moved. The ids are STABLE: the skill, the
# README's triage table and the ROLLOUT verify recipe all cite them, and
# `tests/test_plugin_surface.py` pins each one to its README row.
CHECK_IDS: tuple[str, ...] = (
    "platform",
    "channel",
    "config-route",
    "config-parse",
    "config-authorship",
    "schema",
)

# id -> the function that answers it, given the machine. A producer returns a
# LIST because several of these are per-store: a passing personal-store canary
# must not be able to stand in for a project store that answers nothing.
_PRODUCERS: dict[str, Callable[[Machine], list[Check]]] = {}


def _produces(check_id: str) -> Callable:
    def register(fn: Callable[[Machine], list[Check]]) -> Callable:
        _PRODUCERS[check_id] = fn
        return fn

    return register




# --- the harness's own settings, in the scopes it reads them from ------------
#
# Measured on 2.1.241, out of the shipped binary: the managed directory is
# `/Library/Application Support/ClaudeCode` on macOS, `/etc/claude-code`
# elsewhere and `C:\Program Files\ClaudeCode` on Windows, and the file in it
# is `managed-settings.json`. The user scope is `$CLAUDE_CONFIG_DIR` when set
# and `~/.claude` otherwise; the project scopes are `.claude/settings.json` and
# `.claude/settings.local.json` under the directory the session stands in.
#
# Read rather than resolved-through: doctor reports what the harness was told,
# and reimplementing the harness's precedence would make this a second opinion
# about a question the harness has already answered. What the precedence order
# below is for is naming WHICH file to edit — a remedy that said "your
# settings" over four candidate files is a remedy nobody can act on.
CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
SETTINGS_NAME = "settings.json"
LOCAL_SETTINGS_NAME = "settings.local.json"
MANAGED_SETTINGS_NAME = "managed-settings.json"
# The plugin, spelled the way `enabledPlugins` and `pluginConfigs` key it:
# `<plugin>@<marketplace>`. Both halves are `memkit`, which is a coincidence of
# naming and not a rule, so it is written once here.
PLUGIN_KEY = "memkit@memkit"
OPTION_KEY = "memkitConfig"


def _managed_dir() -> str:
    if sys.platform == "darwin":
        return "/Library/Application Support/ClaudeCode"
    return "/etc/claude-code"


class Settings:
    """One settings file: where it is, what it holds, and why it does not.

    A file that is present and unparseable is its own state, and it is the
    field anti-pattern the prior-art survey names: a harness that meets a parse
    error and silently replaces the file with a stub takes the adopter's
    configuration with it. Doctor never repairs one; it says which file and
    what the parser said.
    """

    __slots__ = ("scope", "path", "data", "error")

    def __init__(self, scope: str, path: str) -> None:
        self.scope = scope
        self.path = path
        self.data: dict = {}
        self.error = ""
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return
        if not isinstance(blob, dict):
            self.error = "top level is not an object"
            return
        self.data = blob

    @property
    def present(self) -> bool:
        return os.path.isfile(self.path)


def settings_scopes() -> list[Settings]:
    """Every scope, most authoritative first."""
    user = os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")
    cwd = os.getcwd()
    return [
        Settings("managed", os.path.join(_managed_dir(), MANAGED_SETTINGS_NAME)),
        Settings("user", os.path.join(user, SETTINGS_NAME)),
        Settings("project", os.path.join(cwd, ".claude", SETTINGS_NAME)),
        Settings("local", os.path.join(cwd, ".claude", LOCAL_SETTINGS_NAME)),
    ]


def authored_configs(state_dir: str) -> set:
    """The absolute config paths init's journal claims to have written.

    The journal is append-only JSONL and a partial line is a crash, not a
    corruption: a record that does not parse is skipped rather than taken as
    evidence that nothing was authored. Reading it the other way would turn one
    interrupted init into a `config-authorship` FAIL against memkit's own file.
    """
    out = set()
    path = os.path.join(state_dir, INIT_JOURNAL_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("authored_config"):
                    claimed = record.get("path")
                    if isinstance(claimed, str):
                        out.add(claimed)
    except OSError:
        return out
    return out


class Machine:
    """What the checks read, resolved once for the whole run.

    One object rather than each producer reaching for `os.environ` itself, for
    the reason `_config_state` exists in the hook: two surfaces deriving the
    same answer separately is how they come to disagree, and a diagnostic whose
    halves disagree is worse than no diagnostic.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.explicit_config = config_path
        self.settings = settings_scopes()
        self.state_dir = _state_dir_candidate()
        # The config the WRAPPER settled on, which is the whole of what doctor
        # knows about the rungs: `bin/memkit` resolves them in POSIX sh and
        # exports the answer, so re-resolving them here would be a second copy
        # of the one rule the whole design rests on — and a second copy that
        # agreed would prove nothing, while one that disagreed would be a
        # diagnostic contradicting the thing it diagnoses. What doctor does
        # instead is report the wrapper's OUTPUT against the wrapper's INPUTS,
        # which is what makes the set-but-wrong option visible: the option is
        # in the environment, and the resolved config is not.
        self.resolved_config = config_path or os.environ.get(CONFIG_ENV) or ""
        self.option_value = os.environ.get(
            "CLAUDE_PLUGIN_OPTION_" + OPTION_KEY.upper(), ""
        )
        self.plugin_data = os.environ.get(PLUGIN_DATA_ENV, "")
        self._parsed = False
        self._config = None
        self._config_error = ""

    @property
    def plugin(self) -> bool:
        return bool(os.environ.get(PLUGIN_ENV))

    @property
    def rung_two(self) -> str:
        """`$CLAUDE_PLUGIN_DATA/memkit.json`, or nothing.

        Skipped entirely when the variable is unset rather than built from an
        empty expansion, and refused when it is relative, for the two reasons
        the wrapper gives: `${unset}/memkit.json` is `/memkit.json`, and a
        relative value names whatever directory the session stands in.
        """
        if not self.plugin_data or not os.path.isabs(self.plugin_data):
            return ""
        return os.path.join(self.plugin_data, GENERATED_CONFIG_NAME)

    def settings_option(self) -> tuple:
        """The literal `memkitConfig` the harness was told, and which scope
        said so.

        THE ONLY READER THAT CAN SEPARATE THE TWO SILENT STATES. A `memkitConfig`
        typo'd by one character leaves the wrapper blanking the path before the
        hook runs, so the trust marker records `trust:unconfigured` —
        byte-identical to never-configured — and the wrapper's excellent stderr
        line is unreachable because the harness swallows hook stderr. The
        person who typed the path is the one person who can be certain a config
        was meant to exist, and this is where what they typed is written down.
        """
        for scope in self.settings:
            configs = scope.data.get("pluginConfigs")
            if not isinstance(configs, dict):
                continue
            entry = configs.get(PLUGIN_KEY)
            if not isinstance(entry, dict):
                continue
            options = entry.get("options")
            if not isinstance(options, dict):
                continue
            value = options.get(OPTION_KEY)
            if isinstance(value, str) and value:
                return value, scope
        return "", None

    def config(self):
        """The parsed config, or None, with the reason parked beside it.

        Parsed at most once: `config-parse`, `schema`, `store-roots` and every
        per-store check ask, and four parses of one file is four chances for a
        config edited mid-run to give two surfaces different answers.
        """
        if not self._parsed:
            self._parsed = True
            if not self.resolved_config:
                return None
            try:
                self._config = load_config(self.resolved_config)
            except ConfigError as exc:
                self._config_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                # `json.load` on a deeply nested document raises RecursionError,
                # which `load_config` does not convert. A config that takes the
                # diagnostic down is the one state this command may not have.
                self._config_error = f"{type(exc).__name__}: {exc}"
        return self._config

    @property
    def config_error(self) -> str:
        self.config()
        return self._config_error


# --- the machine itself ------------------------------------------------------


@_produces("platform")
def _platform(machine: Machine) -> list[Check]:
    """macOS is the platform every scenario runs on; Linux is where the
    adopters are.

    Linux is INFO rather than PASS and the wording is the whole point: nothing
    is known to break there and no scenario proves it does not. Calling it PASS
    would be this report making the claim it exists to stop other surfaces
    making.
    """
    if sys.platform == "darwin":
        return [Check("platform", PASS, "macOS, the platform the scenarios run on")]
    if sys.platform.startswith("win") or sys.platform == "cygwin":
        return [
            Check(
                "platform",
                FAIL,
                f"{sys.platform}: memkit is not supported on Windows",
                "Windows is unsupported. The wrappers are POSIX sh and the "
                "paths are POSIX paths; there is no configuration that makes "
                "this work.",
                actor=USER,
                terminal=True,
            )
        ]
    return [
        Check(
            "platform",
            INFO,
            f"{sys.platform}: unverified — nothing is known to break here, and "
            "no scenario runs here",
        )
    ]


@_produces("channel")
def _channel(machine: Machine) -> list[Check]:
    """Which install this is, because every later remedy is phrased for it.

    Three channels ship memkit and they do not share a repair: a plugin install
    is fixed with `claude plugin`, a nix install with a rebuild, and a pip one
    with pip. A remedy that guessed would send an adopter to a command their
    channel does not have — which is the failure the search-binary naming split
    exists to prevent one layer down.
    """
    if machine.plugin:
        root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
        where = f", payload at {root}" if root else ""
        return [Check("channel", INFO, f"plugin install ({PLUGIN_ENV} set){where}")]
    module = getattr(sys.modules[__name__], "__file__", "") or ""
    if module.startswith("/nix/store/"):
        return [Check("channel", INFO, "nix install (the package is in /nix/store)")]
    return [
        Check(
            "channel",
            INFO,
            "python install (pip or uvx): no plugin wrapper and no nix store "
            "path, so nothing registers a hook automatically",
        )
    ]


# --- the config, its route, and who wrote it ---------------------------------


def _rungs(machine: Machine) -> tuple:
    """The routes this channel really does consult, from the hook's own list.

    EXACTLY TWO on the plugin channel, and the count is load-bearing. A third
    rung reading a `memkit.json` beside the wrappers was deleted because a
    plugin install is a clone of a pinned commit, so a file in the payload tree
    is a file the repo can ship — and a config decides both which directories
    an every-prompt hook reads and which binary it exec's. A remedy naming a
    third rung would teach an adopter to recreate it.
    """
    return PLUGIN_CONFIG_ROUTES if machine.plugin else CONFIG_ROUTES


@_produces("config-route")
def _config_route(machine: Machine) -> list[Check]:
    """Which route answered, and — the half nothing else in the product can
    do — what the option SAYS versus what resolved.

    The set-but-wrong `memkitConfig` is the highest-cost silent state in the
    field log: the install succeeds, `plugin details` still reports `Hooks (1)`,
    no soak record is written at all, and the trust marker records
    `trust:unconfigured` — the same bytes a never-configured install writes. The
    two want opposite remedies. One wants `/memkit:init`; the other wants one
    character fixed in a path the adopter already typed once.
    """
    option, scope = machine.settings_option()
    routes = ", ".join(_rungs(machine))
    where = f", set in {scope.scope} settings" if scope else ""

    if option and machine.resolved_config != os.path.expanduser(option):
        # The option is set and did not answer. Either it names something that
        # is not there, or something else won — and the detail says which,
        # because the two are different repairs.
        if machine.resolved_config:
            return [
                Check(
                    "config-route",
                    FAIL,
                    f'option: "{option}"{where}. In use: '
                    f'"{_display_path(machine.resolved_config)}". Two answers '
                    "to one question, and the hook takes the second",
                    "Decide which config this install serves and make the "
                    "option name it, or clear the other route. A hook reading "
                    "a config the option does not name is a hook reading "
                    "directories nobody pointed it at.",
                    actor=USER,
                )
            ]
        expanded = os.path.expanduser(option)
        why = (
            "exists but cannot be read by this process"
            if os.path.exists(expanded)
            else "does not exist"
        )
        return [
            Check(
                "config-route",
                FAIL,
                f'option: "{option}"{where}, which {why}. In use: nothing. '
                "This install is inert, and that is byte-identical to never "
                "having been configured",
                f"Reinstall with the corrected path, or edit {OPTION_KEY} in "
                f"{scope.path if scope else 'your settings'}. The install "
                "itself is fine; the path is one character off.",
                actor=USER,
            )
        ]

    if not machine.resolved_config:
        if machine.plugin:
            return [
                Check(
                    "config-route",
                    FAIL,
                    "no config on either rung this install reads "
                    f"({routes}), so it is inert: no stores, no pointers, "
                    "exit 0 on every prompt",
                    "Run /memkit:init, which writes a config and points this "
                    "install at it.",
                    actor=USER,
                )
            ]
        return [
            Check(
                "config-route",
                FAIL,
                f"no config on any route this install reads ({routes}), so it "
                "is inert: no stores, no pointers, exit 0 on every prompt",
                "Run /memkit:init, or write the config by hand and name it "
                f"with --config or ${CONFIG_ENV}.",
                actor=USER,
            )
        ]

    if machine.explicit_config:
        rung = "--config, this invocation only"
    elif option:
        rung = f"the {OPTION_KEY} install option{where}"
    elif machine.rung_two and machine.resolved_config == machine.rung_two:
        rung = f"${PLUGIN_DATA_ENV}/{GENERATED_CONFIG_NAME}"
    else:
        rung = f"${CONFIG_ENV}"
    return [
        Check(
            "config-route",
            INFO,
            f'"{_display_path(machine.resolved_config)}", via {rung}. Routes '
            f"this channel consults: {routes}",
        )
    ]


@_produces("config-parse")
def _config_parse(machine: Machine) -> list[Check]:
    """A config that is present and cannot be honoured is never green.

    The error string is the CLI's own, verbatim, because it names the file, the
    field and the cause — and a diagnostic that paraphrased would be a second
    wording of a message the adopter may already have seen somewhere else.
    """
    if not machine.resolved_config:
        return [
            Check(
                "config-parse",
                UNKNOWN,
                "no config resolved, so there is nothing to parse",
            )
        ]
    if machine.config_error:
        return [
            Check(
                "config-parse",
                FAIL,
                machine.config_error,
                "Fix the file the message names. Until it parses this install "
                "is inert, and the hook is fail-open, so nothing else says so.",
                actor=USER,
            )
        ]
    cfg = machine.config()
    if cfg is None:
        return [
            Check("config-parse", UNKNOWN, "the config could not be loaded")
        ]
    return [
        Check(
            "config-parse",
            PASS,
            f"{cfg.path} parses; schema {SCHEMA}, {len(cfg.stores)} store(s)",
        )
    ]


@_produces("config-authorship")
def _config_authorship(machine: Machine) -> list[Check]:
    """A rung-2 config nobody claims to have written.

    `$CLAUDE_PLUGIN_DATA` is harness-owned and payload-WRITABLE — memkit's own
    hook writes `trust.json` there — so a release could write a `memkit.json`
    beside it on one prompt and be honoured by every later, clean release. The
    escalation over "a malicious payload already runs code" is persistence and
    laundering, and it is real.

    Init never writes that file. What it writes is the journal entry claiming
    the configs it did author, which is what makes an UNCLAIMED one detectable
    at all.
    """
    path = machine.rung_two
    if not path:
        return [
            Check(
                "config-authorship",
                PASS,
                f"no ${PLUGIN_DATA_ENV} rung on this install, so there is no "
                "payload-writable config to claim",
            )
        ]
    if not os.path.exists(path):
        return [
            Check(
                "config-authorship",
                PASS,
                f"{path} does not exist, which is what every install memkit "
                "wrote looks like",
            )
        ]
    if path in authored_configs(machine.state_dir):
        return [
            Check(
                "config-authorship",
                PASS,
                f"{path} exists and memkit's init journal claims it",
            )
        ]
    return [
        Check(
            "config-authorship",
            FAIL,
            f"{path} exists and no init journal entry claims it. memkit did "
            "not write this file. It sits in a directory the plugin payload "
            "can write to, and it decides which directories the every-prompt "
            "hook reads",
            f"Read {path}. If you wrote it, that is fine and this check "
            "cannot know. If you did not, delete it: something with write "
            "access to the plugin data directory put it there.",
            actor=USER,
        )
    ]


@_produces("schema")
def _schema(machine: Machine) -> list[Check]:
    """The config's declared schema against the one this build speaks.

    Read out of the RAW file rather than off the parsed object, because a
    mismatch is exactly the case where there is no parsed object: `Config`
    refuses a number it does not speak, so a check that read the parse would
    only ever be able to report agreement.

    Nothing here bumps `SCHEMA`, and `--migrate` is out of this milestone, so
    the remedy names the BUILD: install the memkit that speaks the config's
    number, rather than editing the number in the file.
    """
    if not machine.resolved_config:
        return [Check("schema", UNKNOWN, "no config resolved")]
    try:
        with open(machine.resolved_config, encoding="utf-8") as f:
            raw = json.load(f)
        declared = raw.get("schema") if isinstance(raw, dict) else None
    except (OSError, ValueError) as exc:
        return [
            Check(
                "schema",
                UNKNOWN,
                f"the config could not be read to find its schema: {exc}",
            )
        ]
    if declared == SCHEMA:
        return [Check("schema", PASS, f"config schema {declared}, build {SCHEMA}")]
    return [
        Check(
            "schema",
            FAIL,
            f"config schema {declared!r}, this build speaks {SCHEMA}",
            "Install the memkit build that speaks this config's schema. "
            "Editing the number in the file does not change what the fields "
            "mean.",
            actor=USER,
        )
    ]


# --- the envelope ------------------------------------------------------------


def verdict(checks: list[Check]) -> str:
    """One line, and the rule behind it is load-bearing.

    Counting non-PASS instead of FAIL makes green unreachable: `harness-stamp`
    mismatches for every adopter off the pinned build, `channel` is always
    INFO, and `subagent-delivery` is UNKNOWN until the subagent path ships. A
    criterion nobody can satisfy is a criterion nobody reads.

    The unverified count is reported anyway, because "nothing is broken" and
    "nothing is broken that I could check" are different sentences and the
    reader is entitled to both.
    """
    fails = sum(1 for c in checks if c.status == FAIL)
    unverified = sum(1 for c in checks if c.status in (UNVERIFIED, UNKNOWN))
    if fails == 0:
        return "OK"
    return f"PROBLEMS: {fails} FAIL, {unverified} unverified"


def report(checks: list[Check], line: str) -> str:
    """The human text, rendered FROM the checks and from nothing else.

    This function takes the list; it does not go and ask the machine again.
    That is the property the whole envelope rests on — a report derived from a
    second pass could disagree with the checks beside it, and the disagreement
    would be invisible because each half is individually plausible. Here a
    divergence is not a bug that testing might miss; it has nowhere to come
    from.
    """
    lines = []
    width = max([len(c.id) for c in checks] + [1])
    for check in checks:
        lines.append(
            f"{LABELS[check.status].ljust(_LABEL_WIDTH)}  "
            f"{check.id.ljust(width)}  {check.detail}".rstrip()
        )
    lines.append("")
    lines.append("VERDICT: " + line)
    remedies = [c for c in checks if c.status != PASS and c.remedy]
    if remedies:
        lines.append("")
        lines.append("What to do")
        for check in remedies:
            # WHO acts, on every remedy line. An agent that acted on a `user`
            # remedy would be editing the harness's own configuration on its
            # own authority, and the JSON carries the same field for the same
            # reason.
            lines.append(
                f"  {check.id} [{check.actor}] {check.remedy}"
            )
    return "\n".join(lines)


def envelope(checks: list[Check], ran_at: int | None = None) -> dict:
    line = verdict(checks)
    return {
        "schema": ENVELOPE_SCHEMA,
        "verdict": line,
        "ran_at": int(time.time()) if ran_at is None else ran_at,
        "report": report(checks, line),
        "checks": [c.as_dict() for c in checks],
    }


def collect(machine: Machine, wanted: list[str] | None = None) -> list[Check]:
    """Every requested check, in declared order.

    A producer that raises is a check that answered UNKNOWN, never a doctor
    that died: the reader is somebody whose install is already misbehaving, and
    a traceback in place of the other twenty answers is the worst thing this
    command can do. The exception type is named so the failure is reportable
    rather than merely survived.
    """
    out: list[Check] = []
    for check_id in CHECK_IDS:
        if wanted is not None and check_id not in wanted:
            continue
        producer = _PRODUCERS[check_id]
        try:
            out.extend(producer(machine))
        except Exception as exc:  # noqa: BLE001 - see the docstring
            out.append(
                Check(
                    check_id,
                    UNKNOWN,
                    f"the check itself failed: {type(exc).__name__}: {exc}",
                    "This is a defect in memkit, not in your setup. The other "
                    "checks in this report still stand.",
                )
            )
    return out


SUMMARY = "report whether retrieval is actually working on this machine"

EPILOG = """\
Statuses: PASS / INFO / ASSUMPTIONS-UNVERIFIED / UNKNOWN / FAIL.
All-green is zero FAIL — INFO, ASSUMPTIONS-UNVERIFIED and UNKNOWN never block.
An agent may act only on a check whose actor is `agent` and whose `terminal` is
false; every other remedy is for the person to read and decide.

Exit codes: 0 when the verdict is OK, 1 when any check FAILs."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="the whole envelope on stdout, including the human report",
    )
    parser.add_argument(
        "--check",
        action="append",
        metavar="ID",
        dest="only",
        help="run only this check; repeatable",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="the memkit config to diagnose (default: whatever this install "
        "resolves)",
    )


EXIT_OK = 0
EXIT_PROBLEMS = 1
# argparse's, and the dispatcher's, and not reassignable: naming a check that
# does not exist IS a usage error, and the alternative — exiting 1 — would tell
# a caller its install is broken when its argument was.
EXIT_USAGE = 2


def run(args: argparse.Namespace, extra: list[str] | None = None) -> int:
    # The dispatcher parses with `parse_known_args` while a pending subcommand
    # still has to survive flags it does not declare, so an argument this
    # parser did not recognise arrives here rather than being refused. Refusing
    # it is the point: `memkit doctor --jsn` silently running a full doctor and
    # printing the human report is a caller that believes it got JSON.
    if extra:
        print(
            "memkit doctor: unrecognised arguments: " + " ".join(extra),
            file=sys.stderr,
        )
        return EXIT_USAGE
    wanted = getattr(args, "only", None)
    if wanted:
        unknown = [w for w in wanted if w not in CHECK_IDS]
        if unknown:
            print(
                "memkit doctor: no such check: "
                + ", ".join(sorted(unknown))
                + "\nchecks: "
                + ", ".join(CHECK_IDS),
                file=sys.stderr,
            )
            return EXIT_USAGE
    machine = Machine(getattr(args, "config", None))
    checks = collect(machine, wanted)
    blob = envelope(checks)
    if getattr(args, "as_json", False):
        print(json.dumps(blob, indent=2, sort_keys=False))
    else:
        print(blob["report"])
    return EXIT_OK if blob["verdict"] == "OK" else EXIT_PROBLEMS
