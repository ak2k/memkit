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
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

from memkit.memory_prompt_recall import (
    BUILD_BUSY,
    BUILD_OK,
    BUILD_PARTIAL,
    BUILD_REBUILT,
    BUILD_UNREADABLE,
    CONFIG_ENV,
    CONFIG_ROUTES,
    DOCTOR_ENV,
    EXCLUDE_BASENAMES,
    FRAME_TAG,
    GENERATED_CONFIG_NAME,
    INIT_JOURNAL_NAME,
    MARKER_NAME,
    PLUGIN_CONFIG_ROUTES,
    PLUGIN_DATA_ENV,
    PLUGIN_ENV,
    SCHEMA,
    SOAK_LOG_NAME,
    SWEEP_STAMP_NAME,
    TASK_OUTCOME_PREFIX,
    ConfigError,
    _corpus_files,
    _cwd_digest,
    _display_path,
    _fts_db,
    _search_root,
    _session_state_path,
    _state_dir_candidate,
    _store_live_dir,
    _version,
    load_config,
    recall,
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
    "store-roots",
    "corpus-root",
    "index-state",
    "canary-retrieval",
    "hook-path",
    "hook-ever-fired",
    "gate-outcomes",
    "plugin-enabled",
    "registrations-count",
    "plugin-diagnostics",
    "subagent-delivery",
    "harness-stamp",
    "auto-memory",
    "build",
    "interpreter",
    "state-dir",
    "hooks-layout",
    "uninstall-story",
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
        # The checker route, probed at most once per process on the channels
        # that have no wrapper to have decided it already.
        self._route: tuple | None = None
        # Set by the one check that executes anything, and read by the one that
        # reports what is left behind. A disclosure that was printed whether or
        # not the run happened is a disclosure nobody can rely on — `--check
        # state-dir` on its own runs no hook.
        self.hook_probed = False

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


# --- the stores, the corpus, and the index -----------------------------------


# What may sit at a store root beside `search/` without being a stranded
# memory. A `README.md` explaining the store to a human is a legitimate file to
# keep there, and the ledgers are already excluded from retrieval everywhere.
# Everything else at that level is a file its author expected to be retrieved.
EXCLUDE_STRAY = frozenset({"README.md", "CONTRIBUTING.md", "LICENSE.md"})

# The canary memory init seeds, and the query doctor asks for it. The query
# lives HERE rather than with init because init verifies its own work through
# it: one spelling, so a canary that init cannot find is a canary that is
# really not there rather than two commands asking different questions.
CANARY_NAME = "memkit-canary.md"


def canary_query(nonce: str) -> str:
    """The fixed query, which can only be answered by the file init wrote.

    Three terms, not one. The prompt gate drops anything under two content
    words, so a bare nonce retrieves nothing and the check would fail on every
    healthy install — which is the false RED that matches this design's false
    green.
    """
    return f"memkit canary {nonce}"


def _stranded(root: str) -> list[str]:
    """Markdown a tiered store no longer retrieves, because it sits above
    `search/`.

    The single most expensive silent state in the field log. Creating
    `search/` in a flat store un-retrieves everything above it in one step —
    which is what the agent-writes-memories recipe causes on the first memory
    an agent writes — and every diagnostic stayed green while it happened.
    Three of four reviewers hit it and two lost the memory the quick start had
    just had them create.

    Only the store's own top level: a directory beside `search/` is somebody's
    own filing and not a tier this build knows about, and naming those would
    make the check unusable on the store it exists to protect.
    """
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".md"):
            continue
        if name in EXCLUDE_BASENAMES or name in EXCLUDE_STRAY:
            continue
        if os.path.isfile(os.path.join(root, name)):
            out.append(name)
    return out


@_produces("store-roots")
def _store_roots(machine: Machine) -> list[Check]:
    """Every store, where it resolves, and how it is gated.

    Without this the config could point anywhere and pass every other check —
    `config-parse` says the file is well-formed and says nothing about what it
    names, and the roots are resolved LAZILY, so a store naming a root the
    config never defines raises only when something asks for it.
    """
    cfg = machine.config()
    if cfg is None:
        return [Check("store-roots", UNKNOWN, "no config to resolve roots from")]
    if not cfg.stores:
        return [
            Check(
                "store-roots",
                FAIL,
                f"{_display_path(cfg.path)} declares no stores, so this "
                "install has nothing to search and never will",
                "Add a store to the config, or run /memkit:init to write one.",
                actor=USER,
            )
        ]
    lines = []
    broken = []
    for store in cfg.stores:
        try:
            root, source = cfg.root_with_source(store.live_root)
        except ConfigError as exc:
            broken.append(str(exc))
            continue
        gate = store.cwd_gate or "ungated"
        edit = "same" if store.edit_root == store.live_root else store.edit_root
        lines.append(
            f"{store.id} ({store.role}): {_display_path(os.path.join(root, store.dir))}"
            f" [root {store.live_root} via {source}; edit_root {edit}; "
            f"cwd_gate {gate}]"
        )
    if broken:
        return [
            Check(
                "store-roots",
                FAIL,
                "; ".join(broken),
                "A store names a root the config does not define. Retrieval "
                "raises when something asks for it, which on the hook path is "
                "a silent no-match.",
                actor=USER,
            )
        ]
    return [Check("store-roots", INFO, "; ".join(lines))]


@_produces("corpus-root")
def _corpus_root(machine: Machine) -> list[Check]:
    """One row per store: what retrieval would actually read under it.

    Per store rather than once, because the states are per store and they do
    not share a remedy — a gated project store standing outside its own repo is
    working correctly, and a store whose directory nobody created is not.
    """
    cfg = machine.config()
    if cfg is None:
        return [Check("corpus-root", UNKNOWN, "no config to resolve stores from")]
    searched = cfg.searched_stores()
    out = []
    for store in cfg.stores:
        if store not in searched:
            out.append(
                Check(
                    "corpus-root",
                    INFO,
                    f"{store.id}: gated to {store.cwd_gate}, so this session "
                    "does not read it. That is the gate working",
                )
            )
            continue
        live = cfg.store_dir(store, "live")
        if not os.path.isdir(live):
            out.append(
                Check(
                    "corpus-root",
                    FAIL,
                    f"{store.id}: {_display_path(live)} is not a directory, so "
                    "the store is configured and not on disk",
                    f"Create {_display_path(live)}/search/ and put memories in "
                    "it, or run /memkit:init.",
                    actor=USER,
                )
            )
            continue
        root = _search_root(live)
        tiered = root != live
        stranded = _stranded(live) if tiered else []
        if stranded:
            out.append(
                Check(
                    "corpus-root",
                    FAIL,
                    f"{store.id}: {', '.join(stranded)} sit at "
                    f"{_display_path(live)} while the corpus root is "
                    f"{_display_path(root)}. Retrieval reads none of them",
                    "Move those files into search/, or point the store's `dir` "
                    "at the directory that holds them. Nothing else reports "
                    "this: every other surface stays green over a store that "
                    "is mostly dark.",
                    actor=USER,
                )
            )
            continue
        files = _corpus_files(root)
        if not files:
            out.append(
                Check(
                    "corpus-root",
                    INFO,
                    f"{store.id}: {_display_path(root)} holds no markdown "
                    "retrieval would consider, so this store answers nothing",
                )
            )
            continue
        layout = "tiered (search/)" if tiered else "flat — no search/ yet"
        out.append(
            Check(
                "corpus-root",
                PASS,
                f"{store.id}: {files} file(s) under {_display_path(root)}, "
                f"{layout}",
            )
        )
    return out or [Check("corpus-root", UNKNOWN, "no stores")]


def _build_record(root: str) -> tuple:
    """The `.build` sidecar for one corpus root: (record, error).

    THE INDEX IS NEVER OPENED. Opening it syncs it, and a sync rebuilds
    whatever the walk finds stale — a diagnostic that repairs the state it is
    measuring cannot report on it, and "never indexed" and "indexed, and the
    corpus turned out to be empty" would become the same answer again.
    """
    sidecar = _fts_db(root)[: -len(".db")] + ".build"
    try:
        with open(sidecar, encoding="utf-8") as f:
            return json.load(f), sidecar, ""
    except FileNotFoundError:
        return None, sidecar, ""
    except (OSError, ValueError) as exc:
        return None, sidecar, str(exc)


@_produces("index-state")
def _index_state(machine: Machine) -> list[Check]:
    """How each corpus root was LAST indexed, read out of the sidecar.

    Honours the sidecar's own reader's rule, which is a contract rather than
    advice: an outcome this build does not recognise is treated as NOT-OK, and
    `files` is read as a corpus census only under `ok`. That rule is what lets
    the outcome vocabulary grow without an older reader mistaking a new failure
    state for a healthy one.
    """
    cfg = machine.config()
    if cfg is None:
        return [Check("index-state", UNKNOWN, "no config to resolve stores from")]
    searched = cfg.searched_stores()
    out = []
    for store in cfg.stores:
        live = _store_live_dir(cfg, store, searched)
        if live is None:
            continue
        record, sidecar, error = _build_record(live)
        if error:
            out.append(
                Check(
                    "index-state",
                    UNKNOWN,
                    f"{store.id}: {_display_path(sidecar)} could not be read "
                    f"({error}), so how this corpus was last indexed is not "
                    "knowable from here",
                )
            )
            continue
        if record is None:
            out.append(
                Check(
                    "index-state",
                    UNKNOWN,
                    f"{store.id}: no index record at {_display_path(sidecar)} "
                    "— this corpus has never been indexed, which is what an "
                    "install that has never served a prompt looks like",
                )
            )
            continue
        outcome = record.get("outcome") if isinstance(record, dict) else None
        files = record.get("files") if isinstance(record, dict) else None
        if outcome == BUILD_OK:
            out.append(
                Check(
                    "index-state",
                    PASS,
                    f"{store.id}: last index ok over {files} file(s)",
                )
            )
        elif outcome == BUILD_UNREADABLE:
            out.append(
                Check(
                    "index-state",
                    FAIL,
                    f"{store.id}: the corpus could not be read at all on the "
                    "last index, so the index holds nothing for it",
                    "Check the permissions on the store directory. Retrieval "
                    "answers nothing here and reports no error.",
                    actor=USER,
                )
            )
        elif outcome in (BUILD_BUSY, BUILD_REBUILT, BUILD_PARTIAL):
            out.append(
                Check(
                    "index-state",
                    INFO,
                    f"{store.id}: last index {outcome}; the file count is a "
                    "floor rather than a census under this outcome",
                )
            )
        else:
            # The reader's rule: unrecognised is NOT-OK, and `files` is not a
            # census under it. A newer build wrote this record.
            out.append(
                Check(
                    "index-state",
                    UNKNOWN,
                    f"{store.id}: index outcome {outcome!r}, which this build "
                    "does not recognise. Not read as ok, and the file count "
                    "is not read as a census",
                )
            )
    return out or [Check("index-state", UNKNOWN, "no store offers a corpus here")]


@_produces("canary-retrieval")
def _canary_retrieval(machine: Machine) -> list[Check]:
    """One row per store this session can search: does init's own memory come
    back?

    Per store, because a passing personal-store canary would otherwise stand in
    for a project store that is gated, missing or unindexed — and the personal
    store is the one that passes from anywhere.

    The query is the nonce, so this can only be answered by the file init
    wrote. A fixed phrase would match the adopter's own memories and pass while
    proving nothing.
    """
    cfg = machine.config()
    if cfg is None:
        return [Check("canary-retrieval", UNKNOWN, "no config")]
    if not cfg.canary_nonce:
        return [
            Check(
                "canary-retrieval",
                UNKNOWN,
                "this config records no canary nonce, so there is no memory "
                "whose retrieval proves the store rather than the corpus. "
                "Configs written before init, and by hand, have none",
            )
        ]
    searched = cfg.searched_stores()
    out = []
    for store in cfg.stores:
        live = _store_live_dir(cfg, store, searched)
        if live is None:
            continue
        hits = recall(canary_query(cfg.canary_nonce), dirs=[live])
        found = [h for h in hits if os.path.basename(h) == CANARY_NAME]
        if found:
            out.append(
                Check(
                    "canary-retrieval",
                    PASS,
                    f"{store.id}: {_display_path(found[0])} came back for the "
                    "fixed query",
                )
            )
        else:
            out.append(
                Check(
                    "canary-retrieval",
                    FAIL,
                    f"{store.id}: nothing came back for the fixed query over "
                    f"{_display_path(live)}. The store is configured and does "
                    "not answer",
                    f"Check that {CANARY_NAME} is under this store's search/ "
                    "directory and carries the config's canary_nonce in its "
                    "description. Re-running /memkit:init reseeds it.",
                    actor=USER,
                )
            )
    return out or [
        Check(
            "canary-retrieval",
            UNKNOWN,
            "no store offers a corpus in this directory, so nothing was "
            "queried",
        )
    ]


# --- the path that actually serves pointers ----------------------------------


# How long doctor waits for the installed hook. The registration's own timeout
# is 15 seconds and a first run on a cold index spends most of it building, so
# a bound at the registration's number would report a healthy install as a
# hang. Longer, and the elapsed time is reported either way — a latency figure
# on the adopter's own corpus is one of the things the field survey asked for
# and the author's number cannot supply.
HOOK_PROBE_TIMEOUT = 25

# The wrapper a plugin install puts the registration on. Read from
# `CLAUDE_PLUGIN_ROOT` rather than from `hooks.json`, because that variable is
# what the harness itself expands the registration against — and because the
# root arrives with a TRAILING SLASH (measured on 2.1.238 and still true on
# 2.1.241), which `os.path.join` absorbs and naive string arithmetic does not.
HOOK_WRAPPER = "bin/memkit-hook"


def _installed_hook(machine: Machine) -> tuple:
    """The command the harness would run on a prompt, and how it was found.

    THE POINT OF THE CHECK IS THAT IT IS THE INSTALLED ONE. Running the module
    in this process would prove that retrieval works and nothing about the
    wrapper, the interpreter it resolves, or the registration that reaches it —
    which is the whole span between a store that answers and a session that
    stays quiet.

    A registered command that is not a bare executable path is reported rather
    than run: the harness hands it to a shell, and a diagnostic that evaluated
    a shell fragment out of a settings file would be executing whatever that
    file says on a machine whose configuration is already in doubt.
    """
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if root:
        path = os.path.join(root, HOOK_WRAPPER)
        if os.path.isfile(path):
            return [path], f"the plugin's own wrapper at {_display_path(path)}"
        return [], f"{_display_path(path)} is not there"
    for scope in machine.settings:
        events = scope.data.get("hooks")
        if not isinstance(events, dict):
            continue
        for entry in events.get("UserPromptSubmit") or []:
            for spec in (entry or {}).get("hooks") or []:
                command = (spec or {}).get("command")
                if not isinstance(command, str) or "memkit" not in command:
                    continue
                if os.path.isfile(command) and os.access(command, os.X_OK):
                    return [command], f"the {scope.scope}-settings registration"
                return (
                    [],
                    f"the {scope.scope}-settings registration runs "
                    f'"{command}", which is not an executable file this can '
                    "run on its own",
                )
    return [], "nothing registers a UserPromptSubmit hook for memkit"


def _probe_hook(machine: Machine, command: list, prompt: str) -> tuple:
    """One real run of the installed hook. Returns (stdout, stderr, code, ms).

    Against the REAL state directory and the REAL config, which is the whole
    value of it: a scratch cache would keep doctor's footprint at zero and
    would also force a cold index build on every run and report green over an
    index the adopter never uses. What that costs is one soak record and
    possibly one sweep, and the `state-dir` check says so.

    The session id is fresh per run and the state it leaves is removed after,
    because the hook offers each memory once per session: a fixed id would make
    the SECOND doctor run report no pointer, which is a false FAIL on a healthy
    install.
    """
    machine.hook_probed = True
    session = "memkit-doctor-" + secrets.token_hex(6)
    payload = json.dumps(
        {
            "session_id": session,
            "prompt": prompt,
            "cwd": os.getcwd(),
            "hook_event_name": "UserPromptSubmit",
        }
    )
    started = time.monotonic()
    try:
        out = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=HOOK_PROBE_TIMEOUT,
            env=dict(os.environ, **{DOCTOR_ENV: "1"}),
        )
    except subprocess.TimeoutExpired:
        return "", "", None, int((time.monotonic() - started) * 1000)
    except OSError as exc:
        return "", str(exc), None, int((time.monotonic() - started) * 1000)
    finally:
        _forget_probe_session(session)
    return out.stdout, out.stderr, out.returncode, int(
        (time.monotonic() - started) * 1000
    )


def _forget_probe_session(session: str) -> None:
    """Remove the session ledger the probe's own run wrote.

    Best-effort, and not a substitute for the sweep: what this prevents is one
    file per doctor invocation accumulating in a directory whose growth is
    itself one of the things doctor reports on.
    """
    state = _session_state_path(session)
    with contextlib.suppress(OSError):
        os.unlink(state)
    stem = state[: -len(".json")] if state.endswith(".json") else state
    parent = os.path.dirname(stem) or "."
    base = os.path.basename(stem) + ".dup-"
    with contextlib.suppress(OSError):
        for name in os.listdir(parent):
            if name.startswith(base):
                with contextlib.suppress(OSError):
                    os.unlink(os.path.join(parent, name))


@_produces("hook-path")
def _hook_path(machine: Machine) -> list[Check]:
    """Run the installed hook once, and say whether a pointer came out.

    DOCTOR MAY NEVER REPORT GREEN WITHOUT THIS. A fixed-query retrieval proves
    the store; it proves nothing about the wrapper that finds the config, the
    interpreter that runs the module, or the registration that reaches either —
    and that span is exactly where both walkthroughs' installs were broken
    while every other light was green.
    """
    command, how = _installed_hook(machine)
    if not command:
        return [
            Check(
                "hook-path",
                UNKNOWN,
                f"no hook was run: {how}",
                "Nothing serves pointers on this machine until a hook is "
                "registered. On the plugin channel that is `claude plugin "
                "install`; on nix it is the home-manager module.",
                actor=USER,
            )
        ]
    cfg = machine.config()
    nonce = cfg.canary_nonce if cfg is not None else ""
    if not nonce:
        stdout, stderr, code, ms = _probe_hook(
            machine, command, "memkit doctor probe prompt"
        )
        if code is None:
            return [
                Check(
                    "hook-path",
                    FAIL,
                    f"{how} did not finish inside {HOOK_PROBE_TIMEOUT}s "
                    f"({stderr[:120]})",
                    "On every prompt this is a turn delayed to the "
                    "registration's timeout and then abandoned.",
                )
            ]
        return [
            Check(
                "hook-path",
                INFO,
                f"{how} ran in {ms}ms and exited {code}. Without a canary "
                "nonce there is no query whose answer would prove delivery, "
                "so this says the path runs and not that it serves",
            )
        ]

    stdout, stderr, code, ms = _probe_hook(machine, command, canary_query(nonce))
    if code is None:
        return [
            Check(
                "hook-path",
                FAIL,
                f"{how} did not finish inside {HOOK_PROBE_TIMEOUT}s. {stderr[:120]}",
                "The registration gives up at its own timeout and your prompt "
                "goes through without pointers. A first run on a large store "
                "builds the index; if this repeats, the store is too large or "
                "the index cannot be written.",
                actor=USER,
            )
        ]
    if CANARY_NAME in stdout and FRAME_TAG in stdout:
        return [
            Check(
                "hook-path",
                PASS,
                f"{how} emitted a framed pointer to {CANARY_NAME} in {ms}ms",
            )
        ]
    return [
        Check(
            "hook-path",
            FAIL,
            f"{how} exited {code} in {ms}ms and emitted no pointer to "
            f"{CANARY_NAME}. stderr: {stderr[:200] or '(empty)'}",
            "The store answers and the installed path does not, so the break "
            "is between them: the wrapper's config resolution, the "
            "interpreter it picked, or the registration itself. The "
            "config-route and interpreter checks in this report say which.",
        )
    ]


# Every outcome the hook can write, and the one line that says what it means.
# Mechanizes the README's own triage table so an adopter reading a histogram
# does not have to go and look each name up — and a test pins the two together,
# because the vocabulary grows without a version bump and a name that arrived
# here without arriving there is a record nobody can read.
OUTCOME_REASONS = {
    "injected": "pointers were written into the prompt",
    "gate:envelope": "the prompt began with an editor or tool envelope",
    "gate:empty": "the prompt was empty after stripping",
    "gate:slash": "the prompt began with `/` — a slash command, not a question",
    "gate:short": "the prompt was under three words",
    "gate:long": "the prompt was over 4000 characters — a paste, not a question",
    "gate:stopwords": "the prompt was all common words",
    "gate:nodirs": "nothing to search: no config, or no store on disk and in "
    "scope here",
    "nomatch": "the stores were searched and nothing came back",
    "deduped": "every match had already been offered this session",
    "floored": "matches existed and none cleared the relevance bar",
    "gate:budget:weak": "the session's pointer budget is spent and nothing "
    "beat the weakest",
    "gate:budget": "the session budget is spent, on a ledger this build cannot "
    "reason about",
    "error": "retrieval raised; the turn was unaffected",
    "killed": "the hook ran out of time and gave up rather than delaying the "
    "prompt",
    "output-lost": "pointers were built and the write did not land",
    "dup-registration": "two installs on one machine registered the same hook",
}

# How much of the log a histogram is built over. The file grows one line per
# invocation and is deliberately unswept, so a check that read all of it would
# be the slowest thing in the report on the machine that has been running it
# longest.
GATE_WINDOW = 400


def _soak_tail(state_dir: str, limit: int) -> list:
    """The last `limit` parseable records. A torn final line is a crash mid-
    append, not a corrupt log, and is skipped."""
    path = os.path.join(state_dir, SOAK_LOG_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        with contextlib.suppress(ValueError):
            record = json.loads(line)
            if isinstance(record, dict):
                out.append(record)
    return out


def _prompt_records(records: list) -> list:
    """The per-prompt population, per the log's own published rule.

    `concludes: false` marks a record that is about the machine rather than
    about a prompt, and doctor's own probe carries `doctor: true`. Both are
    excluded here, or a report about how often prompts inject would be counting
    the runs doctor itself made.
    """
    return [
        r
        for r in records
        if r.get("concludes") is not False and not r.get("doctor")
    ]


def _when(record: dict) -> str:
    """A record's timestamp as a person reads it. `ts` is seconds since the
    epoch and a report that printed the integer would be one more thing to go
    and convert."""
    stamp = record.get("ts")
    if not isinstance(stamp, int):
        return "an unrecorded time"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


@_produces("hook-ever-fired")
def _hook_ever_fired(machine: Machine) -> list[Check]:
    """Has this hook ever run, and has it ever injected HERE?

    The adopter's first question, and until now the only witness was a file
    documented for downstream analyzers — walk friction 4, where the adopter
    found their first successful injection by guessing at `log.jsonl`.

    Three answers rather than two, because they want different next moves: no
    log at all is an install that has never been configured, a log with no
    injection here is a store the gate or the corpus is keeping out of this
    project, and an injection here is the thing working.
    """
    records = _prompt_records(_soak_tail(machine.state_dir, GATE_WINDOW))
    if not records:
        return [
            Check(
                "hook-ever-fired",
                UNKNOWN,
                f"no {SOAK_LOG_NAME} in {_display_path(machine.state_dir)}: "
                "this hook has never run. An install that was never configured "
                "writes none, deliberately",
            )
        ]
    here = _cwd_digest()
    mine = [r for r in records if r.get("cwd") == here]
    injected_here = [r for r in mine if r.get("outcome") == "injected"]
    last = records[-1]
    when = _when(last)
    if injected_here:
        latest = _when(injected_here[-1])
        return [
            Check(
                "hook-ever-fired",
                PASS,
                f"last injected in this directory at {latest}; {len(mine)} of "
                f"the last {len(records)} records are from here",
            )
        ]
    if mine:
        return [
            Check(
                "hook-ever-fired",
                INFO,
                f"{len(mine)} of the last {len(records)} records are from this "
                f"directory and none injected; the last was "
                f"{mine[-1].get('outcome')!r}. See gate-outcomes",
            )
        ]
    return [
        Check(
            "hook-ever-fired",
            INFO,
            f"the hook has run ({len(records)} records, last {last.get('outcome')!r} "
            f"at {when}) and never in this directory",
        )
    ]


@_produces("gate-outcomes")
def _gate_outcomes(machine: Machine) -> list[Check]:
    """The mechanized *Why nothing appeared* table: what actually happened, in
    counts, each rendered with the table's own reason.

    Always INFO. Every one of these is a state the hook is designed to reach,
    and a histogram is evidence rather than a verdict — "nothing passed the
    floor", "there was nothing to search" and "retrieval raised" are three
    different answers and none of them is broken by itself.
    """
    records = _prompt_records(_soak_tail(machine.state_dir, GATE_WINDOW))
    if not records:
        return [
            Check("gate-outcomes", INFO, "no records yet, so nothing to count")
        ]
    counts: dict = {}
    for record in records:
        outcome = record.get("outcome")
        if isinstance(outcome, str):
            counts[outcome] = counts.get(outcome, 0) + 1
    parts = []
    for outcome, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        # An outcome this build does not know is REPORTED rather than dropped:
        # the vocabulary grows without a version bump, and a reader that
        # silently discarded a name it did not recognise would compute a rate
        # over a denominator nobody checked.
        reason = OUTCOME_REASONS.get(outcome, "an outcome this build does not know")
        parts.append(f"{outcome} {count} ({reason})")
    took = sorted(r["ms"] for r in records if isinstance(r.get("ms"), int))
    median = f"; median {took[len(took) // 2]}ms" if took else ""
    return [
        Check(
            "gate-outcomes",
            INFO,
            f"last {len(records)} prompts: " + ", ".join(parts) + median,
        )
    ]


# --- coexistence -------------------------------------------------------------


# The harness build memkit's claims about the harness were MEASURED against:
# the option-name mangling, the trailing slash on the plugin root, the
# exit-2-blocks-the-turn behaviour the zero-argument pin rests on. Pinned to
# the workflows' own `CLAUDE_CODE_VERSION` by a test, because a stamp that
# drifted from the build CI measures on is a stamp that reports agreement
# nobody established.
MEASURED_HARNESS = "2.1.238"

# Where the harness keeps the built-in memory feature's per-project state.
# Measured on 2.1.241: `<config dir>/projects/<sanitized cwd>/memory/`, with
# the cwd sanitized by replacing `/` and `.` with `-`. The lock beside it is
# how "armed" and "actually running" are told apart.
CONSOLIDATE_LOCK = ".consolidate-lock"
# An hour, which is the harness's own consolidation interval. A lock older than
# that is a run that finished, not one in flight.
CONSOLIDATE_RECENT = 3600


def _memkit_registrations(machine: Machine) -> list:
    """Every way this machine has asked for memkit's hook to run.

    The runtime half of this — the `dup-registration` fingerprint — is loud and
    cannot name the entry, and is blind to the likeliest duplicate of all: a
    plugin entry and a settings entry naming ONE config of one release, where
    the version stamp is a hash of identical bytes. Counting registrations is
    the half that can name which one to remove.
    """
    found = []
    for scope in machine.settings:
        events = scope.data.get("hooks")
        if not isinstance(events, dict):
            continue
        for entry in events.get("UserPromptSubmit") or []:
            for spec in (entry or {}).get("hooks") or []:
                command = (spec or {}).get("command")
                if isinstance(command, str) and (
                    "memkit" in command or "memory_prompt_recall" in command
                ):
                    found.append(
                        f'{scope.scope} settings ({_display_path(scope.path)}): '
                        f'"{command}"'
                    )
    if _plugin_enabled(machine) is True:
        found.append(f"the {PLUGIN_KEY} plugin registration")
    return found


def _plugin_enabled(machine: Machine):
    """True, False, or None for "this machine has no opinion".

    Three states, because the middle one is the trap: a disabled plugin still
    reports `Hooks (1)` from `plugin details`, and only `plugin list` disagrees.
    Both walkthroughs met that and read it as a working install.
    """
    for scope in machine.settings:
        enabled = scope.data.get("enabledPlugins")
        if isinstance(enabled, dict) and PLUGIN_KEY in enabled:
            return bool(enabled[PLUGIN_KEY])
    return None


@_produces("registrations-count")
def _registrations_count(machine: Machine) -> list[Check]:
    """Exactly one, or say which entries to choose between.

    Two registrations both serving one prompt is a silent lost update from
    inside: each process injects, each writes the session ledger, and the later
    write wins. What the user sees is pointers that come and go for no reason.
    """
    found = _memkit_registrations(machine)
    if len(found) == 1:
        return [Check("registrations-count", PASS, f"one registration: {found[0]}")]
    if not found:
        if machine.plugin:
            return [
                Check(
                    "registrations-count",
                    FAIL,
                    "this process was started by the plugin's own wrapper and "
                    "no settings scope records the plugin as enabled, so "
                    "nothing here says a hook is registered",
                    "Check `claude plugin list`. A plugin that is installed "
                    "and disabled still reports `Hooks (1)` from `plugin "
                    "details`.",
                    actor=USER,
                )
            ]
        return [
            Check(
                "registrations-count",
                INFO,
                "no memkit hook registration in any settings scope and no "
                "enabled plugin, so nothing runs on a prompt here",
            )
        ]
    return [
        Check(
            "registrations-count",
            FAIL,
            f"{len(found)} registrations serve every prompt: " + "; ".join(found),
            "Remove all but one. Both run, both write this session's ledger, "
            "and the later write wins — which shows up as pointers that come "
            "and go for no reason rather than as an error.",
            actor=USER,
        )
    ]


@_produces("plugin-enabled")
def _plugin_enabled_check(machine: Machine) -> list[Check]:
    """`claude plugin list` should be the first check, and this is it.

    The trap it names: `plugin details` reports a registered hook on a plugin
    that is switched off. Only `plugin list` disagrees, and nothing sends an
    adopter there.
    """
    enabled = _plugin_enabled(machine)
    if enabled is None:
        return [
            Check(
                "plugin-enabled",
                INFO,
                f"no settings scope mentions {PLUGIN_KEY}, so this machine has "
                "no plugin install to enable or disable",
            )
        ]
    if enabled:
        return [Check("plugin-enabled", PASS, f"{PLUGIN_KEY} is enabled")]
    return [
        Check(
            "plugin-enabled",
            FAIL,
            f"{PLUGIN_KEY} is installed and DISABLED. `claude plugin details` "
            "still reports `Hooks (1)` in this state, which reads as a working "
            "install",
            f"claude plugin enable {PLUGIN_KEY}",
            actor=USER,
        )
    ]


@_produces("plugin-diagnostics")
def _plugin_diagnostics(machine: Machine) -> list[Check]:
    """What the trust gate and the duplicate detector recorded.

    Otherwise the instrumentation those two write has no reader on an adopter's
    machine — the marker is a file in a plugin data directory nobody is told
    about, and the `dup-registration` records sit in a log documented for
    downstream analyzers.

    `actor: user` on every remedy here, because the marker records REFUSALS and
    a refusal is a setup fact rather than something an agent may act on.
    """
    marker = (
        os.path.join(machine.plugin_data, MARKER_NAME)
        if machine.plugin_data
        else ""
    )
    records = []
    if marker and os.path.isfile(marker):
        with contextlib.suppress(OSError, ValueError):
            with open(marker, encoding="utf-8") as f:
                blob = json.load(f)
            if isinstance(blob, dict):
                loaded = blob.get("records")
                records = loaded if isinstance(loaded, list) else []
    dups = [
        r
        for r in _soak_tail(machine.state_dir, GATE_WINDOW)
        if r.get("outcome") == "dup-registration"
    ]
    if not records and not dups:
        return [
            Check(
                "plugin-diagnostics",
                PASS,
                "no refusals recorded and no duplicate registration seen at "
                "runtime",
            )
        ]
    outcomes: dict = {}
    for record in records:
        name = record.get("outcome")
        if isinstance(name, str):
            outcomes[name] = outcomes.get(name, 0) + 1
    where = len({r.get("cwd") for r in records if r.get("cwd")})
    parts = [f"{name} x{count}" for name, count in sorted(outcomes.items())]
    if dups:
        parts.append(f"dup-registration x{len(dups)} in the soak log")
    return [
        Check(
            "plugin-diagnostics",
            INFO,
            f"{len(records)} refusal(s) across {where} directory/ies: "
            + ", ".join(parts),
            "`trust:unconfigured` means the hook refused before reading a "
            "prompt because no config resolved — see config-route. "
            "`dup-registration` means two installs served one prompt — see "
            "registrations-count.",
            actor=USER,
        )
    ]


@_produces("subagent-delivery")
def _subagent_delivery(machine: Machine) -> list[Check]:
    """Whether memories reach a subagent, and whether that has ever happened.

    Registered-and-never-fired and fired-and-refused are different states with
    different next moves, and both look like silence. UNKNOWN is the answer
    while the subagent path is not in the build at all — a state the closed
    status set already has, and one that does not block green.
    """
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    registered = False
    if root:
        with contextlib.suppress(OSError, ValueError):
            with open(os.path.join(root, "hooks", "hooks.json"), encoding="utf-8") as f:
                blob = json.load(f)
            for entry in (blob.get("hooks") or {}).get("PreToolUse") or []:
                matcher = (entry or {}).get("matcher")
                if matcher and "Agent" in str(matcher):
                    registered = True
    if not registered:
        return [
            Check(
                "subagent-delivery",
                UNKNOWN,
                "no PreToolUse-on-Agent entry in this payload's hooks.json, so "
                "the subagent path is not in this build. Subagents get no "
                "pointers and nothing is wrong",
            )
        ]
    task = [
        r
        for r in _soak_tail(machine.state_dir, GATE_WINDOW)
        if isinstance(r.get("outcome"), str)
        and r["outcome"].startswith(TASK_OUTCOME_PREFIX)
    ]
    if not task:
        return [
            Check(
                "subagent-delivery",
                INFO,
                "registered, and it has never fired here",
            )
        ]
    last = task[-1]["outcome"]
    if last == TASK_OUTCOME_PREFIX + "injected":
        return [
            Check(
                "subagent-delivery",
                PASS,
                f"last subagent brief was served at {_when(task[-1])}",
            )
        ]
    return [
        Check(
            "subagent-delivery",
            INFO,
            f"registered and firing; the last outcome was {last!r} rather than "
            "a delivery",
        )
    ]


@_produces("harness-stamp")
def _harness_stamp(machine: Machine) -> list[Check]:
    """The harness this build's claims were measured against, versus the one
    running.

    NEVER BLOCKS GREEN. Harness releases outpace stamps, so a mismatch is the
    normal case for every adopter who is not on the pinned build — and a
    criterion that counted it would make all-green unreachable for almost
    everybody, which is how a report stops being read.
    """
    binary = shutil.which("claude")
    if binary is None:
        return [
            Check(
                "harness-stamp",
                UNKNOWN,
                f"no `claude` on PATH to ask, so the running harness version is "
                f"not knowable from here. memkit's claims were measured "
                f"against {MEASURED_HARNESS}",
            )
        ]
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [
            Check(
                "harness-stamp",
                UNKNOWN,
                f"`claude --version` could not be run ({type(exc).__name__}); "
                f"memkit's claims were measured against {MEASURED_HARNESS}",
            )
        ]
    running = (out.stdout.split() or [""])[0]
    if running == MEASURED_HARNESS:
        return [
            Check("harness-stamp", PASS, f"harness {running}, the build memkit "
                  "measured its harness claims against")
        ]
    return [
        Check(
            "harness-stamp",
            UNVERIFIED,
            f"harness {running or '?'}; memkit measured its harness claims "
            f"against {MEASURED_HARNESS}. Nothing is known to have changed, "
            "and nothing here re-measured it",
        )
    ]


def _sanitized_cwd() -> str:
    """The harness's own per-project directory name for this cwd.

    Measured on 2.1.241: `/` and `.` both become `-`, so
    `/Users/x/.config/nix` is `-Users-x--config-nix`.
    """
    return re.sub(r"[/.]", "-", os.getcwd())


@_produces("auto-memory")
def _auto_memory(machine: Machine) -> list[Check]:
    """The harness's own memory feature, running beside memkit's.

    The one differentiator the field survey found unclaimed: none of the six
    competitors handles built-in auto-memory coexistence at all. Two stores
    writing memories about the same work, in two formats, with two retrieval
    paths, is a state an adopter should choose rather than discover.
    """
    setting = None
    where = ""
    for scope in machine.settings:
        for key in ("autoDreamEnabled", "autoMemoryEnabled"):
            if key in scope.data:
                setting = (key, scope.data[key])
                where = scope.scope
                break
        if setting:
            break
    config_dir = os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")
    project = os.path.join(config_dir, "projects", _sanitized_cwd())
    recent = ""
    for candidate in (
        os.path.join(project, CONSOLIDATE_LOCK),
        os.path.join(project, "memory", CONSOLIDATE_LOCK),
    ):
        with contextlib.suppress(OSError):
            age = int(time.time() - os.stat(candidate).st_mtime)
            if age < CONSOLIDATE_RECENT:
                recent = f"; a consolidation ran {age}s ago"
            else:
                recent = f"; last consolidation {age // 3600}h ago"
            break
    if setting is None:
        return [
            Check(
                "auto-memory",
                PASS,
                "no auto-memory setting in any scope" + (recent or ""),
            )
        ]
    key, value = setting
    if not value:
        return [
            Check("auto-memory", PASS, f"{key} is off in {where} settings" + recent)
        ]
    return [
        Check(
            "auto-memory",
            INFO,
            f"{key} is ON in {where} settings{recent}. The harness writes and "
            "consolidates its own memories under "
            f"{_display_path(project)}/memory/, beside memkit's store",
            "Two memory systems on one project is a choice rather than a "
            'fault. To run memkit alone, set "autoDreamEnabled": false in '
            f"{_display_path(config_dir)}/settings.json.",
            actor=USER,
        )
    ]


# --- the machine, and what is left behind ------------------------------------


# The two files the nix channel links into the harness's hook directory, and
# the layout the ROLLOUT runbook's per-host verify asserts by eye. A machine
# reader for a recipe that was only ever run by hand.
NIX_HOOK_FILES = ("memory-prompt-recall.py", "common-words.txt")
NIX_STORE = "/nix/store/"

# The checker's floor, and the same two numbers `bin/lib/common.sh` holds. It
# lives in two files by necessity — one of them is POSIX sh and cannot import
# the other — and a test scrapes them against each other.
CHECKER_FLOOR = (3, 12)
# The hook's floor, which is NOT the checker's. A stock macOS python is 3.9.6
# and every current Linux distribution clears it.
HOOK_FLOOR = (3, 9)

ROUTE_ENV = "MEMKIT_CHECKER_ROUTE"
ROUTE_CMD_ENV = "MEMKIT_CHECKER_CMD"


def build_facts() -> tuple:
    """(package version, hook version, payload sha) — the three answers to
    "which build am I on", each None when this install cannot derive it.

    A precondition for reading any other line of this report, and until now no
    command anywhere answered it: a critic filed the absence of `--version` as
    a defect against all four binaries.
    """
    package = None
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        package = version("memkit")
    hook_version = None
    with contextlib.suppress(Exception):
        hook_version = _version()
    payload = None
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if root and os.path.isdir(os.path.join(root, ".git")):
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            out = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0:
                payload = out.stdout.strip()[:12]
    return package, hook_version, payload


def version_line() -> str:
    """What `memkit --version` prints. One line, three facts, and the ones
    this install cannot derive are named as unknown rather than omitted — a
    missing field reads as a field that does not exist."""
    package, hook_version, payload = build_facts()
    return (
        f"memkit {package or '(no installed distribution)'} "
        f"hook:{hook_version or '?'} payload:{payload or '(not a clone)'}"
    )


@_produces("build")
def _build(machine: Machine) -> list[Check]:
    package, hook_version, payload = build_facts()
    if not any((package, hook_version, payload)):
        return [
            Check(
                "build",
                UNKNOWN,
                "no version is derivable here: no installed distribution, the "
                "hook module could not be hashed, and the payload is not a "
                "clone",
            )
        ]
    return [Check("build", INFO, version_line())]


def _checker_route(machine: Machine) -> tuple:
    """(route, command) for checker-backed work.

    Read from the environment when the wrapper resolved it, which is the whole
    point of the wrapper exporting it: the route is decided once per invocation
    so that two subcommands cannot pick differently. Probed only where there is
    no wrapper — the pip and uvx channels — and cached on the machine, because
    each probe is a fork.
    """
    route = os.environ.get(ROUTE_ENV)
    if route:
        # Split on whitespace and nothing cleverer: it is a space-joined
        # string, and the wrapper's own contract says so.
        return route, (os.environ.get(ROUTE_CMD_ENV) or "").split()
    if machine._route is None:
        machine._route = _probe_checker_route()
    return machine._route


def _probe_checker_route() -> tuple:
    if sys.version_info[:2] >= CHECKER_FLOOR:
        return "python", [sys.executable, "-m", "memkit.memory_integrity"]
    for name in ("python3.14", "python3.13", "python3.12", "python3"):
        found = shutil.which(name)
        if not found:
            continue
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            out = subprocess.run(
                [found, "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and "(3, 1" in out.stdout:
                pair = out.stdout.strip().strip("()").split(",")
                if (int(pair[0]), int(pair[1])) >= CHECKER_FLOOR:
                    return "python", [found, "-m", "memkit.memory_integrity"]
    if shutil.which("uvx"):
        return "uvx", ["uvx", "memory-integrity"]
    return "none", []


def _recorded_interpreter(machine: Machine) -> str:
    """The `interpreter` the config records, read as a field rather than by the
    wrapper's line scrape. Doctor has a JSON parser; the wrapper does not."""
    if not machine.resolved_config:
        return ""
    with contextlib.suppress(OSError, ValueError):
        with open(machine.resolved_config, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            value = raw.get("interpreter")
            if isinstance(value, str):
                return value
    return ""


@_produces("interpreter")
def _interpreter(machine: Machine) -> list[Check]:
    """Which python runs the hook, and which route runs the checker.

    The stock-mac case is the one this exists for: `python3` there is 3.9.6 and
    the checker's floor is 3.12, so an install that works perfectly for
    retrieval cannot regenerate a ledger without `uvx`. That is INFORMATION,
    never a failure — and reporting WHICH route resolved is what makes the
    claim scoreable instead of a shrug.
    """
    running = ".".join(str(n) for n in sys.version_info[:3])
    route, command = _checker_route(machine)
    recorded = _recorded_interpreter(machine)
    honoured = ""
    if recorded:
        expanded = os.path.expanduser(recorded)
        if not (os.path.isfile(expanded) and os.access(expanded, os.X_OK)):
            honoured = (
                f'. The config records "interpreter": "{recorded}", which is '
                "not an executable file, so the wrapper falls back to the "
                "python3 on PATH"
            )
        elif os.path.realpath(expanded) != os.path.realpath(sys.executable):
            honoured = (
                f'. The config records "{recorded}" and this process is '
                f"{_display_path(sys.executable)}"
            )
    if route == "none":
        return [
            Check(
                "interpreter",
                FAIL,
                f"hook interpreter {running}; NO checker route: no python "
                f"meets {CHECKER_FLOOR[0]}.{CHECKER_FLOOR[1]} and there is no "
                f"uvx{honoured}",
                "Install python 3.12 or newer, or install uv. Until then any "
                "command that regenerates a ledger refuses by name and writes "
                "nothing — a seeded memory with no ledger row is a broken "
                "store, so half-completing is worse than not starting.",
                actor=USER,
                terminal=True,
            )
        ]
    if sys.version_info[:2] < HOOK_FLOOR:
        return [
            Check(
                "interpreter",
                FAIL,
                f"hook interpreter {running}, below the {HOOK_FLOOR[0]}."
                f"{HOOK_FLOOR[1]} floor the hook imports under",
                "Record an absolute path to a python 3.9 or newer as "
                '"interpreter" in the memkit config.',
                actor=USER,
                terminal=True,
            )
        ]
    where = " ".join(command) if command else "?"
    if route == "uvx":
        return [
            Check(
                "interpreter",
                INFO,
                f"hook interpreter {running}; checker route uvx ({where}), "
                f"because no local python meets {CHECKER_FLOOR[0]}."
                f"{CHECKER_FLOOR[1]}. Retrieval is unaffected{honoured}",
            )
        ]
    if honoured:
        return [
            Check(
                "interpreter",
                INFO,
                f"hook interpreter {running}; checker route {route} "
                f"({where}){honoured}",
            )
        ]
    return [
        Check(
            "interpreter",
            PASS,
            f"hook interpreter {running}; checker route {route} ({where})",
        )
    ]


def _dir_size(path: str) -> tuple:
    """(files, bytes) under one directory, one level deep.

    The state directory is flat by construction and can hold five figures of
    files, so this is the cheapest complete answer rather than a walk.
    """
    files = 0
    total = 0
    with contextlib.suppress(OSError), os.scandir(path) as entries:
        for entry in entries:
            files += 1
            with contextlib.suppress(OSError):
                total += entry.stat(follow_symlinks=False).st_size
    return files, total


def _mib(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MiB"


@_produces("state-dir")
def _state_dir_check(machine: Machine) -> list[Check]:
    """What is in the shared cache, how big it is, and what is deliberately
    never collected.

    A number nobody had until it was asked for: on the author's own machine
    this directory held 14,349 files and 289 MB, of which one file was the soak
    log the analyzers treat as their corpus.

    This check also discloses doctor's own footprint. `hook-path` runs the
    installed hook once against this directory, which appends a soak record and
    may run the sweep, and a read-only claim that quietly did that would be the
    kind of thing this whole report exists to stop.
    """
    path = machine.state_dir
    if not os.path.isdir(path):
        parent = os.path.dirname(path)
        degraded = (
            ""
            if os.access(parent, os.W_OK)
            else f". {_display_path(parent)} is not writable, so the hook would "
            "fall back to a private temporary directory and every session "
            "would start cold"
        )
        return [
            Check(
                "state-dir",
                INFO,
                f"{_display_path(path)} does not exist: nothing has written "
                f"derived state here. An install nobody configured writes "
                f"none, deliberately{degraded}",
            )
        ]
    files, total = _dir_size(path)
    log = os.path.join(path, SOAK_LOG_NAME)
    log_size = 0
    with contextlib.suppress(OSError):
        log_size = os.stat(log).st_size
    swept = "never swept"
    with contextlib.suppress(OSError):
        stamp = os.stat(os.path.join(path, SWEEP_STAMP_NAME)).st_mtime
        swept = "last swept " + time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(stamp)
        )
    footprint = (
        ". This doctor run appended one soak record here and may have run the "
        "sweep"
        if machine.hook_probed
        else ""
    )
    return [
        Check(
            "state-dir",
            INFO,
            f"{_display_path(path)}: {files} file(s), {_mib(total)}; "
            f"{SOAK_LOG_NAME} {_mib(log_size)}; {swept}{footprint}",
            f"{SOAK_LOG_NAME} is deliberately never collected — the soak "
            "analyzers treat it as their corpus — so it grows one line per "
            "invocation. Everything else here is a cache that rebuilds itself.",
            actor=USER,
        )
    ]


@_produces("hooks-layout")
def _hooks_layout(machine: Machine) -> list[Check]:
    """The nix channel's layout, as the rollout runbook asserts it by eye.

    `n/a` off that channel rather than absent: a check that vanished would look
    like one that had not run.
    """
    module = getattr(sys.modules[__name__], "__file__", "") or ""
    if not module.startswith(NIX_STORE):
        return [
            Check(
                "hooks-layout",
                INFO,
                "n/a: this is not the nix channel, which is the only one with "
                "a layout to assert",
            )
        ]
    config_dir = os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")
    hooks = os.path.join(config_dir, "hooks")
    wrong = []
    for name in NIX_HOOK_FILES:
        entry = os.path.join(hooks, name)
        if not os.path.islink(entry):
            wrong.append(f"{_display_path(entry)} is not a symlink")
        elif not os.path.realpath(entry).startswith(NIX_STORE):
            wrong.append(
                f"{_display_path(entry)} points outside {NIX_STORE} "
                f"({os.path.realpath(entry)})"
            )
    if wrong:
        return [
            Check(
                "hooks-layout",
                FAIL,
                "; ".join(wrong),
                "A tracked hook file that is a regular file rather than a "
                "store symlink is the conversion defect the rollout runbook "
                "names. Look before cleaning: a `.backup` beside it is the "
                "only copy of what it used to hold.",
                actor=USER,
            )
        ]
    return [
        Check(
            "hooks-layout",
            PASS,
            f"{', '.join(NIX_HOOK_FILES)} under {_display_path(hooks)} are "
            f"symlinks into {NIX_STORE}",
        )
    ]


@_produces("uninstall-story")
def _uninstall_story(machine: Machine) -> list[Check]:
    """What leaving takes with it, and what it does not.

    The store sits outside every plugin-managed path BY DESIGN, so no uninstall
    sweep reaches it — which is right, and is exactly the thing an adopter
    removing memkit needs told rather than left to discover. Always INFO: none
    of this is a fault.
    """
    cfg = machine.config()
    canaries = []
    if cfg is not None:
        for store in cfg.stores:
            with contextlib.suppress(ConfigError):
                live = cfg.store_dir(store, "live")
                canaries.append(
                    _display_path(os.path.join(_search_root(live), CANARY_NAME))
                )
    survives = [f"your stores{' (' + ', '.join(canaries) + ')' if canaries else ''}"]
    if machine.resolved_config:
        survives.append(_display_path(machine.resolved_config))
    survives.append(f"{_display_path(machine.state_dir)} (index, log, journal)")
    return [
        Check(
            "uninstall-story",
            INFO,
            "`claude plugin uninstall memkit@memkit` removes the payload and "
            "the plugin data directory; `--keep-data` keeps the second. "
            "Neither touches: " + "; ".join(survives),
            "The canary memories above are memkit's own and are safe to "
            "delete by hand; everything else in your stores is yours. Nothing "
            "removes them for you, because the store is deliberately outside "
            "every plugin-managed path.",
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
