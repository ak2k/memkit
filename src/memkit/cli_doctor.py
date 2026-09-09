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

READ-ONLY MEANS: no store write, no config write, no settings write. Nothing
doctor does can change what an adopter would lose.

It is not "touches nothing", and the difference is disclosed rather than
finessed. Three things it does write, all of them derived state that rebuilds
itself:

- `hook-path` executes the installed wrapper, because a fixed-query retrieval
  proves the store and not the path that serves pointers. That run appends one
  soak record and may trigger the hourly sweep.
- `canary-retrieval` searches each store, which syncs — and on a damaged index
  rebuilds — the FTS cache for that corpus. Deliberately the warm one the
  adopter uses: a scratch cache would report green over an index nobody reads.
- both of the above resolve the state directory, which creates it when it is
  absent.

The `state-dir` check says so in the report, and `docs/ADMISSION.md` says it
where somebody decides whether to install.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable

from memkit import harness_memory
from memkit._exec import (
    CHILD_ENV_KEEP,
    CheckerRoute,
    GitRoute,
    Untrusted,
    _execute,
    _under_cwd,
    checker_argv,
    require_executable,
    resolve,
    run_git,
)
from memkit.memory_prompt_recall import (
    BUILD_BUSY,
    BUILD_OK,
    BUILD_PARTIAL,
    BUILD_REBUILT,
    BUILD_TRUNCATED,
    BUILD_UNREADABLE,
    CONFIG_ENV,
    CONFIG_ROUTES,
    DOCTOR_ENV,
    ERRLOG_NAME,
    EXCLUDE_BASENAMES,
    EXCLUDE_DIRS,
    FRAME_NONCE_BYTES,
    FRAME_TAG,
    GENERATED_CONFIG_NAME,
    HARNESS_TIMEOUT,
    INDEX_FILE_MAX_BYTES,
    MARKER_NAME,
    PLUGIN_CONFIG_ROUTES,
    PLUGIN_DATA_ENV,
    PLUGIN_ENV,
    SCHEMA,
    SOAK_LOG_NAME,
    SWEEP_STAMP_NAME,
    TASK_OUTCOME_PREFIX,
    TASK_POPULATION,
    ConfigError,
    _corpus_files,
    _cwd_digest,
    _display_cap,
    _display_path,
    _excluded,
    _fts_db,
    _search_root,
    _session_state_path,
    _state_dir_candidate,
    _store_live_dir,
    _version,
    claim_holds,
    expand_home,
    journal_config_claims,
    load_config,
    path_refusal,
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

# And what ONE PATH inside a detail may cost, because the bound above is only
# half the rule. A path is the part of a sentence whose length somebody else
# decides — a session's own cwd, a value out of a cloned settings file — and a
# single long one fills the whole budget, so what gets cut is every other
# sentence in the row. That failure is worse than a truncated path: the row
# reads as a confident verdict with the verdict missing.
#
# Generous rather than tight, and measured against the longest HONEST path this
# report prints: a derived default is a config directory plus a project key,
# and a key is a whole absolute path with its separators replaced. Two nested
# absolute paths in one string is 250 characters before anything is wrong. This
# is here to stop one value eating a row, not to fit a column.
PATH_SHOWN = 300


# The frame's delimiters as a LITERAL, neutralised in doctor's own output.
#
# MERGE SEAM, and the reason this lives here rather than in the hook's
# `strip_unsafe`. Track A defanged the frame inside the shared sanitizer with
# a grapheme-skeleton match, so that `</memkit́-pointers>` — an accent on
# the `t`, which a reader resolves as the tag anyway — was caught too. Track B
# deleted that rule and the measurement is the argument: over five rounds each
# widening or narrowing of a rule that READS text produced the sibling of the
# defect it had just closed, because the spans an honest store writes and the
# spans a forger writes overlap on every feature the text carries. What
# replaced it is structural — a per-run nonce the writer of a store file
# cannot see, and a line-position invariant no retrieved text can break.
#
# That settles the POINTER BLOCK, and it is not resurrected here. What this is
# instead is a fixed-string substitution on a DIFFERENT surface: doctor's
# report is relayed verbatim into a model's context and sits inside no frame
# of memkit's own, so there is no delimiter for it to close and nothing here
# has to judge whether a span resembles one. It replaces two literals and
# reads no text, which is why it does not inherit Track B's finding.
_FRAME_LITERAL = re.compile(r"</?" + re.escape(FRAME_TAG))


# What may follow a home spelling without the match being a different name.
# TWO RULES AND NOT ONE: a key's own separator is the character a path uses to
# start a component, so `-home-u-git-app` is a directory under `-home-u` while
# `/home/u-git-app` is not one under `/home/u`. A single rule either leaves the
# key spelling of home in the report or renames a sibling to a directory that
# is not there.
_PATH_TAIL = r"(?![\w.-])"
_KEY_TAIL = r"(?!\w)"


def _homes() -> tuple:
    """Every spelling of the home directory a string could carry, compiled.

    FOUR ROUTES, ONE REDACTION. Home reaches a row as the path the environment
    spells it, as the path that resolves to — the two differ for as long as
    `$HOME` is a symlink, and the harness derives its project keys from the
    resolved one — and as either of those with every separator replaced, which
    is what a project key is. Each route was closed where it was found, for one
    branch of one check; applied here, where every detail and every remedy
    passes, the rule holds whichever branch of whichever check produced the row.

    LONGEST FIRST, so a spelling nested inside another does not leave the tail
    of the longer one standing on its own.
    """
    paths: list = []
    for path in (os.path.expanduser("~"), os.path.realpath(os.path.expanduser("~"))):
        if path and path != os.sep and path not in paths:
            paths.append(path)
    keys: list = []
    for key in (harness_memory.key_spelling(path) for path in paths):
        if key and key not in keys:
            keys.append(key)
    spellings = [(path, _PATH_TAIL) for path in paths]
    spellings += [(key, _KEY_TAIL) for key in keys]
    return tuple(
        re.compile(re.escape(spelling) + tail)
        for spelling, tail in sorted(spellings, key=lambda pair: -len(pair[0]))
    )


def _bound(text: str) -> str:
    """One display string, sanitized, redacted and bounded, in that order.

    Sanitizing after bounding would let a truncation land inside an escape
    sequence and produce a string the sanitizer never saw whole. Redacting
    before the cut for the same shape of reason: a home spelling the cut landed
    inside is one no substitution can reach afterwards.
    """
    text = _FRAME_LITERAL.sub("(" + FRAME_TAG, sanitize(text))
    for spelling in _homes():
        text = spelling.sub("~", text)
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
    "task-outcomes",
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
    "hook-errors",
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
#
# THREE KEYS ARE RESOLVED RATHER THAN REPORTED, in `harness_memory`:
# `autoMemoryDirectory`, `autoMemoryEnabled` and `autoDreamEnabled`. They are
# the exception the paragraph above allows for, and what earns it is that
# their effect is a DIRECTORY and a switch the auto-memory check has to name:
# whether what the harness writes is retrievable is a question about one path,
# so a report listing every value it found would answer a different one.
#
# Their order is `harness_memory.SCOPE_ORDER`, measured on 2.1.258 out of the
# resolver itself, and it is NOT the order this file lists the scopes in: the
# harness reads `.claude/settings.local.json` and then the checked-in
# `.claude/settings.json` ahead of user settings, so a checkout outranks the
# adopter. It is also not what the shipped schema description claims — that
# text says the directory key is ignored in a checked-in file "for security",
# and the resolver consults it on every non-interactive run whatever the
# folder trust state.
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


# WHY A SCOPE DID NOT ANSWER, as three states rather than one message. They are
# three different things to have happened to an adopter's machine and three
# different repairs: a syntax error is theirs to fix, a refused open is
# somebody's to grant and never theirs to edit — the managed scope is an
# administrator's root-owned policy file — and anything else is a question
# about what is at that path at all. Collapsed into one string, the sentence
# written for the commonest of them gets asserted over the other two.
UNPARSED = "unparsed"
FORBIDDEN = "forbidden"
UNREADABLE = "unreadable"


class Settings:
    """One settings file: where it is, what it holds, and why it does not.

    A file that is present and unparseable is its own state, and it is the
    field anti-pattern the prior-art survey names: a harness that meets a parse
    error and silently replaces the file with a stub takes the adopter's
    configuration with it. Doctor never repairs one; it says which file and
    what the parser said.
    """

    __slots__ = ("scope", "path", "data", "error", "failure", "adopter_owned")

    def __init__(self, scope: str, path: str, adopter_owned: bool = True) -> None:
        self.scope = scope
        self.path = path
        # WHO CAN WRITE THIS FILE. `managed` and `user` are the adopter's and
        # their administrator's; `project` and `local` sit in whatever
        # directory the session stands in, which on a cloned repository means
        # its author's. Carried as data because the distinction decides
        # whether a command in the file may be EXECUTED, and a rule that
        # important should not be a scope-name comparison repeated at each
        # reader.
        self.adopter_owned = adopter_owned
        self.data: dict = {}
        self.error = ""
        # WHICH of `UNPARSED`, `FORBIDDEN`, `UNREADABLE`, or "" for a scope
        # that answered — read off the exception rather than written once for
        # the case that motivated the message.
        self.failure = ""
        if not path:
            return
        # THE OPEN IS THE TEST. `os.path.isfile` swallows its own `OSError` and
        # answers False, so a settings file sitting in a directory this process
        # may not traverse read as no file at all — the one unreadable state
        # this class was blind to, and the one that let a row conclude from
        # settings nobody had read. Only the errors that mean "nothing is
        # there" return silently.
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            return
        except PermissionError as exc:
            self.failure = FORBIDDEN
            self.error = str(exc)
            return
        except OSError as exc:
            self.failure = UNREADABLE
            self.error = str(exc)
            return
        except ValueError as exc:
            self.failure = UNPARSED
            self.error = str(exc)
            return
        if not isinstance(blob, dict):
            self.failure = UNPARSED
            self.error = "top level is not an object"
            return
        self.data = blob


def _option_in(scope: Settings) -> str:
    """The literal `memkitConfig` one settings scope records, or ""."""
    configs = scope.data.get("pluginConfigs")
    if not isinstance(configs, dict):
        return ""
    entry = configs.get(PLUGIN_KEY)
    if not isinstance(entry, dict):
        return ""
    options = entry.get("options")
    if not isinstance(options, dict):
        return ""
    value = options.get(OPTION_KEY)
    return value if isinstance(value, str) and value else ""


def _session_cwd() -> str:
    """Where this session stands, or "" once that directory is gone.

    THE DIRECTORY THIS PROCESS STANDS IN CAN BE REMOVED UNDER IT — an agent's
    session workdir cleaned up by something else, a torn-down worktree.
    `os.getcwd()` then raises, and this runs from `Machine.__init__`, so an
    unguarded call turns both commands into a traceback where their published
    tables promise a closed set of exit codes. A scope whose path cannot be
    spelled is a scope with no file in it, which is the same answer as an
    install that has none.

    ONE WALK PER RUN, kept on `Machine` and passed to everything that needs it:
    the settings scopes and the harness's project key are two facts about one
    directory, and a second walk answers differently from the first if that
    directory moved between them.
    """
    try:
        return os.getcwd()
    except OSError:
        return ""


def settings_scopes(cwd: str | None = None) -> list[Settings]:
    """Every scope, most authoritative first."""
    user = os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")
    if cwd is None:
        cwd = _session_cwd()
    # THE TRUSTED SCOPE'S LOCATION IS AN ENVIRONMENT VARIABLE. Whatever can set
    # `$CLAUDE_CONFIG_DIR` — direnv in a checkout, a wrapper script — decides
    # where the `user` scope is read from, so pointed inside the session's own
    # directory it is the project scope under another name. The `project` and
    # `local` entries below are already untrusted by their paths; this is the
    # same rule for the one whose path is somebody's to choose.
    user_owned = not _under_cwd(os.path.join(user, SETTINGS_NAME))
    # ONE FILE IS ONE SCOPE. Run from the directory that holds the config
    # directory — an adopter's own home, which is where a first `memkit doctor`
    # is most often typed — `.claude/` under the cwd IS the user scope, and
    # read a second time as `project` the adopter's own settings file was
    # reported as checked into a repository and travelling with every clone.
    # Both scopes go, not just `project`: what these two entries model is a
    # directory somebody else's checkout decides, and here there is no such
    # directory to model.
    project_dir = (
        ""
        if not cwd or _resolved(os.path.join(cwd, ".claude")) == _resolved(user)
        else os.path.join(cwd, ".claude")
    )
    return [
        Settings("managed", os.path.join(_managed_dir(), MANAGED_SETTINGS_NAME)),
        Settings("user", os.path.join(user, SETTINGS_NAME),
                 adopter_owned=user_owned),
        Settings(
            "project",
            os.path.join(project_dir, SETTINGS_NAME) if project_dir else "",
            adopter_owned=False,
        ),
        Settings(
            "local",
            os.path.join(project_dir, LOCAL_SETTINGS_NAME) if project_dir else "",
            adopter_owned=False,
        ),
    ]


# How much of the PARSER's own message a row quotes. `json`'s messages carry a
# line and a column and are short; the cap is here because the string comes
# from `str(exc)` on an arbitrary file and a row that let it run is a row one
# settings file can empty of everything else.
PARSER_SHOWN = 120

# And how much of the WHOLE note, however many scopes contribute to it. Capping
# the parts was not capping the note: one scope reaches ~490 characters between
# its prose, its path and its parser message, so two of them filled a 600-byte
# detail on their own and `_bound`, which cuts from the end, took the row's own
# verdict with them. A qualifier that erases what it qualifies says less than
# nothing.
#
# HALF THE BUDGET, and that is the whole of the guarantee: the note goes first,
# so bounding it to half of `DETAIL_MAX_BYTES` leaves the other half for the row
# — and every row here already leads with its own verdict clause, for the same
# reason. What a long note now costs is its own tail rather than the answer.
NOTE_SHOWN = DETAIL_MAX_BYTES // 2


def _unparsed_settings(scopes: list) -> str:
    """What a settings file that would not parse costs the rows below it, or "".

    A SCOPE THAT DID NOT PARSE IS NOT A SCOPE THAT DECLARES NOTHING, and every
    reader here has been treating the two as one state: `data` is `{}` either
    way, so a key set in that file reads as unset and whatever the row then
    concludes rests on a file nobody read. `Settings` has recorded the parser's
    message since it was written and nothing has ever printed it — which is
    how a `settings.json` with one stray comma produced a row telling its
    author to set, in that same file, the key it already sets.

    WHAT THE HARNESS DOES WITH THE SAME FILE IS NOT CLAIMED. This says what
    THIS process read, because the anti-pattern the field survey names is a
    harness that meets a parse error and silently replaces the file with a
    stub — and a diagnostic that asserted either behaviour would be guessing
    with the adopter's configuration.

    The verdict before the path, and the parser's text last: `_bound` cuts from
    the end, and of the three facts here the one an adopter's own file decides
    the length of is the one worth losing.

    THE VERB COMES OFF THE EXCEPTION. "Could not be parsed" over a file that
    parses perfectly and was merely refused is an assertion about content this
    process never saw — the exact overstatement this whole check exists to
    stop, aimed at memkit's own diagnostic.
    """
    return _display_cap(
        "; ".join(
            f"{scope.scope} settings {_FAILED[scope.failure]}, so its keys read "
            f"as unset here: {_shown(scope.path)} "
            f"({_display_cap(scope.error, PARSER_SHOWN)})"
            for scope in scopes
            if scope.failure
        ),
        NOTE_SHOWN,
    )


# What happened to a scope, in the words of the thing that happened. One per
# failure kind, because a message that names a failure mode has to be derived
# from the exception rather than written once for the common case.
_FAILED = {
    UNPARSED: "could not be parsed",
    FORBIDDEN: "could not be read: this process is not permitted to open it",
    UNREADABLE: "could not be read",
}


def _unparsed_remedy(scopes: list) -> str:
    """The repair that comes before any other this report could offer.

    NEVER "set the key in that file". Any remedy naming a value to add is
    advice to edit a file whose next read will discard the edit along with
    everything else in it, and that is the shape of this finding: the row that
    could not read the file told its author to write to it.

    AND NEVER "edit it" FOR A FILE NOBODY HERE COULD OPEN. One repair per
    failure kind, in the order an adopter can act on them: a refused open is
    not repaired by editing, and the managed scope's file is an
    administrator's root-owned policy — telling its reader to make it parse,
    and then to keep a copy of it, is advice to edit and copy a file they were
    just denied.
    """
    parts = []
    for kind in (UNPARSED, FORBIDDEN, UNREADABLE):
        named = [scope for scope in scopes if scope.failure == kind]
        if not named:
            continue
        paths = ", ".join(_shown(scope.path) for scope in named)
        if kind == UNPARSED:
            parts.append(
                f"Make {paths} parse as JSON before setting any key in it — "
                "while it does not, nothing in it is read here and a value "
                "added to it changes nothing. Keep a copy first: a file this "
                "report could not parse is one whose contents are still yours."
            )
        elif kind == FORBIDDEN:
            whose = (
                " — for a managed policy file that is your administrator, not "
                "you"
                if any(scope.scope == "managed" for scope in named)
                else ""
            )
            parts.append(
                f"Ask whoever owns {paths} for read access, or run this "
                f"as a user who has it{whose}. Do not edit it on this "
                "report's account: nothing here has seen what is in it."
            )
        else:
            parts.append(
                f"Find out what is at {paths}: the open failed for a reason "
                "that is neither a syntax error nor a permission, so what the "
                "harness reads there is not something this report can say."
            )
    return " ".join(parts)


def _relative_to_cwd(path: str) -> str:
    """`path` as a remedy standing in this directory spells it, or "".

    NORMALISED on both sides and never an escape upwards: a `..` answer is not
    a spelling any remedy here uses, and matching on one would suppress a
    remedy naming some other file entirely.
    """
    cwd = _session_cwd()
    if not path or not cwd:
        return ""
    try:
        relative = os.path.relpath(os.path.normpath(path), os.path.normpath(cwd))
    except (OSError, ValueError):
        return ""
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return ""
    return relative


def _with_unparsed(rows: list, scopes: list) -> list:
    """Every row of a producer that reads settings, told what it could not read.

    ONE WRAPPER at the producer's return rather than a sentence at each branch,
    because the rule is about the INPUT and not about any one answer: a row
    whose verdict came from a walk of the settings scopes is a row that a file
    it could not open can falsify, whichever branch it took. A PASS becomes
    INFO for that reason — the harm this whole check class exists to stop is a
    confident answer resting on a file nobody read.
    """
    note = _unparsed_settings(scopes)
    if not note:
        return rows
    fix = _unparsed_remedy(scopes)
    # EVERY SPELLING A REMEDY USES. The one this rule most needs to drop is
    # `_CHECKOUT_REMEDY`, and it names the file the way an adopter standing in
    # that directory would — relatively; matched on the absolute form alone it
    # walked straight through, and one remedy then said both "make
    # settings.local.json parse" and "set a key in settings.local.json".
    #
    # The RAW absolute path is not a third spelling: what is searched is a
    # remedy off an already-bounded `Check`, where a path under home has been
    # re-spelled `~` — and for a path that is not under home `_shown` returns
    # the same characters back.
    spellings = tuple(
        spelling
        for scope in scopes
        if scope.failure
        for spelling in (_shown(scope.path), _relative_to_cwd(scope.path))
        if spelling
    )
    return [
        Check(
            row.id,
            INFO if row.status == PASS else row.status,
            _detail(note, row.detail),
            # A REMEDY THAT NAMES THE UNREADABLE FILE IS DROPPED rather than
            # appended to. Every one of them names a key to set there, and
            # that is an instruction to write into a file whose next read
            # discards it along with everything else — the half of this
            # finding that is worse than the missing diagnostic.
            " ".join(
                part
                for part in (
                    fix,
                    "" if any(s in row.remedy for s in spellings) else row.remedy,
                )
                if part
            ),
            actor=USER,
            terminal=row.terminal,
        )
        for row in rows
    ]


def authored_configs(state_dir: str) -> set:
    """The config paths init's journal claims, whose claims still cover what is
    at them.

    The journal is append-only JSONL and a partial line is a crash, not a
    corruption: a record that does not parse is skipped rather than taken as
    evidence that nothing was authored. Reading it the other way would turn one
    interrupted init into a `config-authorship` FAIL against memkit's own file.

    A claim is CHECKED against the file rather than taken on the path, because
    the config's claim is written before its file: a crash in that window
    leaves a claim on a path that has nothing at it, and something else can
    then create a config there. `claim_holds` is where that rule lives, shared
    with the hook so the two readers of one journal cannot come to read it
    differently.

    NOTHING at all when the journal's own directory sits inside the session's:
    `$XDG_CACHE_HOME` is an environment variable, so a checkout that exports
    one and checks in a `memory-recall/init-journal.jsonl` gets to write this
    function's answer. What hangs off that answer is whether doctor may run a
    config, and whether init may overwrite one — two authorisations a
    repository must not be able to grant itself. It is the same rule
    `settings_scopes` applies to `$CLAUDE_CONFIG_DIR`, on the file that
    records what was written rather than on the settings that ask for it.
    """
    if _under_cwd(state_dir):
        return set()
    claims = journal_config_claims(state_dir)
    return {path for path, records in claims.items() if claim_holds(path, records)}


class Machine:
    """What the checks read, resolved once for the whole run.

    One object rather than each producer reaching for `os.environ` itself, for
    the reason `_config_state` exists in the hook: two surfaces deriving the
    same answer separately is how they come to disagree, and a diagnostic whose
    halves disagree is worse than no diagnostic.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.explicit_config = config_path
        self.cwd = _session_cwd()
        self.settings = settings_scopes(self.cwd)
        self.state_dir = _state_dir_candidate()
        # The config the WRAPPER settled on, which is the whole of what doctor
        # knows about the rungs: `bin/memkit` resolves them in POSIX sh and
        # exports the answer, so re-resolving them here would be a second copy
        # of the one rule the whole design rests on — and a second copy that
        # agreed would prove nothing, while one that disagreed would be a
        # diagnostic contradicting the thing it diagnoses. What doctor does
        # instead is compare the literal `memkitConfig` a settings scope
        # records against the config this process resolved. That comparison is
        # what makes the set-but-wrong option visible, and it reads SETTINGS
        # rather than the environment: the environment variable reaches hook
        # processes and not this one, so a second copy of the answer here
        # would be an input nothing reads and a comment nobody could trust.
        # What the install ITSELF resolved, kept apart from the answer
        # `--config` overrode it with: the difference is what decides whether
        # this run may execute anything under that config. See `may_probe`.
        self.ambient_config = os.environ.get(CONFIG_ENV) or ""
        self.resolved_config = config_path or self.ambient_config
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

        ADOPTER-OWNED SCOPES ONLY. A memkit config names the interpreter the
        wrapper execs and the directories the every-prompt hook reads, and
        `project`/`local` settings sit in whatever directory the session
        stands in — on a cloned repository, its author's files. A route out of
        one of those is REPORTED by `repository_option` and acted on by
        nothing: not this diagnostic's own probe, and not init, which would
        otherwise write the config where a checkout said to.
        """
        for scope in self.settings:
            value = _option_in(scope)
            if value and scope.adopter_owned:
                return value, scope
        return "", None

    def repository_option(self) -> tuple:
        """The `memkitConfig` a scope in the session's directory records.

        Its own reader because it is its own kind of fact: not a route this
        install serves, but a thing a checkout asked for. Quoted in the report
        so an adopter learns it was asked, with an `actor: user` remedy —
        reading a file before trusting it is not work an agent does on its own
        behalf.
        """
        for scope in self.settings:
            if scope.adopter_owned:
                continue
            value = _option_in(scope)
            if value:
                return value, scope
        return "", None

    def may_probe(self) -> tuple:
        """(True, "") when the config in play is one this install itself
        reads.

        `--config` says "diagnose THIS config", and the only way to honour it
        through a real wrapper run is to hand the wrapper the option variable
        the harness would have set — after which the wrapper execs the
        `interpreter` that config records. So the flag is a way to choose a
        program to run, and the doctor skill pre-approves the argv that passes
        it.

        The rule is that doctor may only cause an execution this install
        already performs: the ambient route (what every prompt does), a config
        init's journal claims to have written, or one an adopter-owned
        settings scope names. Anything else is reported and not run.
        """
        if not self.explicit_config:
            return True, ""
        target = os.path.realpath(self.explicit_config)
        if _under_cwd(target):
            # Before the three acceptance routes, because it closes all of
            # them at once: a config inside the session directory is a file
            # the checkout wrote, and every route below is a way for the
            # checkout to name its own.
            return False, (
                "it resolves inside the directory this session stands in, so "
                "it is a file the checkout supplied rather than a config this "
                "install reads"
            )
        if self.ambient_config and os.path.realpath(self.ambient_config) == target:
            return True, ""
        option, _scope = self.settings_option()
        if option and os.path.realpath(expand_home(option)) == target:
            return True, ""
        for claimed in authored_configs(self.state_dir):
            if os.path.realpath(claimed) == target:
                return True, ""
        return False, (
            f"no route on this install names it: not ${CONFIG_ENV}, not the "
            f"{OPTION_KEY} option in a settings scope you own, and no init "
            "journal entry claims to have written it"
        )

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
    """Which route answered, plus anything a checkout asked for and did not
    get.

    Two rows rather than one where a repository named a config: the second is
    not a route this install serves, and merging it into the first would put a
    path nobody vouched for where a reader looks for the answer.
    """
    return _with_unparsed(
        _resolved_route(machine) + _repository_route(machine), machine.settings
    )


def _repository_route(machine: Machine) -> list[Check]:
    """A `memkitConfig` recorded by a settings scope in the session's own
    directory.

    REPORTED, NEVER FOLLOWED. `.claude/settings.json` is the checkout's file,
    and a memkit config names the interpreter the wrapper execs — so a remedy
    telling an agent to re-run with `--config <that path>` is the whole of the
    route between cloning a repository and running its code as the user. The
    row exists because the adopter should still learn the repository asked;
    the `actor` is theirs because reading a file before trusting it is not
    work an agent does on its own behalf.
    """
    option, scope = machine.repository_option()
    if not option:
        return []
    return [
        Check(
            "config-route",
            INFO,
            f'{scope.scope} settings in this directory record {OPTION_KEY}: '
            f'"{option}". Not followed here, and not written to by init: that '
            "file belongs to whatever checkout the session stands in",
            "Read that file, and the config it names, before trusting either. "
            "A memkit config names the directories the every-prompt hook reads "
            "and the interpreter it execs. If it is yours, set the same value "
            "in your own user settings and it will be used.",
            actor=USER,
        )
    ]


def _resolved_route(machine: Machine) -> list[Check]:
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

    shape = path_refusal(expand_home(option)) if option else ""
    if shape:
        # A SECOND SHAPE OF THE SET-BUT-WRONG OPTION, and the one every check
        # that stats the path answers yes about: `//x/memkit.json` is a file
        # the kernel opens happily and one `memkit_resolve_config` refuses
        # before the hook starts. Reported with the wrapper's own sentence,
        # because the two are one rule.
        return [
            Check(
                "config-route",
                FAIL,
                f'option: "{option}"{where}, which {shape}. The wrapper '
                "refuses that shape and runs as if no config were given, so "
                "this install is inert however readable the path looks",
                f"Reinstall with a canonical absolute path, or edit "
                f"{OPTION_KEY} in "
                f"{scope.path if scope else 'your settings'}. The file may be "
                "fine; the spelling of the path is not.",
                actor=USER,
            )
        ]

    if option and machine.resolved_config != expand_home(option):
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
        expanded = expand_home(option)
        if os.path.isfile(expanded) and os.access(expanded, os.R_OK):
            # The option names a real config and THIS PROCESS was not given it.
            # That is what running the binary from a shell looks like: the
            # option reaches hook processes and nothing else, so a FAIL here
            # would report a healthy install as broken every time somebody
            # diagnosed it by hand — the false RED that matches the false green
            # this whole command exists to prevent.
            #
            # What it cannot distinguish is a hook that is not receiving the
            # option either, which is why it says so and points at the check
            # that can.
            return [
                Check(
                    "config-route",
                    INFO,
                    f'option: "{option}"{where}, which exists and is '
                    "readable. In use by THIS process: nothing — the option "
                    "reaches hook processes only, so a run from a shell needs "
                    "--config",
                    f"To diagnose the config the hook reads, re-run with "
                    f"--config {option}. Whether the hook is receiving it is "
                    "what plugin-enabled and hook-path answer.",
                )
            ]
        why = (
            "exists and cannot be read by this process"
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
                    "Run /memkit:init. On this channel it writes the config "
                    f"to ${PLUGIN_DATA_ENV}/{GENERATED_CONFIG_NAME}, which is "
                    "a rung the hook reads without any option being set.",
                    actor=USER,
                )
            ]
        return [
            Check(
                "config-route",
                FAIL,
                f"no config on any route this install reads ({routes}), so it "
                "is inert: no stores, no pointers, exit 0 on every prompt",
                f"Run {_init_command(machine)}, or write the config by hand "
                "and name it "
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
            f"{_display_path(cfg.path)} parses; schema {SCHEMA}, "
            f"{len(cfg.stores)} store(s)",
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

    Init is the one thing that legitimately writes that file — on a plugin
    install with no `memkitConfig` option, it is the only rung the wrapper
    reads — and it journals every config it authors. That journal is what
    makes an UNCLAIMED one detectable at all.
    """
    return _rung_two_authorship(machine) + _unserialised_writes(machine)


def _unserialised_writes(machine: Machine) -> list[Check]:
    """Any config the journal records having been written without the lock.

    THE JOURNAL'S ONLY READER FOR THIS. The lock is best-effort by design — a
    setup command must not fail because a lock could not be taken, and a
    filesystem with no working `flock` degrades to what every earlier build
    did — but an unserialised write is the one case where a store can go
    missing from a config two inits wrote. Without a check that says so, the
    person who hits exactly that failure has to know to grep a JSONL file
    nothing documents.

    Never a FAIL: a lock that could not be taken is a fact about one write, not
    a broken install, and the store it would explain is usually there.
    """
    claims = journal_config_claims(machine.state_dir)
    unserialised = sorted(
        path
        for path, records in claims.items()
        if any(record.get("unlocked") for record in records)
    )
    if not unserialised:
        return []
    return [
        Check(
            "config-authorship",
            INFO,
            "written while another init held the lock, or with no working "
            f"lock at all: {', '.join(_display_path(p) for p in unserialised)}"
            ". Two inits racing on one config can lose one of their appends, "
            "and this is the record that a write was not serialised",
            "Check that the config lists every store you expect. If one is "
            "missing, run init again for it — the merge is additive, and a "
            "second run adds what the first lost.",
            actor=USER,
        )
    ]


def _rung_two_authorship(machine: Machine) -> list[Check]:
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
                f"Add a store to the config, or run {_init_command(machine)} "
                "to write one.",
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
                    f"it, or run {_init_command(machine)}.",
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
        elif outcome == BUILD_TRUNCATED:
            # ITS OWN ARM, because it is the only outcome that means "this
            # corpus is indexed INCOMPLETELY" — the rest describe how the run
            # went, this one describes what is missing from retrieval — and
            # that is the sentence an adopter whose memory stopped coming back
            # is here to read. Both of the emitter's causes are named, because
            # the reason string it builds names both and they send a reader to
            # different places: over the per-file byte cap is a file to split,
            # out of the run's budget resolves itself on the next run.
            out.append(
                Check(
                    "index-state",
                    INFO,
                    f"{store.id}: last index truncated — part of the corpus "
                    "was not indexed, either over the per-file byte cap or "
                    "out of the run's budget, so retrieval here is INCOMPLETE "
                    "and the file count is a floor rather than a census. The "
                    "next run carries on from where this one stopped",
                    "If this outcome persists, look for a memory over the "
                    f"{INDEX_FILE_MAX_BYTES}-byte file cap: budget truncation "
                    "converges over the following runs and the file cap never "
                    "does, so a file above it is never retrievable until it is "
                    "split.",
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
                    f"description. Re-running {_init_command(machine)} "
                    "reseeds it.",
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


# The event doctor drives the hook with, and therefore the registration whose
# timeout its run has to be read against.
PROBE_EVENT = "UserPromptSubmit"

# How much longer than production doctor waits for the installed hook. A first
# run on a cold index spends most of the registration's budget building, so a
# bound AT the registration's number would report a healthy install as a hang.
#
# HEADROOM rather than a second timeout, and that is the whole of the change:
# a flat 25 here was a number about the same deadline as the registration's 15
# with nothing tying the two, so moving one left the other measuring against a
# deadline that no longer existed. Worse, the check compared its elapsed time
# against NEITHER, so a hook that took 20s — served, and killed by the harness
# at 15 with the prompt going through empty — was reported as the path
# working. The probe still finishes what production would have cut off,
# because the elapsed time is worth having; it is no longer read as a pass.
HOOK_PROBE_HEADROOM = 10


def _registered_timeout(event: str) -> int | None:
    """The timeout this payload's `hooks.json` registers for `event`.

    None when there is no payload, no entry for the event, or nothing
    well-formed to read: the caller falls back to the hook module's own copy
    of the harness budget, which is the number the hook itself works to.
    """
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if not root:
        return None
    with contextlib.suppress(OSError, ValueError, TypeError, AttributeError):
        with open(os.path.join(root, "hooks", "hooks.json"), encoding="utf-8") as f:
            blob = json.load(f)
        for entry in (blob.get("hooks") or {}).get(event) or []:
            for registration in (entry or {}).get("hooks") or []:
                timeout = (registration or {}).get("timeout")
                # `bool` is an `int` and a `True` here is not a budget.
                if (
                    isinstance(timeout, int)
                    and not isinstance(timeout, bool)
                    and timeout > 0
                ):
                    return timeout
    return None


def _probe_budget() -> tuple:
    """`(what production allows, what this probe waits)`, both in seconds.

    Two values from ONE source. The first is what the adopter's own
    registration gives the hook and is what a completed run is judged
    against; the second is that plus the headroom above.
    """
    allowed = _registered_timeout(PROBE_EVENT) or HARNESS_TIMEOUT
    return allowed, allowed + HOOK_PROBE_HEADROOM


# The wrapper a plugin install puts the registration on. Read from
# `CLAUDE_PLUGIN_ROOT` rather than from `hooks.json`, because that variable is
# what the harness itself expands the registration against — and because the
# root arrives with a TRAILING SLASH (measured on 2.1.238 and still true on
# 2.1.241), which `os.path.join` absorbs and naive string arithmetic does not.
HOOK_WRAPPER = "bin/memkit-hook"


def _init_command(machine: Machine) -> str:
    """How to reach init ON THIS CHANNEL.

    Skills ship only in the plugin payload, so `/memkit:init` is a command a
    nix or pip adopter's harness does not have — and those are the channels the
    rollout runbook sends to doctor first. A remedy that guessed would send
    them to a command they cannot run, which is exactly the failure the channel
    check exists one screen earlier to prevent.
    """
    if machine.plugin:
        return "/memkit:init"
    return "`memkit init --dry-run`, then `memkit init --confirm <digest>`"


NO_HOOK_REMEDY = (
    "Nothing serves pointers on this machine until a hook is registered. On "
    "the plugin channel that is `claude plugin install`; on nix it is the "
    "home-manager module."
)


def _payload_roots(machine: Machine) -> list:
    """Where this install's own payload might be, best first.

    `CLAUDE_PLUGIN_ROOT` when the harness set it, and THIS MODULE'S OWN
    LOCATION otherwise, on the plugin channel. The second is not a fallback for
    tidiness: doctor is the command an adopter runs from a shell, and a shell
    gets none of the plugin's environment — so a derivation that needed one
    would leave the payload unlocatable in exactly the state somebody reaches
    for diagnosis.
    It is the same reason each wrapper derives its tree from `$0` rather than
    from the variable.

    The root arrives with a TRAILING SLASH from the harness (measured on
    2.1.238 and still true on 2.1.241), which `os.path.join` absorbs.
    """
    roots = []
    harness = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if harness:
        roots.append(harness)
    module = getattr(sys.modules[__name__], "__file__", "") or ""
    # ONLY on the plugin channel. Off it, a `bin/memkit-hook` sitting beside
    # this module is a source checkout rather than this machine's
    # registration — and probing it would report on a hook nothing here has
    # ever run. `bin/memkit` exports the marker, so a shell invocation through
    # the wrapper qualifies and a bare import does not.
    if module and machine.plugin:
        # <payload>/src/memkit/cli_doctor.py -> <payload>
        # realpath, not abspath: on a mac `/var` is a symlink to
        # `/private/var`, so the two spellings of one temp directory are the
        # same tree and a caller comparing against a resolved path finds
        # neither.
        derived = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(module)
        )))
        if derived not in roots:
            roots.append(derived)
    return roots


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

    NOTHING A REPOSITORY WROTE IS EVER EXECUTED, and that is the sharper half
    of the same rule. `.claude/settings.json` and `.claude/settings.local.json`
    sit in the directory the session stands in, so on a cloned repository they
    are its author's files. This command is model-invocable and its skill
    pre-approves the exact argv, so running it inside somebody else's checkout
    would be that checkout choosing a program to run as the user, with the
    session's whole environment — every token in it — inherited by the child.
    Claude Code gates project-scoped hooks behind a folder-trust prompt; there
    is no such gate here, and there does not need to be one: those scopes are
    REPORTED instead, quoted, which is the half of the check that was ever
    worth having from a directory nobody vouched for.
    """
    for root in _payload_roots(machine):
        path = os.path.join(root, HOOK_WRAPPER)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return [path], f"the plugin's own wrapper at {_display_path(path)}", ""
    reported = ()
    for scope in machine.settings:
        events = scope.data.get("hooks")
        if not isinstance(events, dict):
            continue
        for entry in events.get("UserPromptSubmit") or []:
            for spec in (entry or {}).get("hooks") or []:
                command = (spec or {}).get("command")
                if not isinstance(command, str) or "memkit" not in command:
                    continue
                if not scope.adopter_owned:
                    # Kept as the fallback answer rather than returned at once:
                    # an adopter-owned entry further down the list is still
                    # worth running, and this one is worth SAYING either way.
                    reported = reported or (
                        f"the {scope.scope}-settings registration runs "
                        f'"{command}". That file is in the directory this '
                        "session stands in, so it is not run from here — a "
                        "diagnostic that executed what a checkout registered "
                        "would be running whatever the checkout chose",
                        "This directory registers a UserPromptSubmit hook. "
                        "Read the command above before trusting it: a "
                        "project-scoped hook is whatever the checkout's "
                        "author put there, and Claude Code asks you about "
                        "those separately.",
                    )
                    continue
                if not (os.path.isfile(command) and os.access(command, os.X_OK)):
                    return (
                        [],
                        f"the {scope.scope}-settings registration runs "
                        f'"{command}", which is not an executable file this '
                        "can run on its own",
                        NO_HOOK_REMEDY,
                    )
                if _under_cwd(command):
                    # Defence in depth, for what the scope rule cannot see: the
                    # scope says an adopter wrote the ENTRY and says nothing
                    # about who wrote the file it points at.
                    return (
                        [],
                        f"the {scope.scope}-settings registration runs "
                        f'"{command}", which resolves inside this directory. '
                        "Not run from here",
                        "Read the command above before trusting it. A hook "
                        "whose program lives in the directory the session "
                        "stands in is that directory's choice, whichever "
                        "settings scope names it.",
                    )
                return [command], f"the {scope.scope}-settings registration", ""
    if reported:
        return [], reported[0], reported[1]
    return [], "nothing registers a UserPromptSubmit hook for memkit", NO_HOOK_REMEDY


# What the hook probe carries over from THIS session, named rather than
# reached for inline. The child's environment is built from an allow-list, so
# every one of these is a deliberate re-admission — and the probe's whole
# purpose is to run the wrapper the way the harness runs it, which means the
# harness's own plugin variables have to arrive. Declared as a tuple so the
# check's detail can print what the run inherited: a probe whose environment
# differs from a real invocation's is a probe whose result means something
# else, and the difference has to be readable rather than argued.
HOOK_PROBE_FORWARD = (
    PLUGIN_DATA_ENV,
    CONFIG_DIR_ENV,
    "CLAUDE_PLUGIN_ROOT",
    # The harness's own plugin option, which is how an install DELIVERS the
    # config on the channel this probe exists to exercise. Dropping it would
    # leave the probe testing a wrapper with no config to find and calling the
    # silence a failed retrieval. `env_extra` sets the same name when
    # `--config` was given, and is applied after this, so the flag still wins.
    "CLAUDE_PLUGIN_OPTION_" + OPTION_KEY.upper(),
    "XDG_CACHE_HOME",
)


# The variables the HOOK ITSELF reads, so a probe can say which of them this
# session has and it did not pass on. The child's environment is built rather
# than inherited, which is the property that makes it safe; the cost is that a
# probe is not a real invocation, and the difference has to be reported rather
# than left for an adopter to discover by disbelieving a FAIL.
# NOT `MEMKIT_CONFIG` and not `MEMKIT_PLUGIN`: the wrapper sets both itself,
# hard and in both directions, so forwarding them would make the probe test a
# delivery the install does not perform. What is listed is what a SESSION
# supplies and the wrapper reads.
HOOK_READS_ENV = (
    PLUGIN_DATA_ENV,
    CONFIG_DIR_ENV,
    "CLAUDE_PLUGIN_ROOT",
    "CLAUDE_PLUGIN_OPTION_" + OPTION_KEY.upper(),
    "XDG_CACHE_HOME",
    "HOME",
)


def _probe_env_gap() -> list:
    """Names this session carries that the hook reads and the probe drops.

    Non-empty means the probe and a real invocation differ in a way that could
    explain a failure, which is the difference between "your install is broken"
    and "this measurement does not settle it".

    On the shipped lists this is always empty, and that is the point: it is the
    allow-list's own incompleteness made visible. A name added to what the
    wrapper reads, without being added to what the probe forwards, turns every
    hook-path failure into an UNKNOWN that says which name — rather than into a
    FAIL that sends an adopter to repair a store that works.
    """
    carried = set(CHILD_ENV_KEEP) | set(HOOK_PROBE_FORWARD)
    return sorted(
        name
        for name in HOOK_READS_ENV
        if name not in carried and os.environ.get(name)
    )


def _probe_hook(
    machine: Machine, command: list, prompt: str, timeout: int
) -> tuple:
    """One real run of the installed hook. Returns (stdout, stderr, code, ms).

    `timeout` is PASSED IN rather than read here, because the caller reports
    the elapsed time against the registration's own budget and the two numbers
    have to come from one read. A second `_probe_budget()` call would agree
    with the first in every run anyone can construct, and would let the row
    quote a bound that was not the one governing the run.

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
    # The DELTA, not a copy of the environment: `_execute` BUILDS the child's
    # from an allow-list, and there is no `env=` to hand it a snapshot of
    # `os.environ` with.
    env = {DOCTOR_ENV: "1"}
    if machine.explicit_config:
        # `--config` says "diagnose THIS config", and the wrapper reads its own
        # rungs — it deliberately ignores `$MEMKIT_CONFIG` — so the only way to
        # honour the flag through a real wrapper run is to hand it the option
        # variable the harness would have set.
        #
        # What that costs is stated in the check's own detail rather than
        # hidden: with the flag, this proves the wrapper, the interpreter and
        # retrieval, and NOT that the install's own route delivers the config.
        # Without it, the run tests the delivery too.
        env["CLAUDE_PLUGIN_OPTION_" + OPTION_KEY.upper()] = machine.explicit_config
    try:
        out = _execute(
            command,
            input=payload,
            timeout=timeout,
            env_extra=env,
            env_forward=HOOK_PROBE_FORWARD,
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


# One pointer line, as `_pointer_line` builds it: `- <path> — <description>`
# with the evidence tags after. Matching the SHAPE rather than a substring is
# what makes the probe a delivery check: the canary's name appearing anywhere
# in stdout is something a stub, a stale wrapper or a hook that died after
# printing can all arrange.
_POINTER = re.compile(r"^- (\S.*?) — ", re.MULTILINE)

# The frame's opening delimiter, as `_frame_tag` draws it on both paths:
# the shared tag, a `-`, FRAME_NONCE_BYTES bytes of hex, and the attributes
# the opener declares (`lines=N`). Captured so
# the closer can be built from the SAME draw — two frames in one output
# would carry two nonces, and pairing an opener with another's closer is
# how a probe reads a truncated frame as a closed one.
_FRAME_OPEN = re.compile(
    rf"<({re.escape(FRAME_TAG)}-[0-9a-f]{{{FRAME_NONCE_BYTES * 2}}})(?:\s[^>\n]*)?>"
)


def _delivered_canary(stdout: str, code, cfg) -> tuple:
    """(True, "") when this output really is a pointer to the canary.

    Four things have to hold, and each was reachable without the others: the
    hook exited 0, the frame is there and closed, a pointer line parses out of
    it, and the path that line names is the canary and is a file that exists.
    A pointer names something to open, so one naming a path that is not there
    is a line the agent cannot act on — counting it as a delivery is the same
    false green one exit code up.
    """
    if code != 0:
        return False, f"it exited {code}"
    inside = None
    for match in _FRAME_OPEN.finditer(stdout):
        closed = f"</{match.group(1)}>"
        rest = stdout[match.end():]
        if closed in rest:
            inside = rest.split(closed, 1)[0]
            break
    if inside is None:
        return False, "the output carries no closed pointer frame"
    paths = _POINTER.findall(inside)
    if not paths:
        return False, "the frame holds no pointer line"
    named = [path for path in paths if os.path.basename(path) == CANARY_NAME]
    if not named:
        return False, (
            "the pointers are to " + ", ".join(os.path.basename(p) for p in paths[:3])
        )
    for path in named:
        # `~`-expanded, because that is how a pointer renders a path: the
        # emitter writes them `~`-relative on purpose, so they are unambiguous
        # from any cwd. The probe runs the hook with this process's own
        # environment, so the same `~` resolves to the same home.
        if os.path.isfile(os.path.expanduser(path)):
            return True, ""
    return False, f"it points at {named[0]}, which does not exist"


@_produces("hook-path")
def _hook_path(machine: Machine) -> list[Check]:
    """Run the installed hook once, and say whether a pointer came out.

    DOCTOR MAY NEVER REPORT GREEN WITHOUT THIS. A fixed-query retrieval proves
    the store; it proves nothing about the wrapper that finds the config, the
    interpreter that runs the module, or the registration that reaches either —
    and that span is exactly where both walkthroughs' installs were broken
    while every other light was green.
    """
    command, how, remedy = _installed_hook(machine)
    if not command:
        return [
            Check("hook-path", UNKNOWN, f"no hook was run: {how}",
                  remedy, actor=USER)
        ]
    may, why = machine.may_probe()
    if not may:
        # The wrapper execs the `interpreter` its config records, so probing
        # under a config this install does not itself read is that config
        # choosing a program to run as the user — and this command's skill
        # pre-approves the argv that would do it. The signal is not dropped
        # silently: the row says what did not happen and what would let it.
        return [
            Check(
                "hook-path",
                UNKNOWN,
                f'no hook was run: --config named "'
                f'{_display_path(machine.explicit_config or "")}" and {why}. '
                "A config names the interpreter the wrapper execs, so this "
                "does not run one nothing here vouches for",
                f"Diagnose the config this install reads by running with no "
                f"--config. To diagnose that file, point ${CONFIG_ENV} at it "
                "in your own shell and re-run, or set it as the "
                f"{OPTION_KEY} option in settings you own.",
                actor=USER,
            )
        ]
    cfg = machine.config()
    nonce = cfg.canary_nonce if cfg is not None else ""
    # Read once, and both from the registration: `allowed` is what production
    # gives the hook and is what a COMPLETED run is judged against, `waited`
    # is what this probe is willing to sit through so the elapsed time exists
    # to judge.
    allowed, waited = _probe_budget()
    if not nonce:
        stdout, stderr, code, ms = _probe_hook(
            machine, command, "memkit doctor probe prompt", waited
        )
        if code is None:
            return [
                Check(
                    "hook-path",
                    FAIL,
                    f"{how} did not finish inside {waited}s, which is "
                    f"already {HOOK_PROBE_HEADROOM}s past the {allowed}s "
                    f"the registration allows it ({stderr[:120]})",
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

    stdout, stderr, code, ms = _probe_hook(
        machine, command, canary_query(nonce), waited
    )
    if code is None:
        return [
            Check(
                "hook-path",
                FAIL,
                f"{how} did not finish inside {waited}s, which is already "
                f"{HOOK_PROBE_HEADROOM}s past the {allowed}s the "
                f"registration allows it. {stderr[:120]}",
                "The registration gives up at its own timeout and your prompt "
                "goes through without pointers. A first run on a large store "
                "builds the index; if this repeats, the store is too large or "
                "the index cannot be written.",
                actor=USER,
            )
        ]
    delivered, why = _delivered_canary(stdout, code, cfg)
    if delivered:
        supplied = (
            "; the config came from --config, so this proves the wrapper and "
            "retrieval and not that the install's own route delivers it"
            if machine.explicit_config
            else ""
        )
        if ms > allowed * 1000:
            # DELIVERED, and too late to have been delivered. The probe
            # outwaits production on purpose, so reaching here means the run
            # finished in a window the harness would have ended: at this
            # latency the prompt goes through with no pointers and the hook
            # records `killed`. Reporting it as a pass is the one wrong green
            # this check is least entitled to, since the adopter came here to
            # ask whether the path serves.
            #
            # Not a FAIL: a first run on a cold index legitimately spends more
            # than the budget building, once, and a red verdict for every
            # fresh install would cost this report the reader it is written
            # for. What repeats is visible in `gate-outcomes` as `killed`.
            return [
                Check(
                    "hook-path",
                    INFO,
                    f"{how} emitted a framed pointer to {CANARY_NAME}, but "
                    f"took {ms}ms where the registration allows {allowed}s — "
                    "at this latency the harness ends the hook first and the "
                    f"prompt goes through with no pointers{supplied}",
                    "A first run on a cold index does this once and the next "
                    "is warm. If it repeats, the corpus is too large for the "
                    "budget or the index cannot be written — check "
                    "gate-outcomes for `killed`, and index-state for a "
                    "truncated sync.",
                    actor=USER,
                )
            ]
        return [
            Check(
                "hook-path",
                PASS,
                f"{how} emitted a framed pointer to {CANARY_NAME} in "
                f"{ms}ms{supplied}",
            )
        ]
    gap = _probe_env_gap()
    detail = (
        f"{how} exited {code} in {ms}ms and delivered no pointer to "
        f"{CANARY_NAME}: {why}. The probe ran under a built environment "
        f"({', '.join(CHILD_ENV_KEEP)} plus "
        f"{', '.join(HOOK_PROBE_FORWARD)}) and a PATH rebuilt from the entries "
        f"no checkout can steer. stderr: {stderr[:200] or '(empty)'}"
    )
    if gap:
        # UNKNOWN, because this run is not the thing it claims to measure. A
        # variable the hook reads, that this session has and the probe did not
        # pass on, is a difference between the probe and a real invocation —
        # and reporting that as a failure sends an adopter to repair a store
        # that is working.
        return [
            Check(
                "hook-path",
                UNKNOWN,
                f"{detail} This session also carries {', '.join(gap)}, which "
                "the hook reads and the probe does not forward, so a real "
                "invocation and this one did not see the same environment",
                "Re-run without those set, or set them the way your install "
                "does, to get a result that describes the installed path.",
                actor=USER,
            )
        ]
    return [
        Check(
            "hook-path",
            FAIL,
            detail,
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
    "index-unavailable": "a store was asked and could not answer — an index "
    "mid-rebuild, an unreadable corpus, or a query the budget ran out under",
    "gate:event": "a prompt-shaped payload arrived under an event name this "
    "hook did not register for",
    # Written before the dispatch chose a path, so it belongs to neither
    # vocabulary: the payload never said which one it was.
    "main:badpayload": "stdin held no JSON object — empty, truncated, "
    "malformed, or valid JSON that is not an object",
    # --- the subagent path -----------------------------------------------
    #
    # A SEPARATE VOCABULARY, rendered in the same histogram. The two hooks
    # serve different populations — one prompt each, one Agent spawn each —
    # and the prefix is what keeps a rate computed over either from being a
    # rate over an unknown mixture of both. Every name here has a row in the
    # README's outcome table and the two are pinned to each other by
    # `test_every_outcome_the_readme_publishes_has_a_reason_doctor_can_render`.
    "task:injected": "pointers were appended to a subagent's brief",
    "task:envelope": "the brief began with an editor or tool envelope",
    "task:empty": "the brief was empty after stripping",
    "task:slash": "the brief began with `/`",
    "task:short": "the brief was under three words",
    "task:stopwords": "the brief was all common words",
    "task:nodirs": "nothing to search: no config, or no store on disk and in "
    "scope here",
    "task:nomatch": "the stores were searched and nothing came back",
    "task:deduped": "every match had already been offered for this tool call",
    "task:floored": "matches existed and none cleared the relevance bar",
    "task:index-unavailable": "a store was asked and could not answer; "
    "parallel spawns share one index and a contender can lose the race to a "
    "cold build",
    "task:oversize": "the brief plus its pointers would exceed the 16 KiB "
    "write bound, so the pointers were dropped whole",
    "task:unsafe": "the emission did not match the one permitted output "
    "shape, so nothing was written — this one is a defect report",
    "task:notool": "the hook was called for a tool other than `Agent`",
    "task:event": "an `Agent` call arrived under an event name this build "
    "does not recognise",
    "task:nobrief": "the tool call carried no `prompt` string to read",
    "task:unencodable": "the brief carried a lone surrogate, so the emission "
    "cannot be written as UTF-8 at all",
    "task:killed": "the subagent hook ran out of time and gave up rather than "
    "stalling the spawn",
    "task:output-lost": "pointers were built and the write did not land",
    "task:error": "the subagent path raised; the turn was unaffected",
}

# The TRUST MARKER's vocabulary, which is a different file's and therefore a
# different map. `trust.json` is what a plugin install with no config writes
# instead of a soak log — creating the shared state directory is a mutation
# nobody asked for — so these two names never appear in `log.jsonl` and the
# README documents them in the same table under that caveat. Kept apart from
# `OUTCOME_REASONS` rather than folded into it because that map is pinned to
# the log table by an equality assertion in both directions, and a name in it
# that the log cannot hold would be a documented outcome nothing writes.
#
# Two names, and both are glossed for a reason the other half of this file
# already records: `_plugin_diagnostics` explained one of them in its remedy
# and printed the other bare, which left the adopter whose config is broken —
# the one this check exists for — reading the undefined name.
MARKER_REASONS = {
    "trust:unconfigured": "no config resolved on any route this install reads, "
    "so the hook refused before it would have created the state directory. A "
    "`memkitConfig` that is set and WRONG records this too: the wrapper "
    "refuses the path before the hook learns one was named",
    "trust:config-error": "a config was found and could not be used — "
    "unreadable, unparseable, or a schema this build does not speak",
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


def _gloss(outcome: object, reasons: dict | None = None) -> str:
    """What an outcome MEANS, in the one place that decides it.

    Every renderer of an outcome name goes through here, and that is the whole
    point: the map held twenty `task:*` reasons that no reader ever consulted,
    because the one check rendering those names printed them raw. A reader
    holding its own idea of how to say a name is a reason nobody reads, and
    doctor carried four of them.

    `reasons` because there are two maps and they may not merge: the soak log's
    vocabulary and the trust marker's are written to different files by
    different code paths, and `OUTCOME_REASONS` is pinned to the README's log
    table by an equality assertion in both directions. Which map a caller reads
    is a fact about which file it read the record out of.

    An unrecognised name is a STATEMENT rather than a silence, for the same
    reason `_index_state` treats an unrecognised index outcome as not-ok: these
    vocabularies grow without a version bump, so meeting a name from a newer
    build is the expected case and not an error.
    """
    if not isinstance(outcome, str):
        return "a record with no outcome name"
    table = OUTCOME_REASONS if reasons is None else reasons
    return table.get(outcome, "an outcome this build does not know")


def _prompt_records(records: list) -> list:
    """The per-prompt population, per the log's own published rule.

    `concludes: false` marks a record that is about the machine rather than
    about a prompt, and doctor's own probe carries `doctor: true`. Both are
    excluded here, or a report about how often prompts inject would be counting
    the runs doctor itself made.

    `population` is the third exclusion and it is not redundant with the first:
    the subagent path writes both stamps, so today either one alone would do —
    but they are two facts and a record carrying one without the other has to
    land somewhere deliberate rather than in whichever population happened to
    ask first. A spawn is never a prompt, whatever else its record says.
    """
    return [
        r
        for r in records
        if r.get("concludes") is not False
        and not r.get("doctor")
        and not _stamped_task(r)
    ]


def _stamped_task(record: dict) -> bool:
    """The log's own discriminator for the subagent path, in the one place that
    spells it. Every reader below is built out of this and `_names_task`, so
    the partition is one fact each rather than four predicates that agree."""
    return record.get("population") == TASK_POPULATION


def _names_task(record: dict) -> bool:
    """Named out of the subagent vocabulary, whatever the stamp says. The
    NAMING CONVENTION rather than the discriminator, and only ever used as a
    tripwire — a reader that partitioned on it would have to learn each new
    outcome's name to keep counting the same population."""
    outcome = record.get("outcome")
    return isinstance(outcome, str) and outcome.startswith(TASK_OUTCOME_PREFIX)


def _task_records(records: list) -> list:
    """The subagent population, per the log's own published discriminator.

    What a COUNT is taken over, so it is the stamp alone: a record both
    populations could claim would otherwise be counted twice, and a rate
    nobody can reproduce is the defect this check was added to close.
    """
    return [r for r in records if _stamped_task(r)]


def _stray_task_records(records: list) -> list:
    """Records named out of the subagent vocabulary that the discriminator did
    not claim.

    The tripwire for the partition itself. Neither population may silently
    absorb these and neither may silently drop them: a `task:` outcome without
    the stamp is either a build older than the stamp or an emitter that grew a
    record some other way, and both are things a reader has to be told rather
    than have averaged into a count.
    """
    return [r for r in records if _names_task(r) and not _stamped_task(r)]


def _subagent_records(records: list) -> list:
    """Every record the subagent path wrote, by EITHER witness, in log order.

    What an EXISTENCE question is asked over, which is the opposite trade from
    the count above: "has a subagent ever been served here" may not miss its
    evidence because the record carried the other one of two witnesses. The two
    halves above are disjoint by construction — one requires the stamp and the
    other requires its absence — so this union double-counts nothing.
    """
    return [r for r in records if _stamped_task(r) or _names_task(r)]


def _histogram(records: list) -> str:
    """`name count (what it means)`, commonest first, with the median latency.

    One renderer for both populations, because they are the same question
    asked over two sets of records and two copies of this would drift the way
    the reasons themselves did.
    """
    counts: dict = {}
    for record in records:
        outcome = record.get("outcome")
        if isinstance(outcome, str):
            counts[outcome] = counts.get(outcome, 0) + 1
    parts = [
        # An outcome this build does not know is REPORTED rather than dropped:
        # the vocabulary grows without a version bump, and a reader that
        # silently discarded a name it did not recognise would compute a rate
        # over a denominator nobody checked.
        f"{outcome} {count} ({_gloss(outcome)})"
        for outcome, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    took = sorted(r["ms"] for r in records if isinstance(r.get("ms"), int))
    median = f"; median {took[len(took) // 2]}ms" if took else ""
    return ", ".join(parts) + median


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
    everything = _soak_tail(machine.state_dir, GATE_WINDOW)
    records = _prompt_records(everything)
    if not records:
        if not everything and not os.path.isfile(
            os.path.join(machine.state_dir, SOAK_LOG_NAME)
        ):
            return [
                Check(
                    "hook-ever-fired",
                    UNKNOWN,
                    f"no {SOAK_LOG_NAME} in {_display_path(machine.state_dir)}: "
                    "this hook has never run. An install that was never "
                    "configured writes none, deliberately",
                )
            ]
        # The file is THERE and holds nothing this question is about — most
        # often because doctor's own probe put the only record in it two checks
        # ago. Telling an adopter to look for a file they will find is the one
        # thing a report whose value is measurement may not do.
        return [
            Check(
                "hook-ever-fired",
                UNKNOWN,
                f"{SOAK_LOG_NAME} holds {len(everything)} record(s) and none "
                "of them from a prompt — a doctor probe and the search "
                "command both write here. No prompt has been served yet",
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
                f"{mine[-1].get('outcome')!r} ({_gloss(mine[-1].get('outcome'))}). "
                "See gate-outcomes",
            )
        ]
    return [
        Check(
            "hook-ever-fired",
            INFO,
            f"the hook has run ({len(records)} records, last {last.get('outcome')!r} "
            f"— {_gloss(last.get('outcome'))} — at {when}) and never in this "
            "directory",
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

    THE PER-PROMPT POPULATION ONLY, and it says so. It counted that population
    from the day it was written and labelled the count "last N prompts", which
    is true — but a log now holding two populations makes a count that names
    one of them and never mentions the other read as a count of everything.
    `task-outcomes` is the other half, and the two never merge: a rate over the
    union is a rate over an unknown mixture of a 15-second budget and a
    7-second one.
    """
    records = _prompt_records(_soak_tail(machine.state_dir, GATE_WINDOW))
    if not records:
        return [
            Check("gate-outcomes", INFO, "no records yet, so nothing to count")
        ]
    return [
        Check(
            "gate-outcomes",
            INFO,
            f"last {len(records)} prompts (this hook only; subagent spawns are "
            f"counted by task-outcomes): " + _histogram(records),
        )
    ]


@_produces("task-outcomes")
def _task_outcomes(machine: Machine) -> list[Check]:
    """The same histogram over the SUBAGENT population.

    Its own row rather than a share of `gate-outcomes`, because the populations
    may not be summed — one record per prompt against one per spawn, two gates,
    two budgets — and because an adopter whose subagents get nothing is here to
    read counts. `subagent-delivery` answers whether the path is wired up and
    names the LAST outcome; "did it ever work" and "what keeps happening" are
    different questions, and `task:floored` twenty times over (the bar, or the
    corpus) is a different next move from `task:nodirs` twenty times over (the
    config, or the cwd gate) or `task:oversize` (nothing in the store to fix).
    Until this row existed the only way to ask was `tail`-ing the log by hand,
    which is what the README told a reader to do.

    Always INFO, for the reason its sibling is: every one of these is a state
    the path is designed to reach.
    """
    window = _soak_tail(machine.state_dir, GATE_WINDOW)
    records = _task_records(window)
    stray = _stray_task_records(window)
    # NAMED rather than absorbed. A `task:` record the discriminator did not
    # claim is either older than the stamp or built some other way, and the one
    # thing a report may not do with it is average it into a count silently —
    # which is the defect this whole check was added to close, arriving from
    # the other direction.
    aside = (
        f". {len(stray)} record(s) name a {TASK_OUTCOME_PREFIX} outcome without "
        f"the {TASK_POPULATION!r} population stamp and are counted in neither "
        "population; this build cannot say which one they belong to"
        if stray
        else ""
    )
    if not records:
        return [
            Check(
                "task-outcomes",
                INFO,
                "no subagent records yet, so nothing to count" + aside,
            )
        ]
    return [
        Check(
            "task-outcomes",
            INFO,
            f"last {len(records)} subagent spawns: " + _histogram(records) + aside,
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

# Where the harness keeps the built-in memory feature's per-project state:
# `<config dir>/projects/<project key>/memory/`, derived in `harness_memory`.
# The key was measured on 2.1.241 as the sanitized cwd and RE-MEASURED on
# 2.1.258, against the binary and code.claude.com/docs/en/memory, as the
# sanitized git repository root — so every worktree and subdirectory of one
# checkout shares one directory. The lock beside it is how "armed" and
# "actually running" are told apart.
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
    """Exactly one, or say which entries to choose between — over every scope
    this process could read.

    Two registrations both serving one prompt is a silent lost update from
    inside: each process injects, each writes the session ledger, and the later
    write wins. What the user sees is pointers that come and go for no reason.
    """
    return _with_unparsed(_registration_rows(machine), machine.settings)


def _registration_rows(machine: Machine) -> list[Check]:
    """The count, before anything qualifies it."""
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
    return _with_unparsed(_plugin_enabled_rows(machine), machine.settings)


def _plugin_enabled_rows(machine: Machine) -> list[Check]:
    """The answer the scopes that parsed give."""
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
    unreadable = ""
    if marker and os.path.isfile(marker):
        try:
            with open(marker, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError) as exc:
            # NOT "no refusals". A marker that is there and cannot be read is
            # the evidence for the install this check exists to diagnose, and
            # reporting its absence as a clean bill is the same false green
            # every other check here is written against.
            unreadable = str(exc)
            blob = None
        if isinstance(blob, dict):
            loaded = blob.get("records")
            records = loaded if isinstance(loaded, list) else []
        elif blob is not None:
            unreadable = f"its top level is a {type(blob).__name__}"
    dups = [
        r
        for r in _soak_tail(machine.state_dir, GATE_WINDOW)
        if r.get("outcome") == "dup-registration"
    ]
    if unreadable:
        return [
            Check(
                "plugin-diagnostics",
                UNKNOWN,
                f"{_display_path(marker)} exists and could not be read "
                f"({unreadable}), so what this install refused is not knowable "
                "from here",
                "That file is the only record of a refusal the harness "
                "swallowed. Read it, or delete it and reproduce the state.",
                actor=USER,
            )
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
    # THE REASON BESIDE EACH NAME, from the marker's own map. The remedy below
    # glossed `trust:unconfigured` and left `trust:config-error` — the other
    # half of a two-name vocabulary — explained nowhere in the report, so the
    # adopter whose config is broken read the one name that was not defined.
    # Same class as the `task:*` reasons above and found the same way, by
    # enumerating what reads an outcome rather than by meeting it.
    parts = [
        f"{name} x{count} ({_gloss(name, MARKER_REASONS)})"
        for name, count in sorted(outcomes.items())
    ]
    if dups:
        parts.append(
            f"dup-registration x{len(dups)} in the soak log "
            f"({_gloss('dup-registration')})"
        )
    return [
        Check(
            "plugin-diagnostics",
            INFO,
            f"{len(records)} refusal(s) across {where} directory/ies: "
            + ", ".join(parts),
            "A `trust:` refusal is about the config this install resolved — "
            "see config-route, which is the one surface that can tell "
            "\"named and wrong\" from \"never named\". `dup-registration` "
            "means two installs served one prompt — see registrations-count.",
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
                (
                    "no PreToolUse-on-Agent entry in this payload's hooks.json"
                    if root
                    else "there is no plugin payload here to register a "
                    "PreToolUse hook from"
                )
                + ", so the subagent path is not in this build. Subagents get "
                "no pointers and nothing is wrong",
            )
        ]
    # EITHER WITNESS, because the question here is existence rather than a
    # rate: a record carrying the population stamp and a record merely named
    # out of this path's vocabulary are both proof the path fired. Where
    # `task-outcomes` takes the stamp alone, since a count has to be over one
    # defined population, this one may not miss its evidence for having looked
    # at the other witness.
    task = _subagent_records(_soak_tail(machine.state_dir, GATE_WINDOW))
    if not task:
        # WHAT WAS CHECKED, in the words of what was read. The evidence behind
        # this row is one file inside the payload, so what it establishes is
        # that the payload DECLARES the hook — not that the harness registered
        # it. An install where the harness accepted `UserPromptSubmit` and
        # dropped `PreToolUse` produces this row too, and is indistinguishable
        # from a healthy install nobody has spawned a subagent in.
        #
        # The second state is by far the commoner one, so the first would hide
        # behind it indefinitely. Hence a remedy: the ambiguity is cheap to
        # resolve once a reader knows there is one, and `plugin details`
        # separates the two in one line.
        return [
            Check(
                "subagent-delivery",
                INFO,
                "this payload declares the PreToolUse/Agent hook, and no "
                "subagent has fired here yet — which is also what a harness "
                "that registered only the per-prompt hook looks like from "
                "inside the payload",
                "Run `claude plugin details " + PLUGIN_KEY + "`: `Hooks (2)` "
                "means both are registered and this row is simply waiting for "
                "a spawn, `Hooks (1)` means the harness took one entry and "
                "not the other and the plugin needs reinstalling. Either way, "
                "spawning a subagent and re-running this makes the answer "
                "positive rather than absent.",
                actor=USER,
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
    # THE REASON, not just the name. Every `task:*` outcome has had a line in
    # `OUTCOME_REASONS` since the day the vocabulary landed and this is the
    # check that renders those names — it printed them raw, so all twenty of
    # those lines were unreachable and the row an adopter met said `task:killed`
    # and stopped. Sent through the same gloss as both histograms, which is
    # also what makes a name from a newer build say so rather than sit there
    # looking like a word the reader ought to recognise.
    return [
        Check(
            "subagent-delivery",
            INFO,
            f"registered and firing; the last outcome was {last!r} "
            f"({_gloss(last)}) rather than a delivery",
            "`task-outcomes` counts the whole window rather than naming the "
            "last record, which is the row to read if this keeps happening.",
            actor=USER,
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
    try:
        binary = resolve("claude")
    except Untrusted as exc:
        return [
            Check(
                "harness-stamp",
                UNKNOWN,
                f"no `claude` this may run ({exc}), so the running harness "
                f"version is not knowable from here. memkit's claims were "
                f"measured against {MEASURED_HARNESS}. A `claude` found only "
                "through the session's own PATH is not asked: that lookup is "
                "one a checkout steers",
            )
        ]
    try:
        out = _execute([binary, "--version"], timeout=30)
    except (OSError, subprocess.SubprocessError, Untrusted) as exc:
        # The MESSAGE, not just the class. `Untrusted` names which rule refused
        # and what it refused; `OSError` names the errno. A reader told only
        # the type has to reproduce the failure to learn anything from it, and
        # a refusal that reports the same observable as an ordinary negative
        # result is not a refusal.
        return [
            Check(
                "harness-stamp",
                UNKNOWN,
                f"`claude --version` could not be run "
                f"({type(exc).__name__}: {exc}); memkit's claims were "
                f"measured against {MEASURED_HARNESS}",
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


# How many project directories the inventory summary names before it counts the
# rest. Five is what fits beside the other evidence inside `DETAIL_MAX_BYTES`;
# the total is stated whatever the list shows, so the number a reader acts on is
# never the length of the list.
INVENTORY_SHOWN = 5


def _count(number: int, one: str, many: str) -> str:
    return f"{number} {one if number == 1 else many}"


def _detail(*parts: str) -> str:
    """One detail from the sentences that have something to say.

    FIXED FACTS FIRST at every call site, because `_bound` cuts from the end
    and only the list of project directories has a length an adopter's own
    machine decides.
    """
    return "; ".join(part for part in parts if part)


def _shown(path: str) -> str:
    """One path as a detail prints it: `~`-relative, and bounded.

    Both halves at one call, because they were two rules applied at different
    call sites: every path went through `_display_path` and none of them
    through a cap, so the whole of `DETAIL_MAX_BYTES` was one value's to spend.
    """
    return _display_cap(_display_path(path), PATH_SHOWN)


def _resolved(path: str) -> str:
    """One path in the spelling both sides of a containment test are compared in.

    NORMALISED AT THE COMPARISON AND NEVER AT A SOURCE. `harness_dir` hands
    back NFC because that is what the harness writes to, and a corpus root out
    of `memkit.json` is spelled however that file spells it — so a composed
    character made two strings out of one directory and the comparison below
    answered about the encoding. Applied to whatever each side happens to
    carry, rather than to one of them on the way in, because the pair that
    reaches here comes by two routes and only one of them has a normaliser.

    No guard of its own: a path that will not resolve is `realpath`'s to raise
    on, and the callers' own `except` is where that answer belongs.
    """
    return unicodedata.normalize("NFC", os.path.realpath(path))


def _within(child: str, parent: str) -> bool:
    """Whether `child` is `parent` or sits under it, symlinks resolved.

    Resolved on both sides, because the two paths compared here reach this
    function by different routes — one typed into a settings file, one built
    from the config's roots — and on a mac `/tmp` and `/var` are symlinks that
    make two spellings of one directory differ character by character.

    NOT `_exec._under_cwd`, whose containment arithmetic is the same one: the
    safe answer when a path will not resolve is the opposite there. That one
    says "inside" so an unresolvable path stays untrusted; this one says
    "outside" so no path that cannot be resolved is reported as retrieved.

    `ValueError` beside `OSError` because the child is adopter-supplied text:
    `realpath` raises that, not `OSError`, on the embedded NUL a settings file
    may legally carry.
    """
    try:
        child = _resolved(child)
        parent = _resolved(parent)
    except (OSError, ValueError):
        return False
    return child == parent or child.startswith(parent + os.sep)


def _pruned(directory: str, root: str) -> bool:
    """Whether the indexing walk refuses to descend from `root` to `directory`.

    THE INDEXER'S OWN PREDICATE, not a second copy of it: `_fts_scan` asks
    `_excluded` of the whole path it built, so that is what is asked here, of
    the same path — `root` as the config spells it, with the components the
    walk would descend through appended. A corpus-relative test of memkit's
    own answered "retrieved" about a store whose root sits under an `archive/`
    or `hot/` component, where the walk indexes nothing at all.

    That consequence is the HOOK's behaviour and it is left alone here: a store
    under a pruned name is genuinely unindexed today, doctor's job is to report
    it, and whether the walk should exclude on the absolute path at all is a
    separate question about a module this row does not own.

    RESOLVED on both sides before the components are taken, because the
    harness writes to what the path resolves to: a link inside the corpus root
    pointing into `hot/` is a directory the walk never descends into, and its
    own spelling says nothing about that.
    """
    try:
        relative = os.path.relpath(_resolved(directory), _resolved(root))
    except (OSError, ValueError):
        # Only reachable if the pair stopped resolving since `_within` said
        # they did; the answer that claims no retrieval is the one to give.
        return True
    return _excluded(os.path.join(root, relative))


def _store_relation(machine: Machine, directory: str) -> tuple:
    """`(store id, corpus root, how the two overlap, whether a config was read)`.

    THE FOURTH VALUE SEPARATES TWO EMPTY ANSWERS. A config this run could not
    read produces the same first three as a directory that was compared with
    every store and found outside all of them, and the sentence a caller hangs
    off that emptiness — "outside every store, so nothing retrieves what lands
    there" — is a claim about every store on the machine. Made from a file
    nothing opened, it is this row's most alarming answer given for free.

    EVERY configured store, not `searched_stores()`. A store gated to a root
    this session is standing outside of still holds whatever the harness writes
    into it, so gating it out here would answer "outside every store" about a
    directory that is already right — and the answer would change with the
    directory the adopter happened to run doctor from.

    FIVE overlaps rather than in-or-out, because the harness does something
    different in each: `at` and `over` are the two spellings of a configured
    directory that CONTAINS the store's memories, which is what puts them
    inside the harness's own rewrite; `pruned` is inside the corpus root and
    below a name the indexing walk never descends into; `flat` is inside a
    store that has no `search/` yet, so it is retrieved only until one appears;
    `inside` is the one that is retrieved.

    EVERY store is asked, and harm dominates. Returning from the first store
    with any relation at all asked the at/over question per store and never
    across them, so with one store nested in another `memkit.json`'s
    declaration order decided the answer — and the order that read as a PASS
    was the one where the configured directory holds another store's whole
    corpus root.
    """
    cfg = machine.config()
    if cfg is None:
        return "", "", "", False
    # The first store the directory is merely INSIDE, kept rather than
    # returned: a later store whose corpus root this directory holds is the
    # worse state and the one worth reporting.
    within: tuple = ()
    for store in cfg.stores:
        # A store naming a root the config does not define raises here, and
        # `store-roots` is the check that owns saying so.
        with contextlib.suppress(ConfigError, OSError):
            live = cfg.store_dir(store, "live")
            root = _search_root(live)
            # Asked in this order because both tests are true when the two
            # paths are one directory, and that case belongs to the rewrite.
            if _within(root, directory):
                same = _within(directory, root)
                return store.id, root, "at" if same else "over", True
            if _within(directory, root) and not within:
                tiered = os.path.join(live, "search")
                within = (store.id, root, _how_inside(directory, root, tiered))
    return (within or ("", "", "")) + (True,)


def _how_inside(directory: str, root: str, tiered: str) -> str:
    """Which of the three inside-a-corpus-root states `directory` is in.

    `search/` DECIDES CONTAINMENT, not the fallback root: `_search_root`
    answers with the store root while a store has no tier layout, so a
    directory beside the store's memories reads as inside one — and the moment
    a `search/` appears, the same directory is above the corpus root and
    nothing in it is retrieved. That is the state this file's `corpus-root`
    case calls the single most expensive silent one in the field log, and it is
    reached by creating a directory. So the question asked here is the one
    `_nearest_store` already answers when it recommends a value: is this inside
    the corpus root the store WILL have. `tiered` is that root, `<live>/search`
    whether or not it is there yet, and on a store already laid out by tier it
    is `root` itself.
    """
    if _pruned(directory, root):
        return "pruned"
    if not _within(directory, tiered):
        return "flat"
    return "inside"


def _placed(machine: Machine, directory: str) -> tuple:
    """`(retrieved, what to say about it, what a remedy should name)`.

    The middle answer is why this is not a bool. A directory inside a corpus
    root but under a pruned name is in the store and out of retrieval at once;
    a directory that HOLDS the corpus root is retrieved and rewritten at once.
    Containment alone reports both as the state this check exists to reach.
    """
    store, root, how, known = _store_relation(machine, directory)
    if not store:
        return (
            False,
            "is outside every store, so nothing retrieves what lands there"
            if known
            else "cannot be placed: this run did not read a config, so no "
            "store was compared with it",
            "",
        )
    safe = os.path.join(root, harness_memory.SAFE_SUBDIR)
    if how in ("at", "over"):
        where = (
            f"is {store}'s corpus root itself"
            if how == "at"
            else f"holds {store}'s corpus root {_shown(root)}"
        )
        return (
            False,
            f"{where}, so every memory already in that store is one the "
            "harness re-serialises the first time an agent writes or edits "
            "it: name slugified, other keys moved under metadata",
            safe,
        )
    if how == "pruned":
        return (
            False,
            f"is inside {store}'s corpus root {_shown(root)} but under "
            f"a name retrieval prunes ({', '.join(sorted(EXCLUDE_DIRS))}), so "
            f"nothing written there is indexed",
            # A corpus root under a pruned component prunes its own
            # `auto-memory` too, and advice to move there is the state the
            # adopter is already in. Nothing named sends the caller to the
            # store walk instead.
            "" if _pruned(safe, root) else safe,
        )
    if how == "flat":
        # `root` is the STORE root here, which is what `_search_root` answers
        # while there is no `search/` — so the corpus root this names is the
        # one the store gets the moment anything creates it.
        return (
            False,
            f"is inside {store}, which has no search/ yet, and above the "
            f"corpus root {_shown(os.path.join(root, 'search'))} that "
            "store gets the moment one exists: creating it stops anything "
            "there being retrieved",
            os.path.join(root, "search", harness_memory.SAFE_SUBDIR),
        )
    if not os.path.isdir(root):
        # CONTAINMENT IS NOT EXISTENCE: `realpath` resolves a path nothing has
        # created, so a store that is configured and not on disk still contains
        # every directory named under it. Passed on that, this row said what
        # the harness writes is retrieved in the same envelope as the
        # `corpus-root` FAIL saying the store is not there.
        return (
            False,
            f"is inside {store}'s corpus root {_shown(root)}, which is "
            "not on disk — see corpus-root — so nothing retrieves what lands "
            "there",
            safe,
        )
    return (
        True,
        f"is inside {store}'s corpus root {_shown(root)}, so what the "
        f"harness writes there is retrieved",
        "",
    )


def _nearest_store(machine: Machine, directory: str) -> tuple:
    """`(store id, where to point the key, stores declared)` for the store this
    directory is most plausibly meant for.

    A remedy prints ONE value, and with several stores declared the first one
    in the config file is an arbitrary answer to which store was meant. Nearest
    by shared path is a guess as well, and it is the guess an adopter can check
    by reading it; the personal store breaks a tie, being the one STORE.md
    tells them to set once and forget about.
    """
    cfg = machine.config()
    if cfg is None:
        return "", "", 0
    mine = directory.split(os.sep)
    best = ("", "", -1, False)
    for store in cfg.stores:
        # One candidate fewer when a store names a root the config does not
        # define; `store-roots` is the check that reports it.
        with contextlib.suppress(ConfigError, OSError):
            # `<live>/search` rather than `_search_root`'s answer, which falls
            # back to the store root on a store not yet laid out by tier: this
            # value is one an adopter is being told to SET, and set to a store
            # root it puts every memory above the corpus root and out of
            # retrieval.
            root = os.path.join(cfg.store_dir(store, "live"), "search")
            shared = 0
            for ours, theirs in zip(mine, root.split(os.sep)):
                if ours != theirs:
                    break
                shared += 1
            personal = store.role == "personal"
            if (shared, personal) > (best[2], best[3]):
                best = (store.id, os.path.join(root, harness_memory.SAFE_SUBDIR),
                        shared, personal)
    return best[0], best[1], len(cfg.stores)


def _odd_switch(key: str, value, scope: str) -> str:
    """What to say about a switch whose value is neither true nor false.

    Only `false` turns either of these off, so a file carrying `0` or `""` has
    the feature RUNNING while reading, to a person and to a truthiness test
    alike, as though it were off. Quoting the value back is the whole remedy an
    adopter needs. An explicit `null` is not quoted, because the harness reads
    it as the key being absent.
    """
    if not scope or isinstance(value, bool):
        return ""
    return (
        f'"{key}" is {_display_cap(json.dumps(value), 20)} in {scope} settings, '
        "which is not true or false; only false turns it off"
    )


# What a checked-in settings file costs, said once. It is the whole of why a
# value this scope decided is reported rather than acted on, and it is too long
# to repeat in the two remedies that need it.
_CHECKOUT_COST = (
    "A checked-in .claude/settings.json travels with every clone, and the "
    "harness applies it: always in a `claude -p` run, a hook or a subagent, "
    "whatever the folder trust state, and in an interactive session once you "
    "trust the folder"
)

# And what to do about it. NOT "set it in your user settings", which is the
# obvious advice and does nothing: `user` ranks BELOW the checked-in file in
# the measured order, so a value there is masked for as long as that file sets
# the key. `.claude/settings.local.json` outranks it, which is the one edit
# that does not require touching the repository — and being untracked is a
# CONVENTION rather than something git enforces, so the instruction says to
# keep it that way instead of asserting that it already is.
_CHECKOUT_REMEDY = (
    "either take the key out of that file or set it in "
    ".claude/settings.local.json, which outranks it — and keep that file "
    "untracked, which nothing but convention makes it. User settings rank "
    "below both and change nothing while it is set"
)


def _checkout_remedy(decide_what: str, scope: str, config_dir: str) -> str:
    """What to tell an adopter about a value a file in this tree decided.

    ONE FUNCTION for what were two spellings of the same paragraph, because
    the pair had drifted: whether a scope's file travels with a clone is a
    different fact per scope, and a remedy that told the adopter to move the
    key into `.claude/settings.local.json` was nonsense addressed to the run
    where that file is what set it.

    WHICH FILE HOLDS THE KEY is the other per-scope fact, and the same
    paragraph got it wrong for the `user` scope. `$CLAUDE_CONFIG_DIR` pointing
    inside the session's directory moves the trusted scope into this tree
    without `settings.local.json` being involved at all, so the convention to
    check is about a file the adopter would find the key absent from.
    """
    if scope == harness_memory.CHECKOUT_SCOPE:
        return f"{_CHECKOUT_COST}. To decide {decide_what}, {_CHECKOUT_REMEDY}."
    out_of = (
        f"{_shown(config_dir)}/{SETTINGS_NAME}, which is where "
        "$CLAUDE_CONFIG_DIR put the settings this run reads as yours"
        if scope == "user"
        else f"it — and check that {LOCAL_SETTINGS_NAME} is untracked, because "
        "a bare `git add` tracks it and then a clone carries it too"
    )
    return (
        f"That value is in a file in the directory this session stands in "
        f"rather than in your own settings. To decide {decide_what}, take the "
        f"key out of {out_of}."
    )


def _checkout_source(scope: str) -> tuple:
    """`(how to name this scope's file, what it costs)` for a scope the adopter
    does not own.

    WHAT IS ASSERTED IS WHAT IS KNOWN. Only `.claude/settings.json` is the file
    a repository checks in by design, so only that one is described as
    travelling with every clone. `.claude/settings.local.json` is untracked by
    convention alone, and memkit does not measure which: answering it means
    running git inside the session's own checkout, which `cli_init._git_tracked`
    refuses outright — `ls-files` executes `core.fsmonitor` from the config of
    whatever repository the path belongs to, and no `-c` override closes a
    surface a repository can always add a key to. So the file is named, the
    convention is named as one, and the value is reported rather than passed.
    """
    if scope == harness_memory.CHECKOUT_SCOPE:
        return (
            f"a checked-in .claude/{SETTINGS_NAME}",
            "that file is checked into this repository and travels with every "
            "clone",
        )
    if scope == "local":
        return (
            f".claude/{LOCAL_SETTINGS_NAME} in this directory",
            f"{LOCAL_SETTINGS_NAME} is untracked by convention rather than by "
            "anything git enforces, and this file is in the directory this "
            "session stands in",
        )
    # The `user` scope, once `$CLAUDE_CONFIG_DIR` puts it inside the session's
    # own directory: the trusted scope under another name.
    return (
        f"{scope} settings",
        "that file is in the directory this session stands in, so it is not "
        "one your own machine placed",
    )


def _env_switch_note(forced, value: str) -> str:
    """What to say about `$CLAUDE_CODE_DISABLE_AUTO_MEMORY`, or "".

    BOTH DIRECTIONS ARE REPORTED, and neither is passed. Forced on, the row's
    most confident sentence would be false while every settings file agrees
    with it. Forced off, what this process carries is not necessarily what an
    adopter's interactive sessions carry — doctor can read one environment,
    and the claim would be about all of them.
    """
    if forced is None:
        return ""
    spelled = _display_cap(json.dumps(value), 40)
    if forced:
        return (
            f"${harness_memory.DISABLE_ENV} is set to {spelled}, which the "
            "harness reads as an instruction to RUN auto-memory before it "
            "opens a settings file at all: no settings scope turns it off "
            "while that value is in the environment"
        )
    return (
        f"${harness_memory.DISABLE_ENV} is set to {spelled}, which turns "
        "auto-memory off in every process that carries it — doctor reads its "
        "own environment, which need not be the one your sessions run in"
    )


def _env_switch_remedy(forced) -> str:
    """The remedy for a switch an environment variable decided."""
    if forced:
        return (
            f"Unset ${harness_memory.DISABLE_ENV} wherever it is exported — a "
            "shell profile, a direnv file in this checkout, a wrapper script — "
            f'or set it to 1. While it holds that value, "'
            f'{harness_memory.ENABLED_KEY}": false changes nothing.'
        )
    return (
        f"That switch is off only for processes that inherit "
        f"${harness_memory.DISABLE_ENV}. To make it true of every session, set "
        f'"{harness_memory.ENABLED_KEY}": false in your own settings as well.'
    )


def _override_note(names: tuple) -> str:
    """What to say about an environment override of the DIRECTORY, or "".

    Named rather than resolved: each of these needs a resolver of its own, and
    re-deriving three more harness surfaces is how this row comes to name a
    directory with confidence and be wrong. What it stops is the PASS.
    """
    if not names:
        return ""
    return (
        f"an environment override is in effect ({', '.join(names)}), so where "
        "the harness writes is not what any settings file says and doctor "
        "does not resolve it"
    )


def _adopter_owns(scopes: list, name: str) -> bool:
    """Whether the scope that decided a key is one the ADOPTER owns.

    The predicate `settings_scopes` already computes, asked here rather than
    re-derived: a comparison against one scope NAME left `local` — a file in
    the session's own directory, outranking the one memkit does flag — passing
    as though the adopter had written it.

    An undeclared key has no scope and no owner to doubt.
    """
    for scope in scopes:
        if scope.scope == name:
            return scope.adopter_owned
    return True


def _declared_below(scopes: list, key: str, name: str, value) -> str:
    """The highest scope BELOW `name` declaring `key` differently, or "".

    The only case a precedence order decides anything. Compared as JSON text
    rather than with `==`, because `False == 0` in Python and the harness reads
    those two as opposite answers.
    """
    if name not in harness_memory.SCOPE_ORDER:
        return ""
    by_name = {scope.scope: scope for scope in scopes}
    below = harness_memory.SCOPE_ORDER[harness_memory.SCOPE_ORDER.index(name) + 1 :]
    for lower in below:
        scope = by_name.get(lower)
        if scope is None:
            continue
        other = scope.data.get(key)
        if other is not None and json.dumps(other) != json.dumps(value):
            return lower
    return ""


def _default_memory_dir(machine: Machine, config_dir: str) -> tuple:
    """`(where the harness writes for this project, why it cannot be said)`.

    Exactly one of the two is ever non-empty. Both refusals are the same kind
    of fact — a path this row cannot derive — and neither is an error: a
    session whose directory was removed under it and a project key too long
    for the harness's own limit are states an adopter can be in.
    """
    if not machine.cwd:
        return "", (
            "the session directory was removed underneath this run, so where "
            "the harness writes for this project cannot be derived"
        )
    try:
        return harness_memory.default_dir(config_dir, machine.cwd), ""
    except ValueError as exc:
        return "", (
            "where the harness writes for this project is unknown: "
            f"{_display_cap(str(exc), PATH_SHOWN + 160)}"
        )


def _consolidation_recency(default: str) -> str:
    """What the consolidation lock says about the last background run, or "".

    TWO SPELLINGS OF ONE LOCK, in the order the harness writes them: beside the
    memory directory and inside it. The first hit answers — a stale lock in one
    place and a fresh one in the other is one run, reported once.
    """
    if not default:
        return ""
    for candidate in (
        os.path.join(os.path.dirname(default), CONSOLIDATE_LOCK),
        os.path.join(default, CONSOLIDATE_LOCK),
    ):
        with contextlib.suppress(OSError):
            age = int(time.time() - os.stat(candidate).st_mtime)
            if age < 0:
                # Clocks disagree — a lock written on another machine, or
                # before a time step. An elapsed time is the one thing this
                # cannot report, and a negative one reads as a typo.
                return "the consolidation lock is dated in the future"
            if age < CONSOLIDATE_RECENT:
                return f"a consolidation ran {age}s ago"
            return f"last consolidation {age // 3600}h ago"
    return ""


def _inventoried(machine: Machine, config_dir: str) -> tuple:
    """`(retrieved, in a store, outside, what the walk could not read, where)`.

    THE RELATION IS ASKED OF EVERY DIRECTORY, not only of the linked ones.
    Being reached through a link is how a project directory comes to be
    somewhere other than under the config directory, but it is not the only
    way: `$CLAUDE_CONFIG_DIR` can name a directory inside a corpus root, and
    then every project directory under it is in a store while nothing here had
    asked. The count that said otherwise is this row's loudest number.

    THREE BUCKETS, because `inside` is the only relation that is retrieved and
    "outside every store" is a sentence about the other four being false. A
    directory at or above a corpus root, or under a pruned name, is in the
    store and unreached from where it is — which is neither of the two answers
    the caller used to have.

    THE FOURTH VALUE IS A SENTENCE, not a flag, because what a caller owes an
    enumeration failure is the same thing every other branch owes: saying what
    it could not read. Empty when the walk succeeded, and every count beside
    it is then a count of what is there rather than of what could be listed.

    THE FIFTH IS THE PATH THAT SENTENCE NAMES, carried out rather than
    re-derived: the remedy has to name the same directory the detail does, and
    the walk fails on a project directory or a `memory/` as readily as on
    `projects/`.
    """
    retrieved: list = []
    held: list = []
    outside: list = []
    found, read_ok, unreadable = harness_memory.inventory(config_dir)
    for project in found:
        how = _store_relation(machine, project.path)[2]
        bucket = retrieved if how == "inside" else outside if not how else held
        bucket.append(project)
    unread = (
        ""
        if read_ok
        else (
            "what the harness has already written cannot be counted: "
            f"{_shown(unreadable)} could not be read"
        )
    )
    return retrieved, held, outside, unread, unreadable


def _unreadable_remedy(unreadable: str) -> str:
    """The repair for the directory this process could not enumerate.

    THE PATH THE WALK STOPPED ON, not `projects/`: a single project directory
    or a single `memory/` fails the walk too, and naming `projects/` over
    either of those sends the adopter to a directory that is already readable.

    NEVER "move them" and never "switch it off": both are advice about
    memories whose number this run does not know, and the only honest first
    step is making the directory answer.
    """
    return (
        f"Make {_shown(unreadable)} readable, then run this again — until it "
        "lists, nothing here can say how much the harness has already written "
        "or where it went."
    )


# The repair for a `$CLAUDE_CONFIG_DIR` that is not absolute. Unsetting it
# first, because the harness's own default is what almost every adopter wants
# and a relative value is far likelier to be an accident than a choice.
_UNROOTED_REMEDY = (
    "Unset $CLAUDE_CONFIG_DIR, or give it an absolute path — a relative one "
    "names a different directory from every directory you start a session in, "
    "and nothing here can say which of them the harness used."
)


def _already_placed(retrieved: list, held: list) -> str:
    """What to say about project directories a store already holds, or "".

    `linked` DECIDES THE WORDING AND NOTHING ELSE: an adopter who wired a
    directory into a store with a symlink is told it is wired, and one whose
    config directory simply sits inside a corpus root is told the directory is
    in there — the same relation, reached two ways, and telling the second one
    it is "linked into a store" would send them looking for a link nobody made.
    """
    wired = [project for project in retrieved if project.linked]
    plain = [project for project in retrieved if not project.linked]
    return _detail(
        f"{_count(len(wired), 'project directory', 'project directories')} "
        f"{'is' if len(wired) == 1 else 'are'} already linked into a store"
        if wired
        else "",
        f"{_count(len(plain), 'project directory', 'project directories')} "
        f"{'is' if len(plain) == 1 else 'are'} already inside a store's corpus "
        "root"
        if plain
        else "",
        f"{_count(len(held), 'project directory', 'project directories')} "
        f"{'is' if len(held) == 1 else 'are'} inside a store and not retrieved "
        "from where it is"
        if held
        else "",
    )


def _left_behind(outside: list) -> str:
    """The count of memories no store holds, or "".

    ITS OWN SENTENCE because the branch that most needed it had no inventory
    at all: "auto-memory is off; memkit is the only memory system here" is
    true in the present tense over a config directory holding nineteen
    memories nothing retrieves, and it is the sentence that stops an adopter
    looking for them.
    """
    if not outside:
        return ""
    return (
        f"{_count(len(outside), 'project directory', 'project directories')} "
        f"{'holds' if len(outside) == 1 else 'hold'} "
        f"{_count(sum(project.memories for project in outside), 'memory', 'memories')} "
        "outside every store"
    )


# Where an adopter is sent for memories that are already written and already
# outside every store. NOT a settings change: the feature being off does not
# move a file, and the section named here is the one that says what moving
# them costs and what carries the coupling.
ADOPT_ADVICE = (
    'moving them is "Where your agent\'s own memories land" in docs/STORE.md'
)


@_produces("auto-memory")
def _auto_memory(machine: Machine) -> list[Check]:
    """The harness's own memory feature, qualified by what could not be read.

    THREE OF THIS ROW'S FOUR SETTINGS READS decide its status — both switches
    and the directory — and a scope that would not parse answers all three the
    same way a scope that says nothing does. So the wrapper is the row's
    outermost rule rather than a sentence inside one branch of it.

    AND NO BRANCH OF THIS ROW PASSES. Every answer here is assembled from a
    settings walk, an environment this one process inherited, and a directory
    listing — three readings of a machine that a variable set in another
    session, a file this user cannot open, or a harness release read from a
    minified bundle can each make wrong, and the module says so at every one of
    them. A PASS is this report's claim that a question is CLOSED, and no
    branch here can close it: whichever one answers, something it did not read
    could contradict the answer. The narrower promise is the one that holds —
    say what was read and let a reader weigh it — so the settled answer is INFO
    with no remedy, and its detail is word for word what a PASS carried.
    """
    return _with_unparsed(_auto_memory_rows(machine), machine.settings)


def _auto_memory_rows(machine: Machine) -> list[Check]:
    """The harness's own memory feature, running beside memkit's.

    The one differentiator the field survey found unclaimed: none of the six
    competitors handles built-in auto-memory coexistence at all. Two stores
    writing memories about the same work, in two formats, with two retrieval
    paths, is a state an adopter should choose rather than discover.

    What the branches turn on is WHERE the harness writes, because that is what
    decides whether the adopter has one corpus or two: a directory inside a
    store is retrieved and is not a second system at all. Each switch is read
    in the first scope that declares it, independently of the other — they
    answer different questions, and the earlier reading, which took whichever
    key it met first, reported `autoDreamEnabled` while `autoMemoryEnabled`
    sat in the same file deciding whether the feature ran.

    WHO DECIDED is the other axis, and it is asked of every value on this row
    regardless of what the value is. A scope the adopter does not own is
    reported and never passed — a clone that turns the feature ON over an
    adopter who turned it off is the direction that leaves two memory systems
    running on a machine whose owner believes there is one.

    `$CLAUDE_CONFIG_DIR` is one of those scopes when it points inside the
    session's own directory, and the inventory it enumerates is then a count of
    whatever that tree put there.
    """
    enabled, enabled_scope = harness_memory.switch(
        machine.settings, harness_memory.ENABLED_KEY
    )
    # THE ENVIRONMENT DECIDES BEFORE ANY SETTINGS FILE IS OPENED, on both
    # axes — see `harness_memory.DISABLE_ENV` and `OVERRIDE_ENV`. Read from the
    # 2.1.258 code: a falsy word in that variable RUNS the feature and returns
    # before the settings walk, so the state this row calls "memkit is the only
    # memory system here" is one an environment variable can make false while
    # every settings scope agrees with it.
    forced, forced_value = harness_memory.env_switch()
    # ONE ANSWER to "did something outside every settings file decide part of
    # this row", carried into every detail and gating every settled answer in
    # it: a settled answer is a claim about the machine, and neither of these
    # is answerable from the environment this one process happens to have
    # inherited.
    environed = _detail(
        _env_switch_note(forced, forced_value),
        _override_note(harness_memory.overrides()),
    )
    dream, dream_scope = harness_memory.switch(
        machine.settings, harness_memory.DREAM_KEY
    )
    # Asked of the SETTINGS value before the environment replaces it: a scope
    # holding `0` is a file somebody hand-edited meaning `false`, and that is
    # worth saying whichever way the environment then decided.
    odd_enabled = _odd_switch(harness_memory.ENABLED_KEY, enabled, enabled_scope)
    if forced is not None:
        enabled = forced
    config_value = os.environ.get(CONFIG_DIR_ENV) or ""
    config_dir = config_value or os.path.expanduser("~/.claude")
    # THE SAME VARIABLE `settings_scopes` guards, guarded the same way. Read
    # raw it decided three things — the directory printed as the derived
    # default, the inventory whose emptiness settles the row rather than
    # leaving it a remedy, and the file the remedy names — so a repository
    # that redirects it moved this row to its most reassuring answer.
    # ASKED OF THE VALUE, NOT OF THE DERIVED DEFAULT. `~/.claude` is under the
    # cwd whenever a session stands anywhere in home, so the containment test
    # alone told an adopter in their own home directory that a checkout had
    # redirected their harness — and offered them a remedy for a variable
    # their environment does not carry.
    steered = bool(config_value) and _under_cwd(config_dir)
    # AND A RELATIVE ONE NAMES NOTHING. `..`-relative lands outside the
    # session's directory, so the containment guard above says nothing about
    # it, and every path this row prints is then one that resolves only from
    # where this process happens to stand — including the ones in the remedy.
    unrooted = (
        "$CLAUDE_CONFIG_DIR is a relative path, so every directory named here "
        "resolves only from the directory this session stands in: "
        f"{_display_cap(config_value, PATH_SHOWN)}"
        if config_value and not os.path.isabs(config_value)
        else ""
    )
    # THE CLAUSE BEFORE THE PATH, because `_bound` cuts from the end and the
    # path is the part an adopter's own machine decides the length of. Read
    # here rather than two hundred lines down: it says the count below belongs
    # to a directory the tree chose, which is as true of the branch that
    # reports the switch off as of the one that reports where it writes.
    redirected = (
        "$CLAUDE_CONFIG_DIR points inside the directory this session stands "
        "in, so both that path and what is counted in it are this tree's "
        f"choice: {_shown(config_dir)}"
        if steered
        else ""
    )
    default, underived = _default_memory_dir(machine, config_dir)
    configured, where = harness_memory.configured_dir(machine.settings)
    # THE LOCK BESIDE THE DIRECTORY IN USE. Read from the derived one whatever
    # the settings said, this reported the recency of a directory the harness
    # stopped writing to the moment `autoMemoryDirectory` was set — and said
    # nothing about the one it writes to now.
    recent = _consolidation_recency(configured or default)
    # THE SAME QUESTION `_store_relation` ASKS, asked once for the row: every
    # placement sentence below rests on a config this run parsed, and without
    # one "outside every store" and "already inside a store" are both unread.
    store_unknown = (
        ""
        if machine.config() is not None
        else "no config this run could read, so nothing here compared what "
        "the harness writes with a store"
    )

    # `is False` and not falsiness: JSON `0`, `""`, `[]` and `{}` are every one
    # of them a value the harness goes on writing under, and read as off they
    # produce this row's most confident sentence over a machine with two memory
    # systems on it.
    #
    # THE SWITCH'S OWNER, whichever way it was switched. The disclosure used to
    # live inside this branch, so it fired for a clone that turned the feature
    # off and never for one that turned it back on.
    switch_theirs = bool(enabled_scope) and not _adopter_owns(
        machine.settings, enabled_scope
    )
    switch_source, switch_travels = (
        _checkout_source(enabled_scope) if switch_theirs else ("", "")
    )
    # THE INVENTORY BEFORE THE VERDICT, on EVERY branch: a switch that is off
    # stops the harness WRITING, and every memory it wrote before is still on
    # disk, so the branch that returned without a walk is the branch whose
    # sentence stops an adopter looking for them. A directory a store already
    # holds is not a second memory system, and a link into a corpus root is the
    # wiring docs/STORE.md recommends — counted as one, this row alarms about
    # the state it exists to send adopters to.
    in_store, held, outside, unread, unreadable = _inventoried(machine, config_dir)
    # EVERY DISCLOSURE IN ONE TUPLE, and the same tuple gates, details and
    # chooses the remedy. These were three lists that drifted: the gate held
    # two of them, the detail four, and the row went on making its most
    # reassuring claim while an environment variable decided where the harness
    # writes and a checked-in path decided what was counted.
    disclosures = (environed, redirected, unrooted, unread, store_unknown)
    unsure = _detail(*disclosures)
    # ONE REMEDY PER DISCLOSURE, because the gate is wider than the two it was
    # written for. A disclosure with no repair of its own gets none rather than
    # the repair for whichever one happened to be last in the ternary — and
    # `environed` is one of those: nothing here can unset a variable in the
    # environment the adopter's sessions actually run in.
    unsure_remedy = (
        _unreadable_remedy(unreadable)
        if unread
        else _UNROOTED_REMEDY
        if unrooted
        else ""
    )
    # THE ONLY CASE THE SCOPE ORDER DECIDES ANYTHING. With one declaring scope
    # every candidate precedence picks it; with two that disagree, this row's
    # most consequential sentence rests on an order inferred rather than
    # measured, so it is reported instead of passed. Asked before the branches
    # rather than inside one: ordered behind the checkout branch it never fired
    # for the state it matters most in, a clone deciding the switch over a
    # scope that declares it otherwise.
    disputed = _declared_below(
        machine.settings, harness_memory.ENABLED_KEY, enabled_scope, enabled
    )
    if enabled is False:
        left = _left_behind(outside)
        if left:
            left = f"{left}, and {ADOPT_ADVICE}"
        placed = _already_placed(in_store, held)
        # THE CLAIM ONLY WHERE THE INVENTORY BEARS IT OUT. "memkit is the only
        # memory system here" beside a count of memories no store holds is a
        # sentence contradicted two clauses later; off is still off, so this
        # stays settled and says the narrower true thing. A walk that FAILED
        # bears nothing out either way, which is the second condition here.
        # The third: the checkout branch appends "while this checkout says so"
        # to this string, and a clone that can switch the feature back on is
        # not a machine where one memory system is settled.
        off = (
            "auto-memory is off in this process's environment"
            if forced is not None
            else f"auto-memory is off in {enabled_scope} settings"
            + (
                ""
                if left or unsure or switch_theirs
                else "; memkit is the only memory system here"
            )
        )
        if forced is not None:
            # NOT A PASS, and not the settings' answer either: the variable
            # outranks every scope. The scope is still disclosed when the
            # adopter does not own it, because a session started without this
            # variable is one that file decides.
            return [
                Check(
                    "auto-memory",
                    INFO,
                    _detail(
                        off,
                        unsure,
                        left,
                        placed,
                        f'"{harness_memory.ENABLED_KEY}" is also set in '
                        f"{switch_source}, and {switch_travels}"
                        if switch_theirs
                        else "",
                        recent,
                        odd_enabled,
                    ),
                    _env_switch_remedy(forced),
                    actor=USER,
                )
            ]
        if switch_theirs:
            return [
                Check(
                    "auto-memory",
                    INFO,
                    _detail(
                        f"{off} while this checkout says so — the value is in "
                        f"{switch_source}, and {switch_travels}",
                        # BOTH DISCLOSURES, not whichever branch ran first.
                        # Who decided and which file wins are separate
                        # questions, and the clone is the case where the
                        # second one has teeth.
                        f"{disputed} settings declare it otherwise, and which "
                        "file wins is inferred for this key rather than read"
                        if disputed
                        else "",
                        unsure,
                        left,
                        placed,
                        recent,
                    ),
                    _checkout_remedy("it yourself", enabled_scope, config_dir),
                    actor=USER,
                )
            ]
        if disputed:
            return [
                Check(
                    "auto-memory",
                    INFO,
                    _detail(
                        f"auto-memory is off in {enabled_scope} settings and "
                        f"{disputed} settings declare it otherwise. Which file "
                        "wins is read from the harness's directory resolver "
                        "and inferred for this key, which resolves through an "
                        "accessor that reports no scope",
                        unsure,
                        left,
                        placed,
                        recent,
                    ),
                    f"Settle it in one place: take "
                    f'"{harness_memory.ENABLED_KEY}" out of {disputed} '
                    f"settings, or out of {enabled_scope} settings, so no "
                    "order decides it.",
                    actor=USER,
                )
            ]
        if unsure:
            # A settled answer here is a claim about what is on the machine,
            # and the walk that would have borne it out is the one that failed.
            return [
                Check(
                    "auto-memory",
                    INFO,
                    _detail(off, unsure, left, placed, recent),
                    unsure_remedy,
                    actor=USER,
                )
            ]
        return [Check("auto-memory", INFO, _detail(off, left, placed, recent))]

    # The feature is running, and a file this tree carries may be why.
    switched_on = (
        f'"{harness_memory.ENABLED_KEY}" is on in {switch_source}, and '
        f"{switch_travels}"
        if switch_theirs
        else ""
    )
    if configured is not None:
        checkout = not _adopter_owns(machine.settings, where)
        source, travels = (
            _checkout_source(where) if checkout else (f"{where} settings", "")
        )
        retrieved, says, target = _placed(machine, configured)
        # THE KEY, THE FILE AND THE VERDICT BEFORE THE VALUE, which is
        # `_detail`'s own rule applied where it was not: this path is a
        # clone's to choose, and a 560-character one filled the whole budget
        # and cut three verdict sentences off the end. What was left read as a
        # confident answer with no answer in it — a prose-shaped value could
        # render a forged one.
        #
        # TWO PARTS rather than one, because a string is as long as the value
        # inside it: written as a single sentence the verdict sat wherever the
        # value put it, and two ordinary disclosures were enough to push the
        # whole of it past the cut. Split, the sentence goes with the fixed
        # facts and the value goes last, where `_bound` is meant to reach.
        named = (
            f"{harness_memory.DIRECTORY_KEY} in {source} names a directory "
            f"that {says}"
        )
        named_value = f"{harness_memory.DIRECTORY_KEY} is {_shown(configured)}"
        # THE WALK GATES THIS ANSWER TOO. The configured directory being inside
        # a store says nothing about the projects the harness wrote before the
        # key was set, and this branch returned before the inventory was ever
        # asked — so the one state that voids every other branch's answer, a
        # listing that would not answer, was invisible in the only branch that
        # settles silently. The parallel derived-directory guard below already
        # gated on it.
        if (
            retrieved
            and not checkout
            and not switch_theirs
            and not held
            and not outside
            and not unsure
        ):
            return [
                Check(
                    "auto-memory",
                    INFO,
                    _detail(named, recent, odd_enabled, named_value),
                )
            ]
        # WHO DECIDES OUTRANKS WHERE IT POINTS, and the order matters for more
        # than the string: the store walk below exists to name a value in a
        # remedy these two branches do not use, so asking for one was a walk of
        # every configured store whose answer was overwritten two lines later.
        if checkout:
            remedy = _checkout_remedy(
                "where your agent writes", where, config_dir
            )
        elif switch_theirs:
            remedy = _checkout_remedy(
                "whether it runs at all", enabled_scope, config_dir
            )
        elif unsure_remedy:
            # AHEAD OF THE STORE ADVICE, because moving a directory is advice
            # about a corpus this run could not enumerate: the first step is
            # making it answer. On the disclosures that have no repair of their
            # own this is empty and the store advice stands.
            remedy = unsure_remedy
        else:
            aside = ""
            if not target:
                store, target, declared = _nearest_store(machine, configured)
                aside = (
                    f" ({store}'s, of the {declared} stores you have)"
                    if declared > 1 and store
                    else ""
                )
            remedy = (
                f'Point it at a directory of the harness\'s own inside a '
                f'corpus root — "{harness_memory.DIRECTORY_KEY}": '
                f'"{_shown(target or "<store>/search/auto-memory")}"{aside} '
                "— retrieval recurses into it and the rewrite reaches only "
                'what is in it. Or leave it there deliberately: "Where your '
                'agent\'s own memories land" in docs/STORE.md says what each '
                "choice costs."
            )
        return [
            Check(
                "auto-memory",
                INFO,
                _detail(
                    named,
                    travels,
                    *disclosures,
                    switched_on,
                    recent,
                    odd_enabled,
                    named_value,
                ),
                remedy,
                actor=USER,
            )
        ]

    here = False
    if underived:
        first = underived
    elif os.path.isdir(default):
        first = (
            f"the harness writes this project's memories to "
            f"{_shown(default)} (project key from the git root)"
        )
        here, says, _target = _placed(machine, default)
        if here:
            first = f"{first}, and that directory {says}"
    else:
        first = (
            f"the harness would write to {_shown(default)} "
            "(derived from the git root)"
        )

    # A remedy NEVER SENDS THE ADOPTER INTO THE CHECKOUT — and "set it in your
    # user settings" is worse than nothing while a scope above them sets the
    # key, which is the whole of what `_checkout_remedy` exists to say.
    if steered:
        switch_off = (
            'To run memkit alone, set "autoMemoryEnabled": false in the '
            "settings.json under your own ~/.claude, and unset "
            "$CLAUDE_CONFIG_DIR or point it back there."
        )
    else:
        switch_off = (
            'To run memkit alone, set "autoMemoryEnabled": false in '
            f"{_shown(config_dir)}/settings.json."
        )
    # THE TWO FACTS THAT VOID THE ANSWER AHEAD OF THE ONE THE ROW IS ABOUT.
    # `first` ends in a path and carries a second one inside `says`, so the
    # sentence saying this whole count belongs to a directory the tree chose
    # was the one `_bound` cut — on an ordinary machine, with no long value in
    # sight.
    fixed = (
        *disclosures,
        first,
        switched_on,
        recent,
        odd_enabled,
        _odd_switch(harness_memory.DREAM_KEY, dream, dream_scope),
        f"auto-dream is off in {dream_scope}: no background consolidation"
        if dream is False
        else "",
        _already_placed(in_store, held),
    )
    if (
        here
        and not outside
        and not held
        and not unsure
        and not switch_theirs
    ):
        return [Check("auto-memory", INFO, _detail(*fixed))]
    counted = _left_behind(outside)
    listed = ""
    if outside:
        listed = ", ".join(
            f"{project.key} ({project.memories})"
            for project in outside[:INVENTORY_SHOWN]
        )
        if len(outside) > INVENTORY_SHOWN:
            listed += f" + {len(outside) - INVENTORY_SHOWN} more"
    return [
        Check(
            "auto-memory",
            INFO,
            # The KEYS last and their count with the other fixed facts:
            # `_bound` cuts from the end, and a real key is a whole absolute
            # path with its separators replaced, so five of them are longer
            # than everything else in this row put together.
            _detail(*fixed, counted, listed),
            # THE WHOLE REMEDY when a scope above the adopter's turned the
            # feature on, rather than a sentence appended to the store advice:
            # where the harness writes is not the question while somebody else
            # decides whether it writes, and the two paragraphs together do not
            # fit inside this string's own bound.
            _checkout_remedy("whether it runs at all", enabled_scope, config_dir)
            if switch_theirs
            else "Two memory systems on one project is a choice rather than a "
            "fault. To put what the harness writes inside the store, set "
            f'"{harness_memory.DIRECTORY_KEY}" to an {harness_memory.SAFE_SUBDIR}/ '
            'directory under a corpus root — "Where your agent\'s own memories '
            f"land\" in docs/STORE.md has the value and the trap. {switch_off}",
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
# What an adopter with no route is told, in one place because two commands say
# it. LOCATING is not PROVISIONING: `uv python find` answers with a path it can
# already see, so a machine with uv and no 3.12 is told the one-time command
# rather than having an interpreter downloaded for it by a diagnostic.
NO_CHECKER_REMEDY = (
    "Install python 3.12 or newer — `uv python install 3.12` if you have uv, "
    "or your platform's package manager — and re-run. Until then any command "
    "that regenerates a ledger refuses by name and writes nothing: a seeded "
    "memory with no ledger row is a broken store, so half-completing is worse "
    "than not starting."
)
# The hook's floor, which is NOT the checker's. A stock macOS python is 3.9.6
# and every current Linux distribution clears it.
HOOK_FLOOR = (3, 9)



def build_facts() -> tuple:
    """(package version, hook version, payload sha) — the three answers to
    "which build am I on", each None when this install cannot derive it.

    A precondition for reading any other line of this report, and until now no
    command anywhere answered it: a critic filed the absence of `--version` as
    a defect against all four binaries.
    """
    package = _installed_version()
    if package is None:
        # THE PAYLOAD'S OWN MANIFEST, which is the channel the skills run from
        # and the one where `importlib.metadata` can never answer: a plugin
        # install does not pip-install the package, and the marketplace pins by
        # url+sha rather than cloning, so the sha below is empty too. Without
        # this, two of the three facts were unknown on exactly the channel that
        # needs them, with the release number sitting unread in the payload.
        for root in (os.environ.get("CLAUDE_PLUGIN_ROOT", ""),):
            if root:
                package = _manifest_version(root)
    hook_version = None
    with contextlib.suppress(Exception):
        hook_version = _version()
    payload = None
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if root and os.path.isdir(os.path.join(root, ".git")):
        with contextlib.suppress(OSError, subprocess.SubprocessError, Untrusted):
            out = run_git(GitRoute.HEAD_SHA, repo=root, timeout=15)
            if out.returncode == 0:
                payload = out.stdout.strip()[:12]
    return package, hook_version, payload


def _installed_version():
    """The distribution's version, or None where there is no distribution."""
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        return version("memkit")
    return None


def _manifest_version(root: str):
    """The version the plugin manifest declares, or None."""
    with contextlib.suppress(OSError, ValueError):
        path = os.path.join(root, ".claude-plugin", "plugin.json")
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        if isinstance(blob, dict):
            found = blob.get("version")
            if isinstance(found, str) and found:
                return found
    return None


def version_line() -> str:
    """What `memkit --version` prints. One line, three facts, and the ones
    this install cannot derive are named as unknown rather than omitted — a
    missing field reads as a field that does not exist."""
    package, hook_version, payload = build_facts()
    # ONE TOKEN per fact, so the line survives being read by a shell. It used
    # to interpolate `(no installed distribution)` into the version position,
    # which makes `awk '{print $2}'` yield `(no` — a fragment of prose where a
    # caller reads a version. Where a fact is genuinely unknown the token says
    # so, and the provenance goes in a trailing parenthetical.
    return (
        f"memkit {package or 'unknown'} hook:{hook_version or 'unknown'} "
        f"payload:{payload or 'unknown'}"
        + ("" if payload else " (installed from a pinned archive, not a clone)")
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
    """`(CheckerRoute, interpreter path)` for checker-backed work.

    THE PROCESS'S OWN ANSWER, probed once and cached on the machine. Two
    subcommands of one invocation must not pick differently, and a per-process
    cache is that guarantee with no ambient channel to carry it — an
    environment variable that answers this question is an environment variable
    that chooses the code memkit runs.
    """
    if machine._route is None:
        machine._route = _probe_checker_route()
    return machine._route


# The interpreter LOCATOR, not a package fetcher. `uv python find` answers
# with a path and resolves no name from any index; `--no-python-downloads` and
# `UV_PYTHON_DOWNLOADS=never` say the same thing twice, once as a flag and once
# as a declared environment entry, because a uv old enough not to know the flag
# errors rather than downloading.
#
# What runs afterwards is THIS PAYLOAD'S OWN checker — already on disk, because
# a plugin install is a git clone at a pinned sha — so the checker and the hook
# are the same release by construction, rather than two releases that agree
# only while two pins do.
#
# Locating is not provisioning, and that is the cost: on a machine with `uv`
# and no 3.12 anywhere, this refuses and names `uv python install 3.12` rather
# than downloading an interpreter nobody asked for. An implicit interpreter
# download triggered by a diagnostic is a large unconsented side effect.
_UV_FIND = ("python", "find", "--no-python-downloads", "--no-project")


def _probe_checker_route() -> tuple:
    if sys.version_info[:2] >= CHECKER_FLOOR:
        return CheckerRoute.SELF, sys.executable
    for name in ("python3.14", "python3.13", "python3.12", "python3"):
        try:
            found = resolve(name)
        except Untrusted:
            continue
        with contextlib.suppress(
            OSError, subprocess.SubprocessError, ValueError, Untrusted
        ):
            # `-I`: `python -c` puts the session directory on `sys.path`,
            # and `site` imports a `sitecustomize.py` it finds there before
            # the `-c` line runs — so the version probe would be the checkout
            # executing code. Isolated mode drops that entry and the
            # environment with it.
            out = _execute(
                [found, "-I", "-c", "import sys; print(sys.version_info[:2])"],
                timeout=15,
            )
            if out.returncode == 0 and "(3, 1" in out.stdout:
                pair = out.stdout.strip().strip("()").split(",")
                if (int(pair[0]), int(pair[1])) >= CHECKER_FLOOR:
                    return CheckerRoute.LOCAL, found
    with contextlib.suppress(OSError, subprocess.SubprocessError, Untrusted):
        out = _execute(
            [
                resolve("uv"),
                *_UV_FIND,
                f"{CHECKER_FLOOR[0]}.{CHECKER_FLOOR[1]}",
            ],
            timeout=30,
            env_extra={"UV_PYTHON_DOWNLOADS": "never"},
        )
        located = out.stdout.strip()
        if out.returncode == 0 and located:
            require_executable(located)
            return CheckerRoute.UV_MANAGED, located
    return CheckerRoute.NONE, ""


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
    route, interpreter = _checker_route(machine)
    recorded = _recorded_interpreter(machine)
    honoured = ""
    if recorded:
        # `expand_home` and `path_refusal`, in the wrapper's own order: this
        # field names the binary exec'd on every prompt, and the wrapper vets
        # its SHAPE before it ever asks whether the file is executable. Asking
        # only the second question reported `/proc/self/exe` as honoured while
        # the wrapper refused it, and `~someone/python3` as "not an executable
        # file" — true, and the wrong repair.
        expanded = expand_home(recorded)
        shape = path_refusal(expanded)
        if shape:
            honoured = (
                f'. The config records "interpreter": "{recorded}", which '
                f"{shape}, so the wrapper refuses it by name and falls back "
                "to the python3 on PATH"
            )
        elif not (os.path.isfile(expanded) and os.access(expanded, os.X_OK)):
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
    floor = f"{CHECKER_FLOOR[0]}.{CHECKER_FLOOR[1]}"
    if route is CheckerRoute.NONE:
        return [
            Check(
                "interpreter",
                FAIL,
                f"hook interpreter {running}; NO checker route: no python on "
                f"this machine meets {floor}, and `uv` located none "
                f"either{honoured}",
                NO_CHECKER_REMEDY,
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
    # `_display_path` on the binary, like every other path this report prints:
    # the detail is pasted into issues, and an absolute interpreter path under
    # `/Users/<name>` or `/home/<name>` carries the username while every
    # neighbouring line has been shortened.
    command = checker_argv(route, interpreter)
    where = " ".join([_display_path(command[0]), *command[1:]])
    if route is CheckerRoute.UV_MANAGED:
        return [
            Check(
                "interpreter",
                INFO,
                f"hook interpreter {running}; checker route {route.value} "
                f"({where}), because no python on PATH meets {floor} and `uv` "
                f"located one. Retrieval is unaffected{honoured}",
            )
        ]
    if honoured:
        return [
            Check(
                "interpreter",
                INFO,
                f"hook interpreter {running}; checker route {route.value} "
                f"({where}){honoured}",
            )
        ]
    return [
        Check(
            "interpreter",
            PASS,
            f"hook interpreter {running}; checker route {route.value} ({where})",
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
        # The nearest EXISTING ancestor, not the immediate parent. On a fresh
        # macOS account `~/.cache` does not exist — the platform uses
        # `~/Library/Caches` — and `os.access` returns False for a path that is
        # not there, so the guard could not tell a missing parent from a
        # read-only one and told the adopter their cache was unwritable. The
        # hook calls `makedirs`, so a missing parent is created and the
        # fallback is never reached.
        parent = os.path.dirname(path)
        while parent and not os.path.isdir(parent):
            nxt = os.path.dirname(parent)
            if nxt == parent:
                break
            parent = nxt
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
        ". This doctor run appended one soak record here, synced each store's "
        "index, and may have run the sweep"
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
    goes = []
    if machine.resolved_config:
        # A config on rung 2 lives IN the plugin data directory, so it is one
        # of the things `uninstall` takes. That is the right lifetime for a
        # file init can regenerate, and it is exactly the sort of thing an
        # adopter should be told before they run the command rather than
        # after.
        where = survives
        if machine.rung_two and machine.resolved_config == machine.rung_two:
            where = goes
        where.append(_display_path(machine.resolved_config))
    survives.append(f"{_display_path(machine.state_dir)} (index, log, journal)")
    taken = (
        "; ".join(goes) + " goes with the plugin data directory unless you "
        "pass --keep-data — init regenerates it. "
        if goes
        else ""
    )
    return [
        Check(
            "uninstall-story",
            INFO,
            "`claude plugin uninstall memkit@memkit` removes the payload and "
            "the plugin data directory; `--keep-data` keeps the second. "
            + taken
            + "Neither touches: " + "; ".join(survives),
            "The canary memories above are memkit's own and are safe to "
            "delete by hand; everything else in your stores is yours. Nothing "
            "removes them for you, because the store is deliberately outside "
            "every plugin-managed path.",
            actor=USER,
        )
    ]


# How much of the wrappers' error log to quote. Enough to show a repeat and
# short enough that one broken install does not fill the report.
ERRLOG_TAIL = 6


@_produces("hook-errors")
def _hook_errors(machine: Machine) -> list[Check]:
    """Say where the hook's stderr went.

    The refusals in `bin/lib/common.sh` are among the clearest text this
    project contains and in the product they are unreachable: Claude Code
    swallows hook stderr, and `claude --debug -p` showed zero hook lines in
    three separate attempts across two walkthroughs. Without this check
    doctor's best remedy for a whole class of failures is still "there is a
    message you cannot see".

    An empty log is a PASS and an absent one is a PASS for the same reason: the
    wrappers only write here when they refuse something, so nothing to say is
    the healthy state. The file is written only when the state directory
    already exists, so its absence on a never-configured install is expected
    rather than evidence.
    """
    path = os.path.join(machine.state_dir, ERRLOG_NAME)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
    except FileNotFoundError:
        return [
            Check(
                "hook-errors",
                PASS,
                f"no {ERRLOG_NAME}: the wrappers have refused nothing here, or "
                "the state directory did not exist when they tried",
            )
        ]
    except OSError as exc:
        return [
            Check(
                "hook-errors",
                UNKNOWN,
                f"{_display_path(path)} could not be read ({exc})",
            )
        ]
    if not lines:
        return [Check("hook-errors", PASS, f"{ERRLOG_NAME} is empty")]
    when = ""
    with contextlib.suppress(OSError):
        when = " last written " + time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(os.stat(path).st_mtime)
        )
    tail = " | ".join(lines[-ERRLOG_TAIL:])
    return [
        Check(
            "hook-errors",
            INFO,
            f"{len(lines)} line(s) in {ERRLOG_NAME}{when}. Last: {tail}",
            "These are the messages the harness swallowed. Each one names what "
            "the wrapper refused and why; the config-route and interpreter "
            "checks in this report are the two that usually explain them.",
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
                    "This is a defect in memkit, not in your setup, and "
                    "nothing you or an agent can do here changes it. The other "
                    "checks in this report still stand.",
                    actor=USER,
                    terminal=True,
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


def run(args: argparse.Namespace) -> int:
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
    try:
        machine = Machine(getattr(args, "config", None))
    except Exception as exc:  # noqa: BLE001 - the alternative is a traceback
        # `collect()` wraps every producer, and this line sat above it. The
        # docstring there says a traceback in place of the other twenty
        # answers is the worst thing this command can do, and a `Machine`
        # that cannot be built is exactly when that was still reachable.
        print(
            envelope(
                [
                    Check(
                        "install",
                        UNKNOWN,
                        f"nothing could be read about this machine: "
                        f"{type(exc).__name__}: {exc}",
                        "Run this from a directory that exists — the one this "
                        "session stands in may have been removed under it — "
                        "and if it does exist, report this with the message "
                        "above.",
                        actor=USER,
                    )
                ]
            )["report"]
        )
        return EXIT_PROBLEMS
    checks = collect(machine, wanted)
    blob = envelope(checks)
    if getattr(args, "as_json", False):
        print(json.dumps(blob, indent=2, sort_keys=False))
    else:
        print(blob["report"])
    return EXIT_OK if blob["verdict"] == "OK" else EXIT_PROBLEMS
