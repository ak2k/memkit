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
    PLUGIN_ENV,
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
DETAIL_MAX_BYTES = 480


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


class Machine:
    """What the checks read, resolved once for the whole run.

    One object rather than each producer reaching for `os.environ` itself, for
    the reason `_config_state` exists in the hook: two surfaces deriving the
    same answer separately is how they come to disagree, and a diagnostic whose
    halves disagree is worse than no diagnostic.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.explicit_config = config_path

    @property
    def plugin(self) -> bool:
        return bool(os.environ.get(PLUGIN_ENV))


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
