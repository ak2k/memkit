"""`memkit init` — a consented, journalled, converging setup.

The problem it solves is not "typing a config is tedious". It is that the four
manual steps the README taught could not be followed correctly by the people
who wrote them: three of four reviewers put their first memory where nothing
would ever retrieve it, the quick start's third step broke on paste because an
unconfigured install deliberately creates no state directory, and the one
worked example in the docs could not produce the output beside it.

So this writes the whole thing, and the shape is a two-turn handshake rather
than a command that acts:

    memkit init --dry-run            a manifest of every path and every write,
                                     plus a digest. ZERO writes.
    memkit init --confirm <digest>   recompute, refuse if anything moved,
                                     re-emit the manifest, then apply.

THE DIGEST BINDS THE TARGET STATE, not the human's view of it. Every action
records what is at its path NOW; a file that appeared, changed or vanished
between the two calls changes the digest and the second call refuses. "Relay
this verbatim" is an instruction to a model and not a control, which is why
`--confirm` puts the applied text into the transcript itself rather than
trusting that the first text was ever shown.

REFUSALS WRITE NOTHING. Not "clean up on failure" — nothing is written until
every refusal has been checked, because a half-made store is worse than none:
a seeded memory with no ledger row is a store the checker calls broken, and an
adopter who ran a setup command has no reason to look for one.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import unicodedata

from memkit import harness_memory
from memkit._exec import (
    CheckerRoute,
    GitRoute,
    Untrusted,
    _execute,
    _under_cwd,
    checker_argv,
    run_git,
)
from memkit.cli_doctor import (
    CANARY_NAME,
    CONFIG_DIR_ENV,
    EXCLUDE_STRAY,
    INTERPRETER_ROUTES,
    NO_CHECKER_REMEDY,
    OPTION_KEY,
    USER,
    Machine,
    _checker_route,
    _store_relation,
    _within,
    authored_configs,
    canary_query,
    fts5_available,
    interpreter_refusal,
)
from memkit.memory_prompt_recall import (
    CONFIG_ENV,
    DEFAULT_SEARCH_CLI,
    EXCLUDE_BASENAMES,
    GENERATED_CONFIG_NAME,
    INIT_JOURNAL_NAME,
    PLUGIN_DATA_ENV,
    PLUGIN_SEARCH_CLI,
    SCHEMA,
    _display_path,
    _plugin_install,
    _utf8,
    append_record,
    claim_holds,
    expand_home,
    journal_config_claims,
    path_refusal,
    state_token,
)

SUMMARY = "create a store and wire this machine up to it"

# The manifest's own version. A consumer that parsed the text — and one will,
# because the skill relays it verbatim into a model's context — needs to know
# when its shape changed.
MANIFEST_SCHEMA = 1

# How long to wait for another init's advisory lock before giving up and
# proceeding unlocked. Long enough to cover a concurrent config merge, which is
# a read, a dict update and a rename; short enough that a caller waiting on
# this command never wonders whether it is wedged.
LOCK_WAIT_SECONDS = 10.0

# What init creates when nothing says otherwise. `~/notes` because that is what
# the README's own worked example has always used, and an adopter who followed
# it once should find init converging on the same directory rather than
# creating a second store beside it.
DEFAULT_STORE = "~/notes"
# The config, when neither `--config` nor the install option names one, OFF the
# plugin channel. `--config` and `$MEMKIT_CONFIG` are the two routes pip and
# nix read and both take a path the adopter names, so a file under `~/.config`
# is one they can point either route at.
#
# It is NOT the plugin channel's answer. The wrapper reads exactly two rungs
# and this is neither, so a config written here on that channel is a config the
# hook can never read — see `_resolve_config`.
DEFAULT_CONFIG = "~/.config/memkit/memkit.json"

# The operations a plan can hold. Each is one filesystem effect, journalled at
# the moment it happens.
CREATE_DIR = "create-dir"
CREATE_FILE = "create-file"
REWRITE_FILE = "rewrite-file"
APPEND_LINE = "append-line"
MERGE_CONFIG = "merge-config"
SETTINGS_WRITE = "settings-write"
VERIFY = "verify"


def _sha(data: str) -> str:
    """Every digest this command takes, through the hook's total encoding.

    The one chokepoint, and it has to be total because what flows through it
    is outside text: a config path off argv, which `sys.argv` decodes with
    `surrogateescape`, and file content and store paths that reach it the same
    way. A strict encode raises `UnicodeEncodeError` on any of them, and
    `--dry-run` takes this digest before it prints the manifest — so the
    command that exists to show an adopter what it would write instead shows
    them a traceback out of a hashing helper, having written nothing and said
    nothing about why.
    """
    return hashlib.sha256(_utf8(data)).hexdigest()


# What is at a path now, as one comparable token: `absent`, `dir`, or the
# content hash of a file. This is the half of the digest that makes it bind to
# the TREE rather than to the request, and it is the hook's function rather
# than a second copy here — the journal records these tokens and the hook is
# one of the two things that reads them back.
_state_of = state_token


class Action:
    """One filesystem effect, and everything needed to describe or journal it."""

    __slots__ = (
        "op",
        "path",
        "before",
        "content",
        "note",
        "authored_config",
        "payload",
        "group",
        "confine",
    )

    def __init__(
        self,
        op: str,
        path: str,
        content: str = "",
        note: str = "",
        authored_config: bool = False,
        payload: object = None,
        group: str = "",
        confine: str = "",
    ) -> None:
        self.op = op
        self.path = path
        self.before = _state_of(path)
        self.content = content
        self.note = note
        self.authored_config = authored_config
        # Which SUMMARY LINE the manifest folds this action into, or "" for an
        # action the reader sees a line of its own for. Adoption copies one
        # harness directory at a time and there can be hundreds of them; a
        # manifest that listed every file would bury the writes that are not
        # copies. It is outside `key()` deliberately: what the digest binds is
        # the effect, and how the effect is printed is not part of it.
        self.group = group
        # THE ROOT THIS EFFECT HAS TO LAND INSIDE, or "" for one that answers
        # to no root. A copy is planned against a path; what a write reaches is
        # that path with every symlink in it followed, and the two are the same
        # thing only until somebody plants a link. Outside `key()` for the same
        # reason `group` is: it constrains the effect rather than being one, so
        # a plan that gained it is not a different request.
        self.confine = confine
        # What a MERGE_CONFIG action re-derives its content from at apply time.
        # `content` is the merge as it would land against the tree the plan was
        # built over; `payload` is what has to be merged in whatever the tree
        # holds when the lock is finally taken.
        self.payload = payload

    @property
    def after(self) -> str:
        if self.op == CREATE_DIR:
            return "dir"
        if self.op == VERIFY:
            return self.before
        return "file:" + _sha(self.content)

    @property
    def redundant(self) -> bool:
        """True when this action has already happened.

        The convergence rule in one property: a double init produces an empty
        manifest because every action finds the state it would have made.
        """
        return self.op != VERIFY and self.before == self.after

    def key(self) -> str:
        return "\0".join((self.op, self.path, self.before, self.after))


class Refusal(Exception):
    """A named reason init will not proceed. Nothing has been written."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name
        self.message = message


class _RecordNotWritten(Exception):
    """The mutation happened and the record of it did not.

    Its own type, and deliberately not an `OSError`: one handler around the
    write and the record after it named the write as the thing that failed
    while carrying the journal's errno in the parenthesis, about a file already
    on disk with the right bytes. The two failures have different recoveries
    and cannot share a sentence.
    """

    def __init__(self, exc: OSError) -> None:
        super().__init__(f"{type(exc).__name__}: {exc}")


class Plan:
    """Everything init would do, in the order it would do it."""

    def __init__(self, actions: list, notes: list, store: str = "") -> None:
        self.actions = actions
        self.notes = notes
        # The store this plan was built against, carried rather than re-derived
        # by whoever reports on the finished run: the default and the flag are
        # resolved in one place and a second reading of them is a second
        # answer. Outside `key()` and outside the digest — it names the request
        # rather than being an effect of it, and every effect that lands inside
        # it is already an action with its own path.
        self.store = store

    @property
    def writes(self) -> list:
        """The actions that would change the filesystem.

        Separate from `pending` because verification is not a write and must
        not make a converged install look like an unconverged one: a second
        init has nothing to do and still has something to check.
        """
        return [a for a in self.actions if a.op != VERIFY and not a.redundant]

    @property
    def pending(self) -> list:
        """What a confirm would actually perform, verification included."""
        return [a for a in self.actions if a.op == VERIFY or not a.redundant]

    @property
    def digest(self) -> str:
        """A hash of the whole plan INCLUDING the redundant actions.

        Including them is what makes `--confirm` able to tell "already done"
        from "somebody changed this underneath me": a plan whose actions all
        read as redundant is a converged install, and a plan with one action
        missing entirely is a different request.
        """
        return _sha("\n".join(a.key() for a in self.actions))[:16]

    def render(self) -> str:
        """The manifest, as the human is meant to read it.

        Every path, every write, and the two things a path alone does not say:
        where a symlink actually lands, and whether the file is tracked by git —
        an `@-import` line added to a tracked `CLAUDE.md` is a commit somebody
        did not intend to make.
        """
        lines = [
            "memkit init — what this would do",
            "",
        ]
        pending = self.pending
        if not self.writes:
            lines.append("Nothing to write. Every path init would create")
            lines.append("already holds what it would put there, so this")
            lines.append("install is already set up. The check below still runs.")
            lines.append("")
        seen_groups = set()
        for action in pending:
            if action.group:
                # A SUMMARY LINE PER SOURCE DIRECTORY, AND THEN EVERY FILE.
                # The count on its own was the whole consent surface for the
                # writes this command exists to make, on a command whose named
                # harm is a wrong copy — and it hid the one case where a
                # destination is not the path it is spelled as, since the
                # summary carried no room for "resolves to".
                if action.group in seen_groups:
                    continue
                seen_groups.add(action.group)
                members = [a for a in pending if a.group == action.group]
                lines.append(
                    f"  {action.op:<14} {len(members)} "
                    f"{'file' if len(members) == 1 else 'files'} "
                    f"{action.group}"
                )
                for member in members:
                    lines.append(
                        f"                 {_display_path(member.path)}"
                    )
                    if member.note:
                        lines.append(f"                 {member.note}")
                    lines.extend(_path_detail(member))
                continue
            lines.append(f"  {action.op:<14} {_display_path(action.path)}")
            if action.note:
                lines.append(f"                 {action.note}")
            lines.extend(_path_detail(action))
        if self.notes:
            lines.append("")
            lines.extend(self.notes)
        lines.append("")
        lines.append(f"digest: {self.digest}")
        lines.append(
            "To apply exactly this: memkit init --confirm " + self.digest
        )
        # NOT sanitized line by line. `sanitize` collapses runs of whitespace,
        # which is right for a description and wrong for a manifest: the
        # indentation is what makes a list of paths readable, and this text is
        # relayed verbatim into a transcript a person reads. The adopter-
        # controlled part of every line is a PATH, and those go through
        # `_display_path`, which strips what was never visible and leaves the
        # spacing exactly as it was — because a path with two spaces in it is
        # a path, and a collapsed one names nothing.
        return "\n".join(lines)


def _path_detail(action) -> list:
    """The two things a path alone does not say, for one action.

    Shared by both branches of the manifest rather than written out in each:
    the grouped branch grew its own copy of the path line and not of these, and
    a destination that resolves somewhere else is exactly the case a grouped
    line must not be the one to drop.
    """
    out = []
    resolved = _terminal_realpath(action.path)
    if resolved != os.path.abspath(action.path):
        out.append(f"                 -> resolves to {_display_path(resolved)}")
    if action.before != "absent":
        out.append(f"                 (exists: {action.before.split(':')[0]})")
    return out


def _terminal_realpath(path: str) -> str:
    """Where a path really lands, following every symlink in it.

    The manifest shows this whenever it differs, because "write to
    ~/notes/search/" and "write into whatever ~/notes points at" are different
    consents and the second one is the one being asked for.
    """
    return os.path.realpath(path)


# --- the content init writes -------------------------------------------------


def _canary_nonce(config_path: str) -> str:
    """The token doctor searches for, derived rather than random.

    Derived, and that is a decision: a random token would be regenerated on
    every run, so the dry-run's digest and the confirm's would never match and
    a converged install would look like a changed one. What the nonce has to be
    is unlikely to appear in the adopter's own corpus, which a derivation over
    an absolute path satisfies as well as randomness does. It is not a secret;
    nothing is authorised by holding it.

    Keyed on the CONFIG and not on the store, because one config holds one
    `canary_nonce` and may hold several stores: doctor asks one fixed query
    per store, so a per-store token would leave every store but one answering
    nothing for it. Two inits appending different stores to one config derive
    the same nonce and seed canaries that agree.
    """
    return "mkc" + _sha(config_path)[:10]


def _canary_body(nonce: str) -> str:
    query = canary_query(nonce)
    # The description on ONE line, and it is the same string the ledger row
    # carries. A folded YAML scalar would be two sources for one sentence, and
    # the ledger is generated from the frontmatter — so a reader that folded
    # differently would produce a row that reads as drift.
    return f"""---
name: memkit-canary
description: {_canary_description(nonce)}
type: reference
---

{query}

`memkit doctor` runs a fixed query for the token above. When this file comes
back, three things are working at once: the store is on disk and in scope, the
index holds it, and the hook that serves prompts can reach both.

Delete this file whenever you like. Nothing depends on it except the
`canary-retrieval` check, which then reports that this store answers nothing
for the fixed query — which is true, and is the point of a canary.
"""


def _canary_description(nonce: str) -> str:
    """The one line the ledger row carries, kept under the checker's cap.

    The cap is the CHECKER's 155 and not the hook's 157: a memory written to
    the hook's ceiling fails the check, and init must never seed a store that
    its own checker rejects.
    """
    return (
        f"{canary_query(nonce)} — proof retrieval reaches this store. "
        "`memkit doctor` searches for this token; delete it once you have "
        "your own."
    )


def _memory_ledger(store: str) -> str:
    return f"""# {os.path.basename(store) or 'memories'} — hot tier

Memories that load into every session. Keep this file small: the recall hook
never points at `hot/`, because anything in this tier is already in context.

Hand-written. `SEARCH.md` beside it is generated and is not.

## Index
"""


def _search_ledger(store: str) -> str:
    """The preamble a store with no SEARCH.md of its own gets.

    THE ROWS ARE NEVER THIS FUNCTION'S. `_search_ledger_text` keeps only what
    precedes `## Index` and generates every row from the store's own
    frontmatter, so what this stands in for is a file that is not there yet —
    which is what makes a second init on a fresh store find the ledger it
    wrote already correct.
    """
    return f"""# {os.path.basename(store) or 'memories'} — retrieval-only ledger

Generated from each memory's `description:` frontmatter. Never hand-edited —
run the integrity checker with `--write` after adding a memory.

## Index
"""


def _config_entries(*, store: str, store_id: str) -> dict:
    """The root and the store one init contributes to a config.

    Split from the file body because a second init against the same config
    must be able to ADD these to whatever is already there rather than replace
    it. Two adopters' worth of stores in one config is a state the design
    admits, and a setup command that clobbered the first one would be the
    lost update the whole handshake exists to prevent.
    """
    return {
        "root": (store_id, {"kind": "path", "path": store}),
        "store": {
            "id": store_id,
            # Personal, and therefore ungated: a store the adopter reaches
            # from anywhere is the one that makes the first prompt after init
            # produce a pointer. A project store needs a `cwd_gate` and a
            # repository to gate to, and init has neither to guess from.
            "role": "personal",
            "dir": ".",
            "live_root": store_id,
            "sub_indexes": [],
        },
    }


def _merge_config(
    existing: str, *, nonce: str, interpreter: str, entries: dict, where: str = ""
) -> str:
    """`existing` with one store's root and entry added, and nothing removed.

    Every field is set only where it is ABSENT. A second init must not
    retarget the first one's interpreter or renumber its nonce — the nonce in
    particular is what doctor's fixed query is, and changing it would make
    every canary already on disk stop answering.
    """
    blob: dict = {}
    if existing.strip():
        try:
            loaded = json.loads(existing)
        except ValueError as exc:
            raise Refusal(
                "unparseable-config",
                f"the config at {_display_path(where)} does not parse as JSON "
                f"({exc}). init will not replace a file it cannot read, and "
                "exiting 1 with a traceback would tell a caller memkit cannot "
                "run at all when one comma is in the wrong place.",
            ) from exc
        if not isinstance(loaded, dict):
            raise Refusal(
                "unparseable-config",
                f"the config at {_display_path(where)} parses, and its top "
                f"level is a {type(loaded).__name__} rather than an object. "
                "init will not replace it.",
            )
        blob = loaded
    blob.setdefault("schema", SCHEMA)
    blob.setdefault("interpreter", interpreter)
    blob.setdefault(
        "search_cli", PLUGIN_SEARCH_CLI if _plugin_install() else DEFAULT_SEARCH_CLI
    )
    blob.setdefault("canary_nonce", nonce)
    roots = blob.setdefault("roots", {})
    name, spec = entries["root"]
    if isinstance(roots, dict):
        roots.setdefault(name, spec)
    stores = blob.setdefault("stores", [])
    if isinstance(stores, list) and not any(
        isinstance(s, dict) and s.get("id") == entries["store"]["id"] for s in stores
    ):
        stores.append(entries["store"])
    # No `citations` block, ever. It is optional, and an empty one makes the
    # first checker run an adopter does report two warnings about a feature
    # they never opted into.
    # `ensure_ascii=False` for the reason `_settings_with` gives: a store path
    # or an existing value outside ASCII comes back as an escape that carries
    # the same value and reads as a different file.
    return json.dumps(blob, indent=2, ensure_ascii=False) + "\n"


# --- the refusals ------------------------------------------------------------
#
# Every one of these is checked BEFORE the first byte is written, and that is
# the whole design rather than an implementation detail. "Clean up on failure"
# leaves a window where a crash between two mutations produces a half-made
# store, and a half-made store is worse than none: a seeded memory with no
# ledger row is a store the checker calls broken, and an adopter who just ran a
# setup command has no reason to go looking for one.
#
# Each refusal is NAMED. The name is the half a caller can branch on and the
# sentence is the half a person can act on — an agent given only prose parses
# it, and an agent given only a token relays a token.


def _refuse_path(what: str, path: str) -> None:
    """The wrapper's own admission rule, applied to what init would WRITE.

    `path_refusal` is the one spelling of `memkit_path_refusal`, and this is
    the whole reason it is one: the shell decides what the hook will read and
    init decides what to write, so a rule they hold separately is a rule they
    will hold differently. When that happened, init wrote a config the wrapper
    refused, and the adopter got a store, a clean integrity check, exit 0 and
    silence on every prompt — with the manifest saying the config would be
    read via the option.

    Two names rather than one, because the two repairs differ: a relative path
    is a path the adopter has to make absolute, and a non-canonical one is a
    path whose meaning depends on who resolves it.
    """
    why = path_refusal(path)
    if not why:
        return
    if why == "is not an absolute path":
        raise Refusal(
            "relative-path",
            f"the {what} path {path!r} is not absolute. A relative path names "
            "a different directory in every session, and the one thing a "
            "memory store may not be is a different store per directory.",
        )
    raise Refusal(
        "non-canonical-path",
        f"the {what} path {path!r} {why}. `bin/lib/common.sh` refuses that "
        "shape before the hook starts, so a config written there is one the "
        "hook can never read — a store, a clean integrity check, exit 0, and "
        "silence on every prompt.",
    )


def _inside(path: str, root: str) -> bool:
    """Whether `path` lands inside `root`, following symlinks on both.

    Terminal realpath on both sides, because the interesting cases are the
    ones a prefix test misses: a store that IS a symlink into plugin data, and
    a `CLAUDE.md` symlinked into a store.
    """
    if not root:
        return False
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    return real_path == real_root or real_path.startswith(real_root + os.sep)


def _writable_ancestor(path: str) -> str:
    """The nearest existing directory above `path`, or "" if none is found."""
    current = os.path.dirname(os.path.abspath(path))
    seen = set()
    while current and current not in seen:
        if os.path.isdir(current):
            return current
        seen.add(current)
        current = os.path.dirname(current)
    return ""


def _refuse_unwritable(what: str, path: str) -> None:
    """Fail before the first byte, not halfway through.

    The check is on the terminal realpath's nearest existing ancestor, because
    that is what the write will actually go through — a symlink into a
    read-only tree is writable by every test that looks at the link.
    """
    target = os.path.realpath(path)
    if os.path.exists(target):
        if not os.access(target, os.W_OK):
            raise Refusal(
                "not-writable",
                f"the {what} {_display_path(path)} exists and this process "
                "cannot write to it.",
            )
        return
    parent = _writable_ancestor(target)
    if not parent or not os.access(parent, os.W_OK):
        raise Refusal(
            "not-writable",
            f"the {what} {_display_path(path)} cannot be created: "
            f"{_display_path(parent or os.path.dirname(target))} is not "
            "writable by this process.",
        )


def _foreign_canary(store: str, nonce: str) -> str:
    """The nonce an existing canary in this store carries, if it is not ours.

    "" when there is no canary, or when it is already ours. The nonce is keyed
    on the CONFIG so that one fixed query answers for every store that config
    names; the cost is that two configs over one store disagree about it, and
    rewriting the file would silently take the first config's `canary-retrieval`
    check away.
    """
    path = os.path.join(store, "search", CANARY_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            body = f.read(4096)
    except (OSError, ValueError):
        # swallow: nothing discloses this one. A canary this process cannot
        # read reads as "no other config owns this store", so
        # `canary-belongs-to-another-config` does not fire and no line says
        # why; the write half is still safe, because the create refuses a
        # differing destination.
        return ""
    found = re.search(r"\bmkc[0-9a-f]{10}\b", body)
    if not found or found.group(0) == nonce:
        return ""
    return found.group(0)


# The suffix `_write_atomically` gives its temporary. A pid rather than a
# random token, so a leftover says which process died holding it — and so this
# recognises one without having to guess at arbitrary names.
_STRANDED_TEMPORARY = re.compile(r"\.\d+\.tmp$")


def _stranded_temporaries(store: str) -> list:
    """Every `<name>.<pid>.tmp` under the store, said out loud and left alone.

    A kill between a write's temporary and its rename leaves a COMPLETE copy of
    the content at that name, mode 0600, inside the store — and nothing names it
    afterwards: no manifest line, no ledger row, and neither the integrity
    checker nor doctor looks for anything that is not `.md`. "Bytes in the store
    that no ledger row names" is what this command promises not to leave, so the
    promise is kept by saying where they are.

    SAYING, NOT SWEEPING. A removal is a write: it would have to enter the
    manifest and the digest the adopter approves, and destroying inside somebody
    else's store is the one thing adoption never does. Whoever reads the line
    can see what the bytes are before deciding.
    """
    found = []
    for here, dirs, names in os.walk(store):
        dirs.sort()
        for name in sorted(names):
            if _STRANDED_TEMPORARY.search(name):
                found.append(os.path.join(here, name))
    if not found:
        return []
    shown = found[:3]
    lines = [
        f"{len(found)} stranded temporary "
        f"{'file' if len(found) == 1 else 'files'} under "
        f"{_display_path(store)}: a write interrupted between its temporary and "
        "its rename leaves the content it was about to write at "
        "`<name>.<pid>.tmp`, and nothing else names it — no manifest line, no "
        "ledger row, and neither the integrity checker nor doctor looks past "
        "`.md`. init leaves them where they are, because deleting inside your "
        "store is not something a setup command does. Read them and remove "
        "them yourself."
    ]
    lines += [f"  stranded: {_display_path(path)}" for path in shown]
    if len(found) > len(shown):
        lines.append(f"  stranded: and {len(found) - len(shown)} more.")
    return lines


def _stray_markdown(store: str) -> list:
    """Markdown at a store's root that a `search/` would strand."""
    out = []
    try:
        names = sorted(os.listdir(store))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".md"):
            continue
        if name in EXCLUDE_BASENAMES or name in EXCLUDE_STRAY:
            continue
        if os.path.isfile(os.path.join(store, name)):
            out.append(name)
    return out


def check_refusals(
    machine: Machine,
    *,
    config_path: str,
    store_path: str,
    wire_claude_md: bool,
    auto_dream_off: bool,
    adopt_auto_memory: bool,
    auto_memory_off: bool,
    interpreter: str | None = None,
) -> None:
    """Every reason init will not proceed, in the order they are cheapest to
    answer and most terminal to meet."""
    if sys.platform.startswith("win") or sys.platform == "cygwin":
        raise Refusal(
            "windows",
            "memkit is not supported on Windows. The wrappers are POSIX sh "
            "and the paths are POSIX paths; there is no configuration that "
            "makes this work, and an obscure failure later would be worse "
            "than this sentence now.",
        )
    # THE FIELD THIS COMMAND WRITES IS THE ONE THE WRAPPER DOES NOT PROBE, so
    # everything that makes it trustworthy has to happen here. Three questions
    # in the order they can be answered: is it a path this build would act on,
    # is it an executable file, and can it serve — the last of which needs the
    # binary started, because a python below the floor and one whose sqlite3
    # has no FTS5 are both ordinary executable files.
    if interpreter:
        named = expand_home(interpreter)
        why = path_refusal(named)
        if not why and not (
            os.path.isfile(named) and os.access(named, os.X_OK)
        ):
            why = "is not an executable file"
        if not why:
            why = interpreter_refusal(named)
        if why:
            raise Refusal(
                "interpreter-unusable",
                f"--interpreter {interpreter} {why}. This value is written "
                'into the config as "interpreter" and exec\'d on every prompt '
                "without being probed again, so a value that cannot serve is "
                "an install that answers nothing and reports nothing. Nothing "
                "was written.",
            )
    else:
        resolved = _interpreter()
        if not (os.path.isfile(resolved) and os.access(resolved, os.X_OK)):
            raise Refusal(
                "no-interpreter",
                f"no usable interpreter resolved ({resolved!r}). The config "
                "init writes records the python that will read every prompt, "
                "and recording one that cannot run is an install that answers "
                "nothing.",
            )
        # The FLOOR needs no asking — this process imported the package, so it
        # clears it. FTS5 does: it is a property of the sqlite this python was
        # built against, and a build without it runs everything here and
        # answers every search with a failure.
        if not fts5_available():
            raise Refusal(
                "interpreter-unusable",
                f"{_display_path(resolved)} is the python running this "
                "command, and its sqlite3 has no FTS5 — which is the table the "
                "whole index is. Recording it would give you a store, a green "
                "integrity check, and nothing back on any prompt.\n"
                + INTERPRETER_ROUTES,
            )
    _refuse_path("config", config_path)
    _refuse_path("store", store_path)

    data_dir = os.environ.get(PLUGIN_DATA_ENV, "")
    if data_dir and os.path.isabs(data_dir) and _inside(store_path, data_dir):
        raise Refusal(
            "store-in-plugin-data",
            f"{_display_path(store_path)} is inside the plugin's data "
            "directory, which `claude plugin uninstall` removes unless you "
            "remember `--keep-data`. A memory store must outlive the plugin "
            "that reads it.",
        )
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if plugin_root and _inside(store_path, plugin_root):
        raise Refusal(
            "store-in-plugin-root",
            f"{_display_path(store_path)} is inside the plugin payload, which "
            "is a clone of a pinned commit. A store there is a store the "
            "repository can ship, and it is replaced wholesale on the next "
            "update.",
        )

    if os.path.exists(store_path) and not os.path.isdir(store_path):
        raise Refusal(
            "not-a-directory",
            f"{_display_path(store_path)} exists and is not a directory. A "
            "store is a directory of markdown; init would have created the "
            "state directory and the config before finding that out, which is "
            "a half-made setup where the contract promises a refusal.",
        )
    for what, path in (("config", config_path), ("store", store_path)):
        _refuse_unwritable(what, path)

    # An id already in the config, naming somewhere else. Ids are how a store
    # is addressed everywhere — the config, `--debug-config`, doctor's
    # per-store rows — so two stores answering to one id is a store that
    # exists on disk and is never read. The merge keeps the first, silently.
    taken = _store_id_conflict(config_path, _store_id(store_path), store_path)
    if taken:
        raise Refusal(
            "store-id-taken",
            f"{_display_path(config_path)} already has a store called "
            f"{_store_id(store_path)!r} and it is {_display_path(taken)}, not "
            f"{_display_path(store_path)}. Ids are derived from the "
            "directory's own name, so two stores called `notes` in different "
            "places collide — pass a store path whose last segment differs, or "
            "rename one.",
        )

    # The writes that land OUTSIDE memkit's own paths, and the rule is the same
    # for every one of them: a target that resolves inside a memory store is a
    # memory file that edits the harness's configuration. Nothing in this build
    # writes a memory, but the store is a directory an agent is told to write
    # into.
    for flag, target, what in (
        (wire_claude_md, _claude_md(machine), "CLAUDE.md"),
        (
            auto_dream_off or adopt_auto_memory or auto_memory_off,
            _settings_path(machine),
            "settings.json",
        ),
    ):
        if not flag:
            continue
        _refuse_path(what, target)
        if _under_cwd(target):
            raise Refusal(
                "config-dir-in-session-directory",
                f"{_display_path(target)} resolves inside the directory this "
                f"session stands in, so ${CONFIG_DIR_ENV} names the checkout "
                "rather than your own harness configuration. An @-import "
                "written there loads into every session in that project and "
                "not into yours, and a setting written there leaves the one "
                "you asked to change untouched.",
            )
        if _inside(target, store_path):
            raise Refusal(
                "store-resident-target",
                f"{_display_path(target)} resolves inside the memory store. A "
                "file the harness reads as configuration must not also be a "
                "file an agent is told to write memories into.",
            )
        _refuse_unwritable(what, target)

    # BEFORE EITHER SET OF GATES BELOW, because all of them read `scope.data`
    # and a scope that could not be parsed arrives with an empty one — which
    # is indistinguishable from a scope declaring nothing. One trailing comma
    # in `settings.local.json` therefore silences exactly the refusals that
    # protect an adopter from a redirect they did not ask for, in a file the
    # harness itself would also fail to read. The reader already records the
    # error; doctor consults it, and these have to as well.
    if adopt_auto_memory or auto_memory_off:
        by_scope = {scope.scope: scope for scope in machine.settings}
        for name in harness_memory.SCOPE_ORDER:
            scope = by_scope.get(name)
            if scope is None or not scope.error:
                continue
            raise Refusal(
                "settings-unreadable",
                f"{name} settings ({_display_path(scope.path)}) cannot be "
                f"read: {scope.error}. Every check this flag makes about "
                f'"{harness_memory.DIRECTORY_KEY}" and '
                f'"{harness_memory.ENABLED_KEY}" asks that file what it '
                "declares, and a file that will not parse answers nothing "
                "rather than answering no — so proceeding would write your "
                "harness settings on the strength of a question nobody could "
                "ask. The harness cannot read it either. Fix the syntax and "
                "run this again.",
            )

    # AND ABOVE EVERY SCOPE THOSE GATES READ, THE ENVIRONMENT. Two variables
    # decide these same two questions before the harness opens a settings file
    # at all — `harness_memory.DISABLE_ENV` for whether auto-memory runs,
    # `OVERRIDE_ENV` for where it writes — and no gate here consulted either:
    # `--auto-memory-off` under a value spelling "run it" printed "the harness
    # then neither reads nor writes auto-memory" over a write nothing could
    # make true, and `--adopt-auto-memory` under a directory override promised
    # every project's new memories would land in the store while the override
    # sends them elsewhere. Both were accepted, and a digest was offered for
    # them. Asked of the readers doctor asks, so the two commands cannot come
    # to disagree about what the environment decided.
    if auto_memory_off:
        # THREE-VALUED, AND ONLY ONE OF THE THREE IS A CONFLICT. A value the
        # harness reads as "off" agrees with the write, and one it reads as
        # neither leaves the settings to decide; `forced` is true only for the
        # spellings that RUN the feature over every scope below them.
        forced, spelled = harness_memory.env_switch()
        if forced:
            raise Refusal(
                "auto-memory-forced-on",
                f"${harness_memory.DISABLE_ENV} is set to {spelled!r}, which "
                "the harness reads as an instruction to RUN auto-memory "
                "before it opens a settings file at all: no settings scope "
                "turns it off while that value is in the environment. "
                f'--auto-memory-off writes "{harness_memory.ENABLED_KEY}": '
                f"false into {_display_path(_settings_path(machine))}, and "
                "the manifest says the harness then neither reads nor writes "
                "auto-memory — which that variable makes false. Unset "
                f"${harness_memory.DISABLE_ENV} wherever it is exported — a "
                "shell profile, a direnv file, a wrapper script — or set it "
                "to 1, and run this again.",
            )
    if adopt_auto_memory:
        # NAMED, NEVER RESOLVED — see `harness_memory.OVERRIDE_ENV`. Each of
        # these takes a resolver of its own, so what memkit can say honestly
        # is that one is in effect and that it does not know the directory.
        # An unresolved override is exactly the state in which the redirect
        # this flag writes cannot be checked against anything, and the flag's
        # own sentence claims to know where every project writes next.
        overridden = harness_memory.overrides()
        if overridden:
            raise Refusal(
                "auto-memory-overridden",
                "an environment override is in effect ("
                + ", ".join("$" + name for name in overridden)
                + "), so where the harness writes is not what any settings "
                "file says and memkit does not resolve it. "
                f'--adopt-auto-memory writes "{harness_memory.DIRECTORY_KEY}": '
                f'"{_home_form(_redirect_dir(store_path))}" into '
                f"{_display_path(_settings_path(machine))}, and the manifest "
                "says every project's new memories land there from then on — "
                "which is not true of a directory a variable chose first. "
                "Unset it wherever it is exported and run this again, or drop "
                "the flag: copying what is already written is a separate "
                "decision from redirecting what is written next.",
            )

    # BOTH OF THESE ARE --adopt-auto-memory's ALONE. `--auto-memory-off` writes
    # one boolean and has to stay idempotent: refusing it because the feature
    # is already off would take away the convergence every other flag here has,
    # and a second run of it is an empty manifest either way.
    if adopt_auto_memory:
        by_scope = {scope.scope: scope for scope in machine.settings}
        # THE SCOPE SET FOR THE BOOLEAN IS A FAIL-SAFE CHOICE AND NOT A
        # MEASURED ONE. `harness_memory.switch` resolves this key in the
        # precedence the harness was measured to use and answers with the one
        # scope that decides it; this reads every scope ABOVE the one init
        # writes and refuses on any `false` it finds there. The two differ only
        # where a lower scope says false and a higher one says true, and the
        # costs are not symmetric: a refusal costs a flag the adopter drops,
        # and a redirect written for a feature nobody turned on costs a
        # directory they now have to clean up.
        #
        # THE USER SCOPE IS EXCLUDED because it is the scope `--auto-memory-off`
        # itself writes. Refusing on it would make one memkit flag unusable
        # because of state another memkit flag wrote, with a hand edit of the
        # settings file the only way back; the two flags are mutually exclusive,
        # so no single invocation can undo it. Adopting what the harness wrote
        # BEFORE it was switched off is legal — the copy is of files that
        # already exist — and the off state is disclosed in the manifest
        # instead.
        for name in _scopes_outranking_user():
            scope = by_scope.get(name)
            # `is False` and not falsiness: JSON `0`, `""` and `[]` are all
            # values the harness goes on writing under.
            if scope is None or scope.data.get(harness_memory.ENABLED_KEY) is not False:
                continue
            raise Refusal(
                "auto-memory-off",
                f'"{harness_memory.ENABLED_KEY}": false is set in {name} '
                f"settings ({_display_path(scope.path)}), so the harness "
                "writes no auto-memory at all — and --adopt-auto-memory would "
                "copy what is there and then point a switched-off feature at "
                "your store. Turn it back on in that file if you want new "
                "memories to land there, or drop the flag: nothing new is "
                "going to be written for any project while that value is "
                "false.",
            )
        planned = _redirect_dir(store_path)
        for name in harness_memory.SCOPE_ORDER:
            scope = by_scope.get(name)
            if scope is None:
                continue
            value = scope.data.get(harness_memory.DIRECTORY_KEY)
            if value is None:
                continue
            # A value that is not a string is one the harness rejects and falls
            # back to its default from — still not the planned directory, and
            # still somebody's declaration to leave alone.
            if isinstance(value, str) and os.path.normpath(
                expand_home(value)
            ) == os.path.normpath(planned):
                continue
            shown = (
                _display_path(expand_home(value)) if isinstance(value, str)
                else repr(value)
            )
            # WHICH SCOPE DECIDES IT, asked of the same reader doctor asks, so
            # the two commands cannot come to disagree about it. `None` is a
            # declaration the harness would not use, which sends it to its own
            # default rather than to the next scope down.
            winner, decides = harness_memory.configured_dir(machine.settings)
            says = (
                f"the harness reads {decides} settings for it"
                if winner is not None
                else "no scope declares a value the harness would use, so it "
                "writes to its own default directory instead"
            )
            raise Refusal(
                "auto-memory-redirected",
                f'"{harness_memory.DIRECTORY_KEY}" is already set to {shown} '
                f"in {name} settings, and {says}. --adopt-auto-memory writes "
                f'"{_home_form(planned)}" into the user scope '
                f"({_display_path(_settings_path(machine))}), which would "
                "either be ignored or move where your agent writes without "
                "your having said so. Point that setting at the store "
                "yourself and run this again, or drop the flag — copying what "
                "is already written is a separate decision from redirecting "
                "what is written next.",
            )

    if auto_memory_off:
        by_scope = {scope.scope: scope for scope in machine.settings}
        for name in _scopes_outranking_user():
            scope = by_scope.get(name)
            if scope is None:
                continue
            value = scope.data.get(harness_memory.ENABLED_KEY)
            # `false` up there is not a conflict — the feature is already off
            # and the note below says which scope did it. Anything else
            # declared is a value that WINS over the one being written, so the
            # write cannot produce the effect the manifest promises for it.
            if value is None or value is False:
                continue
            raise Refusal(
                "auto-memory-outranked",
                f'"{harness_memory.ENABLED_KEY}" is set to {value!r} in {name} '
                f"settings ({_display_path(scope.path)}), which the harness "
                f"reads ahead of the user scope. --auto-memory-off writes "
                f'"{harness_memory.ENABLED_KEY}": false into '
                f"{_display_path(_settings_path(machine))}, where that "
                "declaration would outrank it — the flag would report success "
                "and the harness would go on writing auto-memory. Change it "
                "where it is set, or drop the flag.",
            )

    if _inside(config_path, machine.state_dir):
        raise Refusal(
            "config-in-state-dir",
            f"{_display_path(config_path)} is inside "
            f"{_display_path(machine.state_dir)}, which holds derived state "
            "and which the hook sweeps on its own schedule. The sweep keeps "
            "what init's journal claims and collects nothing whose name it "
            "does not recognise, so this would probably survive — and 'would "
            "probably survive' is not a property to hang a config on. Put it "
            "somewhere nothing collects.",
        )

    route, _found = _checker_route(machine)
    if route is CheckerRoute.NONE:
        raise Refusal(
            "no-checker-route",
            "no python on this machine meets the integrity checker's floor "
            "and `uv` located none either, so init cannot verify the store it "
            "would seed. A seeded memory whose ledger nobody checked is a "
            "store the checker calls broken, and half-completing is worse "
            "than not starting. " + NO_CHECKER_REMEDY,
        )

    seeded = _foreign_canary(store_path, _canary_nonce(config_path))
    if seeded:
        raise Refusal(
            "canary-belongs-to-another-config",
            f"{_display_path(store_path)} already holds a canary carrying "
            f"nonce {seeded!r}, and this config's is "
            f"{_canary_nonce(config_path)!r}. The nonce is keyed on the config "
            "so that one config's fixed query answers for all of its stores; "
            "two configs over one store means rewriting the canary would take "
            "the other one's doctor check away. Point this init at the config "
            "that already owns the store, or give it a store of its own.",
        )

    if os.path.isdir(store_path):
        # THE INDEX THE ADOPTER WROTE. `MEMORY.md` is excluded from the stray
        # scan above by name, so a store holding nothing else passes
        # `flat-store-adoption` — and the plan then CREATES a file that is
        # already there, replacing an index whose rows are what loads into
        # every session. init converges on its own work and never overwrites
        # what somebody else wrote; this is that rule on the file where losing
        # it is least recoverable, since nothing else records those rows.
        #
        # Byte-identical is not somebody else's: that is what a re-run after a
        # partial one looks like, and refusing it would take away the recovery
        # the incomplete exit code tells the adopter to perform.
        ledger = os.path.join(store_path, "MEMORY.md")
        if os.path.exists(ledger):
            try:
                with open(ledger, encoding="utf-8") as f:
                    held = f.read()
            except OSError as exc:
                raise Refusal(
                    "adopted-memory-index",
                    f"{_display_path(ledger)} exists and cannot be read "
                    f"({exc}). init would create that file, and it will not "
                    "replace one it cannot first see.",
                ) from exc
            except UnicodeDecodeError:
                # A file that is not UTF-8 is not the file init generates, and
                # saying so is the whole answer. It used to reach the caller as
                # a traceback from `read()`, which is the one shape this
                # command promises never to produce: `--dry-run` is
                # pre-approved and an adopter meets it with no refusal name to
                # act on. Not byte-identical is a state the branch below
                # already handles.
                held = None
            if held != _memory_ledger(store_path):
                raise Refusal(
                    "adopted-memory-index",
                    f"{_display_path(ledger)} already exists and is not the "
                    "file init generates, so it is an index somebody wrote. "
                    "Its rows are what loads into every session and nothing "
                    "else records them, so init will not write over it. Adopt "
                    "this store by hand, or pass --store with a directory "
                    "that has no MEMORY.md in it.",
                )
        stray = _stray_markdown(store_path)
        if stray and not os.path.isdir(os.path.join(store_path, "search")):
            raise Refusal(
                "flat-store-adoption",
                f"{_display_path(store_path)} already holds markdown at its "
                f"root ({', '.join(stray)}) and has no search/. Creating one "
                "would un-retrieve every one of those files in a single step, "
                "silently, with every diagnostic still green. Migrate first: "
                f"mkdir {_display_path(store_path)}/search && mv "
                f"{_display_path(store_path)}/*.md "
                f"{_display_path(store_path)}/search/ — then re-run init.",
            )

    if os.path.exists(config_path) and config_path not in authored_configs(
        machine.state_dir
    ):
        raise Refusal(
            "foreign-config",
            f"{_display_path(config_path)} exists and no init journal entry "
            "claims it, so memkit did not write it. init converges on its own "
            "work and never overwrites a config somebody else wrote — that "
            "file decides which directories the every-prompt hook reads.",
        )


# --- the plan ----------------------------------------------------------------


def _resolve_config(machine: Machine, named: str | None) -> str:
    """Where the config goes — which has to be a path this install will READ.

    The INSTALL OPTION wins over any default, because an adopter who passed
    `--config memkitConfig=<path>` has already said where they want it and a
    config written anywhere else would leave the option pointing at nothing —
    the highest-cost silent state in the whole field log, created by the
    command that exists to prevent it.

    ON THE PLUGIN CHANNEL WITH NO OPTION, the answer is rung 2 and not a
    tidy-looking path under `~/.config`. The wrapper reads two rungs and
    nothing else; a config anywhere else is one the hook can never see, so an
    adopter would get a store, a green integrity check, an exit 0 and silence
    on every prompt — with doctor telling them to run the command that just
    ran. `bin/lib/common.sh` already names init as the one thing that will
    ever legitimately write that path, and the journal entry init makes is
    what `config-authorship` reads to tell memkit's own file from a planted
    one.

    Its lifetime is the trade, and it is the right one for this file: plugin
    data dies with `claude plugin uninstall` unless `--keep-data`, and a
    config init can regenerate is exactly the kind of thing that should. The
    journal, which a later undo needs, lives in the state directory instead.
    """
    if named:
        return expand_home(named)
    option, _scope = machine.settings_option()
    if option:
        # `expand_home`, not `os.path.expanduser`: the option value is a
        # string typed into an install command and never expanded by a shell,
        # and the wrapper expands exactly `~` and `~/`. Expanding `~someone`
        # here would admit a path the wrapper leaves relative and refuses.
        return expand_home(option)
    if machine.plugin:
        rung_two = machine.rung_two
        if not rung_two:
            raise Refusal(
                "no-config-route",
                "this is a plugin install with no `memkitConfig` option set "
                f"and no usable ${PLUGIN_DATA_ENV}, so there is nowhere to "
                "put a config that the hook would read. Writing one anyway "
                "would leave you with a store, a clean integrity check and a "
                "hook that says nothing forever.\n"
                "Set the option — `claude plugin install memkit@memkit --yes "
                "--config memkitConfig=<absolute path>`, or `/plugin "
                "configure memkit@memkit` — and run init again. Or pass "
                "`--config <absolute path>` here and point $MEMKIT_CONFIG at "
                "it yourself.",
            )
        return rung_two
    return expand_home(DEFAULT_CONFIG)


def _config_route_note(machine: Machine, config_path: str) -> str:
    """Which route will read the config this plan writes.

    In the manifest because "a config was written" and "the hook can read it"
    are two facts, and conflating them is how an install ends up configured
    and inert. The reader is being asked to consent to a write; what they
    care about is whether it does anything.
    """
    option, _scope = machine.settings_option()
    if option and expand_home(option) == config_path:
        return f"Read via the {OPTION_KEY} install option"
    if machine.plugin and config_path == machine.rung_two:
        return f"Read via ${PLUGIN_DATA_ENV}/{GENERATED_CONFIG_NAME}"
    if machine.plugin:
        return (
            "WARNING: this path is on neither rung a plugin install reads "
            f"(the {OPTION_KEY} option, ${PLUGIN_DATA_ENV}/"
            f"{GENERATED_CONFIG_NAME}), so the hook will not see it"
        )
    return f"Read via --config or ${CONFIG_ENV}"


def _interpreter() -> str:
    """The absolute python this process is, which is the one that will read
    every prompt if the wrapper honours the record.

    `sys.executable` resolved: a venv's `python3` is a symlink, and recording
    the link records a path whose target the adopter can move.
    """
    return os.path.realpath(sys.executable)


def _chosen_interpreter(named: str | None) -> str:
    """The python this run will RECORD, which is not always the one it is.

    `--interpreter` exists because the config's `interpreter` field was
    reachable only through this command and this command was reachable only
    through a config it had already written: an adopter whose every candidate
    python is unusable had no first move. Naming one here is that move, and it
    is the only one of the three routes that leaves the answer written down
    rather than living in an environment.

    Resolved for the reason `_interpreter` resolves: a venv's `python3` is a
    symlink, and the field is read by a wrapper that cannot follow one back to
    a target the adopter has moved. The SHAPE is judged before this, in
    `check_refusals`, against the value as typed — resolving first would turn a
    relative path into an absolute one inside whatever directory the session
    stands in, which is the value that rule exists to refuse.
    """
    if named:
        return os.path.realpath(expand_home(named))
    return _interpreter()


def _store_id(store: str) -> str:
    """A config id derived from the store's own directory name.

    Ids appear in `--debug-config`, in doctor's per-store rows and in the
    inert message, so `notes` reads better than `store-0` — and a store the
    adopter named is one they can recognise in a report.
    """
    base = os.path.basename(store.rstrip(os.sep)) or "memories"
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in base)
    return cleaned.strip("-") or "memories"


def _store_id_conflict(config_path: str, store_id: str, store_path: str) -> str:
    """The directory an existing store of this id already names, or "".

    Read out of the config rather than tracked separately: the config is where
    the collision would land, and a second source for "which ids are taken"
    would be a second thing to keep in step with it.
    """
    with contextlib.suppress(OSError, ValueError):
        with open(config_path, encoding="utf-8") as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            return ""
        roots = blob.get("roots")
        for store in blob.get("stores") or []:
            if not isinstance(store, dict) or store.get("id") != store_id:
                continue
            spec = (roots or {}).get(store.get("live_root")) or {}
            existing = spec.get("path") if isinstance(spec, dict) else None
            if isinstance(existing, str):
                existing = os.path.expanduser(existing)
                if os.path.realpath(existing) != os.path.realpath(store_path):
                    return existing
            return ""
    return ""


def _git_tracked(path: str) -> bool:
    """Whether git would consider this file part of a repository.

    Not a hard refusal — an adopter may well keep their `CLAUDE.md` in a
    dotfiles repo on purpose — but an `@-import` line appended to a tracked
    file is a commit somebody did not intend to make, and the manifest is the
    place that says so before it happens.
    """
    # The RESOLVED path, for the same reason the write follows the link: a
    # `CLAUDE.md` symlinked into a dotfiles repo is tracked there, and asking
    # git about the link's own directory answers about the wrong tree.
    path = os.path.realpath(path)
    parent = os.path.dirname(path) or "."
    # `init --dry-run` is the PRE-APPROVED half of the handshake — the skill
    # grants it with no permission prompt — so this call must not be able to
    # start a program somebody else chose. Which git runs and what it is told
    # to forget are settled by the route table; WHERE it stands is settled
    # here, and it is a parameter of the call rather than a word inside an
    # argv.
    #
    # A repository's own `.git/config` names programs git runs — `ls-files
    # --error-unmatch` executes `core.fsmonitor`, reproduced twice per call —
    # so pointing this at a directory inside the session is handing a checkout
    # the same primitive under another name. `_trusted_git` silences that
    # config surface; the refusal below is the half a `-c` cannot buy, because
    # a repository can always add a key nobody thought to override.
    #
    # Unresolvable, or refused, means the warning is not made rather than made
    # by whatever was in front.
    if _under_cwd(parent):
        return False
    try:
        out = run_git(GitRoute.TRACKED, repo=parent, path=path, timeout=15)
    except (OSError, subprocess.SubprocessError, Untrusted):
        # swallow: nothing discloses this one. The manifest loses its
        # "tracked by git" warning and says nothing about having lost it.
        return False
    return out.returncode == 0


def _import_line(store: str) -> str:
    return f"@{os.path.join(store, 'MEMORY.md')}"


def _claude_md(machine: Machine) -> str:
    return os.path.join(_harness_config_dir(), "CLAUDE.md")


def _harness_config_dir() -> str:
    """Where the harness keeps its settings and its own memories.

    One reader rather than the expression repeated at each use: the two
    directories this command now writes about — the settings file it may
    change and the projects tree it may copy out of — are the same answer to
    one question, and two copies of it is how they come to disagree.
    """
    return os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")


def _settings_path(machine: Machine) -> str:
    return os.path.join(_harness_config_dir(), "settings.json")


def _scopes_outranking_user() -> tuple:
    """The scope names the harness resolves BEFORE the one init writes.

    Every settings write this command makes goes to the user scope, and the
    first scope in `SCOPE_ORDER` to declare a key is the one that decides it.
    So a declaration in any of these is one the write cannot change the effect
    of, whatever the manifest line under it promises.
    """
    order = harness_memory.SCOPE_ORDER
    return order[: order.index(USER)]


# The operations whose target is a FILE. `VERIFY` names the store directory
# and writes nothing, so it is not one of them.
_FILE_OPS = (CREATE_FILE, MERGE_CONFIG, SETTINGS_WRITE, APPEND_LINE)


def _refuse_incompatible_types(actions: list) -> None:
    """Every planned path, checked for a type that will not take its write.

    The store ROOT already had this rule, with the reason written beside it:
    init would otherwise create the state directory and the config before
    finding out. Its DESCENDANTS did not, so a regular file at
    `<store>/search` printed a normal manifest, and `--confirm` then made the
    0700 state directory, wrote the config, and exited 6 at `os.makedirs` —
    a half-configured install after a command whose contract is that it
    refuses safely.

    Over the ACTIONS rather than a list of paths, so an action added later is
    covered by being in the plan.
    """
    for action in actions:
        if not os.path.exists(action.path):
            continue
        if action.op == CREATE_DIR and not os.path.isdir(action.path):
            raise Refusal(
                "not-a-directory",
                f"{_display_path(action.path)} exists and is not a directory, "
                "and init would have to create one there. Nothing has been "
                "written; move that file aside and run this again.",
            )
        if action.op in _FILE_OPS and os.path.isdir(action.path):
            raise Refusal(
                "not-a-file",
                f"{_display_path(action.path)} exists and is a directory, and "
                "init would have to write a file there. Nothing has been "
                "written; move that directory aside and run this again.",
            )


# --- adopting the harness's own auto-memory ----------------------------------
#
# THE CHECKER'S RULES, RESTATED RATHER THAN IMPORTED. `memory_integrity`
# requires 3.12 and this module answers to the 3.9 floor `bin/memkit` runs the
# dispatcher on, so the two cannot share one definition of what a ledger row
# is. What closes that gap is evidence rather than care: a 3.12 case
# regenerates the ledger init wrote using the checker's own generator and
# requires the bytes to be equal, so a rule that drifts here fails there.
#
# The reason any of this is here at all: a memory under `search/` with no row
# in SEARCH.md is an ORPHAN, which fails the VERIFY step init runs on its own
# work. Adoption that left the store failing its own checker would hand the
# adopter a broken store and an exit 6 out of the command that made it.

# Ledgers and sub-indexes are never memories, at any depth, and are recognised
# by FILENAME — which is what lets the harness's own `MEMORY.md` travel beside
# the memories it indexed without owing anybody a row.
_LEDGER_NAMES = frozenset({"MEMORY.md", "SEARCH.md", "INDEX.md"})
_INDEX_HEADING = "## Index"
# The checker's cap, which is the hook's 157-character truncation less the
# ellipsis. A memory written to the hook's ceiling fails the check.
_MAX_DESC_CHARS = 155
# A plain YAML scalar may not START with one of these, and may not CONTAIN
# ": " or " #".
_YAML_INDICATORS = set("-?:,[]{}#&*!|>%@`")
# `tier:` is the layout the tier directories replaced, and the checker rejects
# a file still carrying it.
_TIER_RE = re.compile(r"^tier:\s*\S+", re.MULTILINE)
# A markdown link's destination, the way the checker reads a ledger's rows —
# within one line, so a description carrying an unbalanced bracket cannot
# take the destination off the row below it.
_LINK_RE = re.compile(r"\(([^)\n]+\.md)\)")
# The same links read the way the checker's DEAD-LINK rule reads them, which
# is not the same question: a row's destination need not end in `.md` to fail
# that check, so an index rowing a `.png` that is not coming wedges the store
# exactly as a `.md` row does.
_MD_LINK_RE = re.compile(r"!?\[[^\[\]\n]*\]\(([^)\n]*)\)")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
# The checker's own list. A destination with no slash in it is a path only if
# it ends in one of these; anything else is an anchor or a word and is left
# alone rather than guessed at.
_PATH_SUFFIXES = (
    ".md",
    ".py",
    ".sh",
    ".nix",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
)
# A heading, whose text stands in for a description the file does not have.
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(\S.*?)[ \t]*$", re.MULTILINE)

# Where adopted memories land inside the corpus root, one directory per
# harness project key. Under `search/` because that is what retrieval reads,
# and NOT under the directory the redirect points at: what the harness writes
# is rewritten by it, and what this copies must not be.
ADOPT_DIRNAME = "projects"
# What adoption will not carry. A megabyte is two orders of magnitude past any
# memory in any store here, so a file over it is something else that happens to
# end in `.md` — and the whole file is read into the manifest's digest.
ADOPT_MAX_BYTES = 1 << 20
# What one directory entry may be, in bytes. The POSIX `NAME_MAX` every
# filesystem this runs on holds to, as a constant rather than an `os.pathconf`
# of the destination: the answer would have to be asked of the nearest ancestor
# that exists yet, which at plan time is not the directory the copy lands in,
# and a value read off the wrong filesystem is worse than the floor.
_NAME_MAX_BYTES = 255
# And what the write adds to it. `_write_atomically` writes beside the file and
# renames over, and the name it writes beside is `<name>.<pid>.tmp` — so the
# name the plan approves is not the longest name the write actually creates.
# The pid is measured at the widest a 32-bit `pid_t` can be spelled rather than
# at this process's own: the plan and the write are two processes, and a plan
# approved by a four-digit pid must still land under a six-digit one.
_TMP_SUFFIX_BYTES = len(".") + 10 + len(".tmp")


def _frontmatter_of(text: str) -> dict:
    """Top-level scalar keys of a leading `---` block, values RAW.

    The checker's own reader, restated: not a YAML parser, because the raw
    text is what its description rules judge and the two must agree about
    which line carries the description.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    out: dict = {}
    for line in text[3 : end if end != -1 else len(text)].splitlines():
        if not _is_frontmatter_key(line):
            continue
        key, _sep, value = line.partition(":")
        out.setdefault(key.strip(), value.strip())
    return out


def _is_frontmatter_key(line: str) -> bool:
    """Whether this line of a `---` block declares a top-level key.

    Nested keys (the harness buries everything it does not recognise under
    `metadata:`) are indented and are not; a comment is not; a line with no
    colon, or a spaced word before one, is not.
    """
    if not line or line[0].isspace() or line.startswith("#"):
        return False
    key, sep, _value = line.partition(":")
    return bool(sep and key.strip() and " " not in key.strip())


def _scalar_of(raw: str):
    """The value a raw frontmatter scalar carries, or None when it is not one.

    `memory_integrity._scalar` without its error strings. A `None` here is
    exactly a `DESC-BAD` there.
    """
    if not raw:
        return None
    if raw[0] in "\"'":
        quote = raw[0]
        if len(raw) < 2 or raw[-1] != quote:
            return None
        inner = raw[1:-1]
        if quote == "'":
            return inner.replace("''", "'")
        if re.search(r'(?<!\\)"', inner):
            return None
        return inner.replace('\\"', '"')
    if raw[0] in _YAML_INDICATORS:
        return None
    if ": " in raw or raw.endswith(":"):
        return None
    if " #" in raw:
        return None
    return raw


def _quotable(raw: str):
    """The text a description line carries when plain style cannot hold it,
    or None when the line is not text this may read.

    `_scalar_of` answers None to two different questions. One is "this value
    is not readable"; the other is "this value is readable and plain style may
    not carry it" — an inner `": "`, a trailing colon, a ` #`. The second class
    is a sentence somebody wrote, and quoting is precisely what `_as_scalar`
    does with one.

    NONE FOR ANYTHING WHOSE FIRST CHARACTER CLAIMS A STRUCTURE: a quote that
    has to close, and may close on a line this reader never sees; a block
    indicator whose value is the lines below it; an anchor, an alias, a flow
    collection. Those are not text with an awkward character in it, and
    quoting the one line they begin would be inventing a description rather
    than keeping one.
    """
    if not raw or raw[0] in "\"'" or raw[0] in _YAML_INDICATORS:
        return None
    return raw


def _as_scalar(value: str) -> str:
    """`value` written so that reading it back gives `value`.

    Plain where a plain scalar is legal and double-quoted where it is not,
    which is every shape the checker's rejection set names: a leading
    indicator, an inner `": "`, a trailing colon, an inner ` #`.

    A TRAILING BACKSLASH is dropped before quoting. The checker's reader would
    hand it back happily; a real YAML parser reads the `\\"` it makes as an
    escaped quote and the scalar as never closed, and this text goes into a
    file other tools read.
    """
    value = value.rstrip("\\")
    if value and _scalar_of(value) == value:
        return value
    return '"' + value.replace('"', '\\"') + '"'


def _body_of(text: str) -> str:
    """Everything after the frontmatter block, or all of it when there is none.

    A `---` THAT NEVER CLOSES OPENS NOTHING. Returning "" for it left
    `_first_heading` searching an empty string, so a file whose very next line
    was a heading got its description from the file name instead — and the
    line this treats as a body is the same one `_with_description` writes into
    for that shape, which is what keeps the two readings of it together.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    return text[end:] if end != -1 else text[3:]


def _first_heading(text: str) -> str:
    found = _HEADING_RE.search(text)
    return found.group(1).strip() if found else ""


def _normalise(text: str, stem: str) -> tuple:
    """(`text` with a description and a label a ledger row can carry, the
    rules applied).

    "" for the rule when the file already had one, which is the case adoption
    must leave alone — a copy that rewrote a description the checker accepts
    would be an edit to somebody's memory for no gain.

    BODIES ARE NEVER TOUCHED. Only the frontmatter's description line is
    written, and only when the file has nothing usable there: what an adopter
    consented to is a copy, and the one thing that makes a copy fail the
    store's own checker is a description it cannot read.
    """
    raw = _frontmatter_of(text).get("description", "")
    value = _scalar_of(raw)
    rules = []
    if value is None:
        # QUOTED BEFORE ANYTHING STANDS IN FOR IT. The adopter's own sentence
        # is the description; a heading and a file name are guesses at one, and
        # the guess was taken for every line plain style could not carry — a
        # sentence became a slug, losslessly recoverable the whole time.
        carried = _clean(_quotable(raw) or "")
        if carried:
            value = carried
            rules.append(
                "description holds what a plain scalar cannot carry — quoted "
                "as it stands"
            )
    if value is None:
        heading = _clean(_first_heading(_body_of(text)))
        # CLEANED, both of them. The stem is a filename, which may hold a
        # newline, and this value goes onto a frontmatter line and then into a
        # ledger row — each of which a newline ends, taking what follows with
        # it. `name:` was cleaned here and the description beside it was not.
        value = heading or _clean(stem)
        rules.append(
            "description "
            + ("was missing" if not raw else "could not be read")
            + " — taken from "
            + ("its first heading" if heading else "the file name")
        )
    if len(value) > _MAX_DESC_CHARS:
        rules.append(
            f"description was {len(value)} characters — truncated to "
            f"{_MAX_DESC_CHARS}"
        )
        value = value[: _MAX_DESC_CHARS - 1] + "…"
    if rules:
        if not text.startswith("---"):
            rules.append("no frontmatter block — one was added")
        text = _with_description(text, _as_scalar(value), stem)
    # AFTER the block exists, and asked whether there are other rules or not: a
    # file whose description the checker already reads can still carry a name
    # that ends the row's link.
    text, relabelled = _relabel(text, stem)
    if relabelled:
        rules.append(relabelled)
    return text, "; ".join(rules)


def _with_description(text: str, scalar: str, stem: str) -> str:
    """`text` carrying exactly this description, and nothing else changed.

    THREE SHAPES, and the middle one is the reason this is not a regex. A file
    with no `---` at all gets a block prepended, carrying the name the row will
    use. A file whose block already declares a description has its FIRST one
    rewritten — first, because that is the one the reader takes — along with
    the indented lines under it, which are the rest of the value it replaced
    and would otherwise attach somebody else's sentence to this description. A
    file with a block that declares none gets the line straight after the
    opening `---`, which is also where a `---` that never closes has to take
    it.
    """
    if not text.startswith("---"):
        return (
            f"---\nname: {_as_scalar(_label(stem))}\ndescription: {scalar}\n"
            f"---\n\n{text}"
        )
    return _with_scalar(text, "description", scalar)


def _with_scalar(text: str, key: str, scalar: str) -> str:
    """`text`'s frontmatter block carrying exactly this `key:` line.

    The two shapes that edit a block that is already there, shared by the
    description and the label because the arithmetic is the same one and a
    second copy of it is how the continuation-line rule comes to hold for one
    key and not the other. Callers hold the third shape — a file with no block
    at all, which needs both lines at once.
    """
    line = f"{key}: {scalar}"
    end = text.find("\n---", 3)
    rows = text[3 : end if end != -1 else len(text)].split("\n")
    at = None
    for index, row in enumerate(rows):
        if _is_frontmatter_key(row) and row.partition(":")[0].strip() == key:
            at = index
            break
    if at is None:
        rows.insert(1, line)
    else:
        rows[at] = line
        while at + 1 < len(rows) and rows[at + 1][:1] in (" ", "\t"):
            del rows[at + 1]
    return text[:3] + "\n".join(rows) + (text[end:] if end != -1 else "")


def _clean(value: str) -> str:
    """A filename's stem with what a frontmatter line cannot hold taken out.

    POSIX admits every byte but NUL and `/` in a filename, a newline included,
    and one written into a block unescaped ends the line it is on — taking the
    description below it with it.

    Also what every adopter-controlled string goes through before it is
    rendered into the manifest: a memory named
    `a\n  create-file    ~-.claude-settings.json\nb.md` put two correctly
    indented action lines into the surface a human reads before typing
    `--confirm`.
    """
    return "".join(c for c in value if c.isprintable()).strip()


def _findable(value: str) -> str:
    """`_clean`'s counterpart for a name the reader has to go find on disk.

    DELETING THE BYTE RENAMES THE THING. `_clean` is right for text nobody
    types back, and wrong for the one line telling an adopter which directory
    was passed over: a key holding a tab was reported as `keywithtab`, a
    spelling `ls` matches nothing with. Escaping keeps every byte accounted
    for and still ends the line where the line ends, which is the whole of
    what `_clean` was protecting.
    """
    return "".join(c for c in repr(value) if c.isprintable())


# What a ledger ROW's label may not carry. A row is
# `- [label](link) — description`, so a `]` in the label ends the link early
# and everything after it is markdown nobody wrote.
_LINK_SYNTAX = frozenset("[]()")


def _label(value: str) -> str:
    """`_clean`'s stricter sibling: a label a generated row can carry."""
    return "".join(c for c in _clean(value) if c not in _LINK_SYNTAX).strip()


def _link_target(value: str) -> str:
    """`_label`'s counterpart for the LINK half of a row: what `(...)` holds.

    Stricter than `_label` in one place: a label may carry a space and a link
    destination may not, because the reader ends the link at the first one. A
    path segment holding a space therefore rows a link to a PREFIX of itself,
    which is a path that is usually not there at all.
    """
    return "".join(c for c in _label(value) if not c.isspace())


def _relabel(text: str, stem: str) -> tuple:
    """(`text` carrying a label a row can carry, the rule applied).

    THE LABEL IS THE ONE HALF OF A ROW TAKEN RAW — the description goes
    through `_scalar_of` and the link is built from a path. A `name:` of
    `x](hot/forged.md) — forged row` rows a link to a file that is not there
    and leaves the real memory with no usable row, and the same goes for a
    file NAMED that way, since the stem is the label when no `name:` declares
    one.

    THE ADOPTED TEXT IS WHAT IS REWRITTEN, not init's own row, and that is the
    whole reason this exists as a normalisation rule rather than as a sanitiser
    at the point the row is built: the checker regenerates that row from this
    file's frontmatter, so a label init cleaned over a file it did not would
    agree with the checker exactly until the next `--write`.

    `(text, "")` for everything a row can already carry, which is every
    ordinary memory — a name is rewritten only when the line as it stands
    would end the link.
    """
    raw = _frontmatter_of(text).get("name", "")
    shown = raw or stem
    if _label(shown) == shown:
        return text, ""
    # The VALUE where the raw line parses as one, so a quoted name keeps its
    # quotes rather than gaining a second pair.
    value = _scalar_of(raw) if raw else None
    safe = _label(value if value is not None else shown) or _label(stem)
    if not safe:
        # Nothing to write that would name anything. `_plan_adoption` skips the
        # file for the same reason it skips one with no readable description.
        return text, ""
    return _with_scalar(text, "name", _as_scalar(safe)), (
        "name carried markdown-link syntax or characters a row cannot hold — "
        f"rewritten to {safe!r}"
    )


def _regular_text(path: str, errors: str = "strict") -> tuple:
    """(the file's text, why it is not one). Exactly one is set.

    OPENED FIRST AND ASKED WHAT IT IS SECOND, on the descriptor rather than on
    the name: a `stat` before the open answers about whatever the name held
    then, and the thing a planner must never meet is an object whose open does
    not return. `O_NONBLOCK` is what makes the open of a FIFO return at all —
    `--dry-run` is the turn that exists to be read before anything is written,
    and one that hangs with no output has no turn after it to recover from.

    `O_NOFOLLOW` is deliberately NOT set. A store may hold a memory that is a
    symlink — `_memories_under` yields one under its own name, exactly as the
    checker does — so the question here is what the name finally resolves to,
    which is what `fstat` on the open descriptor answers.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        raise
    except OSError as exc:
        return None, f"cannot be read ({type(exc).__name__})"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None, "is not a regular file"
        with open(fd, encoding="utf-8", errors=errors, closefd=False) as f:
            return f.read(), ""
    except UnicodeDecodeError:
        return None, "is not UTF-8"
    except (OSError, ValueError) as exc:
        return None, f"cannot be read ({type(exc).__name__})"
    finally:
        os.close(fd)


def _read_source(path: str) -> tuple:
    """(the file's text, why it cannot be adopted). Exactly one is set.

    Read as TEXT, with the platform's newline translation, because that is how
    `state_token` reads the destination back: a plan whose content the state
    token could never equal would copy the same file on every run and never
    converge.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return None, f"cannot be read ({type(exc).__name__})"
    if size > ADOPT_MAX_BYTES:
        return None, f"is {size} bytes, over the {ADOPT_MAX_BYTES}-byte cap"
    try:
        return _regular_text(path)
    except FileNotFoundError:
        return None, "cannot be read (FileNotFoundError)"


def _held_text(path: str) -> tuple:
    """(what is at `path`, whether it could be read at all).

    `(None, True)` for a path with nothing at it; `(None, False)` for one this
    process cannot read or decode — or one that is not a regular file at all —
    which adoption treats as DIVERGED rather than as a traceback, or a hang,
    out of `--dry-run`.
    """
    try:
        text, why = _regular_text(path)
    except FileNotFoundError:
        return None, True
    return (None, False) if why else (text, True)


def _sub_indexes(store: str, config_path: str) -> list:
    """The sub-index files the existing config declares for THIS store.

    Read out of the config's own `stores[]` entry, matched on the id VERIFY
    will read it under — a sub-index owns its members' rows, and generating a
    second row for one of them in SEARCH.md is the checker's `DOUBLE-LEDGER`.
    """
    out = []
    with contextlib.suppress(OSError, ValueError):
        with open(config_path, encoding="utf-8") as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            return out
        wanted = _store_id(store)
        for entry in blob.get("stores") or []:
            if not isinstance(entry, dict) or entry.get("id") != wanted:
                continue
            for name in entry.get("sub_indexes") or []:
                if isinstance(name, str) and name:
                    out.append(os.path.join(store, name))
    return out


def _sub_index_members(store: str, config_path: str) -> set:
    """Every store-relative path a sub-index of this store already rows.

    KEYED ON THE RESOLVED PATH, because that is what the checker's
    `_sub_members` keys on — its `_rows` resolves every link against the
    sub-index's own directory. So a sub-index that rows a symlinked `.md` file
    claims the file the link points AT, and the link itself stays unclaimed
    and gets a SEARCH.md row here, which is exactly the row `--write`
    generates for it. Measured: on that store the checker's own `--write` exits
    0 and leaves the ledger init produced byte-identical. Keying this on the
    unresolved path instead would make the two disagree.
    """
    out = set()
    root = os.path.realpath(store)
    for sub in _sub_indexes(store, config_path):
        with contextlib.suppress(OSError, ValueError):
            text, why = _regular_text(sub, errors="replace")
            if why:
                continue
            for link in _LINK_RE.findall(text):
                if "://" in link:
                    continue
                target = os.path.realpath(os.path.join(os.path.dirname(sub), link))
                if os.path.basename(target) in _LEDGER_NAMES:
                    continue
                rel = os.path.relpath(target, root)
                if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                    out.add(rel)
    return out


def _memories_under(store: str, tier: str) -> list:
    """Every memory file under one tier of `store`, store-relative and sorted.

    The checker's `_live` restated for one tier: `rglob("*.md")` less the
    ledger names, which yields a symlinked FILE under its own name and never
    descends a symlinked directory — the same two answers `os.walk` gives with
    `followlinks` left alone.
    """
    out = []
    for dirpath, _dirs, names in os.walk(os.path.join(store, tier)):
        out.extend(
            os.path.relpath(os.path.join(dirpath, name), store)
            for name in names
            if name.endswith(".md") and name not in _LEDGER_NAMES
        )
    return sorted(out)


def _rows_on_disk(store: str, config_path: str) -> tuple:
    """({store-relative path: row} for the memories under `search/`, notes).

    THE HALF THAT WAS MISSING. SEARCH.md was written from the canary alone, so
    an init against a store that already held memories replaced a ledger of
    their rows with a ledger of one — every one of them an ORPHAN at the next
    check and unreachable from the file that indexes them.

    Read the way the checker reads them, decode errors replaced rather than
    raised, so the rows this produces are the rows it would generate.

    ROWS FOR THE `search/` HALF AND NOTES FOR THE REST. The checker's `_live`
    walks `hot/` too, and generates no row for anything it finds there: a hot
    memory is rowed in MEMORY.md, which is hand-written and which no `--write`
    regenerates. So init cannot supply those rows — putting them in SEARCH.md
    instead is worse and was measured (`MISROWED: ./hot/h.md is hot but rowed
    in SEARCH.md`) — and what it can do is name them in the manifest, since
    every unrowed one is an `ORPHAN` at the VERIFY step init runs on its own
    work and the exit code arrives after the store is on disk.
    """
    out: dict = {}
    notes = []
    claimed = _sub_index_members(store, config_path)
    for link in _memories_under(store, "search"):
        if link in claimed:
            continue
        path = os.path.join(store, link)
        try:
            text, why = _regular_text(path, errors="replace")
        except FileNotFoundError:
            text, why = "", "cannot be read (FileNotFoundError)"
        if why:
            # NAMED, not suppressed. A row silently dropped here is a memory
            # this ledger stops carrying, and the checker that reads the tree
            # rather than the ledger calls that an ORPHAN — on a store init
            # has just declared correct. `_read_source` names a reason for
            # every adoption-side read failure and this is the same promise.
            notes.append(
                f"  no row: {_display_path(path)} {why}, so SEARCH.md carries "
                "no row for it and the check below will call it an orphan"
            )
            continue
        front = _frontmatter_of(text)
        desc = _scalar_of(front.get("description", ""))
        # A description the checker cannot read produces no row THERE either —
        # it produces `DESC-BAD`. Generating one here would make init's ledger
        # differ from the checker's.
        if desc is not None:
            out[link] = (
                front.get("name") or os.path.splitext(os.path.basename(link))[0],
                link,
                desc,
            )
    hot = _unrowed_hot(store)
    if hot:
        notes.append(
            f"  no row: {len(hot)} {'memory' if len(hot) == 1 else 'memories'} "
            f"under {_display_path(os.path.join(store, 'hot'))} "
            f"({', '.join(_clean(link) for link in hot)}) "
            f"{'has' if len(hot) == 1 else 'have'} no row in "
            "MEMORY.md, which is hand-written and which nothing regenerates — "
            "the check below reports each one as an orphan until you add them"
        )
    return out, notes


def _unrowed_hot(store: str) -> list:
    """The hot memories MEMORY.md does not row, store-relative and sorted.

    Read through `_LINK_RE` against the ledger's own directory, which is how
    the checker decides what a ledger rows — so a store whose hot index is
    complete produces nothing here whatever init would have written.
    """
    ledger = os.path.join(store, "MEMORY.md")
    held, _readable = _held_text(ledger)
    rowed = set()
    for link in _LINK_RE.findall(held or ""):
        if "://" in link:
            continue
        rowed.add(
            os.path.relpath(os.path.realpath(os.path.join(store, link)),
                            os.path.realpath(store))
        )
    return [
        link
        for link in _memories_under(store, "hot")
        if os.path.relpath(os.path.realpath(os.path.join(store, link)),
                           os.path.realpath(store)) not in rowed
    ]


def _search_ledger_text(store: str, entries: list) -> str:
    """SEARCH.md over this whole store, in the form the checker generates.

    THE PREAMBLE IS THE EXISTING FILE'S, verbatim through `## Index`, because
    that is what `--write` would keep: a store whose ledger carries a sentence
    somebody wrote must not have it replaced by a setup command. With no file
    there, the text init would have written stands in for it, which is what
    makes a second init on a fresh store find its own ledger already correct.
    """
    old = _search_ledger(store)
    # Through the same reader the destinations go through, so a ledger this
    # process cannot decode falls back to the default preamble rather than
    # raising out of `--dry-run`. That is today's behaviour — the file was
    # clobbered unconditionally before this — and not a new refusal.
    held, _readable = _held_text(os.path.join(store, "SEARCH.md"))
    if held is not None:
        old = held
    head, sep, _rest = old.partition(_INDEX_HEADING)
    preamble = (head + sep) if sep else old.rstrip("\n") + f"\n\n{_INDEX_HEADING}"
    # LABEL, THEN THE PATH THE ROW POINTS AT. Both sides key on the label
    # alone and Python's sort is stable, so two rows carrying the same label
    # fall out in whatever order each side happened to build its list: the
    # checker's is `_live()`, every memory path in order, and init's is the
    # inventory's — most memories first. The store then fails LEDGER-DRIFT on
    # a ledger init has just written. The link settles it, split into
    # components because that is how the checker's own list is ordered: it
    # sorts `Path` objects, which compare a directory name against a directory
    # name and never against the separator after it.
    body = "\n".join(
        f"- [{label}]({link}) — {desc}"
        for label, link, desc in sorted(
            entries, key=lambda row: (row[0].lower(), row[1].split(os.sep))
        )
    )
    return f"{preamble}\n\n{body}\n"


def _fenced(text: str) -> tuple:
    """`text`'s lines with fenced code blanked, and the line a fence opens on
    and is never closed — 0 when every one of them is closed.

    RESTATED RATHER THAN IMPORTED, for the reason `_rows_pointing_nowhere` is
    restated: `memory_integrity` exits at import below 3.12 and this module
    answers to the 3.9 floor the dispatcher runs on. Two rules carry it — a
    fence opens on three or more backticks or tildes indented at most three
    columns, and closes only on the same character, at least as long, and
    carrying no info string — and a 3.12 case runs the real checker over a
    store this rule passed, so a restatement that drifts fails there.

    BLANKED AND NOT DROPPED, because a link the checker never reads is a link
    adoption may not refuse a memory for. An unterminated fence is an error the
    checker raises; a link inside a closed one is a quoted example it masks, and
    a note about shell commands is where both of them live.
    """
    out: list = []
    fence = ""
    opened = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        match = _FENCE_RE.match(line)
        if fence:
            out.append("")
            if (
                match
                and match.group(1)[0] == fence[0]
                and len(match.group(1)) >= len(fence)
                and not line.strip().strip(fence[0])
            ):
                fence = ""
            continue
        if match:
            fence, opened = match.group(1), lineno
            out.append("")
            continue
        out.append(line)
    return out, (opened if fence else 0)


def _would_wedge_the_store(
    text: str, dest: str, desc: str, store: str, landing: set
) -> str:
    """Why the check init runs over its own work would go red on this copy, or "".

    WHAT ADOPTION LANDS HAS TO PASS THE CHECK ADOPTION THEN RUNS. The rule was
    asked only of files whose NAME is a ledger name, and the copy it protects
    happens in every branch: an ordinary memory carrying a relative link went in
    unread, the confirm exited 6 on a store it had just built, and no re-run
    cleared it — `memory-integrity --write` regenerates the same row, and the
    next dry-run says there is nothing left to do.

    THE DESCRIPTION IS READ FROM SOMEWHERE ELSE THAN THE MEMORY IS. It is lifted
    verbatim into a row in SEARCH.md at the store ROOT, so a link in it resolves
    from there and not from beside the file — which is why a description linking
    a sibling that IS adopted alongside still leaves the store red. Asked first
    because it is the specific answer whenever both are true.

    A FENCE IS NOT A LINK QUESTION AT ALL, and it is the cheapest door of the
    three: a memory that ends mid-example — an ordinary shape for a note about
    shell commands — needs no link anywhere in it to wedge the store.

    A REASON, NOT A BOOLEAN, because the answer is a line the adopter reads
    before consenting: what is skipped and why is the whole difference between
    this and the exit 6 it replaces.
    """
    masked, opened = _fenced(text)
    if opened:
        return (
            f"it opens a code fence on line {opened} that is never closed, so "
            "every line below it is masked out of the link and citation checks "
            "the store is read by"
        )
    if desc:
        nowhere = _rows_pointing_nowhere(
            desc, os.path.join(store, "SEARCH.md"), store, landing
        )
        if nowhere:
            return (
                f"its description carries a link to {', '.join(nowhere)}, and "
                "the ledger row generated from it is read from the store root, "
                "where that resolves to no file"
            )
    unresolved = _rows_pointing_nowhere(
        "\n".join(masked), dest, store, landing
    )
    if unresolved:
        return (
            f"{', '.join(unresolved)} "
            f"{'points' if len(unresolved) == 1 else 'point'} at no file this "
            "store is getting"
        )
    return ""


def _adoptable(machine: Machine, store: str, project) -> bool:
    """Whether adoption will copy out of this project directory at all.

    A LINK IS SOMEBODY'S ANSWER ONLY WHERE IT LANDS IN A STORE. A memory
    directory wired into a store — which is what docs/STORE.md tells an
    adopter to do — holds memories that store already has, and copying through
    it would duplicate every one of them. A link that lands anywhere else has
    answered nothing: those memories are outside every store exactly as an
    unlinked directory's are.

    EVERY ANSWER `_store_relation` HAS, not the one spelling of it. That
    function distinguishes four overlaps, and `docs/STORE.md` prescribes `at`
    — a link to the corpus root itself — so a test for `inside` alone walked
    back in through the documented wiring and copied the store's whole corpus,
    canary included, under a project key.

    AND THE STORE THIS RUN IS WRITING, ASKED OF THE STORE ROOT. `_store_relation`
    reads the config, which on the run that creates one does not name this store
    yet, and it only ever measures against the corpus root, so a tier beside it
    is no relation at all. Both gaps land the same way — a directory the adopter
    already pointed into the store, reported as outside every store and copied
    back into it — and neither test closes the other.

    DOCTOR'S OWN PREDICATE, term for term, because the claim that the two
    commands report one number is only worth making if one function decides
    it. The three link flags asked alone made doctor count a memory directory
    symlinked outside every store and init call it "already redirected,
    skipped" — reported as handled, so never adopted and never chased.
    """
    return not (
        project.linked
        and (
            _within(project.path, store)
            or _store_relation(machine, project.path)[2]
        )
    )


def _entry_kind(path: str) -> str:
    """What a directory entry IS, asked without following or opening it.

    `os.lstat` and nothing else: the entries this is asked about are the ones
    the walk could not classify, and a `stat` through an unresolved symlink or
    an `open` of a named pipe is how a disclosure comes to hang where the thing
    it discloses only returned early.
    """
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return "an entry this scan cannot stat"
    if stat.S_ISLNK(mode):
        return "a symlink resolving to no file"
    if stat.S_ISDIR(mode):
        return "a directory"
    if stat.S_ISFIFO(mode):
        return "a named pipe"
    if not stat.S_ISREG(mode):
        return "not a regular file"
    return "not a file the walk could list"


def _memory_the_walk_did_not_list(known: list) -> list:
    """Every harness memory entry the inventory dropped, one line each.

    THE WALK IS A DIAGNOSTIC AND IT DROPS WHAT IT CANNOT STAT. An entry that
    raises — a symlink loop, a chain past the kernel's limit — and an entry
    that is merely not a regular file are both dropped on their own, and a
    memory directory whose whole listing raises is dropped with everything in
    it. None of them leaves a trace: not in the count, not in the copy plan,
    and not in any line of the manifest, so an adopter reads "0 memories
    outside every store" about a directory holding two.

    THE SAME QUESTION ASKED FROM HERE, because the walk cannot answer it: what
    it returns is what it managed to read, and the gap is only visible against
    the directory itself. It is a read and stays one — names off `scandir`,
    `os.lstat` for what an entry is, nothing followed and nothing opened — so
    the disclosure cannot hang where the thing it discloses returned early.

    A directory holding only `MEMORY.md` is not a gap: the walk drops it on
    purpose, because the index is not a memory anyone is missing.
    """
    out: list = []
    listed = {project.key: set(project.files) for project in known}
    base = os.path.join(_harness_config_dir(), "projects")
    try:
        with os.scandir(base) as entries:
            projects = sorted((entry.name, entry.path) for entry in entries)
    except OSError as exc:
        # NOT THERE AND CANNOT BE READ ARE DIFFERENT ANSWERS. A machine whose
        # harness has never run has no `projects/` and nothing was dropped;
        # every other errno means this scan does not know what it is missing,
        # and the walk it reconciles against returns the same empty inventory
        # for both.
        if exc.errno != errno.ENOENT:
            out.append(
                "the harness project directory could not be listed "
                f"({exc.strerror or exc}), so whether it holds memories this "
                "run passed over is unknown "
                f"({_display_path(base)})"
            )
        return out
    for key, path in projects:
        memory = os.path.join(path, "memory")
        try:
            with os.scandir(memory) as entries:
                names = sorted(
                    entry.name
                    for entry in entries
                    if entry.name.endswith(".md")
                )
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                out.append(
                    f"{_findable(key)}: this directory could not be listed "
                    f"({exc.strerror or exc}), so what it holds is not adopted, "
                    "counted or named anywhere else here "
                    f"({_display_path(memory)})"
                )
            continue
        if key not in listed:
            if not any(name != harness_memory.INDEX_NAME for name in names):
                continue
            out.append(
                f"{_findable(key)}: the walk returned none of the "
                f"{len(names)} `.md` "
                f"{'entry' if len(names) == 1 else 'entries'} this directory "
                "holds, so nothing in it is adopted, counted or named "
                "anywhere else here — the walk keeps an entry only where it "
                "can stat one as a regular file "
                f"({_display_path(memory)})"
            )
            continue
        out.extend(
            f"{_findable(key)}: `{_clean(name)}` is "
            f"{_entry_kind(os.path.join(memory, name))}, so it is not adopted "
            "or counted"
            for name in names
            if name not in listed[key]
        )
    return out


def _auto_memory_notes(machine: Machine, store: str, known: list) -> list:
    """What the harness has written, said in every manifest.

    THE INVENTORY IS THE CALLER'S, and shared with the adoption planner: this
    sentence is the reader's check on the copy plan below it, and two scans of
    one tree is how a note saying 18 directories comes to sit above a manifest
    that copied 19.

    Unconditional, because the state it describes is the one an adopter cannot
    see: two memory systems on one machine, one of them writing where nothing
    retrieves. A flag they never heard of is not an answer to that.
    """
    mine = [project for project in known if _adoptable(machine, store, project)]
    memories = sum(project.memories for project in mine)
    out = []
    if mine:
        out.append(
            f"{len(mine)} project memory "
            f"{'directory holds' if len(mine) == 1 else 'directories hold'} "
            f"{memories} {'memory' if memories == 1 else 'memories'} outside "
            "every store — --adopt-auto-memory copies them into "
            + _display_path(os.path.join(store, "search", ADOPT_DIRNAME))
            + "/<project key>/ and redirects new ones there; "
            "docs/STORE.md#where-your-agents-own-memories-land has the "
            "by-hand version."
        )
    else:
        out.append("No harness auto-memory to adopt.")
    out.extend(
        f"{_findable(project.key)}: already redirected, skipped "
        f"({_display_path(project.path)})"
        for project in known
        if not _adoptable(machine, store, project)
    )
    out.extend(_memory_the_walk_did_not_list(known))
    # THE LIMIT IS ON DERIVING A KEY, NOT ON READING ONE. `KEY_MAX` refuses
    # inside `project_key`, which adoption never calls: adoption reads names
    # the harness already chose, and a directory the harness itself wrote over
    # the limit is a real one holding real memories, so refusing it would
    # leave them behind. What the adopter cannot see is that such a name is a
    # truncation with a hash after it, so the directory landing in the store
    # is not the project path — and the hash is not one this package measures.
    hashed = [
        project for project in mine
        if len(project.key) > harness_memory.KEY_MAX
    ]
    out.extend(
        f"{_findable(project.key)}: over {harness_memory.KEY_MAX} characters, "
        "so the harness truncated the project path and hashed the rest — this "
        "directory name does not spell the project it belongs to, and memkit "
        "cannot say which one it is"
        for project in hashed
    )
    return out


def _as_spelled(base: str, entries: list, target: str) -> str | None:
    """The name `base` really holds `target` under, or None if it holds none.

    ONE RULE FOR EVERY LEVEL A SPELLING IS DECIDED AT: the project key and the
    file name are the two halves of the path a row points at, and both are
    asked here.

    `os.path.realpath` follows links and does not canonicalise CASE, so on a
    case-insensitive filesystem — APFS, the default on macOS — a path opens an
    entry whose name is spelled some other way and every comparison between the
    two still says they are the same path. The ledger row has to
    carry the name the checker enumerates off disk, so the name is asked of
    the OS by identity rather than derived from a rule about which spellings a
    given filesystem folds together — which is a property of the mount and not
    of the string.
    """
    for name in entries:
        with contextlib.suppress(OSError):
            if os.path.samefile(os.path.join(base, name), target):
                return name
    return None


def _name_fits(name: str) -> bool:
    """Whether `name` still fits one directory entry once the write extends it.

    ONE RULE, ASKED TWICE. The plan approves a name and the write creates a
    longer one, so a rule stated only at the plan approves names that cannot
    land: a 250-byte memory name passed the dry-run, failed `os.open` on the
    temporary at apply time, and left a store the checker called broken with
    nothing copied into it and every re-run doing the same.

    BYTES, not characters. A name is bytes to the kernel, and a 130-character
    name of two-byte characters is a 260-byte entry that a character count says
    is comfortably short.
    """
    return len(_utf8(name)) + _TMP_SUFFIX_BYTES <= _NAME_MAX_BYTES


def _checker_link(dest: str) -> str:
    """A row's destination as the integrity check's link parser reads it.

    RESTATED RATHER THAN IMPORTED, for the reason `_rows_pointing_nowhere` is
    restated: `memory_integrity` exits at import below 3.12 and this module
    answers to the 3.9 floor the dispatcher runs on. Two rules carry it, both
    read off that parser — `_dest` takes the first whitespace token and strips
    `<>`, and `_link_path` then CUTS THE DESTINATION AT THE FIRST `#`, since
    everything after one is an anchor into a document rather than part of the
    path. A 3.12 case runs the real checker over a store this rule passed, so
    a restatement that drifts fails there.

    WHAT IT IS FOR is the comparison, not the value: a destination that comes
    back unchanged is one the check will look for where the row put it, and a
    destination that comes back SHORTER is a row pointing at a path nothing is
    at. `#` is the only character that reaches that second answer through a
    generated row — the link half is always `search/<key>/<name>`, so no
    leading segment can read as a URL scheme and the `<>` strip never bites —
    which is why the skip lines that use this name it.
    """
    raw = dest.strip()
    if not raw:
        return ""
    return raw.split()[0].strip("<>").split("#", 1)[0].strip()


def _rows_pointing_nowhere(text: str, dest: str, store: str, landing: set) -> list:
    """The destinations in `text` that resolve to no file, read from `dest`.

    WHAT AN INDEX PROMISES, not what the walk happened to see. A row for a
    memory the adopter deleted by hand names a file the inventory never
    enumerated, so a rule computed from the inventory's leftovers is satisfied
    by it vacuously and the index is copied carrying a row for nothing.

    RESOLVES MEANS WHAT THE CHECKER MEANS BY IT, restated rather than called:
    `memory_integrity` exits at import below 3.12 and this module answers to
    the 3.9 floor the dispatcher runs on, so its `_link_path` cannot be
    imported here. A 3.12 case runs the checker's own link check over a store
    this rule passed, so a restatement that drifts fails there.

    Two deliberate differences from that rule, both toward refusing to copy:
    code fences and inline spans are not masked, so a link quoted as an
    example counts, and a link is answered against the DESTINATION rather than
    the source — a file landing in this run resolves, one only in the source
    directory does not.
    """
    out: list = []
    root = os.path.realpath(store)
    here = os.path.dirname(dest)
    for raw in _MD_LINK_RE.findall(text):
        if not raw.strip():
            continue
        target = _checker_link(raw)
        if not target or _SCHEME_RE.match(target):
            continue
        if "/" not in target and not target.lower().endswith(_PATH_SUFFIXES):
            continue
        path = os.path.expanduser(target)
        if os.path.isabs(path):
            path = os.path.realpath(path)
            rel = os.path.relpath(path, root)
            if rel == os.pardir or rel.startswith(os.pardir + os.sep):
                continue
        else:
            path = os.path.normpath(os.path.join(here, path))
        if path in landing or os.path.exists(path):
            continue
        shown = _clean(target)
        if shown not in out:
            out.append(shown)
    return out


def _plan_adoption(machine: Machine, store: str, known: list) -> tuple:
    """(actions, ledger rows, notes) for every harness memory this would adopt.

    COPY, NEVER MOVE, AND NEVER OVERWRITE. A destination that already holds
    the same bytes is redundant and the action disappears from the manifest; a
    destination holding anything else is `diverged` — named, left exactly as it
    is, and no reason to refuse the rest. The originals are not touched on any
    path, so the worst outcome of a wrong guess here is a file to delete.

    AND NEVER ANYWHERE BUT THE PATH IT NAMES. A destination is a path until
    something writes to it, and then it is that path with every link in it
    followed: a dangling symlink at one reads as absent through `state_token`,
    so the copy was planned, and the write created the link's target directory
    and put a memory in it. What this plans, it can prove lands where the
    manifest and the ledger row both say it does — which is a stronger claim
    than landing inside the store, and the one the checker measures.
    """
    base = os.path.join(store, "search", ADOPT_DIRNAME)
    # What that directory already holds, read once: planning writes nothing,
    # so it does not change under the loop below.
    entries: list = []
    with contextlib.suppress(OSError):
        entries = os.listdir(base)
    actions: list = []
    rows: list = []
    skipped: list = []
    diverged: list = []
    normalised = 0
    already = 0
    payload = 0
    directories = 0
    for project in known:
        if not _adoptable(machine, store, project):
            continue
        # THE KEY IS THE OTHER HALF OF THE PATH THE ROW POINTS AT, and it
        # gets the test the file name gets one screen below, for the same
        # reason. It is a directory entry name read off disk and never
        # re-derived, so it holds whatever the adopter's disk holds, and
        # `relpath(dest, store)` writes it between the `(` and `)` of a
        # generated row: a `)` in it ends the link early, a space ends it at
        # the space, and a newline ends the manifest line above it.
        if _link_target(project.key) != project.key:
            skipped.append(
                f"{_findable(project.key)}: the project key holds a character "
                "no manifest line and no ledger row could carry — a link "
                "ends at the first `)`, at a space, or at a newline"
            )
            continue
        # AND THE CHARACTER THAT PASSES THAT TEST AND STILL LOSES THE FILE.
        # `#` is printable, is not link syntax and is not whitespace, so the
        # rule above keeps it — and the integrity check's own link parser
        # reads a destination only as far as the first one. A key holding one
        # is copied, rowed, and then read as a path that stops before it: the
        # check init runs over its own work goes red, and no re-run repairs a
        # row memkit generated from a name that is still on disk.
        if _checker_link(project.key) != project.key:
            skipped.append(
                f"{_findable(project.key)}: the project key holds a `#`, and "
                "the integrity check reads a link destination only as far as "
                "the first one — the row this run would write would point at "
                "the path before it, which is no file"
            )
            continue
        # AND IT MUST NOT BE A NAME THE CHECK READS AS A MEMORY. A key is a
        # DIRECTORY entry created under `search/`, and the checker enumerates
        # memories by suffix off the whole subtree — `rglob("*.md")`, which
        # matches a directory as readily as a file. A key ending in `.md`
        # therefore lands as a directory every rule then opens as one: the
        # check init runs over its own work dies `IsADirectoryError` rather
        # than reporting anything, `--write` opens it too, and no re-run
        # clears it. The suffix is spelled exactly as the glob spells it, so
        # this refuses no key the check would have been happy with.
        if project.key.endswith(".md"):
            skipped.append(
                f"{_findable(project.key)}: the project key ends in `.md`, so "
                "the directory a copy would create under this store is a name "
                "the integrity check enumerates as a memory and opens as a "
                "file"
            )
            continue
        target = os.path.join(base, project.key)
        # ONCE PER PROJECT, AND ON THE RESOLVED PATH. The link that moves a
        # write is as often a directory halfway up as the leaf, and
        # `os.makedirs` follows one as happily as `open` does — so a linked
        # `projects/` directory sends every copy under it somewhere the
        # manifest does not name, without a single leaf being a link.
        #
        # WHERE IT LANDS AGAINST WHERE IT IS NAMED, and not containment. The
        # ledger row is `relpath(dest, store)`, spelled lexically, while the
        # bytes go to the resolved path the checker enumerates — so a link
        # that stays INSIDE the store still moves the write off its own row,
        # and init and `memory-integrity --write` then rewrite each other's
        # answer every run. Containment falls out of the comparison: a key is
        # one directory entry name, so the relative path can never climb out.
        #
        # The name is built on the STORE'S OWN realpath rather than on
        # `abspath(target)`, because a store reached through a link — an
        # external volume, a dotfiles tree — resolves every destination in it
        # somewhere else and still lands each one exactly where its row says.
        named = os.path.join(
            os.path.realpath(store), os.path.relpath(target, store)
        )
        if os.path.realpath(target) != named:
            diverged.append(
                f"{_display_path(target)} resolves to "
                f"{_display_path(_terminal_realpath(target))} — every copy "
                "into it would land at a path this manifest does not name, so "
                "nothing was written for this project"
            )
            continue
        # AND SPELLED THE WAY THE DISK SPELLS IT. The comparison above is made
        # of path strings, and a case-insensitive filesystem hands the same
        # directory to two of them: a key `-home-U` opens the `-home-u` that is
        # already there and every path test still passes. The bytes then land
        # in the directory that exists and the row names the one that does not
        # — the checker calls it an orphan, `memory-integrity --write` repairs
        # the row to the on-disk name, and the next init writes the key's own
        # spelling back, which is the oscillation the guard above exists to
        # stop, reached without a single link.
        spelled = _as_spelled(base, entries, target)
        if spelled is not None and spelled != project.key:
            diverged.append(
                f"{_display_path(target)} is the directory this store already "
                f"holds as `{_clean(spelled)}` — this filesystem does not tell "
                "the two spellings apart, so every copy would land in that one "
                "while the row named this one, and nothing was written for "
                "this project"
            )
            continue
        # THE SAME RULE, ONE LEVEL DOWN. The key is half the path a row points
        # at and the file name is the other half, so the spelling has to be
        # asked of the disk at both — a memory renamed `alpha.md` -> `Alpha.md`
        # in the harness opens the `alpha.md` this store already holds, reads
        # as "already adopted", and the row generated for it names a spelling
        # the disk does not carry. Read once, before the loop: planning writes
        # nothing, so it does not change underneath.
        held_entries: list = []
        with contextlib.suppress(OSError):
            held_entries = os.listdir(target)
        # RESOLUTION DISCLOSED AT BOTH ENDS OF A COPY. `_path_detail` says
        # where a destination really lands; the source is the same question
        # asked of the bytes being read, and a `memory` directory symlinked
        # elsewhere is supported rather than refused — so the group line,
        # which is the only line naming the source, has to say so.
        source = _display_path(project.path)
        resolved_source = _terminal_realpath(project.path)
        if resolved_source != os.path.abspath(project.path):
            source += f" (resolves to {_display_path(resolved_source)})"
        group = f"from {source} -> {_display_path(target)}{os.sep}"
        mine: list = []
        # THE NAMES THIS STORE ALREADY HOLDS A DIFFERENT FILE UNDER. Nothing
        # is written for them and they are not in `mine`, but the file a row
        # points at is there — so the index rule below must not count them as
        # files left outside the store. Their own `diverged:` line says what
        # actually happened to each.
        divergent: set = set()
        # WHAT A COPY WOULD COST IF IT IS DROPPED. The rule below runs after the
        # loop and can take a file back out of the plan, so the row it would
        # have written, the line naming it and the counts it moved all have to
        # be reachable from the action rather than from the loop that built it.
        held_back: dict = {}
        # LEDGER NAMES LAST, and stable so nothing else moves. A ledger is
        # copied with no rewriting at all, rows included, so whether it can be
        # copied depends on what the rest of this loop leaves behind — and an
        # answer read before the loop finishes is not that answer.
        for name in sorted(project.files, key=lambda n: n in _LEDGER_NAMES):
            # THROUGH `_clean`, EVERY TIME. A filename is adopter-controlled
            # text that lands in the surface a human reads before typing
            # `--confirm`, and a newline in one forges a whole manifest line.
            # The paths on these lines go through `_display_path` instead,
            # which strips the same characters and keeps the spacing a path
            # needs.
            shown = f"{_clean(project.key)}/{_clean(name)}"
            # AND A NAME NO LINE CAN CARRY IS NOT COPIED AT ALL. Sanitising
            # the note leaves the destination path line, which must keep its
            # spacing byte for byte to name a file that exists — so the only
            # honest answer for a name holding a newline is not to write a
            # line about it. The LINK half of the row is the same argument
            # with no way out at all: a row points at a file by writing its
            # path between `(` and `)`, so a file whose name holds either one
            # is a file no row can point at, whatever the label says.
            # ONE TEST FOR BOTH, AND IT IS THE KEY'S TEST: the name is the
            # other half of the same path a row points at, so it answers to
            # the same rule — `_link_target` drops what is not printable, then
            # the link syntax, then whitespace, and a space in a destination
            # ends the link at the space just as surely as a `)` does.
            if _link_target(name) != name:
                skipped.append(
                    f"{shown}: the file name holds a character no manifest "
                    "line and no ledger row could carry — a link ends at the "
                    "first `)`, at a space, or at a newline"
                )
                continue
            # THE SAME CHARACTER, THE OTHER HALF OF THE SAME PATH. Kept as
            # a clause of its own rather than folded into the rule above,
            # because the two answer to different parsers: that one is the
            # ledger's row SYNTAX, and this one is what the check does with a
            # destination the syntax accepted.
            if _checker_link(name) != name:
                skipped.append(
                    f"{shown}: the file name holds a `#`, and the integrity "
                    "check reads a link destination only as far as the first "
                    "one — the row this run would write would point at the "
                    "path before it, which is no file"
                )
                continue
            if not _name_fits(name):
                skipped.append(
                    f"{shown}: the file name is {len(_utf8(name))} bytes, and "
                    f"the {_TMP_SUFFIX_BYTES} the write adds for the temporary "
                    "it renames from put it past the "
                    f"{_NAME_MAX_BYTES}-byte limit on one directory entry — a "
                    "copy planned for it could not land"
                )
                continue
            if name in project.linked_files:
                skipped.append(
                    f"{shown}: the file is a symlink, so its bytes "
                    "live outside the directory being copied"
                )
                continue
            source = os.path.join(project.path, name)
            text, why = _read_source(source)
            if text is None:
                skipped.append(f"{shown}: {why}")
                continue
            dest = os.path.join(target, name)
            # THE LEAF, the directory above it having been checked once for
            # the whole project. A dangling link reads as absent through
            # `state_token`, so the copy was planned and the write followed
            # it: out of the store where it pointed out, and into `hot/` — an
            # index nothing regenerates — where it pointed back in.
            if os.path.islink(dest):
                divergent.add(name)
                diverged.append(
                    f"{_display_path(dest)} is a symlink to "
                    f"{_display_path(_terminal_realpath(dest))} — a copy would "
                    "write through it, at a path this manifest does not name, "
                    "so nothing was written"
                )
                continue
            spelled_file = _as_spelled(target, held_entries, dest)
            if spelled_file is not None and spelled_file != name:
                divergent.add(name)
                diverged.append(
                    f"{_display_path(dest)} is the file this store already "
                    f"holds as `{_clean(spelled_file)}` — this filesystem does "
                    "not tell the two spellings apart, so the copy would land "
                    "in that one while the row named this one, and nothing was "
                    "written for it"
                )
                continue
            # AN INDEX IS A CLAIM ABOUT THE FILES IT ROWS. A ledger name is
            # copied byte for byte and nothing regenerates it, so a row it
            # carries for a file this store is not getting arrives pointing at
            # nothing — and the integrity check init runs over its own work
            # goes red on adoption's own skip rules, on a store the adopter
            # can only clear by hand-editing a file memkit copied. What the
            # check reads of a memory is asked of every copy below, once the
            # plan is whole; an index is asked here as well, because what an
            # index promises is answered against what the rest of this loop
            # leaves behind.
            #
            # ASKED OF THE ROWS AND OF THE INVENTORY BOTH. The rows are the
            # claim, and they name files the walk never saw; the inventory's
            # leftovers catch what a row shape this rule does not read would
            # have missed. Decided here, after the files it indexes, because
            # what an index promises is answered against what is landing.
            if name in _LEDGER_NAMES:
                landing = {
                    os.path.normpath(action.path)
                    for action in actions + mine
                    if action.op == CREATE_FILE
                }
                unresolved = _rows_pointing_nowhere(text, dest, store, landing)
                copying = {os.path.basename(action.path) for action in mine}
                omitted = [
                    _clean(n) for n in project.files
                    if n not in _LEDGER_NAMES
                    and n not in copying
                    and n not in divergent
                ]
                reasons = []
                if unresolved:
                    reasons.append(
                        f"{', '.join(unresolved)} "
                        f"{'points' if len(unresolved) == 1 else 'point'} at "
                        "no file this store is getting"
                    )
                if omitted:
                    reasons.append(
                        f"{', '.join(omitted)} "
                        f"{'was' if len(omitted) == 1 else 'were'} left behind"
                    )
                if reasons:
                    skipped.append(
                        f"{shown}: it is an index of the directory it came "
                        "from, copied with no rewriting, and "
                        + " and ".join(reasons)
                        + " — so a row it carries would be a row for a file "
                        "that is not there"
                    )
                    continue
            rule = ""
            row = None
            desc = ""
            if name not in _LEDGER_NAMES:
                text, rule = _normalise(text, os.path.splitext(name)[0])
                if _TIER_RE.search(text[:4096]):
                    skipped.append(
                        f"{shown}: carries a `tier:` line, which "
                        "the checker rejects — tier is the directory now"
                    )
                    continue
                front = _frontmatter_of(text)
                desc = _scalar_of(front.get("description", ""))
                # NOT REACHABLE TODAY, and kept: `_normalise` either leaves a
                # description `_scalar_of` already read or writes one through
                # `_as_scalar`, whose output it reads back by construction. It
                # is the guard on that construction rather than on an input.
                if desc is None:
                    skipped.append(
                        f"{shown}: no description this store's "
                        "ledger could carry was derivable from it"
                    )
                    continue
                label = front.get("name") or os.path.splitext(name)[0]
                # FAIL-CLOSED on the label as well. `_relabel` rewrites a name
                # a row cannot carry; what reaches here is the one case it
                # cannot rewrite — a name and a file name that are BOTH
                # nothing but link syntax — and a row built from it would end
                # its own link.
                if _label(label) != label:
                    skipped.append(
                        f"{shown}: no label this store's ledger could carry "
                        "was derivable from it"
                    )
                    continue
                link = os.path.relpath(dest, store)
                # THE PATH THOSE TWO HALVES COMPOSE, ASKED OF THE PARSER THAT
                # WILL READ IT. Each half is tested where it is read, and this
                # is the string a row actually carries — the only thing the
                # check opens. Deliberately belt and braces: a rule that tests
                # the parts and never the whole is one the next part walks
                # past, and what that costs here is a store that fails its own
                # integrity check on a row nothing regenerates.
                if _checker_link(link) != link:
                    skipped.append(
                        f"{shown}: the destination a row would carry is not "
                        "the path the integrity check reads back out of it, "
                        "so the row would point at no file"
                    )
                    continue
                row = (
                    label,
                    link,
                    desc,
                )
            held, readable = _held_text(dest)
            if not readable:
                divergent.add(name)
                diverged.append(
                    f"{_display_path(dest)} exists and cannot be read, so what "
                    f"is there was not compared with {_display_path(source)}"
                )
                continue
            if held is not None and held != text:
                divergent.add(name)
                diverged.append(
                    f"{_display_path(dest)} exists and differs from "
                    f"{_display_path(source)} — left exactly as it is"
                )
                continue
            if held is not None:
                already += 1
            elif rule:
                normalised += 1
            payload += len(_utf8(text))
            mine.append(
                Action(CREATE_FILE, dest, text,
                       note=f"{_clean(name)}: {rule}" if rule else "",
                       group=group, confine=store)
            )
            held_back[dest] = (
                shown, desc, row, len(_utf8(text)),
                "already" if held is not None else ("normalised" if rule else ""),
            )
        # AFTER THE LOOP, AND TO A FIXPOINT. `landing` is what this run is
        # getting, and it is built one file at a time — a link answered inline
        # would be answered against a half-built set, so a memory whose target
        # simply had not been planned yet would be skipped for pointing at a
        # file that does arrive. Dropping one file can strand a link in
        # another, so the question is asked again until nobody moves.
        while mine:
            landing = {
                os.path.normpath(action.path)
                for action in actions + mine
                if action.op == CREATE_FILE
            }
            for action in mine:
                shown, desc, _row, size, counted = held_back[action.path]
                why = _would_wedge_the_store(
                    action.content, action.path, desc, store, landing
                )
                if why:
                    break
            else:
                break
            skipped.append(f"{shown}: {why}")
            mine.remove(action)
            del held_back[action.path]
            payload -= size
            if counted == "already":
                already -= 1
            elif counted == "normalised":
                normalised -= 1
        rows.extend(
            held_back[action.path][2] for action in mine
            if held_back[action.path][2] is not None
        )
        if not mine:
            continue
        if not directories:
            actions.append(
                Action(
                    CREATE_DIR,
                    base,
                    note="one directory per harness project key, inside the "
                    "corpus root so retrieval reaches them and outside the "
                    "directory the harness rewrites.",
                    confine=store,
                )
            )
        directories += 1
        actions.append(Action(CREATE_DIR, target, confine=store))
        actions.extend(mine)
    files = sum(1 for action in actions if action.op == CREATE_FILE)
    notes = [
        f"Adoption: {files} files ({payload} bytes) from {directories} project "
        f"{'directory' if directories == 1 else 'directories'} -> "
        f"{_display_path(base)}{os.sep}<project key>{os.sep}, "
        f"{normalised} normalised, {len(skipped)} skipped, {already} already "
        f"adopted, {len(diverged)} diverged."
    ]
    # THE MODE IS PART OF WHAT IS BEING CONSENTED TO. These are a person's
    # private notes, and a copy of one is as sensitive as the note — so it
    # lands at what memkit writes everything else at rather than at whatever
    # the source happened to carry, and the manifest says so before the copy
    # is made rather than leaving it to a `stat` afterwards.
    if files:
        notes.append(
            "  Each copy lands mode 0600 — readable by you and by nothing "
            "else, whatever the original carried. The originals keep their "
            "own; a destination that already exists keeps its own too."
        )
    # EVERY ONE OF THEM, uncapped. A count an adopter cannot reconcile against
    # their own `ls` is the number this list exists to make checkable, and the
    # diverged lines in particular are the only place a file adoption declined
    # to touch is ever named.
    notes.extend(f"  skipped: {line}" for line in skipped)
    notes.extend(f"  diverged: {line}" for line in diverged)
    return actions, rows, notes


def _home_form(path: str) -> str:
    """`path` written the way a settings file should carry it.

    `~/`-prefixed under `$HOME` because the harness expands exactly that, and
    a settings file that survives a moved home directory is worth the two
    characters. `$HOME` rather than the password database, for the reason
    `expand_home` reads it: the value has to mean the same thing to the
    process that wrote it and the shell that resolved it.
    """
    home = os.environ.get("HOME", "")
    if home and path.startswith(home + os.sep):
        return "~/" + path[len(home) + 1 :]
    return path


def _redirect_dir(store: str) -> str:
    """Where `--adopt-auto-memory` points the harness, resolved.

    A DIRECTORY OF THE HARNESS'S OWN inside the corpus root, never the corpus
    root itself. Measured on 2.1.258: every `.md` written or edited under the
    configured directory is re-serialised — name slugified, other top-level
    keys buried under `metadata:` — and the gate is a string prefix on that
    path. Pointed at the corpus root, that is every memory in the store;
    pointed here, it is only what the harness itself wrote.
    """
    # The harness NFC-normalizes the setting before it uses it; recorded in
    # any other spelling, the directory it writes to is a different one on a
    # normalization-sensitive filesystem.
    return unicodedata.normalize(
        "NFC", os.path.join(store, "search", harness_memory.SAFE_SUBDIR)
    )


def build_plan(
    machine: Machine,
    *,
    store: str | None = None,
    config: str | None = None,
    wire_claude_md: bool = False,
    auto_dream_off: bool = False,
    adopt_auto_memory: bool = False,
    auto_memory_off: bool = False,
    interpreter: str | None = None,
) -> Plan:
    """Everything init would do, computed against the tree as it is now."""
    config_path = _resolve_config(machine, config)
    store_path = expand_home(store or DEFAULT_STORE)
    check_refusals(
        machine,
        config_path=config_path,
        store_path=store_path,
        wire_claude_md=wire_claude_md,
        auto_dream_off=auto_dream_off,
        adopt_auto_memory=adopt_auto_memory,
        auto_memory_off=auto_memory_off,
        interpreter=interpreter,
    )
    # AFTER the refusals and never before: this is the value that goes into the
    # config, and the refusal above is what established it can serve.
    interpreter_path = _chosen_interpreter(interpreter)
    nonce = _canary_nonce(config_path)
    store_id = _store_id(store_path)
    # ONE SCAN, read by the adoption planner and by the note it sits above.
    # It is a `scandir` of every project entry under the harness's config
    # directory, and the two readers have to describe one snapshot: a second
    # scan is how the note and the manifest under it come to disagree.
    # THE WALK'S FAILURE FLAG IS NOT READ HERE, and that is a choice rather
    # than a drop: `inventory` reports that something would not answer and
    # names the first such path, while `_memory_the_walk_did_not_list` asks the
    # same tree again and reports every one of them WITH ITS ERRNO, which is
    # what a manifest owes an adopter. Read here too, one unreadable directory
    # would take two lines, one of them the vaguer.
    known = harness_memory.inventory(_harness_config_dir())[0]
    # WHOSE STORES THE MEMBERSHIP QUESTION IS ABOUT: the config this run is
    # writing, which is not always the one the session resolved. `--config`,
    # an install option and the plugin's second rung all name a config the
    # environment does not, and asked of the session's the predicate answers
    # "outside every store" about a directory symlinked INTO the store being
    # written — so adoption follows the link and lands a second copy of every
    # memory under a second project key, on a store that then fails its own
    # check.
    membership = (
        machine
        if machine.resolved_config == config_path
        else Machine(config_path)
    )
    adopted: list = []
    rows: list = []
    adoption_notes: list = []
    if adopt_auto_memory:
        adopted, rows, adoption_notes = _plan_adoption(
            membership, store_path, known
        )
    canary_link = os.path.join("search", CANARY_NAME)
    # THE WHOLE STORE'S ROWS, not just the ones this run writes. A file already
    # under `search/` owes SEARCH.md a row whoever put it there, and the
    # planned writes win over what is on disk because they are what will be
    # there when the checker runs. A `diverged` destination contributes no
    # planned row and keeps the one its own text produces.
    ledger_rows, ledger_notes = _rows_on_disk(store_path, config_path)
    ledger_rows[canary_link] = (
        "memkit-canary", canary_link, _canary_description(nonce),
    )
    # THROUGH THE SUB-INDEX EXCLUSION `_rows_on_disk` APPLIES, because these
    # are rows for the same store and a declared sub-index owns its members'
    # rows outright. Merged in raw, a row the adopter had moved into one came
    # back into SEARCH.md on the next run and the store failed the check init
    # runs on its own work with DOUBLE-LEDGER — the one path that writes new
    # rows being the one that skipped the guard beside it.
    claimed = _sub_index_members(store_path, config_path)
    for row in rows:
        if row[1] in claimed:
            continue
        ledger_rows[row[1]] = row
    actions = [
        Action(
            CREATE_DIR,
            machine.state_dir,
            note="the shared derived-state directory, mode 0700. An install "
            "nobody has configured never creates this; init is the thing that "
            "asked.",
        ),
        Action(CREATE_DIR, os.path.dirname(config_path)),
        Action(
            MERGE_CONFIG,
            config_path,
            _merge_config(
                _read_or_empty(config_path),
                nonce=nonce,
                interpreter=interpreter_path,
                entries=_config_entries(store=store_path, store_id=store_id),
                where=config_path,
            ),
            note=f"adds root and store {store_id!r}; records interpreter "
            f"{_display_path(interpreter_path)} and canary nonce {nonce}. "
            "Existing stores are kept. "
            + _config_route_note(machine, config_path),
            authored_config=True,
            payload={
                "nonce": nonce,
                "interpreter": interpreter_path,
                "entries": _config_entries(store=store_path, store_id=store_id),
            },
        ),
        # THE ROOT IS THE CONTAINMENT ROOT, so it is the one action under the
        # store that carries no `confine`: `_refuse_escape` judges the landing
        # place against `join(realpath(confine), relpath(path, confine))`, and
        # for a path that IS the root that name ends in `/.` and never equals
        # its own realpath — every run would refuse. Confining it to its PARENT
        # answers identically on a symlinked root, on a fresh root under a
        # symlinked parent, on a plain root and on a deep root whose parents do
        # not exist; what it would cost is `--store <path>/`, where the trailing
        # slash makes the relative path `.` and the same comparison refuses a
        # spelling of their own store the adopter is entitled to.
        Action(CREATE_DIR, store_path),
        # search/ FIRST, and the order in this list is the order they are made.
        # The trap init exists to prevent is a flat store that grows a `search/`
        # later: the moment that directory appears, every memory above it stops
        # being retrieved, silently, with every diagnostic still green.
        #
        # EVERY WRITE UNDER THE STORE ANSWERS TO THE SAME ROOT, whoever
        # authored the action. Adoption's copies carried `confine` and the
        # store's own writes did not, so a `search/` swapped for a link
        # between the two turns sent init's own files somewhere no manifest
        # line names — under a guard that was already there.
        Action(
            CREATE_DIR,
            os.path.join(store_path, "search"),
            note="memories live here. A store without it retrieves from its "
            "root, and gaining one later un-retrieves everything above it.",
            confine=store_path,
        ),
        Action(
            CREATE_DIR,
            os.path.join(store_path, "hot"),
            note="memories that load into every session, and which the hook "
            "never points at because they are already in context.",
            confine=store_path,
        ),
        Action(
            CREATE_FILE,
            os.path.join(store_path, "MEMORY.md"),
            _memory_ledger(store_path),
            confine=store_path,
        ),
        Action(
            CREATE_FILE,
            os.path.join(store_path, "search", CANARY_NAME),
            _canary_body(nonce),
            note="one memory, so the store answers something on the first "
            "prompt and doctor has a fixed query that can only match this file.",
            confine=store_path,
        ),
        # BEFORE the verification, and that is the whole reason they are in
        # this list rather than appended past it the way the settings writes
        # are: VERIFY runs the integrity checker over the finished store, and
        # a copied memory whose ledger row landed after the check would be an
        # orphan the check could not have seen.
        *adopted,
        Action(
            CREATE_FILE,
            os.path.join(store_path, "SEARCH.md"),
            _search_ledger_text(store_path, list(ledger_rows.values())),
            note="generated from the frontmatter of every memory under "
            "search/, in the form the integrity checker generates.",
            confine=store_path,
        ),
        Action(
            VERIFY,
            store_path,
            note="run the integrity checker over the finished store. Not with "
            "--write: a regeneration would repair whatever init got wrong and "
            "then report success.",
        ),
    ]
    # THE INVENTORY LINE IS UNCONDITIONAL and the adoption lines are not.
    # What the harness has already written is a fact about this machine an
    # adopter cannot see from inside memkit, and a flag they have never heard
    # of is not an answer to it.
    notes = (
        _auto_memory_notes(membership, store_path, known)
        + adoption_notes
        + ledger_notes
        + _stranded_temporaries(store_path)
    )
    if wire_claude_md:
        target = _claude_md(machine)
        # FileNotFoundError is the create case and takes the empty default;
        # anything else is a file that is there and cannot be seen. The
        # manifest calls this operation `append-line` and its note says
        # "appends", so substituting an empty string for a read failure made
        # the effect a truncation of the adopter's own instructions under a
        # consent given for an append.
        try:
            with open(target, encoding="utf-8") as f:
                existing = f.read()
        except FileNotFoundError:
            existing = ""
        except (OSError, ValueError) as exc:
            raise Refusal(
                "unreadable-claude-md",
                f"{_display_path(target)} exists and could not be read "
                f"({exc}). init appends to that file and will not replace one "
                "it cannot see.",
            ) from exc
        line = _import_line(store_path)
        # CONVERGE, do not duplicate — and the convergence lives in `_appended`
        # alone, which is also what the apply path re-derives through. A second
        # guard here would make the plan and the write disagree about when the
        # line is already present, which is the one question they both answer.
        actions.append(
            Action(
                APPEND_LINE,
                target,
                _appended(existing, line),
                note=f"appends {line}",
                payload={"line": line},
            )
        )
        notes.append(
            "The @-import puts each HOT memory's description in every session "
            "— one line per memory, from MEMORY.md — and not its body. The "
            "bodies stay files to open. That is narrower than it sounds and it "
            "is why this is behind a flag."
        )
        if _git_tracked(target):
            notes.append(
                f"WARNING: {_display_path(target)} is tracked by git. This "
                "adds a line you will be asked to commit."
            )
    if auto_dream_off:
        target = _settings_path(machine)
        actions.append(
            Action(
                SETTINGS_WRITE,
                target,
                _settings_with_auto_dream_off(target),
                # "no other KEY", not "no other byte": the file is re-serialised
                # from its own parse, so a settings.json indented some other way
                # comes back at two spaces. That converges after one write and
                # changes nothing a reader of the file means by it, but a
                # manifest is a promise about a write and this is what the write
                # does.
                note=(
                    'sets "autoDreamEnabled": false and changes no other key. '
                    "The file is rewritten from its own parse, so its "
                    "indentation becomes two spaces"
                ),
                payload={"autoDreamEnabled": False},
            )
        )
        notes.append(
            "Turning auto-dream off stops BACKGROUND CONSOLIDATION and "
            "nothing else: the harness goes on writing its own memories "
            "beside memkit's. --auto-memory-off is the flag that stops the "
            "writing, and --adopt-auto-memory is the one that moves what is "
            "written into the store."
        )
    if adopt_auto_memory:
        target = _settings_path(machine)
        redirect = _home_form(_redirect_dir(store_path))
        actions.append(
            Action(
                SETTINGS_WRITE,
                target,
                _settings_with(target, {harness_memory.DIRECTORY_KEY: redirect}),
                note=(
                    f'sets "{harness_memory.DIRECTORY_KEY}": "{redirect}" and '
                    "changes no other key. From then on EVERY project's new "
                    "memories land flat in that one directory, and the "
                    "MEMORY.md the harness keeps beside them is loaded into "
                    "every session. The file is rewritten from its own parse, "
                    "so its indentation becomes two spaces"
                ),
                payload={harness_memory.DIRECTORY_KEY: redirect},
            )
        )
        notes.append(
            "Sessions already running keep the directory they started with "
            "until they are restarted, so run these two turns again "
            "afterwards to sweep up whatever they wrote in between — a second "
            "run copies only what is new."
        )
        notes.append(
            "The originals are left exactly where they are: adoption copies "
            "and never moves. Deleting what it copied is yours to do, once "
            "you have looked at what landed."
        )
        notes.append(
            "What the harness writes there LATER has no SEARCH.md row until "
            "the integrity checker generates one (`memory-integrity --write` "
            "on the pip and nix channels). Those files are retrievable "
            "before that — the hook reads the tree, not the ledger — so a "
            "missing row costs a line in an index and not a memory."
        )
        # THE OFF STATE THIS ADOPTS UNDER, said out loud. Copying what the
        # harness already wrote is still worth doing while the feature is off,
        # but the redirect half of the flag then points at a directory nothing
        # will add to, and an adopter who set that boolean in an earlier turn
        # is owed the sentence rather than a refusal they cannot clear.
        by_scope = {scope.scope: scope for scope in machine.settings}
        user_scope = by_scope.get(USER)
        if (
            user_scope is not None
            and user_scope.data.get(harness_memory.ENABLED_KEY) is False
        ):
            notes.append(
                f'Auto-memory is switched off: "{harness_memory.ENABLED_KEY}": '
                "false is set in user settings "
                f"({_display_path(user_scope.path)}). This copies what the "
                "harness wrote BEFORE it was switched off, and the redirect "
                "above takes effect only if you turn it back on — nothing new "
                "lands in that directory while the boolean is false. Setting "
                "it back to true is an edit to that file; memkit has no flag "
                "that does it."
            )
        # AND THE SAME STATE AN ENVIRONMENT VARIABLE PUTS A MACHINE IN, said
        # in the same place. `harness_memory.DISABLE_ENV` turns the feature
        # off ahead of every settings scope, so the redirect above is written
        # for a feature no process carrying that value will exercise — and
        # this one process's environment need not be the one the adopter's
        # sessions run in, which is why the off direction is disclosed here
        # rather than refused in the preflight.
        forced, spelled = harness_memory.env_switch()
        if forced is False:
            notes.append(
                "Auto-memory is switched off by the environment: "
                f"${harness_memory.DISABLE_ENV} is set to {spelled!r}, which "
                "the harness reads ahead of every settings file. This copies "
                "what it wrote while the feature was on; nothing new lands in "
                "the directory above for any session carrying that value. "
                "memkit reads its own environment, which need not be the one "
                "your sessions run in."
            )
    if auto_memory_off:
        target = _settings_path(machine)
        actions.append(
            Action(
                SETTINGS_WRITE,
                target,
                _settings_with(target, {harness_memory.ENABLED_KEY: False}),
                note=(
                    f'sets "{harness_memory.ENABLED_KEY}": false and changes '
                    "no other key. The harness then neither reads nor writes "
                    "auto-memory. The file is rewritten from its own parse, "
                    "so its indentation becomes two spaces"
                ),
                payload={harness_memory.ENABLED_KEY: False},
            )
        )
        notes.append(
            "Turning auto-memory off leaves memkit as the only memory system "
            "on this machine. What the harness has already written stays "
            "where it is — --adopt-auto-memory is what copies it into the "
            "store."
        )
        # WHO ACTUALLY DECIDED IT. A scope ahead of the user one saying false
        # already turns the feature off, so the action line above is true
        # about the world and false about its own cause: the write converges
        # the user scope and changes nothing. Anything OTHER than false up
        # there is a refusal, not a note.
        by_scope = {scope.scope: scope for scope in machine.settings}
        for name in _scopes_outranking_user():
            scope = by_scope.get(name)
            if scope is None:
                continue
            if scope.data.get(harness_memory.ENABLED_KEY) is not False:
                continue
            notes.append(
                f'Auto-memory is already off: "{harness_memory.ENABLED_KEY}": '
                f"false is set in {name} settings "
                f"({_display_path(scope.path)}), which the harness reads "
                "ahead of the user scope. That scope is what decides it; this "
                "write only makes the user scope agree."
            )
            break
    # AFTER the plan is complete and before anything acts on it. Run where the
    # list was still being built, it checked eight of the ten actions — the two
    # the flags add were appended below it — so the one preflight whose job is
    # to see the whole plan saw the part that never varies.
    _refuse_incompatible_types(actions)
    return Plan(actions, notes, store_path)


# The complete set of settings keys init may write, as data.
#
# The rule this enforces is "the plugin never enables itself", and
# `enabledPlugins` is the key that would do it — but the guard is an ALLOWLIST
# rather than a check on that name, because the next key with the same power
# has not been named yet and a denylist only ever catches the ones somebody
# thought of.
SETTINGS_KEYS_INIT_MAY_WRITE = frozenset(
    {"autoDreamEnabled", "autoMemoryDirectory", "autoMemoryEnabled"}
)


def _settings_with(path: str, changes: dict) -> str:
    """The settings file with `changes` applied and everything else left alone.

    Read-modify-write over a file the adopter owns, so a parse failure is a
    refusal rather than a rewrite: the field anti-pattern the prior-art survey
    names is a tool that meets a parse error and replaces the file with a stub,
    taking the whole configuration with it.
    """
    disallowed = sorted(set(changes) - SETTINGS_KEYS_INIT_MAY_WRITE)
    if disallowed:
        raise Refusal(
            "enabled-plugins",
            "init would write " + ", ".join(disallowed) + " into your "
            "settings. The only keys it may write are "
            + ", ".join(sorted(SETTINGS_KEYS_INIT_MAY_WRITE))
            + " — a plugin that enabled itself would be a plugin deciding its "
            "own access.",
        )
    blob: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise Refusal(
                    "unparseable-settings",
                    f"{_display_path(path)} exists and its top level is not a "
                    "JSON object. init will not replace it.",
                )
            blob = loaded
    except FileNotFoundError:
        blob = {}
    except (OSError, ValueError) as exc:
        raise Refusal(
            "unparseable-settings",
            f"{_display_path(path)} exists and could not be read as JSON "
            f"({exc}). init will not replace a settings file it cannot "
            "understand — that is how a whole configuration gets lost.",
        ) from exc
    blob.update(changes)
    # `ensure_ascii=False`, because this is a read-modify-write over a file
    # somebody else wrote and then has to review. Escaping every character
    # outside ASCII changes no value and rewrites every line of their own
    # prose that held one, so the diff for a one-key edit is unreadable.
    return json.dumps(blob, indent=2, ensure_ascii=False) + "\n"


def _settings_with_auto_dream_off(path: str) -> str:
    return _settings_with(path, {"autoDreamEnabled": False})


def _appended(existing: str, line: str) -> str:
    """`existing` with `line` at the end, or unchanged if it is already there."""
    if line in existing.splitlines():
        return existing
    if not existing.strip():
        return line + "\n"
    return existing.rstrip("\n") + "\n" + line + "\n"


# --- the command -------------------------------------------------------------


EXIT_OK = 0
# argparse's and the dispatcher's, and not reassignable.
EXIT_USAGE = 2
# A named refusal, and nothing was written. Its own code rather than 1, because
# 1 already means two things a caller has to tell apart — the wrapper could not
# start, and doctor found problems — and neither of them is "you asked for
# something this will not do". A skill branches on this to relay the reason to
# the person and stop, which is a different move from retrying.
# A subcommand that understood the request and will not do it — a store inside
# the plugin payload, a stale digest, an unparseable settings file. Its own
# number because 1 already carries two meanings a caller has to tell apart (the
# wrapper could not start; doctor found problems), and neither is "you asked
# for something this will not do". The move it calls for is to relay the reason
# and stop, not to retry with different arguments.
#
# `cli.py` re-exports this rather than declaring its own, so the number the
# help table advertises is the number the process returns.
EXIT_REFUSED = 5
# Started and did not finish. Distinct from a refusal because the move is
# different: a refusal wants something changed before re-running, and this
# wants the run repeated — the journal says how far it got and every action
# already done is a no-op the second time.
EXIT_INCOMPLETE = 6


EPILOG = """\
Two turns, and the second one is not the same command with a flag:

  memkit init --dry-run           print a manifest of every path and every
                                  write, plus a digest. Writes nothing.
  memkit init --confirm <digest>  recompute the manifest, refuse if anything
                                  under it changed, re-emit it, then apply.

The digest binds the state of the tree, not the text you read. Pass the same
flags to both calls: a different request produces a different digest.

Exit codes: 0 done (or the manifest printed) / 2 usage error / 5 refused, and
nothing was written — stderr names which refusal, and `interpreter-unusable` is
the one --interpreter returns / 6 started and did not finish; the journal says
how far. Recover with the two turns, not by repeating the last one: what landed
has changed the digest, so re-run --dry-run and confirm the digest THAT
prints.

6 is also what a run that performed every action and then failed its own
integrity check returns. That report prints the checker's output and names
each file it is red on, saying whether this run wrote it — a file this run
did not write is one no re-run will touch."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="print the manifest and the digest; write nothing",
    )
    mode.add_argument(
        "--confirm",
        metavar="DIGEST",
        help="apply exactly the plan that produced DIGEST",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help=f"where the memory store goes (default: {DEFAULT_STORE})",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="where the config goes (default: the memkitConfig install option, "
        f"else {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--interpreter",
        metavar="PATH",
        help="the absolute python to record as the one that runs the hook "
        "(default: the python running this command). Probed before it is "
        "written: a path below the 3.9 floor, or one whose sqlite3 has no "
        "FTS5, is refused and nothing is written",
    )
    parser.add_argument(
        "--wire-claude-md",
        action="store_true",
        dest="wire_claude_md",
        help="append an @-import of the store's MEMORY.md to your CLAUDE.md. "
        "Its own consent, because it writes to a file that is yours",
    )
    parser.add_argument(
        "--auto-dream-off",
        action="store_true",
        dest="auto_dream_off",
        help='set "autoDreamEnabled": false, which stops the harness\'s '
        "BACKGROUND CONSOLIDATION only — it goes on writing its own memories. "
        "--auto-memory-off is the flag that stops the writing",
    )
    # NOT required, and mutually exclusive: adopting what the harness has
    # written and switching the feature off are opposite answers to one
    # question, and argparse turns the pair into the exit 2 the dispatcher
    # already promises for a usage error.
    auto_memory = parser.add_mutually_exclusive_group()
    auto_memory.add_argument(
        "--adopt-auto-memory",
        action="store_true",
        dest="adopt_auto_memory",
        help="copy the memories the harness has written under its own config "
        "directory into the store, and point it there for the ones it writes "
        "next. Copies, never moves",
    )
    auto_memory.add_argument(
        "--auto-memory-off",
        action="store_true",
        dest="auto_memory_off",
        help='set "autoMemoryEnabled": false, so the harness writes no '
        "memories of its own at all and memkit is the only one here",
    )


def run(args: argparse.Namespace) -> int:
    # EVERYTHING that can refuse is inside this, INCLUDING the line that reads
    # the machine. A refusal raised one line above it printed a traceback and
    # exited 1 — the code the published table reads as "memkit could not start
    # at all" — so an agent that met a decision went off to reinstall a
    # working install. `Machine()` was that line: it reads the session
    # directory, which can be removed under this process.
    try:
        machine = Machine()
        config_path = _resolve_config(machine, getattr(args, "config", None))
        plan = build_plan(
            machine,
            store=getattr(args, "store", None),
            config=getattr(args, "config", None),
            wire_claude_md=getattr(args, "wire_claude_md", False),
            auto_dream_off=getattr(args, "auto_dream_off", False),
            adopt_auto_memory=getattr(args, "adopt_auto_memory", False),
            auto_memory_off=getattr(args, "auto_memory_off", False),
            interpreter=getattr(args, "interpreter", None),
        )
    except Refusal as refusal:
        return _refuse(refusal)
    except OSError as exc:
        # The class the filesystem raises when it moves under a process — a
        # removed session directory is the reproduced one. NOT `Exception`:
        # exit 5 says "memkit decided not to", and mapping an unexpected
        # failure to it would file a defect under a decision. Nothing has been
        # written at this point, so refusing is honest for what this catches.
        return _refuse(
            Refusal(
                "unreadable-machine",
                f"nothing could be read about this machine ({exc}). The "
                "directory this session stands in may have been removed under "
                "it — `cd` somewhere that exists and run this again.",
            )
        )
    if getattr(args, "dry_run", False):
        print(plan.render())
        return EXIT_OK

    supplied = getattr(args, "confirm", "")
    if supplied != plan.digest:
        return _refuse(
            Refusal(
                "stale-digest",
                f"the plan you approved was {supplied}; the plan against this "
                f"tree is {plan.digest}. Something under the manifest changed, "
                "or these flags are not the ones the manifest was built from. "
                "Re-run --dry-run, read what it says now, and confirm that.",
            )
        )
    # THE APPLIED TEXT, IN THE TRANSCRIPT. "Relay this verbatim" is an
    # instruction to a model and not a control, so the only way to be sure the
    # human saw what is about to happen is to put it where the turn itself
    # records it — beside the writes rather than one turn earlier.
    print(plan.render())
    print()
    print("applying:")
    # No `except Refusal` here: `apply_plan` owns every refusal raised past its
    # first write and reports it as incomplete, because exit 5's promise is
    # about the filesystem rather than about where the exception came from.
    applied = len(plan.pending)
    code = apply_plan(machine, plan, config_path)
    if code == EXIT_OK:
        # THE LINE THAT SAYS IT HAPPENED. Every other outcome names itself on
        # stderr and success named nothing at all, so a run that did the whole
        # plan ended its transcript at the word `applying:` — the same last
        # line a run that died between the header and its first write would
        # leave. The count is the manifest's own `pending`, because this stands
        # under the list the person was asked to approve.
        print(
            f"applied: {applied} {'action' if applied == 1 else 'actions'}. "
            f"Store {_display_path(plan.store)}, config "
            f"{_display_path(config_path)}."
        )
    return code


def _refuse(refusal: Refusal) -> int:
    """One refusal, named, on stderr, with nothing written.

    The NAME is the half a caller can branch on and the sentence is the half a
    person can act on, so both go out — an agent that had only prose would
    parse it, and an agent that had only a name would relay a token.
    """
    print(
        f"memkit init: refused ({refusal.name})\n{refusal.message}",
        file=sys.stderr,
    )
    return EXIT_REFUSED


# --- applying, and the journal that describes it -----------------------------


def _read_or_empty(path: str) -> str:
    """The file's text, "" when it is not there — and a REFUSAL when it is
    there and cannot be read.

    Absence and unreadability were the same answer, so a config this process
    could write but not read was merged into `{}` and renamed over: every root
    and store the adopter had accumulated gone, under a manifest line that says
    "Existing stores are kept". That is the field anti-pattern the settings
    writer beside it already refuses — a tool that meets a read failure and
    replaces the file with a stub — and this is the file that decides which
    directories an every-prompt hook reads.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # swallow: nothing to read is not a read that failed — the
        # `unreadable-config` refusal below is where a failed one goes.
        return ""
    except (OSError, ValueError) as exc:
        raise Refusal(
            "unreadable-config",
            f"{_display_path(path)} exists and could not be read ({exc}). "
            "init converges on its own work and will not replace a config it "
            "cannot see — that file names the directories the hook reads.",
        ) from exc


class Journal:
    """One record per mutation, written AT the mutation.

    Not batched at the end, and that is the whole point: a crash between two
    mutations has to leave a journal that describes what happened, or the
    re-run cannot tell what it already did and a later `--undo` has nothing to
    undo.

    It lives in the state directory rather than under `${CLAUDE_PLUGIN_DATA}`,
    because `claude plugin uninstall` removes plugin data unless `--keep-data`
    — which would delete the record shipped precisely so a later undo is
    possible, and strand the out-of-harness fallback with no config to point at.
    """

    def __init__(self, state_dir: str, manifest: str) -> None:
        self.path = os.path.join(state_dir, INIT_JOURNAL_NAME)
        self.manifest = manifest
        self.run = _sha(f"{manifest}{time.time()}{os.getpid()}")[:12]
        # WHAT THIS RUN'S OWN WRITES LEFT, per path. The settings write refuses
        # a file that moved out from under the manifest, and one run can write
        # the same settings file twice — a flag per key — so it has to be able
        # to tell its own predecessor's landing from somebody else's edit. The
        # journal is where that is already recorded; this is the same record
        # kept in memory, because the question is only ever about this run.
        self.landed: dict = {}
        # THE FILES A RED INTEGRITY CHECK NAMED, kept until the loop is done.
        # VERIFY is not the last action, so the sentence that says whether this
        # run wrote them cannot be true when the check itself reports.
        self.checker_named: list = []

    def record(
        self,
        action: Action,
        after: str,
        locked: bool | None = None,
        expects: str | None = None,
    ) -> None:
        record = {
            "v": 1,
            "run": self.run,
            "manifest": self.manifest,
            "ts": int(time.time()),
            "op": action.op,
            "path": action.path,
            "before": None if action.before == "absent" else action.before,
            "after": after,
            "authored_config": action.authored_config,
        }
        if expects is not None:
            record["expects"] = expects
        if locked is False:
            # Only on the unserialised write. Its absence means the ordinary
            # case, so an existing reader does not have to learn a key to keep
            # reading; its presence is the one thing worth going back for when
            # a store turns out to be missing from a config two inits wrote.
            record["unlocked"] = True
        line = json.dumps(
            record,
            separators=(",", ":"),
        )
        # Line-buffered append and an explicit flush: the next mutation must
        # not be able to happen before this record is on disk, because the only
        # thing a crash leaves behind is what got there first.
        #
        # `append_record` is what keeps a torn predecessor from swallowing
        # this record: the corruption is the absence of a separator before a
        # new one, not the atomicity of the torn write, so it closes at this
        # level rather than needing a short `write()` to be atomic.
        try:
            append_record(self.path, line, fsync=True)
        except OSError as exc:
            raise _RecordNotWritten(exc) from exc
        self.landed[action.path] = after


class _Lock:
    """An advisory lock around the config's read-modify-write.

    `os.replace` makes the file untearable and does nothing at all about a LOST
    APPEND: two inits that both read the config, both add their own store and
    both write, leave one store. The lock is what serialises read→merge→write→
    journal into one step.

    Best-effort by design — a filesystem with no working `flock` degrades to
    the behaviour every earlier build had, which is the same race and no worse.
    A setup command must not fail because a lock could not be taken.
    """

    def __init__(self, state_dir: str) -> None:
        self.path = os.path.join(state_dir, "init.lock")
        self._fd = None
        # Whether serialisation was actually obtained. A caller that cannot
        # tell a serialised write from an unserialised one cannot tell a lost
        # append from a write that never raced, so the journal records it.
        self.held = False

    def __enter__(self):
        try:
            import fcntl

            self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            # NON-BLOCKING, retried against a bound. A plain `LOCK_EX` has no
            # timeout, so a live process holding this file hung `init
            # --confirm` forever with no output — indistinguishable to the
            # caller from a slow checker run, on a command an agent invoked and
            # is waiting on. Giving up and proceeding is what this lock already
            # does when `flock` is unavailable, so the bounded wait adds no new
            # failure mode; it only stops the one that never ends.
            deadline = time.monotonic() + LOCK_WAIT_SECONDS
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            self.held = True
        except (ImportError, OSError):
            if self._fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._fd)
            self._fd = None
        return self

    def __exit__(self, *_exc) -> None:
        if self._fd is None:
            return
        with contextlib.suppress(ImportError, OSError):
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self._fd)
        self._fd = None


def _refuse_escape(path: str, confine: str) -> str:
    """Refuse a write that would not land at the path it is named as.

    Returns the ONE resolution of `path` it judged, for the caller to write.
    A second `realpath` of the same name is a second question, asked of
    whatever the name points at by then: the answer this one refused on and
    the answer the write follows would be different values, and only the
    first was ever looked at.

    FAIL-CLOSED, and it is checked here rather than only at plan time because
    the two are different moments: a link planted between the dry-run and the
    confirm turns a path the manifest proved landed where it said into one that
    does not, and every write below follows links by design.

    THE SAME QUESTION THE PLAN ASKED, word for word — landing place against
    spelled name, built on the root's own realpath so a store that is itself
    on a symlinked path still resolves every destination onto its own row.
    Containment is the weaker claim, and asking it here while the plan asked
    the stronger one left the gap: a link planted BELOW the store that
    resolves back into it stays inside and still sends the bytes to a path no
    manifest line and no ledger row names.
    """
    resolved = os.path.realpath(path)
    if not confine:
        return resolved
    named = os.path.join(
        os.path.realpath(confine), os.path.relpath(path, confine)
    )
    if resolved == named:
        return resolved
    raise Refusal(
        "escapes-store",
        f"{_display_path(path)} resolves to "
        f"{_display_path(_terminal_realpath(path))} rather than to the path it "
        f"is named as under {_display_path(confine)}. The manifest you approved "
        "describes a copy that lands where its own line says, and following a "
        "link off that would write somebody's memory to a path nobody read. "
        "Nothing further was written. Re-run `init --dry-run` for a manifest "
        "of what is left.",
    )


def _write_atomically(
    path: str,
    content: str,
    mode: int = 0o600,
    expect: str | None = None,
    confine: str = "",
) -> str:
    """Write beside and rename over, returning the state token that landed.

    The same care the session ledger takes, for the same reason: `open(path,
    "w")` destroys the old file before writing the new one, so anything that
    stops the write in between leaves a valid prefix of an invalid file — and
    for a config, a valid prefix is a config that names half a store.

    `expect` is what the plan the human approved said was at this path. The
    digest binds the plan to the tree at PLAN time, and between the confirm's
    digest check and this write another process can create a path the manifest
    described as absent — so the manifest's "create" would replace a file its
    reader never saw. Where the plan said absent the create is EXCLUSIVE, which
    closes that window rather than narrowing it; otherwise the state is
    re-derived here, immediately before the write.
    """
    # THROUGH THE LINK, not over it. An adopter whose `~/.claude/settings.json`
    # is a symlink into a dotfiles or nix repo is the common case, and the
    # manifest already advertises that it understands one — it prints where
    # each path resolves. Replacing the link would leave an untracked regular
    # file, the repo copy orphaned and unchanged, and the next `home-manager
    # switch` reaching nothing.
    # ASKED OF THE PATH AS NAMED, and written at the value that answer was
    # about. The guard's question is where the named path lands against what
    # its name says — a path already replaced by its own realpath cannot be
    # asked it, since under a store that is itself a link the resolved form is
    # not even spelled inside the root it answers to — so the guard resolves
    # it, and hands back the one value it judged rather than leaving the name
    # to be resolved a second time here.
    path = _refuse_escape(path, confine)
    if expect is not None and state_token(path) != expect:
        raise Refusal(
            "changed-underfoot",
            f"{_display_path(path)} is not what the manifest you approved "
            f"described ({expect}). Something wrote to it between the dry-run "
            "and now, and applying a plan built against the file that was "
            "there would replace a file nobody read. Re-run `init --dry-run` "
            "for a manifest of what is left.",
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # An EXISTING file keeps its own permissions. The `mode` argument is for a
    # file being created; a settings file somebody deliberately chmod'd 600 —
    # they commonly carry an API key — must not come back 644 from a command
    # whose stated scope is one key.
    with contextlib.suppress(OSError):
        mode = stat.S_IMODE(os.stat(path).st_mode)
    # THE PLAN'S OWN RULE, ASKED AGAIN WHERE THE LONGER NAME IS ACTUALLY MADE.
    # Unreachable from a plan this module built, and that is the point: the
    # rule lives in one function, and a caller that grew a path the planner
    # never measured gets a decision rather than an ENAMETOOLONG traceback.
    if not _name_fits(os.path.basename(path)):
        raise Refusal(
            "name-too-long",
            f"{_display_path(path)} is written by creating "
            f"`<name>.<pid>.tmp` beside it and renaming over, and that name "
            f"is past the {_NAME_MAX_BYTES}-byte limit on one directory "
            "entry, so nothing was written for it.",
        )
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        # The mode goes on before the first byte, so the content never exists
        # at whatever the umask would have given it.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if expect == "absent":
            # LINK rather than replace: `os.link` fails when the name is taken,
            # so the filesystem decides whether the path was free at the moment
            # of the write. The check above narrows the window; this closes it.
            try:
                os.link(tmp, path)
            except FileExistsError as exc:
                raise Refusal(
                    "changed-underfoot",
                    f"{_display_path(path)} was created while this ran. The "
                    "manifest you approved described it as absent, and "
                    "replacing a file nobody read is not what "
                    "creating one means. Re-run `init --dry-run` for a "
                    "manifest of what is left.",
                ) from exc
            except OSError:
                # A filesystem with no hard links (some network and FUSE
                # mounts). The check above still stands; what is lost is the
                # atomicity of it, not the rule.
                os.replace(tmp, path)
            else:
                os.unlink(tmp)
        else:
            os.replace(tmp, path)
    except BaseException:
        # `Refusal` is a plain `Exception`, so an `except OSError` here left
        # `<target>.<pid>.tmp` beside the target on the one path that raises
        # one — `changed-underfoot`, holding the content it was about to
        # write, in the memory store next to `MEMORY.md` or in `~/.claude`.
        # Nothing collects it: the sweep only reaches the state directory and
        # `_stray_markdown` only counts `.md`, so it survived every re-run of
        # the two-turn recovery the refusal itself advertises.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return "file:" + _sha(content)


def _run_checker(machine: Machine, config_path: str) -> tuple:
    """Run the checker against `config_path`, or say why nothing ran.

    The argv is RECONSTRUCTED from the route and one hole: nothing is parsed,
    nothing is spliced, and no environment variable contributes a word.
    `--confirm`'s permission prompt shows `memkit init --confirm <digest>` and
    never this argv, so consent for the command is not consent for whatever a
    session's PATH supplied — which means the argv may not come from anywhere
    a session can write.

    ONE condition decides that nothing runs, and it is the route.
    """
    route, interpreter = _checker_route(machine)
    try:
        argv = checker_argv(route, interpreter)
    except Untrusted as exc:
        return 1, f"no checker route: {exc}"
    try:
        out = _execute(
            [*argv, "--config", config_path],
            timeout=300,
            # This package's own `src`, and `PYTHONSAFEPATH` so the session
            # directory is not the first place `-m` looks for the module named
            # on it. Named explicitly because the child's environment is built
            # from an allow-list that has no PYTHON* name in it, which is what
            # makes this one exception visible.
            env_extra={"PYTHONPATH": _package_path(), "PYTHONSAFEPATH": "1"},
        )
    except (OSError, subprocess.SubprocessError, Untrusted) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return out.returncode, (out.stdout + out.stderr).strip()


def _package_path() -> str:
    """This package's own `src` on PYTHONPATH for the checker subprocess.

    The plugin channel never pip-installs memkit, so `python -m
    memkit.memory_integrity` finds nothing unless the tree is put in front of
    it — the same reason `bin/memkit` prepends it.

    THIS TREE AND NOTHING ELSE. The session's own `PYTHONPATH` used to be
    concatenated on the end, which put back the one variable most worth
    removing at the one call site that most needed it removed: `python -m`
    reads the module it runs out of this string, so a session that exported a
    `PYTHONPATH` named the code the checker imported.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# A checker finding, as the checker prints one: two leading spaces, a code in
# capitals, the path it is about, and an em dash before the reason. Every rule
# with a line to point at spells that path `{where}:{line}`, so the number is
# matched and dropped here rather than left glued to the file name, where it
# names nothing on disk.
_CHECKER_FINDING = re.compile(r"^ +([A-Z][A-Z0-9-]*): (\S+?)(?::\d+)? — ")
# The rules whose sentence runs straight on from the path with no em dash to
# end it. Named one code at a time because a general "code, path, anything"
# shape would also match the rules whose second word is a DIRECTORY or a
# ledger's row count, and neither is a file an adopter can be told to fix.
_CHECKER_PLAIN_FINDING = re.compile(r"^ +(ROW-LOST|STALE): (\S+?)(?::\d+)? ")
# The line that opens the checker's report for ONE store: its id, the root it
# verified, and where that root came from. The checker verifies every store
# the config names, and every path it prints is spelled relative to the root
# of the store whose block it is in — so the roots are read in the order they
# are announced and the blocks are counted against them.
_CHECKER_STORE = re.compile(r"^(\S+) store: verified in (.+?)  \(")
# And the line that opens one store's block, whatever its verdict. A warning
# block names files too, and an `[OK]` block is what makes the count right for
# the blocks after it.
_CHECKER_BLOCK = re.compile(r"^\[(?:FAIL|WARN|OK)\]\s")


def _files_the_checker_names(output: str) -> list:
    """Every file a red check named, absolute, in the order it named them.

    RESOLVED AGAINST THE STORE THE FINDING CAME OUT OF. The checker verifies
    every store the config names, each in a block of its own, and every path
    it prints is relative to THAT store's root. Joined instead onto the root
    of the store this run happened to create, an older store's `DEAD-LINK`
    became a finding against a file of the same name under the new one — whose
    own block said `[OK]` — and the recovery sentence sent the adopter at a
    re-run that could not touch the broken file. So the roots are read off the
    `<id> store: verified in <root>` lines in the order they are announced,
    and the verdict lines are counted against them: the nth block is the nth
    store.

    AMBIGUOUS IS UNATTRIBUTED. A finding printed before any block, or in a
    block past the last root announced, is one nothing here can place — and a
    file named on a guess is worse than a file not named, because the sentence
    under it tells the adopter what to do about it.

    ROW-LOST spells its path from the store's PARENT instead, so the store's
    own directory name is dropped when it is what stands between the path and
    a file that is there.

    A WARNING NAMES A FILE TOO. Warnings share the block errors are printed
    in, so a store that is red for one reason can carry a warning about a
    second file; attributing that file as well is the question this answers
    — did this run write it — asked of every file the block names.
    """
    named: list = []
    roots: list = []
    store = ""
    blocks = 0
    for line in output.splitlines():
        announced = _CHECKER_STORE.match(line)
        if announced is not None:
            roots.append(announced.group(2))
            continue
        if _CHECKER_BLOCK.match(line) is not None:
            store = roots[blocks] if blocks < len(roots) else ""
            blocks += 1
            continue
        found = _CHECKER_FINDING.match(line) or _CHECKER_PLAIN_FINDING.match(line)
        if found is None or not store:
            continue
        rel = found.group(2)
        full = os.path.normpath(os.path.join(store, rel))
        head, _, tail = rel.partition("/")
        if (
            not os.path.lexists(full)
            and tail
            and head == os.path.basename(os.path.normpath(store))
        ):
            full = os.path.normpath(os.path.join(store, tail))
        if full not in named and os.path.lexists(full):
            named.append(full)
    return named


def _report_red_verify(journal: Journal) -> None:
    """What a red check found, and whether THIS run put it there.

    Exit 6 is returned both by a run that stopped partway and by one that did
    everything it said it would and then failed its own integrity check, and
    the recoveries are opposite. A run that stopped converges on a fresh
    dry-run and confirm; a file this run declined to write is one no re-run
    will touch, because the next manifest has nothing to say about it. This
    sentence is what tells an adopter which of the two they are holding.
    """
    mine = [path for path in journal.checker_named if path in journal.landed]
    theirs = [path for path in journal.checker_named if path not in journal.landed]
    lines = [
        "memkit init: every action in the manifest was performed. This exit "
        "code is the integrity checker's verdict on the finished store, not a "
        "run that stopped partway."
    ]
    lines.extend(f"  {_display_path(path)} — this run wrote it" for path in mine)
    lines.extend(
        f"  {_display_path(path)} — this run did not write it" for path in theirs
    )
    if theirs:
        lines.append(
            "A file this run did not write is one no re-run will change: the "
            "next `init --dry-run` has nothing to say about it. Fix it where "
            "it is — or move it out of the store — and run the checker again."
        )
    if mine:
        lines.append(
            "Re-run `init --dry-run` for a manifest of what is left and "
            "confirm THAT digest; what landed has moved the old one."
        )
    print("\n".join(lines), file=sys.stderr)


def apply_plan(machine: Machine, plan: Plan, config_path: str) -> int:
    """Perform the plan, journalling each mutation as it happens."""
    journal = Journal(machine.state_dir, plan.digest)
    # The exit code a red integrity check earns, held until the loop is done.
    incomplete = EXIT_OK
    for action in plan.pending:
        try:
            code = _perform(machine, journal, action, config_path)
        except Refusal as refusal:
            # VERIFY's red answer is deferred to the end of the loop; a refusal
            # that ends the loop early still owes it.
            if incomplete != EXIT_OK:
                _report_red_verify(journal)
            # A refusal raised BELOW the first write is not a refusal any more.
            # Exit 5 promises "nothing was written" and the skill's table tells
            # the agent so, and this became reachable the moment the settings
            # write started re-deriving under the lock: `_settings_with`
            # refuses an unparseable file at apply time, by which point the
            # config and the store are on disk. The reason still goes to the
            # caller; only the code changes, to the one whose move is to fix
            # what the message names and run init again.
            print(
                f"memkit init: refused mid-apply ({refusal.name})\n"
                f"{refusal.message}\nWhat was done before it is recorded in "
                f"{_display_path(journal.path)}. Fix what the message names, "
                "then re-run `init --dry-run` for a fresh digest and confirm "
                "that.",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE
        except _RecordNotWritten as unrecorded:
            print(
                f"memkit init: {action.op} {_display_path(action.path)} was "
                "performed and the journal record of it was not written "
                f"({unrecorded}). The write itself succeeded, so what is on "
                "disk is what the manifest describes; what is missing is the "
                f"line in {_display_path(journal.path)} saying so. Fix what "
                "that errno names, then re-run `init --dry-run` for a fresh "
                "digest and confirm that. The new manifest may re-plan this "
                "write, which costs nothing: a destination already holding "
                "these bytes is left alone, and a differing one is refused.",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE
        except OSError as exc:
            # Everything before this is journalled and is a no-op on the next
            # run. Naming the journal is what makes "re-run it" a safe
            # instruction rather than a hopeful one.
            print(
                f"memkit init: {action.op} {_display_path(action.path)} failed "
                f"({type(exc).__name__}: {exc}). What was done before it is "
                f"recorded in {_display_path(journal.path)}. Re-run "
                "`init --dry-run` for a fresh digest — what landed has changed "
                "the old one — and confirm that; the new manifest lists only "
                "what is left.",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE
        if code != EXIT_OK:
            # A CHECKER THAT IS UNHAPPY IS NOT A REASON TO STOP WRITING.
            # VERIFY is the only action that reports a code, it reports it
            # about a store that is already on disk, and the actions after it
            # are what makes that store the place the harness writes to.
            # Returned from here it left the memories copied AND the harness
            # still writing outside the store — the exact half-state the
            # redirect exists to end, reached by any of the several inputs
            # that turn the check red. What is deferred is the code, not a
            # guard: every action after this one is performed under all of its
            # own, and a refusal from one still stops the run.
            incomplete = code
    # HERE AND NOT AT THE CHECK, because "every action was performed" is only
    # true once the loop has ended: the settings write that makes the store the
    # place the harness writes to comes after VERIFY.
    if incomplete != EXIT_OK:
        _report_red_verify(journal)
    return incomplete


def _perform(
    machine: Machine, journal: Journal, action: Action, config_path: str
) -> int:
    """One action, then its journal record. In that order, and never batched.

    A crash between two mutations has to leave a journal that describes what
    happened; a batch written at the end describes a run that finished, which
    is the one case the record is not needed for.
    """
    if action.op == CREATE_DIR:
        # 0700 for the shared state directory and for nothing else: it
        # holds the soak log and the index, whose filenames are
        # predictable, and a mode a group could read is the symlink
        # pre-planting hazard the location was chosen to avoid.
        mode = 0o700 if action.path == machine.state_dir else 0o755
        # The same containment the file writes get, because `os.makedirs`
        # follows a symlinked component just as happily and a directory made
        # outside the store is where the files after it would land.
        #
        # AND THE DIRECTORY IS MADE AT THE ONE RESOLUTION THAT WAS JUDGED.
        # `os.makedirs` re-traverses every component of the name, so a link
        # swapped in after the guard returned is a second answer to the same
        # question — the guard approved one path and the directory appeared at
        # another, with the memories written under it following.
        made = _refuse_escape(action.path, action.confine)
        os.makedirs(made, mode=mode, exist_ok=True)
        journal.record(action, "dir")
    elif action.op == MERGE_CONFIG:
        with _Lock(machine.state_dir) as lock:
            # WHOSE content is being merged in. The re-read is deliberate and
            # is what lets a concurrent init's store survive this one's write —
            # but the digest an adopter approved describes the file as it was
            # at the dry-run, so a file that changed since is content they were
            # never shown, and merging it forward publishes it under their
            # consent.
            #
            # The two cases are distinguishable and only one of them is fine.
            # A peer init's write is CLAIMED in the journal both processes
            # append to; anything else is not. So a change memkit can account
            # for merges, and a change it cannot is the `changed-underfoot`
            # refusal the sibling write paths make with `expect`.
            now = state_token(action.path)
            if now != action.before and not claim_holds(
                action.path, journal_config_claims(machine.state_dir).get(
                    action.path, []
                )
            ):
                raise Refusal(
                    "changed-underfoot",
                    f"{_display_path(action.path)} is not what the manifest "
                    f"you approved described ({action.before}), and no init "
                    "journal claims what is there now. Something outside "
                    "memkit wrote it between the dry-run and now, and merging "
                    "that forward would publish content nobody read under the "
                    "digest you approved. Re-run `init --dry-run` for a "
                    "manifest of what is left.",
                )
            payload = action.payload if isinstance(action.payload, dict) else {}
            merged = _merge_config(
                _read_or_empty(action.path),
                nonce=str(payload.get("nonce", "")),
                interpreter=str(payload.get("interpreter", "")),
                entries=payload.get("entries") or {},
                where=action.path,
            )
            # WRITE-AHEAD, and it is the config's alone. Between the file
            # landing and its record being fsynced, every future init — dry-run
            # included — refused `foreign-config` and told the adopter memkit
            # did not write the file memkit had just written: no store, no
            # documented recovery, and the only manual fix deleting a config
            # the refusal exists to protect. The claim costs a record that may
            # describe a write that did not happen, which `authored_configs`
            # already tolerates: it keys on the flag and the path, and a claim
            # on a file that is not there answers nothing.
            # WHAT IT EXPECTS TO LAND, on the claim itself. A claim on a
            # path is not a claim on whatever turns up at that path: if the
            # crash lands in this window and something else then creates a
            # config there, the readers have to be able to tell that file from
            # this one, and the digest is the only thing that can.
            expects = "file:" + _sha(merged)
            journal.record(action, "pending", locked=lock.held, expects=expects)
            after = _write_atomically(action.path, merged)
            journal.record(action, after, locked=lock.held)
    elif action.op == VERIFY:
        code, output = _run_checker(machine, config_path)
        if code != 0:
            journal.checker_named = _files_the_checker_names(output)
            print(
                "memkit init: the store was created and the integrity "
                f"checker is not happy with it:\n{output}",
                file=sys.stderr,
            )
            # INCOMPLETE, not refused: the store is on disk. A caller told
            # "refused" would believe nothing was written and go looking
            # for a store that is right there.
            return EXIT_INCOMPLETE
        journal.record(action, "verified")
    elif action.op in (SETTINGS_WRITE, APPEND_LINE):
        # RE-DERIVED under the lock, exactly as the config merge is. init is
        # invoked from inside a live session, so the harness owns and actively
        # writes `settings.json` for the whole run — and these are the LAST
        # actions, after an integrity-checker subprocess that may take minutes.
        # A payload frozen at plan time is an edit against a file that has
        # moved, and the manifest's promise ("changes nothing else") is only
        # true against the file as it is when the write happens.
        payload = action.payload if isinstance(action.payload, dict) else {}
        with _Lock(machine.state_dir) as lock:
            if action.op == SETTINGS_WRITE:
                # RE-DERIVED IS NOT THE SAME AS UNREAD. Merging a key forward
                # into whatever is there publishes that content under a digest
                # taken against a file the adopter saw — and this file decides
                # where an agent writes its memories, so the bytes beside the
                # key matter. The sibling config merge asks the same question
                # in the same shape; the difference is that a peer init's
                # claim can answer it there, and here nothing but this run's
                # own earlier write can, since a flag per key writes the file
                # twice.
                now = state_token(action.path)
                if now != action.before and now != journal.landed.get(
                    action.path
                ):
                    raise Refusal(
                        "changed-underfoot",
                        f"{_display_path(action.path)} is not what the "
                        f"manifest you approved described ({action.before}). "
                        "Something outside memkit wrote it between that "
                        "manifest and this write — the last action, after an "
                        "integrity check that can run for minutes — and "
                        "setting a key in a file nobody read would publish "
                        "the rest of it under the digest you approved. Re-run "
                        "`init --dry-run` for a manifest of what is left.",
                    )
                content = _settings_with(action.path, payload)
            else:
                content = _appended(
                    _read_or_empty(action.path), str(payload.get("line", ""))
                )
            after = _write_atomically(action.path, content, mode=0o644)
            journal.record(action, after, locked=lock.held)
    else:
        # ONLY the frozen-content actions carry `expect`. The config merge, the
        # settings write and the CLAUDE.md append deliberately re-derive their
        # content under the lock against whatever the file holds now — that is
        # what makes two concurrent inits with distinct appends both survive —
        # so for those, a file that moved is the case being handled rather than
        # a reason to stop.
        # NO MODE, so these take `_write_atomically`'s own 0600. An adopted
        # copy of a note somebody deliberately kept private is as sensitive as
        # the note, and this branch writes it: widening it to 0644 published a
        # 0600 source to everyone on the machine, out of a command whose whole
        # subject is where private memories live. An existing file keeps its
        # own permissions either way.
        after = _write_atomically(
            action.path,
            action.content,
            expect=action.before,
            confine=action.confine,
        )
        journal.record(action, after)
    return EXIT_OK
