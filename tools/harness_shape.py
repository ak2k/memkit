#!/usr/bin/env python3
"""Record the SHAPE of a Claude Code config directory, and none of its contents.

memkit's tests run against corpora memkit itself built. What an adopter has is
a directory the HARNESS wrote — thousands of project entries, a handful of them
holding memories, one of them a symlink into somewhere else, one key long
enough that the harness starts hashing it — and a fixture nobody measured
agrees with whatever the code already does. This captures the measurable half
of a real machine: counts, sizes, frontmatter flags, symlink flags, index row
counts, and the settings keys the memory feature reads. `tests/data/
harness_shapes/` holds what it captured, and the regression tests run against
trees rebuilt from those files.

WHAT IT REFUSES TO CARRY: no memory body, no description text, no real file
name, no real repository, host or user name. Keys and file names become
deterministic pseudonyms that keep the segment count, paired with the ORIGINAL
key length — those two are what make a rebuilt tree exercise the same paths,
because what a 164-character key tests is the length, not the letters. So
`--anonymise` is the DEFAULT rather than a flag somebody has to remember, and
`--raw` is refused outright when its output would land inside a git worktree:
the failure this closes is a debugging run whose redirect happened to point at
the repository.

AND NO FREE STRING ANYWHERE ELSE. The rows that are not pseudonyms are the
ones that leaked — a version hint, an install method, a hook event key, a
settings switch — because each looked like a fixed vocabulary and none of them
is: the harness writes what it is given, and what an adopter gives it is
`2.1.263-alice-patched`, a path to their own binary, and a gate named after
their employer. Every one of those is shape-constrained on the way out, so the
whole of an anonymised shape is pseudonyms, numbers, booleans and values from
a list written down here.

STDLIB ONLY, AND IT RUNS ON 3.8 — below this repository's own floor, because
the floor that binds here is the oldest interpreter on a machine worth
capturing, and the host this tool was first piped to ran 3.8.18. It ran there
unmodified, and `tests/test_harness_shape.py` pins the syntax level so it keeps
doing so. The machines are other people's — reached over ssh, with no uv and no
memkit on them, and nothing of this file on their disk at all:

    ssh host 'sudo -n python3 - --config-dir /h/USER/.claude' < tools/harness_shape.py

No third-party import, no `memkit` import, no read of `__file__`. It is a dev
tool like `tools/mutation_sweep.py`, outside the installed payload.

The handful of constants below are DUPLICATED from `memkit` rather than
imported, for the reason above. `tests/test_harness_shape.py` runs this tool
and `memkit.harness_memory.inventory` over one tree and requires them to agree,
which is what keeps the copies in step.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import stat
import sys
import time

SCHEMA = 1

# The one file in an auto-memory directory that is an index rather than a
# memory. Its NAME is carried through anonymisation because a consumer has to
# rebuild it under that name for the code under test to recognise it.
INDEX_NAME = "MEMORY.md"

# `cli_doctor.CONSOLIDATE_LOCK`: how "consolidation is armed" and
# "consolidation is running" are told apart.
CONSOLIDATE_LOCK = ".consolidate-lock"

# EXACT, CASE-SENSITIVE, and an allowlist rather than a filter: everything else
# in a settings file is the adopter's own configuration, and a shape that
# carried whatever it did not recognise would be a shape nobody can promise
# anything about.
MEMORY_KEYS = ("autoMemoryEnabled", "autoMemoryDirectory", "autoDreamEnabled")
DIRECTORY_KEY = "autoMemoryDirectory"

# What `autoMemoryDirectory` becomes when anonymised. A literal, not a
# pseudonym: the value is a path on somebody's machine, and what a consumer
# needs from it is that the key was SET, not where it pointed.
PATH_PLACEHOLDER = "<path>"

# And what a switch set to something that is not a boolean becomes. A literal
# for the same reason, and NOT `null`: the harness reads any non-null as on,
# so a shape that recorded null for a switch somebody had set inverted the one
# fact it was keeping — a tree rebuilt from it reports the feature off where
# the captured machine had it on. No settings file carries this string with a
# meaning of its own.
SET_PLACEHOLDER = "<set>"

# The one plugin name kept literally, because a shape is also how memkit finds
# out whether it was installed on the machine that was captured.
KEPT_PLUGIN = "memkit"

# How much of a memory file is read to classify its frontmatter. The fence is
# the first thing in the file; the largest memory measured anywhere was 46 KB,
# and a cap keeps one pathological file from turning a capture into a read of
# somebody's whole disk.
FRONTMATTER_BYTES = 65536

_FENCE = "---"
# A top-level key is a literal prefix and is tested as one; only the nested
# `type:` under `metadata:` needs a pattern, for the indent it is known by.
_NESTED_TYPE_RE = re.compile(r"^\s+type:")
# The harness's own index row: `- [title](file.md) — hook`.
_INDEX_ROW_RE = re.compile(r"^\s*[-*]\s+\[[^\]]*\]\(([^)]*)\)")
# A leading numeric run, which is all of a version string this SORTS on.
_VERSION_RE = re.compile(r"^(\d+(?:\.\d+)*)")
# And what a version is allowed to BE in an anonymised shape. A different
# pattern from the one above and deliberately so: sorting wants the numeric
# prefix of whatever it is handed, and the privacy rule wants the whole string
# to be that prefix. `fullmatch`, never `match` — a prefix test admits
# `2.1.263-jgomes-patched` entire, which is a colleague's username in a field
# nobody reads and nothing checks.
_VERSION_FULL_RE = re.compile(r"\d+(\.\d+)*")
# `installMethod` as the harness spells it, and an ALLOWLIST rather than a
# pattern: the field has been seen holding an absolute path to the binary, so
# anything unrecognised becomes `other` rather than travelling. Widening this
# costs a fixture nothing; carrying an unrecognised value costs the invariant.
INSTALL_METHODS = ("global", "native", "local", "npm", "unknown")
INSTALL_OTHER = "other"
# The hook events the harness dispatches, and an ALLOWLIST for the same
# reason `INSTALL_METHODS` is one: a settings file can carry any key under
# `hooks`, and "letters and nothing else" is a rule an organisation's own gate
# satisfies — `AcmeComplianceGate` is a name, and it passed. Widening this
# costs a fixture nothing; a pattern that admits an unlisted key costs the
# invariant.
HOOK_EVENTS = (
    "Notification",
    "PermissionDenied",
    "PermissionRequest",
    "PostToolBatch",
    "PostToolUse",
    "PreCompact",
    "PreToolUse",
    "SessionEnd",
    "SessionStart",
    "Stop",
    "SubagentStop",
    "UserPromptSubmit",
)


# --- settings ---------------------------------------------------------------


# TEST-ONLY SEAM, and named here so it is obvious what it is. `_managed_dir()`
# answers with a MACHINE path, which is what makes the managed scope the one
# thing a capture reads outside `--config-dir` — and therefore the one scope no
# test can put a file in. Read in this function and nowhere else in this
# repository; nothing that produces a shape worth committing sets it.
MANAGED_DIR_ENV = "MEMKIT_SHAPE_MANAGED_DIR"


def _managed_dir() -> str:
    """`cli_doctor._managed_dir`, measured on 2.1.241 out of the binary."""
    override = os.environ.get(MANAGED_DIR_ENV)
    if override:
        return override
    if sys.platform == "darwin":
        return "/Library/Application Support/ClaudeCode"
    return "/etc/claude-code"


def _open_regular(path: str, errors: str = "strict"):
    """`path` open for reading, or an `OSError` — and never a wait.

    Two things `open()` will not do here. A FIFO blocks the open until
    somebody writes to the other end, and a capture is unattended, on a host
    the operator may get one run at, where a hang is indistinguishable from a
    slow NFS walk and produces nothing at all. A device or a directory answers
    a read with something that is not a file's contents. `O_NONBLOCK` makes
    the open return, and `fstat` on the fd decides about the thing that was
    actually opened rather than about the name it was reached by — so what
    every caller here gets for anything that is not a plain file is the
    OSError it already books as a state or a counted read error.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", path)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, encoding="utf-8", errors=errors)


def _read_json(path: str):
    """The parsed object at `path`, or None for anything that is not one.

    None for BOTH "not there" and "there and unusable", because the two are
    told apart by the caller: `_settings` asks the filesystem whether the path
    exists and records the second as its own state. Omitting it said "no
    settings at that scope", which is a different machine — and one a
    materialiser reproduces by writing a file that does not parse.
    """
    try:
        with _open_regular(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _settings_scope(data: dict, names: _Pseudonyms, anonymise: bool) -> dict:
    """One settings file, reduced to the memory feature's own surface.

    Every value that leaves here is SHAPE-CONSTRAINED under `anonymise`, and
    the reason is that a settings file is the adopter's, not the harness's:
    the two switches are booleans only because the harness reads them as
    booleans, and a string found under one is somebody's own value that this
    tool has no business carrying. The same rule runs over the hook keys, and
    the placeholder over the directory.

    A null is neither of those. The harness's own `!= null` test makes an
    explicit null ABSENT, so a key set to one describes a machine with the
    feature off — and both branches below would have said it was on. It is
    carried through as the null it is, which is a value a rebuilt tree can
    write and is still distinguishable from a key the file never declared:
    that one is omitted here.
    """
    memory_keys = {}
    for key in MEMORY_KEYS:
        if key not in data:
            continue
        value = data[key]
        if value is None:
            memory_keys[key] = None
        elif key == DIRECTORY_KEY:
            memory_keys[key] = PATH_PLACEHOLDER if anonymise else value
        elif anonymise and not isinstance(value, bool):
            # PRESENT, and not a switch. The placeholder says the key was set
            # to something the harness reads as on, which is the fact a
            # rebuilt tree needs; the value itself is the adopter's.
            memory_keys[key] = SET_PLACEHOLDER
        else:
            memory_keys[key] = value
    hooks = data.get("hooks")
    # EVENT NAMES ONLY. The value under each is a list of matchers and shell
    # COMMANDS — paths, flags and whatever else somebody wired up, which is the
    # single richest source of real names in a settings file. The KEYS are the
    # other half of that and were carried verbatim: an event name is letters,
    # so anything else under `hooks` is a name and gets a pseudonym.
    events = sorted(hooks) if isinstance(hooks, dict) else []
    plugins = data.get("enabledPlugins")
    listed = sorted(plugins) if isinstance(plugins, dict) else []
    return {
        "unreadable": False,
        "memory_keys": memory_keys,
        # SORTED AFTER the pseudonyms are assigned, not before. The numbering
        # follows the real sort order because determinism needs it to, but the
        # ORDER of the emitted list is then a fact about the output rather
        # than about where a redacted key fell in the alphabet.
        "hooks": (
            sorted(names.hook(event) for event in events) if anonymise else events
        ),
        # Sorted after the pseudonyms are assigned, for the reason above.
        "plugins": (
            sorted(names.plugin(key) for key in listed) if anonymise else listed
        ),
    }


def _default_config_dir() -> str:
    """The config directory this process's own harness would use.

    One spelling for the argparse default and for the ownership test below, so
    a machine whose harness lives under `$CLAUDE_CONFIG_DIR` answers the same
    question both times.
    """
    return os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"


def _is_own_config_dir(config_dir: str) -> bool:
    """Whether `config_dir` is the harness config directory of THIS process.

    `realpath` on both sides: a home reached through a link, and `~/.claude`
    itself linked into a dotfiles checkout, are both ordinary and both would
    fail a string comparison.
    """
    return os.path.realpath(os.path.expanduser(config_dir)) == os.path.realpath(
        os.path.expanduser(_default_config_dir())
    )


def _settings(
    config_dir: str, names: _Pseudonyms, anonymise: bool, managed: bool
) -> dict:
    """The two scopes a whole-tree capture can honestly read.

    `user` and the platform `managed` file, and NOT doctor's `local` or
    `project`: those two are `.claude/settings.json` and
    `.claude/settings.local.json` under the directory a SESSION stands in, and
    a capture of a config directory has no cwd to resolve them against. A
    scope whose file is absent is omitted rather than emitted empty, because
    "no managed settings on this machine" and "managed settings that set
    nothing" are different machines.

    AND `managed` IS A MACHINE PATH, which is `_harness`'s rule turned around:
    that file sits at a fixed platform location whatever `--config-dir` names
    — `cli_doctor`'s own `settings_scopes` reads it the same way — so it
    describes the config directory being captured only when that directory is
    one this machine's harness actually runs on. It is therefore read when the
    caller says so and not otherwise, and `main` says so for the config
    directory this process's own harness would use, or when `--managed` is
    passed for the tree the operator names on a host they are capturing whole.
    A shape captured either other way omits the scope, and `capture` records
    WHETHER IT LOOKED alongside it: an omitted scope under a capture that
    looked is a machine with no policy file, and an omitted scope under one
    that did not is a fact about the run. Without that row the two states were
    byte-identical, and a consumer reading the artifact — which is all a
    materialiser has — could only take the absence for the machine's.
    """
    scopes = [("user", os.path.join(config_dir, "settings.json"))]
    if managed:
        scopes.append(
            ("managed", os.path.join(_managed_dir(), "managed-settings.json"))
        )
    found = {}
    for scope, path in scopes:
        data = _read_json(path)
        if data is not None:
            found[scope] = _settings_scope(data, names, anonymise)
        elif os.path.lexists(path):
            # PRESENT, and not readable as an object. Omitting it would say
            # this machine has no settings at that scope, which is a different
            # machine — the same distinction the docstring above draws between
            # an absent file and one that sets nothing.
            found[scope] = {
                "unreadable": True,
                "memory_keys": {},
                "hooks": [],
                "plugins": [],
            }
    return found


def _version_sort_key(name: str) -> tuple:
    match = _VERSION_RE.match(name)
    if match is None:
        return ((), name)
    return (tuple(int(part) for part in match.group(1).split(".")), name)


def _harness(config_dir: str, anonymise: bool) -> dict:
    """A version hint and an install method, or nulls.

    RELATIVE TO `config_dir` AND NEVER TO `$HOME`, which is what makes a
    capture of a temporary directory reproducible: the round-trip test builds a
    config dir under `tmp_path`, and a lookup that fell back to the operator's
    own `~/.claude.json` would put that machine's version in a fixture built
    from a tree that has none.

    BOTH VALUES ARE ADOPTER-CONTROLLED, which is why neither travels
    unrecognised. `version_hint` is a directory name under `versions/` or a
    string out of `.claude.json`, so a hand-built `2.1.263-alice-patched` is a
    username in a field nobody reads; `installMethod` has been seen holding an
    absolute path to the binary. A shape says the machine had a version and an
    install method, never which one somebody typed.
    """
    parent = os.path.dirname(os.path.abspath(config_dir))
    versions = os.path.join(parent, ".local", "share", "claude", "versions")
    hint = None
    try:
        with os.scandir(versions) as entries:
            installed = [entry.name for entry in entries]
    except OSError:
        installed = []
    if installed:
        hint = max(installed, key=_version_sort_key)
    claude_json = None
    for candidate in (
        os.path.join(config_dir, ".claude.json"),
        os.path.join(parent, ".claude.json"),
    ):
        claude_json = _read_json(candidate)
        if claude_json is not None:
            break
    claude_json = claude_json or {}
    if hint is None:
        seen = claude_json.get("lastReleaseNotesSeen")
        hint = seen if isinstance(seen, str) else None
    install = claude_json.get("installMethod")
    install = install if isinstance(install, str) else None
    if anonymise:
        if hint is not None and not _VERSION_FULL_RE.fullmatch(hint):
            # The highest entry is the one somebody hand-built, and rejecting
            # it whole loses the releases installed beside it: a beta or a
            # patched build sitting next to 2.1.258 is an ordinary machine,
            # and the hint it deserves is the highest entry that IS a version.
            usable = [name for name in installed if _VERSION_FULL_RE.fullmatch(name)]
            hint = max(usable, key=_version_sort_key) if usable else None
        if install is not None and install not in INSTALL_METHODS:
            install = INSTALL_OTHER
    return {"version_hint": hint, "install": install}


# --- frontmatter ------------------------------------------------------------


def _description_len(line: str) -> int:
    """The length of one `description:` value, as the code that judges it counts.

    A LENGTH, which is the whole of what a shape says about a description: the
    rule that fires on one is `>155 characters`, so the number is what a
    rebuilt corpus has to reproduce and the text is what it must not carry.
    Which makes it the CHECKER's number or nothing, and the checker is
    `memory_integrity._scalar` — so this counts what that counts WHEREVER
    THAT RETURNS A VALUE.

    AND A NUMBER WHERE IT DOES NOT. `_scalar` rejects five spellings outright
    — an empty value, a quote that never closes, a plain scalar opening on a
    YAML indicator, one holding `": "`, one holding `" #"` — and answers
    DESC-BAD rather than a length. This returns the length of what was
    written for all five, because a shape records what a file holds and the
    verdict on it is the consumer's to re-take from the rebuilt tree.

    ONE LINE, no folding. Neither real reader folds: the checker's frontmatter
    parser skips every indented continuation as a nested key, and the recall
    hook's regex is `(.+)$` without DOTALL. A folded length was a number
    nothing in this repository decides anything on.

    AND WITHOUT THE QUOTES, which `_scalar` strips before comparing against
    its cap. Measured over the 72 harness-written memories on one machine: 50
    were recorded two characters long, and one sat at 154 by the checker and
    156 here — crossing the boundary the shape exists to reproduce, on the
    side that fires a rule the machine does not.

    A lone `>` or `|` is a block scalar, which the checker refuses outright as
    DESC-BAD and the hook reads as a one-character description. The number
    here is what is written; adjudicating between them is the consumer's.
    """
    value = line.split(":", 1)[1].strip()
    if value[:1] in ('"', "'") and len(value) >= 2 and value[-1] == value[0]:
        inner = value[1:-1]
        value = inner.replace("''", "'") if value[0] == "'" else inner.replace(
            '\\"', '"'
        )
    return len(value)


def _frontmatter(text: str) -> dict:
    """Which of the four frontmatter facts a memory file carries.

    No YAML parser, because there is none in the standard library and this runs
    where nothing can be installed. The fence is `---` on line 1 through the
    next `---` line: an unterminated one is not frontmatter, which is also how
    a file that merely opens with a horizontal rule stays uncounted.
    """
    absent = {
        "has_frontmatter": False,
        "has_description": False,
        "description_len": None,
        "has_name": False,
        "has_type": False,
    }
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != _FENCE:
        return absent
    end = None
    for index in range(1, len(lines)):
        if lines[index].rstrip() == _FENCE:
            end = index
            break
    if end is None:
        return absent
    fence = lines[1:end]
    has_name = has_type = has_description = False
    description_len = None
    in_metadata = False
    for line in fence:
        if line.strip() and line[:1] not in (" ", "\t"):
            in_metadata = line.startswith("metadata:")
        if line.startswith("name:"):
            has_name = True
        if line.startswith("type:") or (in_metadata and _NESTED_TYPE_RE.match(line)):
            has_type = True
        if not has_description and line.startswith("description:"):
            has_description = True
            description_len = _description_len(line)
    return {
        "has_frontmatter": True,
        "has_description": has_description,
        "description_len": description_len,
        "has_name": has_name,
        "has_type": has_type,
    }


def _read_head(path: str):
    """The first `FRONTMATTER_BYTES` of `path`, or None if it could not be read.

    None rather than `""`: an unreadable file and an empty one produced the
    same four `false` flags, and a capture taken over ssh under `sudo -n` into
    an NFS home is exactly where the difference lives.
    """
    try:
        with _open_regular(path, errors="replace") as handle:
            return handle.read(FRONTMATTER_BYTES)
    except OSError:
        return None


# --- pseudonyms -------------------------------------------------------------


class _Pseudonyms:
    """Deterministic stand-ins that keep the structure and drop the names.

    One counter per kind, assigned on FIRST APPEARANCE while the walk runs over
    sorted keys — so the same capture of the same tree produces the same file
    twice, which is what makes a fixture reviewable in a diff.

    Per SEGMENT rather than per key: a project key is the sanitised repository
    path, so two checkouts under one home share a prefix, and a rebuilt tree
    that lost the sharing would not have the shape that was measured. What
    survives is that two keys shared a segment, never which one.
    """

    __slots__ = ("_segments", "_files", "_plugins", "_markets", "_hooks")

    def __init__(self) -> None:
        self._segments = {}
        self._files = {}
        self._plugins = {}
        self._markets = {}
        self._hooks = {}

    @staticmethod
    def _assign(table: dict, value: str, prefix: str) -> str:
        if value not in table:
            table[value] = f"{prefix}{len(table) + 1}"
        return table[value]

    def key(self, key: str) -> str:
        # `split("-")` keeps the empty leading segment an absolute path
        # sanitises to, and an empty segment stays empty: the separator run in
        # `-Users-x--config-nix` is a fact about the original path.
        return "-".join(
            "" if not segment else self._assign(self._segments, segment, "s")
            for segment in key.split("-")
        )

    def file(self, name: str) -> str:
        if name == INDEX_NAME:
            return name
        return self._assign(self._files, name, "m") + ".md"

    def hook(self, event: str) -> str:
        # The harness's whole list of events is public and written down above.
        # Anything else under `hooks` is a key somebody chose, which is where
        # an org name lives.
        if event in HOOK_EVENTS:
            return event
        return self._assign(self._hooks, event, "h")

    def plugin(self, key: str) -> str:
        # memkit by name, its marketplace not: whether memkit is installed is
        # the fact a shape is allowed to keep, and where somebody hosts their
        # own marketplace is not. WHICHEVER WAY THE KEY IS SPELLED — the
        # exception used to fire only on `<plugin>@<marketplace>`, so a bare
        # `memkit` anonymised to `p<n>` and took with it the fact the
        # exception exists to keep.
        plugin, sep, marketplace = key.partition("@")
        left = plugin if plugin == KEPT_PLUGIN else self._assign(
            self._plugins, plugin, "p"
        )
        if not sep:
            return left
        return left + "@" + self._assign(self._markets, marketplace, "q")


# --- the walk ---------------------------------------------------------------


def _lock_age(project_dir: str, memory_dir: str, now: float):
    """Age of the consolidation lock in seconds, or None.

    THE PROJECT DIRECTORY FIRST, then the memory directory — `cli_doctor`'s
    own order, and the reason to copy it rather than reason about it is that
    both files can exist at once. That is exactly the state this function's
    docstring used to name as why it looks in two places: a lock that moved
    from the outer spelling to the inner one. Taking the inner one first, a
    tree rebuilt from `lock_age_s` reports an age doctor never would.

    Never negative: a lock stamped in the future is a clock that disagrees,
    not a consolidation that has not started yet.
    """
    for candidate in (
        os.path.join(project_dir, CONSOLIDATE_LOCK),
        os.path.join(memory_dir, CONSOLIDATE_LOCK),
    ):
        try:
            stamp = os.stat(candidate).st_mtime
        except OSError:
            continue
        return max(0, int(now - stamp))
    return None


def _outside(target: str) -> bool:
    """Whether an index row names something other than a file in this directory.

    Judged from the STRING, with no filesystem touched. A row target is
    adopter-authored text, and joined onto the memory directory an absolute
    one wins the join outright while a `../..` one walks out of it — so the
    lookup that scored the row was a stat of whatever path somebody wrote, run
    over ssh under `sudo -n` on a machine this tool is a guest on. It is also
    the accuracy bug: `/etc/passwd` exists, so that row scored as SATISFIED.
    """
    if not target or os.path.isabs(target) or target in (os.curdir, os.pardir):
        return True
    return "/" in target or "\\" in target or os.sep in target


def _index(memory_dir: str, listed: list) -> dict:
    """Row counts for `MEMORY.md`, and how many of the rows point at nothing.

    `dangling_rows` is a COUNT and never a name. NOTHING JUDGES A
    HARNESS-WRITTEN INDEX TODAY — no rule in this repository reads one — so
    this is the number an ORPHAN-style rule would be judged on if it existed,
    and it is here because a rebuilt corpus has to be able to reproduce it.
    `truncated` is how the count says it was taken off less than the file.

    CAPPED at `FRONTMATTER_BYTES` like every other read here. The cap exists
    so one pathological file cannot turn a capture into a read of somebody's
    whole disk, and an index was the one read that did not honour it.
    """
    path = os.path.join(memory_dir, INDEX_NAME)
    try:
        with _open_regular(path, errors="replace") as handle:
            text = handle.read(FRONTMATTER_BYTES + 1)
    except OSError:
        return {"rows": 0, "dangling_rows": 0, "truncated": False}
    truncated = len(text) > FRONTMATTER_BYTES
    head = text[:FRONTMATTER_BYTES]
    lines = head.splitlines()
    if truncated and lines and not head.endswith(("\n", "\r")):
        # What the cap cut is a fragment of a line — UNLESS it fell on a line
        # boundary, where the last line is whole and dropping it loses a row
        # the file has.
        lines.pop()
    present = set(listed)
    rows = dangling = 0
    for line in lines:
        match = _INDEX_ROW_RE.match(line)
        if match is None:
            continue
        rows += 1
        target = match.group(1).strip()
        if target in present:
            continue
        # `lexists`, so a row pointing at a dead symlink counts as present:
        # the file is there to be moved, which is what the row is about.
        if _outside(target) or not os.path.lexists(
            os.path.join(memory_dir, target)
        ):
            dangling += 1
    return {"rows": rows, "dangling_rows": dangling, "truncated": truncated}


def _memory_dir(
    key: str,
    project_dir: str,
    project_is_symlink: bool,
    memory_dir: str,
    listed: list,
    names: _Pseudonyms,
    anonymise: bool,
    now: float,
) -> tuple:
    """One project's memory directory, and how many files could not be read.

    THREE LINK FLAGS, matching `harness_memory.ProjectMemory`, because
    whatever copies this directory owes each of them a different answer: the
    memory directory's own, the project directory it sits in, and each file
    inside. The tool captured one of the three, so a rebuilt tree could not
    reproduce the other two and the equivalence test could not see them.

    THREE, and not the fourth. Where a linked memory directory POINTED was
    recorded too — `in-shape` or `external` — and nothing in this repository
    reads it: no rule fires on the distinction, no materialiser rebuilds it,
    and the value cost a `realpath` of somebody else's path on a host this is
    a guest on. It is gone until the work that reads it lands.
    """
    files = []
    read_errors = 0
    for name, linked in listed:
        path = os.path.join(memory_dir, name)
        failed = False
        try:
            # `lstat`: the size of the LINK, never of what it points at. A
            # memory file that is a link out of the directory had the target's
            # size and the target's description length recorded, which is a
            # measurement of a file outside the capture.
            size = os.lstat(path).st_size
        except OSError:
            size = None
            failed = True
        record = {
            "name": names.file(name) if anonymise else name,
            "size": size,
            "is_symlink": linked,
        }
        head = ""
        if not linked:
            # NEVER OPENED WHEN IT IS A LINK. The bytes are somebody else's
            # file, reached through a name in this directory.
            head = _read_head(path)
            if head is None:
                head, failed = "", True
        record.update(_frontmatter(head))
        files.append(record)
        if failed:
            read_errors += 1
    names_listed = [name for name, _ in listed]
    return (
        {
            "key": names.key(key) if anonymise else key,
            "key_len": len(key),
            "is_symlink": os.path.islink(memory_dir),
            "project_is_symlink": project_is_symlink,
            "files": files,
            "index": (
                _index(memory_dir, names_listed)
                if INDEX_NAME in names_listed
                else None
            ),
            "lock_age_s": _lock_age(project_dir, memory_dir, now),
        },
        read_errors,
    )


def capture(config_dir: str, anonymise: bool = True, managed: bool = False) -> dict:
    """The whole shape of one config directory.

    `managed` defaults OFF because every other row here is read out of
    `config_dir` and that one is read off the machine: a caller who has not
    established that the two are the same machine's gets the tree it named and
    nothing from beside it.

    A project directory is LISTED when it holds at least one `*.md`, index
    included — so an index-only directory appears, with its rows and no
    memories, which is a state two of memkit's own rules fire on. What
    `memkit.harness_memory.inventory` calls a corpus is the narrower set a
    consumer derives from this: the listed directories holding some file other
    than the index.
    """
    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    names = _Pseudonyms()
    now = time.time()
    projects_root = os.path.join(config_dir, "projects")
    try:
        with os.scandir(projects_root) as entries:
            keys = sorted((entry.name, entry.is_symlink()) for entry in entries)
    except FileNotFoundError:
        # A config directory with no `projects/` is a machine with nothing
        # written yet, and that is a shape. EVERY OTHER failure is not: the
        # directory is there and could not be listed, and the zeros below it
        # read as a healthy empty machine — which over `sudo -n` into an NFS
        # home under root-squash is the reachable failure, on a host the
        # operator may not get a second run at. It leaves here as an
        # exception, and `main` turns it into a message and an exit 2.
        keys = []
    memory_dirs = []
    memory_dirs_total = skipped = read_errors = 0
    for key, project_is_symlink in keys:
        project_dir = os.path.join(projects_root, key)
        memory_dir = os.path.join(project_dir, "memory")
        try:
            is_dir = stat.S_ISDIR(os.stat(memory_dir).st_mode)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            # `os.path.isdir` answers False for an unreadable directory as
            # well as for an absent one, so a project nobody could read used
            # to miss BOTH counters and read as a project with no memories.
            skipped += 1
            continue
        if not is_dir:
            continue
        memory_dirs_total += 1
        listed = []
        try:
            with os.scandir(memory_dir) as entries:
                for entry in entries:
                    if not entry.name.endswith(".md"):
                        continue
                    try:
                        # `is_file` FOLLOWS, and stays following: a dangling
                        # link is not a memory to `harness_memory.inventory`
                        # either, and these two are held to the same answer.
                        # What changes is the blast radius — `DirEntry.is_file`
                        # swallows FileNotFoundError and nothing else, so a
                        # link somebody looped (ELOOP) or a file the capture
                        # cannot stat (EACCES, which `sudo -n` into an NFS
                        # home under root-squash reaches) used to throw away
                        # the listing of every memory beside it.
                        if not entry.is_file():
                            continue
                        found = (entry.name, entry.is_symlink())
                    except OSError:
                        read_errors += 1
                        continue
                    listed.append(found)
            listed.sort()
        except OSError:
            # One unreadable directory is a fact about the machine, not a
            # reason to report nothing about the other three thousand.
            skipped += 1
            continue
        if not listed:
            continue
        record, failed = _memory_dir(
            key, project_dir, project_is_symlink, memory_dir, listed,
            names, anonymise, now,
        )
        read_errors += failed
        memory_dirs.append(record)
    return {
        "schema": SCHEMA,
        "tool": "harness_shape",
        "anonymised": anonymise,
        "harness": _harness(config_dir, anonymise),
        "settings": _settings(config_dir, names, anonymise, managed),
        # WHETHER THE MACHINE'S POLICY FILE WAS LOOKED FOR. An absent `managed`
        # scope meant three things at once — no such file, `--managed` not
        # passed, or the tree not this machine's — and a consumer reading the
        # artifact cannot tell which. No `schema` bump goes with it: every
        # field a reader of the older fixtures already reads is still there and
        # still means the same thing, and what catches a fixture that predates
        # the row is the field-set gate rather than the number.
        "settings_managed_read": managed,
        "projects_total": len(keys),
        "memory_dirs_total": memory_dirs_total,
        # TWO counters, because a half-failed capture that is
        # byte-indistinguishable from a complete one is a capture nobody can
        # act on: `skipped` is a memory directory that could not be listed,
        # `read_errors` a file inside one that could not be measured.
        "skipped": skipped,
        "read_errors": read_errors,
        "memory_dirs": memory_dirs,
    }


# --- the command ------------------------------------------------------------


def _landing_dir(destination: str) -> str:
    """The directory a write to `destination` actually lands in.

    `realpath` of the ORIGINAL dirname, never of an absolute path built first:
    `abspath` collapses `<link>/..` textually while the kernel resolves the
    link and then takes `..` from its TARGET, so the two answers name
    different directories and the lexical one is not where the bytes go. Every
    question below — is this a checkout, is a link standing in the path — is
    asked about this directory, and the open uses it too, so nothing is
    checked about one path and written to another.
    """
    return os.path.realpath(os.path.dirname(destination) or os.curdir)


def _inside_worktree(directory: str) -> bool:
    """Whether a file written in `directory` would land in a git checkout.

    A DIRECTORY and not a file, because the two callers name one differently:
    `--out` is a file that does not exist yet, so its parent is what there is
    to walk, and a stdout redirect names no path at all — only the directory
    the process stands in. `os.path.exists` rather than `isdir`: a linked
    worktree's `.git` is a FILE, and those are exactly the trees a
    worktree-per-unit workflow writes in.
    """
    current = os.path.realpath(directory)
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _stdout_is_a_file() -> bool:
    """Whether stdout is a REDIRECT rather than a terminal, a pipe or a device.

    A regular file is the one spelling of stdout that leaves bytes on disk
    under a name this command line never mentions, which is why `--raw >
    somewhere` escaped a refusal keyed on `--out`. A tty, a pipe — `| jq` and
    the documented ssh flow are both pipes — and `/dev/null` are untouched.
    """
    try:
        return stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode)
    except (AttributeError, OSError, ValueError):
        # No usable fd is not a redirect. It is also not a file this can
        # overwrite, so answering False refuses nothing that matters.
        return False


def _stdout_destination():
    """The path a redirected stdout writes to, or None if it cannot be had.

    THE DESTINATION IS THE QUESTION, and the working directory was standing in
    for it: a `--raw` run started outside a checkout and redirected into one
    was allowed, and one started inside it and redirected safely outside was
    refused. Both answers were about the wrong directory.

    `F_GETPATH` on darwin and `/proc/self/fd` on linux, which are the two
    platforms this is piped to; a kernel that answers neither leaves the
    caller with the cwd test, which over-refuses rather than under-refuses.
    """
    if not _stdout_is_a_file():
        return None
    fd = sys.stdout.fileno()
    if sys.platform.startswith("linux"):
        try:
            return os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            return None
    if sys.platform == "darwin":
        try:
            import fcntl
        except ImportError:
            return None
        try:
            # 50 is `F_GETPATH`, which the fcntl module does not name.
            answer = fcntl.fcntl(fd, 50, b"\0" * 1024)
        except (OSError, ValueError):
            return None
        if isinstance(answer, bytes):
            return answer.split(b"\0", 1)[0].decode("utf-8", "replace") or None
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness_shape.py",
        description="Capture the shape of a Claude Code config directory.",
    )
    parser.add_argument(
        "--config-dir",
        default=_default_config_dir(),
        help="the harness config directory (default: $CLAUDE_CONFIG_DIR or ~/.claude)",
    )
    parser.add_argument("--out", help="write here instead of stdout")
    parser.add_argument(
        "--managed",
        action="store_true",
        help=(
            "read the platform managed-settings.json as well; implied when "
            "--config-dir is this machine's own, and needed only for another "
            "config directory on a machine you are capturing whole"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--anonymise",
        action="store_true",
        help="pseudonymise keys, file names and plugins (the default)",
    )
    mode.add_argument(
        "--raw",
        action="store_true",
        help="keep real names; for reading your own machine, never for a fixture",
    )
    args = parser.parse_args(argv)

    if args.raw:
        destination = args.out or _stdout_destination()
        if destination is not None:
            if _inside_worktree(_landing_dir(destination)):
                sys.stderr.write(
                    f"harness_shape: --raw refuses to write inside a git "
                    f"worktree ({destination}); a shape committed with real "
                    f"names is the one mistake this tool exists to prevent\n"
                )
                return 2
        elif _stdout_is_a_file() and _inside_worktree(os.getcwd()):
            # THE FALLBACK, for a kernel that will not name fd 1. The working
            # directory is the wrong question — a redirect from outside a
            # checkout into one is the leak, and this cannot see it — so it is
            # what is left rather than what is asked, and it over-refuses.
            sys.stderr.write(
                "harness_shape: --raw refuses a redirect to a file from "
                "inside a git worktree; pass --out so the destination can be "
                "checked, and name one outside the checkout\n"
            )
            return 2
    config_dir = os.path.expanduser(args.config_dir)
    if not os.path.isdir(config_dir):
        sys.stderr.write(f"harness_shape: {config_dir} is not a directory\n")
        return 2

    try:
        shape = capture(
            config_dir,
            anonymise=not args.raw,
            # The machine's policy file describes the tree being captured when
            # the tree is one this machine's harness runs on. `--managed` is
            # how the documented ssh flow says so for another user's home
            # under `sudo -n`, where the process's own default is root's.
            managed=args.managed or _is_own_config_dir(config_dir),
        )
    except OSError as exc:
        # A capture that could not read the projects directory reports no
        # shape rather than an empty one.
        where = getattr(exc, "filename", None) or config_dir
        sys.stderr.write(f"harness_shape: {where}: {exc.strerror or exc}\n")
        return 2
    text = json.dumps(shape, indent=2) + "\n"
    if not args.out:
        sys.stdout.write(text)
        return 0
    parent = os.path.dirname(os.path.abspath(args.out))
    resolved = _landing_dir(args.out)
    if resolved != parent:
        # O_NOFOLLOW guards the LAST component only, so a link one level up
        # chose the file that got truncated: `--out real/linkdir/shape.json`
        # wrote through `linkdir` and overwrote whatever `target.json` behind
        # it was. Refused rather than followed, and the resolved path is named
        # so an operator whose home really is reached through a link — or who
        # named /tmp on a mac — can pass that path instead.
        sys.stderr.write(
            f"harness_shape: {args.out}: a symlink stands in this path, which "
            f"chooses what gets overwritten; it resolves to {resolved}\n"
        )
        return 2
    # AFTER the refusals, not before: a rejected --out used to leave the
    # directories it had already created behind it.
    os.makedirs(resolved, exist_ok=True)
    # The RESOLVED directory joined to the final name, so the path that was
    # checked is the path that gets opened.
    target = os.path.join(resolved, os.path.basename(args.out))
    try:
        # O_NOFOLLOW, because `open(path, "w")` follows a link already sitting
        # at that name and truncates what it points at. This runs under
        # `sudo -n` on hosts it is a guest on, so a link somebody else planted
        # in the destination directory is a choice about what gets
        # overwritten — and no capture is worth writing through one.
        #
        # O_NONBLOCK for the same reason one level along: a FIFO at this name
        # would block the open until somebody read from it, and a capture that
        # hangs on a host reached once over ssh is a capture nobody gets.
        # No O_TRUNC — what this opens is not yet known to be a file worth
        # truncating, and the decision is taken below on the fd itself.
        fd = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            # The mode a plain `open(path, "w")` would have created: this
            # change is about not following a link, not about tightening
            # permissions, and umask applies to both the same way.
            0o666,
        )
    except OSError as exc:
        # O_NOFOLLOW reports ELOOP, which reads as a broken filesystem rather
        # than as the refusal it is.
        why = (
            "a symlink sits at this name, and --out refuses to follow one"
            if os.path.islink(target)
            else (exc.strerror or str(exc))
        )
        sys.stderr.write(f"harness_shape: {args.out}: {why}\n")
        return 2
    # ON THE FD, not on the name: between a check by name and the open, the
    # name can be made to mean something else. What was opened is what these
    # two questions are about.
    try:
        info = os.fstat(fd)
        # A hard link is a link O_NOFOLLOW cannot see, and it is the same
        # write primitive as a symlink: a second name for the destination
        # inode, planted by whoever can write in this directory, choosing what
        # gets truncated.
        if info.st_nlink > 1:
            why = (
                "another name points at this file, and --out refuses to "
                "overwrite through one"
            )
        elif not stat.S_ISREG(info.st_mode):
            why = "--out writes a regular file, and this name is not one"
        else:
            why = None
            # Only now, when what is held is a file with one name.
            os.ftruncate(fd, 0)
    except OSError as exc:
        why = exc.strerror or str(exc)
    if why is not None:
        os.close(fd)
        sys.stderr.write(f"harness_shape: {args.out}: {why}\n")
        return 2
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
