#!/usr/bin/env python3
"""Record the SHAPE of a Claude Code config directory, and none of its contents.

memkit's tests run against corpora memkit itself built. What an adopter has is
a directory the HARNESS wrote — thousands of project entries, a handful of them
holding memories, one of them a symlink into somewhere else, one key long
enough that the harness starts hashing it — and a fixture nobody measured
agrees with whatever the code already does. This captures the measurable half
of a real machine: counts, sizes, frontmatter flags, symlink flags, index row
counts, and the settings keys the memory feature reads. `tests/data/
harness_shapes/` holds what it captured, and the tests read those files back to
hold the artifact to what it promises: no real name anywhere in one, the field
set and the value types a reader may rely on, and no shape in that directory
the repository does not track. Rebuilding a tree from a shape and running the
corpus tests against it is what these were captured for, and is not built yet.

WHAT IT REFUSES TO CARRY: no memory body, no description text, no real file
name, no real repository, host or user name. Keys and file names become
deterministic pseudonyms that keep the segment count, paired with the ORIGINAL
key length — those two are what make a rebuilt tree exercise the same paths,
because what a 164-character key tests is the length, not the letters. So
`--anonymise` is the DEFAULT rather than a flag somebody has to remember, and
`--raw` is refused wherever this can SEE where its output lands inside a git
worktree — `--out`, and a stdout redirected straight at a file. The failure
that closes is a debugging run whose redirect happened to point at the
repository. A pipe is not visible to it: the far end of the ssh recipe below
is a machine this process cannot ask about, so what happens to those bytes is
yours to get right.

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

    ssh host 'sudo -n python3 - --config-dir /h/USER/.claude --managed' \
        < tools/harness_shape.py

`--managed` is on that line rather than left to the default because under
`sudo -n` this process's home is root's: the directory named is not the one
this harness would use, so the flag is the only thing that makes a capture
of somebody else's whole machine include that machine's policy file.

No third-party import, no `memkit` import, no read of `__file__`. It is a dev
tool like `tools/mutation_sweep.py`, outside the installed payload.

The handful of constants below are DUPLICATED from `memkit` rather than
imported, for the reason above. `tests/test_harness_shape.py` holds the copies
in step with two comparisons, because neither covers the other: it runs this
tool and `memkit.harness_memory.inventory` over one tree and requires the two
WALKS to agree, and it asserts each copied CONSTANT against the value `memkit`
holds — a walk agrees whenever both sides read one tree the same way, which is
still the case when both are wrong about a name.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import re
import stat
import subprocess
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

# And how much of a settings file is parsed. `_read_json` reads a whole file
# into memory before the parser sees any of it, and a settings file is the
# adopter's: without a cap, one pathological file is a capture that leaves by
# a door this tool did not choose. Generously above any real one — a bound,
# not a size anybody should meet.
SETTINGS_BYTES = 4 * 1024 * 1024

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


def _regular_fd(path: str) -> int:
    """A descriptor on `path`, or an `OSError` — and never a wait.

    Two things `open()` will not do here. A FIFO blocks the open until
    somebody writes to the other end, and a capture is unattended, on a host
    the operator may get one run at, where a hang is indistinguishable from a
    slow NFS walk and produces nothing at all. A device or a directory answers
    a read with something that is not a file's contents. `O_NONBLOCK` makes
    the open return, and `fstat` on the fd decides about the thing that was
    actually opened rather than about the name it was reached by — so what
    every caller here gets for anything that is not a plain file is the
    OSError it already books as a state or a counted read error.

    The guard is the descriptor and not the stream, so the two readers below
    are the same open: one text, one bytes for the read that is CAPPED, a cap
    counted in bytes having to be applied to bytes.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_regular(path: str, errors: str = "strict"):
    """`path` open for reading as text, through the guard above."""
    return os.fdopen(_regular_fd(path), encoding="utf-8", errors=errors)


def _open_regular_bytes(path: str):
    """`path` open for reading as bytes, through the guard above."""
    return os.fdopen(_regular_fd(path), "rb")


def _read_json(path: str):
    """The parsed object at `path`, or None for anything that is not one.

    None for BOTH "not there" and "there and unusable", because the two are
    told apart by the caller: `_settings` asks the filesystem whether the path
    exists and records the second as its own state. Omitting it said "no
    settings at that scope", which is a different machine — and one a
    materialiser reproduces by writing a file that does not parse.

    AND FOR A FILE PAST THE CAP, which earns the same None a third way. The
    parse takes the whole file into memory first, so the size of somebody
    else's settings file decided how much this capture allocated; past the
    cap it is a scope this run could not read, which is a state the caller
    already books. READ AS BYTES for the reason the frontmatter read is:
    `read(n)` on a text stream counts characters, and the cap is named in
    bytes because bytes are what came off the disk.
    """
    try:
        with _open_regular_bytes(path) as handle:
            raw = handle.read(SETTINGS_BYTES + 1)
        if len(raw) > SETTINGS_BYTES:
            return None
        data = json.loads(raw)
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

    AND A NUMBER WHERE IT DOES NOT. `_scalar` rejects six kinds of value
    outright — an empty one, a quote that never closes, a double-quoted one
    holding an unescaped quote, a plain scalar opening on a YAML indicator,
    one holding `": "`, one holding `" #"` — and answers DESC-BAD rather than
    a length. This returns the length of what was written for every one of
    them, and what is lost is the malformedness: a materialiser rebuilding a
    description of that length writes a WELL-FORMED one, so the rebuilt tree
    cannot reproduce DESC-BAD and no consumer can re-take that verdict. A
    shape carries the length and drops the verdict, which is accepted until a
    rule needs it.

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
    here is what is written, which is the hook's answer.
    """
    value = line.split(":", 1)[1].strip()
    if value[:1] in ('"', "'") and len(value) >= 2 and value[-1] == value[0]:
        inner = value[1:-1]
        value = inner.replace("''", "'") if value[0] == "'" else inner.replace(
            '\\"', '"'
        )
    return len(value)


def _top_level_key(line: str):
    """The key one frontmatter line declares, or None if it declares none.

    `memory_integrity._frontmatter`'s own test, spelled its way rather than
    as a literal prefix: `description : text` is a description to YAML and to
    the checker, and read here as a prefix it was no description at all —
    four false flags and a null length for a file the checker measures. So
    the colon is partitioned rather than matched, the key stripped, a key
    holding a space refused, and an indented or commented line skipped.
    """
    if not line or line[0].isspace() or line.startswith("#"):
        return None
    key, sep, _ = line.partition(":")
    key = key.strip()
    if not sep or not key or " " in key:
        return None
    return key


def _frontmatter(text: str, truncated: bool = False) -> dict:
    """Which of the four frontmatter facts a memory file carries.

    No YAML parser, because there is none in the standard library and this runs
    where nothing can be installed. The fence is a line 1 STARTING `---`
    through the next line starting `---`, which is the checker's own test:
    `----` opens frontmatter there, and a shape is only worth carrying if it
    is the answer the checker would give about the same file. A BOM, a fence
    below line 1 and an empty document are not frontmatter to either.

    THE ONE SPELLING THE TWO DISAGREE ON is a fence that never closes in a
    read the CAP CUT SHORT, and `truncated` is how this hears that: the
    checker reads an unclosed fence as frontmatter running to the end of the
    file, and a read that stopped at `FRONTMATTER_BYTES` cannot tell a file
    whose fence closes just past the cap from one whose fence never closes at
    all. Reporting the keys that happen to sit above the cut is a set of keys
    nobody has. An unclosed fence in a read that reached the end of the file
    is not that case — nothing was cut, so it is read the checker's way, to
    the end.

    AND A FENCE PAIR ENCLOSING NO TOP-LEVEL KEY is not frontmatter to either.
    The checker returns its keys and has none to return; reporting the fence
    as present with all four facts false is a shape that disagrees with the
    rule a consumer would run over the rebuilt file.
    """
    absent = {
        "has_frontmatter": False,
        "has_description": False,
        "description_len": None,
        "has_name": False,
        "has_type": False,
    }
    lines = text.splitlines()
    if not lines or not lines[0].startswith(_FENCE):
        return absent
    end = None
    for index in range(1, len(lines)):
        if lines[index].startswith(_FENCE):
            end = index
            break
    if end is None:
        if truncated:
            return absent
        end = len(lines)
    # What the checker reads is the TEXT after the opening `---`, so the
    # remainder of line 1 is a frontmatter line to it and `---name: x`
    # declares `name`. A line-based read that started at line 2 dropped it,
    # which showed up only once the enclosed keys had to be counted.
    fence = [lines[0][len(_FENCE):]] + lines[1:end]
    if not any(_top_level_key(line) is not None for line in fence):
        return absent
    has_name = has_type = has_description = False
    description_len = None
    in_metadata = False
    for line in fence:
        key = _top_level_key(line)
        if line.strip() and not line[:1].isspace():
            in_metadata = key == "metadata"
        if key == "name":
            has_name = True
        if key == "type" or (in_metadata and _NESTED_TYPE_RE.match(line)):
            has_type = True
        if not has_description and key == "description":
            has_description = True
            description_len = _description_len(line)
    return {
        "has_frontmatter": True,
        "has_description": has_description,
        "description_len": description_len,
        "has_name": has_name,
        "has_type": has_type,
    }


def _resolves_inside(path: str, directory: str) -> bool:
    """Whether the name `path` resolves to something under `directory`.

    THE ONE RULE FOR FOLLOWING A LINK, and it is the walk's own boundary
    rather than a judgement about the link: a name that resolves inside the
    directory being walked reaches bytes this capture is already reading, and
    one that resolves outside reaches somebody else's file through a name in
    here. `realpath` on both sides, because the memory directory is itself a
    link on a machine where the harness's own tree is wired somewhere else,
    and a comparison of the two spellings answers no for every file in it.
    """
    root = os.path.realpath(directory)
    return os.path.realpath(path).startswith(root + os.sep)


def _read_head(path: str) -> tuple:
    """`(head, truncated, size)` for the first `FRONTMATTER_BYTES` of `path`.

    `head` is None if it could not be read, rather than `""`: an unreadable
    file and an empty one produced the same four `false` flags, and a capture
    taken over ssh under `sudo -n` into an NFS home is exactly where the
    difference lives.

    AND THE CAP SAYS SO, which is what `truncated` is: a frontmatter fence
    that closes past the cap reads here as no frontmatter at all — four false
    flags, a null length and no counter moved — which is byte-identical to a
    file that genuinely has none. One byte over the cap is what tells the two
    apart, and it is the same answer `_index` already gives about its own.

    READ AS BYTES AND DECODED AFTER, because the cap is named in bytes and a
    text stream's `read(n)` counts CHARACTERS: 40,000 CJK characters are
    40,000 to a text read and 120,000 bytes off the disk, which is the read
    the cap exists to bound. Cutting on a byte boundary can halve a character,
    and `replace` is what stands in its place — the same answer this read has
    always given for bytes that are not UTF-8.

    AND THE SIZE COMES BACK WITH IT, off the descriptor this already holds,
    because it is the only size that describes the same inode as the facts
    read here: a caller asking the NAME afterwards asks about whatever that
    name means by then, which for a link is the link and for a file rewritten
    under the walk is a different file. `size` is None where nothing was
    opened, and the caller answers for that case itself.
    """
    try:
        with _open_regular_bytes(path) as handle:
            raw = handle.read(FRONTMATTER_BYTES + 1)
            # AFTER the read, not at the open: a file that grows past the cap
            # while it is being read would otherwise be recorded as truncated
            # beside a size at or below the cap.
            size = os.fstat(handle.fileno()).st_size
    except OSError:
        return None, False, None
    head = raw[:FRONTMATTER_BYTES].decode("utf-8", "replace")
    return head, len(raw) > FRONTMATTER_BYTES, size


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

    EXACTLY the targets that ESCAPE, which is what the rule is for. Any
    separator at all counted as an escape, so `hot/beads.md` — a real file one
    directory down, inside the directory being walked — was booked dangling
    without being looked at, and a tiered index read as entirely broken.
    """
    if not target or os.path.isabs(target) or target in (os.curdir, os.pardir):
        return True
    collapsed = os.path.normpath(target)
    return (
        os.path.isabs(collapsed)
        or collapsed == os.pardir
        or collapsed.startswith(os.pardir + os.sep)
    )


class _Unlooked(Exception):
    """A row this run was not allowed to look at, carrying its target."""


def _row_present(root: int, target: str) -> bool:
    """Whether `target` names something in the directory `root` is open on.

    COMPONENT BY COMPONENT, each asked of the level above it. `lexists` leaves
    only the LAST component unfollowed and follows every one before it, so a
    row reading `link-out/x.md` stats somebody else's file through a name in
    here — the same stat the string rule exists to prevent, one component
    over, under a `sudo -n` that was given this directory and not the target.

    `O_NOFOLLOW` on each intermediate component is what makes that a dangling
    row rather than a walk out of the directory, and the last is a `stat` that
    does not follow either: a row pointing at a dead symlink counts as
    present, because the file is there to be moved and that is what the row is
    about.

    THREE ANSWERS AND NOT TWO. `False` is a row that points at nothing, and
    `_Unlooked` is a row this run was not allowed to ask about — two states a
    single `False` reported as the same broken index, on a machine where being
    refused a directory is the documented case rather than the strange one.
    """
    parts = os.path.normpath(target).split(os.sep)
    current = None
    try:
        # ACQUIRED INSIDE the handler that releases it. Taken one line above,
        # an exhausted descriptor table escaped every handler in the walk and
        # ended the capture of the whole machine, where the row that asked for
        # the descriptor is the only thing that could not be measured.
        current = os.dup(root)
        for part in parts[:-1]:
            below = _open_dir(part, dir_fd=current)
            os.close(current)
            current = below
        os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
    except ValueError:
        # `ValueError` and not only `OSError`: a row target is adopter-authored
        # text and a NUL byte in it is valid UTF-8 that survives the read, and
        # `os.stat` and `os.open` refuse a path holding one before the kernel
        # sees it. A row nothing can look up is a row pointing at nothing,
        # which is what `lexists` — guarded the same way in the stdlib — used
        # to answer here; narrower, one such row ended the capture of the whole
        # machine.
        return False
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            # THE ROW REALLY POINTS AT NOTHING: the name is not there, or a
            # component of it is not a directory. `ELOOP` sits here with
            # `ENOTDIR` because a link at a component is a decision this
            # function made rather than a failure — `O_NOFOLLOW` answers one
            # errno on one kernel and the other on the next for the same link.
            return False
        # ANY OTHER ERRNO IS A QUESTION NOBODY GOT TO ASK, and answering it
        # `False` scored the row on a lookup that never ran: EACCES on a tier
        # — `sudo -n` into an NFS home under root-squash, which this file names
        # as the expected failure — booked every row beneath it as dangling
        # with `skipped` and `read_errors` both left at zero.
        raise _Unlooked(target) from None
    finally:
        if current is not None:
            os.close(current)
    return True


def _index(memory_dir: str, listed: list) -> tuple:
    """Row counts for `MEMORY.md`, and how many of the rows point at nothing.

    A PAIR: the counts, and how many rows this run was refused a look at. The
    second number belongs to the whole capture rather than to the index — see
    `read_errors` — so it leaves here separately instead of as a field.

    `dangling_rows` is a COUNT and never a name. NOTHING JUDGES A
    HARNESS-WRITTEN INDEX TODAY — no rule in this repository reads one — so
    this is the number an ORPHAN-style rule would be judged on if it existed,
    and it is here because a rebuilt corpus has to be able to reproduce it.
    `truncated` is how the count says it was taken off less than the file.

    CAPPED at `FRONTMATTER_BYTES` like every other read here. The cap exists
    so one pathological file cannot turn a capture into a read of somebody's
    whole disk, and an index was the one read that did not honour it.

    None when the index is a link out of the directory being walked, on the
    same rule every other file here is read by: the rows would be counted off
    somebody else's file, reached through a name in this directory, under a
    `sudo -n` that was given the directory and not the target. The file is
    still listed, with its `is_symlink` flag — a null index beside a linked
    `MEMORY.md` is a directory whose index was not read, and a null index
    beside no `MEMORY.md` at all is a directory that has none.

    None ALSO when the index is there and its bytes could not be read: a mode
    the capture has no rights to, a name that is not a plain file, the NFS
    home under `sudo -n`. Zeroes said the index had no rows, which is a
    measurement nobody took and is what a genuinely empty index records; the
    read failure is counted, once, on the file's own pass through the
    directory listing.
    """
    path = os.path.join(memory_dir, INDEX_NAME)
    if os.path.islink(path) and not _resolves_inside(path, memory_dir):
        return None, 0
    head, truncated, _ = _read_head(path)
    if head is None:
        return None, 0
    lines = head.splitlines()
    if truncated and lines and not head.endswith(("\n", "\r")):
        # What the cap cut is a fragment of a line — UNLESS it fell on a line
        # boundary, where the last line is whole and dropping it loses a row
        # the file has.
        lines.pop()
    present = set(listed)
    rows = dangling = unlooked = 0
    try:
        # ONCE, and every row is judged from it: a directory reopened by name
        # per row is a different directory each time somebody wants it to be.
        root = _open_dir(os.path.realpath(memory_dir))
    except OSError:
        # No descriptor on the directory the rows are about, so no row in it
        # can be looked at — a count nobody took, which is what None says.
        return None, 0
    try:
        for line in lines:
            match = _INDEX_ROW_RE.match(line)
            if match is None:
                continue
            rows += 1
            target = match.group(1).strip()
            if target in present:
                continue
            if _outside(target):
                dangling += 1
                continue
            try:
                found = _row_present(root, target)
            except _Unlooked:
                unlooked += 1
                continue
            if not found:
                dangling += 1
    finally:
        os.close(root)
    return (
        {"rows": rows, "dangling_rows": dangling, "truncated": truncated},
        unlooked,
    )


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

    WHAT IS OPENED is decided by `_resolves_inside` and by nothing else, for
    every `.md` here including the index: a link back into this directory
    names bytes the capture is reading anyway, and a link out of it names
    somebody else's file. A memory directory that is itself a link is followed
    — that is the home-manager machine, wired somewhere else on purpose — and
    the rule is then about the directory it resolves to.
    """
    files = []
    read_errors = 0
    for name, linked in listed:
        path = os.path.join(memory_dir, name)
        failed = False
        size = None
        head = ""
        truncated = False
        unreadable = False
        # A LINK IS FOLLOWED ONLY BACK INTO THIS DIRECTORY. One that resolves
        # out of it reaches somebody else's file through a name in here, and
        # `sudo -n` was given the directory rather than the target.
        if not linked or _resolves_inside(path, memory_dir):
            # ONE RECORD, ONE INODE, ONE INSTANT: the size arrives with the
            # read, off the descriptor the read held. Asked of the name
            # afterwards it described whatever the name meant then — a link
            # back into this directory is opened and read, and an `lstat` does
            # not follow it, so a nine-byte name was recorded as carrying its
            # target's 120-character description.
            head, truncated, size = _read_head(path)
            if head is None:
                head, failed, unreadable = "", True, True
        else:
            # AND A FILE NOBODY OPENED IS NOT ONE EITHER. The branch above is
            # not taken at all here, so the four facts have no reader behind
            # them and the record says so the same way. `read_errors` does not
            # move: nothing failed, the rule declined to look.
            unreadable = True
        if unreadable:
            try:
                # `lstat`, and ONLY where nothing was opened: the size of the
                # LINK, never of what it points at. A memory file that is a
                # link out of the directory had the target's size recorded,
                # which is a measurement of a file outside the capture — and
                # where the read itself failed, the name is all there is.
                size = os.lstat(path).st_size
            except OSError:
                failed = True
        record = {
            "name": names.file(name) if anonymise else name,
            "size": size,
            "is_symlink": linked,
        }
        record["unreadable"] = unreadable
        # A FILE NOBODY COULD READ IS NOT A FILE WITH NO FRONTMATTER, which is
        # what the four flags and the null length said — the same record a
        # readable file with a plain first line produces, and the only trace
        # was the whole-capture `read_errors` total, which no record can be
        # attributed to. Null is what the index answers about its own
        # unreadable read; here it is per fact, because the name, the size and
        # the link flag come from an `lstat` that did succeed.
        if unreadable:
            record.update(dict.fromkeys(_frontmatter(""), None))
            record["frontmatter_truncated"] = None
        else:
            record.update(_frontmatter(head, truncated))
            record["frontmatter_truncated"] = truncated
        files.append(record)
        if failed:
            read_errors += 1
    names_listed = [name for name, _ in listed]
    index = None
    if INDEX_NAME in names_listed:
        # A ROW NOBODY WAS ALLOWED TO LOOK AT IS COUNTED WITH THE FILES NOBODY
        # COULD READ. `dangling_rows` is a finding about the machine's index
        # and there is no field that says a lookup was refused, so a refusal
        # counted there is a wrong number in a document that exits 0.
        index, unlooked = _index(memory_dir, names_listed)
        read_errors += unlooked
    return (
        {
            "key": names.key(key) if anonymise else key,
            "key_len": len(key),
            "is_symlink": os.path.islink(memory_dir),
            "project_is_symlink": project_is_symlink,
            "files": files,
            "index": index,
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
    memory_dirs = []
    memory_dirs_total = skipped = read_errors = 0
    keys = []
    projects_total = 0
    try:
        with os.scandir(projects_root) as entries:
            for entry in entries:
                projects_total += 1
                try:
                    keys.append((entry.name, entry.is_symlink()))
                except OSError:
                    # A per-entry failure is a fact about that ENTRY. `scandir`
                    # answers `is_symlink` from the readdir record where the
                    # filesystem supplies one and stats the name where it does
                    # not, so on the filesystems that do not the one project a
                    # capture cannot reach used to end the run for all of them.
                    # It is counted here and still counted in `projects_total`.
                    skipped += 1
    except FileNotFoundError:
        if projects_total or os.path.lexists(projects_root):
            # THE NAME IS THERE and does not resolve — `projects/` as a link
            # to somewhere that has been moved or unmounted, which `scandir`
            # reports the same way it reports no name at all. That machine has
            # projects and this run cannot see them, so it is the failure
            # below and not the empty one above it.
            #
            # AND THE NAME IS ASKED AFTER THE RAISE, so a `projects/` that went
            # away UNDER the walk — a dropped mount, a moved link — is a name
            # nobody can find by the time the question is put, and the arm read
            # a machine mid-unmount as a machine with nothing on it. An
            # iteration that already served an entry is not that machine, and
            # unlike the name, that fact cannot change while this is asked.
            raise
        # A config directory with no `projects/` is a machine with nothing
        # written yet, and that is a shape. EVERY OTHER failure is not: the
        # directory is there and could not be listed, and the zeros below it
        # read as a healthy empty machine — which over `sudo -n` into an NFS
        # home under root-squash is the reachable failure, on a host the
        # operator may not get a second run at. It leaves here as an
        # exception, and `main` turns it into a message and an exit 2.
        keys = []
        projects_total = skipped = 0
    keys.sort()
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
        "projects_total": projects_total,
        "memory_dirs_total": memory_dirs_total,
        # TWO counters, because a half-failed capture that is
        # byte-indistinguishable from a complete one is a capture nobody can
        # act on: `skipped` is a project this run could not look into — the
        # entry itself unreadable, or its memory directory unlistable —
        # `read_errors` a file inside one that could not be measured, or an
        # index row whose lookup was refused.
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
    different directories and the lexical one is not where the bytes go. This
    is where the descriptor below is opened, and every question after that is
    asked of the descriptor rather than of a name.
    """
    return os.path.realpath(os.path.dirname(destination) or os.curdir)


def _open_dir(path: str, dir_fd=None) -> int:
    """A descriptor for the DIRECTORY at `path`, or an `OSError`.

    `O_DIRECTORY` so a file standing at the name is refused rather than
    opened, and `O_NOFOLLOW` so a link planted at it is not followed.
    """
    return os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=dir_fd,
    )


class _Refused(Exception):
    """A destination this will not write to, carrying the operator's line."""


class _Landing:
    """The directory `--out` lands in, held OPEN for as long as it is needed.

    Three rounds of this file closed one spelling each of the same defect — a
    link at the name, a link above it, a `..` taken from a link's target, a
    second hard name — and the fourth was a parent directory renamed between
    the check and the open, which no spelling of a name can close: what a name
    means is the attacker's to change, and they get to choose when. So the
    parent is opened ONCE, every question is asked of that descriptor, and the
    file is created relative to it. The directory that was judged is the
    directory written into by construction rather than by string equality.

    The parent may not exist yet, so what is opened is the deepest ancestor
    that does and the rest are made below it at write time — after the
    refusals, because a rejected `--out` that left half a path behind it is
    the same guard failing one step earlier. Judging the ancestor answers the
    same question: what is made below it is made empty, and no `.git` can
    appear in a directory this run just created.
    """

    def __init__(self, directory: str, leaf: str) -> None:
        base = directory
        missing = []
        while not os.path.isdir(base):
            base, tail = os.path.split(base)
            if not tail:
                break
            missing.append(tail)
        self.directory = directory
        self.leaf = leaf
        self.missing = list(reversed(missing))
        # The name of the level the descriptor was opened at, for the one
        # question that cannot be asked of a descriptor.
        self.judged = base
        # The level a leaf was created in, taken only once there is a leaf to
        # remove — see `discard`.
        self.made_in = None
        # And WHICH FILE was created in it, as `(st_dev, st_ino)` off the
        # descriptor that made it. A name is not an identity here.
        self.created = None
        self.fd = _open_dir(base)

    def close(self) -> None:
        """Every descriptor this took, whether or not it created anything."""
        if self.made_in is not None:
            os.close(self.made_in)
            self.made_in = None
        os.close(self.fd)

    def inside_worktree(self) -> bool:
        return _worktree_above(self.fd, self.judged)

    def create(self, judge_worktree: bool = False) -> int:
        """The destination created below the judged directory, or a refusal.

        `O_EXCL`: `--out` CREATES its destination and never writes over one.
        Whatever is already at the name — a file, a link, a second name for
        somebody else's file — is a choice about what this capture destroys,
        made by whoever could write in that directory, and this runs under
        `sudo -n` on hosts it is a guest on. The mode is owner-only for the
        same reason and applies because the inode is this run's: `--raw`
        carries real usernames, org names and repository paths, and a default
        umask leaves them readable by everybody else logged into that machine.

        No `O_NOFOLLOW` at this name and no fd-side questions after it: with
        `O_EXCL` a link is EEXIST whether it dangles or not, and what comes
        back is an inode this call made, so there is nothing left to ask about
        what was opened.

        A LEVEL THAT ALREADY EXISTS IS A REFUSAL and not a level to adopt.
        Every name in `missing` was absent when the descriptor above it was
        judged, so one that is there now was made by somebody else while the
        capture ran, and nothing has asked what it is; the whole point of
        judging the ancestor is that what is made below it is made empty.

        `judge_worktree` re-asks the checkout question of the descriptor the
        file was just created relative to, which is the only descriptor that
        answers about where the bytes are actually going: the pre-flight
        answered before the capture, and a whole capture is time enough to
        rename the judged directory into a checkout. The file is `O_EXCL`-fresh
        and empty, so unlinking it destroys nothing.

        AND THE LEVEL THE LEAF WAS MADE IN IS KEPT, because the other thing
        that can happen to this file is a write that fails part-way, and by
        then the name is not a way back to that directory — see `discard`.
        The INODE is kept with it and for the same reason: the level is where
        the file was made, and the name in it is not proof of which file.
        """
        current = os.dup(self.fd)
        try:
            for name in self.missing:
                made = True
                try:
                    os.mkdir(name, dir_fd=current)
                except FileExistsError:
                    # Whatever is at the name is opened below rather than
                    # judged here, so a plain file still refuses as the
                    # `Not a directory` it is.
                    made = False
                below = _open_dir(name, dir_fd=current)
                os.close(current)
                current = below
                if not made:
                    raise _Refused(
                        f"{name} already stands under this path, and --out "
                        f"makes the levels it writes below"
                    )
            try:
                fd = os.open(
                    self.leaf,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=current,
                )
            except FileExistsError:
                raise _Refused(_occupant(current, self.leaf)) from None
            # WHAT THIS RUN MADE, off the fresh descriptor and not off the
            # name: by the time anything is removed the handle is closed, so
            # `os.fstat` is no longer available to ask and the name is the one
            # thing somebody else can have changed in between.
            mine = os.fstat(fd)
            self.created = (mine.st_dev, mine.st_ino)
            if judge_worktree:
                try:
                    inside = _worktree_above(current, self.directory)
                except _Refused:
                    self._discard(fd, current)
                    raise
                if inside:
                    self._discard(fd, current)
                    raise _Refused(
                        "--raw refuses to write inside a git worktree; a shape "
                        "committed with real names is the one mistake this "
                        "tool exists to prevent"
                    )
            try:
                self.made_in = os.dup(current)
            except OSError:
                # With no descriptor of the level there is no way to remove
                # this afterwards, so it goes now: an empty file at the
                # operator's name would refuse their next run for nothing.
                self._discard(fd, current)
                raise
            return fd
        finally:
            os.close(current)

    def discard(self) -> bool:
        """Whether what this run created is gone from the destination name.

        A write that failed part-way left a truncated document at the
        operator's own name with mode 0600, and the `O_EXCL` create then
        refused their retry: two guarantees that are each right and compose
        into a destination that is neither written nor retryable, on a host an
        unattended capture may not get a second run at. What `O_EXCL` proved is
        that an INODE is this run's, and the descriptor of the level the leaf
        was made in carries that proof only as far as the directory: the name
        inside it is still whatever it now means, so a file moved aside and
        replaced at the name was destroyed by a removal that reported success.
        The inode recorded at the create is what the name is held to, and a
        name carrying anything else is left alone.

        The stat and the unlink are two calls and the window between them is
        real: a replacement published inside it, or an inode number reused
        there, is still removed. What that costs is bounded by what the guard
        above it already refuses, and closing it takes a different shape —
        writing into a private directory and publishing with a link that
        refuses an existing destination, so the name this run removes is one
        nobody else can reach.

        FALSE IS THE ANSWER THE CALLER'S LINE TURNS ON rather than a second
        failure to report: a directory whose mode changed under the run keeps
        the file, and an operator told otherwise is sent at a retry that will
        be refused.
        """
        if self.made_in is None:
            return False
        gone = True
        try:
            # `os.stat` and not `os.lstat`: only the first is in
            # `os.supports_dir_fd` on the 3.8 floor this has to run on, and
            # `follow_symlinks=False` so a link planted at the name answers
            # about itself rather than about what it points at.
            now = os.stat(self.leaf, dir_fd=self.made_in, follow_symlinks=False)
            if (now.st_dev, now.st_ino) != self.created:
                gone = False
            else:
                os.unlink(self.leaf, dir_fd=self.made_in)
        except FileNotFoundError:
            # Somebody else took the name away, which leaves the retry exactly
            # where this removing it would have.
            pass
        except OSError:
            gone = False
        os.close(self.made_in)
        self.made_in = None
        return gone

    def _discard(self, fd: int, current: int) -> None:
        """The just-created leaf removed, before anything is written to it."""
        os.close(fd)
        # A name somebody else has already taken away is not a name to chase:
        # the refusal stands either way, and no byte of the shape was written.
        # And a name somebody else has taken OVER is not this run's to remove,
        # for the reason `discard` gives — this window is shorter, not absent.
        with contextlib.suppress(OSError):
            now = os.stat(self.leaf, dir_fd=current, follow_symlinks=False)
            if (now.st_dev, now.st_ino) == self.created:
                os.unlink(self.leaf, dir_fd=current)


def _occupant(fd: int, leaf: str) -> str:
    """Why `leaf` could not be created below `fd`, in an operator's terms.

    The refusal is one rule — the name is taken — but which way it is taken is
    what tells a staged link apart from a second run of the same command.
    """
    try:
        # `os.stat` and not `os.lstat`: only the first is in
        # `os.supports_dir_fd` on the 3.8 floor this has to run on.
        info = os.stat(leaf, dir_fd=fd, follow_symlinks=False)
    except OSError:
        return "something stands at this name, and --out creates its destination"
    if stat.S_ISLNK(info.st_mode):
        return "a symlink sits at this name, and --out refuses to follow one"
    if stat.S_ISDIR(info.st_mode):
        # Before the link count, which every directory fails.
        return "a directory stands at this name, and --out creates its destination"
    if not stat.S_ISREG(info.st_mode):
        return "this name is not a file, and --out creates its destination"
    if info.st_nlink > 1:
        return (
            "another name points at this file, and --out refuses to overwrite "
            "through one"
        )
    return (
        "a file is already at this name, and --out creates its destination "
        "rather than writing over one"
    )


def _git_says_worktree(directory: str) -> bool:
    """git's own answer about `directory`, for the trees no `.git` name marks.

    A work tree attached to a BARE repository — the dotfiles pattern, and a
    `GIT_DIR` exported over ssh — has no `.git` anywhere inside it, and `git
    status` still lists a shape written there as untracked in a real checkout,
    one `git add -A` from being committed. A name walk cannot see that, and no
    walk can: which directory is a work tree is a fact about a repository
    somewhere else. So the last question is asked of git.

    A host with no git on PATH is left with the walk's answer, which is the
    behaviour every earlier round had.
    """
    try:
        answer = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return answer.returncode == 0 and bool(answer.stdout.strip())


def _worktree_above(fd: int, directory: str) -> bool:
    """Whether a file created in the directory `fd` names lands in a checkout.

    The walk is on DESCRIPTORS: `.git` is asked of each level with `dir_fd`,
    and the next level is that level's own `..`. A directory renamed or
    replaced while this runs is still the one being judged, and it is the one
    the caller goes on to create in; a walk by name answers about whatever
    each name means at the moment that step is taken.

    `stat` without following rather than `exists`: a linked worktree's `.git`
    is a FILE and a link at that name marks a checkout too, dangling or not.
    The top is where a directory and its own `..` are the same inode.

    `GIT_WORK_TREE` is compared by inode along the same walk, so a work tree
    named through a different path — or one the destination merely sits
    inside — still matches. `directory` is the name of the level the walk
    STARTED at, and it is carried only for the question git is asked when the
    walk finds nothing.
    """
    marks = set()
    declared = os.environ.get("GIT_WORK_TREE")
    if declared:
        try:
            named = os.stat(declared)
        except OSError:
            pass
        else:
            marks.add((named.st_dev, named.st_ino))
    current = os.dup(fd)
    try:
        while True:
            try:
                os.stat(".git", dir_fd=current, follow_symlinks=False)
                return True
            except OSError:
                pass
            here = os.fstat(current)
            if (here.st_dev, here.st_ino) in marks:
                return True
            try:
                above = os.open(
                    "..", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), dir_fd=current
                )
            except OSError as exc:
                # A level this cannot open is a level this cannot clear, and
                # the walk stopping early would report the checkout it never
                # reached as no checkout at all.
                raise _Refused(
                    f"a directory above this one will not open, so whether "
                    f"the destination is inside a git worktree cannot be "
                    f"answered: {exc.strerror or exc}"
                ) from None
            os.close(current)
            current = above
            info = os.fstat(current)
            if (info.st_dev, info.st_ino) == (here.st_dev, here.st_ino):
                break
    finally:
        os.close(current)
    return _git_says_worktree(directory)


def _inside_worktree(directory: str) -> bool:
    """Whether a file written in `directory` would land in a git checkout.

    A DIRECTORY and not a file, because the two callers name one differently:
    `--out` is a file that does not exist yet, so its parent is what there is
    to walk, and a stdout redirect names no path at all — only the directory
    the process stands in.

    A directory that will not open is a REFUSAL and not a False. A landing
    directory at mode 0333 is writable and searchable by a shell and opaque to
    `O_RDONLY`, so answering False there let the walk report an unread
    checkout as no checkout — the leak this whole mechanism exists to stop,
    with exit 0 and an empty stderr.
    """
    try:
        fd = _open_dir(os.path.realpath(directory))
    except OSError as exc:
        raise _Refused(
            f"this directory will not open, so whether it is inside a git "
            f"worktree cannot be answered: {exc.strerror or exc}"
        ) from None
    try:
        return _worktree_above(fd, os.path.realpath(directory))
    finally:
        os.close(fd)


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


def _drop_stdout() -> None:
    """Fd 1 pointed at the null device, after a failure has been reported.

    The interpreter flushes `sys.stdout` once more on its way out, and a
    stream whose write or flush just failed fails there too — a second report,
    after this run has already made its one, on the channel a wrapper reads as
    this tool's.
    """
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        # The descriptor NUMBER and not `sys.stdout.fileno()`: the stream this
        # stands in for may be too far gone to answer for itself.
        os.dup2(null, 1)
    except OSError:
        pass
    finally:
        os.close(null)


def main(argv=None) -> int:
    """The whole run, and the one place its exit contract is decided.

    ONE boundary for the whole family. Four separate closures each guarded one
    member at one site — a NUL byte reaching `os.stat`, a descriptor taken
    above the handler that releases it, an unguarded `..` open, an unguarded
    write to a pipe — and what they had in common was an exception set
    narrower than the syscall's, one site at a time, with the next site
    unguarded. A capture is read by a wrapper on the far end of an ssh pipe
    that reads the number: every way the machine underneath this run can fail
    is one line and a 2 here.

    `ValueError` and `UnicodeError` alongside `OSError` because a config
    directory is somebody else's text — a NUL byte in a path reaches the
    stdlib as a `ValueError` before the kernel sees it, a name that is not
    UTF-8 reaches an encode as a `UnicodeError` — and `RecursionError`
    because the depth of an index chain is theirs to choose too.
    `MemoryError` for the same reason one step further out: the SIZE of what
    is read is theirs as well, and a parse that ran out is a machine that
    failed under this run rather than a fault in it. The caps are what make it
    unlikely; the boundary is what makes it a 2 when it happens anyway.

    Interior code catches only to BOOK a measurement — a file it could not
    read, a row it could not look at, an entry it skipped — or to carry a
    decision this tool made for a reason of its own, which keeps its own line.
    """
    try:
        return _run(argv)
    except (OSError, ValueError, RecursionError, UnicodeError, MemoryError) as exc:
        # Fd 1 before the report and not after it: the failure may BE the
        # stdout write, whose bytes are still buffered.
        _drop_stdout()
        where = getattr(exc, "filename", None)
        why = getattr(exc, "strerror", None) or exc
        detail = f"{where}: {why}" if where else f"{why}"
        sys.stderr.write(f"harness_shape: {detail}\n")
        return 2


def _run(argv) -> int:
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

    landing = None
    if args.out:
        if os.path.basename(args.out) in ("", os.curdir, os.pardir):
            # Before the symlink check below, which these spellings fail for a
            # reason it does not have: `abspath` drops the trailing separator
            # and `realpath` keeps it, so the two disagree with no link
            # anywhere, and the operator is told to look for one.
            sys.stderr.write(
                f"harness_shape: {args.out}: this names a directory and not a "
                f"file to create; --out writes one file, so name it inside "
                f"that directory\n"
            )
            return 2
        # `abspath` calls `os.getcwd()` and `realpath`'s `os.readlink` is
        # unguarded, so both of these raise for a working directory that has
        # been removed. Nothing catches them here: the boundary in `main`
        # answers for every failure of the machine underneath this run.
        parent = os.path.dirname(os.path.abspath(args.out))
        resolved = _landing_dir(args.out)
        if resolved != parent:
            # O_NOFOLLOW guards the LAST component only, so a link one
            # level up chose the file that got truncated: `--out
            # real/linkdir/shape.json` wrote through `linkdir` and
            # overwrote whatever `target.json` behind it was. Refused
            # rather than followed, and the resolved path is named so an
            # operator whose home really is reached through a link — or who
            # named /tmp on a mac — can pass that path instead.
            sys.stderr.write(
                f"harness_shape: {args.out}: a symlink stands in this path, "
                f"which chooses what gets overwritten; it resolves to "
                f"{resolved} — on macOS /var is itself a link, so a path "
                f"under $TMPDIR lands here and the resolved one above is "
                f"the path to pass\n"
            )
            return 2
        landing = _Landing(resolved, os.path.basename(args.out))
    try:
        return _capture_and_write(args, landing)
    finally:
        if landing is not None:
            landing.close()


def _capture_and_write(args, landing) -> int:
    """The run itself, with the destination already open where there is one."""
    if args.raw:
        destination = args.out or _stdout_destination()
        try:
            if destination is not None:
                inside = (
                    landing.inside_worktree()
                    if landing is not None
                    else _inside_worktree(_landing_dir(destination))
                )
                if inside:
                    sys.stderr.write(
                        f"harness_shape: --raw refuses to write inside a git "
                        f"worktree ({destination}); a shape committed with real "
                        f"names is the one mistake this tool exists to prevent\n"
                    )
                    return 2
            elif _stdout_is_a_file() and _inside_worktree(os.getcwd()):
                # THE FALLBACK, for a kernel that will not name fd 1. The
                # working directory is the wrong question — a redirect from
                # outside a checkout into one is the leak, and this cannot see
                # it — so it is what is left rather than what is asked, and it
                # over-refuses.
                sys.stderr.write(
                    "harness_shape: --raw refuses a redirect to a file from "
                    "inside a git worktree; pass --out so the destination can "
                    "be checked, and name one outside the checkout\n"
                )
                return 2
        except _Refused as exc:
            # An unanswerable question is not a no. `--raw` carries real
            # names, so the run that cannot establish where the bytes land
            # writes none of them.
            sys.stderr.write(
                f"harness_shape: {destination or os.getcwd()}: {exc.args[0]}\n"
            )
            return 2
    config_dir = os.path.expanduser(args.config_dir)
    if not os.path.isdir(config_dir):
        sys.stderr.write(f"harness_shape: {config_dir} is not a directory\n")
        return 2

    # A capture that could not read the projects directory reports no shape
    # rather than an empty one, and nothing here catches for that: leaving by
    # the boundary is what makes an unread tree an exit 2 with no shape on
    # stdout.
    shape = capture(
        config_dir,
        anonymise=not args.raw,
        # The machine's policy file describes the tree being captured when
        # the tree is one this machine's harness runs on. `--managed` is
        # how the documented ssh flow says so for another user's home
        # under `sudo -n`, where the process's own default is root's.
        managed=args.managed or _is_own_config_dir(config_dir),
    )
    text = json.dumps(shape, indent=2) + "\n"
    if not args.out:
        if sys.stdout is None:
            # FD 1 CLOSED BEFORE THE INTERPRETER STARTED leaves no stream to
            # fail: CPython sets `sys.stdout` to None, the write is an
            # `AttributeError`, and that is not one of the ways a machine
            # fails, so the boundary does not map it. What failed is the
            # destination, and it failed the way a missing descriptor does.
            raise OSError(errno.EBADF, os.strerror(errno.EBADF), "stdout")
        sys.stdout.write(text)
        # FLUSHED INSIDE the contract. The interpreter's own flush happens
        # after this returns, where a reader that has gone away is an ignored
        # exception, an exit nobody chose and no line naming this tool — and
        # a shape of eight projects or more is already past the stream's
        # buffer, so which door an operator leaves by depended on the size of
        # the machine being captured.
        sys.stdout.flush()
        return 0
    try:
        fd = landing.create(judge_worktree=args.raw)
    except _Refused as exc:
        # A refusal is a decision with a reason of its own, so it carries its
        # own line; a destination that merely failed leaves by the boundary.
        sys.stderr.write(f"harness_shape: {args.out}: {exc.args[0]}\n")
        return 2
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        # WHAT THIS RUN CREATED GOES WITH THE FAILURE, and the line says which
        # way it went: the bytes that did land are a truncated document at the
        # operator's name, and the create would refuse their next run over it.
        note = (
            "nothing was left at this name"
            if landing.discard()
            else "the partial file it wrote is still there"
        )
        sys.stderr.write(
            f"harness_shape: {args.out}: {exc.strerror or exc}; {note}\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
