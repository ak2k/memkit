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
# A hook EVENT name as the harness spells every event it dispatches: letters
# and nothing else. A settings file can carry any key under `hooks`, and the
# ones that are not events are somebody's own gate, named after their org.
_HOOK_EVENT_RE = re.compile(r"[A-Za-z]+")


# --- settings ---------------------------------------------------------------


def _managed_dir() -> str:
    """`cli_doctor._managed_dir`, measured on 2.1.241 out of the binary."""
    if sys.platform == "darwin":
        return "/Library/Application Support/ClaudeCode"
    return "/etc/claude-code"


def _read_json(path: str):
    """The parsed object at `path`, or None for anything that is not one.

    None for BOTH "not there" and "there and unusable", because the two are
    told apart by the caller: `_settings` asks the filesystem whether the path
    exists and records the second as its own state. Omitting it said "no
    settings at that scope", which is a different machine — and one a
    materialiser reproduces by writing a file that does not parse.
    """
    try:
        with open(path, encoding="utf-8") as handle:
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
    """
    memory_keys = {}
    for key in MEMORY_KEYS:
        if key not in data:
            continue
        value = data[key]
        if key == DIRECTORY_KEY:
            memory_keys[key] = PATH_PLACEHOLDER if anonymise else value
        elif anonymise and not isinstance(value, bool):
            # PRESENT, and not a switch. `null` says the key was set to
            # something the harness reads as on, which is the fact a rebuilt
            # tree needs; the value itself is the adopter's.
            memory_keys[key] = None
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
        "plugins": [names.plugin(key) if anonymise else key for key in listed],
    }


def _settings(config_dir: str, names: _Pseudonyms, anonymise: bool) -> dict:
    """The two scopes a whole-tree capture can honestly read.

    `user` and the platform `managed` file, and NOT doctor's `local` or
    `project`: those two are `.claude/settings.json` and
    `.claude/settings.local.json` under the directory a SESSION stands in, and
    a capture of a config directory has no cwd to resolve them against. A
    scope whose file is absent is omitted rather than emitted empty, because
    "no managed settings on this machine" and "managed settings that set
    nothing" are different machines.
    """
    found = {}
    for scope, path in (
        ("user", os.path.join(config_dir, "settings.json")),
        ("managed", os.path.join(_managed_dir(), "managed-settings.json")),
    ):
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
            hint = None
        if install is not None and install not in INSTALL_METHODS:
            install = INSTALL_OTHER
    return {"version_hint": hint, "install": install}


# --- frontmatter ------------------------------------------------------------


def _folded_len(fence: list, start: int) -> int:
    """The length of one `description:` scalar, continuation lines included.

    A LENGTH, which is the whole of what a shape says about a description: the
    rule that fires on one is `>155 characters`, so the number is what a
    rebuilt corpus has to reproduce and the text is what it must not carry.
    Folded the way YAML folds it, one space per line break, and a lone block
    indicator is dropped because `>` is syntax rather than description.
    """
    head = fence[start].split(":", 1)[1]
    parts = [head.strip()]
    for line in fence[start + 1 :]:
        if line[:1] not in (" ", "\t") or not line.strip():
            break
        parts.append(line.strip())
    if parts and parts[0] in (">", "|", ">-", "|-", ">+", "|+"):
        parts = parts[1:]
    return len(" ".join(part for part in parts if part).strip())


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
    for index, line in enumerate(fence):
        if line.strip() and line[:1] not in (" ", "\t"):
            in_metadata = line.startswith("metadata:")
        if line.startswith("name:"):
            has_name = True
        if line.startswith("type:") or (in_metadata and _NESTED_TYPE_RE.match(line)):
            has_type = True
        if not has_description and line.startswith("description:"):
            has_description = True
            description_len = _folded_len(fence, index)
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
        with open(path, encoding="utf-8", errors="replace") as handle:
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
        # An event name is letters, and the harness's whole list of them is
        # public. Anything else under `hooks` is a key somebody chose, which
        # is where an org name lives.
        if _HOOK_EVENT_RE.fullmatch(event):
            return event
        return self._assign(self._hooks, event, "h")

    def plugin(self, key: str) -> str:
        plugin, sep, marketplace = key.partition("@")
        if not sep:
            return self._assign(self._plugins, key, "p")
        # memkit by name, its marketplace not: whether memkit is installed is
        # the fact a shape is allowed to keep, and where somebody hosts their
        # own marketplace is not.
        left = plugin if plugin == KEPT_PLUGIN else self._assign(
            self._plugins, plugin, "p"
        )
        return left + "@" + self._assign(self._markets, marketplace, "q")


# --- the walk ---------------------------------------------------------------


def _lock_age(project_dir: str, memory_dir: str, now: float):
    """Age of the consolidation lock in seconds, or None.

    The memory directory first and the project directory second, because the
    lock has been seen in both and the inner one is the current spelling. Never
    negative: a lock stamped in the future is a clock that disagrees, not a
    consolidation that has not started yet.
    """
    for candidate in (
        os.path.join(memory_dir, CONSOLIDATE_LOCK),
        os.path.join(project_dir, CONSOLIDATE_LOCK),
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

    `dangling_rows` is a COUNT and never a name. It is the number an adopter's
    own index is judged on, and the one a rebuilt corpus has to reproduce for
    the ORPHAN rule to fire the same number of times. No such rule reads a
    harness-written index today; this is the number one would need, and
    `truncated` is how it says it is not the whole file.

    CAPPED at `FRONTMATTER_BYTES` like every other read here. The cap exists
    so one pathological file cannot turn a capture into a read of somebody's
    whole disk, and an index was the one read that did not honour it.
    """
    path = os.path.join(memory_dir, INDEX_NAME)
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read(FRONTMATTER_BYTES + 1)
    except OSError:
        return {"rows": 0, "dangling_rows": 0, "truncated": False}
    truncated = len(text) > FRONTMATTER_BYTES
    lines = text[:FRONTMATTER_BYTES].splitlines()
    if truncated and lines:
        # What the cap cut is a fragment of a line, not a row.
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
    config_root: str,
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
    is_symlink = os.path.islink(memory_dir)
    target_kind = None
    if is_symlink:
        try:
            resolved = os.path.realpath(memory_dir)
        except OSError:
            # A link this process cannot resolve is a third state, and calling
            # it external would be a guess about where it points.
            target_kind = "unresolved"
        else:
            inside = resolved == config_root or resolved.startswith(
                config_root + os.sep
            )
            target_kind = "in-shape" if inside else "external"
    names_listed = [name for name, _ in listed]
    return (
        {
            "key": names.key(key) if anonymise else key,
            "key_len": len(key),
            "is_symlink": is_symlink,
            "project_is_symlink": project_is_symlink,
            "symlink_target_kind": target_kind,
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


def capture(config_dir: str, anonymise: bool = True) -> dict:
    """The whole shape of one config directory.

    A project directory is LISTED when it holds at least one `*.md`, index
    included — so an index-only directory appears, with its rows and no
    memories, which is a state two of memkit's own rules fire on. What
    `memkit.harness_memory.inventory` calls a corpus is the narrower set a
    consumer derives from this: the listed directories holding some file other
    than the index.
    """
    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    config_root = os.path.realpath(config_dir)
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
        try:
            with os.scandir(memory_dir) as entries:
                listed = sorted(
                    (entry.name, entry.is_symlink())
                    for entry in entries
                    if entry.name.endswith(".md") and entry.is_file()
                )
        except OSError:
            # One unreadable directory is a fact about the machine, not a
            # reason to report nothing about the other three thousand.
            skipped += 1
            continue
        if not listed:
            continue
        record, failed = _memory_dir(
            key, project_dir, project_is_symlink, memory_dir, listed,
            config_root, names, anonymise, now,
        )
        read_errors += failed
        memory_dirs.append(record)
    return {
        "schema": SCHEMA,
        "tool": "harness_shape",
        "anonymised": anonymise,
        "harness": _harness(config_dir, anonymise),
        "settings": _settings(config_dir, names, anonymise),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness_shape.py",
        description="Capture the shape of a Claude Code config directory.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude",
        help="the harness config directory (default: $CLAUDE_CONFIG_DIR or ~/.claude)",
    )
    parser.add_argument("--out", help="write here instead of stdout")
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
        if args.out and _inside_worktree(
            os.path.dirname(os.path.abspath(args.out))
        ):
            sys.stderr.write(
                f"harness_shape: --raw refuses to write inside a git worktree "
                f"({args.out}); a shape committed with real names is the one "
                f"mistake this tool exists to prevent\n"
            )
            return 2
        if not args.out and _stdout_is_a_file() and _inside_worktree(os.getcwd()):
            # THE SPELLING THE DOCSTRING NAMES. "A debugging run whose
            # redirect happened to point at the repository" is `--raw >
            # somewhere`, and the guard above sees only the run where somebody
            # typed the destination and could read it back.
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
        shape = capture(config_dir, anonymise=not args.raw)
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
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
