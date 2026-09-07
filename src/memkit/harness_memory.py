"""Where the harness's own memory feature writes, and what it has written.

Claude Code writes auto-memory itself, flat, one directory per project, under
its config directory — so an adopter arriving at memkit already has a corpus,
and the question every surface here answers is where that corpus IS. Doctor
names it; a later command offers to move it. Both need the same three answers,
and deriving them twice is how two commands come to disagree about which
directory holds an adopter's memories.

MEASURED on 2.1.258, out of the shipped binary and against live `claude -p`
runs, superseding an earlier reading of the 2.1.232 and 2.1.240 schema text:

- `autoMemoryEnabled` decides whether the feature runs at all. False and the
  harness neither reads nor writes auto-memory.
- `autoMemoryDirectory` is a path, `~/` expanded, used AS-IS for every project:
  one flat directory, no per-project subdirectory under it.
- `autoDreamEnabled` false stops background consolidation ONLY. Memories are
  still written, which is why it is not the switch that turns the feature off.
- Unset, the directory is `<config dir>/projects/<project key>/memory`, and
  the key comes from the GIT REPOSITORY ROOT rather than from the cwd: every
  worktree of one repository shares one directory, and a SUBMODULE keys on
  itself.

A CHECKED-IN `.claude/settings.json` REDIRECTS IT. The schema description says
the key is "Ignored if set in projectSettings (checked-in .claude/settings.json)
for security"; that sentence is false in this build. The resolver consults
`projectSettings` whenever its trust gate passes, and the gate short-circuits
to true for every non-interactive invocation — `claude -p`, hooks, the SDK,
subagents — whatever the folder trust state is. An interactive session applies
it once the folder is trusted. So a clone can decide where an agent writes its
memories, which is why a value decided by that file is reported rather than
passed.

WHAT LANDS IN THAT DIRECTORY IS REWRITTEN. On every Write or Edit of a `.md`
file whose normalised path merely STARTS WITH the configured directory — a
string prefix, with no `realpath` — a file with parseable frontmatter is
re-serialised: `name` slugified, every other top-level key buried under
`metadata`, `node_type`, a session id and a timestamp added, comments and key
order lost. So pointing the key at a corpus root hands memkit's own memories to
somebody else's serialiser. A directory of the harness's own under the corpus
root is what keeps both: retrieval recurses, and the rewrite reaches only what
is inside it.

WHETHER THAT PREFIX ENDS IN A SEPARATOR IS NOT MEASURED, and it decides one
case: a SIBLING whose name merely starts with the configured directory's —
`auto-memory-old/` beside `auto-memory/` — is inside the rewrite under a bare
prefix test and outside it under a separator-terminated one. Every directory
this package recommends is one it also recommends creating, so nothing memkit
says rests on the answer; it is recorded because the remedy's promise that "the
rewrite reaches only what is in it" is exactly as wide as this.

THE ENVIRONMENT OUTRANKS EVERY SETTINGS FILE, on two surfaces this module reads
and one it does not. `CLAUDE_CODE_DISABLE_AUTO_MEMORY` decides the switch
before a settings file is opened at all (`env_switch`), and three more
variables decide the DIRECTORY the same way (`OVERRIDE_ENV`) — of which memkit
resolves none, and says so rather than naming a directory that is not the one
being written to.

`memoryDir` is not a key the harness reads. It was memkit's own earlier reading
of this feature, and a remedy naming it changed nothing on the adopter's
machine.
"""

from __future__ import annotations

import os
import re
import unicodedata

from memkit.memory_prompt_recall import (
    _repo_common_dir,
    _repo_root,
    _RootUnknown,
)

# The harness's own sanitiser for a project key: EVERY character outside
# `[A-Za-z0-9]` becomes `-`, one for one and with no run collapsing, so
# `/Users/x/.config/nix` keys as `-Users-x--config-nix` and a path holding a
# space or an underscore loses it the same way. Measured on 2.1.258.
_SANITIZE = re.compile(r"[^A-Za-z0-9]")

# Where the harness stops spelling the path out and starts hashing it: a key
# longer than this is truncated to it and given a `-<base36>` suffix derived
# from the unsanitised path. THE HASH IS NOT MEASURED, so this package cannot
# name that directory and says so rather than naming the prefix — a prefix is
# a directory nothing writes to, which is the answer this module refuses
# everywhere else.
KEY_MAX = 200

# The one file in an auto-memory directory that is an INDEX rather than a
# memory. It is carried in `files` because anything moving the directory has to
# move it too, and it is never counted, because a directory holding nothing but
# an index holds no memories.
INDEX_NAME = "MEMORY.md"

# A SUBMODULE keeps its git state under the SUPERPROJECT's, at
# `<super>/.git/modules/<name>`, and no `commondir` sits beside it — so the
# common-dir walk lands on `<super>/.git/modules`, one answer for every
# submodule of that superproject.
_SUBMODULE_GIT_DIR = os.sep + ".git" + os.sep + "modules" + os.sep

# The scopes the harness resolves these keys in, most authoritative first, and
# the FIRST ONE THAT DECLARES THE KEY WINS: policy settings, then the
# `--settings` flag, then `.claude/settings.local.json`, then the checked-in
# `.claude/settings.json`, then user settings — so `local` outranks `project`,
# and both outrank `user`.
#
# MEASURED FOR THE DIRECTORY, INFERRED FOR THE TWO BOOLEANS, and the difference
# is stated because nothing else here records it. `autoMemoryDirectory` is
# resolved through an entry resolver that RETURNS THE SCOPE its value came
# from, so which file won is observable. Read from the 2.1.258 code, the two
# switches go through a plain merged-settings accessor that reports no scope at
# all — one precedence almost certainly, but two code paths, and applying this
# tuple to them is an inference this package has not measured.
#
# It only decides anything when TWO scopes declare one key with different
# values; with one declaring scope every candidate order picks it. That is why
# doctor reports rather than passes the disagreeing case instead of guessing.
#
# The `--settings` scope has no entry here because it cannot be one: it names a
# file chosen per invocation, on a command line this process never sees.
SCOPE_ORDER = ("managed", "local", "project", "user")

# The one scope whose file is checked into the repository and travels with
# every clone of it. Named rather than compared inline, because what hangs off
# it is a status: a value this file decided is REPORTED, never passed.
CHECKOUT_SCOPE = "project"

# What to point `autoMemoryDirectory` at inside a corpus root, rather than at
# the root. Retrieval recurses, so a memory here is found; the harness's
# rewrite is a string prefix on the configured directory, so keeping that
# directory below memkit's own files is what keeps them out of its serialiser.
SAFE_SUBDIR = "auto-memory"

DIRECTORY_KEY = "autoMemoryDirectory"
ENABLED_KEY = "autoMemoryEnabled"
DREAM_KEY = "autoDreamEnabled"

# The variable the harness reads BEFORE any settings file, and the two word
# lists it reads it with. Read from the 2.1.258 code: the gate lower-cases and
# trims the value, takes `1/true/yes/on` as "do not run" and — this is the
# direction that matters — takes `0/false/no/off` as "RUN", returning before
# the settings are consulted at all. So a machine with `autoMemoryEnabled:
# false` in every scope has the feature running, and every settings file says
# otherwise.
#
# A value in neither list, and an empty one, decide nothing: both fall through
# to the settings, which is why this is a three-valued answer rather than a
# bool.
DISABLE_ENV = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
_ENV_OFF = ("1", "true", "yes", "on")
_ENV_ON = ("0", "false", "no", "off")

# The variables that decide the DIRECTORY above every settings scope, in the
# order the harness consults them. Read from the 2.1.258 code, and NOT
# resolved here: each needs a resolver of its own (a cowork path, a remote
# projects root, a literal project key), and re-deriving three more surfaces
# is how this module comes to name a directory the harness does not use. What
# a report can do with them honestly is say one is in effect.
OVERRIDE_ENV = (
    "CLAUDE_COWORK_MEMORY_PATH_OVERRIDE",
    "CLAUDE_CODE_REMOTE_MEMORY_DIR",
    "CLAUDE_CODE_PROJECT_DIR_NAME",
)


class ProjectMemory:
    """One project directory the harness has written memories into.

    `files` carries `MEMORY.md` when it is there and `memories` never counts
    it: the two numbers answer different questions — how many files a move has
    to carry, and how many memories an adopter actually has.

    THREE LINK FLAGS rather than one, because whatever copies this directory
    owes each of them a different answer: a linked memory directory is already
    wired somewhere else and must not be moved at all, a linked project
    directory is reached through a link somebody else made, and a linked file
    inside is a memory whose bytes live outside the directory being copied.
    """

    __slots__ = ("key", "path", "files", "is_symlink", "linked_project", "linked_files")

    def __init__(
        self,
        key: str,
        path: str,
        files: list,
        is_symlink: bool,
        linked_project: bool = False,
        linked_files: tuple = (),
    ) -> None:
        self.key = key
        self.path = path
        self.files = files
        self.is_symlink = is_symlink
        self.linked_project = linked_project
        self.linked_files = linked_files

    @property
    def memories(self) -> int:
        return sum(1 for name in self.files if name != INDEX_NAME)

    @property
    def linked(self) -> bool:
        """Whether anything about this directory is reached through a link."""
        return bool(self.is_symlink or self.linked_project or self.linked_files)


def project_key(cwd: str) -> str:
    """The harness's own directory name for the project `cwd` is in.

    FROM THE REPOSITORY, not from the directory: the key is derived from the
    git root, so a linked worktree and a subdirectory of the main checkout key
    to the same place. That is what makes "the memories for this project" one
    answer rather than one per tree, and it is why the walk goes through the
    git COMMON dir — a linked worktree's own git dir is under the main
    checkout's, and its `commondir` file is the only thing that says so.
    Measured on 2.1.258: a linked worktree wrote to its main checkout's key.

    PHYSICAL, not the spelling the caller used. The harness keys on the
    process's own `cwd`, which the kernel has already resolved, so a directory
    reached through a symlink keys to what the link points at. Measured on
    2.1.258: a session in a symlinked directory outside any repository wrote to
    the target's key, not the link's.

    `cwd` is the caller's to pass and has to be ABSOLUTE. Doctor has already
    resolved where it stands for the settings scopes, and a second walk here
    could answer differently from the first if the directory moved between
    them. An empty or relative one is REFUSED with `ValueError` rather than
    answered: `""` is exactly what a caller holds once `os.getcwd()` has
    failed, and keyed it collapses to `<config dir>/projects/memory`, a path
    that reads as derived and is nowhere.

    A key over `KEY_MAX` characters is refused for the same reason. The harness
    truncates and appends a hash of the untruncated path, and that hash was not
    measured — so the only names this could return are one the harness does not
    use and a prefix that is no directory at all.

    Never raises for anything the FILESYSTEM does. Every way the walk can fail
    — no repository above `cwd`, a session directory that was removed
    underneath the process, a `.git` file nothing can read, a path the OS
    refuses to resolve at all — falls back to the cwd itself, which is what the
    harness keys on outside a repository anyway.
    """
    if not os.path.isabs(cwd):
        raise ValueError(f"{cwd!r} is not an absolute directory")
    key = _SANITIZE.sub("-", _project_path(cwd))
    if len(key) > KEY_MAX:
        # THE EXPLANATION BEFORE THE PATH, because doctor interpolates this
        # message into a detail that is bounded from the end, and the path here
        # is the session's own cwd — the one part of the sentence whose length
        # the adopter's machine decides. Led with the path, a deep enough
        # directory cut the reason it was refused off the end of the row.
        raise ValueError(
            f"the project key is {len(key)} characters; over {KEY_MAX} the "
            "harness appends a hash suffix memkit has not measured, so the "
            f"directory it uses cannot be named: {cwd!r}"
        )
    return key


def _project_path(cwd: str) -> str:
    """The path the harness keys on: the repository above `cwd`, or `cwd`.

    RESOLVED on every path out of this function, including the ones that give
    up: `_repo_root` resolves before it walks, so only the fallbacks could
    return the spelling the caller happened to use.

    A SUBMODULE KEYS ON ITSELF, and that is the one branch here that is a
    reading rather than a measurement. Measured on 2.1.258: the harness
    rewrites a git dir to another root only when it finds a `commondir` beside
    it, which a linked worktree has and a submodule does not. What was not
    measured is a live submodule session writing a memory — so if the harness
    ever keys one on `<super>/.git/modules`, every submodule of one
    superproject shares a directory and this names the wrong one. The test
    covering it asserts the two submodules differ, which is the shape of the
    claim rather than the harness's own answer.
    """
    try:
        resolved = os.path.realpath(cwd)
    except (OSError, ValueError):
        return cwd
    try:
        root = _repo_root(resolved)
        if root is None:
            return resolved
        common = _repo_common_dir(root)
        if common is None:
            return resolved
        if _SUBMODULE_GIT_DIR in common:
            # The submodule's own worktree root — see the docstring for what
            # was measured and what was read.
            return root
        return os.path.dirname(common)
    except (_RootUnknown, OSError, ValueError):
        return resolved


def default_dir(config_dir: str, cwd: str) -> str:
    """Where the harness writes this project's memories with no setting.

    NO trailing separator, though a directory that reads as a file is one an
    adopter tries to open. Every surface prints this path through
    `_display_path`, which re-spells anything under `$HOME` relative to it and
    normalises the separator away in doing so — so the separator survived only
    for a config directory outside HOME, which is no real install, and a
    promise kept on the fixtures alone is worse than none.

    Refuses a cwd that is not absolute, through `project_key`.
    """
    return os.path.join(config_dir, "projects", project_key(cwd), "memory")


def inventory(config_dir: str) -> list:
    """Every project directory under `config_dir` that holds memories.

    A directory qualifies on holding at least one `*.md` that is not the index.
    Direct children only, because the harness writes flat — a `search/` below
    one of these is somebody else's tree, and recursing into it would count
    files no adoption should touch.

    `config_dir` is the harness's own configuration directory, which is the
    ADOPTER's to place: this enumerates names in it and opens nothing, so a
    `$CLAUDE_CONFIG_DIR` pointed somewhere unexpected costs a report about the
    wrong directory and nothing more.

    COST: one `os.scandir` of `projects/` and one per project directory, and
    nothing below that. There is no time budget and no cap on the result, and
    that is a decision this shape earns rather than a gap: a real config
    directory here holds 3918 project entries of which 19 hold memories, so the
    walk is two levels of `scandir` over names, with no read of a file. The
    link flags for the project directory and for each listed file come off the
    `scandir` entries the walk already holds; only the memory directory's own
    flag costs a `stat`, once per qualifying project. A cap would make the
    count this reports a number an adopter cannot reconcile with their own
    `ls`.

    Unreadable entries are skipped rather than raising: a diagnostic that dies
    on one unreadable directory reports nothing about the other 3917.
    `ValueError` beside `OSError` for the same reason `_project_path` catches
    both — `scandir` raises that, not `OSError`, on an embedded NUL, and this
    path is built from an environment variable.

    Sorted by memory count descending, then by key — the order a report that
    can show only a few of them wants, and stable for two directories holding
    the same number.
    """
    try:
        with os.scandir(os.path.join(config_dir, "projects")) as entries:
            projects = [
                (entry.name, entry.path, entry.is_symlink()) for entry in entries
            ]
    except (OSError, ValueError):
        return []
    found = []
    for key, path, linked_project in projects:
        memory = os.path.join(path, "memory")
        try:
            with os.scandir(memory) as entries:
                listed = sorted(
                    (entry.name, entry.is_symlink())
                    for entry in entries
                    if entry.name.endswith(".md") and entry.is_file()
                )
        except (OSError, ValueError):
            continue
        files = [name for name, _ in listed]
        if not any(name != INDEX_NAME for name in files):
            continue
        found.append(
            ProjectMemory(
                key,
                memory,
                files,
                os.path.islink(memory),
                linked_project,
                tuple(name for name, linked in listed if linked),
            )
        )
    found.sort(key=lambda project: (-project.memories, project.key))
    return found


def switch(scopes, key: str) -> tuple:
    """`(value, scope name)` for one settings key, or `(None, None)`.

    `SCOPE_ORDER`, not the order the scopes arrive in: these keys are read in
    the harness's own precedence because the answer is a switch and a directory
    the report has to NAME, and a report that named every value it found would
    be answering a different question. An explicit `null` is absence, which is
    what the harness's own `!= null` test makes it.

    THAT PRECEDENCE IS AN INFERENCE FOR THE TWO BOOLEANS — see `SCOPE_ORDER`.
    The scope name comes back beside the value so a caller can say which file
    decided, and so a caller that cares can ask whether a lower one disagrees:
    this returns the first answer, never a claim that it is the only one.
    """
    by_name = {scope.scope: scope for scope in scopes}
    for name in SCOPE_ORDER:
        scope = by_name.get(name)
        if scope is not None and scope.data.get(key) is not None:
            return scope.data[key], name
    return None, None


def harness_dir(value):
    """The directory the harness would USE for this value, or None.

    THE VALIDATOR, not a test of the string a caller passed in. Read from the
    2.1.258 code, in this order:

    - a falsy value is refused before anything else looks at it;
    - only a LEADING `~/` expands, and only against the home directory — and
      the remainder is refused outright when it normalises to `.`, `..` or
      anything below `..`, so `~/..` is not the parent of home, it is nothing;
    - the value is then NORMALISED and stripped of trailing separators, and
      only then tested for being absolute, at least three characters, and free
      of a NUL. (Its refusal of a bare `C:` drive letter is not mirrored: on
      the platforms memkit runs on that string is not absolute, so the branch
      could never be reached or tested.)

    The order is the whole of it. Testing the expanded string first accepted
    `~`, `~/`, `~/..` and `/a/` — four values the harness replaces with the
    default directory — and a report that names a directory the harness does
    not write to is the defect this module exists to close.

    NFC, because the harness normalises the string it returns and a Linux
    filesystem holds the two spellings of a composed character as two
    different directories.

    ONE REFUSAL IS NOT MIRRORED: the harness's validator ends with a predicate
    this reading could not resolve out of the minified bundle, so a value
    memkit calls usable may still be one the harness declines. That direction
    is disclosed rather than guessed at — the alternative is refusing values
    the harness accepts, which is the same wrong answer pointed the other way.
    """
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("~" + os.sep):
        remainder = value[2:]
        above = os.path.normpath(remainder or os.curdir)
        if above in (os.curdir, os.pardir) or above.startswith(os.pardir + os.sep):
            return None
        value = os.path.join(os.path.expanduser("~"), remainder)
    path = os.path.normpath(value).rstrip(os.sep) or os.sep
    if not os.path.isabs(path) or len(path) < 3 or "\x00" in path:
        return None
    return unicodedata.normalize("NFC", path)


def usable_dir(value) -> bool:
    """Whether the harness would use this `autoMemoryDirectory` value."""
    return harness_dir(value) is not None


def env_switch() -> tuple:
    """`(whether the feature runs, the value that said so)`, or `(None, "")`.

    ABOVE EVERY SETTINGS SCOPE and answered before them — see `DISABLE_ENV`.
    The value is returned as it was spelled rather than as it was read, because
    what an adopter has to go and change is the spelling.
    """
    raw = os.environ.get(DISABLE_ENV)
    if not raw:
        return None, ""
    word = raw.strip().lower()
    if word in _ENV_OFF:
        return False, raw
    if word in _ENV_ON:
        return True, raw
    return None, ""


def overrides() -> tuple:
    """Every `OVERRIDE_ENV` name set to a non-empty value, in the harness's
    own order. Named, never resolved."""
    return tuple(name for name in OVERRIDE_ENV if os.environ.get(name))


def configured_dir(scopes) -> tuple:
    """`(directory, scope name)` the harness would use, or `(None, None)`.

    DECIDED BY THE FIRST SCOPE THAT DECLARES THE KEY, whatever it declares.
    Measured on 2.1.258: the resolver takes the first non-null value it finds
    and validates afterwards, so a value it rejects falls through to the
    default directory and NOT to the next scope. Read the other way — skipping
    a bad value and reporting a lower scope's good one — this names a directory
    the harness does not write to, which is the whole defect this module
    exists to close.

    EXPANDED, AND NORMALISED THE WAY THE HARNESS NORMALISES IT. `~/notes/search`
    and the path it expands to are the same directory, and a comparison against
    a store's corpus root that used the unexpanded spelling would report
    "outside every store" for a setting pointing straight at one. `harness_dir`
    answers with the string the harness resolves to, so what is compared and
    what is printed are both the directory being written to. Display puts the
    `~` back.
    """
    value, name = switch(scopes, DIRECTORY_KEY)
    if name is None:
        return None, None
    directory = harness_dir(value)
    if directory is None:
        return None, None
    return directory, name
