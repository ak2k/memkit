"""Where the harness's own memory feature writes, and what it has written.

Claude Code writes auto-memory itself, flat, one directory per project, under
its config directory — so an adopter arriving at memkit already has a corpus,
and the question every surface here answers is where that corpus IS. Doctor
names it; a later command offers to move it. Both need the same three answers,
and deriving them twice is how two commands come to disagree about which
directory holds an adopter's memories.

MEASURED, on the 2.1.232, 2.1.240 and 2.1.258 binaries and against
code.claude.com/docs/en/memory:

- `autoMemoryEnabled` decides whether the feature runs at all. False and the
  harness neither reads nor writes auto-memory.
- `autoMemoryDirectory` is a path, `~/` expanded, used AS-IS for every project:
  one flat directory, no per-project subdirectory under it.
- `autoDreamEnabled` false stops background consolidation ONLY. Memories are
  still written, which is why it is not the switch that turns the feature off.
- Unset, the directory is `<config dir>/projects/<project key>/memory/`, and
  the key comes from the GIT REPOSITORY ROOT rather than from the cwd: every
  worktree and every subdirectory of one repository share one directory.

`memoryDir` is not a key the harness reads. It was memkit's own earlier reading
of this feature, and a remedy naming it changed nothing on the adopter's
machine.
"""

from __future__ import annotations

import os
import re

from memkit.memory_prompt_recall import (
    _repo_common_dir,
    _repo_root,
    _RootUnknown,
    expand_home,
)

# The harness's own sanitiser for a project key: `/` and `.` both become `-`,
# so `/Users/x/.config/nix` keys as `-Users-x--config-nix`.
_SANITIZE = re.compile(r"[/.]")

# The one file in an auto-memory directory that is an INDEX rather than a
# memory. It is carried in `files` because anything moving the directory has to
# move it too, and it is never counted, because a directory holding nothing but
# an index holds no memories.
INDEX_NAME = "MEMORY.md"

# The key whose value is a directory this package must name, and the scopes the
# harness honours it in — most authoritative first. `project` is absent
# deliberately: the harness IGNORES `autoMemoryDirectory` in a checked-in
# `.claude/settings.json`, so that cloning a repository cannot redirect where
# an agent writes. Reading it there would make this report name a directory
# nothing writes to.
DIRECTORY_KEY = "autoMemoryDirectory"
DIRECTORY_SCOPES = ("managed", "local", "user")

ENABLED_KEY = "autoMemoryEnabled"
DREAM_KEY = "autoDreamEnabled"


class ProjectMemory:
    """One project directory the harness has written memories into.

    `files` carries `MEMORY.md` when it is there and `memories` never counts
    it: the two numbers answer different questions — how many files a move has
    to carry, and how many memories an adopter actually has.
    """

    __slots__ = ("key", "path", "files", "is_symlink")

    def __init__(
        self, key: str, path: str, files: list, is_symlink: bool
    ) -> None:
        self.key = key
        self.path = path
        self.files = files
        self.is_symlink = is_symlink

    @property
    def memories(self) -> int:
        return sum(1 for name in self.files if name != INDEX_NAME)


def project_key(cwd: str) -> str:
    """The harness's own directory name for the project `cwd` is in.

    FROM THE REPOSITORY, not from the directory: the key is derived from the
    git root, so a linked worktree and a subdirectory of the main checkout key
    to the same place. That is what makes "the memories for this project" one
    answer rather than one per tree, and it is why the walk goes through the
    git COMMON dir — a linked worktree's own git dir is under the main
    checkout's, and its `commondir` file is the only thing that says so.

    `cwd` is the caller's to pass. Doctor has already resolved where it stands
    for the settings scopes, and a second walk here could answer differently
    from the first if the directory moved between them.

    Never raises. Every way the walk can fail — no repository above `cwd`, a
    session directory that was removed underneath the process, a `.git` file
    nothing can read — falls back to the cwd itself, which is what the harness
    keys on outside a repository anyway.
    """
    return _SANITIZE.sub("-", _project_path(cwd))


def _project_path(cwd: str) -> str:
    try:
        root = _repo_root(cwd)
        if root is None:
            return cwd
        common = _repo_common_dir(root)
        if common is None:
            return cwd
        return os.path.dirname(common)
    except (_RootUnknown, OSError):
        return cwd


def default_dir(config_dir: str, cwd: str) -> str:
    """Where the harness writes this project's memories with no setting.

    Spelled with its trailing separator, because every surface that has this
    string prints it, and a directory that reads as a file is one an adopter
    tries to open.
    """
    return os.path.join(config_dir, "projects", project_key(cwd), "memory", "")


def inventory(config_dir: str) -> list:
    """Every project directory under `config_dir` that holds memories.

    A directory qualifies on holding at least one `*.md` that is not the index.
    Direct children only, because the harness writes flat — a `search/` below
    one of these is somebody else's tree, and recursing into it would count
    files no adoption should touch.

    COST: one `os.scandir` of `projects/` and one per project directory, and
    nothing below that. There is no time budget and no cap on the result, and
    that is a decision this shape earns rather than a gap: a real config
    directory here holds 3918 project entries of which 19 hold memories, so the
    walk is two levels of `scandir` over names, with no `stat` of a file and no
    read of one. A cap would make the count this reports a number an adopter
    cannot reconcile with their own `ls`.

    Unreadable entries are skipped rather than raising: a diagnostic that dies
    on one unreadable directory reports nothing about the other 3917.

    Sorted by memory count descending, then by key — the order a report that
    can show only a few of them wants, and stable for two directories holding
    the same number.
    """
    found = []
    try:
        with os.scandir(os.path.join(config_dir, "projects")) as entries:
            projects = [(entry.name, entry.path) for entry in entries]
    except OSError:
        return found
    for key, path in projects:
        memory = os.path.join(path, "memory")
        try:
            with os.scandir(memory) as entries:
                files = sorted(
                    entry.name
                    for entry in entries
                    if entry.name.endswith(".md") and entry.is_file()
                )
        except OSError:
            continue
        if not any(name != INDEX_NAME for name in files):
            continue
        found.append(
            ProjectMemory(key, memory, files, os.path.islink(memory))
        )
    found.sort(key=lambda project: (-project.memories, project.key))
    return found


def configured_dir(scopes) -> tuple:
    """`(directory, scope name)` the harness would use, or `(None, None)`.

    The scopes are iterated in `DIRECTORY_SCOPES` order rather than in the
    order they arrive, because that order is the harness's answer to which file
    wins and this is the one key doctor resolves rather than merely reports.
    What hangs off it is a directory the report has to NAME.

    EXPANDED. `~/notes/search` and the path it expands to are the same
    directory, and a comparison against a store's corpus root that used the
    unexpanded spelling would report "outside every store" for a setting
    pointing straight at one. Display puts the `~` back.
    """
    by_name = {scope.scope: scope for scope in scopes}
    for name in DIRECTORY_SCOPES:
        scope = by_name.get(name)
        if scope is None:
            continue
        value = scope.data.get(DIRECTORY_KEY)
        if isinstance(value, str) and value:
            return expand_home(value), name
    return None, None
