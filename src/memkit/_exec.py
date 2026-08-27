"""The one place this package starts a process, and the rules that govern it.

A MODULE rather than a section of the hook file, and the boundary is the
point: `memory_prompt_recall` — the file the harness runs on every prompt —
imports nothing from here, which makes "the every-prompt path starts no
process" a fact a reader can check by looking at its imports rather than a
claim to be audited call site by call site.

ONE CHOKEPOINT, package-wide, for the same reason each rule below is one
predicate: a rule held at each call site is a rule the next call site will
not have. Every module in this package starts its processes through
`_execute` and resolves every program NAME through `_trusted_which`, and
`tests/test_doctor.py` walks the AST of every file under `src/memkit` to
assert there is no second way out — a module added later is covered by
discovery rather than by somebody remembering to add it to a list.

Three inputs decide what a program IS, and all three are attacker-writable on
the surfaces this package exposes to a model:

  which file        - a PATH lookup, or a path somebody else spelled
  which environment - an interpreter reads its module path out of it, a
                      loader preloads what it is told to, a shell sources
                      a file one variable names
  which directory   - git takes the name of a program to run from the
                      configuration of whatever repository it is standing in

A path being ABSOLUTE settles none of them. Trust is decided by origin.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _under_cwd(path: str) -> bool:
    """Whether `path` lands inside the directory this session stands in."""
    try:
        cwd = os.path.realpath(os.getcwd())
        target = os.path.realpath(path)
    except OSError:
        return True
    return target == cwd or target.startswith(cwd + os.sep)


class _Untrusted(Exception):
    """A program whose identity this package may not take from where it found
    it."""


def _may_execute(path: str) -> bool:
    """THE ONE RULE for every program any module in this package starts.

    Program identity comes from an adopter-owned settings scope, from the
    plugin's own payload, or from a pinned absolute path — never from
    something a repository can write and never from a PATH lookup. `memkit
    doctor` is model-invocable and its skill pre-approves the exact argv, so
    running it inside somebody else's checkout must not be that checkout
    choosing a program to run as the user, with the session's whole
    environment inherited by the child.

    Kept as one predicate rather than a rule repeated at each call site,
    because the failure mode is a call site that does not apply it — a rule
    held in five places is a rule the sixth will not have.
    """
    if not path or not os.path.isabs(path):
        return False
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    if path == sys.executable:
        # The interpreter ALREADY RUNNING is not a program anything chose
        # here — this code is what it is executing. The session-directory rule
        # below would refuse it for an adopter whose venv sits inside the
        # project they are standing in, which is where a python project puts
        # one by default, and would stop nothing: the process is already
        # theirs.
        return True
    return not _under_cwd(path)


def _trusted_path_entries() -> list:
    """The PATH entries a repository cannot steer.

    An EMPTY entry is the current directory, spelled the way every shell reads
    it, and a relative one is the same thing under another name — so a
    `PATH=:/usr/bin` inherited from the session hands the lookup to whatever
    the checkout ships. Entries under the session directory go for the same
    reason a `node_modules/.bin` or a direnv-exported venv is the checkout's
    choice, and entries inside the payload go because a plugin's own tree is a
    clone of a pinned commit: it may supply memkit's wrappers, not the
    harness binary memkit asks questions of.
    """
    try:
        cwd = os.path.realpath(os.getcwd())
    except OSError:
        # The session directory can be removed under this process. Nothing
        # can then be said about which entries are inside it, and the safe
        # direction for a rule about what may be EXECUTED is to admit
        # nothing rather than to admit everything.
        return []
    payload = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    payload_real = os.path.realpath(payload) if payload else ""
    entries = []
    for entry in (os.environ.get("PATH") or "").split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            continue
        try:
            real = os.path.realpath(entry)
        except OSError:
            continue
        if real == cwd or real.startswith(cwd + os.sep):
            continue
        if payload_real and (
            real == payload_real or real.startswith(payload_real + os.sep)
        ):
            continue
        entries.append(entry)
    return entries


def _trusted_which(name: str) -> str:
    """`name` resolved against `_trusted_path_entries`, or "".

    Never `shutil.which`: that reads the session's PATH, which is an input a
    repository steers through direnv, a checked-in venv or a
    `node_modules/.bin` — and the result is a program this package then runs.
    `shutil.which` is banned outright package-wide rather than wrapped, so
    there is no spelling of the untrusted lookup left to reach for.

    A NAME, never a path: a word carrying a separator is not a question a PATH
    lookup answers, and `shutil.which` would have returned such a word
    unexamined. It is refused here and goes to `_may_execute` instead, which
    is the predicate that can judge a path.
    """
    if not name or os.sep in name or (os.altsep and os.altsep in name):
        return ""
    for entry in _trusted_path_entries():
        candidate = os.path.join(entry, name)
        if _may_execute(candidate):
            return candidate
    return ""


# The variables that make a trusted binary somebody else's program. Each one
# names code the child loads before it reaches its own first instruction: a
# dynamic loader preload, an interpreter's module path or startup file, the
# file a POSIX shell sources for a non-interactive run, the program git runs
# in place of a diff or a credential prompt. All of them arrive from whatever
# launched this process, which on the pre-approved surfaces is a session a
# checkout steers through direnv.
_CHILD_ENV_DROP = (
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_FALLBACK_FRAMEWORK_PATH",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    "PYTHONEXECUTABLE",
    "PYTHONUSERBASE",
    "PYTHONINSPECT",
    "NODE_OPTIONS",
    "NODE_REPL_EXTERNAL_MODULE",
    "PERL5OPT",
    "PERL5LIB",
    "RUBYOPT",
    "RUBYLIB",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_EXTERNAL_DIFF",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_PROXY_COMMAND",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
    "GIT_EDITOR",
    "GIT_SEQUENCE_EDITOR",
    "GIT_PAGER",
    "PAGER",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_NAMESPACE",
)

# Whole families rather than names, because the family is what the variable
# is: `GIT_CONFIG_KEY_7` is as good as `GIT_CONFIG_KEY_0`, and an exported
# shell function is code that runs when the child's shell starts.
_CHILD_ENV_DROP_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_", "BASH_FUNC_")


def _child_env(base: dict | None = None) -> dict:
    """The environment a child of this package may be handed.

    Program identity does not stop at the executable: the child reads its own
    code out of the environment, and PATH decides what IT resolves next. So
    the variables that name code are removed and PATH is replaced by the
    entries `_trusted_path_entries` admits — the same rule applied one process
    down, rather than the rule ending at the boundary this process controls.

    A caller that genuinely needs one of these back sets it explicitly through
    `env_extra`, which makes the exception visible at its own call site
    instead of leaving the whole class admitted for everyone.
    """
    env = dict(os.environ if base is None else base)
    for name in _CHILD_ENV_DROP:
        env.pop(name, None)
    for name in [n for n in env if n.startswith(_CHILD_ENV_DROP_PREFIXES)]:
        env.pop(name, None)
    env["PATH"] = os.pathsep.join(_trusted_path_entries())
    return env


def _execute(argv: list, *, env_extra: dict | None = None, **kw):
    """The one place any module in this package starts a process.

    Raises `_Untrusted` rather than running anything whose program fails
    `_may_execute`, so a call site that forgot the rule cannot silently
    execute — the check that called it reports UNKNOWN with a remedy instead,
    which is the answer a diagnostic owes when honouring the rule costs it its
    signal.

    The environment goes through `_child_env` whether the caller supplied one
    or not: a call site that builds its own `env` from `os.environ` would
    otherwise hand the child every variable this gate exists to strip.
    """
    if not argv or not _may_execute(argv[0]):
        raise _Untrusted(argv[0] if argv else "")
    env = _child_env(kw.pop("env", None))
    if env_extra:
        env.update(env_extra)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(argv, env=env, **kw)  # noqa: S603


# What a git invocation must be told to forget. A repository's own
# `.git/config` is not reachable by any environment variable — there is no
# `GIT_CONFIG_LOCAL` — so each key that names a PROGRAM is overridden on the
# command line instead, where a `-c` beats the file. `core.fsmonitor` is the
# one reproduced end to end: `git ls-files --error-unmatch` runs it, twice,
# from the config of whatever repository the path belongs to.
_GIT_NEUTRAL_CONFIG = (
    "core.fsmonitor=",
    "core.hooksPath=/dev/null",
    "core.pager=cat",
    "core.sshCommand=",
    "core.askpass=",
    "core.alternateRefsCommand=",
    "core.attributesFile=/dev/null",
    "diff.external=",
    "gpg.program=",
    "credential.helper=",
    "uploadpack.packObjectsHook=",
    "protocol.ext.allow=never",
)

# The subcommands that render CONTENT, and so consult `.gitattributes` for a
# textconv or an external diff driver. `--no-textconv` and `--no-ext-diff` are
# only accepted by these, which is why the list exists rather than a blanket
# flag.
_GIT_CONTENT_SUBCOMMANDS = ("diff", "log", "show")


# The git-level options that take a VALUE. Without them the walk below stops
# at `<dir>` in `git -C <dir> log ...` and never reaches the subcommand.
_GIT_VALUE_OPTIONS = ("-C", "-c", "--git-dir", "--work-tree", "--namespace")


def _git_argv(git: str, args: list) -> list:
    """`git` plus `args`, with every config key that names a program silenced."""
    flags = ["--no-optional-locks"]
    for setting in _GIT_NEUTRAL_CONFIG:
        flags += ["-c", setting]
    rest = list(args)
    i = 0
    while i < len(rest):
        word = rest[i]
        if word in _GIT_VALUE_OPTIONS:
            i += 2
            continue
        if word.startswith("-"):
            i += 1
            continue
        if word in _GIT_CONTENT_SUBCOMMANDS:
            rest[i + 1 : i + 1] = ["--no-textconv", "--no-ext-diff"]
        break
    return [git, *flags, *rest]


# Config DISCOVERY, switched off at the two levels an environment variable can
# reach. `/dev/null` rather than an unset variable: unset means "look in the
# usual place", and the usual place is a file whichever `$HOME` the session
# exported points at.
#
# The `-c` overrides above already beat every config file, so this is the
# second line rather than the first: what it buys is a key nobody enumerated.
# What it costs is `core.excludesFile` — `git ls-files --others
# --exclude-standard` in the checker no longer honours an adopter's GLOBAL
# ignore file, so a file they ignore everywhere reads as untracked here. The
# checker's blamed set is `.md` under a memory store, which is not what a
# global ignore file is usually about.
_GIT_NEUTRAL_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_COUNT": "0",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
}


def _trusted_git(args: list, **kw):
    """One `git` run: a trusted binary, told to read no configuration that
    names a program.

    Both halves are the rule. `_trusted_which` settles WHICH git runs;
    `_git_argv` and `_GIT_NEUTRAL_ENV` settle what it reads once it is
    running — and the second half is not optional, because git standing in a
    directory somebody else wrote is git being handed a program to run.

    Raises `_Untrusted` when no trusted git resolves, so a caller cannot get a
    silent "not a repository" answer out of a refusal.
    """
    git = _trusted_which("git")
    if not git:
        raise _Untrusted("git")
    extra = dict(_GIT_NEUTRAL_ENV)
    extra.update(kw.pop("env_extra", None) or {})
    return _execute(_git_argv(git, list(args)), env_extra=extra, **kw)
