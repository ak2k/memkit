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


class Untrusted(Exception):
    """A program, a lookup or an environment this package may not act on.

    Raised rather than answered, and that is the module's one structural rule:
    NO function here returns a falsy value to mean "I could not decide."
    `""`, `[]`, `False` and `None` are answers about the world, never about
    the decider's confidence — every defect of this class was one function
    spelling the second like the first and a caller downstream reading it as
    the safe case.
    """


# The old name, for the `except` clauses that spell it. One class, so an
# `except _Untrusted` and an `except Untrusted` cannot come apart.
_Untrusted = Untrusted


def require_executable(path: str) -> None:
    """THE ONE RULE for every program any module in this package starts.

    Returns nothing and raises `Untrusted` naming the reason, because the
    predicate it replaced answered "no" and "I have no idea" with the same
    `False` — and a caller that gets a bool has nowhere to put the reason
    even when there was one.

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
    if not path:
        raise Untrusted("no program was named")
    if not os.path.isabs(path):
        raise Untrusted(f"{path!r} is not an absolute path")
    if not os.path.isfile(path):
        raise Untrusted(f"{path!r} is not a file")
    if not os.access(path, os.X_OK):
        raise Untrusted(f"{path!r} is not executable")
    if path == sys.executable:
        # The interpreter ALREADY RUNNING is not a program anything chose
        # here — this code is what it is executing. The session-directory rule
        # below would refuse it for an adopter whose venv sits inside the
        # project they are standing in, which is where a python project puts
        # one by default, and would stop nothing: the process is already
        # theirs.
        return
    if _under_cwd(path):
        raise Untrusted(
            f"{path!r} resolves inside the directory this session stands in"
        )


def trusted_path() -> list:
    """The PATH entries a repository cannot steer, or `Untrusted`.

    NEVER an empty list, and the reason is measured rather than argued: an
    empty PATH is not "search nothing" to POSIX, it is the CURRENT DIRECTORY.
    A child handed `PATH=""` from a directory holding `memkit-probe-target`
    ran it; the same child with a PATH naming a directory that holds nothing
    refused. So the filter's SUCCESS path — every entry rejected — was the one
    that handed the lookup to the checkout, and the fix is not a better empty
    value but a refusal.

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
    except OSError as exc:
        # The session directory can be removed under this process. Nothing can
        # then be said about which entries are inside it, and a rule about what
        # may be EXECUTED that cannot decide has to say so.
        raise Untrusted(f"the session directory is unreadable: {exc}") from exc
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
    if not entries:
        raise Untrusted(
            "no PATH entry survives the rule: every one is empty, relative, "
            "inside this session's own directory, or inside the plugin payload"
        )
    return entries


def resolve(name: str) -> str:
    """`name` resolved against `trusted_path`, or `Untrusted`.

    Never `shutil.which`: that reads the session's PATH, which is an input a
    repository steers through direnv, a checked-in venv or a
    `node_modules/.bin` — and the result is a program this package then runs.
    `shutil.which` is banned outright package-wide rather than wrapped, so
    there is no spelling of the untrusted lookup left to reach for.

    A NAME, never a path: a word carrying a separator is not a question a PATH
    lookup answers, and `shutil.which` would have returned such a word
    unexamined. It is refused here and goes to `require_executable` instead,
    which is the rule that can judge a path.
    """
    if not name:
        raise Untrusted("no program name was given")
    if os.sep in name or (os.altsep and os.altsep in name):
        raise Untrusted(f"{name!r} carries a path separator, so it is not a name")
    for entry in trusted_path():
        candidate = os.path.join(entry, name)
        try:
            require_executable(candidate)
        except Untrusted:
            continue
        return candidate
    raise Untrusted(f"{name!r} is on no PATH entry this package may search")


# What a child of this package is GIVEN. An allow-list, and the polarity is
# the whole of the decision.
#
# The list this replaces enumerated 41 dangerous names and 3 prefixes, which
# is a bet that nobody will invent a 42nd. Measured on one developer machine,
# 78 names survived it — `GIT_CONFIG_PARAMETERS`, which reopens git's entire
# configuration discovery on its own, among them. Inverted, the question stops
# being "what could a variable do to a child" and becomes "what does a child
# need", which is a short and finite list.
#
# An incomplete deny-list ADMITS the thing nobody thought of. An incomplete
# allow-list REFUSES it, visibly, and the adopter is told which name to add:
# incompleteness stops being a vulnerability and becomes a support ticket.
# `DYLD_*`, `LD_*`, `BASH_FUNC_x%%`, `PYTHON*` and every future member of those
# families are excluded because they were never added.
#
# PATH is not here because it is not FORWARDED — it is rebuilt from the
# entries `trusted_path` admits, so what the child resolves next is governed
# by the same rule as what this process resolved.
CHILD_ENV_KEEP = ("HOME", "LANG", "LC_ALL", "TMPDIR", "TZ")


def child_env(extra: dict = None, forward: tuple = ()) -> dict:  # noqa: RUF013
    """The environment a child of this package is handed.

    `extra` is a route's own additions, values it computed. `forward` NAMES
    session variables to carry over — a declared, printable tuple rather than
    a call site reaching into `os.environ` inline, so what a route inherits is
    something another part of the program can render for a reader.

    A forwarded name that is not set is ABSENT from the result, not empty: a
    child reads `""` as a value and acts on it, and absence is the only
    spelling of "this was not set".
    """
    env = {}
    for name in CHILD_ENV_KEEP + tuple(forward):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env["PATH"] = os.pathsep.join(trusted_path())
    if extra:
        env.update(extra)
    return env


def _execute(
    argv: list,
    *,
    env_extra: dict = None,  # noqa: RUF013
    env_forward: tuple = (),
    **kw,
) -> subprocess.CompletedProcess:
    """The one place any module in this package starts a process.

    Raises `Untrusted` rather than running anything `require_executable`
    refuses, so a call site that forgot the rule cannot silently execute — the
    check that called it reports UNKNOWN with the reason instead, which is the
    answer a diagnostic owes when honouring the rule costs it its signal.

    The environment is BUILT, never derived from the caller's: there is no
    `env=` to pass, because a call site that assembles `dict(os.environ, …)`
    is the whole class of defect this exists to remove.
    """
    if not argv:
        raise Untrusted("no program was named")
    require_executable(argv[0])
    if "env" in kw:
        raise Untrusted(
            "a caller may not supply a child environment; name what the route "
            "adds in env_extra and what it inherits in env_forward"
        )
    env = child_env(env_extra, env_forward)
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


def _trusted_git(args: list, **kw) -> subprocess.CompletedProcess:
    """One `git` run: a trusted binary, told to read no configuration that
    names a program.

    Both halves are the rule. `resolve` settles WHICH git runs; `_git_argv`
    and `_GIT_NEUTRAL_ENV` settle what it reads once it is running — and the
    second half is not optional, because git standing in a directory somebody
    else wrote is git being handed a program to run.

    Raises `Untrusted` when no trusted git resolves, so a caller cannot get a
    silent "not a repository" answer out of a refusal.
    """
    git = resolve("git")
    extra = dict(_GIT_NEUTRAL_ENV)
    extra.update(kw.pop("env_extra", None) or {})
    return _execute(_git_argv(git, list(args)), env_extra=extra, **kw)
