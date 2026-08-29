"""The one place this package starts a process, and the rules that govern it.

A MODULE rather than a section of the hook file, and the boundary is the
point: `memory_prompt_recall` — the file the harness runs on every prompt —
imports nothing from here, which makes "the every-prompt path starts no
process" a fact a reader can check by looking at its imports rather than a
claim to be audited call site by call site.

ONE CHOKEPOINT, package-wide, for the same reason each rule below is one
predicate: a rule held at each call site is a rule the next call site will
not have. Every module in this package starts its processes through
`_execute`, which runs nothing `require_executable` has not approved: an
absolute path to a real executable, named by an adopter-owned scope or the
payload, and NEVER resolved by a PATH lookup — the child's own PATH is what
`trusted_path` leaves after dropping every entry a checkout can steer. And
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

import contextlib
import enum
import os
import subprocess
import sys

from memkit.memory_prompt_recall import _is_process_start_event


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


def child_env(extra: dict | None = None, forward: tuple = ()) -> dict:
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
    env_extra: dict | None = None,
    env_forward: tuple = (),
    timeout: int | None = None,
    cwd: str | None = None,
    input: str | None = None,  # noqa: A002 — subprocess.run spells it this way
) -> subprocess.CompletedProcess:
    """The one place any module in this package starts a process.

    Raises `Untrusted` rather than running anything `require_executable`
    refuses, so a call site that forgot the rule cannot silently execute — the
    check that called it reports UNKNOWN with the reason instead, which is the
    answer a diagnostic owes when honouring the rule costs it its signal.

    EVERY parameter is named here, and the absent `**kw` is the point: the
    gate governs the whole invocation rather than argv[0]. Forwarding a
    caller's keywords let `executable=` substitute a different program behind
    an argv the gate had approved, `shell=True` reinterpret it as a command
    line, and `preexec_fn=` run arbitrary code in the child before it. None of
    those has a spelling here — not a filtered one, not a validated one.

    The environment is likewise BUILT rather than derived from the caller's:
    there is no `env=`, because a call site that assembles `dict(os.environ, …)`
    is the whole class of defect this exists to remove.
    """
    if not argv:
        raise Untrusted("no program was named")
    require_executable(argv[0])
    env = child_env(env_extra, env_forward)
    _WINDOW.append(_Window([str(word) for word in argv]))
    try:
        return subprocess.run(  # noqa: S603
            argv,
            env=env,
            timeout=timeout,
            cwd=cwd,
            input=input,
            capture_output=True,
            text=True,
        )
    finally:
        _WINDOW.pop()


# --- git, as a closed table of routes ----------------------------------------
#
# A caller NAMES a route and supplies typed holes; it never supplies argv. The
# set of git subcommands this package can ever run is therefore finite and
# printable, which turns "which configuration keys can this subcommand reach a
# program through" from an open question into one asked ten times.
#
# That is the answer to a class, not to a key. A `-c` list silences the keys
# somebody thought of — three were missing, one of them
# (`filter.<driver>.clean`) not expressible as a `-c` at all because the
# driver's NAME is chosen by the repository — so the invocations also have to
# stop ASKING for the work that reaches the unlisted ones. Signature
# verification is the case in hand: it is asked for by `log` and by nothing
# else here, so `log.showSignature=false` sits in the two `log` templates
# rather than at their call sites, where the eleventh route would not have it.
#
# `--` is in every template that takes a path, so a path that begins with `-`
# is a path; and every revision arrives as a `Rev`, which has no empty and no
# option-shaped value. Neither is a rule a call site can forget.

# What a git invocation must be told to forget. A repository's own
# `.git/config` is not reachable by any environment variable — there is no
# `GIT_CONFIG_LOCAL` — so each key that names a PROGRAM is overridden on the
# command line instead, where a `-c` beats the file. `core.fsmonitor` is the
# one reproduced end to end: `git ls-files --error-unmatch` runs it, twice,
# from the config of whatever repository the path belongs to.
#
# This list is a SECOND line and is known to be incomplete — see the route
# table's own comment. What bounds the exposure is that the every-prompt path
# runs no git at all, and that the repositories reached from here are ones the
# adopter declared.
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

# Config DISCOVERY, switched off at the levels an environment variable can
# reach. `/dev/null` rather than an unset variable: unset means "look in the
# usual place", and the usual place is a file whichever `$HOME` the session
# exported points at.
#
# `GIT_CONFIG_PARAMETERS` is git's own serialisation of `-c` options and is
# honoured despite every other name here. It cannot arrive — the child's
# environment is built from an allow-list and this is not in it — and it is
# set EMPTY anyway, because a route that one day declares a forward should not
# be able to reopen config discovery by accident.
#
# What this costs is `core.excludesFile` — `ls-files --others
# --exclude-standard` no longer honours an adopter's GLOBAL ignore file, so a
# file they ignore everywhere reads as untracked here. The blamed set is `.md`
# under a memory store, which is not what a global ignore file is usually
# about.
_GIT_NEUTRAL_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_PARAMETERS": "",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
}

# The subcommands that render CONTENT, and so consult `.gitattributes` for a
# textconv or an external diff driver. Only these accept the two flags, which
# is why they are in three templates rather than in the shared prefix.
_NO_CONTENT_DRIVERS = ("--no-textconv", "--no-ext-diff")

# Signature verification names a program through `gpg.<format>.program`, whose
# FORMAT half a repository chooses, so no fixed `-c` covers the family. `log`
# is the only route that can be asked for it, and it is asked here not to be.
_NO_SIGNATURES = ("-c", "log.showSignature=false")


class GitRoute(enum.Enum):
    """Every git invocation this package can make. A closed set."""

    TOPLEVEL = "toplevel"
    HEAD_SHA = "head-sha"
    CHANGED_VS_HEAD = "changed-vs-head"
    CHANGED_SINCE = "changed-since"
    UNTRACKED = "untracked"
    MERGE_BASE = "merge-base"
    BLOB_AT = "blob-at"
    LAST_TOUCH = "last-touch"
    ROW_TOUCH = "row-touch"
    TRACKED = "tracked"


class Rev:
    """A git revision, validated where it is CONSTRUCTED rather than where it
    is used.

    Git reads a leading `-` as an option wherever a revision is expected, and
    `git show` accepts `--output=<file>` — so a config-supplied value starting
    with one turns a read into a write git performs on the config's behalf. No
    placement of `--` fixes a `rev:path` argument, so the shape is refused.

    An empty revision is refused for the reason the guard this replaces got
    wrong: it answered `""`, one of its two call sites checked for that and
    the other did not, and git was handed an empty argument and failed for an
    incidental reason instead of declining to run. A type cannot be half
    checked.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise Untrusted("a revision may not be empty")
        if value.startswith("-"):
            raise Untrusted(f"{value!r} is option-shaped, so it is not a revision")
        if value != value.strip() or "\0" in value:
            raise Untrusted(f"{value!r} is not a well-formed revision")
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Rev({self.value!r})"


def _require_path(value: str, where: str) -> str:
    """A path hole that no `--` protects — `show <rev>:<path>` is one word."""
    if not isinstance(value, str) or not value:
        raise Untrusted(f"{where} may not be empty")
    if value.startswith("-"):
        raise Untrusted(f"{where} {value!r} is option-shaped")
    return value


def _git_template(route: GitRoute, holes: dict) -> list:
    """`route`'s fixed template with its holes filled, or `Untrusted`.

    One branch per member and no default: an unrecognised route is not a route,
    and falls out as a refusal rather than as an argv nobody wrote.
    """
    if not isinstance(route, GitRoute):
        raise Untrusted(f"{route!r} is not a git route")
    if route is GitRoute.TOPLEVEL:
        return ["rev-parse", "--show-toplevel"]
    if route is GitRoute.HEAD_SHA:
        return ["rev-parse", "HEAD"]
    if route is GitRoute.CHANGED_VS_HEAD:
        return ["diff", *_NO_CONTENT_DRIVERS, "-z", "--name-only", "HEAD"]
    if route is GitRoute.CHANGED_SINCE:
        rev = _rev(holes)
        return [
            "diff", *_NO_CONTENT_DRIVERS, "-z", "--name-only", str(rev), "HEAD"
        ]
    if route is GitRoute.UNTRACKED:
        return ["ls-files", "-z", "--others", "--exclude-standard"]
    if route is GitRoute.MERGE_BASE:
        return ["merge-base", str(_rev(holes)), "HEAD"]
    if route is GitRoute.BLOB_AT:
        rev = _rev(holes)
        path = _require_path(_str(holes, "path"), "a blob path")
        return ["show", *_NO_CONTENT_DRIVERS, f"{rev}:{path}"]
    if route is GitRoute.LAST_TOUCH:
        return [
            *_NO_SIGNATURES, "log", *_NO_CONTENT_DRIVERS, "--format=%ct",
            "--name-only", "--no-renames", "--", *_paths(holes),
        ]
    if route is GitRoute.ROW_TOUCH:
        limit = _int(holes, "limit")
        path = _str(holes, "path")
        return [
            *_NO_SIGNATURES, "log", *_NO_CONTENT_DRIVERS, f"-{limit}",
            "--format=%ct", "-p", "--no-color", "--", path,
        ]
    # TRACKED, and there is no `else`: a member added without a template
    # reaches the refusal below rather than an argv nobody wrote.
    if route is GitRoute.TRACKED:
        return ["ls-files", "--error-unmatch", "--", _str(holes, "path")]
    raise Untrusted(f"{route!r} has no template")


def _rev(holes: dict) -> Rev:
    value = holes.get("rev")
    if not isinstance(value, Rev):
        raise Untrusted("this route needs a Rev it did not get")
    return value


def _str(holes: dict, name: str) -> str:
    value = holes.get(name)
    if not isinstance(value, str) or not value:
        raise Untrusted(f"this route needs a non-empty {name} it did not get")
    return value


def _int(holes: dict, name: str) -> int:
    value = holes.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Untrusted(f"this route needs a positive {name} it did not get")
    return value


def _paths(holes: dict) -> list:
    value = holes.get("paths")
    if not isinstance(value, (list, tuple)) or not value:
        raise Untrusted("this route needs at least one path it did not get")
    return [_str({"p": p}, "p") for p in value]


# The adopter's `safe.directory`, read once per process.
#
# `GIT_CONFIG_GLOBAL=/dev/null` closes a config-discovery level and takes this
# key with it, and without it git refuses a repository it considers to have
# dubious ownership — a shared machine, a container bind-mount, a store owned
# by another uid. Measured: two values in `~/.gitconfig` come back from `git
# config --get-all safe.directory` and rc=1 under the neutralisation.
#
# So it is read BEFORE the global config is switched off and re-supplied as
# `-c` entries. Reading configuration starts no program, and every key that
# names one is overridden on this invocation too — the read is the same
# hardened git as every other route, minus the one variable whose absence is
# the thing being repaired.
#
# THE RESIDUAL, stated where the code is: `$HOME` decides which file this
# reads, and `$HOME` is forwarded from the session. What a session that
# controls it can add is permission for git to operate on a repository owned
# by another user — never a program, because no `-c` here survives the
# overrides. That is narrower than what controlling `$HOME` already buys
# elsewhere in this package, and it is not closed by anything here.
_SAFE_DIRECTORIES: list = []


def _safe_directory_flags() -> list:
    if _SAFE_DIRECTORIES:
        return _SAFE_DIRECTORIES[0]
    flags: list = []
    with contextlib.suppress(OSError, subprocess.SubprocessError, Untrusted):
        out = _execute(
            [
                resolve("git"),
                "--no-optional-locks",
                *[word for setting in _GIT_NEUTRAL_CONFIG for word in ("-c", setting)],
                "config",
                "--get-all",
                "safe.directory",
            ],
            timeout=15,
            # Every neutral variable EXCEPT the one that hides this key.
            env_extra={
                name: value
                for name, value in _GIT_NEUTRAL_ENV.items()
                if name != "GIT_CONFIG_GLOBAL"
            },
        )
        if out.returncode == 0:
            for value in out.stdout.splitlines():
                value = value.strip()
                # No leading `-`, or the re-supplied `-c` would be an option.
                if value and not value.startswith("-"):
                    flags += ["-c", f"safe.directory={value}"]
    _SAFE_DIRECTORIES.append(flags)
    return flags


def run_git(
    route: GitRoute, *, repo: str, timeout: int = 30, **holes
) -> subprocess.CompletedProcess:
    """One `git` run: a trusted binary, a template from the closed table above,
    and configuration it is told to forget.

    `repo` is where git STANDS, and it is a directory a caller declared rather
    than a `-C` word inside an argv somebody assembled — which is what makes
    "which repository answered" a thing the signature says.

    Raises `Untrusted` when no trusted git resolves, so a caller cannot get a
    silent "not a repository" answer out of a refusal.
    """
    argv = _git_template(route, holes)
    flags = ["--no-optional-locks"]
    for setting in _GIT_NEUTRAL_CONFIG:
        flags += ["-c", setting]
    flags += _safe_directory_flags()
    return _execute(
        [resolve("git"), *flags, *argv],
        cwd=repo,
        timeout=timeout,
        env_extra=dict(_GIT_NEUTRAL_ENV),
    )


# --- the checker's invocation, reconstructed ---------------------------------
#
# An argv is BUILT from a closed route and one hole. It is never parsed, never
# read out of an environment variable and never passed through from a caller,
# because a command assembled out of an input is an input choosing the code
# that runs — and no amount of validating that input turns it back into a
# command this package chose.
#
# The tail is a CONSTANT with no hole in it, identical on every live route —
# only the interpreter varies, and that one hole goes back through
# `require_executable` immediately before the call. So the set of commands
# this package can ever spell for the checker is three, and they differ in one
# absolute path.

CHECKER_TAIL = ("-m", "memkit.memory_integrity")


class CheckerRoute(enum.Enum):
    """Which interpreter runs the checker. A closed set, not a string.

    A route has no spelling that is not one of these four, and an unrecognised
    value fails at the boundary rather than falling through to a bare-name
    lookup — a parse failure is an exception, never a default. Because the
    argv is derived from the route, "there is no route" also has exactly one
    condition rather than a route name and an empty command list to be held in
    agreement.
    """

    SELF = "self"
    LOCAL = "local"
    UV_MANAGED = "uv-managed"
    NONE = "none"


def checker_argv(route: CheckerRoute, interpreter: str) -> list:
    """The command for `route`, or `Untrusted`.

    NONE is a state callers must handle, and it is spelled as a refusal rather
    than as an empty list, so "this machine has no checker" and "somebody
    forgot to build the command" cannot arrive at a call site looking the
    same.
    """
    if not isinstance(route, CheckerRoute):
        raise Untrusted(f"{route!r} is not a checker route")
    if route is CheckerRoute.NONE:
        raise Untrusted("this machine has no interpreter that can run the checker")
    require_executable(interpreter)
    return [interpreter, *CHECKER_TAIL]


# --- the chokepoint, enforced by the interpreter -----------------------------
#
# The gate above governs argv[0] at the moment a call site asks. This governs
# the invocation the INTERPRETER is about to make, which is one step further
# down and is where the difference between "the argv a gate approved" and "the
# argv that ran" lives. `sys.addaudithook` fires on the runtime event rather
# than on the syntax, so it sees a call reached through a partial, a dict, a
# rebound name or a computed attribute — the shapes a static walk cannot
# resolve and, when it cannot, waves through.
#
# Two states and no third: a window is open for exactly one argv, or nothing
# may start. There is no value here that can be spelled to mean "I could not
# tell".
#
# The window is ONE CALL. A call site that opens it and then runs something
# else is aborted, which is the case a gate on argv[0] cannot see at all.

_WINDOW: list = []


class _Window:
    """One argv, permitted once."""

    __slots__ = ("argv", "spent")

    def __init__(self, argv: list) -> None:
        self.argv = list(argv)
        self.spent = False


def _audit(event: str, args: tuple) -> None:
    if not _is_process_start_event(event):
        return
    if not _WINDOW:
        raise Untrusted(
            f"this package starts no program outside its own gate ({event})"
        )
    window = _WINDOW[-1]
    if event != "subprocess.Popen":
        # The low-level events CPython may raise from inside `subprocess`'s own
        # machinery on some platforms. Permitted only INSIDE a window, and the
        # argv comparison stays on `subprocess.Popen`, which fires first —
        # measured on Darwin/3.9.6 and 3.12.12, where it is the only one.
        return
    argv = list(args[1]) if isinstance(args[1], (list, tuple)) else [args[1]]
    if window.spent:
        raise Untrusted(
            f"the gate was opened for one program and {argv[:1]} is a second"
        )
    if [str(word) for word in argv] != window.argv:
        raise Untrusted(
            f"{argv[:1]} is not the program the gate approved "
            f"({window.argv[:1]})"
        )
    window.spent = True


_INSTALLED: list = []


def enforce_execution_boundary() -> None:
    """Install the runtime chokepoint for this process.

    Called from the entry points rather than at import, for the reason an
    audit hook is a guarantee at all: it cannot be removed. Installing one when
    this module is merely imported would put it in every process that imports
    it — a test runner included, where starting a program is somebody else's
    business.

    Idempotent, because two entry points can be reached in one process
    (`memkit init` runs the checker, which is `memory_integrity`'s entry
    point in another process, but the eval loads a hook module in this one).
    """
    if _INSTALLED:
        return
    _INSTALLED.append(True)
    sys.addaudithook(_audit)

