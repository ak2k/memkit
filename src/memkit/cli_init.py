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
import hashlib
import json
import os
import subprocess
import sys

from memkit.cli_doctor import (
    CANARY_NAME,
    CONFIG_DIR_ENV,
    EXCLUDE_STRAY,
    Machine,
    _checker_route,
    authored_configs,
    canary_query,
)
from memkit.memory_prompt_recall import (
    DEFAULT_SEARCH_CLI,
    EXCLUDE_BASENAMES,
    PLUGIN_DATA_ENV,
    PLUGIN_SEARCH_CLI,
    SCHEMA,
    _display_path,
    _plugin_install,
)

SUMMARY = "create a store and wire this machine up to it"

# The manifest's own version. A consumer that parsed the text — and one will,
# because the skill relays it verbatim into a model's context — needs to know
# when its shape changed.
MANIFEST_SCHEMA = 1

# What init creates when nothing says otherwise. `~/notes` because that is what
# the README's own worked example has always used, and an adopter who followed
# it once should find init converging on the same directory rather than
# creating a second store beside it.
DEFAULT_STORE = "~/notes"
# The config, when neither `--config` nor the install option names one. The
# same default the plugin manifest declares, so an install that took the
# default and an install that skipped the option land in one place.
DEFAULT_CONFIG = "~/.config/memkit/memkit.json"

# The operations a plan can hold. Each is one filesystem effect, journalled at
# the moment it happens.
CREATE_DIR = "create-dir"
CREATE_FILE = "create-file"
REWRITE_FILE = "rewrite-file"
APPEND_LINE = "append-line"
SETTINGS_WRITE = "settings-write"
VERIFY = "verify"


def _sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _state_of(path: str) -> str:
    """What is at `path` now, as one comparable token.

    This is the half of the digest that makes it bind to the TREE rather than
    to the request: `absent`, `dir`, or the content hash of a file. A target
    that changed between the dry-run and the confirm changes this, and the
    confirm refuses without writing anything.
    """
    if os.path.isdir(path):
        return "dir"
    try:
        with open(path, encoding="utf-8") as f:
            return "file:" + _sha(f.read())
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}"
    except ValueError:
        return "file:unreadable-encoding"


class Action:
    """One filesystem effect, and everything needed to describe or journal it."""

    __slots__ = ("op", "path", "before", "content", "note", "authored_config")

    def __init__(
        self,
        op: str,
        path: str,
        content: str = "",
        note: str = "",
        authored_config: bool = False,
    ) -> None:
        self.op = op
        self.path = path
        self.before = _state_of(path)
        self.content = content
        self.note = note
        self.authored_config = authored_config

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


class Plan:
    """Everything init would do, in the order it would do it."""

    def __init__(self, actions: list, notes: list) -> None:
        self.actions = actions
        self.notes = notes

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
        for action in pending:
            lines.append(f"  {action.op:<14} {_display_path(action.path)}")
            if action.note:
                lines.append(f"                 {action.note}")
            resolved = _terminal_realpath(action.path)
            if resolved != os.path.abspath(action.path):
                lines.append(
                    f"                 -> resolves to {_display_path(resolved)}"
                )
            if action.before != "absent":
                lines.append(
                    f"                 (exists: {action.before.split(':')[0]})"
                )
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


def _terminal_realpath(path: str) -> str:
    """Where a path really lands, following every symlink in it.

    The manifest shows this whenever it differs, because "write to
    ~/notes/search/" and "write into whatever ~/notes points at" are different
    consents and the second one is the one being asked for.
    """
    return os.path.realpath(path)


# --- the content init writes -------------------------------------------------


def _canary_nonce(config_path: str, store: str) -> str:
    """The token doctor searches for, derived rather than random.

    Derived, and that is a decision: a random token would be regenerated on
    every run, so the dry-run's digest and the confirm's would never match and
    a converged install would look like a changed one. What the nonce has to be
    is unlikely to appear in the adopter's own corpus, which a derivation over
    two absolute paths satisfies as well as randomness does. It is not a
    secret; nothing is authorised by holding it.
    """
    return "mkc" + _sha(config_path + "\0" + store)[:10]


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


def _search_ledger(store: str, nonce: str) -> str:
    """SEARCH.md exactly as the checker would generate it.

    Written here rather than left to `--write`, and the difference matters: a
    regeneration would produce a correct ledger whatever init put in the file,
    so the checker run that follows would be verifying its own repair. Written
    this way, the run is a real check of what init did.
    """
    preamble = f"""# {os.path.basename(store) or 'memories'} — retrieval-only ledger

Generated from each memory's `description:` frontmatter. Never hand-edited —
run the integrity checker with `--write` after adding a memory.

## Index
"""
    row = (
        f"- [memkit-canary]({os.path.join('search', CANARY_NAME)}) — "
        + _canary_description(nonce)
    )
    return f"{preamble}\n{row}\n"


def _config_body(
    *, store: str, nonce: str, interpreter: str, store_id: str
) -> str:
    """The minimal working config, and every field in it is load-bearing.

    `interpreter` is recorded because PATH probing alone hands the process that
    reads every prompt to whatever direnv/mise/venv shim the launching shell
    carried. `search_cli` is written for the channel that is running, because
    one config file is read by every channel and a name that resolves on one
    resolves to nothing — or to another install's stores — on another. No
    `citations` block at all: it is optional, and a config that declared an
    empty one would make the first checker run an adopter does report two
    warnings about a feature they never opted into.
    """
    blob = {
        "schema": SCHEMA,
        "interpreter": interpreter,
        "search_cli": PLUGIN_SEARCH_CLI if _plugin_install() else DEFAULT_SEARCH_CLI,
        "canary_nonce": nonce,
        "roots": {store_id: {"kind": "path", "path": store}},
        "stores": [
            {
                "id": store_id,
                # Personal, and therefore ungated: a store the adopter reaches
                # from anywhere is the one that makes the first prompt after
                # init produce a pointer. A project store needs a `cwd_gate`
                # and a repository to gate to, and init has neither to guess
                # from.
                "role": "personal",
                "dir": ".",
                "live_root": store_id,
                "sub_indexes": [],
            }
        ],
    }
    return json.dumps(blob, indent=2) + "\n"


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


def _refuse_relative(what: str, path: str) -> None:
    """Absolute after `~` expansion, or it does not count.

    The same rule the wrappers enforce, for the same reason: a relative path
    resolves against whatever directory the session stands in, so an adopter
    who typed `--store notes` would get a different store in every repository
    they open — and a config decides which directories the every-prompt hook
    reads.
    """
    if not os.path.isabs(path):
        raise Refusal(
            "relative-path",
            f"the {what} path {path!r} is not absolute. A relative path names "
            "a different directory in every session, and the one thing a "
            "memory store may not be is a different store per directory.",
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
    interpreter = _interpreter()
    if not (os.path.isfile(interpreter) and os.access(interpreter, os.X_OK)):
        raise Refusal(
            "no-interpreter",
            f"no usable interpreter resolved ({interpreter!r}). The config "
            "init writes records the python that will read every prompt, and "
            "recording one that cannot run is an install that answers nothing.",
        )
    _refuse_relative("config", config_path)
    _refuse_relative("store", store_path)

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

    for what, path in (("config", config_path), ("store", store_path)):
        _refuse_unwritable(what, path)

    # The two writes that land OUTSIDE memkit's own paths, and the rule is the
    # same for both: a target that resolves inside a memory store is a memory
    # file that edits the harness's configuration. Nothing in this build writes
    # a memory, but the store is a directory an agent is told to write into.
    for flag, target, what in (
        (wire_claude_md, _claude_md(machine), "CLAUDE.md"),
        (auto_dream_off, _settings_path(machine), "settings.json"),
    ):
        if not flag:
            continue
        _refuse_relative(what, target)
        if _inside(target, store_path):
            raise Refusal(
                "store-resident-target",
                f"{_display_path(target)} resolves inside the memory store. A "
                "file the harness reads as configuration must not also be a "
                "file an agent is told to write memories into.",
            )
        _refuse_unwritable(what, target)

    route, _command = _checker_route(machine)
    if route == "none":
        raise Refusal(
            "no-checker-route",
            "no python meets the integrity checker's floor and there is no "
            "uvx to provision one, so init cannot verify the store it would "
            "seed. A seeded memory whose ledger nobody checked is a store the "
            "checker calls broken, and half-completing is worse than not "
            "starting.",
        )

    if os.path.isdir(store_path):
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
    """Where the config goes, in the order that makes an install converge.

    The INSTALL OPTION wins over the default, because an adopter who passed
    `--config memkitConfig=<path>` has already said where they want it and a
    config written anywhere else would leave the option pointing at nothing —
    which is the highest-cost silent state in the whole field log, created by
    the command that exists to prevent it.
    """
    if named:
        return os.path.expanduser(named)
    option, _scope = machine.settings_option()
    if option:
        return os.path.expanduser(option)
    return os.path.expanduser(DEFAULT_CONFIG)


def _interpreter() -> str:
    """The absolute python this process is, which is the one that will read
    every prompt if the wrapper honours the record.

    `sys.executable` resolved: a venv's `python3` is a symlink, and recording
    the link records a path whose target the adopter can move.
    """
    return os.path.realpath(sys.executable)


def _store_id(store: str) -> str:
    """A config id derived from the store's own directory name.

    Ids appear in `--debug-config`, in doctor's per-store rows and in the
    inert message, so `notes` reads better than `store-0` — and a store the
    adopter named is one they can recognise in a report.
    """
    base = os.path.basename(store.rstrip(os.sep)) or "memories"
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in base)
    return cleaned.strip("-") or "memories"


def _git_tracked(path: str) -> bool:
    """Whether git would consider this file part of a repository.

    Not a hard refusal — an adopter may well keep their `CLAUDE.md` in a
    dotfiles repo on purpose — but an `@-import` line appended to a tracked
    file is a commit somebody did not intend to make, and the manifest is the
    place that says so before it happens.
    """
    parent = os.path.dirname(path) or "."
    try:
        out = subprocess.run(
            ["git", "-C", parent, "ls-files", "--error-unmatch", path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def _import_line(store: str) -> str:
    return f"@{os.path.join(store, 'MEMORY.md')}"


def _claude_md(machine: Machine) -> str:
    config_dir = os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")
    return os.path.join(config_dir, "CLAUDE.md")


def _settings_path(machine: Machine) -> str:
    config_dir = os.environ.get(CONFIG_DIR_ENV) or os.path.expanduser("~/.claude")
    return os.path.join(config_dir, "settings.json")


def build_plan(
    machine: Machine,
    *,
    store: str | None = None,
    config: str | None = None,
    wire_claude_md: bool = False,
    auto_dream_off: bool = False,
) -> Plan:
    """Everything init would do, computed against the tree as it is now."""
    config_path = _resolve_config(machine, config)
    store_path = os.path.expanduser(store or DEFAULT_STORE)
    check_refusals(
        machine,
        config_path=config_path,
        store_path=store_path,
        wire_claude_md=wire_claude_md,
        auto_dream_off=auto_dream_off,
    )
    nonce = _canary_nonce(config_path, store_path)
    store_id = _store_id(store_path)
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
            CREATE_FILE,
            config_path,
            _config_body(
                store=store_path,
                nonce=nonce,
                interpreter=_interpreter(),
                store_id=store_id,
            ),
            note=f"records interpreter {_display_path(_interpreter())} and "
            f"canary nonce {nonce}",
            authored_config=True,
        ),
        Action(CREATE_DIR, store_path),
        # search/ FIRST, and the order in this list is the order they are made.
        # The trap init exists to prevent is a flat store that grows a `search/`
        # later: the moment that directory appears, every memory above it stops
        # being retrieved, silently, with every diagnostic still green.
        Action(
            CREATE_DIR,
            os.path.join(store_path, "search"),
            note="memories live here. A store without it retrieves from its "
            "root, and gaining one later un-retrieves everything above it.",
        ),
        Action(
            CREATE_DIR,
            os.path.join(store_path, "hot"),
            note="memories that load into every session, and which the hook "
            "never points at because they are already in context.",
        ),
        Action(
            CREATE_FILE,
            os.path.join(store_path, "MEMORY.md"),
            _memory_ledger(store_path),
        ),
        Action(
            CREATE_FILE,
            os.path.join(store_path, "search", CANARY_NAME),
            _canary_body(nonce),
            note="one memory, so the store answers something on the first "
            "prompt and doctor has a fixed query that can only match this file.",
        ),
        Action(
            CREATE_FILE,
            os.path.join(store_path, "SEARCH.md"),
            _search_ledger(store_path, nonce),
            note="generated from the frontmatter above, in the form the "
            "integrity checker generates.",
        ),
        Action(
            VERIFY,
            store_path,
            note="run the integrity checker over the finished store. Not with "
            "--write: a regeneration would repair whatever init got wrong and "
            "then report success.",
        ),
    ]
    notes = []
    if wire_claude_md:
        target = _claude_md(machine)
        existing = ""
        try:
            with open(target, encoding="utf-8") as f:
                existing = f.read()
        except OSError:
            existing = ""
        actions.append(
            Action(
                APPEND_LINE,
                target,
                existing.rstrip("\n") + "\n" + _import_line(store_path) + "\n"
                if existing.strip()
                else _import_line(store_path) + "\n",
                note=f"appends {_import_line(store_path)}",
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
                note='sets "autoDreamEnabled": false and changes nothing else',
            )
        )
        notes.append(
            "Turning auto-dream off stops the harness writing and "
            "consolidating its own memories beside memkit's. It is the only "
            "settings key init will ever write."
        )
    return Plan(actions, notes)


# The complete set of settings keys init may write, as data.
#
# The rule this enforces is "the plugin never enables itself", and
# `enabledPlugins` is the key that would do it — but the guard is an ALLOWLIST
# rather than a check on that name, because the next key with the same power
# has not been named yet and a denylist only ever catches the ones somebody
# thought of.
SETTINGS_KEYS_INIT_MAY_WRITE = frozenset({"autoDreamEnabled"})


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
            "settings. The only key it may write is "
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
    return json.dumps(blob, indent=2) + "\n"


def _settings_with_auto_dream_off(path: str) -> str:
    return _settings_with(path, {"autoDreamEnabled": False})


# --- the command -------------------------------------------------------------


EXIT_OK = 0
# argparse's and the dispatcher's, and not reassignable.
EXIT_USAGE = 2
# A named refusal, and nothing was written. Its own code rather than 1, because
# 1 already means two things a caller has to tell apart — the wrapper could not
# start, and doctor found problems — and neither of them is "you asked for
# something this will not do". A skill branches on this to relay the reason to
# the person and stop, which is a different move from retrying.
EXIT_REFUSED = 5


EPILOG = """\
Two turns, and the second one is not the same command with a flag:

  memkit init --dry-run           print a manifest of every path and every
                                  write, plus a digest. Writes nothing.
  memkit init --confirm <digest>  recompute the manifest, refuse if anything
                                  under it changed, re-emit it, then apply.

The digest binds the state of the tree, not the text you read. Pass the same
flags to both calls: a different request produces a different digest.

Exit codes: 0 done (or the manifest printed) / 2 usage error / 5 refused, and
nothing was written — stderr names which refusal."""


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
        help='set "autoDreamEnabled": false, so the harness stops writing its '
        "own memories beside memkit's. The only settings key init will write",
    )


def run(args: argparse.Namespace) -> int:
    machine = Machine()
    try:
        plan = build_plan(
            machine,
            store=getattr(args, "store", None),
            config=getattr(args, "config", None),
            wire_claude_md=getattr(args, "wire_claude_md", False),
            auto_dream_off=getattr(args, "auto_dream_off", False),
        )
    except Refusal as refusal:
        return _refuse(refusal)
    print(plan.render())
    return EXIT_OK


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
