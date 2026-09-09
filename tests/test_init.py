"""Unit tests for `memkit init`.

Two properties carry most of this file. The first is that a dry-run writes
nothing — asserted with `diff -rq` over the whole scratch profile rather than
by inspecting the one file a case is about, because "it did not create the
config" and "it created nothing" are different claims and only the second one
is what a consent handshake promises.

The second is that the digest binds the TREE. A manifest that bound only the
request would let a file appear between the two turns and be silently
overwritten by a confirm the human approved for a different world.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import pathlib
import signal
import stat
import subprocess
import sys
import time

import pytest

from memkit import _exec, harness_memory
from memkit import cli_doctor as doctor
from memkit import cli_init as init
from memkit import memory_prompt_recall as hook


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_dir = tmp_path / "claude-config"
    project = tmp_path / "project"
    for path in (home, config_dir, project):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(doctor.CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.chdir(project)
    for name in (
        hook.CONFIG_ENV,
        hook.PLUGIN_ENV,
        hook.PLUGIN_DATA_ENV,
        "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG",
        "CLAUDE_PLUGIN_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path
    # `_use_config` sets module globals and clears caches; a case that pointed
    # the reader at a fixture config would otherwise leave every later case in
    # this process reading it.
    hook._use_config(None)


def _which_git() -> str:
    """A trusted `git`, or "" — the same lookup the code under test uses.

    `shutil.which` would answer for a git these cases then could not run,
    which is a skip that hides a real failure.
    """
    try:
        return _exec.resolve("git")
    except _exec.Untrusted:
        return ""


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace(
        dry_run=True,
        confirm=None,
        store=None,
        config=None,
        wire_claude_md=False,
        auto_dream_off=False,
        adopt_auto_memory=False,
        auto_memory_off=False,
        subcommand="init",
    )
    for key, value in kw.items():
        setattr(ns, key, value)
    return ns


def _snapshot(root) -> dict:
    """Every file under `root`, by path and content hash.

    The whole profile, not the one path a case is about: a refusal that
    created the state directory before deciding to refuse would pass every
    assertion written about the file it refused to write.
    """
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                with open(path, "rb") as f:
                    out[os.path.relpath(path, root)] = f.read()
            except OSError as exc:
                # A file this process cannot read is still a file that must be
                # there, unchanged, afterwards. Recording the failure keeps the
                # snapshot total — a helper that raised would fail the case
                # before the code under test ever ran.
                out[os.path.relpath(path, root)] = f"unreadable:{type(exc).__name__}"
        for name in _dirnames:
            out[os.path.relpath(os.path.join(dirpath, name), root) + "/"] = b"(dir)"
    return out


def _plan(profile, **kw) -> init.Plan:
    return init.build_plan(doctor.Machine(), **kw)


# --- the manifest ------------------------------------------------------------


def test_a_dry_run_writes_nothing_at_all(profile, capsys) -> None:
    """`diff -rq` over the whole profile, in effect. Not "the config is not
    there" — "nothing is there that was not there before"."""
    before = _snapshot(profile)
    assert init.run(_args()) == init.EXIT_OK
    assert _snapshot(profile) == before
    printed = capsys.readouterr().out
    assert "digest:" in printed


def test_the_manifest_names_every_path_it_would_touch(profile) -> None:
    """Every path and every write, because the human is being asked to consent
    to those and not to a summary of them."""
    plan = _plan(profile, store=str(profile / "notes"))
    rendered = plan.render()
    for expected in (
        str(profile / "home" / ".cache" / "memory-recall"),
        str(profile / "notes" / "search"),
        str(profile / "notes" / "hot"),
        str(profile / "notes" / "MEMORY.md"),
        str(profile / "notes" / "SEARCH.md"),
        str(profile / "notes" / "search" / doctor.CANARY_NAME),
    ):
        assert hook._display_path(expected) in rendered, expected


def test_the_manifest_keeps_its_own_indentation(profile) -> None:
    """The one place the pointer sanitizer must NOT be applied line by line: it
    collapses runs of whitespace, and the indentation is what makes a list of
    paths readable. This text is relayed verbatim into a transcript a person
    reads."""
    rendered = _plan(profile).render()
    assert any(line.startswith("  create-dir") for line in rendered.splitlines())


def test_a_hostile_path_is_stripped_without_losing_its_spacing(profile) -> None:
    """A path is something to open, so the only permitted edit is removing
    characters that were never visible — a collapsed path with two spaces in it
    names nothing."""
    store = profile / "two  spaces\rand a return"
    rendered = _plan(profile, store=str(store)).render()
    assert "\r" not in rendered
    assert "two  spaces" in rendered


def test_the_manifest_shows_where_a_symlink_actually_lands(profile) -> None:
    """"Write to ~/notes/search" and "write into whatever ~/notes points at"
    are different consents, and only the second one is being asked for."""
    real = profile / "elsewhere"
    real.mkdir()
    link = profile / "home" / "notes"
    link.symlink_to(real)
    rendered = _plan(profile, store=str(link)).render()
    assert "resolves to" in rendered
    assert hook._display_path(str(real)) in rendered


def test_the_grouped_copies_each_say_where_they_actually_land(profile) -> None:
    """AND THE GROUPED BRANCH TOO, which is the branch every adopted memory
    goes through.

    A store on an external volume, in a dotfiles tree or under a synced
    directory is the common shape this is for, and there the summary line
    ("2 files from ... -> ~/notes/search/projects/-k/") names a path that is
    not where a single byte lands. The count folds the copies together; the
    resolution is per file, so it is asserted per file.
    """
    _harness(profile, "-home-u", {"one.md": TRAP, "two.md": TRAP})
    real = profile / "external-volume" / "notes"
    real.mkdir(parents=True)
    store = profile / "home" / "notes"
    store.symlink_to(real)
    lines = _plan(
        profile, store=str(store), adopt_auto_memory=True
    ).render().splitlines()
    grouped = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("create-file") and "files from" in line
    ]
    assert len(grouped) == 1, lines
    for name in ("one.md", "two.md"):
        member = next(
            i for i in range(grouped[0], len(lines))
            if lines[i].strip() == str(store / "search" / "projects" / "-home-u" / name)
            or lines[i].strip() == hook._display_path(
                str(store / "search" / "projects" / "-home-u" / name)
            )
        )
        assert lines[member + 1].strip() == "-> resolves to " + hook._display_path(
            str(real / "search" / "projects" / "-home-u" / name)
        ), lines[member : member + 2]


# --- the digest --------------------------------------------------------------


def test_the_digest_is_stable_across_runs_on_an_unchanged_tree(profile) -> None:
    """Otherwise the handshake cannot be completed at all: the confirm
    recomputes, and a digest that moved on its own would refuse every time."""
    assert _plan(profile).digest == _plan(profile).digest


def test_the_digest_moves_when_the_target_state_moves(profile) -> None:
    """It binds the TREE, not the request. A file that appeared between the two
    turns is a world the human did not approve."""
    store = profile / "notes"
    # The store itself exists in BOTH plans, deliberately. The verification
    # step's `after` is its `before`, so moving the store's own state would
    # move the digest through that action whether or not any other one carried
    # the tree — and the property under test is that a CREATE_DIR whose result
    # is always "dir" still binds what was there first.
    store.mkdir()
    before = _plan(profile, store=str(store)).digest
    (store / "hot").mkdir()
    assert _plan(profile, store=str(store)).digest != before


def test_the_digest_moves_when_the_request_moves(profile) -> None:
    """Pass the same flags to both calls: a different request is a different
    plan and has to be a different digest, or `--confirm <digest>` would apply
    something else under an approved number."""
    plain = _plan(profile).digest
    assert _plan(profile, store=str(profile / "other")).digest != plain
    assert _plan(profile, wire_claude_md=True).digest != plain
    assert _plan(profile, auto_dream_off=True).digest != plain


def test_a_converged_install_manifests_nothing(profile) -> None:
    """Double init is a no-op, and the manifest says so rather than listing
    writes that would change nothing."""
    plan = _plan(profile, store=str(profile / "notes"))
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_OK
    converged = _plan(profile, store=str(profile / "notes"))
    assert converged.writes == []
    assert "already set up" in converged.render()
    # Verification is not a write and still runs: a second init has nothing to
    # do and still has something to check.
    assert [a.op for a in converged.pending] == [init.VERIFY]
    # The digest MOVED, and that is the binding working: it names the state of
    # the tree, and the tree changed. What it must not do is collapse — a plan
    # that dropped its redundant actions would hash the same as one that never
    # had them, and "already done" and "a step went missing" would stop being
    # different answers.
    assert converged.digest != plan.digest
    assert len(converged.actions) == len(plan.actions)
    # And it must not COLLAPSE. On a converged tree every write is redundant,
    # so a digest taken over what is left to do would hash a plan that lost a
    # step identically to one that never had it — "already done" and "a step
    # went missing" are different answers and only one of them is safe to
    # apply.
    dropped = init.Plan(
        [a for a in converged.actions if a.op != init.CREATE_DIR], converged.notes
    )
    assert dropped.pending == converged.pending
    assert dropped.digest != converged.digest


# --- what init writes --------------------------------------------------------


def test_the_store_starts_in_search_and_never_flat(profile) -> None:
    """The layout trap init exists to prevent. A flat store that grows a
    `search/` later un-retrieves everything above it in one step, silently,
    with every diagnostic green — three of four reviewers reproduced that and
    two lost the memory the quick start had just had them create."""
    plan = _plan(profile, store=str(profile / "notes"))
    dirs = [a.path for a in plan.actions if a.op == init.CREATE_DIR]
    assert str(profile / "notes" / "search") in dirs
    assert str(profile / "notes" / "hot") in dirs
    files = [a.path for a in plan.actions if a.op == init.CREATE_FILE]
    # Every memory init writes is UNDER search/. A memory at the store root is
    # the state the trap springs from.
    memories = [f for f in files if f.endswith(".md") and "MEMORY" not in f]
    assert memories
    for path in memories:
        if os.path.basename(path) == "SEARCH.md":
            continue
        assert os.sep + "search" + os.sep in path, path


def test_the_config_records_the_interpreter_and_the_nonce(profile) -> None:
    """PATH probing alone hands the process that reads every prompt to whatever
    shim the launching shell carried."""
    plan = _plan(profile, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.path.endswith("memkit.json")]
    blob = json.loads(action.content)
    assert blob["schema"] == hook.SCHEMA
    assert os.path.isabs(blob["interpreter"])
    assert blob["canary_nonce"]
    assert blob["stores"][0]["role"] == "personal"
    assert "cwd_gate" not in blob["stores"][0]
    # No citations block at all: it is optional, and an empty one makes the
    # first checker run an adopter does report two warnings about a feature
    # they never opted into.
    assert "citations" not in blob


def test_the_config_names_a_search_command_this_channel_ships(profile, monkeypatch):
    """One config file is read by every channel, and a name that resolves on
    one resolves to nothing — or to another install's stores — on another."""
    plain = json.loads(
        [a for a in _plan(profile).actions if a.path.endswith("memkit.json")][0].content
    )
    assert plain["search_cli"] == hook.DEFAULT_SEARCH_CLI

    monkeypatch.setenv(hook.PLUGIN_ENV, "1")
    # A plugin install needs a route the wrapper reads, or init refuses before
    # it gets as far as choosing a command to advertise.
    data = profile / "plugin-data"
    data.mkdir()
    monkeypatch.setenv(hook.PLUGIN_DATA_ENV, str(data))
    plugin = json.loads(
        [a for a in _plan(profile).actions if a.path.endswith("memkit.json")][0].content
    )
    assert plugin["search_cli"] == hook.PLUGIN_SEARCH_CLI


def test_the_canary_description_is_under_the_checkers_cap(profile) -> None:
    """The cap is the CHECKER's 155 and not the hook's 157: a memory written to
    the hook's ceiling fails the check, and init must never seed a store its
    own checker rejects."""
    from memkit import memory_integrity as checker

    plan = _plan(profile, store=str(profile / "notes"))
    (canary,) = [a for a in plan.actions if a.path.endswith(doctor.CANARY_NAME)]
    description = ""
    for line in canary.content.splitlines():
        if line.startswith("description: "):
            description = line[len("description: "):]
        elif description and line.startswith("  "):
            description += " " + line.strip()
        elif description:
            break
    assert description
    assert len(description) <= checker.MAX_DESC_CHARS, len(description)


def test_the_nonce_is_derived_so_the_handshake_can_complete(profile) -> None:
    """A random token would be regenerated on every run, so the dry-run's
    digest and the confirm's would never match and a converged install would
    look like a changed one. What the nonce has to be is unlikely to appear in
    the adopter's own corpus, which a derivation over two absolute paths
    satisfies as well as randomness does."""
    first = init._canary_nonce("/a/config.json")
    assert first == init._canary_nonce("/a/config.json")
    assert first != init._canary_nonce("/z/config.json")


def test_the_config_goes_where_the_install_option_already_points(profile, monkeypatch):
    """An adopter who passed `--config memkitConfig=<path>` has said where they
    want it, and a config written anywhere else leaves the option naming
    nothing — the highest-cost silent state in the field log, created by the
    command that exists to prevent it."""
    named = profile / "elsewhere" / "memkit.json"
    settings = profile / "claude-config" / "settings.json"
    settings.write_text(
        json.dumps(
            {"pluginConfigs": {"memkit@memkit": {"options": {"memkitConfig": str(named)}}}}
        ),
        encoding="utf-8",
    )
    plan = _plan(profile)
    assert any(a.path == str(named) for a in plan.actions), [
        a.path for a in plan.actions
    ]
    # And an explicit --config still wins over both.
    explicit = str(profile / "explicit.json")
    assert any(a.path == explicit for a in _plan(profile, config=explicit).actions)


# --- the command surface -----------------------------------------------------


def _run(*argv: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "memkit.cli", "init", *argv],
        capture_output=True,
        text=True,
        timeout=120,
        env=env if env is not None else os.environ,
    )


def test_a_refusal_raised_before_the_plan_is_still_exit_five(profile) -> None:
    """A refusal is exit 5 wherever it was raised.

    `no-config-route` was raised by the resolver, one line ABOVE the try that
    turns refusals into exit 5, so a plugin install with no option and no
    usable plugin-data directory printed a traceback and exited 1 — which the
    published table reads as "memkit could not start at all". An agent given
    that goes off to reinstall a working install, and the one thing the
    refusal contract promises is that a refusal is a decision rather than a
    crash.
    """
    env = dict(
        os.environ,
        HOME=str(profile / "home"),
        XDG_CACHE_HOME=str(profile / "home" / ".cache"),
        MEMKIT_PLUGIN="1",
    )
    env.pop(hook.PLUGIN_DATA_ENV, None)
    env.pop("CLAUDE_PLUGIN_OPTION_MEMKITCONFIG", None)
    env[doctor.CONFIG_DIR_ENV] = str(profile / "claude-config")
    before = _snapshot(profile)
    out = _run("--dry-run", "--store", str(profile / "notes"), env=env)
    assert out.returncode == init.EXIT_REFUSED, (out.returncode, out.stderr)
    assert "refused (no-config-route)" in out.stderr, out.stderr
    assert "Traceback" not in out.stderr, out.stderr
    assert _snapshot(profile) == before, "a refusal wrote something"


def test_nothing_that_can_refuse_runs_before_the_guard(profile, monkeypatch) -> None:
    """The guard is structural, because the defect was.

    `_refuses` and the inventory scrape both call `build_plan` directly, so a
    refusal that never reaches `_refuse()` counted as covered by both — which
    is precisely how one shipped. The rule this pins is not "these two call
    sites are wrapped" but "nothing that can raise runs outside the wrapper".
    """
    import ast

    source = (
        pathlib.Path(init.__file__).read_text(encoding="utf-8")
        if hasattr(init, "__file__")
        else ""
    )
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    guarded = next(
        (node for node in fn.body if isinstance(node, ast.Try)), None
    )
    assert guarded is not None, "run() has no refusal guard at all"
    assert any(
        isinstance(h.type, ast.Name) and h.type.id == "Refusal"
        for h in guarded.handlers
    ), ast.dump(guarded)
    before = fn.body[: fn.body.index(guarded)]
    calls = [
        ast.unparse(node.func)
        for statement in before
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    # NOTHING, not even `Machine()`. The allowance for that one call is what
    # left the line that reads the session directory outside the guard, and
    # the session directory can be removed under this process.
    assert calls == [], calls

    # And behaviourally, at both sites, so the shape above is not the only
    # thing standing.
    for target in ("_resolve_config", "build_plan"):
        # PUT BACK BY NAME. `monkeypatch.undo()` here would also undo the
        # `profile` fixture's HOME and config directory — it is the same
        # monkeypatch — and the next iteration's unpatched half would plan
        # against the developer's real machine.
        real = getattr(init, target)
        monkeypatch.setattr(
            init,
            target,
            lambda *a, **k: (_ for _ in ()).throw(init.Refusal("synthetic", "no")),
        )
        assert init.run(_args()) == init.EXIT_REFUSED
        monkeypatch.setattr(init, target, real)


def test_init_requires_a_mode_rather_than_defaulting_to_one(profile) -> None:
    """A mutating command with a default mode is one an agent runs by
    accident. Neither mode is the default; the caller says which."""
    out = _run(env=dict(os.environ, HOME=str(profile / "home")))
    assert out.returncode == init.EXIT_USAGE
    assert "--dry-run" in out.stderr and "--confirm" in out.stderr


def test_the_two_modes_are_mutually_exclusive(profile) -> None:
    out = _run("--dry-run", "--confirm", "abc",
               env=dict(os.environ, HOME=str(profile / "home")))
    assert out.returncode == init.EXIT_USAGE


def test_the_help_names_both_turns_and_every_exit_code(profile) -> None:
    """`--help` is the cheapest probe an agent makes, and a two-turn handshake
    it does not describe is one an agent will collapse into one turn."""
    out = _run("--help", env=dict(os.environ, HOME=str(profile / "home")))
    assert out.returncode == 0
    collapsed = " ".join(out.stdout.split())
    assert "--dry-run" in collapsed and "--confirm" in collapsed
    for code in (init.EXIT_OK, init.EXIT_USAGE, init.EXIT_REFUSED):
        assert f"{code} " in collapsed, code
    assert "binds the state of the tree" in collapsed


# --- the refusals ------------------------------------------------------------
#
# Each one asserts over the WHOLE profile rather than over the file it is
# about: "it did not write the config" and "it wrote nothing" are different
# claims, and a refusal that created the state directory before deciding to
# refuse would satisfy the first.


def _claim(profile, config) -> None:
    """Journal the config as one init authored.

    The read and parse refusals below are only reachable for memkit's OWN
    file: a config no journal claims is refused earlier, and for a better
    reason — `foreign-config`, which says memkit did not write it.
    """
    state = profile / "home" / ".cache" / "memory-recall"
    state.mkdir(parents=True, exist_ok=True)
    with open(state / hook.INIT_JOURNAL_NAME, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"v": 1, "op": "merge-config", "path": str(config),
                 "authored_config": True}
            )
            + "\n"
        )


def _refuses(profile, name: str, **kw) -> init.Refusal:
    before = _snapshot(profile)
    with pytest.raises(init.Refusal) as caught:
        _plan(profile, **kw)
    assert caught.value.name == name, caught.value.name
    assert _snapshot(profile) == before, "a refusal wrote something"
    return caught.value


def test_windows_is_refused_by_name_rather_than_met_as_a_failure(profile, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    refusal = _refuses(profile, "windows")
    assert "POSIX" in refusal.message


def test_a_relative_store_or_config_is_refused(profile) -> None:
    """The same rule the wrappers enforce: a relative path names a different
    directory in every session, and the one thing a memory store may not be is
    a different store per directory."""
    _refuses(profile, "relative-path", store="notes")
    _refuses(profile, "relative-path", config="memkit.json")


def test_init_writes_only_where_the_hook_would_read(profile, monkeypatch) -> None:
    """The writer and the readers admit exactly the same paths.

    The no-option door on the plugin channel is closed and the
    malformed-option door beside it was not: `memkitConfig` with a doubled slash —
    which shell variable concatenation at install time produces on its own —
    got a store, a green integrity check and exit 0, while
    `memkit_resolve_config` refused that shape and served every prompt
    nothing. One doubled character, and the manifest asserted the opposite.

    Table-driven over the shapes the wrapper's own rule names, and each case
    asserts NOTHING WAS WRITTEN, because a refusal that got halfway is the
    state this whole command exists to avoid.
    """
    from memkit.memory_prompt_recall import path_refusal

    good = str(profile / "cfg" / "memkit.json")
    for bad in (
        str(profile) + "//cfg/memkit.json",
        str(profile) + "/cfg/./memkit.json",
        str(profile) + "/cfg/../cfg/memkit.json",
        "/proc/self/cwd/memkit.json",
        "/dev/fd/3/memkit.json",
    ):
        assert path_refusal(bad), bad
        refusal = _refuses(profile, "non-canonical-path", config=bad)
        assert bad in refusal.message, refusal.message
        assert path_refusal(bad) in refusal.message, refusal.message
    # The store is admitted by the same rule: `/proc/self/cwd/notes` is
    # absolute and is a different directory in every session.
    _refuses(profile, "non-canonical-path", store="/proc/self/cwd/notes")
    # And an unexpanded `~someone` is what `os.path.expanduser` would have
    # turned into an absolute path the shell leaves alone.
    _refuses(profile, "relative-path", config="~nobody/memkit.json")
    # The control: the same rule admits the ordinary case.
    assert path_refusal(good) == ""
    _plan(profile, config=good, store=str(profile / "notes"))


def test_the_option_rung_is_vetted_the_way_the_wrapper_vets_it(
    profile, monkeypatch
) -> None:
    """The rung init trusts unconditionally is the one the shell vets.

    `--config` is typed at the moment of the run; the `memkitConfig` option was
    typed once, at install, and is read back out of settings — so it is the
    rung where a bad shape survives long enough to be written to.
    """
    def _option(value: str) -> None:
        (profile / "claude-config" / "settings.json").write_text(
            json.dumps(
                {
                    "pluginConfigs": {
                        doctor.PLUGIN_KEY: {"options": {doctor.OPTION_KEY: value}}
                    }
                }
            ),
            encoding="utf-8",
        )

    _option(str(profile) + "//cfg/memkit.json")
    _refuses(profile, "non-canonical-path", store=str(profile / "notes"))
    # `~someone` is the case that separates the two expansions: the shell
    # leaves it alone and refuses it as relative, and `os.path.expanduser`
    # turns it into an absolute path init would have written to.
    _option("~nobody/memkit.json")
    _refuses(profile, "relative-path", store=str(profile / "notes"))


def test_a_store_inside_the_plugin_data_directory_is_refused(profile, monkeypatch):
    """Plugin data dies with the plugin unless somebody remembers
    `--keep-data`. A memory store must outlive the plugin that reads it."""
    data = profile / "plugin-data"
    data.mkdir()
    monkeypatch.setenv(hook.PLUGIN_DATA_ENV, str(data))
    refusal = _refuses(profile, "store-in-plugin-data", store=str(data / "notes"))
    assert "--keep-data" in refusal.message


def test_a_store_reached_by_symlink_into_plugin_data_is_refused(profile, monkeypatch):
    """The case a prefix test misses. The store is `~/notes`; `~/notes` is a
    symlink into plugin data."""
    data = profile / "plugin-data"
    data.mkdir()
    monkeypatch.setenv(hook.PLUGIN_DATA_ENV, str(data))
    link = profile / "home" / "notes"
    link.symlink_to(data)
    _refuses(profile, "store-in-plugin-data", store=str(link))


def test_a_store_inside_the_plugin_payload_is_refused(profile, monkeypatch) -> None:
    """The payload is a clone of a pinned commit: a store there is a store the
    repository can ship, and it is replaced wholesale on the next update."""
    payload = profile / "payload"
    payload.mkdir()
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(payload) + "/")
    _refuses(profile, "store-in-plugin-root", store=str(payload / "store"))


def test_an_unwritable_target_is_refused_before_the_first_byte(profile) -> None:
    locked = profile / "locked"
    locked.mkdir(mode=0o500)
    try:
        _refuses(profile, "not-writable", store=str(locked / "notes"))
    finally:
        locked.chmod(0o700)


def test_a_claude_md_that_resolves_inside_the_store_is_refused(profile, monkeypatch):
    """A file the harness reads as configuration must not also be a file an
    agent is told to write memories into."""
    store = profile / "notes"
    store.mkdir()
    target = profile / "claude-config" / "CLAUDE.md"
    real = store / "CLAUDE.md"
    real.write_text("# mine\n", encoding="utf-8")
    target.symlink_to(real)
    _refuses(
        profile, "store-resident-target", store=str(store), wire_claude_md=True
    )


def test_a_settings_file_that_resolves_inside_the_store_is_refused(profile):
    store = profile / "notes"
    store.mkdir()
    real = store / "settings.json"
    real.write_text("{}", encoding="utf-8")
    (profile / "claude-config" / "settings.json").symlink_to(real)
    _refuses(
        profile, "store-resident-target", store=str(store), auto_dream_off=True
    )


def test_an_unparseable_settings_file_is_refused_rather_than_replaced(profile):
    """The field anti-pattern the prior-art survey names: a tool that meets a
    parse error and replaces the file with a stub takes the whole
    configuration with it."""
    (profile / "claude-config" / "settings.json").write_text(
        "{ not json", encoding="utf-8"
    )
    refusal = _refuses(profile, "unparseable-settings", auto_dream_off=True)
    assert "will not replace" in refusal.message


def test_an_existing_memory_index_is_refused_rather_than_replaced(profile) -> None:
    """`MEMORY.md` is in `EXCLUDE_BASENAMES`, so the stray scan never saw it.

    A store holding only a hand-written index therefore passed
    `flat-store-adoption`, the manifest listed `create-file <store>/MEMORY.md`
    with a bare `(exists: file)` beside it, and `--confirm` exited 0 having
    replaced the adopter's hot-tier rows with the generated template. Nothing
    else records those rows.
    """
    store = profile / "notes"
    store.mkdir()
    mine = store / "MEMORY.md"
    mine.write_text(
        "# My index\n\n## Index\n\n- [a](hot/a.md) — must not be lost\n",
        encoding="utf-8",
    )
    refusal = _refuses(profile, "adopted-memory-index", store=str(store))
    assert "loads into every session" in refusal.message
    assert mine.read_text().startswith("# My index")

    # And the file init itself generates is not somebody else's, so the
    # recovery the incomplete exit code advertises still converges.
    mine.write_text(init._memory_ledger(str(store)), encoding="utf-8")
    assert _plan(profile, store=str(store)) is not None


def test_an_interior_store_conflict_refuses_before_the_first_write(
    profile,
) -> None:
    """The store ROOT had this rule and its descendants did not.

    A regular file at `<store>/search` printed a normal manifest, and
    `--confirm` then created the 0700 state directory and wrote the config
    before `os.makedirs` raised `FileExistsError` — exit 6 and a
    half-configured install, from a command whose contract is that it refuses
    safely.
    """
    store = profile / "notes"
    store.mkdir()
    (store / "search").write_text("not a directory\n", encoding="utf-8")
    refusal = _refuses(profile, "not-a-directory", store=str(store))
    assert "Nothing has been written" in refusal.message

    # The other direction, on a path init writes a FILE to.
    (store / "search").unlink()
    (store / "SEARCH.md").mkdir()
    refusal = _refuses(profile, "not-a-file", store=str(store))
    assert "Nothing has been written" in refusal.message


def test_a_config_dir_inside_the_session_is_refused_for_both_targets(
    profile, monkeypatch
) -> None:
    """A repository choosing where a config is READ from is one shape of the
    defect; this is the same shape on the WRITER.

    `$CLAUDE_CONFIG_DIR` decides where `CLAUDE.md` and `settings.json` go, and
    a checkout that exports one through direnv gets the `@-import` appended to
    the REPOSITORY's `CLAUDE.md` — a file loaded into every session in that
    project — while `--auto-dream-off` rewrites the repository's settings and
    leaves the adopter's own untouched, reporting success for a change nobody
    asked for. `settings_scopes()` already declares this variable untrusted on
    the read side; nothing did on the write side.
    """
    inside = profile / "project" / ".claude"
    inside.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(doctor.CONFIG_DIR_ENV, str(inside))
    for kw in ({"wire_claude_md": True}, {"auto_dream_off": True}):
        refusal = _refuses(profile, "config-dir-in-session-directory", **kw)
        assert doctor.CONFIG_DIR_ENV in refusal.message, refusal.message


def test_an_unreadable_memory_index_is_refused_by_name(profile) -> None:
    """`--dry-run` is the pre-approved half of the handshake, so a traceback
    out of it is a state an adopter meets with no refusal name to act on.

    Two shapes reach the same read and used to end differently: a file the
    process cannot OPEN refused, and one it can open and cannot DECODE raised
    UnicodeDecodeError three frames away. A file that is not UTF-8 is not the
    file init generates, which is the branch beside it.
    """
    store = profile / "store"
    (store / "search").mkdir(parents=True)
    (store / "hot").mkdir(parents=True)
    (store / "MEMORY.md").write_bytes(b"# index\n\xff\xfe not utf-8\n")
    refusal = _refuses(profile, "adopted-memory-index", store=str(store))
    assert "somebody wrote" in refusal.message, refusal.message


def test_the_plan_preflight_sees_the_actions_the_flags_add(
    profile, monkeypatch
) -> None:
    """The preflight's whole job is to see the WHOLE plan: two paths of
    incompatible types in one manifest is a plan that cannot be performed, and
    finding that out at apply time means finding it out after a write.

    Run where the action list was still being built it checked eight of ten —
    the two the flags append came after it — so the one check that has to see
    everything saw the part that never varies.
    """
    import ast

    source = pathlib.Path(init.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and "_refuse_incompatible_types" in
        {getattr(c.func, "id", "") for c in ast.walk(n) if isinstance(c, ast.Call)}
    )
    calls = [
        node.lineno for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "_refuse_incompatible_types"
    ]
    appends = [
        node.lineno for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(getattr(node.func, "value", None), "id", "") == "actions"
        and getattr(node.func, "attr", "") == "append"
    ]
    assert calls and appends, (calls, appends)
    assert min(calls) > max(appends), (
        "the preflight runs before an action the flags append, so it checks "
        "part of the plan"
    )


def test_the_package_hands_the_checker_its_own_src_and_not_the_sessions(
    monkeypatch,
) -> None:
    """`PYTHONPATH` is stripped from every child and was then concatenated back
    on by the one call site that needed to ADD to it — the scrub undone by the
    code that most needed it.

    `python -m` reads the module it runs out of that string, so a session that
    exported a `PYTHONPATH` named the code the checker imported, on the write
    turn, under a consent given for `memkit init --confirm <digest>`.
    """
    monkeypatch.setenv("PYTHONPATH", "/session/evil")
    here = os.path.dirname(os.path.dirname(os.path.abspath(init.__file__)))
    assert init._package_path() == here, init._package_path()
    assert "/session/evil" not in init._package_path()


def test_no_checker_route_is_refused_rather_than_half_completed(profile, monkeypatch):
    """A seeded memory whose ledger nobody checked is a store the checker calls
    broken. Half-completing is worse than not starting."""
    monkeypatch.setattr(
        doctor, "_probe_checker_route", lambda: (_exec.CheckerRoute.NONE, "")
    )
    refusal = _refuses(profile, "no-checker-route")
    assert "uv" in refusal.message
    # The adopter-facing cost of locating rather than provisioning is a named
    # one-time command, not a silent download.
    assert "uv python install 3.12" in refusal.message, refusal.message


def test_adopting_a_flat_store_is_refused_and_the_refusal_names_the_migration(
    profile,
) -> None:
    """The trap, met from the other side. Creating `search/` in a store that
    already holds memories at its root un-retrieves every one of them in a
    single step, silently, with every diagnostic green."""
    store = profile / "notes"
    store.mkdir()
    (store / "postgres-pooling.md").write_text("---\nname: x\n---\nbody\n")
    (store / "README.md").write_text("# not a memory\n")
    refusal = _refuses(profile, "flat-store-adoption", store=str(store))
    assert "postgres-pooling.md" in refusal.message
    # A README at a store root is not a memory and is not named as one.
    assert "README.md" not in refusal.message
    # The one-step migration, spelled out.
    assert "mkdir" in refusal.message and "mv" in refusal.message


def test_a_store_that_already_has_search_is_not_a_flat_store(profile) -> None:
    """The refusal is about the TRANSITION, not the layout: a store already in
    the tiered shape has nothing to strand."""
    store = profile / "notes"
    (store / "search").mkdir(parents=True)
    (store / "README.md").write_text("# fine\n")
    plan = _plan(profile, store=str(store))
    assert plan.writes


def test_a_config_no_journal_claims_is_never_overwritten(profile) -> None:
    """init converges on its own work. That file decides which directories the
    every-prompt hook reads, and a setup command that silently replaced a
    hand-written one would be the memory-poisoning surface of the design."""
    config = profile / "mine.json"
    config.write_text('{"schema": 1}', encoding="utf-8")
    refusal = _refuses(profile, "foreign-config", config=str(config))
    assert "memkit did not write it" in refusal.message


def test_a_config_the_journal_claims_is_converged_on(profile) -> None:
    state = profile / "home" / ".cache" / "memory-recall"
    state.mkdir(parents=True)
    config = profile / "mine.json"
    config.write_text('{"schema": 1}', encoding="utf-8")
    (state / hook.INIT_JOURNAL_NAME).write_text(
        json.dumps(
            {"v": 1, "op": "create-file", "path": str(config), "authored_config": True}
        )
        + "\n",
        encoding="utf-8",
    )
    plan = _plan(profile, config=str(config))
    assert any(a.path == str(config) for a in plan.writes)


def test_init_never_writes_enabled_plugins(profile) -> None:
    """The plugin never enables itself. Enforced over the whole settings diff
    rather than over that one key, because the next key with the same power has
    not been named yet."""
    (profile / "claude-config" / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"memkit@memkit": False}, "theme": "dark"}),
        encoding="utf-8",
    )
    plan = _plan(profile, auto_dream_off=True)
    (action,) = [a for a in plan.actions if a.op == init.SETTINGS_WRITE]
    written = json.loads(action.content)
    assert written["enabledPlugins"] == {"memkit@memkit": False}
    assert written["theme"] == "dark"
    assert written["autoDreamEnabled"] is False


def test_an_interpreter_that_cannot_run_is_refused(profile, monkeypatch) -> None:
    """The config init writes records the python that will read every prompt,
    and recording one that cannot run is an install that answers nothing."""
    monkeypatch.setattr(init, "_interpreter", lambda: str(profile / "no-python"))
    refusal = _refuses(profile, "no-interpreter")
    assert "read every prompt" in refusal.message


def test_the_only_settings_key_init_may_write_is_an_allowlist(profile) -> None:
    """The rule is "the plugin never enables itself", and `enabledPlugins` is
    the key that would do it — but the guard is an allowlist, because the next
    key with the same power has not been named yet and a denylist only catches
    the ones somebody thought of."""
    target = str(profile / "claude-config" / "settings.json")
    assert frozenset(
        {"autoDreamEnabled", "autoMemoryDirectory", "autoMemoryEnabled"}
    ) == init.SETTINGS_KEYS_INIT_MAY_WRITE
    with pytest.raises(init.Refusal) as caught:
        init._settings_with(target, {"enabledPlugins": {"memkit@memkit": True}})
    assert caught.value.name == "enabled-plugins"
    assert "deciding its own access" in caught.value.message
    # And a key nobody has thought of yet is refused by the same rule.
    with pytest.raises(init.Refusal) as caught:
        init._settings_with(target, {"someFutureTrustKey": True})
    assert caught.value.name == "enabled-plugins"


def test_the_refusal_reaches_the_caller_named_and_with_a_reason(profile) -> None:
    """The name is the half a caller branches on and the sentence is the half a
    person acts on. An agent given only prose parses it; one given only a token
    relays a token."""
    out = _run(
        "--dry-run", "--store", "notes",
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert out.returncode == init.EXIT_REFUSED
    assert out.stdout == ""
    assert "refused (relative-path)" in out.stderr
    assert "not absolute" in out.stderr


def test_a_session_directory_removed_underfoot_is_an_exit_code_not_a_traceback(
    profile, monkeypatch
) -> None:
    """`machine = Machine()` sat ABOVE the try that was widened to hold
    everything that can refuse.

    The directory a process stands in can be removed under it — an agent's
    session workdir cleaned up by something else, a torn-down worktree — and
    `Machine.__init__` reads it. Both commands promise a closed set of exit
    codes on every path, and neither `bin/memkit` nor `cli.main` adds a
    handler, so the traceback reached the caller verbatim.

    `os.getcwd` is faked rather than the directory really removed: on macOS
    the real syscall takes about 38 seconds to fail, which would make this
    case a timeout rather than a test.
    """
    def gone():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os, "getcwd", gone)
    assert init.run(_args(dry_run=True)) in (init.EXIT_OK, init.EXIT_REFUSED)
    # And doctor answers rather than raising, from the same construction.
    assert doctor.Machine() is not None


def test_every_refusal_in_the_inventory_is_reachable() -> None:
    """A named refusal nothing can produce is a name in a docstring.

    Scraped from the module rather than listed here, so a refusal added
    without a case that reaches it fails this rather than passing quietly.
    """
    import re as _re

    source = pathlib.Path(init.__file__).read_text(encoding="utf-8")
    raised = set(_re.findall(r'Refusal\(\s*"([a-z-]+)"', source))
    covered = set(_re.findall(
        r'_refuses\(\s*profile,\s*"([a-z-]+)"',
        pathlib.Path(__file__).read_text(encoding="utf-8"),
    ))
    # `enabled-plugins` is reached through its own function rather than
    # through `build_plan`.
    covered |= {"enabled-plugins"}
    # Refusals raised at APPLY time, which `_refuses` cannot reach because it
    # calls `build_plan` directly — the exact blind spot that let a refusal
    # ship without a path to `_refuse()`. Each is named with the case that
    # does reach it, so the allowance cannot quietly become a hole.
    mine = pathlib.Path(__file__).read_text(encoding="utf-8")
    apply_time = {
        "stale-digest": "test_a_stale_digest_refuses_and_writes_nothing",
        "changed-underfoot": (
            "test_a_file_that_arrived_after_the_plan_is_not_written_over"
        ),
        # Raised by `run()` AROUND `build_plan`, so it is reached by running
        # the command rather than by building a plan.
        "unreadable-machine": (
            "test_a_session_directory_removed_underfoot_is_an_exit_code_"
            "not_a_traceback"
        ),
        "escapes-store": (
            "test_a_link_planted_after_the_plan_never_lands_outside_the_store"
        ),
        # The write's half of the length rule. Adoption's planner skips such a
        # name before it can be planned, so this is reached by calling the
        # write rather than by building a plan — which is the point of it.
        "name-too-long": (
            "test_the_write_refuses_a_name_it_could_not_create_a_temporary_for"
        ),
    }
    for name, case in apply_time.items():
        assert f"def {case}(" in mine, (name, case)
    assert raised - covered <= set(apply_time), sorted(raised - covered)
    assert len(raised) >= 12, sorted(raised)


# --- the confirm turn, the journal, and convergence --------------------------


def _confirm(profile, digest, *extra):
    return _run(
        "--confirm", digest, *extra,
        env=dict(
            os.environ,
            HOME=str(profile / "home"),
            XDG_CACHE_HOME=str(profile / "home" / ".cache"),
            CLAUDE_CONFIG_DIR=str(profile / "claude-config"),
        ),
    )


def _dry(profile, *extra):
    return _run(
        "--dry-run", *extra,
        env=dict(
            os.environ,
            HOME=str(profile / "home"),
            XDG_CACHE_HOME=str(profile / "home" / ".cache"),
            CLAUDE_CONFIG_DIR=str(profile / "claude-config"),
        ),
    )


def _digest_of(out) -> str:
    for line in out.stdout.splitlines():
        if line.startswith("digest: "):
            return line.split()[1]
    raise AssertionError(out.stdout + out.stderr)


def test_a_stale_digest_refuses_and_writes_nothing(profile) -> None:
    """The digest binds the state of the TREE. A confirm carrying a number
    computed against a different world is a consent given for something else.
    """
    before = _snapshot(profile)
    out = _confirm(profile, "0000000000000000")
    assert out.returncode == init.EXIT_REFUSED
    assert "refused (stale-digest)" in out.stderr
    assert _snapshot(profile) == before


def test_the_confirm_turn_puts_the_applied_text_in_the_transcript(profile) -> None:
    """"Relay this verbatim" is an instruction to a model and not a control, so
    the only way to be sure the human saw what is about to happen is to put it
    where the turn itself records it — beside the writes rather than one turn
    earlier."""
    manifest = _dry(profile)
    out = _confirm(profile, _digest_of(manifest))
    assert out.returncode == init.EXIT_OK, out.stderr
    assert "memkit init — what this would do" in out.stdout
    assert "applying:" in out.stdout
    # The same manifest, not a summary of it.
    for line in manifest.stdout.splitlines():
        if line.strip().startswith(("create-dir", "create-file", "merge-config")):
            assert line in out.stdout, line


def test_the_journal_names_every_file_the_run_made_and_nothing_it_did_not(profile):
    """A record per mutation, at the mutation. Not batched at the end: a crash
    between two mutations has to leave a journal that describes what happened,
    and a batch written at the end describes a run that finished — the one case
    the record is not needed for."""
    out = _confirm(profile, _digest_of(_dry(profile)))
    assert out.returncode == init.EXIT_OK, out.stderr
    journal = profile / "home" / ".cache" / "memory-recall" / hook.INIT_JOURNAL_NAME
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    journalled = {r["path"] for r in records if r["op"] != init.VERIFY}
    on_disk = set()
    for root, dirnames, filenames in os.walk(profile / "home"):
        for name in dirnames + filenames:
            on_disk.add(os.path.join(root, name))
    made = {
        p
        for p in journalled
        if not p.endswith((hook.INIT_JOURNAL_NAME, "init.lock"))
    }
    assert made <= on_disk, sorted(made - on_disk)
    # And nothing it did not: every journalled path is one the plan named.
    planned = {a.path for a in _plan(profile).actions}
    assert made <= planned, sorted(made - planned)
    # The config's record claims authorship, which is what makes an unclaimed
    # rung-2 config detectable at all.
    claims = [r for r in records if r.get("authored_config")]
    # TWO records for the one config, and the first is the point: a claim
    # written before the file lands is what stops a crash in that window from
    # bricking every later init against memkit's own file.
    assert [r["after"] for r in claims] == ["pending", claims[-1]["after"]]
    assert claims[-1]["after"].startswith("file:")
    assert {r["path"] for r in claims} == {claims[0]["path"]}
    assert claims[0]["path"].endswith("memkit.json")


def test_a_file_that_arrived_after_the_plan_is_not_written_over(
    profile, monkeypatch
) -> None:
    """The digest binds the plan to the tree at PLAN time.

    Between the confirm's digest check and the write, another process can
    create a path the manifest described as absent — and the manifest said
    "create", so the adopter consented to a file appearing where there was
    none, not to one of theirs being replaced. An exclusive create is what
    closes the window rather than narrowing it.
    """
    plan = _plan(profile, store=str(profile / "notes"))
    machine = doctor.Machine()
    target = next(
        a.path for a in plan.pending if a.path.endswith("MEMORY.md")
    )
    assert not os.path.exists(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    theirs = "# their own notes, written between the two turns\n"
    with open(target, "w", encoding="utf-8") as f:
        f.write(theirs)
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_INCOMPLETE
    with open(target, encoding="utf-8") as f:
        assert f.read() == theirs, "confirm wrote over a file it planned to create"


def test_losing_the_exclusive_create_leaves_nothing_beside_the_target(
    profile, monkeypatch
) -> None:
    """The refusal is right; what it left behind was not.

    The pre-check catches the file that arrived before the write. The one that
    arrives between that check and `os.link` is caught by the exclusive create
    itself — and THAT path had already written the temp file. `Refusal` is a
    plain `Exception`, so the `except OSError` cleanup never ran: a
    `<target>.<pid>.tmp` stayed beside `MEMORY.md` in the memory store,
    holding the content the refusal had just declined to write. Nothing
    collects it — the sweep only reaches the state directory and the stray
    scan only counts `.md` — so it survived every re-run of the two-turn
    recovery the refusal itself advertises.
    """
    target = profile / "notes" / "MEMORY.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    real_link = os.link

    def lost(src, dst, **kw):
        # The race, deterministically: the name is taken at the moment of the
        # link and not a moment before it.
        raise FileExistsError(errno.EEXIST, "File exists", dst)

    monkeypatch.setattr(init.os, "link", lost)
    with pytest.raises(init.Refusal) as caught:
        init._write_atomically(str(target), "# theirs\n", expect="absent")
    monkeypatch.setattr(init.os, "link", real_link)
    assert caught.value.name == "changed-underfoot"
    strays = [p.name for p in target.parent.iterdir() if ".tmp" in p.name]
    assert strays == [], strays


def test_a_torn_record_cannot_swallow_the_next_one(profile) -> None:
    """The readers tolerate a torn LAST line; what they could not tolerate was
    the record after it.

    A crash mid-`write()` leaves a fragment with no trailing newline, and the
    next record appended — by any later run, including one that finishes
    cleanly — became part of the same LINE. `for line in f` then saw one
    unparseable line and dropped BOTH: the torn record, which is intended,
    and a complete one that had nothing wrong with it. The two-phase config
    write survived it by accident of ordering; a single-record write does not.
    """
    state = profile / "home" / ".cache" / "memory-recall"
    state.mkdir(parents=True, exist_ok=True)
    journal = state / hook.INIT_JOURNAL_NAME
    good = json.dumps(
        {
            "v": 1,
            "op": "merge-config",
            "authored_config": True,
            "path": str(profile / "memkit.json"),
            "before": None,
            "after": "file:abc",
        },
        separators=(",", ":"),
    )
    journal.write_text('{"v":1,"op":"merge-config","authored_conf', encoding="utf-8")
    hook.append_record(str(journal), good, fsync=True)
    assert list(hook.journal_config_claims(str(state))) == [
        str(profile / "memkit.json")
    ], journal.read_text()

    # And a well-formed file gains no blank lines, so every reader that counts
    # LINES — the soak log's window is one — keeps counting the same things.
    log = state / hook.SOAK_LOG_NAME
    hook.append_record(str(log), good)
    hook.append_record(str(log), good)
    assert len(log.read_text().splitlines()) == 2, log.read_text()


def test_a_crash_between_two_mutations_leaves_a_journal_that_describes_it(
    profile, monkeypatch
) -> None:
    """The whole reason the record is written at the mutation. A batch would
    describe the runs that did not need describing and nothing else."""
    plan = _plan(profile)
    machine = doctor.Machine()
    real = init._write_atomically
    calls = []

    def explode(path, content, mode=0o600, expect=None, confine=""):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real(path, content, mode, expect, confine)

    monkeypatch.setattr(init, "_write_atomically", explode)
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_INCOMPLETE
    journal = profile / "home" / ".cache" / "memory-recall" / hook.INIT_JOURNAL_NAME
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    # Everything before the failure is on the record, and the failure is not.
    assert records
    assert calls[1] not in {r["path"] for r in records}


def test_a_partial_run_converges_when_it_is_run_again(profile, monkeypatch) -> None:
    """Every action already done is a no-op the second time, so re-running is
    the safe instruction the incomplete exit code gives."""
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    real = init._write_atomically
    calls = []

    def explode(path, content, mode=0o600, expect=None, confine=""):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real(path, content, mode, expect, confine)

    monkeypatch.setattr(init, "_write_atomically", explode)
    assert init.apply_plan(machine, _plan(profile), config) == init.EXIT_INCOMPLETE

    monkeypatch.setattr(init, "_write_atomically", real)
    assert init.apply_plan(machine, _plan(profile), config) == init.EXIT_OK
    assert _plan(profile).writes == []


def test_two_inits_appending_different_stores_both_survive(profile) -> None:
    """`os.replace` makes the file untearable and does nothing about a LOST
    APPEND: two inits that both read the config, both add their own store and
    both write leave one store."""
    first = _confirm(
        profile, _digest_of(_dry(profile, "--store", str(profile / "a"))),
        "--store", str(profile / "a"),
    )
    assert first.returncode == init.EXIT_OK, first.stderr
    second = _confirm(
        profile, _digest_of(_dry(profile, "--store", str(profile / "b"))),
        "--store", str(profile / "b"),
    )
    assert second.returncode == init.EXIT_OK, second.stderr
    blob = json.loads(
        (profile / "home" / ".config" / "memkit" / "memkit.json").read_text()
    )
    assert {s["id"] for s in blob["stores"]} == {"a", "b"}
    assert set(blob["roots"]) == {"a", "b"}
    # One nonce for the whole config, so doctor's fixed query answers for both
    # stores rather than for whichever one ran last.
    assert blob["canary_nonce"] == init._canary_nonce(
        str(profile / "home" / ".config" / "memkit" / "memkit.json")
    )


def test_the_config_write_re_reads_under_the_lock(profile, monkeypatch) -> None:
    """The interleave the lock is for: another init committed between this
    one's plan and its write. Writing the plan-time content would silently
    drop that store."""
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    plan = _plan(profile, store=str(profile / "mine"))
    os.makedirs(os.path.dirname(config), exist_ok=True)
    # A peer's config, committed after our plan was built.
    peer = init._merge_config(
        "",
        nonce=init._canary_nonce(config),
        interpreter=sys.executable,
        entries=init._config_entries(store=str(profile / "theirs"), store_id="theirs"),
    )
    with open(config, "w", encoding="utf-8") as f:
        f.write(peer)
    (action,) = [a for a in plan.actions if a.op == init.MERGE_CONFIG]
    assert "theirs" not in action.content, "the fixture is not exercising the race"

    journal = init.Journal(str(machine.state_dir), plan.digest)
    os.makedirs(machine.state_dir, mode=0o700, exist_ok=True)
    # A PEER INIT, which means a journal record — the two processes append to
    # one journal, and a claim is how this one can tell "another init wrote it"
    # from "something else did". Without the record the fixture models an
    # unaccountable writer, which is the case the refusal below is for.
    peer_action = init.Action(
        init.MERGE_CONFIG, config, peer, authored_config=True
    )
    init.Journal(str(machine.state_dir), "peer-digest").record(
        peer_action, hook.state_token(config)
    )
    init._perform(machine, journal, action, config)
    with open(config, encoding="utf-8") as f:
        blob = json.loads(f.read())
    assert {s["id"] for s in blob["stores"]} == {"theirs", "mine"}


def test_a_config_no_init_journal_claims_is_not_merged_forward(
    profile, monkeypatch
) -> None:
    """The digest an adopter approved describes the config as it was at the
    dry-run. Merging whatever is there at apply time publishes content they
    were never shown, under their consent.

    A concurrent init is the case the re-read exists for and it stays working —
    its write is claimed in the journal both processes append to. Anything
    else is a writer memkit cannot account for, and it is refused rather than
    carried forward.
    """
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    plan = _plan(profile, store=str(profile / "mine"))
    os.makedirs(os.path.dirname(config), exist_ok=True)
    os.makedirs(machine.state_dir, mode=0o700, exist_ok=True)
    (action,) = [a for a in plan.actions if a.op == init.MERGE_CONFIG]
    assert action.before == "absent", action.before
    with open(config, "w", encoding="utf-8") as f:
        f.write(json.dumps({"schema": hook.SCHEMA, "roots": {}, "stores": []}))
    journal = init.Journal(str(machine.state_dir), plan.digest)
    with pytest.raises(init.Refusal) as caught:
        init._perform(machine, journal, action, config)
    assert caught.value.name == "changed-underfoot"
    assert "no init journal claims" in caught.value.message


def test_a_second_init_does_not_renumber_the_first_ones_nonce(profile) -> None:
    """Changing it would make every canary already on disk stop answering the
    fixed query, which is the one thing the canary exists to do."""
    config = str(profile / "home" / ".config" / "memkit" / "memkit.json")
    os.makedirs(os.path.dirname(config), exist_ok=True)
    with open(config, "w", encoding="utf-8") as f:
        f.write(
            init._merge_config(
                "", nonce="mkcORIGINAL", interpreter=sys.executable,
                entries=init._config_entries(store=str(profile / "a"), store_id="a"),
            )
        )
    with open(config, encoding="utf-8") as f:
        current = f.read()
    merged = init._merge_config(
        current,
        nonce="mkcSECOND",
        interpreter="/other/python",
        entries=init._config_entries(store=str(profile / "b"), store_id="b"),
    )
    blob = json.loads(merged)
    assert blob["canary_nonce"] == "mkcORIGINAL"
    assert blob["interpreter"] == sys.executable


def test_the_seeded_store_passes_the_checker_and_answers_doctors_query(profile):
    """§5.7's verification, end to end without the harness: a cold init
    produces a store doctor rates with zero FAIL checks and a canary that comes
    back for the fixed query."""
    out = _confirm(profile, _digest_of(_dry(profile)))
    assert out.returncode == init.EXIT_OK, out.stderr
    config = str(profile / "home" / ".config" / "memkit" / "memkit.json")

    # The checker, clean — including the two citation warnings a config with an
    # empty `citations` block would produce about a feature nobody opted into.
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", config],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "CITED-PATHS" not in checked.stdout

    hook._use_config(config)
    machine = doctor.Machine(config)
    checks = doctor.collect(machine, ["canary-retrieval", "corpus-root", "config-parse"])
    assert [c.status for c in checks if c.id == "canary-retrieval"] == [doctor.PASS], [
        c.detail for c in checks
    ]
    assert doctor.verdict(checks) == "OK", [c.detail for c in checks if c.status == "FAIL"]


def test_a_second_confirm_is_a_no_op_that_still_verifies(profile) -> None:
    """Double init converges. The manifest says there is nothing to write and
    the check still runs, because a second init has nothing to do and still has
    something to check."""
    assert _confirm(profile, _digest_of(_dry(profile))).returncode == init.EXIT_OK
    again = _dry(profile)
    assert "already set up" in again.stdout
    applied = _confirm(profile, _digest_of(again))
    assert applied.returncode == init.EXIT_OK, applied.stderr


def test_a_store_that_fails_its_own_check_is_incomplete_and_not_refused(profile):
    """The store is on disk by then. A caller told "refused" would believe
    nothing was written and go looking for a store that is right there — and
    the move that fixes it is to repair the store and re-run, which is what
    the incomplete code means."""
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, _plan(profile), config) == init.EXIT_OK
    # A memory at the store root, which is the layout the checker refuses.
    (profile / "home" / "notes" / "stray.md").write_text(
        "---\nname: stray\ndescription: d\ntype: reference\n---\n\nbody\n",
        encoding="utf-8",
    )
    assert init.apply_plan(machine, _plan(profile), config) == init.EXIT_INCOMPLETE
    assert (profile / "home" / "notes" / "search" / doctor.CANARY_NAME).is_file()


def test_a_failed_write_never_destroys_what_was_already_there(profile, monkeypatch):
    """`open(path, "w")` destroys the old file before writing the new one, so
    anything that stops the write in between leaves a valid prefix of an
    invalid file — and for a config, a valid prefix is a config that names half
    a store."""
    target = profile / "config.json"
    target.write_text('{"schema": 1, "stores": []}', encoding="utf-8")
    original = target.read_text()

    def refuse(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        init._write_atomically(str(target), "{}" * 100)
    assert target.read_text() == original
    # And no scratch file left behind for the next reader to find.
    assert [p.name for p in profile.glob("config.json*")] == ["config.json"]


# --- the two consented writes ------------------------------------------------


def test_the_settings_write_appears_only_with_its_own_flag(profile) -> None:
    """The ONLY settings key init may write, and only when asked. A setup
    command that turned a harness feature off because it seemed tidy would be
    making a decision about somebody else's tool."""
    assert not [a for a in _plan(profile).actions if a.op == init.SETTINGS_WRITE]
    with_flag = _plan(profile, auto_dream_off=True)
    (action,) = [a for a in with_flag.actions if a.op == init.SETTINGS_WRITE]
    assert json.loads(action.content) == {"autoDreamEnabled": False}
    assert any("auto-dream off" in note for note in with_flag.notes)


def test_the_claude_md_import_appears_only_with_its_own_flag(profile) -> None:
    """It writes to a file adopters treat as theirs, on a path where the
    consent that was given was about a memory store."""
    assert not [a for a in _plan(profile).actions if a.op == init.APPEND_LINE]
    plan = _plan(profile, wire_claude_md=True, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.APPEND_LINE]
    assert action.content.strip() == f"@{profile / 'notes' / 'MEMORY.md'}"


def test_the_import_offer_states_the_honest_version_of_what_it_buys(profile):
    """An @-import of MEMORY.md puts each hot memory's DESCRIPTION in every
    session — one line per memory — and not its body. The bodies stay files to
    open. That is narrower than the docs have implied, and it is the reason
    this is behind a flag rather than on by default."""
    plan = _plan(profile, wire_claude_md=True)
    notes = " ".join(plan.notes)
    assert "description" in notes and "not its body" in notes
    assert "files to open" in notes


def test_the_import_converges_rather_than_duplicating(profile) -> None:
    """Re-running must not append the same line twice: a CLAUDE.md that grew
    one import per init is a file the adopter has to clean up by hand."""
    target = profile / "claude-config" / "CLAUDE.md"
    target.write_text("# mine\n", encoding="utf-8")
    plan = _plan(profile, wire_claude_md=True, store=str(profile / "notes"))
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_OK
    body = target.read_text()
    assert body.count("@") == 1, body
    assert body.startswith("# mine")

    again = _plan(profile, wire_claude_md=True, store=str(profile / "notes"))
    assert not [a for a in again.writes if a.op == init.APPEND_LINE]


def test_a_git_tracked_claude_md_is_warned_about_rather_than_refused(profile):
    """An adopter may well keep their CLAUDE.md in a dotfiles repo on purpose.
    What they may not do is find a line in it they will be asked to commit and
    have nobody mention it."""
    repo = profile / "claude-config"
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=60)
    target = repo / "CLAUDE.md"
    target.write_text("# mine\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "CLAUDE.md"],
        cwd=repo, check=True, timeout=60,
    )
    plan = _plan(profile, wire_claude_md=True)
    notes = " ".join(plan.notes)
    assert "tracked by git" in notes
    assert "commit" in notes
    # A warning, not a refusal: the plan still carries the write.
    assert [a for a in plan.actions if a.op == init.APPEND_LINE]


def test_the_generated_config_advertises_a_command_the_agent_can_run(profile):
    """On the plugin channel the config PATH is part of the command, and that
    is what makes it runnable rather than merely spelled correctly: a Bash-tool
    process gets the plugin's bin on PATH and none of the plugin environment,
    so a bare `memkit-recall --search` there answers inert."""
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, _plan(profile), config) == init.EXIT_OK
    cfg = hook.load_config(config)
    assert cfg is not None
    assert hook._advertised_search_cli(cfg) == hook.DEFAULT_SEARCH_CLI

    os.environ[hook.PLUGIN_ENV] = "1"
    try:
        advertised = hook._advertised_search_cli(cfg)
    finally:
        os.environ.pop(hook.PLUGIN_ENV, None)
    assert advertised.startswith(f"{hook.PLUGIN_SEARCH_BINARY} --config ")
    assert advertised.endswith("--search")
    assert config in advertised


# --- the config has to land somewhere the hook will read ---------------------


def test_a_plugin_install_with_no_option_writes_where_the_wrapper_looks(
    profile, monkeypatch
) -> None:
    """The flagship cold path, and it ended configured-but-inert.

    `required: false` lets an install skip `--config`, and the harness then
    writes no `pluginConfigs` entry at all — measured. The wrapper reads
    exactly two rungs, and `~/.config/memkit/memkit.json` is neither, so init
    wrote a config, seeded a store, passed its own integrity check, exited 0,
    and the hook could never read any of it. Doctor then said to run init,
    which converges to "nothing to write" on every retry: a closed loop between
    the two commands this milestone adds.

    Rung 2 is where it goes. `bin/lib/common.sh` already names init as the one
    thing that will ever legitimately write that path, and the journal entry
    init makes is what `config-authorship` reads to tell memkit's own file from
    a planted one.
    """
    data = profile / "plugin-data"
    data.mkdir()
    monkeypatch.setenv(hook.PLUGIN_ENV, "1")
    monkeypatch.setenv(hook.PLUGIN_DATA_ENV, str(data))
    plan = _plan(profile, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.MERGE_CONFIG]
    assert action.path == str(data / "memkit.json"), action.path
    assert action.authored_config is True
    # And the manifest says which route will read it, because "a config was
    # written" and "the hook can read it" were the two facts this conflated.
    assert hook.PLUGIN_DATA_ENV in plan.render()


def test_the_option_still_wins_over_the_plugin_data_rung(profile, monkeypatch):
    """An adopter who passed `--config memkitConfig=<path>` has said where they
    want it, and that is the rung the wrapper tries first."""
    data = profile / "plugin-data"
    data.mkdir()
    named = profile / "chosen" / "memkit.json"
    monkeypatch.setenv(hook.PLUGIN_ENV, "1")
    monkeypatch.setenv(hook.PLUGIN_DATA_ENV, str(data))
    (profile / "claude-config" / "settings.json").write_text(
        json.dumps(
            {"pluginConfigs": {"memkit@memkit": {"options": {"memkitConfig": str(named)}}}}
        ),
        encoding="utf-8",
    )
    plan = _plan(profile, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.MERGE_CONFIG]
    assert action.path == str(named)


def test_a_plugin_install_with_no_route_at_all_is_refused_by_name(
    profile, monkeypatch
) -> None:
    """Writing a config nothing can read is worse than refusing: the adopter
    gets a store, a green integrity check and an exit 0, and a hook that says
    nothing on every prompt forever."""
    monkeypatch.setenv(hook.PLUGIN_ENV, "1")
    monkeypatch.delenv(hook.PLUGIN_DATA_ENV, raising=False)
    refusal = _refuses(profile, "no-config-route", store=str(profile / "notes"))
    assert "plugin configure" in refusal.message or "--config" in refusal.message


def test_off_the_plugin_channel_the_default_path_is_still_right(profile):
    """`$MEMKIT_CONFIG` and `--config` are the routes pip and nix read, and
    both take a path the adopter names — so a default under `~/.config` is a
    file they can point either route at."""
    plan = _plan(profile, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.MERGE_CONFIG]
    assert action.path == str(profile / "home" / ".config" / "memkit" / "memkit.json")


def test_a_config_inside_the_swept_state_directory_is_refused(profile) -> None:
    """The other half of the sweep hazard: init must not create the thing the
    every-prompt hook garbage-collects.

    The sweep now keeps a config its journal claims and collects no `.json`
    whose name it does not recognise, so this is belt and braces — but a setup
    command that put a config into a directory it also sweeps would be one
    ordinary refactor away from eating it, and the refusal costs an adopter
    nothing they cannot get by naming another directory.
    """
    inside = profile / "home" / ".cache" / "memory-recall" / "mine.json"
    refusal = _refuses(profile, "config-in-state-dir", config=str(inside))
    assert "derived state" in refusal.message
    assert "swept" in refusal.message or "collect" in refusal.message


# --- what init may do to a file it did not write -----------------------------


def test_a_config_this_process_cannot_read_is_refused_not_replaced(profile):
    """The field anti-pattern init's own settings writer names, on the file
    that decides which directories an every-prompt hook reads.

    A config that exists, cannot be READ and can be written was merged into
    `{}` and renamed over — every root and store the adopter had accumulated
    gone, with the manifest line one screen earlier saying "Existing stores are
    kept" and no recoverable copy.
    """
    config = profile / "locked.json"
    config.write_text('{"schema": 1, "stores": [{"id": "theirs"}]}', encoding="utf-8")
    _claim(profile, config)
    # Write-only: readable-and-unwritable is caught earlier and better by
    # `not-writable`. The dangerous shape is the one init can act on and
    # cannot see.
    config.chmod(0o200)
    try:
        refusal = _refuses(profile, "unreadable-config", config=str(config))
    finally:
        config.chmod(0o600)
    assert "could not be read" in refusal.message


def test_a_config_that_does_not_parse_is_refused_not_a_traceback(profile):
    """Exit 1 is spoken for as "memkit could not start at all", so a skill
    branching on the code learned the machine cannot run memkit when in fact
    one file has a comma in the wrong place."""
    config = profile / "typo.json"
    config.write_text('{"schema": 1,,}', encoding="utf-8")
    _claim(profile, config)
    refusal = _refuses(profile, "unparseable-config", config=str(config))
    assert str(config) in refusal.message or "typo.json" in refusal.message


def test_a_config_whose_top_level_is_not_an_object_is_refused(profile):
    config = profile / "list.json"
    config.write_text("[1, 2, 3]", encoding="utf-8")
    _claim(profile, config)
    _refuses(profile, "unparseable-config", config=str(config))


def test_a_write_follows_a_symlink_rather_than_replacing_it(profile) -> None:
    """An adopter whose `~/.claude/settings.json` is a symlink into a dotfiles
    or nix repo — the common setup, and the one the manifest's `resolves to`
    line advertises it understands — silently lost the link, leaving an
    untracked regular file and the repo copy orphaned."""
    real = profile / "dotfiles" / "settings.json"
    real.parent.mkdir(parents=True)
    real.write_text('{"theme": "dark"}', encoding="utf-8")
    link = profile / "claude-config" / "settings.json"
    link.symlink_to(real)

    machine = doctor.Machine()
    plan = _plan(profile, auto_dream_off=True, store=str(profile / "notes"))
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_OK
    assert link.is_symlink(), "init replaced the symlink with a regular file"
    assert json.loads(real.read_text())["autoDreamEnabled"] is False
    assert json.loads(real.read_text())["theme"] == "dark"


def test_an_existing_files_permissions_survive_the_write(profile) -> None:
    """`~/.claude/settings.json` commonly carries an `env` block with an API
    key. A command whose stated scope is 'sets autoDreamEnabled and changes
    nothing else' handed a deliberately 0600 file back at 0644."""
    settings = profile / "claude-config" / "settings.json"
    settings.write_text('{"theme": "dark"}', encoding="utf-8")
    settings.chmod(0o600)
    machine = doctor.Machine()
    plan = _plan(profile, auto_dream_off=True, store=str(profile / "notes"))
    assert init.apply_plan(machine, plan, init._resolve_config(machine, None)) == 0
    assert stat.S_IMODE(settings.stat().st_mode) == 0o600, oct(
        settings.stat().st_mode
    )


def test_a_file_init_creates_is_never_world_readable_mid_write(profile):
    """The mode is set on the temporary file before any byte is written, so
    the content never exists at whatever the umask would have given it."""
    machine = doctor.Machine()
    plan = _plan(profile, store=str(profile / "notes"))
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_OK
    assert stat.S_IMODE(os.stat(config).st_mode) == 0o600, oct(
        os.stat(config).st_mode
    )


def test_a_regular_file_at_the_store_path_is_refused_before_any_write(profile):
    """A writable regular file passed the preflight, so init created the state
    directory and the config and then died on CREATE_DIR — a broken partial
    configuration where the contract promises a write-nothing refusal."""
    store = profile / "notes"
    store.write_text("i am a file\n", encoding="utf-8")
    refusal = _refuses(profile, "not-a-directory", store=str(store))
    assert "notes" in refusal.message
    # The STORE ROOT's own sentence, not the generic preflight's. The two
    # refusals share a name and the plan-wide preflight now catches this case
    # as well, so an assertion on the name alone stopped being able to tell
    # which of them answered.
    assert "A store is a directory of markdown" in refusal.message


def test_two_stores_with_the_same_basename_are_refused(profile) -> None:
    """`/one/notes` and `/two/notes` produced the same store id, and the merge
    kept the first: the second store's files and canary exist on disk and the
    configured reader never looks at them."""
    machine = doctor.Machine()
    first = _plan(profile, store=str(profile / "one" / "notes"))
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, first, config) == init.EXIT_OK
    refusal = _refuses(profile, "store-id-taken", store=str(profile / "two" / "notes"))
    assert "notes" in refusal.message


def test_a_crash_before_the_journal_record_does_not_brick_init(profile, monkeypatch):
    """Between the config landing and its journal record being fsynced, every
    future init — dry-run included — refused `foreign-config` and told the
    adopter memkit did not write the file memkit had just written. There was no
    store, no documented recovery, and the only manual fix was deleting a
    config the refusal exists to protect."""
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    plan = _plan(profile, store=str(profile / "notes"))
    real = init.Journal.record
    calls = []

    def die(self, action, after, locked=None, expects=None):
        calls.append(action.op)
        if action.op == init.MERGE_CONFIG and after != "pending":
            raise OSError("no space left on device")
        return real(self, action, after, locked, expects)

    monkeypatch.setattr(init.Journal, "record", die)
    assert init.apply_plan(machine, plan, config) == init.EXIT_INCOMPLETE
    assert os.path.isfile(config), "the config did not land, so this is not the case"
    monkeypatch.setattr(init.Journal, "record", real)
    # The next run converges instead of refusing about memkit's own file.
    again = _plan(profile, store=str(profile / "notes"))
    assert any(a.op == init.MERGE_CONFIG for a in again.actions)


def test_an_unreadable_claude_md_is_refused_rather_than_truncated(profile):
    """The manifest calls the operation `append-line` and its note says
    'appends @...', so the consent given was for an append. Substituting an
    empty string for a read failure made the effect a truncation."""
    target = profile / "claude-config" / "CLAUDE.md"
    target.write_text("# my instructions\n" * 40, encoding="utf-8")
    # Write-only: unwritable is caught earlier and better by `not-writable`.
    # The dangerous shape is the one init can act on and cannot see.
    target.chmod(0o200)
    try:
        refusal = _refuses(
            profile, "unreadable-claude-md", wire_claude_md=True,
            store=str(profile / "notes"),
        )
    finally:
        target.chmod(0o600)
    assert "could not be read" in refusal.message


def test_a_settings_file_that_moved_under_the_manifest_is_not_written(
    profile,
) -> None:
    """THE SIBLING'S QUESTION, ASKED HERE TOO. init is invoked from inside a
    live session, so the harness owns that file for the whole run — and the
    settings write is the LAST action, after an integrity-checker subprocess
    that may take minutes. Setting one key into whatever is there by then
    re-writes the rest of the file as well, under a digest taken against the
    file the adopter actually read: content nobody saw, published as theirs.
    The config merge beside it has refused that shape since it was written.
    """
    machine = doctor.Machine()
    settings = profile / "claude-config" / "settings.json"
    settings.write_text('{"theme": "dark"}', encoding="utf-8")
    plan = _plan(profile, auto_dream_off=True, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.SETTINGS_WRITE]
    # Something writes while the plan is in flight.
    moved = json.dumps({"theme": "dark", "enabledPlugins": {"other@x": True}})
    settings.write_text(moved, encoding="utf-8")
    journal = init.Journal(str(machine.state_dir), plan.digest)
    os.makedirs(machine.state_dir, mode=0o700, exist_ok=True)
    with pytest.raises(init.Refusal) as refusal:
        init._perform(
            machine, journal, action, init._resolve_config(machine, None)
        )
    assert refusal.value.name == "changed-underfoot", refusal.value.name
    assert settings.read_text(encoding="utf-8") == moved, "it wrote anyway"


def test_the_settings_write_lands_when_nothing_moved_under_it(profile) -> None:
    """The other half, and the case a run with two flags is: one settings file,
    two writes, and the second must be able to tell its predecessor's landing
    from somebody else's edit — or a plain `--auto-dream-off
    --adopt-auto-memory` refuses itself.
    """
    machine = doctor.Machine()
    settings = profile / "claude-config" / "settings.json"
    settings.write_text('{"theme": "dark"}', encoding="utf-8")
    _harness(profile, "-home-u", {"note.md": TRAP})
    plan = _plan(
        profile,
        auto_dream_off=True,
        adopt_auto_memory=True,
        store=str(profile / "notes"),
    )
    writes = [a for a in plan.pending if a.op == init.SETTINGS_WRITE]
    assert len({a.path for a in writes}) == 1 and len(writes) == 2, writes
    code = init.apply_plan(machine, plan, init._resolve_config(machine, None))
    assert code == init.EXIT_OK, code
    blob = json.loads(settings.read_text())
    assert blob["theme"] == "dark", blob
    assert blob["autoDreamEnabled"] is False, blob
    assert blob[harness_memory.DIRECTORY_KEY], blob


def test_the_claude_md_append_re_reads_under_the_lock(profile) -> None:
    """Same window, same file class: an append computed at plan time and
    written after a 300-second subprocess is an append against a file that may
    have moved."""
    machine = doctor.Machine()
    target = profile / "claude-config" / "CLAUDE.md"
    target.write_text("# mine\n", encoding="utf-8")
    plan = _plan(profile, wire_claude_md=True, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.APPEND_LINE]
    target.write_text("# mine\nsomething they added meanwhile\n", encoding="utf-8")
    journal = init.Journal(str(machine.state_dir), plan.digest)
    os.makedirs(machine.state_dir, mode=0o700, exist_ok=True)
    init._perform(machine, journal, action, init._resolve_config(machine, None))
    body = target.read_text()
    assert "something they added meanwhile" in body, body
    assert body.rstrip().endswith(init._import_line(str(profile / "notes")))


def test_a_refusal_raised_after_a_write_is_never_reported_as_refused(
    profile, monkeypatch
) -> None:
    """Exit 5 promises "nothing was written", and the skill's table tells the
    agent so.

    `run()` wrapped the whole apply in `except Refusal`, so a refusal raised
    below the first write returned 5 after files had landed — and this stopped
    being hypothetical the moment the settings write started re-deriving under
    the lock, since `_settings_with` refuses an unparseable file at apply time.
    An agent reading 5 goes looking for a machine nothing touched.
    """
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    plan = _plan(profile, auto_dream_off=True, store=str(profile / "notes"))
    # The harness writes something unparseable between plan and apply, which is
    # the window the re-derivation exists for.
    settings = profile / "claude-config" / "settings.json"

    real = init._run_checker

    def corrupt(machine_, config_):
        settings.write_text("{ not json", encoding="utf-8")
        return real(machine_, config_)

    monkeypatch.setattr(init, "_run_checker", corrupt)
    code = init.apply_plan(machine, plan, config)
    assert code == init.EXIT_INCOMPLETE, code
    assert os.path.isfile(config), "nothing landed, so this is not the case"


def test_a_refusal_before_the_first_write_is_still_a_refusal(profile) -> None:
    """The other side, or the change above would have turned every refusal into
    an incomplete run."""
    out = _run(
        "--dry-run", "--store", "notes",
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert out.returncode == init.EXIT_REFUSED, out.stderr


def test_a_store_whose_canary_belongs_to_another_config_is_refused(profile):
    """The nonce is keyed on the CONFIG so one fixed query answers for every
    store that config names. The cost is that two configs over one store
    disagree about it — and rewriting the canary would silently take the first
    config's `canary-retrieval` check away, which is the check that exists to
    say whether that store answers at all."""
    machine = doctor.Machine()
    store = profile / "shared"
    first = _plan(profile, store=str(store))
    assert init.apply_plan(machine, first, init._resolve_config(machine, None)) == 0
    refusal = _refuses(
        profile, "canary-belongs-to-another-config",
        store=str(store), config=str(profile / "second.json"),
    )
    assert "mkc" in refusal.message


def test_the_lock_gives_up_rather_than_waiting_forever(profile, monkeypatch):
    """A plain `LOCK_EX` has no timeout, so a live process holding the file
    hung `init --confirm` with no output — indistinguishable to a waiting
    caller from a slow checker run. Proceeding unlocked is what this lock
    already does when `flock` is unavailable, so the bound adds no new failure
    mode; it removes the one that never ends."""
    machine = doctor.Machine()
    os.makedirs(machine.state_dir, mode=0o700, exist_ok=True)
    monkeypatch.setattr(init, "LOCK_WAIT_SECONDS", 0.2)
    held = init._Lock(str(machine.state_dir))
    held.__enter__()

    # A HARD BOUND of its own. The failure this catches is a hang, and a test
    # that waited for one would hang with it — turning a red into a wedged
    # suite, which is the shape of failure nobody can act on.
    def _fire(signum, frame):
        raise TimeoutError("the lock did not give up")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        started = time.monotonic()
        with init._Lock(str(machine.state_dir)):
            pass
        waited = time.monotonic() - started
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        held.__exit__()
    assert waited < 5, waited
    assert waited >= 0.2, waited


def test_an_unserialised_write_says_so_in_the_journal(profile):
    """A caller cannot tell a lost append from a write that never raced.

    The lock is best-effort by design: a filesystem with no working `flock`,
    or one still held when the bounded wait runs out, proceeds anyway. That is
    the right call for a setup command, and it is also the one case where a
    store can go missing from a config two inits wrote — so the record that
    survives has to say which kind of write it was.
    """

    def contended(fd, flags):
        raise OSError(errno.EWOULDBLOCK, "locked")

    machine = doctor.Machine()
    # The real `_Lock.__enter__`, against a lock it can never take. Bounding
    # the wait at zero is what keeps this from being a ten-second test.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(init, "LOCK_WAIT_SECONDS", 0.0)
        mp.setattr(fcntl, "flock", contended)
        assert (
            init.apply_plan(
                machine,
                _plan(profile, store=str(profile / "notes")),
                init._resolve_config(machine, None),
            )
            == init.EXIT_OK
        )
    merges = [r for r in _journal(machine) if r["op"] == init.MERGE_CONFIG]
    assert merges, "no config write to describe"
    assert all(r.get("unlocked") is True for r in merges), merges

    # And the ordinary path, through the same code with a working `flock`: the
    # key is ABSENT, so a reader that never learnt it keeps reading every
    # record it could read before.
    second = doctor.Machine()
    assert (
        init.apply_plan(
            second,
            _plan(profile, store=str(profile / "other")),
            init._resolve_config(second, None),
        )
        == init.EXIT_OK
    )
    fresh = [r for r in _journal(second) if r["run"] != merges[0]["run"]]
    assert fresh, "the second run wrote nothing"
    assert all("unlocked" not in r for r in fresh), fresh


def _journal(machine) -> list:
    path = pathlib.Path(machine.state_dir) / init.INIT_JOURNAL_NAME
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- residual coverage the round-2 review left standing ----------------------


def test_a_write_keeps_the_mode_and_the_link_of_a_file_that_is_already_there(
    profile,
) -> None:
    """Two properties of the general write path that only the refusal cases
    covered.

    A settings file somebody deliberately chmod'd 600 — they commonly carry an
    API key — must not come back 644 from a command whose stated scope is one
    key; and an adopter whose dotfile is a symlink into a nix or dotfiles repo
    must get the write through the link, or the repo copy is orphaned and the
    next `home-manager switch` reaches nothing.
    """
    target = profile / "settings.json"
    target.write_text("{}\n", encoding="utf-8")
    os.chmod(target, 0o600)
    init._write_atomically(str(target), '{"a": 1}\n', mode=0o644)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert target.read_text() == '{"a": 1}\n'

    real = profile / "dotfiles" / "CLAUDE.md"
    real.parent.mkdir(parents=True)
    real.write_text("original\n", encoding="utf-8")
    link = profile / "home" / "CLAUDE.md"
    link.symlink_to(real)
    init._write_atomically(str(link), "through the link\n", mode=0o644)
    assert link.is_symlink(), "the link was replaced by a regular file"
    assert real.read_text() == "through the link\n"

    # And a file being CREATED gets the mode it was asked for.
    fresh = profile / "fresh.json"
    init._write_atomically(str(fresh), "{}\n", mode=0o644)
    assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o644


def test_a_write_judges_and_writes_one_resolution_of_the_name_it_was_given(
    profile, monkeypatch
) -> None:
    """ONE NAME, ONE ANSWER.

    The containment guard resolved the name to decide whether the write lands
    where its manifest line says, and the write then resolved the same name
    again to decide where to put the bytes. Two questions, two answers, and
    only the first one was looked at: a link swapped between them — the window
    is real, the confirm turn re-plans and then writes — sends the bytes to a
    path nothing judged, under a guard that has already said yes.

    `os.path.realpath` is made to answer differently the second time it is
    asked about this name, which is what a swap looks like from inside the
    process. The assertion is the call count as much as the landing place: a
    fix that resolved twice and happened to agree would pass the second and
    not the first.
    """
    store = profile / "store"
    (store / "search").mkdir(parents=True)
    target = store / "search" / "note.md"
    elsewhere = profile / "elsewhere.md"

    real = os.path.realpath
    answers = []

    def counting(path):
        resolved = real(path)
        if os.fspath(path) == str(target):
            answers.append(resolved)
            if len(answers) > 1:
                return str(elsewhere)
        return resolved

    monkeypatch.setattr(os.path, "realpath", counting)
    init._write_atomically(str(target), "hello\n", confine=str(store))

    assert len(answers) == 1, f"the name was resolved {len(answers)} times"
    assert target.read_text() == "hello\n"
    assert not elsewhere.exists(), "the bytes went to the unjudged answer"


def test_both_consented_writes_leave_a_file_a_person_can_still_read(
    profile, monkeypatch
) -> None:
    """The two writes that land outside memkit's own paths, checked for the
    property nothing else checks: the mode they leave behind."""
    settings = profile / "claude-config" / "settings.json"
    settings.write_text("{}\n", encoding="utf-8")
    os.chmod(settings, 0o600)
    claude_md = profile / "claude-config" / "CLAUDE.md"
    claude_md.write_text("# theirs\n", encoding="utf-8")
    os.chmod(claude_md, 0o640)
    machine = doctor.Machine()
    plan = _plan(
        profile,
        store=str(profile / "notes"),
        wire_claude_md=True,
        auto_dream_off=True,
    )
    config = init._resolve_config(machine, None)
    assert init.apply_plan(machine, plan, config) == init.EXIT_OK
    assert stat.S_IMODE(os.stat(settings).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(claude_md).st_mode) == 0o640
    assert json.loads(settings.read_text())["autoDreamEnabled"] is False


def test_the_import_line_is_added_once_however_the_file_is_spaced(profile) -> None:
    """Convergence is about the LINE, not about the bytes around it.

    A file whose last line has no newline, one with trailing blank lines, and
    one where the import is already the last line all have to end with exactly
    one copy of it — a second `@-import` of the same store is a duplicate the
    harness loads twice.
    """
    line = init._import_line(str(profile / "notes"))
    for existing, expected_tail in (
        ("", line + "\n"),
        ("# heading", "# heading\n" + line + "\n"),
        ("# heading\n\n\n", "# heading\n" + line + "\n"),
        ("# heading\n" + line + "\n", "# heading\n" + line + "\n"),
        (line, line),
        ("# heading\n" + line + "\nmore\n", "# heading\n" + line + "\nmore\n"),
    ):
        got = init._appended(existing, line)
        assert got == expected_tail, (repr(existing), repr(got))
        # And it is idempotent: a second pass changes nothing.
        assert init._appended(got, line) == got, repr(got)
        assert got.count(line) == 1, repr(got)


def test_the_dry_run_never_runs_a_program_the_checkout_supplied(
    profile, monkeypatch
) -> None:
    """`init --dry-run` is the pre-approved half of the handshake.

    It asks git whether a target is tracked, so a checkout that puts its own
    `git` in front of the system one on PATH — a `node_modules/.bin`, a
    direnv-exported venv — chooses a program the pre-approved call then runs as
    the user. The shim here is a SYMLINK out of the session directory, because
    the executable's own path cannot answer that question.
    """
    marker = profile / "PWNED-git.txt"
    hostile = profile / "elsewhere" / "prog"
    hostile.parent.mkdir(parents=True, exist_ok=True)
    hostile.write_text(f"#!/bin/sh\necho pwned > {marker}\nexit 0\n", encoding="utf-8")
    hostile.chmod(0o755)
    shim = profile / "project" / "node_modules" / ".bin"
    shim.mkdir(parents=True)
    (shim / "git").symlink_to(hostile)
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    target = profile / "claude-config" / "CLAUDE.md"
    target.write_text("# theirs\n", encoding="utf-8")
    assert init._git_tracked(str(target)) is False
    assert not marker.exists(), marker.read_text()


def test_the_checker_command_is_built_and_no_input_contributes_a_word(
    profile, monkeypatch
) -> None:
    """The two cases this replaces both asserted that a hostile
    `$MEMKIT_CHECKER_CMD` was RE-RESOLVED — one that the directory in it was
    re-derived, one that an absolute word in it was refused. Both were rules
    about an input that no longer reaches this function, and a rule about an
    input is one more input than a constructed command has.

    `--confirm`'s permission prompt shows `memkit init --confirm <digest>` and
    never this argv, so consent for the command is not consent for whatever a
    session's PATH supplied. What runs is therefore derived, not received:
    THIS process's interpreter and a constant tail.
    """
    hostile = profile / "elsewhere" / "fakepy"
    hostile.parent.mkdir(parents=True, exist_ok=True)
    marker = profile / "PWNED-checker.txt"
    hostile.write_text(
        f"#!/bin/sh\necho pwned > {marker}\nexit 0\n", encoding="utf-8"
    )
    hostile.chmod(0o755)
    shim = profile / "project" / "node_modules" / ".bin"
    shim.mkdir(parents=True)
    for name in ("uvx", "uv", "python3", "python3.12"):
        (shim / name).symlink_to(hostile)
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    for name, value in (
        ("MEMKIT_CHECKER_ROUTE", "python"),
        ("MEMKIT_CHECKER_CMD", f"{hostile} -m memkit.memory_integrity"),
        # The TAIL as well as the interpreter: a constant with a hole in it is
        # an input under another name, and this is the spelling that would put
        # one there.
        ("MEMKIT_CHECKER_TAIL", "-c import os;os.system('x')"),
    ):
        monkeypatch.setenv(name, value)
    # A CONSTANT WITH NO HOLE, asserted on the source. The value alone would
    # not say it: a tail computed from the environment at import time reads
    # back as its default in any process that did not set the variable, which
    # is every process except the one the attack is in.
    import ast

    tree = ast.parse(pathlib.Path(_exec.__file__).read_text(encoding="utf-8"))
    assigned = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "CHECKER_TAIL"
    )
    assert isinstance(assigned, ast.Tuple), ast.dump(assigned)
    assert [
        e.value for e in assigned.elts if isinstance(e, ast.Constant)
    ] == ["-m", "memkit.memory_integrity"], ast.dump(assigned)
    assert _exec.CHECKER_TAIL == ("-m", "memkit.memory_integrity"), (
        _exec.CHECKER_TAIL
    )

    ran: list = []
    real = init.subprocess.run

    def watched(argv, *a, **kw):
        ran.append(list(argv))
        return real([sys.executable, "-c", "raise SystemExit(0)"], *a, **kw)

    monkeypatch.setattr(init.subprocess, "run", watched)
    try:
        init._run_checker(doctor.Machine(), str(profile / "memkit.json"))
    finally:
        monkeypatch.setattr(init.subprocess, "run", real)
    assert ran, "the checker was never invoked, so this proves nothing"
    # This interpreter, and the constant tail. Not the environment's word, not
    # the shim's, and no `memory-integrity` for anything to resolve by name.
    assert ran[0] == [
        sys.executable,
        *_exec.CHECKER_TAIL,
        "--config",
        str(profile / "memkit.json"),
    ], ran[0]
    assert not marker.exists(), marker.read_text()

    # And with no route at all, one condition decides that nothing runs.
    monkeypatch.setattr(
        doctor, "_probe_checker_route", lambda: (_exec.CheckerRoute.NONE, "")
    )
    ran.clear()
    monkeypatch.setattr(init.subprocess, "run", watched)
    try:
        code, detail = init._run_checker(
            doctor.Machine(), str(profile / "memkit.json")
        )
    finally:
        monkeypatch.setattr(init.subprocess, "run", real)
    assert not ran, ran
    assert code == 1
    assert "no checker route" in detail, detail



def test_the_dry_run_runs_git_where_no_configuration_can_name_a_program(
    profile, monkeypatch
) -> None:
    """Which git runs was the first half of this rule; where it runs is the
    second.

    A repository's own `.git/config` is a program-selection surface: `git
    ls-files --error-unmatch` executes `core.fsmonitor`, and `$CLAUDE_CONFIG_DIR`
    is what decides which repository this call stands in. `init --dry-run
    --wire-claude-md` is the pre-approved half of the handshake, so a checkout
    that names a program there gets it run as the user with no prompt. The
    same primitive is reachable from `GIT_CONFIG_COUNT` alone, with no path
    steering at all, which is why both are here.
    """
    if not _which_git():
        pytest.skip("no git")
    repo = profile / "elsewhere" / "dotfiles"
    repo.mkdir(parents=True, exist_ok=True)
    marker = profile / "PWNED-git-config.txt"
    named = profile / "elsewhere" / "fsmon"
    named.write_text(
        f"#!/bin/sh\necho pwned >> {marker}\nexit 0\n", encoding="utf-8"
    )
    named.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=60)
    subprocess.run(
        ["git", "config", "core.fsmonitor", str(named)],
        cwd=repo, check=True, timeout=60,
    )
    target = repo / "CLAUDE.md"
    target.write_text("# theirs\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "CLAUDE.md"],
        cwd=repo, check=True, timeout=60,
    )
    # The staging above ran git WITHOUT the hardening, so it proves the probe
    # can fire on this machine's git at all. A case whose marker could never
    # be written would assert nothing.
    assert marker.exists(), "core.fsmonitor never fired, so this proves nothing"
    marker.unlink()

    # The warning is still made — the hardening may not cost the check its
    # subject — and the named program is not run.
    assert init._git_tracked(str(target)) is True
    assert not marker.exists(), marker.read_text()

    # The environment route: no repository config at all, and no path
    # steering. `GIT_CONFIG_COUNT` names the program on its own.
    plain = profile / "elsewhere" / "plain"
    plain.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=plain, check=True, timeout=60)
    other = plain / "CLAUDE.md"
    other.write_text("# theirs\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "CLAUDE.md"],
        cwd=plain, check=True, timeout=60,
    )
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(named))
    # The same call this function makes, without the hardening: the live
    # probe, so a green assertion below cannot come from a variable this git
    # ignores.
    subprocess.run(
        ["git", "-C", str(plain), "ls-files", "--error-unmatch", str(other)],
        capture_output=True, timeout=60,
    )
    assert marker.exists(), "GIT_CONFIG_COUNT never fired, so this proves nothing"
    marker.unlink()
    assert init._git_tracked(str(other)) is True
    assert not marker.exists(), marker.read_text()


def test_the_dry_run_never_asks_git_about_a_directory_inside_this_session(
    profile,
) -> None:
    """A `-c` override silences the keys somebody thought of.

    A repository can always add one nobody did, so the directory itself is
    refused rather than only disarmed: `$CLAUDE_CONFIG_DIR` pointed into the
    checkout is the checkout asking for git to be run inside it, and the
    warning is not worth that.
    """
    if not _which_git():
        pytest.skip("no git")
    inside = profile / "project" / ".claude"
    inside.mkdir(parents=True, exist_ok=True)
    target = inside / "CLAUDE.md"
    target.write_text("# theirs\n", encoding="utf-8")
    # A REAL repository, and the file really tracked in it, so a False answer
    # here is the refusal and not "there was nothing to find".
    subprocess.run(["git", "init", "-q"], cwd=inside, check=True, timeout=60)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "CLAUDE.md"],
        cwd=inside, check=True, timeout=60,
    )
    assert init._git_tracked(str(target)) is False


def test_no_digest_in_init_dies_on_a_lone_surrogate() -> None:
    """`--dry-run` exists to show an adopter what would be written, and it
    digests before it prints.

    `sys.argv` is decoded with `surrogateescape`, so `--config` naming a path
    the filesystem holds as undecodable bytes hands this command a lone
    surrogate; a store path and a file's content reach the same helper the
    same way. A strict `.encode()` raises `UnicodeEncodeError` on every one of
    them, and the manifest an adopter asked to see is then a traceback out of
    a hashing helper — nothing written, and nothing said about why.

    Same rule as the hook module's, held by the same imported scan rather than
    by a copy of it: every encode names its handler. No exceptions in this
    file — the hook has one argued strict encode because there the raise IS a
    refusal, and nothing here refuses anything by dying.
    """
    from test_memory_prompt_recall import _unhandled_encodes

    surrogate = json.loads('"\\udcff"')
    assert len(init._sha(f"/tmp/memkit{surrogate}.json")) == 64
    # Non-vacuity: the digest still SEPARATES, which is why it is taken over
    # the raw path rather than over a sanitized one.
    assert init._sha(f"/a{surrogate}") != init._sha(f"/b{surrogate}")
    assert init._canary_nonce(f"/tmp/memkit{surrogate}.json").startswith("mkc")

    source = pathlib.Path(init.__file__).read_text(encoding="utf-8")
    assert _unhandled_encodes(source) == []
    # And the scan still sees its subject in this file's own text.
    assert _unhandled_encodes(source + '\ndef f(t):\n    return t.encode("utf-8")\n')


# --- adopting the harness's own auto-memory ----------------------------------


TRAP = "---\nname: app trap\ndescription: app trap one\n---\n# t\nbody\n"
BARE = "# Home note\n\nno frontmatter here\n"


def _harness(profile, key: str, files: dict) -> pathlib.Path:
    """One harness auto-memory directory under the profile's own config dir.

    Where the harness writes, and where the inventory looks: flat, one
    directory per project key, `<config dir>/projects/<key>/memory`.
    """
    memory = profile / "claude-config" / "projects" / key / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (memory / name).write_text(text, encoding="utf-8")
    return memory


def _rows_of(text: str) -> dict:
    """{link: description} for every row in a generated ledger."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("- ["):
            continue
        link = line[line.index("(") + 1 : line.index(")")]
        out[link] = line.split(" — ", 1)[1]
    return out


def test_a_harness_already_pointed_somewhere_else_refuses_by_name(profile) -> None:
    """Where an agent writes its memories is a decision somebody has already
    made, and a setup command that overwrote it would be making it again.

    Named per scope rather than as "your settings", because the harness reads
    four of them: measured on 2.1.258 a checked-in `.claude/settings.json`
    really does redirect auto-memory, so the file to edit is a fact the
    refusal has to carry.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    (profile / "claude-config" / "settings.json").write_text(
        json.dumps({"autoMemoryDirectory": "/elsewhere"}), encoding="utf-8"
    )
    refusal = _refuses(profile, "auto-memory-redirected", adopt_auto_memory=True)
    assert "/elsewhere" in refusal.message
    assert "in user settings" in refusal.message
    # And the scope init would write is named too, so the sentence says what
    # it would have done as well as what it found.
    assert "user scope" in refusal.message

    # A CHECKED-IN settings file outranks the adopter's own, so it is the one
    # named — the whole reason this is per-scope.
    checkout = profile / "project" / ".claude"
    checkout.mkdir(parents=True)
    (checkout / "settings.json").write_text(
        json.dumps({"autoMemoryDirectory": "/from-the-clone"}), encoding="utf-8"
    )
    refusal = _refuses(profile, "auto-memory-redirected", adopt_auto_memory=True)
    assert "in project settings" in refusal.message
    assert "/from-the-clone" in refusal.message

    # AND `settings.local.json` OUTRANKS BOTH — measured on 2.1.258, and the
    # everyday instance of this: an uncommitted file in somebody's own checkout
    # that the harness reads before the settings they think they are editing.
    (checkout / "settings.local.json").write_text(
        json.dumps({"autoMemoryDirectory": "/my-own-untracked-choice"}),
        encoding="utf-8",
    )
    refusal = _refuses(profile, "auto-memory-redirected", adopt_auto_memory=True)
    assert "in local settings" in refusal.message
    assert "/my-own-untracked-choice" in refusal.message


def test_adoption_refuses_while_the_harness_feature_is_switched_off(
    profile, monkeypatch
) -> None:
    """Copying what is there and then pointing a switched-off feature at the
    store would leave an adopter with a redirect nothing acts on and a
    directory to clean up.

    EVERY SCOPE THAT OUTRANKS THE ONE INIT WRITES. `managed` and `local` both
    outrank `user`, so a switch read from either of them is the one deciding
    whether the harness writes anything at all — and each is asserted here by
    name, because a loop that skipped one would leave that adopter's memories
    copied and a redirect written under a feature nobody turned on.

    The USER scope is not one of them: that is the scope `--auto-memory-off`
    writes, and its case is the convergence test below.

    The flag that turns it off is not refused by the same rule: it writes one
    boolean and has to stay idempotent.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    # `settings.local.json` in the checkout, which the harness reads ahead of
    # user settings.
    checkout = profile / "project" / ".claude"
    checkout.mkdir(parents=True)
    (checkout / "settings.local.json").write_text(
        json.dumps({"autoMemoryEnabled": False}), encoding="utf-8"
    )
    refusal = _refuses(profile, "auto-memory-off", adopt_auto_memory=True)
    assert "local settings" in refusal.message
    # Neither refusal is evaluated for the other flag, and neither is
    # evaluated for a plain init.
    assert _plan(profile, auto_memory_off=True).actions
    assert _plan(profile).actions
    (checkout / "settings.local.json").unlink()

    # And managed settings, the administrator's — the one scope the adopter in
    # front of the terminal cannot answer for.
    managed = profile / "managed"
    managed.mkdir()
    monkeypatch.setattr(doctor, "_managed_dir", lambda: str(managed))
    (managed / doctor.MANAGED_SETTINGS_NAME).write_text(
        json.dumps({"autoMemoryEnabled": False}), encoding="utf-8"
    )
    refusal = _refuses(profile, "auto-memory-off", adopt_auto_memory=True)
    assert "managed settings" in refusal.message


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_switching_auto_memory_off_first_still_leaves_adoption_a_path(
    profile,
) -> None:
    """NO STATE MEMKIT WROTE MAKES A MEMKIT FLAG UNRECOVERABLE.

    `--auto-memory-off` writes `"autoMemoryEnabled": false` into user settings.
    Refusing `--adopt-auto-memory` on that same boolean made the second flag
    unusable because of the first, and the two are mutually exclusive — so no
    invocation undoes it and the only way back was hand-editing the settings
    file. Both orders converge instead, and the manifest discloses the off
    state it adopts under.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    off = _dry(profile, "--store", str(store), "--auto-memory-off")
    assert off.returncode == init.EXIT_OK, off.stdout + off.stderr
    landed = _confirm(
        profile, _digest_of(off), "--store", str(store), "--auto-memory-off"
    )
    assert landed.returncode == init.EXIT_OK, landed.stdout + landed.stderr
    settings = profile / "claude-config" / "settings.json"
    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "autoMemoryEnabled": False
    }

    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "switched off" in manifest.stdout, manifest.stdout
    assert "BEFORE it was switched off" in manifest.stdout, manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    copied = store / "search" / init.ADOPT_DIRNAME / "-home-u" / "note.md"
    assert copied.is_file(), out.stdout
    written = json.loads(settings.read_text(encoding="utf-8"))
    assert written["autoMemoryEnabled"] is False
    assert written[harness_memory.DIRECTORY_KEY], written

    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout

    # The other order was never broken, and stays that way.
    other = _dry(profile, "--store", str(store), "--auto-memory-off")
    assert other.returncode == init.EXIT_OK, other.stdout + other.stderr
    assert "Nothing to write" in other.stdout, other.stdout


def test_the_adoption_manifest_names_every_directory_and_every_file(profile) -> None:
    """Consent is given to the paths, so the paths are what the manifest
    lists: the directories it would make, the files it would copy into them,
    the ledger it would regenerate and the one settings key it would set.
    """
    _harness(
        profile,
        "-home-u-git-app",
        {"trap.md": TRAP, "MEMORY.md": "# idx\n- trap\n"},
    )
    _harness(profile, "-home-u", {"note.md": BARE})
    store = profile / "notes"
    before = _snapshot(profile)
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    base = store / "search" / "projects"
    order = [a.path for a in plan.actions]
    ops = [a.op for a in plan.actions]
    dirs = [a.path for a in plan.actions if a.op == init.CREATE_DIR]
    for path in (base, base / "-home-u-git-app", base / "-home-u"):
        assert str(path) in dirs, path
    files = [a.path for a in plan.actions if a.op == init.CREATE_FILE]
    for rel in ("-home-u-git-app/trap.md", "-home-u-git-app/MEMORY.md",
                "-home-u/note.md"):
        assert str(base / rel) in files, rel
    # A directory before the files that go in it, so the preflight meets a
    # type clash before anything is written rather than at `os.makedirs`.
    assert order.index(str(base)) < order.index(str(base / "-home-u"))
    assert order.index(str(base / "-home-u")) < order.index(
        str(base / "-home-u" / "note.md")
    )
    # And every copy and the ledger AHEAD of the verification: a memory whose
    # row landed after the check is an orphan the check could not have seen.
    assert order.index(str(base / "-home-u" / "note.md")) < ops.index(init.VERIFY)
    assert order.index(str(store / "SEARCH.md")) < ops.index(init.VERIFY)
    # ONE settings write, and it is the redirect.
    (settings,) = [a for a in plan.actions if a.op == init.SETTINGS_WRITE]
    assert json.loads(settings.content) == {
        "autoMemoryDirectory": str(store / "search" / "auto-memory")
    }
    # The ledger rows the two memories and not the index that travelled with
    # them: a `MEMORY.md` at any depth is a ledger, never a memory.
    (ledger,) = [a for a in plan.actions if a.path == str(store / "SEARCH.md")]
    rows = _rows_of(ledger.content)
    assert rows["search/projects/-home-u-git-app/trap.md"] == "app trap one"
    assert rows["search/projects/-home-u/note.md"] == "Home note"
    assert "search/projects/-home-u-git-app/MEMORY.md" not in rows
    assert _snapshot(profile) == before


def test_the_confirm_turn_copies_the_memories_and_leaves_the_originals(
    profile,
) -> None:
    """COPY, NEVER MOVE. The worst outcome of a wrong guess here has to be a
    file to delete, so the originals are still there afterwards and the
    settings file gained exactly one key.
    """
    app = _harness(
        profile,
        "-home-u-git-app",
        {"trap.md": TRAP, "MEMORY.md": "# idx\n- trap\n"},
    )
    home = _harness(profile, "-home-u", {"note.md": BARE})
    (profile / "claude-config" / "settings.json").write_text(
        json.dumps({"theme": "dark"}), encoding="utf-8"
    )
    store = profile / "home" / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stderr
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    base = store / "search" / "projects"
    assert (base / "-home-u-git-app" / "trap.md").read_text() == TRAP
    assert (base / "-home-u-git-app" / "MEMORY.md").read_text() == "# idx\n- trap\n"
    landed = (base / "-home-u" / "note.md").read_text()
    assert "description: Home note" in landed
    assert landed.endswith(BARE)
    # The originals, byte for byte.
    assert (app / "trap.md").read_text() == TRAP
    assert (home / "note.md").read_text() == BARE
    # `~/`-form under HOME, and the one key added to what was already there.
    settings = json.loads(
        (profile / "claude-config" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings == {
        "theme": "dark",
        "autoMemoryDirectory": "~/notes/search/auto-memory",
    }
    # A SECOND RUN IS A NO-OP: every action redundant, the count said back,
    # and no settings write left to make.
    again = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert again.writes == []
    assert any("3 already adopted" in note for note in again.notes)
    (settings_action,) = [a for a in again.actions if a.op == init.SETTINGS_WRITE]
    assert settings_action.redundant


def test_a_destination_that_differs_is_named_and_never_written_over(profile) -> None:
    """A file already at the destination is somebody's, whoever put it there.
    It is named, left exactly as it is, and the row the ledger carries comes
    from ITS text — a plan that rowed the source would describe a file that is
    not on disk.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    dest = store / "search" / "projects" / "-home-u" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text(
        "---\nname: mine\ndescription: mine already\n---\n\nkeep me\n",
        encoding="utf-8",
    )
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert not [a for a in plan.actions if a.path == str(dest)]
    assert any("diverged" in note and "note.md" in note for note in plan.notes)
    (ledger,) = [a for a in plan.actions if a.path == str(store / "SEARCH.md")]
    rows = _rows_of(ledger.content)
    assert rows["search/projects/-home-u/note.md"] == "mine already"
    assert dest.read_text() == (
        "---\nname: mine\ndescription: mine already\n---\n\nkeep me\n"
    )


def test_a_destination_that_cannot_be_read_is_diverged_and_not_a_traceback(
    profile,
) -> None:
    """`--dry-run` is the pre-approved turn, so it has one contract above every
    other: it answers. A destination this process cannot decode or open is a
    file that was not compared, which is what `diverged` means — reaching the
    adopter as a traceback out of `read()` leaves them a refusal name they
    cannot act on and no manifest at all.
    """
    _harness(profile, "-home-u", {"note.md": TRAP, "blob.md": TRAP})
    store = profile / "notes"
    base = store / "search" / "projects" / "-home-u"
    base.mkdir(parents=True)
    (base / "blob.md").write_bytes(b"\xff\xfe not utf-8\n")
    shut = base / "note.md"
    shut.write_text("something\n", encoding="utf-8")
    shut.chmod(0)
    try:
        plan = _plan(profile, store=str(store), adopt_auto_memory=True)
        diverged = [n for n in plan.notes if "diverged" in n]
        assert any("blob.md" in n for n in diverged), diverged
        assert any("cannot be read" in n and "note.md" in n for n in diverged), (
            diverged
        )
        assert not [a for a in plan.actions if a.path in (str(shut), str(base / "blob.md"))]
    finally:
        shut.chmod(0o644)


def test_every_description_adoption_writes_is_one_the_checker_can_read(
    profile,
) -> None:
    """FOUR CLASSES, ONE FILE EACH, and the property over all of them is the
    same: the file that lands carries a description the store's own checker
    reads, because a memory it cannot read a description for is what fails the
    VERIFY step init runs on its own work.

    Bodies are never touched. The only edit is the frontmatter's description
    line, and only where there was nothing usable on it.
    """
    _harness(
        profile,
        "-classes",
        {
            # No description at all: the first heading stands in for one.
            "heading.md": "---\nname: h\n---\n\n## The heading line\n\nbody\n",
            # Over the checker's cap: truncated, with the ellipsis counted.
            "toolong.md": "---\nname: t\ndescription: " + "L" * 200 + "\n---\n\nb\n",
            # A description the checker's own reader rejects, replaced by a
            # value that needs quoting to survive the round trip.
            "colon.md": (
                "---\nname: c\ndescription: a thing: with a colon\n---\n\n"
                "# a thing: with a colon\n\nbody\n"
            ),
            # No frontmatter block at all: one is prepended.
            "bare.md": BARE,
        },
    )
    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    base = store / "search" / "projects" / "-classes"
    landed = {
        os.path.basename(a.path): a.content
        for a in plan.actions
        if a.op == init.CREATE_FILE and a.path.startswith(str(base))
    }
    assert set(landed) == {"heading.md", "toolong.md", "colon.md", "bare.md"}
    read_back = {}
    for name, text in landed.items():
        raw = init._frontmatter_of(text).get("description", "")
        value = init._scalar_of(raw)
        assert value is not None, (name, raw)
        assert len(value) <= init._MAX_DESC_CHARS, name
        read_back[name] = value
    assert read_back["heading.md"] == "The heading line"
    truncated = read_back["toolong.md"]
    assert len(truncated) == init._MAX_DESC_CHARS and truncated.endswith("…")
    assert read_back["colon.md"] == "a thing: with a colon"
    assert landed["colon.md"].count('description: "a thing: with a colon"') == 1
    assert landed["bare.md"].startswith("---\n")
    assert landed["bare.md"].endswith(BARE)
    # The two files that already had a readable description are unchanged
    # bytes, which is the rule the other four are the exception to.
    _harness(profile, "-kept", {"fine.md": TRAP})
    kept = _plan(profile, store=str(store), adopt_auto_memory=True)
    (copied,) = [a for a in kept.actions if a.path.endswith("-kept/fine.md")]
    assert copied.content == TRAP
    assert copied.note == ""


def test_what_adoption_will_not_carry_is_named_rather_than_dropped(profile) -> None:
    """One file per class, and every one of them named in the manifest: a
    count an adopter cannot reconcile against their own `ls` is the number the
    list exists to make checkable.
    """
    memory = _harness(profile, "-skips", {"keep.md": TRAP})
    (memory / "binary.md").write_bytes(
        b"---\nname: b\ndescription: d\n---\n\n\xff\xfe\n"
    )
    (memory / "huge.md").write_text(
        "x" * (init.ADOPT_MAX_BYTES + 1), encoding="utf-8"
    )
    (memory / "tiered.md").write_text(
        "---\nname: t\ndescription: d\n---\n\ntier: hot\n", encoding="utf-8"
    )
    outside = profile / "outside.md"
    outside.write_text(TRAP, encoding="utf-8")
    (memory / "linked.md").symlink_to(outside)

    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    skipped = " ".join(n for n in plan.notes if "skipped" in n)
    assert "binary.md: is not UTF-8" in skipped
    assert "huge.md: is" in skipped and "byte cap" in skipped
    assert "tiered.md: carries a `tier:` line" in skipped
    assert "linked.md: the file is a symlink" in skipped
    copied = [
        a.path for a in plan.actions
        if a.op == init.CREATE_FILE and "projects" in a.path
    ]
    assert copied == [str(store / "search" / "projects" / "-skips" / "keep.md")]




def test_only_a_memory_directory_wired_into_a_store_is_already_redirected(
    profile, monkeypatch
) -> None:
    """A LINK IS AN ANSWER ONLY WHERE IT LANDS.

    Adoption skipped every symlinked memory directory as "already redirected"
    while doctor counted the ones landing nowhere as outside every store — so
    a memory directory linked to an ordinary directory was reported as handled
    on one command and as unadopted on the other, and never chased on either.
    The two now ask one predicate, and this asserts the numbers as well as the
    predicate.
    """
    # A store and a config first: "inside a store" is not a question that can
    # be asked before one exists.
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    # The route the hook and doctor read it by. Without one, no store is
    # configured and nothing can be inside one — which both commands already
    # agree about.
    monkeypatch.setenv(
        hook.CONFIG_ENV, str(profile / "home" / ".config" / "memkit" / "memkit.json")
    )

    wired_at = store / "search" / "wired-memories"
    wired_at.mkdir()
    (wired_at / "wired.md").write_text(TRAP, encoding="utf-8")
    wired = profile / "claude-config" / "projects" / "-wired"
    wired.mkdir(parents=True)
    (wired / "memory").symlink_to(wired_at)

    # The PROJECT directory reached through a link, rather than its `memory/`:
    # `harness_memory` carries three link flags and each needs its own answer.
    wired_project = store / "search" / "wired-project"
    (wired_project / "memory").mkdir(parents=True)
    (wired_project / "memory" / "p.md").write_text(TRAP, encoding="utf-8")
    (profile / "claude-config" / "projects" / "-wired-project").symlink_to(
        wired_project
    )

    nowhere = profile / "linked-memories"
    nowhere.mkdir()
    (nowhere / "loose.md").write_text(TRAP, encoding="utf-8")
    loose = profile / "claude-config" / "projects" / "-linked-outside"
    loose.mkdir(parents=True)
    (loose / "memory").symlink_to(nowhere)

    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    notes = " ".join(plan.notes)
    assert "'-wired': already redirected, skipped" in notes
    assert "'-wired-project': already redirected, skipped" in notes
    assert "-linked-outside: already redirected" not in notes
    assert "1 project memory directory holds 1 memory outside every store" in notes
    # And doctor, over the same machine, counts the same one.
    machine = doctor.Machine()
    known = harness_memory.inventory(init._harness_config_dir())
    outside = [p for p in known if init._adoptable(machine, str(store), p)]
    assert [p.key for p in outside] == ["-linked-outside"]
    (row,) = [
        check for check in doctor.collect(machine) if check.id == "auto-memory"
    ]
    assert "1 project directory holds 1 memory outside every store" in row.detail


def test_a_hot_memory_no_index_rows_is_named_before_the_confirm(profile) -> None:
    """The checker walks `hot/` as well as `search/` and generates rows for
    neither: a hot memory's row lives in MEMORY.md, which is hand-written.

    So a store holding one gets an `ORPHAN` from the VERIFY step init runs on
    its own work — after the store is on disk, with exit 6 and nothing in the
    manifest that saw it coming. Rowing them in SEARCH.md instead is worse:
    the checker answers `MISROWED`.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    (store / "hot" / "h.md").write_text(
        "---\nname: h\ndescription: a hot memory\n---\n\nbody\n",
        encoding="utf-8",
    )
    plan = _plan(profile, store=str(store))
    assert any("hot/h.md" in note and "no row" in note for note in plan.notes), (
        plan.notes
    )
    # And it is not answered by rowing it in the generated ledger.
    (ledger,) = [a for a in plan.actions if a.path == str(store / "SEARCH.md")]
    assert "hot/h.md" not in ledger.content
    # And a store whose MEMORY.md DOES row it never reaches this: an index
    # somebody wrote is refused outright, because nothing else records those
    # rows. So the note above covers every store init will actually plan.
    (store / "MEMORY.md").write_text(
        (store / "MEMORY.md").read_text() + "\n- [h](hot/h.md) — a hot memory\n",
        encoding="utf-8",
    )
    _refuses(profile, "adopted-memory-index", store=str(store))


def test_a_memory_whose_row_cannot_be_read_is_named_and_not_dropped(
    profile,
) -> None:
    """A blanket suppress produced no row, no note and no refusal — and the
    checker that reads the tree rather than the ledger calls a memory with no
    row an ORPHAN, on a store init has just declared correct.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    shut = store / "search" / "shut.md"
    shut.write_text("---\nname: s\ndescription: shut\n---\n\nb\n", encoding="utf-8")
    shut.chmod(0)
    try:
        plan = _plan(profile, store=str(store))
        assert any(
            "shut.md" in note and "no row" in note and "PermissionError" in note
            for note in plan.notes
        ), plan.notes
    finally:
        shut.chmod(0o644)




def test_the_managed_scope_is_asked_about_the_redirect_like_every_other(
    profile, monkeypatch
) -> None:
    """MANAGED SETTINGS ARE THE ADMINISTRATOR'S, and the highest-precedence
    scope the harness reads. A hole there is init redirecting where an agent
    writes against site policy — the one scope whose answer the adopter in
    front of the terminal cannot give.
    """
    managed = profile / "managed"
    managed.mkdir()
    monkeypatch.setattr(doctor, "_managed_dir", lambda: str(managed))
    (managed / doctor.MANAGED_SETTINGS_NAME).write_text(
        json.dumps({"autoMemoryDirectory": "/site/policy/memories"}),
        encoding="utf-8",
    )
    _harness(profile, "-home-u", {"note.md": TRAP})
    refusal = _refuses(profile, "auto-memory-redirected", adopt_auto_memory=True)
    assert "in managed settings" in refusal.message
    assert "/site/policy/memories" in refusal.message


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_the_description_cap_is_the_checkers_own(profile) -> None:
    """The truncation arithmetic is pinned and the constant it depends on was
    not. One character of drift is a `DESC-LONG` at the VERIFY step, on a store
    init has just built and just truncated a description for.
    """
    from memkit import memory_integrity as checker

    assert init._MAX_DESC_CHARS == checker.MAX_DESC_CHARS
    assert set(init._LEDGER_NAMES) == set(checker.LEDGER_NAMES)
    assert init._INDEX_HEADING == checker.INDEX_HEADING


def test_a_replaced_description_takes_the_lines_under_it(profile) -> None:
    """A `description:` whose value continued onto indented lines is one value,
    and replacing the first line alone leaves the rest of somebody else's
    sentence attached to the new one — which is why this is not a regex.
    """
    _harness(
        profile,
        "-home-u",
        {"c.md": (
            "---\nname: c\ndescription: >\n  a folded value\n"
            "  that runs on\nkeep: me\n---\n\nbody\n"
        )},
    )
    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    (copy,) = [a for a in plan.actions if a.path.endswith("c.md")]
    block = copy.content.split("\n---", 1)[0]
    assert "a folded value" not in block, block
    assert "that runs on" not in block, block
    # And nothing else in the block is touched.
    assert "keep: me" in block
    assert "a folded value" not in copy.content.split("\n---", 1)[1]


def test_a_memory_a_sub_index_already_rows_is_not_rowed_again(profile) -> None:
    """A sub-index owns its members' rows. Generating a second one in SEARCH.md
    is the checker's `DOUBLE-LEDGER`, and the members are read from the
    sub-index's own text because membership is data rather than convention.

    THREE SHAPES, and the second is the one the halves disagreed about: a
    plain member, a member reached through a symlinked FILE — where the
    checker resolves the link and this must too — and a symlinked DIRECTORY,
    which neither walk descends.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    search = store / "search"
    domain = search / "domain"
    domain.mkdir()
    (domain / "plain.md").write_text(
        "---\nname: plain\ndescription: a plain member\n---\n\nb\n",
        encoding="utf-8",
    )
    (domain / "real.md").write_text(
        "---\nname: real\ndescription: reached through a link\n---\n\nb\n",
        encoding="utf-8",
    )
    (search / "link.md").symlink_to(domain / "real.md")
    elsewhere = profile / "outside-tree"
    elsewhere.mkdir()
    (elsewhere / "never.md").write_text(
        "---\nname: never\ndescription: below a linked directory\n---\n\nb\n",
        encoding="utf-8",
    )
    (search / "linked-dir").symlink_to(elsewhere)
    (domain / "INDEX.md").write_text(
        "## Index\n\n- [plain](plain.md) — a plain member\n"
        "- [real](../link.md) — reached through a link\n",
        encoding="utf-8",
    )
    config = profile / "home" / ".config" / "memkit" / "memkit.json"
    blob = json.loads(config.read_text())
    blob["stores"][0]["sub_indexes"] = ["search/domain/INDEX.md"]
    config.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    plan = _plan(profile, store=str(store))
    (ledger,) = [a for a in plan.actions if a.path == str(store / "SEARCH.md")]
    rows = _rows_of(ledger.content)
    # The member the sub-index rows is NOT rowed again...
    assert "search/domain/plain.md" not in rows
    assert "search/domain/real.md" not in rows
    # ...the link the sub-index resolved through is rowed, because that is the
    # row the checker's own --write generates for it...
    assert "search/link.md" in rows
    # ...and nothing under a symlinked directory is rowed, because neither
    # walk descends one.
    assert not [link for link in rows if "linked-dir" in link]


def test_an_adopted_memory_a_sub_index_rows_is_not_rowed_again(profile) -> None:
    """The exclusion above is applied where the rows are read off disk, and
    the adoption rows were merged in after it without going through it. So the
    one path that writes NEW rows was the one that skipped the guard: an
    adopter who moved an adopted memory's row into a sub-index of their own
    got it put back on the next run, and the store then failed the check init
    runs on its own work with DOUBLE-LEDGER.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    out = _confirm(
        profile,
        _digest_of(_dry(profile, "--store", str(store), "--adopt-auto-memory")),
        "--store", str(store), "--adopt-auto-memory",
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    link = "search/projects/-home-u/note.md"
    search = store / "SEARCH.md"
    assert link in search.read_text(encoding="utf-8")

    # The adopter moves the row into a sub-index of their own and declares it.
    adopted = store / "search" / "projects" / "-home-u"
    (adopted / "INDEX.md").write_text(
        "## Index\n\n- [app trap](note.md) — app trap one\n", encoding="utf-8"
    )
    search.write_text(
        "".join(
            line for line in search.read_text(encoding="utf-8").splitlines(True)
            if link not in line
        ),
        encoding="utf-8",
    )
    config = profile / "home" / ".config" / "memkit" / "memkit.json"
    blob = json.loads(config.read_text())
    blob["stores"][0]["sub_indexes"] = ["search/projects/-home-u/INDEX.md"]
    config.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    (ledger,) = [a for a in plan.actions if a.path == str(search)]
    assert link not in _rows_of(ledger.content), ledger.content


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_the_ledger_over_a_sub_index_is_still_the_one_write_would_leave(
    profile,
) -> None:
    """The fixpoint over the shape the two halves are read differently on: a
    sub-index rowing a memory through a symlink. `--write` regenerates every
    generated ledger, so a store where it changes nothing is a store whose
    SEARCH.md init got right.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    search = store / "search"
    domain = search / "domain"
    domain.mkdir()
    (domain / "real.md").write_text(
        "---\nname: real\ndescription: reached through a link\n---\n\nb\n",
        encoding="utf-8",
    )
    (search / "link.md").symlink_to(domain / "real.md")
    (domain / "INDEX.md").write_text(
        "## Index\n\n- [real](../link.md) — reached through a link\n",
        encoding="utf-8",
    )
    config = profile / "home" / ".config" / "memkit" / "memkit.json"
    blob = json.loads(config.read_text())
    blob["stores"][0]["sub_indexes"] = ["search/domain/INDEX.md"]
    config.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    plan = _plan(profile, store=str(store))
    (action,) = [a for a in plan.actions if a.path == str(store / "SEARCH.md")]
    (store / "SEARCH.md").write_text(action.content, encoding="utf-8")
    mine = (store / "SEARCH.md").read_text()
    written = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config),
         "--write"],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert written.returncode == 0, written.stdout + written.stderr
    assert (store / "SEARCH.md").read_text() == mine


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_the_ledger_init_writes_is_the_one_the_checker_would_generate(
    profile,
) -> None:
    """THE FIXPOINT, and the reason the checker's rules may be restated here at
    all: `memory_integrity` requires 3.12 and this module answers to the 3.9
    floor the dispatcher runs on, so the two cannot share one definition of
    what a ledger row is. What closes the gap is evidence — the checker's own
    generator, over the tree init made, has to produce the bytes init wrote.
    """
    from memkit import memory_integrity as checker

    _harness(
        profile,
        "-home-u-git-app",
        {
            "trap.md": TRAP,
            # A CAPITAL LABEL, deliberately: the rows are sorted case-
            # insensitively and an ASCII sort puts this one first, so a fixture
            # of lower-case names alone would agree with either rule and prove
            # neither.
            "upper.md": (
                "---\nname: Zeta note\ndescription: a capital label\n---\n\nbody\n"
            ),
        },
    )
    _harness(profile, "-home-u", {"note.md": BARE})
    store = profile / "home" / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    ledger = store / "SEARCH.md"
    entries = []
    for path in sorted((store / "search").rglob("*.md")):
        if path.name in checker.LEDGER_NAMES:
            continue
        front = checker._frontmatter(path)
        value, error = checker._scalar(front.get("description", ""))
        assert error is None, (path, error)
        entries.append(
            (front.get("name") or path.stem, os.path.relpath(path, store), value)
        )
    assert len(entries) == 4, entries
    # Non-vacuity: the labels really do sort differently under the two rules,
    # so the equality below is a claim about the ordering as well as the text.
    assert sorted(e[0] for e in entries) != sorted(
        (e[0] for e in entries), key=str.lower
    )
    assert checker._generate(ledger, entries) == ledger.read_text(encoding="utf-8")




def test_a_destination_that_is_a_link_is_named_and_never_written_through(
    profile,
) -> None:
    """A DANGLING SYMLINK AT A DESTINATION READS AS ABSENT.

    `state_token` opens the path, so a link pointing at nothing answers
    "absent" exactly as an empty directory entry does — the copy was planned,
    and the write followed the link, created the directories it named and put
    somebody's memory outside the store. Nothing in the manifest said so.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    dest = store / "search" / "projects" / "-home-u" / "note.md"
    dest.parent.mkdir(parents=True)
    outside = profile / "elsewhere" / "deep" / "planted.md"
    dest.symlink_to(outside)
    # And one that lands back INSIDE the store: containment says nothing about
    # it, and it is still a write at a path the manifest does not name — into
    # `hot/`, whose ledger is hand-written and rows nothing new.
    _harness(profile, "-home-v", {"note.md": TRAP})
    inward = store / "search" / "projects" / "-home-v" / "note.md"
    inward.parent.mkdir(parents=True)
    inward.symlink_to(store / "hot" / "smuggled.md")
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert not [a for a in plan.actions if a.path in (str(dest), str(inward))]
    diverged = [note for note in plan.notes if "diverged" in note]
    assert any(str(outside) in note for note in diverged), diverged
    assert any(str(store / "hot" / "smuggled.md") in note for note in diverged), (
        diverged
    )
    assert any("0 files" in note and "2 diverged" in note for note in plan.notes)
    machine = doctor.Machine()
    assert init.apply_plan(
        machine, plan, init._resolve_config(machine, None)
    ) in (init.EXIT_OK, init.EXIT_INCOMPLETE)
    assert not outside.exists(), "a copy landed outside the store"
    assert not outside.parent.exists(), "a directory was made outside the store"
    assert not (store / "hot" / "smuggled.md").exists(), "a copy went through a link"




def test_a_linked_destination_directory_takes_the_whole_project_with_it(
    profile,
) -> None:
    """THE LINK THAT MOVES A WRITE IS AS OFTEN A DIRECTORY AS THE LEAF.

    `os.makedirs` follows a linked component exactly as `open` does, so a
    linked `search/projects/` sends every copy under it somewhere the manifest
    does not name — with no leaf a link and nothing else to notice it. Both
    shapes are here: one landing outside the store, where containment is the
    only test that answers, and one landing back inside it, where the link
    itself is.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    outside = profile / "elsewhere"
    outside.mkdir()
    (store / "search").mkdir(parents=True)
    (store / "search" / "projects").symlink_to(outside)
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert not [a for a in plan.actions if a.op == init.CREATE_FILE
                and "-home-u" in a.path]
    assert any(str(outside) in note and "diverged" in note for note in plan.notes), (
        plan.notes
    )
    machine = doctor.Machine()
    init.apply_plan(machine, plan, init._resolve_config(machine, None))
    assert list(outside.iterdir()) == [], "a copy landed outside the store"

    # And the project's OWN directory as a link, landing back inside the
    # store: containment says yes and the link is still a path the manifest
    # does not name, so this is the clause that answers.
    (store / "search" / "projects").unlink()
    (store / "search" / "projects").mkdir()
    inward = store / "search" / "somewhere-else"
    inward.mkdir()
    (store / "search" / "projects" / "-home-u").symlink_to(inward)
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert not [a for a in plan.actions if a.op == init.CREATE_FILE
                and "-home-u" in a.path]
    assert any(str(inward) in note and "diverged" in note for note in plan.notes), (
        plan.notes
    )
    assert list(inward.iterdir()) == []


@pytest.mark.parametrize("shape", ("linked-projects-dir", "linked-key-dir"))
def test_a_link_that_lands_back_inside_the_store_still_never_converges(
    profile, shape
) -> None:
    """CONTAINED IS NOT THE SAME AS CONVERGENT, and containment was the only
    question the guard used to ask.

    A link BELOW the store that resolves back INSIDE it passes every
    containment test there is: no leaf is a link, and the resolved path is in
    the store. The row init writes is `relpath(dest, store)`, spelled
    lexically — `search/projects/<key>/note.md` — while the bytes land where
    the link points, which is the name the checker enumerates. Init then exits
    6 on its own LEDGER-DRIFT every run; `memory-integrity --write` repairs the
    row and the next init puts it back. Two tools each undoing the other is
    worse than a refusal, so the project is named and left alone instead.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "home" / f"notes-{shape}"
    projects = store / "search" / "projects"
    elsewhere = store / "search" / "elsewhere"
    elsewhere.mkdir(parents=True)
    landing = elsewhere
    if shape == "linked-projects-dir":
        # No leaf is a link and nothing is outside the store: the whole
        # `projects/` directory is one, pointing at a sibling.
        projects.symlink_to(elsewhere)
    else:
        projects.mkdir()
        (projects / "-home-u").symlink_to(elsewhere)

    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "diverged" in manifest.stdout, manifest.stdout
    assert hook._display_path(str(landing)) in manifest.stdout, manifest.stdout
    assert "0 files" in manifest.stdout, manifest.stdout

    # The confirm runs the integrity check on the store it just built, so an
    # exit 0 here is the checker's answer as well as init's.
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    assert list(landing.iterdir()) == [], "a copy went through the link"

    # AND IT CONVERGES. The oscillation this guards is only visible on the
    # second turn: the first one wrote a row for a path nothing is at, and the
    # second one is where init and the checker start undoing each other.
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert again.returncode == init.EXIT_OK, again.stdout + again.stderr
    assert "Nothing to write" in again.stdout, again.stdout
    settled = _confirm(
        profile, _digest_of(again), "--store", str(store), "--adopt-auto-memory"
    )
    assert settled.returncode == init.EXIT_OK, settled.stdout + settled.stderr


def test_a_store_that_is_itself_a_symlink_is_still_adopted_into(profile) -> None:
    """The other side of the same test, and the reason it compares the landing
    place against the store's OWN realpath rather than against `abspath`.

    A store on an external volume, in a dotfiles tree or under a synced
    directory is a store reached through a link, so EVERY destination in it
    resolves somewhere other than the path it is spelled as — and every one of
    them still lands exactly where its row says. A guard that only asked
    "did anything resolve?" would refuse to adopt into any of them.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    real = profile / "external-volume" / "notes"
    real.mkdir(parents=True)
    store = profile / "home" / "notes"
    store.symlink_to(real)
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert any("1 files" in note and "0 diverged" in note for note in plan.notes), (
        plan.notes
    )
    assert [
        a.path for a in plan.actions
        if a.op == init.CREATE_FILE and a.path.endswith("-home-u/note.md")
    ] == [str(store / "search" / "projects" / "-home-u" / "note.md")]

    # END TO END, because the claim is about the ledger and not the plan: the
    # confirm runs the integrity checker over the store it just built, so an
    # exit 0 is the checker agreeing that the row and the bytes are the same
    # file — reached one way through the link and the other way around it.
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    assert (real / "search" / "projects" / "-home-u" / "note.md").is_file()
    rows = _rows_of((store / "SEARCH.md").read_text(encoding="utf-8"))
    assert "search/projects/-home-u/note.md" in rows, rows


def test_a_link_planted_after_the_plan_never_lands_outside_the_store(
    profile,
) -> None:
    """The plan proves containment against the tree it was built over, and a
    link planted after that proves nothing. Every write the adoption plan makes
    carries the root it has to land inside, and the write itself is where that
    is enforced — the two moments are different and only the second one is the
    one that writes.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    base = store / "search" / "projects"
    copy = next(
        a for a in plan.pending if a.path.endswith("-home-u/note.md")
    )
    assert copy.confine == str(store)
    # Between the plan and the write: the directory the copies go in becomes a
    # link out of the store.
    outside = profile / "elsewhere"
    outside.mkdir()
    base.parent.mkdir(parents=True, exist_ok=True)
    base.symlink_to(outside)
    machine = doctor.Machine()
    code = init.apply_plan(machine, plan, init._resolve_config(machine, None))
    assert code == init.EXIT_INCOMPLETE
    assert list(outside.iterdir()) == [], "the write followed the planted link"


def test_a_link_planted_after_the_plan_never_lands_off_its_own_row(
    profile,
) -> None:
    """AND STAYING INSIDE THE STORE IS NOT LANDING WHERE THE ROW SAYS. The plan
    proves the destination resolves to the name it is spelled as; a link
    planted after it that resolves back into the store passes containment and
    still moves every copy off the row written for it. The write has to ask the
    plan's question rather than a weaker one, or the guard is only as strong as
    where the planted link happens to point.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    base = store / "search" / "projects"
    landing = store / "search" / "landing"
    landing.mkdir(parents=True)
    base.symlink_to(landing)
    machine = doctor.Machine()
    code = init.apply_plan(machine, plan, init._resolve_config(machine, None))
    assert code == init.EXIT_INCOMPLETE
    assert list(landing.iterdir()) == [], "the write followed the planted link"


def test_the_manifest_names_every_file_the_copy_would_write(profile) -> None:
    """A COUNT IS NOT A LIST. Doctor's remedy for this command promises it
    "names every file first", and consent for a command whose named harm is a
    wrong copy has to be given against the destinations rather than against
    their number. The summary line stays: it is what keeps a hundred copies
    from burying the writes that are not copies.
    """
    _harness(profile, "-home-u-git-app", {"trap.md": TRAP, "MEMORY.md": "# idx\n"})
    _harness(profile, "-home-u", {"note.md": BARE})
    store = profile / "notes"
    rendered = _plan(profile, store=str(store), adopt_auto_memory=True).render()
    base = store / "search" / "projects"
    for rel in ("-home-u-git-app/trap.md", "-home-u-git-app/MEMORY.md",
                "-home-u/note.md"):
        assert str(base / rel) in rendered, rel
    # And the summary line each of them hangs under.
    assert "2 files from" in rendered
    assert "1 file from" in rendered




def test_a_name_that_would_end_its_own_row_is_rewritten_in_the_copy(profile) -> None:
    """A ROW IS `- [label](link) — description` AND THE LABEL IS THE RAW HALF.

    A `name:` carrying `](` closes the link early: what follows is markdown
    the adopter never wrote — here a row pointing at a file that does not
    exist — and the real memory is left with no usable row at all. The COPY is
    what gets the safe name, not init's row, because the checker regenerates
    that row from this file and the two have to keep agreeing.
    """
    _harness(
        profile,
        "-home-u",
        {
            "evil.md": (
                "---\nname: x](hot/forged.md) — forged row\n"
                "description: real desc\n---\n\nbody\n"
            )
        },
    )
    store = profile / "notes"
    out = _confirm(
        profile,
        _digest_of(_dry(profile, "--store", str(store), "--adopt-auto-memory")),
        "--store", str(store), "--adopt-auto-memory",
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    copied = (store / "search" / "projects" / "-home-u" / "evil.md").read_text()
    assert "](" not in copied.split("\n---", 1)[0]
    ledger = (store / "SEARCH.md").read_text()
    # ONE row for it, it points at the memory rather than at the file the name
    # named, and the description is still the file's own.
    (row,) = [line for line in ledger.splitlines() if "evil.md" in line]
    assert row.endswith("(search/projects/-home-u/evil.md) — real desc"), row
    # The name's text survives as TEXT; what it may not be is a link.
    assert "](hot/forged.md)" not in ledger


def test_a_file_name_no_manifest_line_can_carry_is_skipped(profile) -> None:
    """POSIX admits a newline in a filename. Rendered raw it forged two
    correctly indented action lines into the surface a human reads before
    typing `--confirm`; sanitised, the note is honest and the destination path
    line beside it still cannot be — a path has to keep its spacing byte for
    byte to name a file that exists. So the file is not copied, and the one
    line that names it is cleaned.
    """
    memory = _harness(profile, "-home-u", {"ok.md": TRAP})
    forged = "a\n  create-file    ~-.claude-settings.json\nb.md"
    (memory / forged).write_text("body\n", encoding="utf-8")
    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    assert not [a for a in plan.actions if ".claude-settings" in a.path]
    assert any(
        "no manifest line and no ledger row could carry" in note
        and "skipped:" in note
        for note in plan.notes
    ), plan.notes
    rendered = plan.render()
    # NOT ONE forged line: every line of the manifest that looks like an action
    # is one the plan holds.
    ops = {init.CREATE_DIR, init.CREATE_FILE, init.SETTINGS_WRITE,
           init.MERGE_CONFIG, init.VERIFY, init.APPEND_LINE, init.REWRITE_FILE}
    printed = [
        line for line in rendered.splitlines()
        if line[:2] == "  " and line[2:3] != " " and line.split()[0] in ops
    ]
    assert len(printed) == len([
        a for a in plan.pending if not a.group
    ]) + len({a.group for a in plan.pending if a.group})


def test_a_file_name_no_row_could_point_at_is_skipped(profile) -> None:
    """The file name is the OTHER half of the path a generated row points at,
    and a space in it is not a character a link destination can carry: the
    reader ends the link at the space, so `al pha.md` rows a link to
    `search/projects/-home-u/al`, a path that is not there. Copied, the store
    init has just built fails init's own integrity check with DEAD-LINK.

    A sound file beside it still adopts: this refuses a file, not a run.
    """
    _harness(profile, "-home-u", {"al pha.md": TRAP, "beta.md": TRAP})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert any(
        "al pha.md" in line
        and "no manifest line and no ledger row could carry" in line
        for line in manifest.stdout.splitlines()
    ), manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    assert sorted(p.name for p in adopted.iterdir()) == ["beta.md"]
    ledger = (store / "SEARCH.md").read_text(encoding="utf-8")
    assert "search/projects/-home-u/beta.md" in ledger
    assert "al pha" not in ledger
    # The store init just built still passes the check it will be measured by.
    from memkit import memory_integrity as checker

    entries = []
    for path in sorted((store / "search").rglob("*.md")):
        if path.name in checker.LEDGER_NAMES:
            continue
        front = checker._frontmatter(path)
        value, error = checker._scalar(front.get("description", ""))
        assert error is None, (path, error)
        entries.append(
            (front.get("name") or path.stem, os.path.relpath(path, store), value)
        )
    assert checker._generate(store / "SEARCH.md", entries) == ledger


@pytest.mark.parametrize(
    "key",
    ["-home-u with spaces", "-home-u\ttab", "key(paren)", "key)close"],
)
def test_a_project_key_no_row_could_point_at_is_skipped(profile, key) -> None:
    """The key is half of the path a row points at, and the harness is not the
    only writer of it — `inventory` reads directory NAMES off disk and never
    re-derives them, so the key is whatever any process running as the adopter
    put under `projects/`. Taken raw it lands between the `(` and `)` of a
    generated row: `key)close` ends its own link at `search/projects/key`, a
    space ends it at the space, and a tab is a character no line can carry.
    Each one left the copies on disk and the store failing its own check.

    A sound key beside it still adopts: this refuses a project, not a run.
    """
    _harness(profile, key, {"alpha.md": TRAP})
    _harness(profile, "-home-ok", {"beta.md": TRAP})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "no manifest line and no ledger row could carry" in manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME
    assert not (adopted / key).exists(), sorted(p.name for p in adopted.iterdir())
    ledger = (store / "SEARCH.md").read_text(encoding="utf-8")
    assert "search/projects/-home-ok/beta.md" in ledger
    assert key not in ledger
    # The store init just built still passes the check it will be measured by.
    from memkit import memory_integrity as checker

    entries = []
    for path in sorted((store / "search").rglob("*.md")):
        if path.name in checker.LEDGER_NAMES:
            continue
        front = checker._frontmatter(path)
        value, error = checker._scalar(front.get("description", ""))
        assert error is None, (path, error)
        entries.append(
            (front.get("name") or path.stem, os.path.relpath(path, store), value)
        )
    assert checker._generate(store / "SEARCH.md", entries) == ledger


@pytest.mark.parametrize(
    ("key", "spelled"),
    [("-home-u\ttab", "'-home-u\\ttab'"), ("-home-u\nnl", "'-home-u\\nnl'")],
)
def test_a_skipped_key_is_named_in_a_spelling_the_disk_holds(
    profile, key, spelled
) -> None:
    """A skip line is an instruction to go look at something, so the name it
    prints has to be one `ls` will match. Deleting the byte that made the key
    unusable named `-home-utab`, a directory nobody has.
    """
    _harness(profile, key, {"alpha.md": TRAP})
    _harness(profile, "-home-ok", {"beta.md": TRAP})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert spelled in manifest.stdout, manifest.stdout
    # The escaping is what keeps the byte on one line: still not one forged
    # action line, and still no raw control character in the surface.
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    rendered = plan.render()
    assert "\t" not in rendered and "\r" not in rendered
    ops = {init.CREATE_DIR, init.CREATE_FILE, init.SETTINGS_WRITE,
           init.MERGE_CONFIG, init.VERIFY, init.APPEND_LINE, init.REWRITE_FILE}
    printed = [
        line for line in rendered.splitlines()
        if line[:2] == "  " and line[2:3] != " " and line.split()[0] in ops
    ]
    assert len(printed) == len([
        a for a in plan.pending if not a.group
    ]) + len({a.group for a in plan.pending if a.group})


def test_an_already_redirected_key_is_named_in_a_spelling_the_disk_holds(
    profile, monkeypatch
) -> None:
    """The sibling skip line, for the same reason: it tells the adopter which
    project directory was passed over, so it has to spell one that is there.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    monkeypatch.setenv(
        hook.CONFIG_ENV, str(profile / "home" / ".config" / "memkit" / "memkit.json")
    )
    wired_at = store / "search" / "wired-memories"
    wired_at.mkdir()
    (wired_at / "wired.md").write_text(TRAP, encoding="utf-8")
    wired = profile / "claude-config" / "projects" / "-home-u\tdone"
    wired.mkdir(parents=True)
    (wired / "memory").symlink_to(wired_at)

    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    notes = "\n".join(plan.notes)
    assert "'-home-u\\tdone': already redirected, skipped" in notes, notes


def test_a_key_over_the_harness_limit_adopts_and_says_it_is_hashed(
    profile,
) -> None:
    """`KEY_MAX` governs DERIVING a key from a cwd; adoption reads names the
    harness already chose. A directory the harness wrote over the limit holds
    real memories, so refusing it would leave them behind — but its name is a
    truncated path with an unmeasured hash after it, which is the one thing
    the adopter cannot tell by looking, so the manifest says it.
    """
    at_limit = "-home-u" + "a" * (harness_memory.KEY_MAX - 7)
    over_limit = "-home-u" + "b" * (harness_memory.KEY_MAX + 43)
    assert len(at_limit) == harness_memory.KEY_MAX
    _harness(profile, at_limit, {"alpha.md": TRAP})
    _harness(profile, over_limit, {"beta.md": TRAP})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert f"'{over_limit}': over 200 characters, so the harness truncated" in (
        manifest.stdout
    ), manifest.stdout
    # The control: a key AT the limit is one the harness spelled out in full.
    assert f"'{at_limit}': over" not in manifest.stdout
    # Both adopt, and the store init just built passes its own VERIFY.
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME
    assert (adopted / at_limit / "alpha.md").read_text(encoding="utf-8") == TRAP
    assert (adopted / over_limit / "beta.md").read_text(encoding="utf-8") == TRAP
    assert "Adoption: 2 files" in out.stdout, out.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_project_key_the_check_would_read_as_a_memory_is_skipped(
    profile,
) -> None:
    """A KEY IS A DIRECTORY ENTRY THIS STORE IS GETTING, and the check init
    runs over its own work enumerates memories by suffix. A harness project
    key ending in `.md` — the harness derives keys from a path, and a path
    can end in a file name — became `search/projects/<key>.md/`, which every
    rule in the checker then opened as a file. `--write` opens it too, so the
    documented recovery could not clear it either.
    """
    over_limit = "-home-u" + "b" * (harness_memory.KEY_MAX + 43)
    _harness(profile, "-home-u-notes.md", {"note.md": TRAP})
    _harness(profile, "-home-u", {"ok.md": BARE})
    _harness(
        profile,
        over_limit,
        {"far.md": "---\nname: far\ndescription: over the key limit\n---\n\nb\n"},
    )
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    # Said on the DRY RUN, where the adopter reads it before consenting.
    assert "'-home-u-notes.md': the project key ends in `.md`" in manifest.stdout, (
        manifest.stdout
    )
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME
    assert not (adopted / "-home-u-notes.md").exists(), sorted(
        p.name for p in adopted.iterdir()
    )
    # The controls, in the same run: an ordinary key and one over the harness's
    # own limit both adopt, and the over-limit one still says it is hashed.
    assert "no frontmatter here" in (
        (adopted / "-home-u" / "ok.md").read_text(encoding="utf-8")
    )
    assert (adopted / over_limit / "far.md").is_file()
    assert f"'{over_limit}': over 200 characters, so the harness truncated" in (
        manifest.stdout
    ), manifest.stdout
    # And the store init just built passes the check it runs over it, rather
    # than dying inside it.
    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "IsADirectoryError" not in checked.stderr, checked.stderr


def test_the_manifest_says_where_a_linked_source_directory_resolves(
    profile,
) -> None:
    """A symlinked `memory/` is SUPPORTED, so this is a disclosure and not a
    refusal — and a copy has two ends. The manifest already says where a
    destination really lands; reading through a link is the same question
    asked of the bytes going in, and the group line is the only one that
    names the source at all.
    """
    elsewhere = profile / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "alpha.md").write_text(TRAP, encoding="utf-8")
    linked = profile / "claude-config" / "projects" / "-home-linked"
    linked.mkdir(parents=True)
    (linked / "memory").symlink_to(elsewhere)
    _harness(profile, "-home-plain", {"beta.md": TRAP})

    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    (group,) = [
        line for line in manifest.stdout.splitlines()
        if "-home-linked" in line and " -> " in line
    ]
    assert f"(resolves to {elsewhere})" in group, group
    # The control: nothing resolves anywhere else, so nothing is said.
    (plain,) = [
        line for line in manifest.stdout.splitlines()
        if "-home-plain" in line and " -> " in line
    ]
    assert "resolves to" not in plain, plain
    # Supported, not refused: the copy still happens.
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME
    assert (adopted / "-home-linked" / "alpha.md").read_text() == TRAP
    assert (elsewhere / "alpha.md").read_text() == TRAP


def test_the_stores_own_writes_answer_to_the_stores_containment_root(
    profile,
) -> None:
    """Adoption's copies carried a containment root and init's own writes did
    not, so the guard covered the actions a reviewer looks at and not the ones
    that build the store. A directory swapped for a link to another directory
    is `dir` before and after, so the digest binds nothing about it and the
    write is the first thing that can see it — which is what a containment
    root is for.
    """
    outside = profile / "outside"
    (outside / "search").mkdir(parents=True)
    store = profile / "notes"
    (store / "search").mkdir(parents=True)
    manifest = _dry(profile, "--store", str(store))
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    (store / "search").rmdir()
    (store / "search").symlink_to(outside / "search")
    out = _confirm(profile, _digest_of(manifest), "--store", str(store))
    # 6, not 5: the link is only visible to the write, so earlier actions have
    # already landed — "started and did not finish" is what happened.
    assert out.returncode == init.EXIT_INCOMPLETE, out.stdout + out.stderr
    assert "refused mid-apply (escapes-store)" in out.stderr, out.stderr
    assert sorted(p.name for p in (outside / "search").iterdir()) == []


def test_a_store_built_over_no_link_at_all_is_written_and_checks_green(
    profile,
) -> None:
    """The control for the containment root: the root itself carries none, and
    cannot — `_refuse_escape` names a path against `relpath(path, confine)`,
    which for the root is `.` and never equals its own resolution. Confining
    it to itself refuses every run there is.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    assert (store / "search").is_dir() and not (store / "search").is_symlink()
    assert (store / "hot").is_dir()
    assert (store / "MEMORY.md").exists() and (store / "SEARCH.md").exists()
    # Exit 0 IS the checker's answer: VERIFY is the last action in the plan and
    # a red one is exit 6.
    assert any(a.op == init.VERIFY for a in _plan(profile, store=str(store)).actions)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_adoption_never_lands_an_index_for_a_directory_it_left_behind(
    profile,
) -> None:
    """WHATEVER ADOPTION LANDS IS GREEN. `MEMORY.md` is a ledger name, so it is
    copied byte for byte and nothing regenerates it — rows and all, including
    the rows it carries for siblings adoption itself declined to copy. Landing
    it put a row for a file that is not there into the store, and the integrity
    check init runs over its own work then went red on adoption's own skip
    rules: a failure the feature manufactured out of nothing but its own
    correctness.

    The mechanism is the smallest one that holds it: a ledger is decided after
    the files it indexes, and it is copied only when every one of them was.
    """
    memory = _harness(profile, "-home-u", {
        "MEMORY.md": "# index\n\n- [gone](gone.md) — a row for a file "
                     "adoption skips\n",
        "note.md": TRAP,
    })
    outside = profile / "outside.md"
    outside.write_text("# gone\n\noutside the directory being copied\n",
                       encoding="utf-8")
    (memory / "gone.md").symlink_to(outside)
    # The control, beside it: a directory copied whole keeps its index.
    _harness(profile, "-home-ok", {"MEMORY.md": "# idx\n", "beta.md": BARE})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "-home-u/gone.md: the file is a symlink" in manifest.stdout
    assert "-home-u/MEMORY.md: it is an index" in manifest.stdout, manifest.stdout
    assert "gone.md was left behind" in manifest.stdout, manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    assert sorted(p.name for p in adopted.iterdir()) == ["note.md"]
    whole = store / "search" / init.ADOPT_DIRNAME / "-home-ok"
    assert sorted(p.name for p in whole.iterdir()) == ["MEMORY.md", "beta.md"]
    # The check init just ran on its own work, run again by hand.
    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_named_pipe_in_the_store_does_not_hold_the_dry_run_open(
    profile,
) -> None:
    """A PLANNER READS REGULAR FILES AND NOTHING ELSE.

    `search/` is walked for `*.md` and each one is opened to read its
    frontmatter, and `open` on a FIFO blocks until somebody writes to the other
    end. The dry-run is the turn that exists to be read before anything is
    written, and one that never returns has no turn after it: no manifest, no
    digest, no output at all, and only a signal ends it.

    The 15-second cap is the assertion. A regression here does not fail a
    comparison, it stops returning — so the case has to be able to fail rather
    than hang the suite behind it.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    store = profile / "notes"
    first = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert first.returncode == init.EXIT_OK, first.stdout + first.stderr
    out = _confirm(
        profile, _digest_of(first), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr

    pipe = store / "search" / "pipe.md"
    os.mkfifo(pipe, 0o600)
    assert stat.S_ISFIFO(os.lstat(pipe).st_mode)
    again = subprocess.run(
        [sys.executable, "-m", "memkit.cli", "init", "--dry-run",
         "--store", str(store), "--adopt-auto-memory"],
        capture_output=True, text=True, timeout=15,
        env=dict(
            os.environ,
            HOME=str(profile / "home"),
            XDG_CACHE_HOME=str(profile / "home" / ".cache"),
            CLAUDE_CONFIG_DIR=str(profile / "claude-config"),
        ),
    )
    assert again.returncode == init.EXIT_OK, again.stdout + again.stderr
    assert f"no row: {pipe} is not a regular file" in again.stdout, again.stdout
    # And it is still a pipe: nothing opened it for writing either.
    assert stat.S_ISFIFO(os.lstat(pipe).st_mode)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_file_the_store_already_holds_is_not_called_one_left_behind(
    profile,
) -> None:
    """A SKIP LINE NAMES A FILE BY WHAT ACTUALLY HAPPENED TO IT.

    The index rule asks what the copy loop left outside the store, and it read
    that off the actions the loop produced — so a destination that DIVERGED,
    which produces no action because adoption declines to overwrite, counted as
    a file left behind. It is not: it is in the store, under bytes the adopter
    put there. The line named a state the file was not in, and it withheld an
    index every one of whose rows resolves.
    """
    _harness(profile, "-home-u", {
        "MEMORY.md": "# index\n\n- [beta](beta.md) — the second memory\n",
        "alpha.md": TRAP,
        "beta.md": BARE,
    })
    store = profile / "notes"
    first = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert first.returncode == init.EXIT_OK, first.stdout + first.stderr
    out = _confirm(
        profile, _digest_of(first), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    assert sorted(p.name for p in adopted.iterdir()) == [
        "MEMORY.md", "alpha.md", "beta.md"
    ]

    # The adopter edits what landed, so the next run declines to overwrite it.
    beta = adopted / "beta.md"
    beta.write_text(beta.read_text(encoding="utf-8") + "\ntheirs\n", "utf-8")
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert again.returncode == init.EXIT_OK, again.stdout + again.stderr
    assert f"diverged: {beta} exists and differs" in again.stdout, again.stdout
    assert "left behind" not in again.stdout, again.stdout
    assert "-home-u/MEMORY.md: it is an index" not in again.stdout, again.stdout
    assert beta.read_text(encoding="utf-8").endswith("theirs\n")


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_adoption_never_lands_an_index_rowing_a_memory_that_is_not_there(
    profile,
) -> None:
    """AN INDEX IS ITS ROWS, not the directory listing beside it. A row for a
    memory the adopter deleted by hand names a file the inventory walk never
    enumerated, so a rule computed from the walk's leftovers was satisfied by
    it vacuously: the index landed, the copies and the settings write landed,
    and the run exited 6 with the store failing its own checker. Every re-run
    then said `Nothing to write.` on the dry-run and exited 6 on the confirm —
    a store only a hand-edit of a file memkit itself copied could clear.

    The dry-run and the confirm are asked of the same tree twice here, because
    "nothing left to do" followed by "this did not finish" is the wedge, and
    one turn cannot see it.
    """
    _harness(profile, "-home-u", {
        "MEMORY.md": "# index\n\n- [gone](gone.md) — a memory deleted by hand\n",
        "note.md": TRAP,
    })
    # The control, beside it: an index whose every row resolves is copied.
    _harness(profile, "-home-ok", {
        "MEMORY.md": "# idx\n\n- [beta](beta.md) — a row that resolves\n",
        "beta.md": BARE,
    })
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "-home-u/MEMORY.md: it is an index" in manifest.stdout, manifest.stdout
    assert "gone.md points at no file this store is getting" in manifest.stdout, (
        manifest.stdout
    )
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    assert sorted(p.name for p in adopted.iterdir()) == ["note.md"]
    whole = store / "search" / init.ADOPT_DIRNAME / "-home-ok"
    assert sorted(p.name for p in whole.iterdir()) == ["MEMORY.md", "beta.md"]
    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    # And it converges: what the dry-run says is left is what the confirm does.
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout
    settled = _confirm(
        profile, _digest_of(again), "--store", str(store), "--adopt-auto-memory"
    )
    assert settled.returncode == init.EXIT_OK, settled.stdout + settled.stderr


def _folds_case(where) -> bool:
    """Whether this filesystem hands the same directory to two spellings."""
    probe = where / "case-probe"
    probe.mkdir(exist_ok=True)
    (probe / "a").write_text("", encoding="utf-8")
    return (probe / "A").exists()


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_key_the_store_already_holds_another_spelling_of_diverges(
    profile,
) -> None:
    """A ROW MAY NOT NAME A SPELLING THE DISK DOES NOT HOLD. The destination
    guard is made of path strings, and APFS — the default on macOS — hands the
    same directory to `-home-U` and `-home-u`: every comparison passes, the
    copy lands in the directory that is there and the row names the one that
    is not. The checker calls that an orphan, `memory-integrity --write`
    repairs the row to the on-disk name, and the next init writes the key's
    spelling back — the two rewriting each other every run, which is the
    failure the guard's own reasoning is about, reached with no link at all.

    Three turns, because one is not enough to see it: init, the repair, and a
    dry-run that has to have nothing left to say.
    """
    if not _folds_case(profile):
        pytest.skip("a case-sensitive filesystem tells the two keys apart")
    _harness(profile, "-home-U", {"alpha.md": TRAP})
    _harness(profile, "-home-ok", {"beta.md": BARE})
    store = profile / "notes"
    adopted = store / "search" / init.ADOPT_DIRNAME
    (adopted / "-home-u").mkdir(parents=True)
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "already holds as `-home-u`" in manifest.stdout, manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    assert list((adopted / "-home-u").iterdir()) == [], "the copy went in anyway"
    ledger = (store / "SEARCH.md").read_text(encoding="utf-8")
    assert "-home-U" not in ledger, ledger
    assert "alpha.md" not in ledger, ledger
    # The control, in the same run: a key the disk holds as itself adopts.
    assert "search/projects/-home-ok/beta.md" in ledger, ledger

    # Turn two: the repair the exit code advertises has nothing to repair.
    config = init._resolve_config(doctor.Machine(), None)
    written = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config),
         "--write"],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert written.returncode == 0, written.stdout + written.stderr
    assert (store / "SEARCH.md").read_text(encoding="utf-8") == ledger

    # Turn three: and init has nothing to put back.
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_file_the_store_already_holds_another_spelling_of_diverges(
    profile,
) -> None:
    """THE SAME RULE, ONE LEVEL DOWN. A row points at a memory by writing the
    project key and the file name into one path, and only the key was being
    asked which spelling the disk really holds. A memory renamed `alpha.md` ->
    `Alpha.md` in the harness directory therefore opened the `alpha.md` this
    store already had — read as "already adopted", so nothing was copied —
    while the generated row named `Alpha.md`. Two rows for one file on disk,
    STALE out of the checker, exit 6, and every re-run the same.

    Three turns, because the wedge is what the second and third do: the store
    stays as it is, the ledger carries one row, and the dry-run that follows
    has nothing left to say.
    """
    if not _folds_case(profile):
        pytest.skip("a case-sensitive filesystem tells the two names apart")
    memory = _harness(profile, "-home-u", {"alpha.md": TRAP})
    _harness(profile, "-home-ok", {"beta.md": BARE})
    store = profile / "notes"
    seed = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert seed.returncode == init.EXIT_OK, seed.stdout + seed.stderr
    out = _confirm(
        profile, _digest_of(seed), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    assert [p.name for p in adopted.iterdir()] == ["alpha.md"]

    # The rename the adopter does by hand, in the harness's own directory.
    (memory / "alpha.md").rename(memory / "Alpha.md")
    assert [p.name for p in memory.iterdir()] == ["Alpha.md"]
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "already holds as `alpha.md`" in manifest.stdout, manifest.stdout
    # The control, in the same run: an untouched memory is still already adopted.
    assert "1 already adopted" in manifest.stdout, manifest.stdout
    again = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert again.returncode == init.EXIT_OK, again.stdout + again.stderr
    assert [p.name for p in adopted.iterdir()] == ["alpha.md"], "the copy went in"
    ledger = (store / "SEARCH.md").read_text(encoding="utf-8")
    rows = [line for line in ledger.splitlines() if "projects/-home-u/" in line]
    assert len(rows) == 1, ledger
    assert "Alpha.md" not in ledger, ledger
    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    settled = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in settled.stdout, settled.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_store_membership_is_asked_of_the_config_being_written(profile) -> None:
    """ONE CONFIG DECIDES MEMBERSHIP, and it is the one this run is writing.
    The predicate that keeps adoption off a memory directory already pointed
    INTO a store was asked of the config the session resolved, while `--config`
    named another — so it answered "outside every store" about a directory
    inside the store being written, adoption followed the link, and a second
    copy of every memory landed under a second project key. The store then
    failed its own check with LEDGER-DRIFT, out of the command that made it.

    The same input with the environment aligned was always answered correctly,
    which is the control the two dry-runs below compare: the answer may not
    depend on a variable that names no config this run touches.
    """
    _harness(profile, "-home-real", {"alpha.md": TRAP})
    store = profile / "notes"
    config = profile / "named-by-the-flag.json"
    named = ("--config", str(config), "--store", str(store), "--adopt-auto-memory")
    seed = _dry(profile, *named)
    assert seed.returncode == init.EXIT_OK, seed.stdout + seed.stderr
    out = _confirm(profile, _digest_of(seed), *named)
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr

    # The harness directory an adopter has already pointed into the store.
    linked = profile / "claude-config" / "projects" / "-home-linked"
    linked.mkdir(parents=True)
    (linked / "memory").symlink_to(store / "search" / init.ADOPT_DIRNAME / "-home-real")
    base = dict(
        os.environ,
        HOME=str(profile / "home"),
        XDG_CACHE_HOME=str(profile / "home" / ".cache"),
        CLAUDE_CONFIG_DIR=str(profile / "claude-config"),
    )
    base.pop("MEMKIT_CONFIG", None)
    manifest = _run("--dry-run", *named, env=base)
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "'-home-linked': already redirected, skipped" in manifest.stdout, (
        manifest.stdout
    )
    # The control: the same request with the environment naming that config.
    aligned = _run("--dry-run", *named, env=dict(base, MEMKIT_CONFIG=str(config)))
    assert aligned.stdout == manifest.stdout, manifest.stdout
    applied = _run("--confirm", _digest_of(manifest), *named, env=base)
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-linked").exists()
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300, env=base,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

def _green(profile, store, config=None) -> subprocess.CompletedProcess:
    """The real integrity checker over `store`, through the config init wrote."""
    if config is None:
        config = profile / "home" / ".config" / "memkit" / "memkit.json"
    return subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )


def _canaries(store) -> list:
    return sorted(
        str(p.relative_to(store))
        for p in store.rglob(doctor.CANARY_NAME)
    )


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_memory_directory_linked_at_a_corpus_root_is_already_redirected(
    profile, monkeypatch
) -> None:
    """THE WIRING THE DOCS PRESCRIBE IS AN ANSWER TOO. `docs/STORE.md` tells an
    adopter to move their memories into the store's `search/` and link the
    harness directory AT it — which makes the relation "at", not "inside", so a
    predicate that skipped only "inside" walked back in through the link and
    copied the store's whole corpus, canary included, under a project key. The
    manifest invited it: the same run called a directory that IS the corpus
    root "outside every store".

    Asserted against the disk and the checker rather than the sentence: no
    project directory, one canary, and a store still green.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    monkeypatch.setenv(
        hook.CONFIG_ENV, str(profile / "home" / ".config" / "memkit" / "memkit.json")
    )
    for name in ("wired1.md", "wired2.md"):
        (store / "search" / name).write_text(
            f"---\nname: {name[:-3]}\ndescription: one the store already holds\n"
            "---\n# w\nbody\n",
            encoding="utf-8",
        )
    rowed = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--write",
         "--config", str(profile / "home" / ".config" / "memkit" / "memkit.json")],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert rowed.returncode == 0, rowed.stdout + rowed.stderr
    # The control: the store is green BEFORE adoption, so any red below is
    # adoption's own.
    base = _green(profile, store)
    assert base.returncode == 0, base.stdout + base.stderr

    wired = profile / "claude-config" / "projects" / "-home-wired"
    wired.mkdir(parents=True)
    (wired / "memory").symlink_to(store / "search")

    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "'-home-wired': already redirected, skipped" in manifest.stdout, (
        manifest.stdout
    )
    assert "outside every store" not in manifest.stdout, manifest.stdout
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-wired").exists()
    assert _canaries(store) == [f"search/{doctor.CANARY_NAME}"], _canaries(store)
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_the_store_this_run_creates_is_a_store_the_membership_guard_can_see(
    profile,
) -> None:
    """MEMBERSHIP IS ABOUT THE STORE, NOT ABOUT THE CONFIG THAT NAMES IT. On the
    run that CREATES the config, no configured store contains anything yet, so a
    memory directory already linked into the store being written answered
    "outside every store" and every memory landed a second time — green, silent,
    and doubled, one row each for two copies of one file.

    The store root, not the corpus root, because a directory under the store but
    outside `search/` is somebody's answer as well.
    """
    store = profile / "notes"
    (store / "search").mkdir(parents=True)
    for name in ("alpha.md", "beta.md", "gamma.md"):
        (store / "search" / name).write_text(
            f"---\nname: {name[:-3]}\ndescription: already where it lands\n"
            "---\n# a\nbody\n",
            encoding="utf-8",
        )
    linked = profile / "claude-config" / "projects" / "-home-first"
    linked.mkdir(parents=True)
    (linked / "memory").symlink_to(store / "search")

    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "'-home-first': already redirected, skipped" in manifest.stdout, (
        manifest.stdout
    )
    assert "outside every store" not in manifest.stdout, manifest.stdout
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-first").exists()
    ledger = (store / "SEARCH.md").read_text(encoding="utf-8")
    for name in ("alpha.md", "beta.md", "gamma.md"):
        assert ledger.count(f"(search/{name})") == 1, ledger
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_memory_directory_linked_under_a_store_but_outside_search_is_skipped(
    profile, monkeypatch
) -> None:
    """A TIER IS NOT THE CORPUS ROOT AND IS STILL THE STORE. `_store_relation`
    only ever measures against `search/`, so it answers "" — no relation at all
    — about `<store>/hot`, and the manifest said a directory inside the store
    held memories outside every store.
    """
    store = profile / "notes"
    out = _confirm(profile, _digest_of(_dry(profile, "--store", str(store))),
                   "--store", str(store))
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    monkeypatch.setenv(
        hook.CONFIG_ENV, str(profile / "home" / ".config" / "memkit" / "memkit.json")
    )
    tier = store / "hot"
    tier.mkdir(parents=True, exist_ok=True)
    (tier / "note.md").write_text(TRAP, encoding="utf-8")
    linked = profile / "claude-config" / "projects" / "-home-hot"
    linked.mkdir(parents=True)
    (linked / "memory").symlink_to(tier)

    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "'-home-hot': already redirected, skipped" in manifest.stdout, (
        manifest.stdout
    )
    assert "outside every store" not in manifest.stdout, manifest.stdout
    # The dry run carries the whole assertion here. A memory under a tier with
    # no row for it leaves the store red before adoption runs, so a confirm's
    # exit code would be the fixture's verdict rather than this rule's.
    assert str(store / "search" / init.ADOPT_DIRNAME / "-home-hot") not in (
        manifest.stdout
    ), manifest.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_memory_directory_landing_in_another_store_is_not_copied_into_this_one(
    profile,
) -> None:
    """ANY STORE'S ANSWER IS AN ANSWER, AND THE CONFIG BEING WRITTEN IS WHO IS
    ASKED. Containment in the store this run writes is not the whole question: a
    machine can have two, and a memory directory the adopter already wired into
    the other one is somewhere retrieval already reaches. Copying it here would
    duplicate that store's corpus into this one, under a project key, with a row
    for each copy and nothing saying so.

    Through `--config`, because the second store is only in the config this run
    is writing: asked of the session's instead, the answer is "outside every
    store" about a directory that is inside one.
    """
    config = profile / "named-by-the-flag.json"
    kept = profile / "archive"
    store = profile / "notes"
    env = dict(
        os.environ,
        HOME=str(profile / "home"),
        XDG_CACHE_HOME=str(profile / "home" / ".cache"),
        CLAUDE_CONFIG_DIR=str(profile / "claude-config"),
    )
    env.pop("MEMKIT_CONFIG", None)
    for path in (kept, store):
        named = ("--config", str(config), "--store", str(path))
        seed = _run("--dry-run", *named, env=env)
        assert seed.returncode == init.EXIT_OK, seed.stdout + seed.stderr
        out = _run("--confirm", _digest_of(seed), *named, env=env)
        assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    (kept / "search" / "kept.md").write_text(TRAP, encoding="utf-8")
    rowed = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--write",
         "--config", str(config)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert rowed.returncode == 0, rowed.stdout + rowed.stderr
    elsewhere = profile / "claude-config" / "projects" / "-home-elsewhere"
    elsewhere.mkdir(parents=True)
    (elsewhere / "memory").symlink_to(kept / "search")

    named = ("--config", str(config), "--store", str(store), "--adopt-auto-memory")
    manifest = _run("--dry-run", *named, env=env)
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "'-home-elsewhere': already redirected, skipped" in manifest.stdout, (
        manifest.stdout
    )
    assert "outside every store" not in manifest.stdout, manifest.stdout
    applied = _run("--confirm", _digest_of(manifest), *named, env=env)
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-elsewhere").exists()
    assert sorted(p.name for p in (kept / "search").iterdir()) == sorted(
        [doctor.CANARY_NAME, "kept.md"]
    )

DESC_LINK = (
    "---\nname: desc link\ndescription: see [the plan](plan.md) for the rest\n"
    "---\n# d\nbody\n"
)
BODY_LINK = (
    "---\nname: body link\ndescription: a plain one\n---\n# b\n\n"
    "see [the plan](plan.md)\n"
)
PLAN = "---\nname: plan\ndescription: the plan itself\n---\n# p\nbody\n"


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_description_linking_nowhere_is_skipped_before_the_confirm(
    profile,
) -> None:
    """A DESCRIPTION IS LIFTED INTO A ROW VERBATIM, so a markdown link in one
    becomes a live link in the ledger memkit generates — and the checker
    resolves that link against the STORE ROOT, where it points at nothing. The
    run exited 6 on a store it had just built, `memory-integrity --write`
    regenerated the same row, and every later run said `Nothing to write.` and
    exited 6 again.

    The rule is asked of the plan, so the adopter reads it before consenting.
    """
    _harness(profile, "-home-d", {"note.md": DESC_LINK})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "-home-d/note.md: its description carries a link" in manifest.stdout, (
        manifest.stdout
    )
    assert "plan.md" in manifest.stdout, manifest.stdout
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-d").exists()
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout
    settled = _confirm(
        profile, _digest_of(again), "--store", str(store), "--adopt-auto-memory"
    )
    assert settled.returncode == init.EXIT_OK, settled.stdout + settled.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_description_linking_a_sibling_that_is_adopted_is_still_skipped(
    profile,
) -> None:
    """THE ROW IS NOT WHERE THE MEMORY IS. The description's link resolves
    beautifully beside the memory — and the row carrying it sits in SEARCH.md at
    the store root, three directories up, so the checker reads it from there and
    finds nothing. A rule that asked the question against the destination
    answered yes and left the store red.

    The sibling still lands: only the memory whose description cannot be carried
    is left behind.
    """
    _harness(profile, "-home-s", {"note.md": DESC_LINK, "plan.md": PLAN})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "-home-s/note.md: its description carries a link" in manifest.stdout, (
        manifest.stdout
    )
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-s"
    assert sorted(p.name for p in adopted.iterdir()) == ["plan.md"]
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_body_link_pointing_at_no_adopted_file_is_skipped(profile) -> None:
    """A relative link in the BODY resolves against the destination, and the
    memory it names stayed in the harness directory: the copy landed, nothing
    was skipped, nothing was said, and the checker called the store broken.

    The control beside it is the one that makes the rule narrow: the same link
    with its target adopted alongside is copied, both files.
    """
    _harness(profile, "-home-b", {"note.md": BODY_LINK})
    _harness(profile, "-home-ok", {"note.md": BODY_LINK, "plan.md": PLAN})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "-home-b/note.md: plan.md points at no file" in manifest.stdout, (
        manifest.stdout
    )
    assert "-home-ok/note.md" not in manifest.stdout.split("skipped:")[-1], (
        manifest.stdout
    )
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-b").exists()
    whole = store / "search" / init.ADOPT_DIRNAME / "-home-ok"
    assert sorted(p.name for p in whole.iterdir()) == ["note.md", "plan.md"]
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_wiki_links_in_a_description_and_a_body_are_still_adopted(profile) -> None:
    """WHAT THE CHECKER WARNS ABOUT IS NOT WHAT IT FAILS ON. A dangling
    `[[wikilink]]` is a WARN and leaves the store green, so refusing to adopt a
    memory carrying one would cost the adopter a real memory for nothing.
    """
    _harness(profile, "-home-w", {
        "note.md": (
            "---\nname: wiki\ndescription: see [[the plan]] for the rest\n"
            "---\n# w\n\nand [[another one]] here\n"
        ),
    })
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "0 skipped" in manifest.stdout, manifest.stdout
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-w"
    assert sorted(p.name for p in adopted.iterdir()) == ["note.md"]
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr

FENCE_OPEN = (
    "---\nname: fenced\ndescription: a note about shell commands\n---\n# f\n\n"
    "run this:\n\n```bash\nmemkit doctor\n"
)
FENCE_CLOSED = (
    "---\nname: fenced\ndescription: a note about shell commands\n---\n# f\n\n"
    "run this:\n\n```bash\nmemkit doctor\nsee [the plan](plan.md)\n```\n\ndone.\n"
)
FENCE_LONGER = (
    "---\nname: fenced\ndescription: a note quoting a fence\n---\n# f\n\n"
    "````\n```\nnot a close\n````\n\ndone.\n"
)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_memory_that_ends_mid_example_is_skipped_before_the_confirm(
    profile,
) -> None:
    """A CODE FENCE OPENED AND NOT CLOSED IS AN ERROR TO THE CHECKER, and it
    needs no link anywhere in the memory: a note about shell commands that ends
    mid-example was copied, the confirm exited 6 on the store it had just built,
    and every re-run said there was nothing to write and exited 6 again.

    The skip chain had the text in hand and asked it about `tier:` lines, name
    length and link syntax — this is the same question, asked where those are.
    """
    _harness(profile, "-home-f", {"note.md": FENCE_OPEN})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "-home-f/note.md: it opens a code fence on line 9" in manifest.stdout, (
        manifest.stdout
    )
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    assert not (store / "search" / init.ADOPT_DIRNAME / "-home-f").exists()
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    again = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert "Nothing to write" in again.stdout, again.stdout
    settled = _confirm(
        profile, _digest_of(again), "--store", str(store), "--adopt-auto-memory"
    )
    assert settled.returncode == init.EXIT_OK, settled.stdout + settled.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_closed_fence_is_adopted_and_so_is_the_example_link_inside_it(
    profile,
) -> None:
    """WHAT THE CHECKER MASKS, ADOPTION MAY NOT REFUSE A MEMORY FOR. A closed
    fence is ordinary prose, and the checker blanks every line inside one before
    it reads links — so a memory quoting `[the plan](plan.md)` in an example is
    a memory with no dead link in it, whatever a rule reading the raw bytes
    would say.

    Beside it, a fence closed by a LONGER run of the same character: a close is
    at least as long as its opening, so the shorter run inside this one opens
    nothing and closes nothing.
    """
    _harness(profile, "-home-c", {"note.md": FENCE_CLOSED})
    _harness(profile, "-home-l", {"note.md": FENCE_LONGER})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "0 skipped" in manifest.stdout, manifest.stdout
    applied = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert applied.returncode == init.EXIT_OK, applied.stdout + applied.stderr
    for key in ("-home-c", "-home-l"):
        adopted = store / "search" / init.ADOPT_DIRNAME / key
        assert sorted(p.name for p in adopted.iterdir()) == ["note.md"]
    checked = _green(profile, store)
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_an_adopted_copy_is_never_more_readable_than_its_original(
    profile,
) -> None:
    """A COPY OF A PRIVATE NOTE IS AS PRIVATE AS THE NOTE. The copy path asked
    for `0644` while the function it asked has its own `0600`, so a memory the
    adopter had deliberately chmod'd `0600` came back readable by everyone on
    the machine — published by the command whose whole subject is where private
    memories live, and disclosed nowhere: the only mode the manifest named was
    `0700` for the cache directory.

    Both sources are here because the widening was invisible from the `0644`
    one: they land at the same mode now, and it is the narrower one.
    """
    memory = _harness(profile, "-home-u", {"open.md": TRAP, "private.md": BARE})
    (memory / "open.md").chmod(0o644)
    (memory / "private.md").chmod(0o600)
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert "Each copy lands mode 0600" in manifest.stdout, manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    for name in ("open.md", "private.md"):
        assert stat.S_IMODE(os.stat(adopted / name).st_mode) == 0o600, name
    assert stat.S_IMODE(os.stat(store / "SEARCH.md").st_mode) == 0o600
    # The originals are not touched on any path, mode included.
    assert stat.S_IMODE(os.stat(memory / "open.md").st_mode) == 0o644
    assert stat.S_IMODE(os.stat(memory / "private.md").st_mode) == 0o600


def test_the_write_refuses_a_name_it_could_not_create_a_temporary_for(
    tmp_path,
) -> None:
    """ONE RULE, ASKED WHERE THE LONGER NAME IS ACTUALLY MADE. Adoption's
    planner skips a name this long before it can be planned, so nothing memkit
    builds reaches this — and that is what it is for: a caller that grew a path
    the planner never measured gets a decision rather than an ENAMETOOLONG
    traceback out of the middle of an apply.
    """
    room = init._NAME_MAX_BYTES - init._TMP_SUFFIX_BYTES
    over = tmp_path / ("o" * (room - len(".md") + 1) + ".md")
    with pytest.raises(init.Refusal) as raised:
        init._write_atomically(str(over), "body\n")
    assert raised.value.name == "name-too-long", raised.value
    assert not list(tmp_path.iterdir()), "something was written anyway"
    fits = tmp_path / ("f" * (room - len(".md")) + ".md")
    init._write_atomically(str(fits), "body\n")
    assert fits.read_text(encoding="utf-8") == "body\n"


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_name_the_write_could_not_land_is_skipped_at_the_dry_run(profile) -> None:
    """NO NAME THE DRY-RUN APPROVES FAILS TO LAND FOR ITS LENGTH. The write
    creates `<name>.<pid>.tmp` beside the file and renames over it, so the
    name the plan measured is not the longest name the write makes. A memory
    name a little under the limit therefore passed the dry-run, failed at
    apply time, and left a store the checker called broken with nothing copied
    into it — and every re-run did the same.

    The lengths here are derived from the rule's own constants, because a
    literal would pass whatever the rule became.
    """
    room = init._NAME_MAX_BYTES - init._TMP_SUFFIX_BYTES
    longest = "n" * (room - len(".md")) + ".md"
    over = "o" * (room - len(".md") + 1) + ".md"
    # Chars comfortably under the bound, bytes over it: the rule counts bytes.
    wide = "é" * (room // 2) + ".md"
    assert len(wide) < room < len(wide.encode()), (len(wide), len(wide.encode()))
    _harness(profile, "-home-u", {longest: TRAP, over: TRAP, wide: TRAP})
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    assert f"{over}: the file name is" in manifest.stdout, manifest.stdout
    assert f"{wide}: the file name is" in manifest.stdout, manifest.stdout
    assert f"{longest}: the file name is" not in manifest.stdout, manifest.stdout
    assert "2 skipped" in manifest.stdout, manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, out.stdout + out.stderr
    adopted = store / "search" / init.ADOPT_DIRNAME / "-home-u"
    assert [p.name for p in adopted.iterdir()] == [longest]
    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_red_integrity_check_still_redirects_the_harness(
    profile, monkeypatch, capsys
) -> None:
    """VERIFY is not the last action and its answer is about a store that is
    already on disk. Returning the moment the checker is unhappy left the
    memories copied into the store AND the harness still writing outside it —
    the half-state the redirect exists to end, and reachable from any of the
    inputs that turn the check red. The code is still INCOMPLETE, the
    checker's own output is still printed, and the manifest order is
    untouched: what changed is only when the code is returned.
    """
    _harness(profile, "-home-u", {"ok.md": TRAP})
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    plan = _plan(
        profile, store=str(profile / "notes"), adopt_auto_memory=True
    )
    ops = [a.op for a in plan.pending]
    assert ops.index(init.VERIFY) < ops.index(init.SETTINGS_WRITE), ops
    monkeypatch.setattr(
        init,
        "_run_checker",
        lambda m, c: (1, "ORPHAN: ./hot/x.md — no row in MEMORY.md"),
    )
    assert init.apply_plan(machine, plan, config) == init.EXIT_INCOMPLETE
    assert "ORPHAN: ./hot/x.md" in capsys.readouterr().err
    settings = json.loads(
        (profile / "claude-config" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings["autoMemoryDirectory"].startswith(str(profile / "notes"))


def test_the_two_shapes_of_exit_six_say_in_their_output_which_one_they_are(
    profile, monkeypatch, capsys
) -> None:
    """EVERY EXIT CODE'S SENTENCE HAS TO BE TRUE OF EVERY RUN THAT RETURNS IT.

    Deferring the red checker's code to the end of the loop made 6 the answer
    for a run that performed every action in its manifest as well as for one
    that genuinely stopped partway, and the published table said only the
    second. The code cannot tell them apart — one number, two states — so the
    output has to, and it does: the finished run says so and names the files
    the check is red on, the stopped run names the refusal that stopped it and
    says nothing about a finished manifest.
    """
    _harness(profile, "-home-u", {"ok.md": TRAP})
    machine = doctor.Machine()
    config = init._resolve_config(machine, None)
    store = profile / "notes"

    # Shape one: everything performed, and then the check is red — on a file
    # this run wrote and on one it did not.
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    mine = next(a.path for a in plan.pending if a.path.endswith("memkit-canary.md"))
    theirs = store / "search" / "not-from-here.md"

    def red(_machine, _config):
        store.mkdir(parents=True, exist_ok=True)
        (store / "search").mkdir(parents=True, exist_ok=True)
        theirs.write_text("# theirs\n", encoding="utf-8")
        return 1, (
            "[FAIL] ./ (0 hot, 2 search, hot ledger 248b)\n"
            "  DESC-BAD: ./search/memkit-canary.md — description empty\n"
            "  ORPHAN: ./search/not-from-here.md — no row in SEARCH.md"
        )

    monkeypatch.setattr(init, "_run_checker", red)
    assert init.apply_plan(machine, plan, config) == init.EXIT_INCOMPLETE
    finished = capsys.readouterr().err
    assert "every action in the manifest was performed" in finished, finished
    assert f"{mine} — this run wrote it" in finished, finished
    assert f"{theirs} — this run did not write it" in finished, finished
    assert "no re-run will change" in finished, finished
    assert "refused mid-apply" not in finished, finished
    # And it really did finish: the last action is the settings write.
    assert json.loads(
        (profile / "claude-config" / "settings.json").read_text(encoding="utf-8")
    )[harness_memory.DIRECTORY_KEY]

    # Shape two: a run that stopped partway. Same exit code, and its output
    # claims nothing about a manifest it did not finish. `_run_checker` is put
    # back by name rather than with `monkeypatch.undo()`, which would also undo
    # the profile fixture's own patches — the two share one instance, and a
    # case that lost `$CLAUDE_CONFIG_DIR` halfway through would plan against
    # the machine running the suite.
    monkeypatch.setattr(init, "_run_checker", lambda _m, _c: (0, ""))
    settings = profile / "claude-config" / "settings.json"
    stopped = _plan(profile, auto_dream_off=True, store=str(profile / "other"))
    settings.write_text('{"theme": "moved under it"}', encoding="utf-8")
    assert init.apply_plan(machine, stopped, config) == init.EXIT_INCOMPLETE
    partway = capsys.readouterr().err
    assert "refused mid-apply (changed-underfoot)" in partway, partway
    assert "every action in the manifest was performed" not in partway, partway
    assert settings.read_text(encoding="utf-8") == '{"theme": "moved under it"}'


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_red_finding_that_carries_a_line_number_is_attributed_too(
    profile,
) -> None:
    """THE PARSER IS FED THE CHECKER'S OWN BYTES, not a hand-typed line.

    The two cases above stub the checker out and hand it a block spelled the
    way no line-numbered rule spells one. Every rule with a line to point at
    writes the path as `file.md:8`, which resolves to nothing on disk, so the
    finding fell out of the attribution and took the recovery paragraph with
    it — on DEAD-LINK, the canonical red an adoption run earns. A store the
    real checker really is red about is the only thing that pins the spelling.
    """
    _harness(
        profile,
        "-home-u",
        {
            "alpha.md": (
                "---\nname: alpha\ndescription: one adopted memory\n---\n\n"
                "See [[not-a-memory-anywhere]] for the rest.\n"
            )
        },
    )
    store = profile / "notes"
    theirs = store / "search" / "not-from-here.md"
    theirs.parent.mkdir(parents=True)
    theirs.write_text(
        "---\nname: theirs\ndescription: a file this run does not write\n---\n\n"
        "See [the other one](./nowhere-at-all.md) for the rest.\n",
        encoding="utf-8",
    )
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout + manifest.stderr
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_INCOMPLETE, out.stdout + out.stderr
    err = out.stderr
    mine = store / "search" / init.ADOPT_DIRNAME / "-home-u" / "alpha.md"
    assert mine.is_file(), err
    # The checker's own spelling, line number and all, on both files.
    assert "DEAD-LINK: ./search/not-from-here.md:" in err, err
    assert "DANGLING-WIKILINK: ./search/" in err, err
    # And the attribution, which is what the spelling used to cost.
    assert f"{mine} — this run wrote it" in err, err
    assert f"{theirs} — this run did not write it" in err, err
    assert "what landed has moved the old one" in err, err
    assert "no re-run will change" in err, err

    # A rule that puts no em dash after the path is attributed as well, again
    # on the checker's own bytes: a ledger row for a file that is not there.
    ledger = store / "SEARCH.md"
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + "- [gone](search/gone.md) — a row for a file that is not there\n",
        encoding="utf-8",
    )
    config = init._resolve_config(doctor.Machine(), None)
    checked = subprocess.run(
        [sys.executable, "-m", "memkit.memory_integrity", "--config", str(config)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, HOME=str(profile / "home")),
    )
    stale = [
        line
        for line in checked.stdout.splitlines()
        if line.strip().startswith("STALE:")
    ]
    assert stale, checked.stdout + checked.stderr
    assert init._files_the_checker_names("\n".join(stale), str(store)) == [str(ledger)]


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the integrity checker's own floor"
)
def test_a_destination_the_adopter_edited_is_named_as_one_this_run_left_alone(
    profile,
) -> None:
    """The contract case, end to end: adoption copies and never overwrites, so
    a destination the adopter has edited is `diverged` and nothing is written
    for it — and a memory with no frontmatter is exactly what the integrity
    checker calls an orphan. The run therefore does everything it said it
    would and exits 6 anyway.

    Re-running the two turns cannot clear that, because the next manifest has
    nothing to say about a file adoption declined. The output has to say so, or
    the recovery the exit code advertises sends the adopter round a loop.
    """
    _harness(profile, "-home-u", {"note.md": BARE})
    store = profile / "notes"
    first = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert first.returncode == init.EXIT_OK, first.stdout + first.stderr
    landed = _confirm(
        profile, _digest_of(first), "--store", str(store), "--adopt-auto-memory"
    )
    assert landed.returncode == init.EXIT_OK, landed.stdout + landed.stderr
    dest = store / "search" / init.ADOPT_DIRNAME / "-home-u" / "note.md"
    dest.write_text("# Home note\n\nedited, and still no frontmatter\n", "utf-8")

    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    assert manifest.returncode == init.EXIT_OK, manifest.stdout
    assert f"diverged: {dest}" in manifest.stdout, manifest.stdout
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_INCOMPLETE, out.stdout + out.stderr
    assert "the integrity checker is not happy" in out.stderr, out.stderr
    assert "every action in the manifest was performed" in out.stderr, out.stderr
    assert f"{dest} — this run did not write it" in out.stderr, out.stderr
    assert "no re-run will change" in out.stderr, out.stderr
    # The file the run declined to write is byte-unchanged, and the source too.
    assert dest.read_text(encoding="utf-8").endswith("still no frontmatter\n")
    source = profile / "claude-config" / "projects" / "-home-u" / "memory" / "note.md"
    assert source.read_text(encoding="utf-8") == BARE


def test_a_description_taken_from_a_file_name_cannot_end_its_own_line(
    profile,
) -> None:
    """The stem feeds two frontmatter lines and only one of them was cleaned.
    A newline in it split the copied block, gave the checker `description: foo`
    and put a literal line break inside the ledger row's link.

    Asked of the normaliser directly: the planner skips such a file outright
    now, and a guard nothing reaches is a guard that stops being true.
    """
    written, rule = init._normalise("plain body\n", "foo\nbar")
    assert "\n" not in written.split("\n---", 1)[0].partition("description:")[2]
    assert "description: foobar" in written
    assert "name: foobar" in written
    assert "the file name" in rule


def test_an_unclosed_frontmatter_opener_still_has_a_first_heading(
    profile,
) -> None:
    """`---` with no closer opens nothing. Read as an unterminated block it
    left the body empty, so a file whose next line is a heading took its
    description from the file name instead — with the heading right there.
    """
    _harness(profile, "-home-u", {"u.md": "---\nname: u\n\n# A Real Heading\n\nb\n"})
    store = profile / "notes"
    plan = _plan(profile, store=str(store), adopt_auto_memory=True)
    (copy,) = [a for a in plan.actions if a.path.endswith("u.md")]
    assert "its first heading" in copy.note
    assert "description: A Real Heading" in copy.content


@pytest.mark.parametrize(
    "name,files",
    [
        (
            "a-name-that-would-end-its-own-link",
            {"ok.md": TRAP, "evil.md": (
                "---\nname: x](hot/forged.md) — forged\n"
                "description: real desc\n---\n\nbody\n"
            )},
        ),
        (
            "a-name-that-is-quoted-and-hostile",
            {"ok.md": TRAP, "q.md": (
                '---\nname: "x](y) z"\ndescription: quoted and hostile\n'
                "---\n\nbody\n"
            )},
        ),
        (
            "a-name-of-nothing-but-link-syntax",
            {"ok.md": TRAP,
             "n.md": "---\nname: ()[]\ndescription: only syntax\n---\n\nbody\n"},
        ),
        (
            "a-file-named-with-link-syntax-and-no-frontmatter",
            {"ok.md": TRAP, "x](y).md": "body with no frontmatter\n"},
        ),
        (
            "a-continued-description-being-replaced",
            {"ok.md": TRAP, "c.md": (
                "---\nname: c\ndescription: >\n  a folded value\n"
                "  that runs on\n---\n\nbody\n"
            )},
        ),
    ],
)
def test_the_ledger_is_still_the_checkers_on_a_hostile_name(
    profile, name, files
) -> None:
    """THE FIXPOINT, over the inputs the sanitising was added for.

    `memory_integrity` needs 3.12 and this module answers to the 3.9 floor, so
    the rules are restated rather than shared — and what makes the restatement
    safe is evidence: the checker's own generator, over the tree init made,
    produces the bytes init wrote. A label init cleaned in its own row and not
    in the file would pass every assertion above and diverge at the next
    `--write`.
    """
    from memkit import memory_integrity as checker

    _harness(profile, "-home-u", files)
    store = profile / "notes"
    manifest = _dry(profile, "--store", str(store), "--adopt-auto-memory")
    out = _confirm(
        profile, _digest_of(manifest), "--store", str(store), "--adopt-auto-memory"
    )
    assert out.returncode == init.EXIT_OK, name + out.stdout + out.stderr
    ledger = store / "SEARCH.md"
    entries = []
    for path in sorted((store / "search").rglob("*.md")):
        if path.name in checker.LEDGER_NAMES:
            continue
        front = checker._frontmatter(path)
        value, error = checker._scalar(front.get("description", ""))
        assert error is None, (path, error)
        entries.append(
            (front.get("name") or path.stem, os.path.relpath(path, store), value)
        )
    assert checker._generate(ledger, entries) == ledger.read_text(encoding="utf-8")
    # Non-vacuity: the hostile file really is in the ledger this compared.
    assert len(entries) >= 2, entries


def test_an_existing_search_ledger_keeps_the_preamble_somebody_wrote(profile) -> None:
    """SEARCH.md was written from the canary alone, so an init over a store
    that already held memories replaced a ledger of their rows with a ledger of
    one — every one of them an orphan at the next check and unreachable from
    the file that indexes them. The preamble is kept for the same reason
    `--write` keeps it: a sentence somebody wrote is not a setup command's to
    replace.
    """
    store = profile / "notes"
    (store / "search").mkdir(parents=True)
    (store / "search" / "mine.md").write_text(
        "---\nname: mine\ndescription: a memory that was here first\n---\n\nbody\n",
        encoding="utf-8",
    )
    (store / "SEARCH.md").write_text(
        "# my own words\n\nkeep this line.\n\n## Index\n\n- [stale](search/gone.md) — x\n",
        encoding="utf-8",
    )
    plan = _plan(profile, store=str(store))
    (ledger,) = [a for a in plan.actions if a.path == str(store / "SEARCH.md")]
    assert ledger.content.startswith("# my own words\n\nkeep this line.\n\n## Index")
    rows = _rows_of(ledger.content)
    assert rows["search/mine.md"] == "a memory that was here first"
    assert "search/gone.md" not in rows
    assert "search/" + doctor.CANARY_NAME in rows


def test_auto_memory_off_writes_one_boolean_and_then_has_nothing_to_do(
    profile,
) -> None:
    """The switch that really stops the harness writing — `--auto-dream-off`
    stops background consolidation and nothing else. Idempotent, so a second
    run over a machine already set this way is an empty manifest rather than a
    refusal.
    """
    plan = _plan(profile, auto_memory_off=True)
    (action,) = [a for a in plan.actions if a.op == init.SETTINGS_WRITE]
    assert json.loads(action.content) == {"autoMemoryEnabled": False}
    (profile / "claude-config" / "settings.json").write_text(
        action.content, encoding="utf-8"
    )
    again = _plan(profile, auto_memory_off=True)
    (second,) = [a for a in again.actions if a.op == init.SETTINGS_WRITE]
    assert second.redundant
    assert not [a for a in again.writes if a.op == init.SETTINGS_WRITE]


@pytest.mark.parametrize(
    "flag", ["adopt_auto_memory", "auto_memory_off"]
)
def test_a_settings_scope_that_will_not_parse_is_not_a_scope_saying_nothing(
    profile, flag
) -> None:
    """Every gate around auto-memory asks a scope what it declares, and reads
    the answer out of `scope.data` — which a file that would not parse arrives
    with empty, exactly as a file declaring nothing does. So one trailing
    comma in `settings.local.json` turned off the refusals that stand between
    an adopter and a redirect they did not ask for, and init planned the
    settings write anyway. The harness cannot read that file either.
    """
    _harness(profile, "-home-u", {"note.md": TRAP})
    checkout = profile / "project" / ".claude"
    checkout.mkdir(parents=True)
    local = checkout / "settings.local.json"
    # What the gates would have refused, one comma short of parsing.
    local.write_text(
        '{"autoMemoryDirectory": "~/elsewhere", "autoMemoryEnabled": true,}',
        encoding="utf-8",
    )
    refusal = _refuses(profile, "settings-unreadable", **{flag: True})
    assert "local settings" in refusal.message
    assert str(local) in refusal.message
    # A plain init reads no scope for these keys and is not refused by it.
    assert _plan(profile).actions


def test_auto_memory_off_will_not_promise_what_a_higher_scope_overrules(
    profile, monkeypatch
) -> None:
    """The flag writes the ONE scope every other scope outranks, and the
    manifest line under it says the harness will then neither read nor write
    auto-memory. With `true` declared in a scope the harness reads first, that
    promise is one the write cannot keep: the adopter would get exit 0 and a
    harness still writing. Refused by name instead, naming the scope and its
    file, which is the rule `--adopt-auto-memory` has been asked in this file
    since it was written.

    `false` up there is a different case: the feature really is off, so the
    write converges the user scope and a note says which scope decided it.
    """
    checkout = profile / "project" / ".claude"
    checkout.mkdir(parents=True)
    local = checkout / "settings.local.json"
    local.write_text(json.dumps({"autoMemoryEnabled": True}), encoding="utf-8")
    refusal = _refuses(profile, "auto-memory-outranked", auto_memory_off=True)
    assert "local settings" in refusal.message
    assert str(local) in refusal.message

    # The administrator's scope, the one the adopter cannot answer for.
    local.unlink()
    managed = profile / "managed"
    managed.mkdir()
    monkeypatch.setattr(doctor, "_managed_dir", lambda: str(managed))
    (managed / doctor.MANAGED_SETTINGS_NAME).write_text(
        json.dumps({"autoMemoryEnabled": True}), encoding="utf-8"
    )
    assert "managed settings" in _refuses(
        profile, "auto-memory-outranked", auto_memory_off=True
    ).message

    # Already off above: not a refusal, and the note says who decided it.
    (managed / doctor.MANAGED_SETTINGS_NAME).write_text(
        json.dumps({"autoMemoryEnabled": False}), encoding="utf-8"
    )
    plan = _plan(profile, auto_memory_off=True)
    assert [a for a in plan.writes if a.op == init.SETTINGS_WRITE]
    assert any(
        "managed settings" in note and "already off" in note
        for note in plan.notes
    ), plan.notes


def test_adopting_and_switching_off_are_not_one_request(profile) -> None:
    """Opposite answers to one question, so argparse refuses the pair as the
    usage error it is — exit 2, which is the code the dispatcher and the
    published table already promise for one."""
    out = _dry(profile, "--adopt-auto-memory", "--auto-memory-off")
    assert out.returncode == init.EXIT_USAGE, out.stdout + out.stderr
    assert "not allowed with" in out.stderr
    assert out.stdout == ""


def test_the_auto_dream_flag_no_longer_claims_to_stop_the_writing(profile) -> None:
    """It stops BACKGROUND CONSOLIDATION only: memories are still written,
    which is why it is not the switch that turns the feature off. The help and
    the note said otherwise, and an adopter who read either of them came away
    with two memory systems and a flag they thought had closed one.
    """
    parser = argparse.ArgumentParser()
    init.add_arguments(parser)
    help_text = parser.format_help()
    assert "--auto-memory-off" in help_text
    assert "--adopt-auto-memory" in help_text
    dream = [
        line for line in help_text.splitlines() if "BACKGROUND CONSOLIDATION" in line
    ]
    assert dream, help_text
    note = " ".join(_plan(profile, auto_dream_off=True).notes)
    assert "BACKGROUND CONSOLIDATION" in note
    assert "--auto-memory-off is the flag that stops the writing" in note
