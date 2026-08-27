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
    return hook._trusted_which("git")


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace(
        dry_run=True,
        confirm=None,
        store=None,
        config=None,
        wire_claude_md=False,
        auto_dream_off=False,
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
    assert calls == ["Machine"], calls

    # And behaviourally, at both sites, so the shape above is not the only
    # thing standing.
    for target in ("_resolve_config", "build_plan"):
        monkeypatch.setattr(
            init,
            target,
            lambda *a, **k: (_ for _ in ()).throw(init.Refusal("synthetic", "no")),
        )
        assert init.run(_args()) == init.EXIT_REFUSED
        monkeypatch.undo()


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


def test_no_checker_route_is_refused_rather_than_half_completed(profile, monkeypatch):
    """A seeded memory whose ledger nobody checked is a store the checker calls
    broken. Half-completing is worse than not starting."""
    monkeypatch.setenv(doctor.ROUTE_ENV, "none")
    monkeypatch.setenv(doctor.ROUTE_CMD_ENV, "")
    refusal = _refuses(profile, "no-checker-route")
    assert "uvx" in refusal.message


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
    assert frozenset({"autoDreamEnabled"}) == init.SETTINGS_KEYS_INIT_MAY_WRITE
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


def test_a_crash_between_two_mutations_leaves_a_journal_that_describes_it(
    profile, monkeypatch
) -> None:
    """The whole reason the record is written at the mutation. A batch would
    describe the runs that did not need describing and nothing else."""
    plan = _plan(profile)
    machine = doctor.Machine()
    real = init._write_atomically
    calls = []

    def explode(path, content, mode=0o600, expect=None):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real(path, content, mode, expect)

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

    def explode(path, content, mode=0o600, expect=None):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real(path, content, mode, expect)

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
    init._perform(machine, journal, action, config)
    with open(config, encoding="utf-8") as f:
        blob = json.loads(f.read())
    assert {s["id"] for s in blob["stores"]} == {"theirs", "mine"}


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


def test_the_settings_write_re_reads_under_the_lock(profile) -> None:
    """init is invoked from inside a live session, so the harness owns and
    actively writes that file for the whole run — and the settings write is the
    LAST action, after an integrity-checker subprocess that may take minutes.
    Anything the harness wrote in between was silently lost."""
    machine = doctor.Machine()
    settings = profile / "claude-config" / "settings.json"
    settings.write_text('{"theme": "dark"}', encoding="utf-8")
    plan = _plan(profile, auto_dream_off=True, store=str(profile / "notes"))
    (action,) = [a for a in plan.actions if a.op == init.SETTINGS_WRITE]
    # The harness writes while the plan is in flight.
    settings.write_text(
        json.dumps({"theme": "dark", "enabledPlugins": {"other@x": True}}),
        encoding="utf-8",
    )
    journal = init.Journal(str(machine.state_dir), plan.digest)
    os.makedirs(machine.state_dir, mode=0o700, exist_ok=True)
    init._perform(machine, journal, action, init._resolve_config(machine, None))
    blob = json.loads(settings.read_text())
    assert blob["autoDreamEnabled"] is False
    assert blob["enabledPlugins"] == {"other@x": True}, blob


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


def test_the_checker_command_is_not_taken_from_the_session_path(
    profile, monkeypatch
) -> None:
    """The wrapper hands the checker over as a space-joined string, and its
    first word came from the wrapper's own unfiltered `command -v`.

    So the word is untrusted whatever its shape. A bare name resolves against
    the entries no checkout can steer, and where no trusted candidate answers
    the call REFUSES: `--confirm`'s permission prompt shows `memkit init
    --confirm <digest>` and never this argv, so consent for the command was
    never consent for whatever the session's PATH happened to supply.
    """
    hostile = profile / "elsewhere" / "prog"
    hostile.parent.mkdir(parents=True, exist_ok=True)
    hostile.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hostile.chmod(0o755)
    shim = profile / "project" / "node_modules" / ".bin"
    shim.mkdir(parents=True)
    (shim / "uvx").symlink_to(hostile)
    # One the checkout supplies, FIRST, and one outside it after — so the
    # answer is a choice between two real candidates rather than a question
    # about what this machine happens to have installed.
    theirs = profile / "usr-bin"
    theirs.mkdir()
    trusted = theirs / "uvx"
    trusted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    trusted.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}:{theirs}:{os.environ['PATH']}")
    monkeypatch.setenv("MEMKIT_CHECKER_ROUTE", "uvx")
    monkeypatch.setenv("MEMKIT_CHECKER_CMD", "uvx --from memkit memory-integrity")
    ran: list = []
    real = init.subprocess.run

    def watched(argv, *a, **kw):
        ran.append(list(argv))
        return real([sys.executable, "-c", "raise SystemExit(0)"], *a, **kw)

    monkeypatch.setattr(init.subprocess, "run", watched)
    init._run_checker(doctor.Machine(), str(profile / "memkit.json"))
    monkeypatch.setattr(init.subprocess, "run", real)
    assert ran, "the checker was never invoked, so this proves nothing"
    assert ran[0][0] == str(trusted), ran[0]

    # And with the trusted candidate gone, nothing runs at all. The fallback
    # that used to stand here ran the checkout's own program on the write turn.
    monkeypatch.setenv("PATH", str(shim))
    ran.clear()
    monkeypatch.setattr(init.subprocess, "run", watched)
    code, detail = init._run_checker(doctor.Machine(), str(profile / "memkit.json"))
    monkeypatch.setattr(init.subprocess, "run", real)
    assert not ran, ran
    assert code == 1
    assert "no trusted checker route" in detail, detail


def test_the_checker_refuses_an_absolute_program_the_environment_named(
    profile, monkeypatch
) -> None:
    """An absolute path is the SHAPE `$MEMKIT_CHECKER_CMD` always carries.

    `bin/lib/common.sh` resolves the python route with `command -v` and
    exports the answer, so every real invocation of this function on the
    plugin channel arrives with an already-absolute first word — and anything
    else that can write this process's environment arrives the same way. A
    rule that let an absolute word through unexamined therefore governed
    nothing on the path it was written for.
    """
    outside = profile / "elsewhere"
    outside.mkdir(exist_ok=True)
    hostile = outside / "fakepy"
    marker = profile / "PWNED-checker.txt"
    hostile.write_text(
        f"#!/bin/sh\necho pwned > {marker}\nexit 0\n", encoding="utf-8"
    )
    hostile.chmod(0o755)
    # NOT on PATH: the variable is the whole of how this program was named.
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("MEMKIT_CHECKER_ROUTE", "python")
    monkeypatch.setenv("MEMKIT_CHECKER_CMD", f"{hostile} -m memkit.memory_integrity")
    code, detail = init._run_checker(doctor.Machine(), str(profile / "memkit.json"))
    assert code == 1, (code, detail)
    assert "no trusted checker route" in detail, detail
    assert not marker.exists(), marker.read_text()


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
