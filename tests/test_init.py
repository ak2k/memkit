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
import json
import os
import subprocess
import sys

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
    return tmp_path


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
            with open(path, "rb") as f:
                out[os.path.relpath(path, root)] = f.read()
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
    before = _plan(profile).digest
    config = profile / "home" / ".config" / "memkit" / "memkit.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    assert _plan(profile).digest != before


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
    for action in plan.actions:
        if action.op == init.CREATE_DIR:
            os.makedirs(action.path, exist_ok=True)
        elif action.op in (init.CREATE_FILE, init.REWRITE_FILE):
            os.makedirs(os.path.dirname(action.path), exist_ok=True)
            with open(action.path, "w", encoding="utf-8") as f:
                f.write(action.content)
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
    first = init._canary_nonce("/a/config.json", "/b/store")
    assert first == init._canary_nonce("/a/config.json", "/b/store")
    assert first != init._canary_nonce("/a/config.json", "/c/store")
    assert first != init._canary_nonce("/z/config.json", "/b/store")


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
