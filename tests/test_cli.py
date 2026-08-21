"""Unit tests for the `memkit` subcommand dispatcher.

Driven as a SUBPROCESS wherever the exit status is the claim. argv routing and
argparse's own refusals are only real from outside the process — in-process
`main([...])` cannot see the console script's wiring, and that wiring is half
of what this file is about: a skill invokes `memkit <subcommand>` and branches
on what comes back.

The dispatcher does nothing yet on purpose. What it has to get right in the
meantime is the answer it gives about the things it cannot do, because the
reader is an agent deciding whether to find another way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from memkit import cli


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    # Through `-m`, not the installed console script: the suite runs from a
    # checkout whose entry points may be stale, and this exercises the same
    # `cli.main` the script's `cli()` calls.
    return subprocess.run(
        [sys.executable, "-m", "memkit.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env if env is not None else os.environ,
    )


def test_the_dispatcher_exit_codes_are_the_numbers_a_skill_hardcodes() -> None:
    """The literals, spelled out.

    Every other case here says `cli.EXIT_*`, which puts the constant on both
    sides of the assertion and stays green through any renumbering. The reader
    that matters is a skill's shell branch, which cannot import anything.

    Deliberately NOT memory-recall's vocabulary, even where the numbers
    coincide: these are separate binaries with separate contracts, and the
    codes are defined in cli.py so that adding one here never has to make
    sense for a search command.
    """
    assert (cli.EXIT_USAGE, cli.EXIT_NOT_IN_BUILD) == (2, 4)


def test_a_bare_invocation_asks_for_a_subcommand_and_says_so_in_its_status() -> None:
    """Usage on stderr, non-zero out. A bare `memkit` asked for something and
    got nothing done, and exiting 0 with usage text is how a caller records a
    successful setup step that never ran."""
    out = _run()
    assert out.returncode == cli.EXIT_USAGE
    assert out.stdout == ""
    assert "usage: memkit" in out.stderr


def test_a_missing_subcommand_is_not_the_same_answer_as_a_wrong_one() -> None:
    """Three states, two codes, and the split is the point.

    "You invoked this wrongly" and "you invoked it correctly and it is not
    here yet" want opposite next moves: the first says try different arguments,
    the second says stop trying and use the fallback. They shared exit 2, so an
    agent could not tell them apart without parsing prose, and the cheap retry
    loop is the one it reaches for.

    2 is argparse's own and cannot be reassigned, so it keeps the usage
    meaning and the state that is not a usage error takes a code of its own.
    """
    usage = _run()
    unknown = _run("frobnicate")
    pending = _run("doctor")

    assert usage.returncode == unknown.returncode == cli.EXIT_USAGE
    assert pending.returncode == cli.EXIT_NOT_IN_BUILD
    assert pending.returncode != usage.returncode


def test_help_names_the_subcommands_that_do_not_exist_yet() -> None:
    """A name that is simply absent reads as "this tool cannot do that", which
    sends an agent off to invent a way; a name that says NOT IN THIS BUILD YET
    sends it to the fallback. The two answers are far apart in consequence and
    only one of them is true."""
    out = _run("--help")
    assert out.returncode == 0
    for name in cli._PENDING:
        assert name in out.stdout
    assert "NOT IN THIS BUILD YET" in out.stdout


def test_both_help_surfaces_carry_the_fallback_not_just_the_name() -> None:
    """`--help` is the cheaper probe and the one an agent tries before running
    anything, so a help page listing two names and no way forward is a dead
    end reached in preference to the refusal that has one. Both levels: the
    top-level epilog, and each subcommand's own `--help`, which argparse
    answers from the SUBparser — bare usage and exit 0, the one shape that
    reads as "this works".

    Exit 0 stays, because that is what `--help` means. The fix is what it
    says, not what it returns.
    """
    top = _run("--help")
    for name, (summary, template) in cli._PENDING.items():
        meanwhile = cli._meanwhile(template)
        assert meanwhile in " ".join(top.stdout.split()), name

        own = _run(name, "--help")
        assert own.returncode == 0
        collapsed = " ".join(own.stdout.split())
        assert summary in collapsed
        assert "NOT IN THIS BUILD YET" in collapsed
        assert meanwhile in collapsed


def test_the_refusal_names_the_search_binary_this_install_actually_has(
    tmp_path,
) -> None:
    """A plugin adopter has `memkit-recall` on PATH and no `memory-recall` at
    all, so a hardcoded fallback sends them to a binary that does not exist —
    which reads as the tool being broken rather than as a name being wrong.

    The command comes from the same `search_cli` the hook's truncation notice
    advertises, so the refusal and the pointer block name one binary by
    construction. The literal is the no-config fallback and nothing else.
    """
    config = tmp_path / "memkit.json"
    config.write_text(
        json.dumps(
            {
                "schema": 1,
                "roots": {"home": {"kind": "path", "path": str(tmp_path)}},
                "stores": [],
                "search_cli": "memkit-recall --search",
            }
        )
    )
    env = dict(os.environ, MEMKIT_CONFIG=str(config))
    out = _run("doctor", env=env)
    assert out.returncode == cli.EXIT_NOT_IN_BUILD
    assert "memkit-recall --search" in out.stderr
    assert "memkit-recall --debug-config" in out.stderr
    assert "memory-recall" not in out.stderr

    # With nothing configured there is nothing better to say, so the shipped
    # default stands.
    bare = dict(os.environ)
    bare.pop("MEMKIT_CONFIG", None)
    assert "memory-recall --search" in _run("doctor", env=bare).stderr


def test_a_config_that_cannot_be_read_never_takes_the_refusal_down(
    tmp_path,
) -> None:
    """A refusal is what a caller reaches when something is already wrong, so
    it is the last surface that may fail — and `--help` reaches it too, which
    this file's own comment calls the cheapest probe there is.

    Reading the config to improve a message is worth doing; it is not worth an
    exit code that says the tool is broken. A `search_cli` of the wrong TYPE
    is the shape that found this: harmless while the only consumer f-stringed
    it, an AttributeError out of `memkit --help` the moment something split it.
    """
    config = tmp_path / "hostile.json"
    for raw in (
        '{"schema": 1, "roots": {}, "stores": [], "search_cli": 123}',
        '{"schema": 99}',
        "{ not json at all",
    ):
        config.write_text(raw)
        env = dict(os.environ, MEMKIT_CONFIG=str(config))

        top = _run("--help", env=env)
        assert top.returncode == 0, (raw, top.stderr)
        # The shipped default, since nothing better could be resolved — and a
        # command that exists either way.
        assert "memory-recall --search" in top.stdout, raw

        refusal = _run("doctor", env=env)
        assert refusal.returncode == cli.EXIT_NOT_IN_BUILD, (raw, refusal.stderr)
        assert "memory-recall --search" in refusal.stderr, raw


def test_the_refusal_quotes_the_exit_codes_from_their_source() -> None:
    """The prose named 3 and 1 as literals while importing the constants that
    define them, so a renumbering would have left the advice confidently
    wrong — and this is advice an agent follows without a second source."""
    from memkit.memory_prompt_recall import EXIT_INERT, EXIT_NO_MATCH

    refusal = _run("doctor").stderr
    assert f"Exit {EXIT_INERT} there" in refusal
    assert f"exit {EXIT_NO_MATCH} means" in refusal


def test_an_unknown_subcommand_is_refused_against_the_list_of_real_ones() -> None:
    out = _run("frobnicate")
    assert out.returncode != 0
    assert "invalid choice" in out.stderr
    # The refusal enumerates what does exist, which is the whole reason to let
    # argparse own this rather than matching the name ourselves.
    assert "doctor" in out.stderr and "init" in out.stderr


def test_a_subcommand_that_has_not_landed_refuses_with_somewhere_to_go() -> None:
    for name, (summary, template) in cli._PENDING.items():
        out = _run(name)
        assert out.returncode == cli.EXIT_NOT_IN_BUILD, out.stderr
        assert out.stdout == ""
        assert f"`{name}`" in out.stderr and "not in this build" in out.stderr
        # Both halves, verbatim: the summary says what the caller wanted and
        # the fallback says what to do instead. A refusal carrying only the
        # first is one an agent can act on by giving up. The fallback is
        # rendered, not the raw template — the command it names is a config
        # value, so the two can only be compared after resolution.
        assert summary in out.stderr
        assert cli._meanwhile(template) in out.stderr
        assert "{search" not in out.stderr, "an unfilled placeholder reached a user"


def test_flags_a_pending_subcommand_will_take_do_not_preempt_its_refusal() -> None:
    """`memkit doctor --json` is the invocation a skill will make, and on this
    build it has to reach the message explaining that doctor is not here.

    parse_known_args is what makes that true. Under parse_args the run dies on
    an unrecognised argument, and an agent that meets argparse's usage text
    there learns that `--json` is the problem — which is both false and the
    kind of false that gets worked around rather than reported.
    """
    plain = _run("doctor")
    flagged = _run("doctor", "--json")
    assert flagged.returncode == plain.returncode
    assert flagged.stderr == plain.stderr
    assert "unrecognized arguments" not in flagged.stderr


def test_every_pending_name_is_a_name_the_parser_accepts() -> None:
    """The help text and the dispatch read the same dict, so a subcommand
    cannot be advertised and then refused as an invalid choice — the failure a
    hand-written help string produces the moment one of the two is edited."""
    for name in cli._PENDING:
        assert cli.main([name]) == cli.EXIT_NOT_IN_BUILD


def test_a_handler_takes_over_from_the_pending_message(monkeypatch) -> None:
    """The seam a later unit extends: one entry in _HANDLERS and the pending
    refusal stops being reachable for that name. Asserted here so the routing
    is covered before there is anything to route — otherwise the first real
    subcommand lands on a dispatch nothing has ever exercised.
    """
    seen: list[list[str]] = []
    monkeypatch.setitem(cli._HANDLERS, "doctor", lambda extra: seen.append(extra) or 0)
    assert cli.main(["doctor", "--json"]) == 0
    # The arguments this parser did not consume reach the handler untouched,
    # which is what lets a subcommand own its own flags.
    assert seen == [["--json"]]
