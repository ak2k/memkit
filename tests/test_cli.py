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

import subprocess
import sys

from memkit import cli
from memkit.memory_prompt_recall import EXIT_ERROR


def _run(*args: str) -> subprocess.CompletedProcess:
    # Through `-m`, not the installed console script: the suite runs from a
    # checkout whose entry points may be stale, and this exercises the same
    # `cli.main` the script's `cli()` calls.
    return subprocess.run(
        [sys.executable, "-m", "memkit.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_bare_invocation_asks_for_a_subcommand_and_says_so_in_its_status() -> None:
    """Usage on stderr, non-zero out. A bare `memkit` asked for something and
    got nothing done, and exiting 0 with usage text is how a caller records a
    successful setup step that never ran."""
    out = _run()
    assert out.returncode != 0
    assert out.stdout == ""
    assert "usage: memkit" in out.stderr


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


def test_an_unknown_subcommand_is_refused_against_the_list_of_real_ones() -> None:
    out = _run("frobnicate")
    assert out.returncode != 0
    assert "invalid choice" in out.stderr
    # The refusal enumerates what does exist, which is the whole reason to let
    # argparse own this rather than matching the name ourselves.
    assert "doctor" in out.stderr and "init" in out.stderr


def test_a_subcommand_that_has_not_landed_refuses_with_somewhere_to_go() -> None:
    for name, (summary, meanwhile) in cli._PENDING.items():
        out = _run(name)
        assert out.returncode == EXIT_ERROR, out.stderr
        assert out.stdout == ""
        assert f"`{name}`" in out.stderr and "not in this build" in out.stderr
        # Both halves, verbatim: the summary says what the caller wanted and
        # the fallback says what to do instead. A refusal carrying only the
        # first is one an agent can act on by giving up.
        assert summary in out.stderr
        assert meanwhile in out.stderr


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
        assert cli.main([name]) == EXIT_ERROR


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
