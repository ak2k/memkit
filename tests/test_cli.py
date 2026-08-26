"""Unit tests for the `memkit` subcommand dispatcher.

Driven as a SUBPROCESS wherever the exit status is the claim. argv routing and
argparse's own refusals are only real from outside the process — in-process
`main([...])` cannot see the console script's wiring, and that wiring is half
of what this file is about: a skill invokes `memkit <subcommand>` and branches
on what comes back.

Both subcommands route now, and `_PENDING` is empty. It is KEPT, and so are
the cases below that drive it through a stand-in name: M3 adds triage, and an
exit code an agent may already have branched on must not change meaning
between releases. A mechanism with no live claimant and no test is one that
rots until the release that needs it.
"""

from __future__ import annotations

import json
import os
import pathlib
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

    assert usage.returncode == unknown.returncode == cli.EXIT_USAGE
    # Nothing is pending in this build, so the third state is exercised
    # through the mechanism M3 will use rather than through a live name.
    assert cli._pending_code("triage") == cli.EXIT_NOT_IN_BUILD
    assert cli.EXIT_NOT_IN_BUILD != cli.EXIT_USAGE


def test_help_names_the_subcommands_that_do_not_exist_yet(monkeypatch) -> None:
    """A name that is simply absent reads as "this tool cannot do that", which
    sends an agent off to invent a way; a name that says NOT IN THIS BUILD YET
    sends it to the fallback. The two answers are far apart in consequence and
    only one of them is true.

    In-process against a stand-in name, because nothing is pending in this
    build. The machinery is what M3 lands on, and a help surface that had
    stopped rendering it would be discovered by whoever adds triage.
    """
    monkeypatch.setitem(
        cli._PENDING, "triage", ("classify a store's memories", "meanwhile: read them")
    )
    rendered = cli._parser().format_help()
    collapsed = " ".join(rendered.split())
    assert "triage" in collapsed
    assert "NOT IN THIS BUILD YET" in collapsed
    assert "Not in this build yet:" in collapsed


def test_the_pending_heading_renders_nothing_when_nothing_is_pending() -> None:
    """An empty "Not in this build yet:" reads as a list that failed to
    render, on the cheapest probe an agent makes."""
    assert cli._PENDING == {}
    assert "Not in this build yet" not in cli._parser().format_help()


def test_both_help_surfaces_carry_the_fallback_not_just_the_name(monkeypatch):
    """`--help` is the cheaper probe and the one an agent tries before running
    anything, so a help page listing two names and no way forward is a dead
    end reached in preference to the refusal that has one. Both levels: the
    top-level epilog, and each subcommand's own `--help`, which argparse
    answers from the SUBparser — bare usage and exit 0, the one shape that
    reads as "this works".

    Exit 0 stays, because that is what `--help` means. The fix is what it
    says, not what it returns.
    """
    # Through a STAND-IN and the in-process parser, because nothing is pending
    # in this build: iterating `_PENDING` skipped every assertion in this body
    # and the case passed without exercising anything. Its two siblings were
    # updated for exactly this reason and this one was not, which left the
    # pending subparser's `description=` — the whole refusal M3 will land on —
    # with no coverage at all. A subprocess cannot see the monkeypatch, so the
    # parser is built here.
    summary = "classify a store's memories"
    template = 'meanwhile: `{search_config}`, and `{search} "<terms>"`'
    monkeypatch.setitem(cli._PENDING, "triage", (summary, template))
    meanwhile = cli._meanwhile(template)

    top = " ".join(cli._parser().format_help().split())
    assert meanwhile in top

    # argparse keeps the subparsers on the one positional it added, and the
    # only way to reach a subparser's own help is through it.
    sub = None
    for action in cli._parser()._actions:
        found = getattr(action, "choices", None)
        if isinstance(found, dict) and "triage" in found:
            sub = found["triage"]
    assert sub is not None, "no subparser for the stand-in name"
    collapsed = " ".join(sub.format_help().split())
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

    The surface is the top-level description, which `--help` prints and which
    every other invocation of this binary builds before it does anything:
    `_parser()` calls `_meanwhile` for it, so a config that could take this
    down would take down every `memkit` command including `--help`.
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
    helped = _run("--help", env=env)
    assert helped.returncode == 0
    assert "memkit-recall --search" in helped.stdout
    assert "memory-recall" not in helped.stdout

    # With nothing configured there is nothing better to say, so the shipped
    # default stands.
    bare = dict(os.environ)
    bare.pop("MEMKIT_CONFIG", None)
    assert "memory-recall --search" in _run("--help", env=bare).stdout


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
        # The three below are the ones that actually exercise the fallback.
        # Everything above is a shape `_config` already folds into None, so
        # `_search_cli` returns the default without raising and the guard is
        # never reached — which is how the guard came to be mutation-green
        # while being the half that rescues `--help`. These raise straight
        # THROUGH that swallow, because the config reader only converts
        # ConfigError and a store that is not a mapping is an AttributeError.
        '{"schema": 1, "roots": {}, "stores": [123]}',
        '{"schema": 1, "roots": {}, "stores": "notalist"}',
        '{"schema": 1, "roots": {}, "stores": [], "citations": {"roots": 5}}',
    ):
        config.write_text(raw)
        env = dict(os.environ, MEMKIT_CONFIG=str(config))

        top = _run("--help", env=env)
        assert top.returncode == 0, (raw, top.stderr)
        # The shipped default, since nothing better could be resolved — and a
        # command that exists either way.
        assert "memory-recall --search" in top.stdout, raw

        # And every other entry point, which builds the same description
        # before it parses a single argument.
        listed = _run("doctor", "--help", env=env)
        assert listed.returncode == 0, (raw, listed.stderr)


def test_the_help_quotes_the_search_clis_exit_codes_from_their_source() -> None:
    """The prose named 3 and 1 as literals while importing the constants that
    define them, so a renumbering would have left the advice confidently
    wrong — and this is advice an agent follows without a second source.

    The two tables collide on both 1 and 4, so the sentence about the OTHER
    binary's table has to be built from that binary's own constants: the
    dispatcher's `EXIT_NO_RUNTIME` renders the same digit as the hook's
    `EXIT_NO_MATCH` and means the opposite thing, which is the collision the
    sentence exists to warn about and the one a shared constant would hide.
    """
    from memkit.memory_prompt_recall import (
        EXIT_CANNOT_START,
        EXIT_INERT,
        EXIT_NO_MATCH,
    )

    epilog = " ".join(_run("--help").stdout.split())
    assert f"there {EXIT_NO_MATCH} means nothing matched" in epilog, epilog
    assert f"{EXIT_CANNOT_START} means it could not start" in epilog, epilog
    assert f"Its {EXIT_INERT} has no counterpart" in epilog, epilog


def test_an_unknown_subcommand_is_refused_against_the_list_of_real_ones() -> None:
    out = _run("frobnicate")
    assert out.returncode != 0
    assert "invalid choice" in out.stderr
    # The refusal enumerates what does exist, which is the whole reason to let
    # argparse own this rather than matching the name ourselves.
    assert "doctor" in out.stderr and "init" in out.stderr


def test_a_subcommand_that_has_not_landed_refuses_with_somewhere_to_go(
    monkeypatch, capsys
) -> None:
    """Driven through a stand-in, because nothing is pending in this build and
    the mechanism is what M3 lands on."""
    summary = "classify a store's memories"
    template = 'meanwhile: `{search_config}`, and `{search} "<terms>"`'
    monkeypatch.setitem(cli._PENDING, "triage", (summary, template))
    assert cli.main(["triage"]) == cli.EXIT_NOT_IN_BUILD
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "`triage`" in captured.err and "not in this build" in captured.err
    # Both halves, verbatim: the summary says what the caller wanted and the
    # fallback says what to do instead. A refusal carrying only the first is
    # one an agent can act on by giving up. The fallback is rendered, not the
    # raw template — the command it names is a config value, so the two can
    # only be compared after resolution.
    assert summary in captured.err
    assert cli._meanwhile(template) in captured.err
    assert "{search" not in captured.err, "an unfilled placeholder reached a user"


def test_an_unrecognised_flag_on_a_landed_subcommand_is_the_callers_mistake():
    """The other side of the parse_known_args trade, now that both subcommands
    declare their own flags.

    While they were listed and unimplemented, `memkit doctor --json` had to
    reach the message explaining the absence rather than dying on an
    unrecognised argument — an agent that met argparse's usage text there
    learned that `--json` was wrong, which is false. With both landed, an
    unrecognised flag really IS the caller's mistake and argparse is the right
    thing to say so.
    """
    out = _run("doctor", "--jsn")
    assert out.returncode == cli.EXIT_USAGE
    assert "unrecognized arguments" in out.stderr
    # And the flag that IS declared is parsed rather than passed through.
    assert _run("doctor", "--json").returncode in (0, 1)


def test_every_pending_name_is_a_name_the_parser_accepts(monkeypatch) -> None:
    """The help text and the dispatch read the same dict, so a subcommand
    cannot be advertised and then refused as an invalid choice — the failure a
    hand-written help string produces the moment one of the two is edited."""
    monkeypatch.setitem(cli._PENDING, "triage", ("classify memories", "meanwhile: -"))
    for name in cli._PENDING:
        assert cli.main([name]) == cli.EXIT_NOT_IN_BUILD


def test_a_handler_takes_over_from_the_pending_message(monkeypatch) -> None:
    """The seam a later unit extends: one entry in _HANDLERS and the pending
    refusal stops being reachable for that name. Asserted here so the routing
    is covered before there is anything to route — otherwise the first real
    subcommand lands on a dispatch nothing has ever exercised.
    """
    seen: list = []
    monkeypatch.setitem(cli._HANDLERS, "doctor", lambda args: seen.append(args) or 0)
    assert cli.main(["doctor", "--json"]) == 0
    # The flags the SUBPARSER declared arrive parsed, which is what lets
    # `memkit doctor --help` list them.
    (args,) = seen
    assert args.as_json is True
    assert args.subcommand == "doctor"


def test_the_diagnostic_form_keeps_the_config_the_search_form_carries(
    tmp_path, monkeypatch
) -> None:
    """On the plugin channel the advertised command carries `--config <path>`,
    and that is the half that makes it runnable rather than merely spelled
    correctly: the agent's Bash tool gets the plugin's `bin/` and none of the
    plugin's environment.

    Rebuilding the diagnostic form from the command's first word dropped it,
    which handed an agent a command that answers `inert` on a serving install
    — the one conclusion the interpolation exists to prevent.
    """
    from memkit import memory_prompt_recall as hook

    config = tmp_path / "memkit.json"
    config.write_text(
        json.dumps({"schema": 1, "roots": {}, "stores": [],
                    "search_cli": "memory-recall --search"})
    )
    monkeypatch.setenv(hook.PLUGIN_ENV, "1")
    monkeypatch.setenv(hook.CONFIG_ENV, str(config))
    hook._use_config(None)
    try:
        rendered = cli._meanwhile("{search} :: {search_config}")
    finally:
        hook._use_config(None)
    assert f"memkit-recall --config {config} --search" in rendered, rendered
    assert f"memkit-recall --config {config} --debug-config" in rendered, rendered


def test_a_whitespace_only_search_cli_is_replaced_before_it_is_split(
    tmp_path, monkeypatch
) -> None:
    """An EMPTY `search_cli` is falsy, so the config applies the default and
    never reaches this code. A whitespace-only one is truthy, so the config
    keeps it — and this runs while the parser is being built, which makes an
    IndexError here a `memkit --help` that does not run at all."""
    from memkit import memory_prompt_recall as hook

    config = tmp_path / "ws.json"
    config.write_text(
        json.dumps({"schema": 1, "roots": {}, "stores": [], "search_cli": "   "})
    )
    monkeypatch.delenv(hook.PLUGIN_ENV, raising=False)
    monkeypatch.setenv(hook.CONFIG_ENV, str(config))
    hook._use_config(None)
    try:
        rendered = cli._meanwhile("{search} :: {search_config}")
    finally:
        hook._use_config(None)
    assert "memory-recall --search" in rendered, rendered
    assert "memory-recall --debug-config" in rendered, rendered


def test_every_exit_code_the_help_advertises_is_one_the_process_returns():
    """The table `--help` renders and the numbers the commands return have to
    be the same objects, not two constants that happen to agree.

    `cli.EXIT_REFUSED` rendered the table while the process returned
    `cli_init.EXIT_REFUSED`, with nothing asserting they were equal — so one
    side could move and `--help` would advertise a code the command never
    returns, the skill's table (pinned to the other side) would disagree with
    the binary's own help, and no test would go red.
    """
    import ast

    from memkit import cli_doctor, cli_init

    # By the SOURCE, not by `is`. Both sides are small ints, which CPython
    # interns — so `cli.EXIT_REFUSED is cli_init.EXIT_REFUSED` is True for two
    # independent literals and the drift this exists to catch is invisible to
    # it. What has to hold is that there is one definition.
    tree = ast.parse(pathlib.Path(cli.__file__).read_text())
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigned[target.id] = node.value
    for name in ("EXIT_REFUSED", "EXIT_INCOMPLETE"):
        value = assigned.get(name)
        assert isinstance(value, ast.Attribute), (name, ast.dump(value or ast.Pass()))
        assert value.attr == name, (name, value.attr)
    assert cli.EXIT_REFUSED == cli_init.EXIT_REFUSED
    assert cli.EXIT_INCOMPLETE == cli_init.EXIT_INCOMPLETE
    # And the one every module spells for itself, because argparse owns it.
    assert cli.EXIT_USAGE == cli_init.EXIT_USAGE == cli_doctor.EXIT_USAGE == 2

    rendered = " ".join(_run("--help").stdout.split())
    for code in (0, cli.EXIT_NO_RUNTIME, cli.EXIT_USAGE, cli.EXIT_NOT_IN_BUILD,
                 cli.EXIT_REFUSED, cli.EXIT_INCOMPLETE):
        assert f"/ {code} " in rendered or f": {code} " in rendered, code
