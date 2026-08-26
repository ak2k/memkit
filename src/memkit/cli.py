"""`memkit <subcommand>` — the dispatcher the setup and diagnosis commands hang off.

A fourth console script rather than more flags on the search CLI, because
these are a different job. That one searches; what an adopter needs
before they can search is a way to set the thing up and a way to ask whether
it worked, and both of those write or read things that have nothing to do with
retrieval. Keeping them apart is also what lets a skill pre-approve one exact
argument shape: an `allowed-tools` entry that had to cover setup, diagnosis and
arbitrary-directory search in one command is an entry that pre-approves reading
any directory.

The dispatch is a dict. A subcommand is one line in `_HANDLERS`, one line in
`_SUBCOMMANDS` and a module beside this one; nothing else about the routing
moves. A name in `_PENDING` with no handler is what a caller meets for a
subcommand that is declared and has not shipped — see `_pending` for why the
names are listed at all before they do anything.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from memkit import cli_doctor
from memkit.memory_prompt_recall import (
    EXIT_CANNOT_START,
    EXIT_INERT,
    EXIT_NO_MATCH,
    _search_cli,
    _self_name,
)

# This binary's exit codes, and deliberately NOT the search CLI's. They are
# different commands with different jobs, so sharing a vocabulary would mean
# every future addition to one had to make sense for the other; a skill
# branching on `memkit doctor` should read this table and nothing else.
#
# 2 is argparse's and cannot be changed: it is what the parser itself returns
# for a usage error or an unknown subcommand, and a handler cannot intercept
# that. So 2 keeps exactly that meaning here — "you invoked this wrongly" — and
# the state that is not a usage error at all gets its own code. A subcommand
# that IS declared and simply has not shipped is not the caller's mistake, and
# an agent that cannot tell the two apart retries with different arguments
# forever.
EXIT_USAGE = 2
EXIT_NOT_IN_BUILD = 4
# Emitted by the plugin's `bin/memkit` wrapper and never by this module: it is
# what a caller gets when the dispatcher could not be STARTED — no interpreter
# resolved, or the plugin payload is incomplete. Declared here anyway, because
# the table an agent branches on has to be complete to be usable, and a code
# that appears only in a shell script is a code nobody can look up. Deliberately
# not EXIT_USAGE: "you invoked this wrongly" sends a caller to retry with
# different arguments against a machine that cannot run memkit at all.
EXIT_NO_RUNTIME = 1
# The reciprocal note, because the two tables SWAP these two numbers and only
# one direction was written down. `memkit-recall` exits 4 for the three
# conditions this exits 1 for — and 1 on its table means "the stores were
# searched and nothing matched", the code that tells an agent to stop looking.
# That is the dangerous direction of the collision.

# What `memkit` will route, with the one-line summary `--help` shows and the
# thing to reach for meanwhile. Listed before they exist on purpose: the reader
# of `--help` here is usually an agent deciding what to do next, and a name
# that is simply absent reads as "this tool cannot do that" — which sends it to
# invent a way — while a name that says "not in this build" sends it to the
# fallback named beside it. The two answers are not close together in
# consequence, and only one of them is true.
#
# The `meanwhile` half is a template rather than a string: `{search}` is filled
# at print time from the resolved config, because the command this install
# actually exposes is a config value. A plugin adopter has `memkit-recall` on
# PATH and no `memory-recall` at all, so a hardcoded fallback would send them
# to a binary that does not exist — which is worse than no fallback, since it
# reads as the tool being broken rather than as a name being wrong. The exit
# codes are interpolated from the constants for the same reason they are
# constants at all.
_PENDING: dict[str, tuple[str, str]] = {
    "init": (
        "create a store and wire this machine up to it",
        "meanwhile: write the config by hand — the schema and a worked example "
        "are in the project README under Config. `{search_config}` says what "
        'this install resolved once you have, and `{search} "<terms>"` whether '
        "the stores answer",
    ),
}


def _meanwhile(template: str) -> str:
    """Fill a `meanwhile` template with the commands this install exposes.

    `_search_cli()` is what the hook's own truncation notice advertises, so the
    refusal and the pointer block name the same command by construction —
    including the `--config <path>` prefix the plugin channel adds, without
    which the command is inert in the Bash tool the agent would run it from.

    Nothing about the config may take this down. A refusal is what a caller
    reaches when something is already wrong, so it is the last surface that
    should be able to fail — and it is reached from `--help`, the cheapest
    probe there is. Reading the config to improve a message is worth doing;
    it is not worth an exit code that says the tool is broken, so any failure
    resolving it falls back to the shipped default rather than propagating.
    """
    try:
        search = _search_cli()
    except Exception:
        # `_self_name()` rather than the module constant: this branch is
        # reachable whenever the config raises something `load_config` does not
        # convert to `ConfigError` — `json.load` on a deeply nested file raises
        # `RecursionError`, which is neither — and falling back to the shipped
        # default there hands a plugin adopter a binary their channel does not
        # ship. It reads only `os.environ` and cannot itself raise.
        search = f"{_self_name()} --search"
    # The debug form is the search command with its MODE FLAG swapped, not the
    # binary re-derived: on the plugin channel the command carries
    # `--config <path>`, which is the half that makes it runnable from the
    # agent's Bash tool, and rebuilding from the first word alone dropped it.
    #
    # The swap is CONDITIONAL on the suffix being there, and the fallback is
    # what it replaced: `re.sub` on a string that does not end in `--search`
    # returns it unchanged, so the diagnostic form would silently become the
    # search command — a command that needs terms, handed to an agent as the
    # thing to run to see what resolved.
    if search.endswith("--search"):
        search_config = search[: -len("--search")] + "--debug-config"
    else:
        # TOTAL, because this runs while the parser is being BUILT: `_parser()`
        # calls `_meanwhile` for its description, so anything that raises here
        # takes down every `memkit` invocation including `--help`, the cheapest
        # probe an adopter runs. A whitespace-only `search_cli` is truthy, so
        # `Config` keeps it and `split()` returns nothing to index.
        head = search.split()
        search_config = f"{head[0] if head else _self_name()} --debug-config"
    return template.format(search=search, search_config=search_config)

# Subcommand -> the function that runs it, given the namespace this parser
# produced and whatever it could not consume.
#
# The second argument exists only while `_PENDING` is non-empty. A pending
# subcommand declares no flags, so `memkit init --dry-run` has to survive
# parsing to reach the message explaining why it does nothing — which forces
# `parse_known_args` on the top level, which in turn means a real subcommand
# has to refuse its own leftovers rather than letting argparse do it. When the
# last pending name goes, both halves of that collapse into `parse_args`.
_HANDLERS: dict[str, Callable[[argparse.Namespace, list[str]], int]] = {
    "doctor": cli_doctor.run,
}

# Subcommand -> its one-line summary, its help epilog, and the function that
# declares its flags. Split from `_HANDLERS` because the parser is built on
# every invocation including `--help`, and a subcommand whose flags were
# declared by the handler itself would have a `--help` that could not list
# them — which is the shape `_PENDING` already produces on purpose and must
# not survive into a subcommand that works.
_SUBCOMMANDS: dict[
    str, tuple[str, str, Callable[[argparse.ArgumentParser], None]]
] = {
    "doctor": (cli_doctor.SUMMARY, cli_doctor.EPILOG, cli_doctor.add_arguments),
}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="memkit",
        # Through `_meanwhile` like every other command this prints. Hardcoded,
        # this line named a binary the plugin channel does not ship on the
        # cheapest probe an agent runs, and no config value could reach it.
        description=_meanwhile(
            "Set up and diagnose a memkit installation. To search one, see "
            '`{search} "<terms>"`.'
        ),
        # The fallbacks belong on the top-level help too, not only on the
        # refusal an agent reaches by running the subcommand: `memkit --help`
        # is the cheaper probe and the one tried first, and it was answering
        # with two names and no way forward.
        epilog="Not in this build yet:\n"
        + "\n".join(f"  {n}: {_meanwhile(m)}" for n, (_, m) in _PENDING.items())
        + f"\n\nExit codes: 0 ok / {EXIT_NO_RUNTIME} memkit could not start at "
        f"all (stderr names what is missing) / {EXIT_USAGE} usage error or "
        f"unknown subcommand / {EXIT_NOT_IN_BUILD} the subcommand exists but "
        "is not in this build."
        "\nThe search CLI's table is its own and swaps these two: there "
        f"{EXIT_NO_MATCH} means nothing matched and {EXIT_CANNOT_START} "
        f"means it could not start. Its {EXIT_INERT} has no counterpart here "
        "at all: there was nothing to search — no config, or no store on disk "
        "and in scope for this directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Which build am I on — the precondition for reading any other answer this
    # binary gives, and until now no command anywhere had it. Rendered from the
    # same three facts doctor's `build` check reports, because two spellings of
    # "which build" is the drift that makes both useless.
    ap.add_argument(
        "--version",
        action="version",
        version=cli_doctor.version_line(),
        help="the installed distribution, the hook's content hash, and the "
        "payload's commit",
    )
    sub = ap.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    for name, (summary, epilog, declare) in _SUBCOMMANDS.items():
        declare(
            sub.add_parser(
                name,
                help=summary,
                description=f"{summary}.",
                epilog=epilog,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
        )
    for name, (summary, template) in _PENDING.items():
        meanwhile = _meanwhile(template)
        # `memkit doctor --help` is the other probe an agent tries, and
        # argparse answers it from the SUBparser — which had bare usage and
        # exit 0, i.e. the one shape that reads as "this works". Help stays
        # exit 0, because that is what --help means, and now carries both
        # halves of the refusal.
        sub.add_parser(
            name,
            help=f"{summary} — NOT IN THIS BUILD YET",
            description=f"{summary}.\n\nNOT IN THIS BUILD YET. {meanwhile}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    return ap


def _pending(name: str) -> int:
    summary, template = _PENDING[name]
    print(
        f"memkit: `{name}` ({summary}) is not in this build.\n"
        f"{_meanwhile(template)}",
        file=sys.stderr,
    )
    return EXIT_NOT_IN_BUILD


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    # parse_known_args, not parse_args: the flags these subcommands will take
    # do not exist yet, and `memkit doctor --json` has to reach the message
    # explaining that rather than dying on an unrecognised argument. An agent
    # that meets argparse's usage text there learns that `--json` is wrong,
    # which is a different and false thing to learn.
    args, extra = ap.parse_known_args(argv)
    if args.subcommand is None:
        # Usage on stderr and a usage exit, because a bare `memkit` asked for
        # something and got nothing done. An unknown subcommand never reaches
        # here — argparse rejects it against the choices above with the same
        # code, which is where the list of what exists belongs.
        ap.print_help(sys.stderr)
        return EXIT_USAGE
    handler = _HANDLERS.get(args.subcommand)
    return handler(args, extra) if handler else _pending(args.subcommand)


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
