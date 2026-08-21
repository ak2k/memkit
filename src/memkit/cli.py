"""`memkit <subcommand>` — the dispatcher the setup and diagnosis commands hang off.

A fourth console script rather than more flags on `memory-recall`, because
these are a different job. `memory-recall` searches; what an adopter needs
before they can search is a way to set the thing up and a way to ask whether
it worked, and both of those write or read things that have nothing to do with
retrieval. Keeping them apart is also what lets a skill pre-approve one exact
argument shape: an `allowed-tools` entry that had to cover setup, diagnosis and
arbitrary-directory search in one command is an entry that pre-approves reading
any directory.

The dispatch is a dict. A later unit adds one line to `_HANDLERS` and a module
beside this one; nothing else about the routing moves. Until then a name in
`_PENDING` with no handler is what a caller meets, by design — see `_pending`
for why the names are listed at all before they do anything.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from memkit.memory_prompt_recall import (
    DEFAULT_SEARCH_CLI,
    EXIT_INERT,
    EXIT_NO_MATCH,
    _search_cli,
)

# This binary's exit codes, and deliberately NOT `memory-recall`'s. They are
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
    "doctor": (
        "report whether retrieval is actually working on this machine",
        "meanwhile: `{search_config}` for what resolved, and "
        '`{search} "<terms>"` for whether the stores answer. '
        f"Exit {EXIT_INERT} there means there was nothing to search — no "
        "config, or no store on disk and in scope for this directory — and "
        f"stderr names which; exit {EXIT_NO_MATCH} means the stores were "
        "searched and nothing matched",
    ),
    "init": (
        "create a store and wire this machine up to it",
        "meanwhile: write the config by hand — the schema and a worked example "
        "are in the project README under Config",
    ),
}


def _meanwhile(template: str) -> str:
    """Fill a `meanwhile` template with the commands this install exposes.

    `_search_cli()` is what the hook's own truncation notice advertises, so the
    refusal and the pointer block name the same binary by construction. It
    falls back to the shipped default when no config resolves, which is the
    only case where there is nothing better to say.

    Nothing about the config may take this down. A refusal is what a caller
    reaches when something is already wrong, so it is the last surface that
    should be able to fail — and it is reached from `--help`, the cheapest
    probe there is. Reading the config to improve a message is worth doing;
    it is not worth an exit code that says the tool is broken, so any failure
    resolving it falls back to the shipped default rather than propagating.
    """
    try:
        search = _search_cli()
        # The debug form is the search command with its mode flag swapped,
        # since `search_cli` is spelled `<binary> --search`.
        binary = search.split()[0]
    except Exception:
        search = DEFAULT_SEARCH_CLI
        binary = search.split()[0]
    return template.format(search=search, search_config=f"{binary} --debug-config")

# Subcommand -> the function that runs it, given the arguments this parser did
# not consume. Empty today.
_HANDLERS: dict[str, Callable[[list[str]], int]] = {}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="memkit",
        description="Set up and diagnose a memkit installation. "
        "To search one, see `memory-recall --search`.",
        # The fallbacks belong on the top-level help too, not only on the
        # refusal an agent reaches by running the subcommand: `memkit --help`
        # is the cheaper probe and the one tried first, and it was answering
        # with two names and no way forward.
        epilog="Not in this build yet:\n"
        + "\n".join(f"  {n}: {_meanwhile(m)}" for n, (_, m) in _PENDING.items())
        + f"\n\nExit codes: 0 ok / {EXIT_NO_RUNTIME} memkit could not start at "
        f"all (stderr names what is missing) / {EXIT_USAGE} usage error or "
        f"unknown subcommand / {EXIT_NOT_IN_BUILD} the subcommand exists but "
        "is not in this build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
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
    return handler(extra) if handler else _pending(args.subcommand)


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
