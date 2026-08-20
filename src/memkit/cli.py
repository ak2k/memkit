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

from memkit.memory_prompt_recall import EXIT_ERROR

# What `memkit` will route, with the one-line summary `--help` shows and the
# thing to reach for meanwhile. Listed before they exist on purpose: the reader
# of `--help` here is usually an agent deciding what to do next, and a name
# that is simply absent reads as "this tool cannot do that" — which sends it to
# invent a way — while a name that says "not in this build" sends it to the
# fallback named beside it. The two answers are not close together in
# consequence, and only one of them is true.
_PENDING: dict[str, tuple[str, str]] = {
    "doctor": (
        "report whether retrieval is actually working on this machine",
        'meanwhile: `memory-recall --debug-config` for what resolved, and '
        '`memory-recall --search "<terms>"` for whether the stores answer '
        "(exit 3 means nothing is configured, 1 means nothing matched)",
    ),
    "init": (
        "create a store and wire this machine up to it",
        "meanwhile: write the config by hand — the schema and a worked example "
        "are in the project README under Config",
    ),
}

# Subcommand -> the function that runs it, given the arguments this parser did
# not consume. Empty today.
_HANDLERS: dict[str, Callable[[list[str]], int]] = {}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="memkit",
        description="Set up and diagnose a memkit installation. "
        "To search one, see `memory-recall --search`.",
    )
    sub = ap.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    for name, (summary, _) in _PENDING.items():
        sub.add_parser(name, help=f"{summary} — NOT IN THIS BUILD YET")
    return ap


def _pending(name: str) -> int:
    summary, meanwhile = _PENDING[name]
    print(
        f"memkit: `{name}` ({summary}) is not in this build.\n{meanwhile}",
        file=sys.stderr,
    )
    return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    # parse_known_args, not parse_args: the flags these subcommands will take
    # do not exist yet, and `memkit doctor --json` has to reach the message
    # explaining that rather than dying on an unrecognised argument. An agent
    # that meets argparse's usage text there learns that `--json` is wrong,
    # which is a different and false thing to learn.
    args, extra = ap.parse_known_args(argv)
    if args.subcommand is None:
        # Usage on stderr and a non-zero exit, because a bare `memkit` asked
        # for something and got nothing done. An unknown subcommand never
        # reaches here — argparse rejects it against the choices above, which
        # is where the list of what exists belongs.
        ap.print_help(sys.stderr)
        return EXIT_ERROR
    handler = _HANDLERS.get(args.subcommand)
    return handler(extra) if handler else _pending(args.subcommand)


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
