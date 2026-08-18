#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""R6: assert src/memkit/common-words.txt is what the pinned wordfreq produces.

The wordlist is a committed build artifact of a maintainer-only generator, and
it is load-bearing for retrieval: a hit whose matched query terms are all in
this list is floored as conversational coincidence. Nothing else in the repo
would notice a hand-edited line, a truncated write, or a regeneration under a
different wordfreq — the file would just be a different retriever, arriving in
a diff that reads as data.

So: regenerate under the pin, compare, restore. Any difference is red.

Two pins have to agree before that means anything. `[project.optional-
dependencies] wordlist` in pyproject.toml is what a maintainer installs; the
PEP 723 block in generate-common-words.py is what `uv run` actually resolves.
If they drift, this check regenerates under one pin while the documented one
says another, and passes. They are compared first.

Run: uv run tools/check-wordlist-reproducible.py
"""

from __future__ import annotations

import difflib
import re
import subprocess
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
GENERATOR = REPO / "tools" / "generate-common-words.py"
ARTIFACT = REPO / "src" / "memkit" / "common-words.txt"

# PEP 723's own reference regex for the inline metadata block.
PEP723 = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)


def pep723_deps(script: Path) -> list[str]:
    matches = [
        m for m in PEP723.finditer(script.read_text()) if m.group("type") == "script"
    ]
    if len(matches) != 1:
        sys.exit(f"{script}: expected exactly one PEP 723 script block, found "
                 f"{len(matches)}")
    body = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in matches[0].group("content").splitlines(keepends=True)
    )
    return list(tomllib.loads(body).get("dependencies", []))


def sole_wordfreq_pin(specs: list[str], where: str) -> str:
    pins = [s for s in specs if s.replace("-", "_").lower().startswith("wordfreq")]
    if len(pins) != 1:
        sys.exit(f"{where}: expected exactly one wordfreq requirement, found {pins}")
    pin = pins[0]
    if "==" not in pin:
        sys.exit(
            f"{where}: wordfreq is not pinned with `==` ({pin!r}). An unpinned "
            "generator makes this whole check assert nothing."
        )
    return pin


def main() -> int:
    project = tomllib.loads(PYPROJECT.read_text())
    declared = sole_wordfreq_pin(
        project["project"]["optional-dependencies"]["wordlist"],
        "pyproject.toml [project.optional-dependencies] wordlist",
    )
    resolved = sole_wordfreq_pin(
        pep723_deps(GENERATOR), f"{GENERATOR.name} PEP 723 dependencies"
    )
    if declared != resolved:
        print(
            f"ERROR: the two wordfreq pins disagree.\n"
            f"  pyproject.toml [wordlist] extra: {declared}\n"
            f"  {GENERATOR.name} PEP 723 block: {resolved}\n"
            "Whichever is right, they have to be the same string: the extra is "
            "what a maintainer installs, the PEP 723 block is what `uv run` "
            "resolves, and this check is only meaningful when they match.",
            file=sys.stderr,
        )
        return 1
    print(f"wordfreq pin: {declared} (pyproject and PEP 723 block agree)")

    committed = ARTIFACT.read_bytes()
    try:
        # The generator writes ARTIFACT in place; that is the only way to run
        # it, so put the committed bytes back before returning either way.
        # `--script` is what makes uv honour the PEP 723 pin rather than the
        # project environment.
        run = subprocess.run(
            ["uv", "run", "--quiet", "--script", str(GENERATOR)],
            cwd=REPO,
            check=False,
        )
        if run.returncode != 0:
            print(
                f"ERROR: the generator exited {run.returncode}; the artifact was "
                "left as committed. This is a broken check, not a failed "
                "assertion — the wordlist was never compared.",
                file=sys.stderr,
            )
            return 2
        regenerated = ARTIFACT.read_bytes()
    finally:
        ARTIFACT.write_bytes(committed)

    if regenerated == committed:
        print(f"{ARTIFACT.relative_to(REPO)}: reproduces byte-for-byte")
        return 0

    diff = difflib.unified_diff(
        committed.decode().splitlines(),
        regenerated.decode().splitlines(),
        fromfile="committed",
        tofile=f"regenerated with {declared}",
        lineterm="",
        n=1,
    )
    print(
        f"ERROR: {ARTIFACT.relative_to(REPO)} is not what {declared} produces.\n"
        "Either the artifact was edited by hand, or the pin moved without the "
        "artifact being regenerated. Both change retrieval for every consumer.\n"
        "Fix: uv run tools/generate-common-words.py, and read the diff.\n",
        file=sys.stderr,
    )
    for i, line in enumerate(diff):
        if i >= 60:
            print("  ... (truncated)", file=sys.stderr)
            break
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
