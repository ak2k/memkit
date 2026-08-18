#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = ["wordfreq"]
# ///
"""Regenerate src/memkit/common-words.txt — the English-common wordlist
behind the recall hook's relevance floor.

The hook needs one boolean per query term: "is this common English?"
(Zipf >= threshold). wordfreq computes continuous frequencies for ~40
languages with a multi-MB table load — far too heavy for a per-prompt
stdlib hook — so this generator compiles the answer down to a static
one-word-per-line file the hook reads as a frozenset (single-digit ms).

Run manually, rarely: English word frequency doesn't drift. Regenerate
only to tune the threshold (then update ZIPF_THRESHOLD in the hook's
common-words test to match) or on a major wordfreq release.
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

ZIPF_THRESHOLD = 3.5
# top_n_list(50_000) reaches down to Zipf ~2.7 — comfortably past the
# threshold, so the >= filter sees every qualifying word.
CANDIDATE_POOL = 50_000

OUT = Path(__file__).resolve().parent.parent / "src/memkit/common-words.txt"


def main() -> None:
    words = sorted(
        w
        for w in top_n_list("en", CANDIDATE_POOL)
        if w.isascii() and w.isalpha() and zipf_frequency(w, "en") >= ZIPF_THRESHOLD
    )
    header = (
        f"# English words with wordfreq Zipf frequency >= {ZIPF_THRESHOLD}\n"
        f"# (wordfreq {version('wordfreq')}; {len(words)} words)\n"
        "# Consumed by memkit's recall hook: a memory hit whose matched\n"
        "# query terms are ALL in this list is floored as conversational\n"
        "# coincidence (unless >=3 terms matched).\n"
        "# Regenerate: uv run tools/generate-common-words.py\n"
    )
    OUT.write_text(header + "\n".join(words) + "\n")
    print(f"wrote {OUT} ({len(words)} words)")


if __name__ == "__main__":
    sys.exit(main())
