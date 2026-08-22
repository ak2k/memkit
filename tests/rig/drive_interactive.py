#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pexpect"]
# ///
"""Drive an interactive `claude` session over a pty, and leave.

`claude -p` is not the same program. It sets `CLAUDE_CODE_ENTRYPOINT=sdk-cli`
— and sets it itself, so scrubbing the variable from the parent's environment
does not change what a hook sees — while a pty run reports `cli`. Harness
behaviour keys on that difference, so any scenario making a claim about what a
PERSON's session does has to be driven through a terminal rather than through
`-p`.

Run through `uv run --script` for the pexpect dependency: the suite proper is
stdlib-only and this is an instrument, not a test dependency. Environment comes
from the caller (`Profile.env()`), which is what keeps it inside a scratch
profile.

usage: drive_interactive.py <project-dir> <prompt> [seconds]
"""

from __future__ import annotations

import os
import sys

import pexpect


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    project, prompt = sys.argv[1], sys.argv[2]
    seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 180

    # SCRATCH, not merely set. `Profile._guard` makes this check for everything
    # it spawns; this script is spawned as a subprocess and had only the
    # weaker half, so it would have driven a real session against the author's
    # own profile — which carries a live memkit registration — as long as the
    # variable pointed anywhere at all.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "")
    assert config_dir, (
        "refusing to drive a session against the real profile — "
        "CLAUDE_CONFIG_DIR must name a scratch directory"
    )
    resolved = os.path.realpath(config_dir)
    real = os.path.realpath(os.path.expanduser("~"))
    assert resolved != os.path.join(real, ".claude"), resolved
    assert not resolved.startswith(real + os.sep) or ".cache" in resolved.split(
        os.sep
    ), f"CLAUDE_CONFIG_DIR is inside the real home: {resolved}"

    child = pexpect.spawn(
        "claude",
        cwd=project,
        # No `env=`: the child inherits this process's environment, which is
        # the profile's — `Profile.env()` is what spawned this script. Passing
        # a copy would be the same environment with an extra step.
        encoding="utf-8",
        timeout=seconds,
        dimensions=(40, 120),
    )
    child.logfile_read = sys.stdout
    # The prompt box is the only reliable "ready" signal; the banner and any
    # tips above it vary by build and by what the profile has already seen.
    child.expect([r"│\s*>", pexpect.TIMEOUT], timeout=60)
    child.send(prompt)
    child.send("\r")
    # A turn is over when the box comes back. Matching on the model's text
    # instead would make the driver depend on what the model chose to say.
    child.expect([r"│\s*>", pexpect.EOF, pexpect.TIMEOUT], timeout=seconds)
    child.send("\x03")  # ctrl-c
    child.send("\x03")
    child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.close(force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
