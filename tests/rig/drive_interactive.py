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
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    project, prompt = sys.argv[1], sys.argv[2]
    seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 180

    # SCRATCH, not merely set — and through the SAME helper `Profile._guard`
    # uses. This script runs as a separate process, so it cannot share the
    # parent's copy; keeping its own was how it ended up deriving the real home
    # from `$HOME`, which every caller has already redirected into the scratch
    # tree. The guard then could not refuse anything.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "")
    assert config_dir, (
        "refusing to drive a session against the real profile — "
        "CLAUDE_CONFIG_DIR must name a scratch directory"
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rig import assert_scratch_config_dir

    assert_scratch_config_dir(config_dir)

    # AFTER the guard, deliberately. `pexpect` is what spawns, and importing it
    # at module scope meant a refusal could not be reached without the
    # dependency present — so the one check standing between this script and a
    # real profile ran second to a `ModuleNotFoundError`.
    import pexpect

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
