#!/usr/bin/env python3
"""Record one hook invocation: argv, env, cwd and the payload on stdin.

Registered by `Profile.register_dump_hooks` at SETTINGS scope, which is what
bounds what it can settle — and it is less than the obvious reading. Measured
on 2.1.239: a settings-scope hook receives NONE of `CLAUDE_PLUGIN_OPTION_*`,
`CLAUDE_PLUGIN_ROOT` or `CLAUDE_PLUGIN_DATA`, so this cannot see the option
arrive; and the `argv` below is this script's own, since the registration runs
`{python} {hookdump} {event}`, so it is never memkit's argv either.

What it does settle is what the harness told a hook of its own: the event, the
payload, the cwd, and the non-plugin half of the environment —
`CLAUDE_CODE_ENTRYPOINT` among it, which is the one claim the pty driver
exists to make. The plugin-side claims are read off memkit's own artifacts
instead; see the tier note in `__init__`.

Writes one file per invocation, named so that `sorted()` is invocation order,
because two subagents spawned in one turn produce two records at the same
second and a single file would keep only the later one.

Never blocks and never fails the turn it is measuring: stdin is read with a
deadline, and any error at all still exits 0. An instrument that can break the
run it observes produces evidence about itself.
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> int:
    event = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    payload = None
    raw = ""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else None
    except Exception:  # noqa: BLE001 - an instrument never fails the run
        pass

    log = os.environ.get("MEMKIT_RIG_HOOKLOG")
    if not log:
        return 0
    record = {
        "event": event,
        # argv WITHOUT the interpreter and script path, i.e. what the
        # registration passed. The zero-argument claim is about this list.
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "env": dict(os.environ),
        "payload": payload,
        "raw_len": len(raw),
        "ts": time.time(),
    }
    name = f"{time.time():.6f}-{os.getpid()}-{event}.json"
    try:
        with open(os.path.join(log, name), "w", encoding="utf-8") as f:
            json.dump(record, f)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
