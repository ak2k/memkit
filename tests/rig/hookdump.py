#!/usr/bin/env python3
"""Record one hook invocation: argv, env, cwd and the payload on stdin.

Registered by `Profile.register_dump_hooks`, this is the rig's only view of
what the harness actually did. Everything the plugin claims about registration
— that it passes zero arguments, that a manifest option arrives under a
particular variable name, that `CLAUDE_PLUGIN_DATA` is present here and absent
there — is a statement about this record.

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
