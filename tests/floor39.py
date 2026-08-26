"""What a stock macOS python must be able to do, executed by one.

Run by `tests/test_packaging.py` against a real 3.9 interpreter. Not a pytest
module: pytest is not installed for that interpreter and does not need to be —
what is being checked is that memkit's own entry points IMPORT and RUN there,
which is a claim about the code and not about a test framework.

The floor was a static pyright pass and nothing else. That catches a PEP-604
annotation evaluated at runtime and a 3.10+ stdlib call it can see the type of;
it does not catch a module-level attribute that exists in the version pyright
was told about and not in the one the harness runs — `sqlite3.SQLITE_BUSY`
landed in 3.11, and a reference to it reachable on 3.9 is a hook that raises on
the machine that most needs it. Executing the thing is the check that sees that
class of failure.

Every failure prints one line and exits non-zero. There is no reporting to do:
either the floor holds or it does not.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, wanted {want!r}")


# --- the hook module imports and its surfaces answer -------------------------

from memkit import memory_prompt_recall as hook  # noqa: E402

check("version_info floor", sys.version_info[:2] >= (3, 9), True)

# The contention/damage split, which is the one place this file's own history
# reaches for a name that does not exist at this floor. `sqlite_errorcode` and
# the `sqlite3.SQLITE_*` constants both landed in 3.11, so the guarded branch
# must not be taken here and the message fallback must answer instead.
check("no sqlite_errorcode at this floor",
      hasattr(sqlite3.OperationalError("x"), "sqlite_errorcode"), False)
check("busy is contention", hook._fts_busy(sqlite3.OperationalError("database is locked")), True)
check("busy is contention (busy)", hook._fts_busy(sqlite3.OperationalError("database is busy")), True)
check("damage is not contention",
      hook._fts_busy(sqlite3.OperationalError("database disk image is malformed")), False)
check("a non-sqlite error is not contention", hook._fts_busy(ValueError("x")), False)

check("prompt gate, short", hook.prompt_gate("hi"), "gate:short")
check("prompt gate, ok", hook.prompt_gate("why do prepared statements break"), None)
check("query builder", hook.build_query("why do prepared statements break") is None, False)
check("sanitizer", hook.sanitize("a\x1b[31m  b"), "a b")

# --- EVERYTHING BELOW HERE RUNS AGAINST A SCRATCH HOME -----------------------
#
# The runner passes the whole environment through, stripping only `MEMKIT_*`,
# so until this block runs `$HOME` is the developer's and `_state_dir()` is
# their real cache. `_sweep()` was called fifteen lines above it: on this
# machine that is 28,000 files, and the run unlinked from them and rewrote
# their cursor. It was flaky as well as destructive — `unlink` is 0 only while
# the real stamp is under an hour old, so the same suite passed at 12:30 and
# failed at 13:30.

home = tempfile.mkdtemp()
os.environ["HOME"] = home
os.environ["XDG_CACHE_HOME"] = os.path.join(home, ".cache")
for name in ("MEMKIT_CONFIG", "MEMKIT_PLUGIN", "CLAUDE_PLUGIN_DATA",
             "CLAUDE_PLUGIN_ROOT", "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG"):
    os.environ.pop(name, None)
os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(home, "claude")
# Asserted rather than assumed: this file is a script with no fixtures, so
# nothing but this line stands between a later check and the real cache.
check("state is hermetic",
      hook._state_dir_candidate().startswith(tempfile.gettempdir()), True)

check("state dir is absolute", os.path.isabs(hook._state_dir_candidate()), True)
check("task state path", os.path.basename(hook._task_state_path("t")).startswith("t-"), True)
check("sweep on an absent dir", hook._sweep()["unlink"], 0)

# --- the dispatcher and both subcommands import ------------------------------

from memkit import cli, cli_doctor, cli_init  # noqa: E402

check("dispatcher has both", sorted(cli._HANDLERS), ["doctor", "init"])
check("doctor checks", len(cli_doctor.CHECK_IDS) > 20, True)
check("version line", "hook:" in cli_doctor.version_line(), True)
check("init default config is absolute after expansion",
      os.path.isabs(os.path.expanduser(cli_init.DEFAULT_CONFIG)), True)

# Every doctor check runs, which is what makes this more than an import test:
# a check that raises is caught and reported as UNKNOWN, so a floor break would
# otherwise hide inside the envelope rather than failing.
report = cli_doctor.envelope(cli_doctor.collect(cli_doctor.Machine()))
broke = [c for c in report["checks"] if "the check itself failed" in c["detail"]]
check("no doctor check raised", [c["id"] for c in broke], [])
check("every check ran", len({c["id"] for c in report["checks"]}),
      len(cli_doctor.CHECK_IDS))

# --- and the hook SERVES a pointer, run as the harness runs it ---------------

store = os.path.join(home, "notes", "search")
os.makedirs(store)
with open(os.path.join(store, "pooling.md"), "w") as f:
    f.write("---\nname: pooling\ndescription: pgbouncer transaction pooling "
            "breaks prepared statements.\ntype: reference\n---\n\n"
            "pgbouncer transaction pooling breaks prepared statements\n")
config = os.path.join(home, "memkit.json")
with open(config, "w") as f:
    json.dump({"schema": 1,
               "roots": {"h": {"kind": "path", "path": os.path.join(home, "notes")}},
               "stores": [{"id": "notes", "dir": ".", "live_root": "h"}]}, f)
out = subprocess.run(
    [sys.executable, os.path.join(REPO, "src", "memkit", "memory_prompt_recall.py")],
    input=json.dumps({"session_id": "floor39", "prompt":
                      "why does pgbouncer transaction pooling break prepared statements"}),
    capture_output=True, text=True, timeout=300,
    env=dict(os.environ, MEMKIT_CONFIG=config),
)
check("the hook exits 0", out.returncode, 0)
check("the hook emitted a pointer", "pooling.md" in out.stdout, True)

if failures:
    for line in failures:
        sys.stderr.write("floor39: " + line + "\n")
    sys.exit(1)
sys.stdout.write(f"floor39: ok on {sys.version.split()[0]}\n")
