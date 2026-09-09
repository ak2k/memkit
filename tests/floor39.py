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

from memkit import cli, cli_doctor, cli_init, harness_memory  # noqa: E402

check("dispatcher has both", sorted(cli._HANDLERS), ["doctor", "init"])
# The harness-memory helpers, called rather than merely imported: both walk the
# filesystem, and a 3.9 break in either is a doctor row that says "the check
# itself failed" on the machine that most needs the answer.
check("a project key is one directory name",
      "/" in harness_memory.project_key(os.getcwd()), False)
check("nothing written is an empty inventory",
      harness_memory.inventory(os.environ["CLAUDE_CONFIG_DIR"]), [])
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

# --- the repository's own file is read, on a checkout built here -------------
#
# EXECUTED rather than type-checked: pyright had the whole of this code and a
# 3.9 configuration, and what it cannot see is the class this file exists for —
# a name that is present in the version it was told about and absent in the one
# the harness runs. The guard is a walk, an open, an fstat and a realpath, so
# it is exactly the shape that breaks that way.

# Realpath'd here, because the walk realpaths what it is handed and `$TMPDIR`
# is behind a symlink on macOS — the shape this file exists to run on.
checkout = os.path.realpath(os.path.join(home, "checkout"))
os.makedirs(os.path.join(checkout, ".git"))
os.makedirs(os.path.join(checkout, "docs", "memories", "search"))
memkit_json = os.path.join(checkout, ".memkit.json")


def write_project(spec):
    with open(memkit_json, "w") as f:
        json.dump({"memkit_project": 1, "store": spec}, f)


write_project({"id": "app", "dir": os.path.join("docs", "memories")})
check("the checkout is found from inside it",
      hook._repo_root(os.path.join(checkout, "docs")), checkout)
project, why = hook._project_store(checkout, {"notes"})
check("the project file is accepted", why, "")
check("the project store is read-only",
      project is not None and project.read_only, True)
check("the project corpus is the directory the file named",
      project is not None and project.resolved_dir,
      os.path.realpath(os.path.join(checkout, "docs", "memories")))

# And the refusal path, which is most of the guard: a `dir` outside the
# checkout is the one every other rejection is shaped like.
write_project({"id": "app", "dir": "/etc"})
refused, why = hook._project_store(checkout, {"notes"})
check("an absolute dir is refused", refused, None)
check("the refusal says which key", "'dir'" in why and "is absolute" in why, True)
os.remove(memkit_json)
check("no file at all is not a refusal", hook._project_store(checkout, set()),
      (None, ""))

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

# --- a nested project file is one file refused, not one prompt lost ----------
#
# The depth is run rather than asserted, because the exception TYPE is what
# differs between interpreters: this scanner answers a document nested past its
# budget with `RecursionError`, a `RuntimeError` the suite's own 3.12 never
# produces within the 4096-byte cap. Half that cap buys 1024 levels, so the
# shape is committable, and uncaught it does not cost the checkout its own
# corpus — it costs every prompt there every store, the user's included.

levels = hook.PROJECT_CONFIG_MAX_BYTES // 4
with open(memkit_json, "w") as f:
    f.write("[" * levels + "]" * levels)
nested, why = hook._project_store(checkout, {"notes"})
check("a nested project file is refused", nested, None)
check("the refusal names the file and the parse",
      why.startswith(hook.PROJECT_CONFIG_NAME + " is not valid JSON:"), True)
out = subprocess.run(
    [sys.executable, os.path.join(REPO, "src", "memkit", "memory_prompt_recall.py")],
    input=json.dumps({"session_id": "floor39n", "prompt":
                      "why does pgbouncer transaction pooling break prepared statements"}),
    capture_output=True, text=True, timeout=300, cwd=checkout,
    env=dict(os.environ, MEMKIT_CONFIG=config),
)
check("the hook exits 0 in that checkout", out.returncode, 0)
check("the user's own store is still served there", "pooling.md" in out.stdout, True)
os.remove(memkit_json)

# --- a `dir` behind a chain of symlinks is one file refused too --------------
#
# The guard's other `RuntimeError`, and the chain itself is the evidence:
# `realpath` recurses once per link, so a `dir` a repository committed behind
# a thousand of them is neither an `OSError` nor a `ValueError`. This one is
# not staged on any interpreter — the recursion limit is what it is here.

links = sys.getrecursionlimit() + 100
os.symlink(os.path.join("docs", "memories"), os.path.join(checkout, "l%d" % (links - 1)))
for i in range(links - 2, -1, -1):
    os.symlink("l%d" % (i + 1), os.path.join(checkout, "l%d" % i))
try:
    os.path.realpath(os.path.join(checkout, "l0"))
    deep = False
except RecursionError:
    deep = True
check("the chain is deep enough to recurse past the limit", deep, True)
write_project({"id": "app", "dir": "l0"})
chained, why = hook._project_store(checkout, {"notes"})
check("a dir behind a symlink chain is refused", chained, None)
check("the refusal names the key and the resolution",
      why, "%s: 'dir' does not resolve: RecursionError" % hook.PROJECT_CONFIG_NAME)
out = subprocess.run(
    [sys.executable, os.path.join(REPO, "src", "memkit", "memory_prompt_recall.py")],
    input=json.dumps({"session_id": "floor39s", "prompt":
                      "why does pgbouncer transaction pooling break prepared statements"}),
    capture_output=True, text=True, timeout=300, cwd=checkout,
    env=dict(os.environ, MEMKIT_CONFIG=config),
)
check("the hook exits 0 behind that chain", out.returncode, 0)
check("the user's own store survives the chain", "pooling.md" in out.stdout, True)
os.remove(memkit_json)
for i in range(links):
    os.remove(os.path.join(checkout, "l%d" % i))

# --- the guarded open, the credential scan, and the read-only branch ---------
#
# The guard above reaches `_project_store` and no line of the trio underneath
# it, and all three are the shape this file exists for: an O_NONBLOCK open, an
# fstat, a lazily compiled alternation, and the one branch that reads a file a
# REPOSITORY chose. A break in any of them at this floor is an every-prompt
# hook that hangs, or one that puts a committed credential in front of a model.

fifo = os.path.join(home, "fifo")
os.mkfifo(fifo)
try:
    hook._regular_fd(fifo)
    check("a fifo is refused as not a regular file", "returned a descriptor",
          "_NotRegular")
except hook._NotRegular:
    pass
except OSError as exc:  # pragma: no cover - a floor break is what this reports
    check("a fifo is refused as not a regular file", type(exc).__name__,
          "_NotRegular")

plain = os.path.join(home, "plain.txt")
with open(plain, "w") as f:
    f.write("x" * 17)
regular_fd, regular_st = hook._regular_fd(plain)
os.close(regular_fd)
check("the guarded open reports the file's own size", regular_st.st_size, 17)

planted_key = "aws_secret_access_key = " + "A" * 40
check("the scan sees a committed key",
      hook._secret_re().search(planted_key) is not None, True)
# The SHAPE is what the scan is about: the same word in prose, with no
# assignment behind it, is a memory somebody wrote about credentials.
check("the scan leaves prose about one alone",
      hook._secret_re().search(
          "the aws_secret_access_key is the field name, and it is not here"
      ) is not None, False)

corpus = os.path.join(checkout, "docs", "memories", "search")
planted = os.path.join(corpus, "creds.md")
with open(planted, "w") as f:
    f.write("---\nname: creds\ndescription: pgbouncer transaction pooling notes\n"
            "type: reference\n---\n\n"
            "pgbouncer transaction pooling breaks prepared statements\n"
            + planted_key + "\n")
terms = ["pgbouncer", "transaction", "pooling"]
corpus_real = os.path.realpath(corpus)
# The evidence the index would have produced, so the two branches below differ
# only in whether a repository chose the corpus.
hook._LEX_MATCHED[planted] = list(terms)
check("a repository's file carrying a key earns no evidence",
      hook._relevance(terms, planted, corpus_real, True), ([], len(terms), "?"))
check("and the same file in a store the user configured earns its own",
      hook._relevance(terms, planted, corpus_real, False),
      (terms, len(terms), "reference"))
os.remove(planted)

if failures:
    for line in failures:
        sys.stderr.write("floor39: " + line + "\n")
    sys.exit(1)
sys.stdout.write(f"floor39: ok on {sys.version.split()[0]}\n")
