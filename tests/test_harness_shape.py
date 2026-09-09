"""What `tools/harness_shape.py` may carry off a machine, and what it may not.

The tool exists so memkit's regression tests can run against corpora the
HARNESS wrote rather than corpora memkit wrote, and the price of that is
reading other people's config directories. So the rule it is held to here is
not "it works": it is that a shape committed to this repository carries no
memory text, no description text and no real name of anything — and that the
counts, sizes and flags it does carry are the ones a rebuilt tree needs in
order to exercise the same paths.

`test_the_committed_shapes_carry_no_names` is the one that gates the artifact
rather than the code, and it reads whatever is in `tests/data/harness_shapes/`
instead of a fixed list of names: a fixture lands here in its own commit, and a
test naming the files in advance would be red until then, while a test naming
them afterwards would skip if one were ever removed.

The NFS fixture was taken off a machine that cannot be captured again, so its
`index.dangling_rows` is the number the tool gave before an index row naming a
file one directory down was looked at rather than written off: it OVER-COUNTS
wherever that machine's indexes were tiered. The darwin one was re-captured
after the fix and its counts are current. `unreadable` was written into the
NFS fixture by a script for the same reason, `false` throughout: that capture
read every file it listed, which is what its `read_errors` total of 0 says.
"""

from __future__ import annotations

import ast
import errno
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from memkit import cli_doctor, harness_memory, memory_integrity

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "harness_shape.py"
SHAPES = REPO / "tests" / "data" / "harness_shapes"

# What an anonymised shape is allowed to say. A key is dashes and `s<n>`
# segments; a file is the index under its own name or `m<n>.md`; a plugin is
# `p<n>@q<n>` with memkit's own name the one exception.
KEY_RE = re.compile(r"^(-|s\d+)*$")
FILE_RE = re.compile(r"^(MEMORY\.md|m\d+\.md)$")
PLUGIN_RE = re.compile(r"^(memkit|p\d+)(@q\d+)?$")
# And the four rows that are not pseudonyms but still carry a string. Written
# out here rather than imported from the tool: a gate that read the tool's own
# allowlist would pass whatever the tool decided to allow, which is the one
# thing an artifact gate must not do.
VERSION_RE = re.compile(r"\d+(\.\d+)*")
INSTALL_OK = frozenset({"global", "native", "local", "npm", "unknown", "other"})
# Re-typed like the two above, and it did not used to be: this was the tool's
# own `[A-Za-z]+` character for character, so it could only catch the tool
# failing to apply its rule and never the rule being wrong — and the rule was
# wrong, because an organisation's gate is letters too.
HOOK_PSEUDONYM_RE = re.compile(r"h\d+")
HOOK_OK = frozenset({
    "Notification", "PermissionDenied", "PermissionRequest", "PostToolBatch",
    "PostToolUse", "PreCompact", "PreToolUse", "SessionEnd", "SessionStart",
    "Stop", "SubagentStop", "UserPromptSubmit",
})
# The two vocabularies a shape ADDRESSES its values by. Every rule above is
# about a value, and a dict key is a string a fixture carries just as surely:
# a scope named after an employer's policy set, holding the exact field set
# the tool emits, passed both gates. Re-typed like the vocabularies above, and
# held to `harness_memory`'s own names by
# `test_the_constants_copied_from_memkit_are_the_ones_memkit_holds`.
SCOPE_OK = frozenset({"managed", "local", "project", "user"})
MEMORY_KEY_OK = frozenset({
    "autoMemoryEnabled", "autoMemoryDirectory", "autoDreamEnabled",
})
# And what every field that is not a string is allowed to BE. Without this a
# path in `files[].size` and a hostname in `lock_age_s` are numbers as far as
# the gate can tell — the leak rules only ever look at the strings they expect
# to find. `bool` and `int` are kept apart in both directions because
# `isinstance(True, int)` is true in Python, and a count is not a flag.
SHAPE_TYPES = {
    "schema": int,
    "projects_total": int,
    "memory_dirs_total": int,
    "skipped": int,
    "read_errors": int,
    "anonymised": bool,
    "settings_managed_read": bool,
}
DIR_TYPES = {
    "key_len": int,
    "is_symlink": bool,
    "project_is_symlink": bool,
    "lock_age_s": (int, type(None)),
}
FILE_TYPES = {
    "size": int,
    "is_symlink": bool,
    "unreadable": bool,
}
# The frontmatter facts, which a record only states when the file was read.
# A record that says it was not read states none of them, so the types are
# read off the record's own answer rather than fixed: `false` there is a
# statement about a read that did not happen.
READ_TYPES = {
    "has_frontmatter": bool,
    "has_description": bool,
    "has_name": bool,
    "has_type": bool,
    "frontmatter_truncated": bool,
    "description_len": (int, type(None)),
}
UNREAD_TYPES = dict.fromkeys(READ_TYPES, type(None))
INDEX_TYPES = {"rows": int, "dangling_rows": int, "truncated": bool}
SCOPE_TYPES = {"unreadable": bool}
# The harness record's two values are read by name below, and a number in
# either of them reaches `VERSION_RE.fullmatch` as a `TypeError` — an error
# the gate raises rather than a name it refuses.
HARNESS_TYPES = {
    "version_hint": (str, type(None)),
    "install": (str, type(None)),
}


def _run(*args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def _shape(*args: str, env=None) -> dict:
    out = _run(*args, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    return json.loads(out.stdout)


# The tool's test-only seam for the one path it reads outside `--config-dir`.
# Spelled here rather than imported, so a rename of the variable fails these
# cases instead of quietly reading the runner's own managed settings.
MANAGED_DIR_ENV = "MEMKIT_SHAPE_MANAGED_DIR"


def _managed_env(directory) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, **{MANAGED_DIR_ENV: str(directory)})


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _memory_dir(config: Path, key: str) -> Path:
    memory = config / "projects" / key / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    return memory


def _by_key(shape: dict) -> dict:
    return {entry["key"]: entry for entry in shape["memory_dirs"]}


DESCRIPTION = "a fact worth keeping"
# QUOTED because it holds `": "`, which is 50 of the 72 harness-written
# memories on the capture host and the case the recorded length was wrong for.
QUOTED = "about memory: a fact"


def _tree(tmp_path: Path) -> Path:
    """One config directory holding every state the shape distinguishes.

    A normal directory with an index and two memories, an index-only one, one
    whose `memory` is empty, one whose `memory` links somewhere else, one
    whose PROJECT directory is a link, one holding a memory FILE that is a
    link, and one whose `memory` links to another directory under the same
    config root — plus a project directory with no `memory` at all, which 5 of
    the 3,923 entries on the machine the NFS fixture came from are. The bulk
    case there is the one above it: 3,918 hold a `memory` directory, and 3,899
    of those hold nothing.

    The last three are named to sort AFTER the others, because pseudonyms are
    assigned by first appearance over sorted keys: a name inserted in the
    middle renumbers every assertion below it and says nothing.
    """
    config = tmp_path / "config"
    normal = _memory_dir(config, "-h-u-git-app")
    _write(
        normal / "kept.md",
        f"---\nname: kept\ndescription: {DESCRIPTION}\n"
        "metadata:\n  type: project\n---\n\nbody\n",
    )
    _write(normal / "plain.md", "no frontmatter at all\n")
    _write(
        normal / "MEMORY.md",
        "# Memory index\n\n- [a](kept.md) — hook\n- [b](gone.md) — hook\n",
    )
    index_only = _memory_dir(config, "-h-u-git-empty")
    _write(index_only / "MEMORY.md", "# Memory index\n\n- [a](gone.md) — hook\n")
    _memory_dir(config, "-h-u-git-bare")
    outside = tmp_path / "elsewhere"
    _write(
        outside / "linked.md",
        f'---\ndescription: "{QUOTED}"\ntype: user\n---\n\nbody\n',
    )
    linked = config / "projects" / "-h-u-git-linked"
    linked.mkdir(parents=True)
    os.symlink(outside, linked / "memory")
    (config / "projects" / "-h-u-git-none").mkdir(parents=True)
    # A project directory reached through a link somebody else made. The
    # package's inventory carries this as its own flag, and a copy has to
    # answer it separately from the memory directory's.
    elsewhere_project = tmp_path / "elsewhere-project"
    _write(elsewhere_project / "memory" / "one.md", "x\n")
    os.symlink(elsewhere_project, config / "projects" / "-h-u-git-plink")
    # A memory FILE that is a link: its bytes live outside the directory being
    # copied, which is the third of the three link states.
    file_linked = _memory_dir(config, "-h-u-git-qlink")
    _write(file_linked / "real.md", "x\n")
    os.symlink(outside / "linked.md", file_linked / "link.md")
    # And a memory directory linked to somewhere INSIDE the config root: two
    # projects then share one set of memories, which is a shape a rebuilt tree
    # has to be able to hold.
    shared = config / "shared"
    _write(shared / "shared.md", "x\n")
    in_shape = config / "projects" / "-h-u-git-rlink"
    in_shape.mkdir(parents=True)
    os.symlink(shared, in_shape / "memory")
    return config


def test_a_shape_round_trips_and_says_what_the_tree_actually_holds(tmp_path) -> None:
    """The capture, read back, against the filesystem it was taken from.

    Every number here is checked against the tree rather than against a literal
    somebody typed: the sizes come from `stat`, the description lengths from
    the strings planted above. A test that compared the output to a
    hand-written expectation would be a test of the expectation.
    """
    config = _tree(tmp_path)
    out = tmp_path / "shape.json"
    # Pointed at an empty directory, so `settings` below is a statement about
    # the capture rather than about whether this runner has a managed file.
    env = _managed_env(tmp_path / "no-managed")
    written = _run("--config-dir", str(config), "--out", str(out), env=env)
    assert written.returncode == 0, written.stdout + written.stderr
    shape = json.loads(out.read_text(encoding="utf-8"))
    # The same capture through the other exit: a shape is a file that gets
    # committed and a stream that gets piped over ssh, and the two have to be
    # the same document.
    assert shape == _shape("--config-dir", str(config), env=env)

    assert shape["schema"] == 1
    assert shape["tool"] == "harness_shape"
    assert shape["anonymised"] is True
    # No `$HOME` lookup anywhere: a temporary config directory has no version
    # and no install method, and a null here is what proves the capture did not
    # reach for the operator's own.
    assert shape["harness"] == {"version_hint": None, "install": None}
    # The counterpart, and the reason it needs a seam: `managed` is the one
    # scope read from a machine path, so on a host that has such a file this
    # capture of a temporary tree would carry it.
    assert shape["settings"] == {}
    assert shape["projects_total"] == 8
    assert shape["memory_dirs_total"] == 7
    assert shape["skipped"] == 0
    assert shape["read_errors"] == 0
    # The bare `memory` directory holds no `*.md`, so it is counted and not
    # listed; the other six are listed, index-only included.
    assert len(shape["memory_dirs"]) == 6

    listed = _by_key(shape)
    assert set(listed) == {
        "-s1-s2-s3-s4", "-s1-s2-s3-s5", "-s1-s2-s3-s6",
        "-s1-s2-s3-s7", "-s1-s2-s3-s8", "-s1-s2-s3-s9",
    }
    normal = listed["-s1-s2-s3-s4"]
    assert normal["key_len"] == len("-h-u-git-app")
    assert normal["is_symlink"] is False
    assert normal["project_is_symlink"] is False
    assert normal["lock_age_s"] is None
    assert normal["index"] == {"rows": 2, "dangling_rows": 1, "truncated": False}
    source = config / "projects" / "-h-u-git-app" / "memory"
    files = {entry["name"]: entry for entry in normal["files"]}
    assert set(files) == {"MEMORY.md", "m1.md", "m2.md"}
    assert files["m1.md"] == {
        "name": "m1.md",
        "size": (source / "kept.md").stat().st_size,
        "is_symlink": False,
        "has_frontmatter": True,
        "has_description": True,
        "description_len": len(DESCRIPTION),
        "has_name": True,
        "has_type": True,
        "frontmatter_truncated": False,
        "unreadable": False,
    }
    assert files["m2.md"]["has_frontmatter"] is False
    assert files["m2.md"]["description_len"] is None
    assert files["m2.md"]["size"] == (source / "plain.md").stat().st_size

    index_only = listed["-s1-s2-s3-s5"]
    assert [entry["name"] for entry in index_only["files"]] == ["MEMORY.md"]
    assert index_only["index"] == {"rows": 1, "dangling_rows": 1, "truncated": False}

    linked = listed["-s1-s2-s3-s6"]
    assert linked["is_symlink"] is True
    assert linked["index"] is None
    assert len(linked["files"]) == 1
    # WITHOUT the quotes, which is how the checker counts before deciding the
    # >155-character rule — see the description case below.
    assert linked["files"][0]["description_len"] == len(QUOTED)
    assert linked["files"][0]["has_type"] is True
    assert linked["files"][0]["has_name"] is False

    # The three link states, each answered separately.
    project_linked = listed["-s1-s2-s3-s7"]
    assert project_linked["project_is_symlink"] is True
    assert project_linked["is_symlink"] is False
    file_linked = listed["-s1-s2-s3-s8"]
    by_name = {entry["name"]: entry for entry in file_linked["files"]}
    assert by_name["m6.md"]["is_symlink"] is False
    assert by_name["m5.md"]["is_symlink"] is True
    # AND NEVER OPENED, which the record states rather than answering the
    # frontmatter question as if it had: the size is the link's rather than
    # the file's, and every fact a read would have produced is null.
    assert by_name["m5.md"]["unreadable"] is True
    assert by_name["m5.md"]["has_frontmatter"] is None
    assert by_name["m5.md"]["description_len"] is None
    assert by_name["m5.md"]["size"] == (
        config / "projects" / "-h-u-git-qlink" / "memory" / "link.md"
    ).lstat().st_size
    # The memory directory shared through a link inside the config root: its
    # own flag is set and its files are the shared ones.
    shared = listed["-s1-s2-s3-s9"]
    assert shared["is_symlink"] is True
    assert [item["name"] for item in shared["files"]] == ["m7.md"]


def test_the_same_tree_captures_to_the_same_bytes_twice(tmp_path) -> None:
    """Pseudonyms are assigned by first appearance over sorted keys, so a
    second capture of an unchanged tree is a zero-line diff. A fixture nobody
    can re-derive is a fixture nobody can review."""
    config = _tree(tmp_path)
    first = _run("--config-dir", str(config))
    second = _run("--config-dir", str(config))
    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout


def test_the_lock_is_read_from_either_place_it_has_been_seen(tmp_path) -> None:
    """`lock_age_s` is how "consolidation is armed" and "consolidation ran an
    hour ago" are told apart, and the file has been seen beside the memory
    directory as well as inside it."""
    config = tmp_path / "config"
    inner = _memory_dir(config, "-a")
    _write(inner / "one.md", "x\n")
    stale = time.time() - 7200
    lock = inner / cli_doctor.CONSOLIDATE_LOCK
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (stale, stale))
    outer = _memory_dir(config, "-b")
    _write(outer / "one.md", "x\n")
    beside = outer.parent / cli_doctor.CONSOLIDATE_LOCK
    beside.write_text("", encoding="utf-8")
    os.utime(beside, (stale, stale))
    listed = _by_key(_shape("--config-dir", str(config), "--raw"))
    assert 7100 < listed["-a"]["lock_age_s"] < 7300
    assert 7100 < listed["-b"]["lock_age_s"] < 7300


# LONG, and that is the point rather than a flourish. Every sentinel in this
# suite used to be under 21 characters, and a one-line weakening of the form
# `return name if len(name) >= 20 else pseudonym` therefore survived the whole
# file — while long names are exactly where real identity lives, because a
# repository called `worktree_discipline_shared_checkouts` says more about
# somebody than `app` does. 28 characters, alphanumeric so it survives the
# harness's own key sanitiser unchanged, and it is planted in EVERY row of an
# anonymised shape that can hold a free string.
SENTINEL = "loxodontaafricanaberthae2026"
MARKET = SENTINEL + "market"
# The same length and no digits, because a hook key that is letters and
# nothing else is what an organisation's own gate looks like — and the rule
# that used to guard that row admitted every one of them. The digits in
# SENTINEL are why it could not see this: `Gate--<sentinel>` failed the old
# pattern twice over, and no case planted a name that passed it.
ALPHA_SENTINEL = "loxodontaafricanaberthaegate"


def _sentinel_tree(tmp_path) -> Path:
    """One config directory with the sentinel in every string-bearing row.

    Ten of them: a project key, a SINGLE-SEGMENT project key, a file name, a
    description, a body, an index row, the auto-memory directory, one of the
    two switches, a hook event key, and a plugin with its marketplace — plus
    the two `harness` rows, a `versions/` directory name and an
    `installMethod`. The list is the point: `assert SENTINEL not in stdout` is
    one assertion, and what makes it a gate is how many places it reaches.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, f"-Users-{SENTINEL}-src-{SENTINEL}")
    _write(
        memory / f"{SENTINEL}-note.md",
        f"---\nname: n\ndescription: about {SENTINEL}\n---\n\n"
        f"the body mentions {SENTINEL}\n",
    )
    _write(
        memory / "MEMORY.md",
        f"# Memory index\n\n- [t]({SENTINEL}-note.md) — hook\n",
    )
    # A key with NO separator in it, which a segment-wise pseudonymiser can be
    # weakened to wave through: the harness spells project keys as sanitised
    # absolute paths, so every real one starts with `-`, and a rule that only
    # looks at multi-segment keys is untestable against a real capture.
    bare = _memory_dir(config, SENTINEL)
    _write(bare / "one.md", "x\n")
    # `versions/` wins over `lastReleaseNotesSeen`, so the directory name is
    # the version row this tree exercises; the other source is pinned by
    # `test_a_version_that_is_not_one_never_travels`.
    (tmp_path / ".local" / "share" / "claude" / "versions" / f"2.1.263-{SENTINEL}").mkdir(
        parents=True
    )
    _write(
        config / ".claude.json",
        json.dumps({"installMethod": f"/Users/{SENTINEL}/.local/bin/claude"}),
    )
    _write(
        config / "settings.json",
        json.dumps(
            {
                # NOT a boolean. The harness reads any non-null as on, so an
                # adopter who typed a path here has a working switch and a
                # value this tool has no business carrying.
                "autoMemoryEnabled": f"/Users/{SENTINEL}/notes",
                "autoDreamEnabled": True,
                # Not `/home/…`: a rule that anonymises one platform's home
                # and passes the other's through is the exact weakening this
                # row exists to catch.
                "autoMemoryDirectory": f"/Users/{SENTINEL}/Library/memory",
                "hooks": {
                    "UserPromptSubmit": [{"hooks": [{"command": SENTINEL}]}],
                    f"Gate--{SENTINEL}": [],
                    ALPHA_SENTINEL: [],
                },
                "enabledPlugins": {f"{SENTINEL}@{MARKET}": True},
            }
        ),
    )
    return config


def test_a_name_in_the_tree_never_reaches_an_anonymised_shape(tmp_path) -> None:
    """The binding rule, exercised in every place a name can hide.

    `--raw` is asserted to LEAK the same sentinel, because a leak test that
    passes against empty output is a leak test that will pass forever.
    """
    config = _sentinel_tree(tmp_path)
    anonymised = _run("--config-dir", str(config))
    assert anonymised.returncode == 0, anonymised.stderr
    assert SENTINEL not in anonymised.stdout
    assert SENTINEL not in anonymised.stderr
    assert ALPHA_SENTINEL not in anonymised.stdout

    shape = json.loads(anonymised.stdout)
    listed = _by_key(shape)
    entry = listed["-s1-s2-s3-s2"]
    # And what it kept instead is the shape: the length of the key, the number
    # of segments, the length of the description, the size of the file.
    assert entry["key_len"] == len(f"-Users-{SENTINEL}-src-{SENTINEL}")
    names = sorted(item["name"] for item in entry["files"])
    assert names == ["MEMORY.md", "m1.md"]
    memories = [item for item in entry["files"] if item["name"] != "MEMORY.md"]
    assert memories[0]["description_len"] == len(f"about {SENTINEL}")
    # The single-segment key is a pseudonym too — the SAME one, because the
    # segment table is shared and a shape's whole claim about keys is that two
    # of them shared a segment, never which.
    assert listed["s2"]["key_len"] == len(SENTINEL)
    assert shape["harness"] == {"version_hint": None, "install": "other"}
    user = shape["settings"]["user"]
    assert user["memory_keys"] == {
        harness_memory.ENABLED_KEY: "<set>",
        harness_memory.DIRECTORY_KEY: "<path>",
        harness_memory.DREAM_KEY: True,
    }
    # Two pseudonyms and a real event name, sorted as OUTPUT: `Gate--<sentinel>`
    # sorts first among the real keys and last here, which is the point.
    assert user["hooks"] == ["UserPromptSubmit", "h1", "h2"]
    # And the artifact gate's own vocabulary, which was this rule restated
    # rather than re-typed and so could not see it being the wrong rule.
    assert ALPHA_SENTINEL not in HOOK_OK
    assert user["plugins"] == ["p1@q1"]

    raw = _run("--config-dir", str(config), "--raw")
    assert raw.returncode == 0, raw.stderr
    assert SENTINEL in raw.stdout, "the anonymised pass proved nothing"
    # And raw keeps them AS THEY ARE, which is what makes the anonymised pass
    # above a statement about the anonymiser rather than about the reader.
    raw_shape = json.loads(raw.stdout)
    assert raw_shape["harness"] == {
        "version_hint": f"2.1.263-{SENTINEL}",
        "install": f"/Users/{SENTINEL}/.local/bin/claude",
    }
    assert f"Gate--{SENTINEL}" in raw_shape["settings"]["user"]["hooks"]
    assert ALPHA_SENTINEL in raw_shape["settings"]["user"]["hooks"]


def test_a_switch_the_harness_reads_as_on_is_recorded_as_on(tmp_path) -> None:
    """The privacy half of this row was right and the value chosen for it
    inverted the fact it was keeping.

    `harness_memory.switch` treats an explicit null as ABSENCE — the harness's
    own `!= null` test is what makes it one — so a shape that recorded null
    for a switch an adopter had set to a path described a machine with the
    feature OFF, which is the opposite of the machine that was captured. The
    comparison is against `switch` itself rather than against a literal,
    because that function is what decides the question.

    And the same road runs the other way. A key DECLARED null is one `switch`
    reads as absent, so recording the placeholder for it says the feature was
    on where the captured machine had it off — the identical inversion, in the
    identical field. Every one of the three keys is put through both, because
    the directory key takes its own branch and inverted the same way.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    typed = f"/Users/{SENTINEL}/notes"
    written = {
        harness_memory.ENABLED_KEY: typed,
        harness_memory.DREAM_KEY: False,
        harness_memory.DIRECTORY_KEY: typed,
    }
    _write(config / "settings.json", json.dumps(written))
    env = _managed_env(tmp_path / "no-managed")
    shape = _shape("--config-dir", str(config), env=env)
    recorded = shape["settings"]["user"]["memory_keys"]
    assert SENTINEL not in json.dumps(recorded)

    nulled = dict.fromkeys(written)
    _write(config / "settings.json", json.dumps(nulled))
    recorded_null = _shape("--config-dir", str(config), env=env)["settings"][
        "user"
    ]["memory_keys"]
    # DECLARED and null, which is not the same document as a file that never
    # mentioned the key: the second omits it here.
    assert set(recorded_null) == set(nulled)

    class _Scope:
        def __init__(self, data):
            self.scope = "user"
            self.data = data

    for source, seen in ((written, recorded), (nulled, recorded_null)):
        for key, captured in source.items():
            was, _ = harness_memory.switch([_Scope({key: captured})], key)
            rebuilt, _ = harness_memory.switch([_Scope({key: seen[key]})], key)
            assert (was is None) == (rebuilt is None), (key, captured, seen[key])
            assert bool(was) == bool(rebuilt), (key, captured, seen[key])


def test_a_version_that_is_not_one_never_travels(tmp_path) -> None:
    """Both sources of `version_hint`, and the install method beside it.

    A version hint is whichever `versions/` entry sorts highest, and
    `lastReleaseNotesSeen` only when there is no such directory — so a rule
    applied to one source and not the other is a rule with a hole nobody
    walking the happy path would find. A stock value has to SURVIVE, which is
    the half that a redact-everything answer would fail.
    """
    cases = ((f"2.1.263-{SENTINEL}", None), ("2.1.258", "2.1.258"))
    for index, (value, kept) in enumerate(cases):
        for source in ("versions", "notes"):
            root = tmp_path / f"{source}{index}"
            config = root / "config"
            config.mkdir(parents=True)
            claude_json = {"installMethod": "native"}
            if source == "versions":
                (root / ".local" / "share" / "claude" / "versions" / value).mkdir(
                    parents=True
                )
            else:
                claude_json["lastReleaseNotesSeen"] = value
            _write(config / ".claude.json", json.dumps(claude_json))
            shape = _shape("--config-dir", str(config))
            assert shape["harness"] == {"version_hint": kept, "install": "native"}, (
                source, value,
            )
    # AND ONE PATCHED DIRECTORY DOES NOT NULL THE WHOLE ROW. The highest entry
    # under `versions/` is the hand-built one, and rejecting it outright threw
    # away the releases installed beside it — a beta next to a release is an
    # ordinary machine, and the fact it had a version is one a rebuilt tree
    # can use.
    root = tmp_path / "beside"
    config = root / "config"
    config.mkdir(parents=True)
    for name in ("2.1.100", "2.1.258", f"2.1.999-{SENTINEL}"):
        (root / ".local" / "share" / "claude" / "versions" / name).mkdir(parents=True)
    out = _run("--config-dir", str(config))
    assert out.returncode == 0, out.stderr
    assert SENTINEL not in out.stdout
    assert json.loads(out.stdout)["harness"]["version_hint"] == "2.1.258"


def test_raw_refuses_to_write_where_a_fixture_would_be_committed(tmp_path) -> None:
    """The one mistake the tool exists to prevent is a `--raw` redirect that
    happened to point at the repository, so the refusal is the guard rather
    than the docstring. Both spellings of a checkout: a `.git` directory, and
    the `.git` FILE a linked worktree has — which is what a worktree-per-unit
    workflow writes in all day."""
    for name, make in (("dir", Path.mkdir), ("file", Path.touch)):
        tree = tmp_path / name
        (tree / "deep").mkdir(parents=True)
        make(tree / ".git")
        out = tree / "deep" / "shape.json"
        refused = _run("--config-dir", str(tmp_path), "--raw", "--out", str(out))
        assert refused.returncode == 2, refused.stdout + refused.stderr
        assert "git worktree" in refused.stderr
        assert not out.exists(), "it refused and wrote anyway"
    # Outside a checkout it writes, and to stdout it always writes: reading
    # your own machine is what the flag is for.
    outside = tmp_path / "shape.json"
    assert _run("--config-dir", str(tmp_path), "--raw", "--out", str(outside)).returncode == 0
    assert json.loads(outside.read_text(encoding="utf-8"))["anonymised"] is False
    assert _run("--config-dir", str(tmp_path), "--raw").returncode == 0


def test_the_headline_docstring_states_the_exemption_the_guard_really_has(
    tmp_path,
) -> None:
    """A guarantee is read or executed, and this one is both.

    The headline said `--raw` was "refused outright" when its output would
    land inside a git worktree. A PIPE is exempt, deliberately and correctly —
    the far end of the documented ssh recipe is a machine this process cannot
    ask about — and the sentence saying so was 590 lines further down, in the
    docstring of the function that implements it. The headline is what an
    operator reads before trusting the guard.
    """
    headline = _tool_module().__doc__ or ""
    claim = headline[: headline.index("AND NO FREE STRING")]
    assert "refused outright" not in claim, claim
    assert "A pipe is not visible to it" in claim, claim
    assert "--out" in claim and "redirected straight at a file" in claim, claim

    # And the exemption is real, which is why the prose has to say it.
    tree = tmp_path / "repo"
    (tree / ".git").mkdir(parents=True)
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    piped = _run("--config-dir", str(config), "--raw")
    assert piped.returncode == 0, piped.stderr
    assert json.loads(piped.stdout)["anonymised"] is False
    redirected = tree / "oops.json"
    with redirected.open("w") as handle:
        refused = subprocess.run(
            [sys.executable, str(TOOL), "--config-dir", str(config), "--raw"],
            stdout=handle, stderr=subprocess.PIPE, text=True, timeout=300,
        )
    assert refused.returncode == 2, refused.stderr
    assert redirected.read_text(encoding="utf-8") == ""


def test_raw_refuses_a_work_tree_that_holds_no_dot_git_at_all(tmp_path) -> None:
    """Which directory is a work tree is a fact about a repository somewhere
    else, so the last word is git's.

    A work tree attached to a BARE repository — the dotfiles pattern, and a
    `GIT_DIR` exported over ssh — has no `.git` at any level, and a name walk
    looking for one calls it open ground. `git status` there lists the shape
    as untracked in a real checkout, one `git add -A` from the single mistake
    this tool exists to prevent. Both spellings: the work tree declared in the
    environment, and the work tree declared only in the repository's config,
    which the environment cannot see and only git can answer.
    """
    for name, configure in (("declared", None), ("configured", "core.worktree")):
        work = tmp_path / name
        work.mkdir()
        bare = tmp_path / (name + ".git")
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, timeout=300)
        env = dict(os.environ, GIT_DIR=str(bare))
        env.pop("GIT_WORK_TREE", None)
        if configure is None:
            env["GIT_WORK_TREE"] = str(work)
        else:
            for key, value in (("core.bare", "false"), (configure, str(work))):
                subprocess.run(
                    ["git", "--git-dir", str(bare), "config", key, value],
                    check=True, timeout=300,
                )
        assert not (work / ".git").exists(), "the case is only about trees without one"
        out = work / "leak.json"
        refused = _run("--config-dir", str(tmp_path), "--raw", "--out", str(out), env=env)
        assert refused.returncode == 2, refused.stdout + refused.stderr
        assert "git worktree" in refused.stderr
        assert not out.exists(), "real names landed in a work tree"

    # And a plain directory with no git anywhere near it is still what the
    # flag is for: the question is asked, and the answer is no.
    open_ground = tmp_path / "mine"
    open_ground.mkdir()
    allowed = _run(
        "--config-dir", str(tmp_path), "--raw", "--out", str(open_ground / "ok.json")
    )
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert json.loads(
        (open_ground / "ok.json").read_text(encoding="utf-8")
    )["anonymised"] is False


def test_a_declared_work_tree_is_matched_by_inode_when_git_cannot_answer(
    tmp_path, monkeypatch,
) -> None:
    """The `GIT_WORK_TREE` branch is the one that answers when git will not,
    and every run that reached it had git answering yes on its own.

    The case above exports `GIT_DIR` beside the work tree, so `git rev-parse`
    succeeds and the refusal it asserts arrives whether or not the environment
    is consulted at all — the inode comparison could be deleted with the suite
    green. Here git's answer is no, and the two properties the comparison
    exists for are the ones a string test cannot give: a destination one level
    INSIDE the declared tree, and that tree named through a different path.
    """
    module = _tool_module()
    work = tmp_path / "work"
    (work / "below").mkdir(parents=True)
    spelling = tmp_path / "other-name"
    os.symlink(work, spelling)
    monkeypatch.setattr(module, "_git_says_worktree", lambda directory: False)
    monkeypatch.setenv("GIT_WORK_TREE", str(spelling))
    assert module._inside_worktree(str(work)) is True
    assert module._inside_worktree(str(work / "below")) is True
    # And the branch answers no as readily: a directory that is not under the
    # declared tree is the open ground the flag is for.
    open_ground = tmp_path / "mine"
    open_ground.mkdir()
    assert module._inside_worktree(str(open_ground)) is False

    # End to end, in the state the docstring names: no git on PATH, so the
    # walk's own answer is the only one there is.
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    env = dict(os.environ, GIT_WORK_TREE=str(spelling), PATH="")
    env.pop("GIT_DIR", None)
    out = work / "below" / "leak.json"
    refused = _run("--config-dir", str(config), "--raw", "--out", str(out), env=env)
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert "git worktree" in refused.stderr
    assert not out.exists(), "real names landed one level inside a declared tree"


def test_raw_refuses_a_redirect_from_inside_a_checkout(tmp_path) -> None:
    """The spelling the tool's own docstring names, which had no guard.

    "A debugging run whose redirect happened to point at the repository" is
    `--raw > somewhere`, and the refusal only ran when `--out` was given —
    the spelling where somebody typed the destination and could read it back.
    A regular-file stdout is the redirect; a terminal, a pipe (`| jq` and the
    documented ssh flow) and `/dev/null` are not, and all three keep working.
    """
    tree = tmp_path / "repo"
    (tree / "tests" / "data").mkdir(parents=True)
    (tree / ".git").mkdir()
    config = tmp_path / "config"
    config.mkdir()
    argv = [sys.executable, str(TOOL), "--config-dir", str(config), "--raw"]
    target = tree / "tests" / "data" / "oops.json"
    with target.open("w") as handle:
        refused = subprocess.run(
            argv, stdout=handle, stderr=subprocess.PIPE, text=True,
            cwd=str(tree), timeout=300,
        )
    assert refused.returncode == 2, refused.stderr
    # It names the file, because fd 1 can be asked what it is; the message
    # that names only the working directory is the fallback for a kernel that
    # will not answer.
    assert "git worktree" in refused.stderr
    assert str(target) in refused.stderr
    assert target.read_text(encoding="utf-8") == "", "it refused and wrote anyway"

    # The three spellings that are not a redirect, from the same directory.
    piped = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(tree), timeout=300
    )
    assert piped.returncode == 0, piped.stderr
    with open(os.devnull, "w") as sink:
        nulled = subprocess.run(
            argv, stdout=sink, stderr=subprocess.PIPE, text=True,
            cwd=str(tree), timeout=300,
        )
    assert nulled.returncode == 0, nulled.stderr
    # And a redirect from OUTSIDE a checkout is what the flag is for.
    mine = tmp_path / "mine.json"
    with mine.open("w") as handle:
        allowed = subprocess.run(
            argv, stdout=handle, stderr=subprocess.PIPE, text=True,
            cwd=str(tmp_path), timeout=300,
        )
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(mine.read_text(encoding="utf-8"))["anonymised"] is False


def test_the_redirect_is_judged_by_where_it_lands_not_where_it_started(
    tmp_path,
) -> None:
    """The working directory was standing in for the destination, and it is
    wrong in both directions.

    One step outside the checkout is all it took: `cd /tmp && harness_shape
    --raw > repo/tests/data/oops.json` wrote real keys, real file names and a
    real install path into the repository and exited 0. The mirror image
    refused a capture of your own machine redirected safely outside the tree,
    which is what the flag is for. fd 1 knows its own name — `F_GETPATH` on
    darwin, `/proc/self/fd` on linux — so the question can be asked about the
    file that is actually being written.
    """
    tree = tmp_path / "repo"
    (tree / "tests" / "data").mkdir(parents=True)
    (tree / ".git").mkdir()
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    argv = [sys.executable, str(TOOL), "--config-dir", str(config), "--raw"]

    landing_inside = tree / "tests" / "data" / "oops.json"
    with landing_inside.open("w") as handle:
        refused = subprocess.run(
            argv, stdout=handle, stderr=subprocess.PIPE, text=True,
            cwd=str(tmp_path), timeout=300,
        )
    assert refused.returncode == 2, refused.stderr
    assert "git worktree" in refused.stderr
    assert landing_inside.read_text(encoding="utf-8") == "", "it refused and wrote"

    landing_outside = tmp_path / "mine.json"
    with landing_outside.open("w") as handle:
        allowed = subprocess.run(
            argv, stdout=handle, stderr=subprocess.PIPE, text=True,
            cwd=str(tree), timeout=300,
        )
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(
        landing_outside.read_text(encoding="utf-8")
    )["anonymised"] is False


def test_the_fallback_for_a_kernel_that_will_not_name_fd_one_still_refuses(
    tmp_path, monkeypatch,
) -> None:
    """The branch above answers on darwin and on linux, which is every machine
    this suite runs on — so the cwd fallback beneath it was executed by
    nothing, on any platform, and could have been deleted with the suite
    green.

    The seam is the answer, not a flag: `_stdout_destination` returning None
    is exactly the kernel the fallback exists for, and the two directions are
    the whole of what it decides. It over-refuses on purpose — a redirect
    started outside a checkout and landing inside one is invisible to it —
    so the case it does catch is the one worth holding it to.
    """
    module = _tool_module()
    tree = tmp_path / "repo"
    (tree / ".git").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    argv = ["--config-dir", str(config), "--raw"]
    monkeypatch.setattr(module, "_stdout_destination", lambda: None)

    for cwd, code in ((tree, 2), (outside, 0)):
        errors = io.StringIO()
        landing = tmp_path / f"shape-{cwd.name}.json"
        monkeypatch.setattr(module.sys, "stderr", errors)
        # A real file, because the tool asks fd 1 what it is at call time.
        with landing.open("w", encoding="utf-8") as handle:
            monkeypatch.setattr(module.sys, "stdout", handle)
            monkeypatch.chdir(cwd)
            assert module.main(argv) == code, errors.getvalue()
        if code == 2:
            assert "refuses a redirect to a file" in errors.getvalue()
            assert landing.read_text(encoding="utf-8") == "", "it refused and wrote"
        else:
            assert errors.getvalue() == ""
            assert json.loads(
                landing.read_text(encoding="utf-8")
            )["anonymised"] is False


def _recorded_lookups(module, monkeypatch) -> list:
    """Every question `_index` puts to the filesystem, as `(name, dir_fd)`.

    BOTH spellings are recorded and not only the one in use: a lookup that
    names a whole path is exactly what reaching through an intermediate
    component looks like, and a recorder watching the descriptor call alone
    would pass a walk that had stopped making it.
    """
    asked = []
    stat, lexists, open_dir = module.os.stat, module.os.path.lexists, module._open_dir

    def recording_stat(path, *args, **kwargs):
        asked.append((path, kwargs.get("dir_fd")))
        return stat(path, *args, **kwargs)

    def recording_lexists(path):
        asked.append((path, None))
        return lexists(path)

    def recording_open_dir(path, dir_fd=None):
        asked.append((path, dir_fd))
        return open_dir(path, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "stat", recording_stat)
    monkeypatch.setattr(module.os.path, "lexists", recording_lexists)
    monkeypatch.setattr(module, "_open_dir", recording_open_dir)
    return asked


def test_an_index_row_is_judged_without_leaving_the_directory(tmp_path) -> None:
    """`dangling_rows` used to be decided by joining adopter-authored text onto
    the memory directory and stat-ing whatever came out.

    Two things wrong with one line: the walk left the directory — over ssh
    under `sudo -n`, so as root on somebody else's machine — and `/etc/passwd`
    exists, so that row scored as SATISFIED and the count was wrong in the
    direction that hides the problem.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    _write(memory / "real.md", "x\n")
    _write(
        memory / "MEMORY.md",
        "# Memory index\n\n"
        "- [a](real.md) — hook\n"
        "- [b](/etc/passwd) — hook\n"
        "- [c](../../../../etc/hosts) — hook\n"
        "- [d](nope.md) — hook\n",
    )
    entry = _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]
    assert entry["index"] == {"rows": 4, "dangling_rows": 3, "truncated": False}


def test_a_row_one_directory_down_is_looked_at_rather_than_written_off(
    tmp_path, monkeypatch,
) -> None:
    """The rule is about targets that ESCAPE, and it was written as "holds a
    separator": a tiered index — `hot/beads.md`, one directory down and inside
    the directory being walked — read as twelve dangling rows out of twelve on
    the machine the darwin fixture came from, which is a corpus a materialiser
    would rebuild with twelve dead rows the machine does not have.

    What must not happen is the stat that made this a rule: a row naming
    `/etc/hosts` is still scored without asking the filesystem about it.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    (memory / "hot").mkdir()
    _write(memory / "sibling.md", "x\n")
    _write(memory / "hot" / "beads.md", "x\n")
    _write(memory.parent / "sibling.md", "x\n")
    _write(
        memory / "MEMORY.md",
        "- [a](sibling.md) — hook\n"
        "- [b](hot/beads.md) — hook\n"
        "- [c](/etc/hosts) — hook\n"
        "- [d](../sibling.md) — hook\n"
        "- [e](nowhere.md) — hook\n",
    )
    entry = _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]
    # `/etc/hosts`, `../sibling.md`, `nowhere.md`. The two contained rows are
    # the ones the count changed for.
    assert entry["index"] == {"rows": 5, "dangling_rows": 3, "truncated": False}
    module = _tool_module()
    asked = _recorded_lookups(module, monkeypatch)
    assert module._index(str(memory), ["MEMORY.md", "sibling.md"]) == (
        {"rows": 5, "dangling_rows": 3, "truncated": False}, 0,
    )
    # WHICH ROWS the filesystem is asked about, which is the half of this the
    # count alone does not say: `hot/beads.md` one directory down and
    # `nowhere.md`, and neither `/etc/hosts` nor `../sibling.md`. Component by
    # component from the directory's own descriptor, so `hot` is a question of
    # its own and the leaves arrive under their last component alone.
    assert [name for name, dir_fd in asked if dir_fd is not None] == [
        "hot", "beads.md", "nowhere.md",
    ]


def test_a_row_is_not_reached_through_a_linked_component(
    tmp_path, monkeypatch,
) -> None:
    """The escape rule judges the STRING, and `lexists` follows every
    component but the last.

    So `link-out/x.md` passes the rule — nothing about that text escapes —
    and the lookup then walks through `link-out` and stats a file outside the
    directory being captured: the stat the rule exists to prevent, one
    component over. The row also scored as SATISFIED, on the strength of
    somebody else's file existing.
    """
    config = tmp_path / "config"
    outside = tmp_path / "elsewhere"
    _write(outside / "x.md", "x\n")
    memory = _memory_dir(config, "-a")
    _write(memory / "own.md", "x\n")
    os.symlink(outside, memory / "link-out")
    # And the row a walk by descriptor must still count as PRESENT: a dead
    # link one directory down is a file that is there to be moved, which is
    # what the row is about — so the last component is not followed either.
    (memory / "hot").mkdir()
    os.symlink(tmp_path / "gone.md", memory / "hot" / "dead.md")
    _write(
        memory / "MEMORY.md",
        "- [a](own.md) — hook\n"
        "- [b](link-out/x.md) — hook\n"
        "- [c](hot/dead.md) — hook\n",
    )
    entry = _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]
    assert entry["index"] == {"rows": 3, "dangling_rows": 1, "truncated": False}

    module = _tool_module()
    asked = _recorded_lookups(module, monkeypatch)
    assert module._index(str(memory), ["MEMORY.md", "own.md"]) == (
        {"rows": 3, "dangling_rows": 1, "truncated": False}, 0,
    )
    # And nothing was asked by a name at all: every question is one component
    # put to a descriptor the walk already holds, which is what keeps a linked
    # component from being followed rather than what notices afterwards.
    assert asked, "a walk that asks nothing proves nothing"
    for name, dir_fd in asked:
        if dir_fd is None:
            # The one question put by name is the directory being walked.
            assert os.path.realpath(name) == os.path.realpath(str(memory)), name
        else:
            assert os.sep not in name, name
    # `link-out` is asked for as a DIRECTORY and refused, so the row is
    # dangling because the walk stopped rather than because it looked.
    assert ("link-out", None) not in asked
    assert "link-out" in [name for name, dir_fd in asked if dir_fd is not None]


def test_a_row_target_holding_a_nul_byte_is_dangling_and_not_the_end_of_the_run(
    tmp_path,
) -> None:
    """An index row is adopter-authored text and 0x00 is valid UTF-8.

    It survives the decode untouched, the row pattern admits it, and then
    `os.stat` and `os.open` raise `ValueError` rather than `OSError` for a
    path holding one — so a walk guarded on `OSError` alone let one row in one
    project's index end the capture of the WHOLE machine: a traceback, an exit
    1 and no shape, with a second healthy project beside it reported not at
    all.

    A row nothing can look up is a row pointing at nothing, which is the
    answer `os.path.lexists` gave here before the walk was hand-rolled — the
    stdlib guards its own `lstat` with `(OSError, ValueError)` for this.
    """
    config = tmp_path / "config"
    poisoned = _memory_dir(config, "-a")
    _write(poisoned / "real.md", "x\n")
    with open(poisoned / "MEMORY.md", "wb") as fh:
        # Written as BYTES: the NUL is never typed into a path here.
        fh.write(b"# Index\n\n- [a](real.md) - hook\n- [b](evil\x00name.md) - hook\n")
    healthy = _memory_dir(config, "-b")
    _write(healthy / "real.md", "x\n")
    _write(healthy / "MEMORY.md", "# Index\n\n- [a](real.md) - hook\n")

    shape = _by_key(_shape("--config-dir", str(config), "--raw"))
    assert shape["-a"]["index"] == {"rows": 2, "dangling_rows": 1, "truncated": False}
    assert shape["-b"]["index"] == {"rows": 1, "dangling_rows": 0, "truncated": False}


def test_a_tier_this_run_may_not_enter_is_not_a_tier_full_of_dead_rows(
    tmp_path,
) -> None:
    """A directory the capture is refused is not a directory of dangling rows.

    `sudo -n` into an NFS home under root-squash is the deployment this tool
    was written for and the thing that turns a readable tier into EACCES, and
    every errno from the component walk answered the same `False`: a tier at
    mode 000 booked every row beneath it as pointing at nothing, with
    `skipped` and `read_errors` both at zero and an exit 0. Nothing in the
    document said a lookup had been refused, so the number a consumer would
    judge the index on was wrong in the direction that looks like a finding.
    """
    if os.geteuid() == 0:
        pytest.skip("root enters a directory at mode 000")
    config = tmp_path / "config"
    tiers = []
    for key in ("-a", "-b", "-c"):
        memory = _memory_dir(config, key)
        _write(memory / "top.md", "x\n")
        _write(memory / "hot" / "deep.md", "x\n")
        _write(memory / "MEMORY.md", "- [a](top.md)\n- [b](hot/deep.md)\n")
        tiers.append(memory / "hot")
    readable = _shape("--config-dir", str(config), "--raw")
    for tier in tiers:
        tier.chmod(0o000)
    try:
        refused = _shape("--config-dir", str(config), "--raw")
    finally:
        for tier in tiers:
            tier.chmod(0o755)
    counts = [
        (entry["index"]["rows"], entry["index"]["dangling_rows"])
        for entry in refused["memory_dirs"]
    ]
    # THE ROWS ARE STILL COUNTED and NONE of them is dangling: the rows are
    # read off the index, and whether the tier can be entered is a fact about
    # this run rather than about the index.
    assert counts == [(2, 0), (2, 0), (2, 0)], counts
    assert counts == [
        (entry["index"]["rows"], entry["index"]["dangling_rows"])
        for entry in readable["memory_dirs"]
    ], "the tier's mode moved a count it is not about"
    # AND THE REFUSAL IS ON THE DOCUMENT, once per row that was not looked at,
    # in the counter that already means "inside a directory this run reached
    # and could not measure".
    assert refused["read_errors"] == 3
    assert refused["skipped"] == 0
    assert readable["read_errors"] == 0


def test_one_row_short_of_a_descriptor_degrades_the_row_and_not_the_capture(
    tmp_path,
) -> None:
    """The descriptor the row walk needs, taken where the walk releases it.

    Taken one line above the `try`, an `EMFILE` on it reached no handler in
    the walk at all and left by the contract boundary: exit 2 and no shape for
    a machine whose every project had been read, which is the one failure mode
    a capture over a pipe cannot recover from. The table is squeezed to
    exactly one free descriptor, which is the whole width of the window — with
    none the interpreter cannot finish its own imports, and with two the walk
    has room.

    The child is the door the exit-contract table uses, because a descriptor
    table with one entry left cannot be inherited across an `exec` that still
    has imports of its own to take.
    """
    config = tmp_path / "config"
    for key in ("-a", "-b", "-c"):
        memory = _memory_dir(config, key)
        _write(memory / "top.md", "x\n")
        _write(memory / "hot" / "deep.md", "x\n")
        _write(memory / "MEMORY.md", "- [a](top.md)\n- [b](hot/deep.md)\n")
    door = _write(tmp_path / "door.py", _HOSTILE_DOOR)
    run = subprocess.run(
        [sys.executable, str(door), str(TOOL), "squeeze", str(config)],
        capture_output=True, text=True, timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Traceback" not in run.stderr, run.stderr
    shape = json.loads(run.stdout)
    assert len(shape["memory_dirs"]) == 3
    # Every row counted, none of them dangling, and one refusal booked per row
    # the walk had no descriptor for.
    assert [
        (entry["index"]["rows"], entry["index"]["dangling_rows"])
        for entry in shape["memory_dirs"]
    ] == [(2, 0), (2, 0), (2, 0)]
    assert shape["read_errors"] == 3
    assert shape["skipped"] == 0


def test_what_is_opened_is_decided_by_where_the_link_lands(tmp_path) -> None:
    """One rule for every `.md` in a memory directory, the index included.

    The per-file read refused a link outright and then handed `MEMORY.md` —
    the one file the rule most obviously covers — to the index reader, which
    opened it: rows counted off a file outside the capture, over `sudo -n`,
    on a host that was given the directory and not the target. A link back
    into the directory is a different thing, because those bytes are being
    read anyway, and it is the case the harness's own tree produces.
    """
    config = tmp_path / "config"
    outside = tmp_path / "elsewhere"
    _write(outside / "real-index.md", "# i\n\n- [a](m.md) — h\n- [b](gone.md) — h\n")
    _write(outside / "memo.md", "---\nname: theirs\n---\n\nbody\n")

    out = _memory_dir(config, "-a")
    _write(out / "m.md", "x\n")
    os.symlink(outside / "real-index.md", out / "MEMORY.md")
    os.symlink(outside / "memo.md", out / "linked.md")

    back = _memory_dir(config, "-b")
    _write(back / "m.md", "x\n")
    _write(back / "index-real.md", "# i\n\n- [a](m.md) — h\n")
    _write(back / "memo-real.md", "---\nname: ours\n---\n\nbody\n")
    os.symlink("index-real.md", back / "MEMORY.md")
    os.symlink("memo-real.md", back / "linked.md")

    shape = _shape("--config-dir", str(config), "--raw")
    left = _by_key(shape)["-a"]
    files = {item["name"]: item for item in left["files"]}
    # LISTED, with the flag that says why the numbers stop where they do: a
    # null index beside a linked `MEMORY.md` is an index that was not read,
    # and a null index beside no `MEMORY.md` is a directory without one.
    assert sorted(files) == ["MEMORY.md", "linked.md", "m.md"]
    assert files["MEMORY.md"]["is_symlink"] is True
    assert left["index"] is None
    # Null and not `false`: the link out was never opened, so there is no read
    # behind the answer to "does it carry a name".
    assert files["linked.md"]["has_name"] is None

    stayed = _by_key(shape)["-b"]
    kept = {item["name"]: item for item in stayed["files"]}
    assert stayed["index"] == {"rows": 1, "dangling_rows": 0, "truncated": False}
    assert kept["MEMORY.md"]["is_symlink"] is True
    assert kept["linked.md"]["has_name"] is True
    # Nothing outside was measured: the sizes are the links', not the targets'.
    assert files["linked.md"]["size"] != (outside / "memo.md").stat().st_size


def test_frontmatter_cut_at_the_cap_says_so_rather_than_reading_as_none(
    tmp_path,
) -> None:
    """Four false flags, a null length and no counter moved is what a file
    with no frontmatter records — and it was also what a file whose closing
    fence sits past the 64 KB cap recorded.

    The index reader already answers this about its own read. The per-file one
    did not, so the one number a rebuilt corpus cannot reproduce was the one
    nothing said anything about.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    cap = _tool_module().FRONTMATTER_BYTES
    filler = "filler\n" * (cap // 7 + 1)
    _write(memory / "far.md", f"---\nname: far\n{filler}---\n\nbody\n")
    _write(memory / "near.md", "---\nname: near\n---\n\nbody\n")
    files = {
        item["name"]: item
        for item in _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]["files"]
    }
    assert files["far.md"]["size"] > cap
    assert files["far.md"]["has_frontmatter"] is False
    assert files["far.md"]["frontmatter_truncated"] is True
    assert files["near.md"]["frontmatter_truncated"] is False
    # The cap still holds: nothing past it was read to reach that answer.
    assert files["near.md"]["has_name"] is True


def test_the_cap_named_in_bytes_is_counted_in_bytes(tmp_path) -> None:
    """A text stream's `read(n)` counts characters, so the 64 KB cap was 64 K
    CHARACTERS: a head of CJK text read 192 KB off a machine this tool is a
    guest on and reported nothing cut, which is the read the cap exists to
    bound and the one flag that says it happened.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    cap = _tool_module().FRONTMATTER_BYTES
    # Three bytes each: half the cap in CHARACTERS is 1.5x it in BYTES.
    wide = "中" * (cap // 2)
    _write(memory / "wide.md", f"---\nname: wide\n---\n{wide}\n")
    head = "---\nname: ascii\n---\n"
    _write(memory / "ascii.md", head + "a" * (cap - len(head)))
    assert (memory / "wide.md").stat().st_size > cap
    assert len((memory / "wide.md").read_text(encoding="utf-8")) < cap
    files = {
        item["name"]: item
        for item in _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]["files"]
    }
    assert files["wide.md"]["frontmatter_truncated"] is True
    # The frontmatter itself is inside the cap either way, and still read.
    assert files["wide.md"]["has_name"] is True
    # An ASCII file exactly at the cap answers what it always did.
    assert (memory / "ascii.md").stat().st_size == cap
    assert files["ascii.md"]["frontmatter_truncated"] is False
    assert files["ascii.md"]["has_name"] is True


def test_a_pathological_index_is_read_to_the_cap_and_says_so(tmp_path) -> None:
    """Every other read here stops at `FRONTMATTER_BYTES` so one file cannot
    turn a capture into a read of somebody's whole disk. The index was the one
    that did not, and a count taken off a truncated read has to say so or it
    is a number a rebuilt corpus cannot reproduce."""
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    _write(memory / "real.md", "x\n")
    _write(memory / "MEMORY.md", "- [a](real.md) — hook\n" * 6000)
    entry = _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]
    assert entry["index"]["truncated"] is True
    assert 0 < entry["index"]["rows"] < 6000
    assert entry["index"]["dangling_rows"] == 0

    # AND WHERE THE CAP FALLS. The last line read is dropped as a fragment,
    # which is right for a cut in the middle of a row and wrong for one that
    # lands on a newline — there the row is whole, and dropping it under-counts
    # a file by one. Rows of exactly 32 characters divide the cap evenly, so
    # the character at the boundary is the newline.
    cap = _tool_module().FRONTMATTER_BYTES
    stem = "- [%s](real.md)\n"
    row = stem % ("a" * (32 - len(stem % "")))
    assert len(row) == 32
    whole = cap // len(row)
    boundary = _memory_dir(config, "-b")
    _write(boundary / "real.md", "x\n")
    _write(boundary / "MEMORY.md", row * (whole + 1))
    counted = _by_key(_shape("--config-dir", str(config), "--raw"))["-b"]["index"]
    assert counted["truncated"] is True
    assert counted["rows"] == whole, counted


def _destination(name: str, root: Path) -> tuple:
    """One spelling of a `--out` destination, built under `root`.

    Every victim these build lives inside `root / "repo"`, which is a git
    checkout, and every destination is named from outside it. That is what
    lets one table ask both questions at once: whether the write went where
    the checks were answered about, and whether `--raw` can be steered into a
    repository by the same spelling.

    Returns the destination, the exit status it must produce, and the fragment
    of the refusal it must say (None where the write is allowed).
    """
    repo = root / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / ".git").mkdir()
    outside = root / "outside"
    outside.mkdir()
    if name == "a symlink at the name":
        _write(repo / "keep.txt", "keep me\n")
        out = outside / "shape.json"
        os.symlink(repo / "keep.txt", out)
        return out, 2, "refuses to follow one"
    if name == "a symlink one directory up":
        _write(repo / "sub" / "target.json", "keep me too\n")
        os.symlink(repo / "sub", outside / "linkdir")
        return outside / "linkdir" / "target.json", 2, "a symlink stands in this path"
    if name == "a dotdot after a symlinked directory":
        # `abspath` collapses this to `outside/base/plain`, which is a real
        # directory nothing here writes to; the kernel resolves the link and
        # then takes `..` from its target, which is inside the checkout.
        _write(repo / "plain" / "out.json", "keep me three\n")
        (repo / "dir").mkdir()
        (outside / "base" / "plain").mkdir(parents=True)
        os.symlink(repo / "dir", outside / "base" / "link")
        spelling = outside / "base" / "link" / ".." / "plain" / "out.json"
        return spelling, 2, "a symlink stands in this path"
    if name == "a hard link at the name":
        _write(repo / "hardtarget.json", "keep me four\n")
        os.link(repo / "hardtarget.json", outside / "hard.json")
        return outside / "hard.json", 2, "overwrite through one"
    if name == "a dangling symlink at the name":
        os.symlink(repo / "never" / "gone.json", outside / "dangling.json")
        return outside / "dangling.json", 2, "refuses to follow one"
    if name == "a directory made on the way to a refusal":
        # `makedirs` used to run before the refusal, through the very link it
        # was about to refuse.
        os.symlink(repo / "sub", outside / "madelink")
        return outside / "madelink" / "made" / "shape.json", 2, "a symlink stands in"
    if name == "a plain file in a plain directory":
        return _write(outside / "plain" / "shape.json", "old\n"), 2, "already at this name"
    if name == "a fifo at the name":
        (outside / "queue").mkdir()
        os.mkfifo(str(outside / "queue" / "shape.json"))
        return outside / "queue" / "shape.json", 2, "this name is not a file"
    if name == "a directory at the name":
        (outside / "adir" / "shape.json").mkdir(parents=True)
        return outside / "adir" / "shape.json", 2, "a directory stands at this name"
    if name == "the null device itself":
        # Not a symlink to it: a link is refused one step earlier, so this is
        # the only spelling that reaches the destination open at all.
        return Path(os.devnull), 2, "this name is not a file"
    if name == "a destination naming no file":
        # A `str` and not a `Path`, which drops the trailing separator.
        return str(outside) + os.sep, 2, "names a directory and not a file"
    if name == "a fresh path":
        return outside / "fresh" / "deep" / "shape.json", 0, None
    raise AssertionError(name)


def _files_under(directory: Path) -> dict:
    """Everything below `directory`: files by inode and bytes, and the names
    of the directories and links that hold them.

    Inode as well as bytes because the attacks differ there — a symlink puts
    new bytes in the victim's inode, a hard link truncates the inode the
    victim is still one name for — and directories because a refused
    destination that created them on its way out is the same guard failing
    one step earlier.
    """
    found = {}
    for path in sorted(directory.rglob("*")):
        key = str(path.relative_to(directory))
        if path.is_symlink():
            found[key] = "link"
        elif path.is_dir():
            found[key] = "dir"
        else:
            found[key] = (path.stat().st_ino, path.read_bytes())
    return found


@pytest.mark.parametrize(
    "spelling",
    [
        "a symlink at the name",
        "a symlink one directory up",
        "a dotdot after a symlinked directory",
        "a hard link at the name",
        "a dangling symlink at the name",
        "a directory made on the way to a refusal",
        "a plain file in a plain directory",
        "a fifo at the name",
        "a directory at the name",
        "the null device itself",
        "a destination naming no file",
        "a fresh path",
    ],
)
def test_out_writes_only_where_its_refusals_were_answered_about(
    tmp_path, spelling
) -> None:
    """`--out` CREATES its destination. Every name that is already taken is a
    refusal, whatever it is taken by.

    A destination is a write primitive: over `sudo -n` on a shared capture
    host — the documented way this runs — anybody who can create a name in the
    directory the operator writes shapes into chooses what gets overwritten.
    Four rounds closed one spelling each and left the next open — a symlink at
    the name, a symlink above it, a `..` the kernel takes from a link's target,
    a hard link no `O_NOFOLLOW` can see, and a parent swapped after the check —
    which is what a table of spellings buys and what it does not: the rule that
    ends the sequence is that nothing that exists is opened at all.

    `--raw` is the same question with the stakes the privacy rule turns on,
    so every spelling is run that way too: whatever the tool decides, the
    checkout is byte-for-byte what it was.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    plain_root = tmp_path / "plain"
    out, code, says = _destination(spelling, plain_root)
    repo = plain_root / "repo"
    before = _files_under(repo)

    run = _run("--config-dir", str(config), "--out", str(out))
    assert run.returncode == code, run.stdout + run.stderr
    if says is not None:
        assert says in run.stderr, run.stderr
        if "a symlink stands in" in says:
            # This one fires for every macOS user on every `--out $TMPDIR/...`,
            # because `/var` is itself a link — so the message says which path
            # to pass rather than reading as an attack somebody staged.
            assert "on macOS /var is itself a link" in run.stderr
        else:
            # A refusal names the reason it found. Every spelling below this
            # line fails for something other than a link standing in the path,
            # and being told to hunt for one that is not there is its own
            # defect: a trailing separator reached that message for four
            # rounds because two spellings of the parent disagree about it.
            assert "a symlink stands in" not in run.stderr, run.stderr
    else:
        assert json.loads(out.read_text(encoding="utf-8"))["anonymised"] is True
    assert _files_under(repo) == before, "it wrote through the destination"

    raw_root = tmp_path / "raw"
    raw_out, raw_code, _ = _destination(spelling, raw_root)
    raw_repo = raw_root / "repo"
    raw_before = _files_under(raw_repo)
    raw = _run("--config-dir", str(config), "--raw", "--out", str(raw_out))
    assert raw.returncode == raw_code, raw.stdout + raw.stderr
    assert _files_under(raw_repo) == raw_before, "real names landed in a checkout"


def test_an_ordinary_destination_is_created_and_never_written_over(tmp_path) -> None:
    """The refusals above are worth nothing if the tool cannot write, and the
    second run of the same command is where a rewrite would happen.

    It does not happen. A capture that would only ever be overwriting its own
    output cannot tell that from overwriting somebody else's, and it decided
    on a host it is a guest on: the second run refuses and names the file, and
    an operator who wants a fresh one removes it. That also makes the mode
    unconditional — `0o600` applies because the inode is this run's, where a
    create-time mode over an existing file applies to nothing at all.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    plain = tmp_path / "plain.json"
    assert _run("--config-dir", str(config), "--out", str(plain)).returncode == 0
    assert json.loads(plain.read_text(encoding="utf-8"))["anonymised"] is True
    # OWNER ONLY. `--raw` carries real usernames, org names and repository
    # paths, and it is written on a host this tool is a guest on — a default
    # umask left it readable by everyone else logged in there. Asserted on the
    # anonymised run, because the two are written by one call and the file
    # that needs the narrower mode is the one somebody forgot they produced.
    assert stat.S_IMODE(plain.stat().st_mode) == 0o600

    was = (plain.stat().st_ino, plain.read_bytes())
    again = _run("--config-dir", str(config), "--out", str(plain))
    assert again.returncode == 2, again.stdout + again.stderr
    assert "already at this name" in again.stderr, again.stderr
    assert (plain.stat().st_ino, plain.read_bytes()) == was, "it wrote over it"
    # A world-readable file left by somebody else is not narrowed by a second
    # run either, because there is no second run: it is refused.
    theirs = tmp_path / "theirs.json"
    theirs.write_text("not mine\n", encoding="utf-8")
    os.chmod(str(theirs), 0o666)
    refused = _run("--config-dir", str(config), "--raw", "--out", str(theirs))
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert theirs.read_text(encoding="utf-8") == "not mine\n"


def test_out_refuses_a_destination_it_cannot_make_rather_than_crashing(
    tmp_path,
) -> None:
    """`main()`'s contract is a message and an exit 2, and the `--out` route
    had two calls outside it.

    A linked worktree's `.git` is a FILE, so `--out <worktree>/.git/shape.json`
    named a directory that cannot be made — and a plain file anywhere in the
    path does the same. Both left a Python traceback and an exit 1 on a host
    reached once over ssh, where a wrapper reads that number and 1 and 2 mean
    different things to it. Nothing was written; nothing is written now.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    midway = tmp_path / "notadir"
    midway.write_text("i am a file\n", encoding="utf-8")
    for out in (worktree / ".git" / "shape.json", midway / "deep" / "shape.json"):
        refused = _run("--config-dir", str(config), "--out", str(out))
        assert refused.returncode == 2, refused.stdout + refused.stderr
        assert "Traceback" not in refused.stderr, refused.stderr
        assert refused.stderr.startswith("harness_shape: "), refused.stderr
        assert len(refused.stderr.strip().splitlines()) == 1, refused.stderr
        assert not out.exists(), "it refused and wrote anyway"
    assert midway.read_text(encoding="utf-8") == "i am a file\n"
    assert (worktree / ".git").read_text(encoding="utf-8").startswith("gitdir: ")


def test_out_writes_into_the_directory_it_judged_however_the_name_moves(
    tmp_path,
) -> None:
    """The fifth spelling of one defect, and the one no name can close.

    The first four were a link at the destination name, a link above it, a
    `..` the kernel takes from a link's target, and a second hard name. This
    is the parent directory REPLACED between the check and the open: every
    refusal answered about the directory the operator named, and the bytes
    went to the one that had taken its place — with `--raw`, into a git
    checkout, exit 0. The destination's parent is opened once and the file is
    created relative to that descriptor, so a rename cannot come between them.

    The count is what makes this a gate rather than a coin toss: a build that
    opens the name has to lose only one of these runs, and on a loaded machine
    the flipping thread can be starved for a long stretch of them.
    """
    config = tmp_path / "config"
    memory = config / "projects" / "-Users-realname-src-realrepo" / "memory"
    memory.mkdir(parents=True)
    _write(memory / "real-secret-name.md", "---\nname: a-real-memory-name\n---\nb\n")
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    dest = tmp_path / "dest"
    dest.mkdir()
    stop = threading.Event()

    def flip() -> None:
        while not stop.is_set():
            try:
                if dest.is_symlink():
                    os.remove(str(dest))
                    os.mkdir(str(dest))
                else:
                    shutil.rmtree(str(dest), ignore_errors=True)
                    os.symlink(str(checkout), str(dest))
            except OSError:
                pass

    flipper = threading.Thread(target=flip)
    flipper.start()
    try:
        for _ in range(200):
            run = _run(
                "--config-dir", str(config), "--raw", "--out", str(dest / "leak.json")
            )
            # The exit code and the stderr are half of what this route
            # promises and were thrown away here: a contended run that crashes
            # is a wrapper on the far end of an ssh pipe told the wrong thing,
            # and 200 of these were reporting only where the bytes went.
            assert run.returncode in (0, 2), run.stdout + run.stderr
            assert "Traceback" not in run.stderr, run.stderr
            assert not (checkout / "leak.json").exists(), (
                "real names landed in a checkout the run never named"
            )
    finally:
        stop.set()
        flipper.join()


def test_out_answers_with_an_exit_2_when_the_directory_it_runs_in_is_gone(
    tmp_path,
) -> None:
    """`--out` reads its own path before anything guards the reading.

    `abspath` on a relative `--out` calls `os.getcwd()`, and a process whose
    working directory has been removed gets `FileNotFoundError` from it — no
    attacker, no race, and the one route that promises a wrapper on the far
    end of an ssh pipe a number rather than a traceback gave it an exit 1.
    `realpath` on the next line is documented not to raise and does, for the
    same reason one level along.

    The removal happens in a CHILD shell, because a test that removes its own
    working directory takes the rest of the suite with it.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    gone = tmp_path / "gone"
    gone.mkdir()
    script = "; ".join(
        [
            f"cd {shlex.quote(str(gone))}",
            f"rmdir {shlex.quote(str(gone))}",
            " ".join(
                shlex.quote(part)
                for part in [
                    "exec", sys.executable, str(TOOL),
                    "--config-dir", str(config), "--out", "out/shape.json",
                ]
            ),
        ]
    )
    refused = subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=300,
    )
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert "Traceback" not in refused.stderr, refused.stderr
    assert refused.stderr.startswith("harness_shape: "), refused.stderr
    assert len(refused.stderr.strip().splitlines()) == 1, refused.stderr
    assert refused.stdout == "", "it failed and emitted a shape anyway"


def _raw_capture_tree(tmp_path):
    """A config directory whose one memory carries a name worth not leaking."""
    config = tmp_path / "config"
    memory = config / "projects" / "-Users-realname-src-realrepo" / "memory"
    memory.mkdir(parents=True)
    _write(memory / "real-secret-name.md", "---\nname: a-real-memory-name\n---\nb\n")
    return config


def test_the_judged_directory_is_re_judged_when_the_file_is_created(
    tmp_path, monkeypatch, capsys,
) -> None:
    """The sixth spelling: the judged directory RENAMED INTO a checkout.

    The five above it move the name and the descriptor answers about the inode,
    which is right and is not enough — the question was asked before the
    capture and the file is made after it, and a capture of a real machine is
    seconds of window. Moving the inode satisfies every rule the descriptor
    enforces, because it is still the directory that was judged and still the
    one written into; it is simply somewhere else now.

    So the question is asked again of the descriptor the file was created
    relative to, and an answer that changed unlinks a file nothing has been
    written to yet. The flipper above is a race and this is the seam: the
    rename is driven once, inside the window, with no thread to lose.
    """
    module = _tool_module()
    config = _raw_capture_tree(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    dest = tmp_path / "dest"
    dest.mkdir()
    real_capture = module.capture

    def rename_into_the_checkout(*args, **kwargs):
        shape = real_capture(*args, **kwargs)
        os.rename(str(dest), str(checkout / "moved_dest"))
        return shape

    monkeypatch.setattr(module, "capture", rename_into_the_checkout)
    argv = ["--config-dir", str(config), "--raw", "--out", str(dest / "leak.json")]
    assert module.main(argv) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err, err
    assert not (checkout / "moved_dest" / "leak.json").exists(), (
        "real names landed in a checkout the destination was moved into"
    )


def test_a_destination_level_that_appears_during_the_capture_is_not_adopted(
    tmp_path, monkeypatch, capsys,
) -> None:
    """`--out a/b/c.json` where `b` does not exist yet, and something else
    makes `b` a checkout while the capture runs.

    Nothing judged `b`: the descriptor is on `a`, and the guarantee that
    carries the judgement down is that everything below it is made by this
    run and made empty. A level that turns out to be there already breaks
    exactly that, so it is a refusal rather than a directory to adopt.

    The arm WITHOUT `--raw` is the one that holds the rule on its own. With
    `--raw` the re-ask at the leaf catches this particular level because it is
    a checkout; an anonymised run asks nothing there, so an adopted level goes
    unnoticed unless the adoption itself is refused.
    """
    module = _tool_module()
    config = _raw_capture_tree(tmp_path)
    land = tmp_path / "land"
    land.mkdir()
    newdir = land / "newdir"
    real_capture = module.capture

    def make_a_checkout(*args, **kwargs):
        shape = real_capture(*args, **kwargs)
        (newdir / ".git").mkdir(parents=True)
        return shape

    monkeypatch.setattr(module, "capture", make_a_checkout)
    argv = ["--config-dir", str(config), "--raw", "--out", str(newdir / "shape.json")]
    assert module.main(argv) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err, err
    assert not (newdir / "shape.json").exists()

    plain = land / "plain"

    def make_a_directory(*args, **kwargs):
        shape = real_capture(*args, **kwargs)
        plain.mkdir()
        return shape

    monkeypatch.setattr(module, "capture", make_a_directory)
    assert module.main(
        ["--config-dir", str(config), "--out", str(plain / "shape.json")]
    ) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err, err
    assert not (plain / "shape.json").exists()

    # The control the deferred creation is for: a level nobody else touches is
    # still made, and the shape still lands in it.
    monkeypatch.undo()
    mine = tmp_path / "land" / "mine" / "shape.json"
    assert module.main(["--config-dir", str(config), "--out", str(mine)]) == 0
    assert json.loads(mine.read_text(encoding="utf-8"))["anonymised"] is True


def test_a_landing_directory_that_will_not_open_is_a_refusal_and_not_a_no(
    tmp_path,
) -> None:
    """A directory a shell can write in and `O_RDONLY` cannot open.

    Mode 0333 grants write and search and no read, so a redirect lands there
    and the walk that decides whether "there" is a checkout cannot take its
    first step. Answering "not a checkout" to a question that was never
    answered is how an un-anonymised shape reached a git checkout at exit 0
    with an empty stderr, and the same directory at 0755 is refused.
    """
    if os.geteuid() == 0:
        pytest.skip("root bypasses the mode bits this case is made of")
    config = _raw_capture_tree(tmp_path)
    tree = tmp_path / "wt"
    (tree / ".git").mkdir(parents=True)
    drop = tree / "drop"
    drop.mkdir()
    landing = drop / "raw.json"
    argv = [sys.executable, str(TOOL), "--config-dir", str(config), "--raw"]
    os.chmod(str(drop), 0o333)
    try:
        with landing.open("w") as handle:
            refused = subprocess.run(
                argv, stdout=handle, stderr=subprocess.PIPE, text=True, timeout=300,
            )
    finally:
        os.chmod(str(drop), 0o755)
    assert refused.returncode == 2, refused.stderr
    assert "Traceback" not in refused.stderr, refused.stderr
    assert landing.read_text(encoding="utf-8") == "", "it refused and wrote"


@pytest.mark.parametrize("route", ["--out", "redirect"])
def test_an_ancestor_that_will_not_open_is_a_refusal_on_both_raw_routes(
    tmp_path, route,
) -> None:
    """The walk to the top of the tree, not just its first step.

    The landing directory opens; a directory ABOVE it at mode 0111 takes the
    `..` step and refuses `O_RDONLY`, and that step had no handler at all — a
    `PermissionError` out of the middle of the walk on both `--raw` routes,
    where the contract is one line and an exit 2 and where the operator is a
    wrapper on the far end of an ssh pipe reading the number.

    An unanswerable question is a refusal, and a refusal names the destination
    it is about: leaving instead by the exit-2 boundary gives the same number
    with `..` as the whole subject, which tells an operator nothing about
    where they asked the bytes to go.
    """
    if os.geteuid() == 0:
        pytest.skip("root reads a directory at mode 0111")
    config = _raw_capture_tree(tmp_path)
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    landing = inner / "raw.json"
    argv = [sys.executable, str(TOOL), "--config-dir", str(config), "--raw"]
    if route == "--out":
        argv += ["--out", str(landing)]
        os.chmod(str(outer), 0o111)
        try:
            run = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        finally:
            os.chmod(str(outer), 0o755)
        assert not landing.exists(), "it could not answer and wrote anyway"
    else:
        with landing.open("w") as handle:
            os.chmod(str(outer), 0o111)
            try:
                run = subprocess.run(
                    argv, stdout=handle, stderr=subprocess.PIPE, text=True,
                    timeout=300,
                )
            finally:
                os.chmod(str(outer), 0o755)
        # The redirect makes the file before this tool starts, so what says it
        # refused is that nothing was written into it.
        assert landing.read_text(encoding="utf-8") == "", "it refused and wrote"
    assert run.returncode == 2, run.stderr
    assert "Traceback" not in run.stderr, run.stderr
    lines = run.stderr.strip().splitlines()
    assert len(lines) == 1, run.stderr
    assert lines[0].startswith("harness_shape:"), run.stderr
    assert os.path.realpath(str(landing)) in lines[0], lines[0]
    assert "worktree cannot be answered" in lines[0], lines[0]
    assert "Permission denied" in lines[0], lines[0]



def test_a_projects_directory_that_is_a_dead_link_is_not_an_empty_machine(
    tmp_path,
) -> None:
    """`os.scandir` raises FileNotFoundError for a name that is not there and
    for one that is there as a broken link, and only the first is a shape.

    A `projects/` linked onto a volume that has been unmounted or a directory
    that has been moved is a machine WITH projects that this run cannot see,
    and it captured as `projects_total: 0` at exit 0 — the healthy empty
    machine the handler's own comment says every other failure must not
    become.

    AND THE PACKAGE ANSWERS THE OTHER WAY ON PURPOSE. `inventory` returns an
    empty list for the same tree, which is right for it and wrong here: a
    capture is one shot over ssh at a machine nobody will visit again, so an
    empty shape that reads as healthy is the failure that survives, while
    `inventory` is a local diagnostic anybody can re-run and must not die on
    one unreadable name having said nothing about the other 3917. The
    equivalence test cannot see this pair, because the tool never returns.
    """
    config = tmp_path / "config"
    config.mkdir()
    os.symlink(tmp_path / "moved-away", config / "projects")
    refused = _run("--config-dir", str(config))
    assert refused.returncode == 2, refused.stdout
    assert refused.stdout == "", "it failed and emitted a shape anyway"
    assert "harness_shape:" in refused.stderr
    assert _inventory(str(config)) == []

    # The control the handler is there for: nothing written yet really is a
    # machine, and it still captures.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _shape("--config-dir", str(empty))["projects_total"] == 0


def test_a_projects_directory_that_goes_away_under_the_walk_is_not_an_empty_machine(
    tmp_path, monkeypatch,
) -> None:
    """The other way into the same arm, and the name cannot answer it.

    `projects/` dropping mid-iteration — an unmounted NFS home, a directory
    moved out from under the walk — raises the same `FileNotFoundError` as a
    directory that was never there, and the name the arm asks about is gone by
    the time it asks. So the re-raise was skipped, every project already
    collected was thrown away, and a machine mid-unmount captured as a healthy
    empty one at exit 0.

    What the arm asks first now is whether the walk had already served an
    entry, which is a fact about this run rather than about a name somebody
    else owns.
    """
    module = _tool_module()
    config = tmp_path / "config"
    for number in range(4):
        _write(
            _memory_dir(config, f"-p{number}") / "MEMORY.md",
            f"---\nname: n{number}\n---\nbody\n",
        )
    projects = str(config / "projects")
    assert module.capture(str(config))["projects_total"] == 4

    real_scandir = os.scandir

    class _DropsAfterOneEntry:
        """One entry served, and then the directory goes the way a mount does."""

        def __init__(self, path) -> None:
            self._inner = real_scandir(path)
            self._path = path
            self._served = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            self._inner.close()
            return False

        def __iter__(self):
            return self

        def __next__(self):
            if self._served:
                shutil.rmtree(self._path, ignore_errors=True)
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self._path
                )
            self._served += 1
            return next(self._inner)

    def scandir(path=".", *args, **kwargs):
        # `module.os` IS the os module, so this patch is GLOBAL: everything
        # that is not the one path being targeted is delegated, file
        # descriptors and other directories included.
        if isinstance(path, str) and path == projects and not args and not kwargs:
            return _DropsAfterOneEntry(path)
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "scandir", scandir)
    with pytest.raises(FileNotFoundError):
        module.capture(str(config))


def test_a_frontmatter_key_is_matched_the_way_the_checker_matches_it(
    tmp_path,
) -> None:
    """`description : text` is a description to YAML and to
    `memory_integrity._frontmatter`, and a literal-prefix test read it as no
    frontmatter description at all.

    The flags are only worth carrying if they are the checker's, so the two
    parsers are compared over the spellings that tell them apart rather than
    against literals typed here.
    """
    lines = {
        "description : spaced out": ("description", "spaced out"),
        "name\t: tabbed": ("name", "tabbed"),
        "description: plain": ("description", "plain"),
        "# description: commented": (None, None),
        "not a key at all": (None, None),
        "two words: value": (None, None),
    }
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    for number, line in enumerate(lines):
        _write(memory / f"m{number}.md", f"---\n{line}\n---\n\nbody\n")
    files = {
        item["name"]: item
        for item in _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]["files"]
    }
    for number, (line, (key, value)) in enumerate(lines.items()):
        seen = memory_integrity._frontmatter(memory / f"m{number}.md")
        assert seen == ({} if key is None else {key: value}), line
        item = files[f"m{number}.md"]
        assert item["has_description"] is (key == "description"), line
        assert item["has_name"] is (key == "name"), line
        if key == "description":
            assert item["description_len"] == len(value), line


def test_the_fence_is_read_the_way_the_checker_reads_it(tmp_path) -> None:
    """Which spellings of the fence open frontmatter, decided by running both
    parsers over the same documents.

    The tool is stdlib-only on a 3.8 floor and cannot import memkit, so the
    two rules are two pieces of code that have to be kept in step by
    something; a table both are run over is that something. `----` opened
    frontmatter to the checker and not to the tool.

    A FENCE THAT NEVER CLOSES IN A READ THE CAP CUT SHORT is the one spelling
    they disagree on, and the disagreement is deliberate: the checker reads an
    unclosed fence as frontmatter running to the end of the file, and a read
    that stopped at `FRONTMATTER_BYTES` cannot tell that from a fence closing
    one byte past the cut, so doing the same would report the keys that happen
    to sit above the cap as a whole frontmatter block. An unclosed fence in a
    read that reached the end of the file is not that case, and is read the
    checker's way.

    A FENCE PAIR ENCLOSING NO TOP-LEVEL KEY is not frontmatter to either. The
    checker has no keys to return there, and a shape that recorded the fence
    as present with all four facts false would disagree with the rule a
    consumer runs over the rebuilt file.
    """
    table = {
        "a bare fence": ("---\nname: x\n---\nbody\n", True),
        "trailing spaces on the fence": ("---   \nname: x\n---\nbody\n", True),
        "CRLF": ("---\r\nname: x\r\n---\r\nbody\r\n", True),
        "a BOM before the fence": ("﻿---\nname: x\n---\nbody\n", False),
        "four dashes": ("----\nname: x\n---\nbody\n", True),
        "a fence below line 1": ("\n---\nname: x\n---\nbody\n", False),
        "an empty document": ("", False),
        "a close on the very next line": ("---\n---\nbody\n", False),
        "a fence enclosing only whitespace": ("---\n   \n---\n", False),
        "an unclosed fence inside the cap": ("---\nname: x\nbody\n", True),
        "the key on the fence line": ("---name: x\n---\nbody\n", True),
        "the key on the fence line, unclosed": ("---name: x\nbody\n", True),
    }
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    for number, (text, _) in enumerate(table.values()):
        with open(memory / f"m{number}.md", "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    files = {
        item["name"]: item
        for item in _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]["files"]
    }
    for number, (label, (_, opens)) in enumerate(table.items()):
        checker = bool(memory_integrity._frontmatter(memory / f"m{number}.md"))
        assert checker is opens, label
        assert files[f"m{number}.md"]["has_frontmatter"] is opens, label
    # And the one they part company on, named rather than left to be found: a
    # fence that never closes in a read the cap cut short. The keys above the
    # cut are not a frontmatter block anybody has, so the tool declines to
    # report one and the checker, which read the whole file, reports the keys.
    cap = _tool_module().FRONTMATTER_BYTES
    with open(memory / "open.md", "w", encoding="utf-8", newline="") as fh:
        fh.write("---\nname: x\n" + "filler line\n" * (cap // 12 + 1))
    assert (memory / "open.md").stat().st_size > cap
    unclosed = _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]["files"]
    assert memory_integrity._frontmatter(memory / "open.md") == {"name": "x"}
    cut = next(item for item in unclosed if item["name"] == "open.md")
    assert cut["has_frontmatter"] is False
    assert cut["frontmatter_truncated"] is True


def test_memkit_is_kept_by_name_whichever_way_the_key_is_spelled(tmp_path) -> None:
    """The exception exists so a shape says whether memkit was installed on the
    machine that was captured, and it only fired on the `<plugin>@<market>`
    spelling — a bare key anonymised to `p<n>` and took the fact with it.

    Neither branch had a test. Both do now, and the marketplace stays a
    pseudonym in both: where somebody hosts their own marketplace is not a
    fact a shape is allowed to keep.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    _write(
        config / "settings.json",
        json.dumps(
            {
                "enabledPlugins": {
                    "memkit": True,
                    "memkit@memkit": True,
                    f"memkit@{SENTINEL}": True,
                    SENTINEL: True,
                }
            }
        ),
    )
    out = _run("--config-dir", str(config), env=_managed_env(tmp_path / "no-managed"))
    assert out.returncode == 0, out.stderr
    assert SENTINEL not in out.stdout
    # Sorted as OUTPUT, so the list order says nothing about where a redacted
    # key fell among the real ones.
    assert json.loads(out.stdout)["settings"]["user"]["plugins"] == [
        "memkit", "memkit@q1", "memkit@q2", "p1",
    ]


def _inventory(config_dir: str) -> list:
    """`harness_memory.inventory`'s projects, whichever shape it returns."""
    found = harness_memory.inventory(config_dir)
    return found[0] if isinstance(found, tuple) else found


def _adoptable(shape: dict) -> dict:
    """The tool's memory directories that hold a file other than the index.

    What `inventory` calls a corpus is this narrower set, and the difference
    between the two questions is where a comparison of the two walks would
    otherwise read as a disagreement.
    """
    return {
        entry["key"]: entry
        for entry in shape["memory_dirs"]
        if any(item["name"] != harness_memory.INDEX_NAME for item in entry["files"])
    }


def test_the_two_walks_answer_the_same_way_about_a_hostile_tree(tmp_path) -> None:
    """The duplication strategy rests on this and was watched on a tree where
    nothing could go wrong.

    The module docstring pins the copies in step by running both walks over
    one tree and requiring agreement — and the tree had no looping link, no
    unreadable directory and no dangling name, so the guard the tool grew for
    exactly those cases put the two walks a whole directory apart without
    anything going red. Every state either walk answers differently about is
    here, and both are asked the same two questions: which directories hold a
    corpus, and what is in them.
    """
    config = tmp_path / "config"

    loop = _memory_dir(config, "-loop")
    _write(loop / "real1.md", "x\n")
    _write(loop / "real2.md", "x\n")
    # A name anybody who can write in the tree can leave there, and the one
    # `DirEntry.is_file` raises ELOOP for rather than answering.
    os.symlink("loop.md", loop / "loop.md")

    dangling = _memory_dir(config, "-dangling")
    _write(dangling / "real.md", "x\n")
    os.symlink(tmp_path / "gone.md", dangling / "dead.md")

    outside = tmp_path / "elsewhere"
    _write(outside / "real-index.md", "# i\n\n- [a](m.md) — h\n")
    _write(outside / "theirs.md", "x\n")
    indexed = _memory_dir(config, "-idxlink")
    _write(indexed / "m.md", "x\n")
    os.symlink(outside / "real-index.md", indexed / "MEMORY.md")

    # A memory directory linked OUT of the config root, and one linked to
    # another directory INSIDE it: both are followed, and a rebuilt tree owes
    # each a different answer.
    linked_out = config / "projects" / "-outlink"
    linked_out.mkdir(parents=True)
    os.symlink(outside, linked_out / "memory")
    shared = config / "shared"
    _write(shared / "shared.md", "x\n")
    linked_in = config / "projects" / "-inlink"
    linked_in.mkdir(parents=True)
    os.symlink(shared, linked_in / "memory")

    blocked = None
    if not ROOT:
        blocked = _memory_dir(config, "-blocked")
        _write(blocked / "real.md", "x\n")
        blocked.chmod(0o000)
    try:
        shape = _shape("--config-dir", str(config), "--raw")
        found = _inventory(str(config))
    finally:
        if blocked is not None:
            blocked.chmod(0o700)

    walked = _adoptable(shape)
    assert set(walked) == {project.key for project in found}
    # Named rather than derived, so a walk that dropped every directory would
    # not pass this by agreeing about nothing.
    expected = {"-loop", "-dangling", "-idxlink", "-outlink", "-inlink"}
    assert set(walked) == expected
    for project in found:
        entry = walked[project.key]
        assert [item["name"] for item in entry["files"]] == project.files, project.key
        assert entry["is_symlink"] is project.is_symlink, project.key
        assert entry["project_is_symlink"] is project.linked_project, project.key
        assert tuple(
            item["name"] for item in entry["files"] if item["is_symlink"]
        ) == project.linked_files, project.key
    # A directory neither walk could list is absent from both, and the tool
    # says so where the package has nowhere to.
    assert shape["skipped"] == (0 if ROOT else 1)
    # The loop is a name and never a file, so neither walk lists it, and the
    # tool books the entry it could not answer for.
    assert walked["-loop"]["files"][0]["name"] == "real1.md"
    assert shape["read_errors"] == 1


def test_the_tool_and_the_package_agree_on_what_a_corpus_is(tmp_path) -> None:
    """The shape is measured by one walk and consumed by another, and the two
    have to name the same directories.

    `harness_memory.inventory` is the package's answer and this tool is a
    stdlib copy of the same rules, deliberately duplicated because it runs
    where memkit is not installed. So the copies are compared rather than
    trusted: the directories holding a file other than the index are exactly
    the inventory's keys, and the symlink flag agrees on each. The tree has an
    index-only directory and an empty one precisely because those are where a
    looser rule would show up as a disagreement.
    """
    config = _tree(tmp_path)
    shape = _shape("--config-dir", str(config), "--raw")
    adoptable = _adoptable(shape)
    found = _inventory(str(config))
    assert set(adoptable) == {project.key for project in found}
    for project in found:
        entry = adoptable[project.key]
        # ALL THREE link flags and the file list, not just the one. The
        # package documents why each needs its own answer, and a comparison of
        # one of them passes while the two copies disagree about what a link
        # is — which is a rebuilt tree that never reaches doctor's linked path.
        assert entry["is_symlink"] is project.is_symlink, project.key
        assert entry["project_is_symlink"] is project.linked_project, project.key
        assert [item["name"] for item in entry["files"]] == project.files, project.key
        assert tuple(
            item["name"] for item in entry["files"] if item["is_symlink"]
        ) == project.linked_files, project.key
    # The index-only directory is the difference between the two questions, and
    # it is a directory the tool reports and the inventory does not.
    assert "-h-u-git-empty" not in adoptable
    assert any(
        entry["key"] == "-h-u-git-empty" for entry in shape["memory_dirs"]
    )


def test_the_managed_scope_anonymises_the_way_the_user_scope_does(tmp_path) -> None:
    """The one scope read from a MACHINE path rather than from `--config-dir`,
    and until now the one no test could reach.

    Its path is fixed, so there was nowhere to put a file: every settings
    assertion in this file is about `user`, and `managed` was covered by
    nothing at all while carrying the same three rows. It is also the scope
    where the richest names live, because a managed settings file is an
    organisation's rather than a person's.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    managed = tmp_path / "managed"
    _write(
        managed / "managed-settings.json",
        json.dumps(
            {
                harness_memory.ENABLED_KEY: True,
                harness_memory.DIRECTORY_KEY: f"/Users/{SENTINEL}/notes",
                "hooks": {f"Gate--{SENTINEL}": [], "PreToolUse": []},
                "enabledPlugins": {f"{SENTINEL}@{MARKET}": True},
            }
        ),
    )
    out = _run("--config-dir", str(config), "--managed", env=_managed_env(managed))
    assert out.returncode == 0, out.stderr
    assert SENTINEL not in out.stdout
    scope = json.loads(out.stdout)["settings"]["managed"]
    assert scope == {
        "unreadable": False,
        "memory_keys": {
            harness_memory.ENABLED_KEY: True,
            harness_memory.DIRECTORY_KEY: "<path>",
        },
        "hooks": ["PreToolUse", "h1"],
        "plugins": ["p1@q1"],
    }


def test_the_machines_policy_file_travels_only_with_its_own_machines_tree(
    tmp_path,
) -> None:
    """Every other row in a shape comes out of `--config-dir`; this one comes
    off the machine, and a capture of a tree that is not this machine's used to
    fold it in anyway.

    That is an org's hook keys and an org's plugin names filed under a key that
    reads as the captured tree's — a copied or temporary directory, which is
    what a fixture is built from. So the scope is read for the config directory
    this process's own harness would use, and otherwise only when the operator
    passes `--managed` to say the tree they named is that machine's: the ssh
    flow names another user's home under `sudo -n`, where the process's own
    default is root's and no automatic test can recognise it.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    managed = tmp_path / "managed"
    _write(
        managed / "managed-settings.json",
        json.dumps({"hooks": {f"Gate--{SENTINEL}": []}}),
    )
    env = _managed_env(managed)
    assert _shape("--config-dir", str(config), env=env)["settings"] == {}
    asserted = _shape("--config-dir", str(config), "--managed", env=env)
    assert asserted["settings"]["managed"]["hooks"] == ["h1"]

    # The same tree, now the one this process's harness would use. Named
    # through a LINK, because a home reached through one — and a `~/.claude`
    # linked into a dotfiles checkout — are both ordinary, and a comparison of
    # the two strings answers no for both.
    link = tmp_path / "link-to-config"
    os.symlink(config, link)
    own = dict(env, CLAUDE_CONFIG_DIR=str(config))
    assert _shape("--config-dir", str(link), env=own)["settings"]["managed"][
        "hooks"
    ] == ["h1"]


def test_a_missing_policy_file_and_a_capture_that_did_not_look_are_told_apart(
    tmp_path,
) -> None:
    """An omitted `managed` scope carried three states at once and recorded
    none of them.

    No such file on the machine, `--managed` not passed, and a tree this
    machine's harness does not run on all produced the same bytes — so a
    consumer reading a shape, which is all a materialiser has, could only take
    the absence for the machine's. The prose in `_settings` disclaimed the
    inference; the artifact still invited it.
    """
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    empty = _managed_env(tmp_path / "no-policy-file")
    machine = tmp_path / "managed"
    _write(machine / "managed-settings.json", json.dumps({"hooks": {}}))
    present = _managed_env(machine)

    declined = _shape("--config-dir", str(config), env=present)
    assert declined["settings_managed_read"] is False
    assert "managed" not in declined["settings"]

    none_there = _shape("--config-dir", str(config), "--managed", env=empty)
    assert none_there["settings_managed_read"] is True
    assert "managed" not in none_there["settings"]

    # The two omissions are now different documents, which is the whole point:
    # before this row they were byte-identical.
    assert declined != none_there

    read = _shape("--config-dir", str(config), "--managed", env=present)
    assert read["settings_managed_read"] is True
    assert read["settings"]["managed"]["unreadable"] is False

    # And the third meaning: the scope is read for this process's own tree
    # without the flag, so the row follows the DECISION rather than the flag.
    own = dict(present, CLAUDE_CONFIG_DIR=str(config))
    assert _shape("--config-dir", str(config), env=own)[
        "settings_managed_read"
    ] is True


ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.mark.skipif(ROOT, reason="root reads a directory nobody else can")
def test_a_half_failed_capture_counts_what_it_could_not_read(tmp_path) -> None:
    """A capture that read half a machine has to be distinguishable from one
    that read all of a smaller machine.

    Three failures, three places. An unreadable PROJECT directory answered
    False to `os.path.isdir` and so missed both counters — it read as a
    project with no memories. An unreadable memory directory was already
    counted. And a file that could not be read recorded `size: 0` and four
    `false` flags, which is also what an empty file records.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    _write(memory / "one.md", "x\n")
    unreadable_file = _write(memory / "two.md", "---\nname: two\n---\n")
    unreadable_file.chmod(0o000)
    blocked_memory = _memory_dir(config, "-b")
    _write(blocked_memory / "one.md", "x\n")
    blocked_memory.chmod(0o000)
    blocked_project = _memory_dir(config, "-c").parent
    _write(blocked_project / "memory" / "one.md", "x\n")
    blocked_project.chmod(0o000)
    try:
        shape = _shape("--config-dir", str(config), "--raw")
    finally:
        for path, mode in (
            (unreadable_file, 0o600), (blocked_memory, 0o700), (blocked_project, 0o700)
        ):
            path.chmod(mode)
    # `-c` is not counted as a memory directory because nothing could tell
    # whether it had one; it is counted as a directory that was skipped.
    assert shape["memory_dirs_total"] == 2
    assert shape["skipped"] == 2
    assert shape["read_errors"] == 1
    files = {item["name"]: item for item in _by_key(shape)["-a"]["files"]}
    # The size is readable here and the CONTENT is not, so the file is listed
    # with its real size and says in its own record that nothing was read off
    # it. (A file whose `lstat` fails records `size: null`, never 0.)
    assert files["two.md"]["size"] == unreadable_file.stat().st_size
    assert files["two.md"]["unreadable"] is True
    assert files["two.md"]["has_frontmatter"] is None
    assert files["one.md"]["unreadable"] is False
    assert files["one.md"]["has_frontmatter"] is False


def test_one_unclassifiable_project_entry_does_not_end_the_capture(
    tmp_path, monkeypatch,
) -> None:
    """The listing of `projects/` used to be one comprehension, so the
    `is_symlink()` of any single entry decided whether the machine got a shape
    at all.

    It cannot be provoked by planting: APFS and ext4 carry the type in the
    readdir record, so `DirEntry.is_symlink` answers without a syscall and
    never raises. The filesystems that supply no `d_type` — NFS among them,
    which is where `sudo -n` into somebody's home lands — stat the name
    instead, and there an entry the capture cannot reach raises. So the
    failing entry is injected: what is under test is the blast radius, and the
    kernel that produces it is not one this suite can stand on.
    """
    config = tmp_path / "config"
    for key in ("-good", "-bad"):
        _write(_memory_dir(config, key) / "one.md", "x\n")
    module = _tool_module()
    real_scandir = module.os.scandir

    class Unreachable:
        """One `projects/` entry whose type cannot be established."""

        def __init__(self, entry) -> None:
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def is_symlink(self):
            if self._entry.name == "-bad":
                raise OSError(errno.EACCES, "Permission denied", self._entry.path)
            return self._entry.is_symlink()

    class Wrapped:
        def __init__(self, entries) -> None:
            self._entries = entries

        def __enter__(self):
            self._entries.__enter__()
            return (Unreachable(entry) for entry in self._entries)

        def __exit__(self, *exc):
            return self._entries.__exit__(*exc)

    monkeypatch.setattr(
        module.os, "scandir", lambda path=".": Wrapped(real_scandir(path))
    )
    shape = module.capture(str(config), anonymise=False)
    assert [entry["key"] for entry in shape["memory_dirs"]] == ["-good"]
    # Still one of the machine's projects, and now one nothing was read from.
    assert shape["projects_total"] == 2
    assert shape["skipped"] == 1


def test_a_project_that_holds_no_memory_directory_is_passed_over(tmp_path) -> None:
    """Two spellings of a project with nothing to read, both reachable by
    anyone who can write in `projects/`: a FILE at the name `memory`, and a
    file where the project directory should be — which makes the `memory`
    lookup a `NotADirectoryError` rather than a missing name.

    Neither is a failure to report, so neither is counted; what matters is
    that the capture finishes and the project beside them is in it.
    """
    config = tmp_path / "config"
    _write(_memory_dir(config, "-real") / "one.md", "x\n")
    _write(config / "projects" / "-file-at-memory" / "memory", "not a directory\n")
    _write(config / "projects" / "-not-a-project", "nor is this\n")
    shape = _shape("--config-dir", str(config), "--raw")
    assert [entry["key"] for entry in shape["memory_dirs"]] == ["-real"]
    assert shape["projects_total"] == 3
    assert shape["memory_dirs_total"] == 1
    assert shape["skipped"] == 0
    assert shape["read_errors"] == 0


@pytest.mark.skipif(ROOT, reason="root reads a file nobody else can")
def test_a_file_nobody_could_read_is_not_a_file_with_no_frontmatter(
    tmp_path,
) -> None:
    """The two records were byte-identical apart from the pseudonym: same
    size, four false flags, a null length — and `frontmatter_truncated: false`
    is a statement about a read that did not happen.

    The whole-capture `read_errors` total was the only trace, and it cannot be
    attributed to a file. The index record has answered this about its own
    read since the round before; this is the same answer one level down.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    _write(memory / "plain.md", "no frontmatter at all\n")
    blocked = _write(memory / "blocked.md", "no frontmatter at all\n")
    blocked.chmod(0o000)
    try:
        shape = _shape("--config-dir", str(config), "--raw")
    finally:
        blocked.chmod(0o600)
    files = {item["name"]: item for item in _by_key(shape)["-a"]["files"]}
    assert files["blocked.md"] != {**files["plain.md"], "name": "blocked.md"}
    assert files["blocked.md"]["size"] == files["plain.md"]["size"]
    assert files["blocked.md"]["unreadable"] is True
    # Nothing asserted about a read that did not happen, in either direction.
    for fact in READ_TYPES:
        assert files["blocked.md"][fact] is None, fact
        assert files["plain.md"][fact] is not None or fact == "description_len"
    assert shape["read_errors"] == 1


def test_a_file_the_rule_declined_to_open_states_no_read(tmp_path) -> None:
    """The last spelling of the same record: a memory file that is a link OUT.

    Nothing opens it — that is the rule, and it is the right rule — and it
    recorded `unreadable: false` beside four `false` flags and a null length,
    which is the record of a file that WAS read and carries no frontmatter.
    The two are told apart the way a failed read already is: every fact a read
    would have produced is null. `read_errors` does not move, because nothing
    failed here; the capture declined to look.
    """
    config = tmp_path / "config"
    outside = tmp_path / "elsewhere"
    theirs = _write(
        outside / "memo.md",
        "---\nname: theirs\ntype: note\ndescription: " + DESCRIPTION + "\n---\n\nb\n",
    )
    memory = _memory_dir(config, "-a")
    _write(memory / "own.md", "---\nname: ours\n---\n\nb\n")
    link = memory / "out.md"
    os.symlink(theirs, link)

    shape = _shape("--config-dir", str(config), "--raw")
    files = {item["name"]: item for item in _by_key(shape)["-a"]["files"]}
    record = files["out.md"]
    assert record["is_symlink"] is True
    # The LINK's size, so nothing outside the capture was measured.
    assert record["size"] == link.lstat().st_size != theirs.stat().st_size
    assert record["unreadable"] is True
    for fact in READ_TYPES:
        assert record[fact] is None, fact
    # The file beside it was read, and says the opposite in every field.
    assert files["own.md"]["unreadable"] is False
    assert files["own.md"]["has_name"] is True
    assert shape["read_errors"] == 0


@pytest.mark.skipif(ROOT, reason="root reads a file nobody else can")
def test_one_entry_that_cannot_be_measured_does_not_take_its_siblings(
    tmp_path,
) -> None:
    """The blast radius of a single bad name in a memory directory.

    `DirEntry.is_file` swallows FileNotFoundError and lets every other OSError
    out, and the handler that caught it wrapped the whole listing — so one
    `ln -s loop.md loop.md`, which anyone who can write in the tree can leave
    there, deleted the directory and every real memory in it from the capture.
    It was booked as `skipped` besides, which says the directory could not be
    listed when in fact it could.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    _write(memory / "good.md", "---\nname: good\n---\n\nbody\n")
    unreadable = _write(memory / "locked.md", "x\n")
    unreadable.chmod(0o000)
    os.symlink("loop.md", memory / "loop.md")
    try:
        shape = _shape("--config-dir", str(config), "--raw")
    finally:
        unreadable.chmod(0o600)
    entry = _by_key(shape)["-a"]
    # The loop is a name in this directory and never a file, so it is not
    # listed; the file whose bytes are unreadable is a file, and what could not
    # be measured is its frontmatter.
    assert [row["name"] for row in entry["files"]] == ["good.md", "locked.md"]
    assert shape["skipped"] == 0
    # One for the loop, one for the head that could not be read.
    assert shape["read_errors"] == 2


@pytest.mark.skipif(ROOT, reason="root reads a file nobody else can")
def test_an_index_that_cannot_be_read_is_not_an_index_with_no_rows(
    tmp_path,
) -> None:
    """An index whose bytes are unreachable recorded zero rows and zero
    dangling rows, which is the answer an empty index gives — so a rebuilt
    corpus reproduced a measurement nobody took. Unknown is the answer the
    linked-out index already gives, and the read failure is what says why."""
    config = tmp_path / "config"
    locked = _memory_dir(config, "-a")
    _write(locked / "real.md", "x\n")
    blocked_index = _write(locked / "MEMORY.md", "- [a](real.md) — hook\n")
    blocked_index.chmod(0o000)
    empty = _memory_dir(config, "-b")
    _write(empty / "real.md", "x\n")
    _write(empty / "MEMORY.md", "# Memory index\n")
    try:
        shape = _shape("--config-dir", str(config), "--raw")
    finally:
        blocked_index.chmod(0o600)
    listed = _by_key(shape)
    assert listed["-a"]["index"] is None
    # The index that could be read and holds no rows keeps saying so, which is
    # the state the null is now distinguishable from.
    assert listed["-b"]["index"] == {
        "rows": 0, "dangling_rows": 0, "truncated": False
    }
    # The file is still listed, and the failure is counted once rather than
    # twice for the two reads of the same name.
    assert [row["name"] for row in listed["-a"]["files"]] == [
        "MEMORY.md", "real.md"
    ]
    assert shape["read_errors"] == 1


@pytest.mark.skipif(ROOT, reason="root reads a directory nobody else can")
def test_a_projects_directory_that_cannot_be_listed_is_an_exit(tmp_path) -> None:
    """The failure the deployment reaches: piped over ssh under `sudo -n` into
    an NFS home, where root-squash turns a root read into EACCES.

    Every number below that point is a zero, so the capture read as a healthy
    machine with no projects and exited 0 — on a host the operator may not get
    a second run at. An ABSENT projects directory still captures, because a
    config directory with nothing written yet really is empty.
    """
    config = tmp_path / "config"
    projects = config / "projects"
    projects.mkdir(parents=True)
    _write(projects / "-a" / "memory" / "one.md", "x\n")
    projects.chmod(0o000)
    try:
        refused = _run("--config-dir", str(config))
    finally:
        projects.chmod(0o700)
    assert refused.returncode == 2, refused.stdout
    assert refused.stdout == "", "it failed and emitted a shape anyway"
    assert "harness_shape:" in refused.stderr

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _shape("--config-dir", str(empty))["projects_total"] == 0


def test_a_settings_file_that_cannot_be_read_is_a_state_not_an_absence(
    tmp_path,
) -> None:
    """"No settings at that scope" and "settings this run could not read" are
    different machines, and the scope was omitted for both."""
    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    _write(config / "settings.json", "{ not json")
    env = _managed_env(tmp_path / "no-managed")
    assert _shape("--config-dir", str(config), env=env)["settings"] == {
        "user": {
            "unreadable": True, "memory_keys": {}, "hooks": [], "plugins": []
        }
    }


def test_a_name_that_is_not_a_file_is_answered_rather_than_waited_on(
    tmp_path,
) -> None:
    """`open()` on a FIFO blocks until somebody writes to the other end.

    A capture runs unattended over ssh on a host the operator may get one run
    at, so a read that waits forever is the whole capture lost with nothing on
    stdout — worse than any wrong number, and indistinguishable from a slow
    NFS walk while it is happening. A settings file that is not a file is the
    same state as a directory at that path, which the tool already covers, and
    the rest of the tree is still captured around it.

    The FIFOs are made by this case. Nothing here goes near a real one.
    """
    config = tmp_path / "config"
    memory = config / "projects" / "-a" / "memory"
    memory.mkdir(parents=True)
    _write(memory / "real.md", "---\nname: a\n---\n")
    # In the tree these are not files and are never listed, so nothing opens
    # them; at the two settings paths the name is opened because it is named.
    os.mkfifo(str(memory / "queue.md"))
    os.mkfifo(str(config / "MEMORY.md"))
    os.mkfifo(str(config / "settings.json"))
    os.mkfifo(str(config / ".claude.json"))
    # A timeout of its own, well above what this tree costs: the rest of the
    # file gives a blocked capture five minutes to look like a slow one, and
    # what this case is about is that it does not block at all.
    run = None
    try:
        run = subprocess.run(
            [sys.executable, str(TOOL), "--config-dir", str(config)],
            capture_output=True, text=True, timeout=30,
            env=_managed_env(tmp_path / "none"),
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the capture is still waiting on a name that is not a file")
    assert run is not None and run.returncode == 0, run
    shape = json.loads(run.stdout)
    assert shape["settings"]["user"]["unreadable"] is True
    assert shape["harness"] == {"version_hint": None, "install": None}
    assert shape["read_errors"] == 0, shape["memory_dirs"]
    assert [item["name"] for item in _by_key(shape)["-s1"]["files"]] == ["m1.md"]


def test_a_device_that_never_ends_is_refused_rather_than_read_forever(
    tmp_path,
) -> None:
    """The other half of the rule the case above only tests one side of.

    `O_NONBLOCK` makes the open return; `S_ISREG` on the fd is what makes the
    READ end. A FIFO and `/dev/null` cannot tell the two apart — both answer
    an empty read, and the capture books `unreadable` either way — so the
    guard that stands between a settings path and an unbounded read had
    nothing asserting it. `/dev/zero` is decisive: the device never reaches
    EOF, and a build without the check reads until the machine gives out.

    THE ORDER HERE IS THE POINT. The guard is asserted directly, before
    anything is asked to read that name, so a build without it fails on the
    line below rather than filling memory from a device with no end — 23 GiB
    in five seconds, measured. Only then is the whole capture run, and what
    it must do is answer in milliseconds.
    """
    if not os.path.exists("/dev/zero"):
        pytest.skip("no /dev/zero on this platform")
    with pytest.raises(OSError):
        _tool_module()._open_regular("/dev/zero")

    config = tmp_path / "config"
    (config / "projects").mkdir(parents=True)
    os.symlink("/dev/zero", str(config / "settings.json"))
    run = None
    try:
        run = subprocess.run(
            [sys.executable, str(TOOL), "--config-dir", str(config)],
            capture_output=True, text=True, timeout=15,
            env=_managed_env(tmp_path / "none"),
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the capture is still reading a device that never ends")
    assert run is not None and run.returncode == 0, run
    assert json.loads(run.stdout)["settings"] == {
        "user": {"unreadable": True, "memory_keys": {}, "hooks": [], "plugins": []}
    }


def _tool_module():
    """`tools/harness_shape.py` loaded as a module, which it otherwise is not.

    It is a dev tool outside the installed package and it imports nothing from
    memkit on purpose, so loading the file is the only way to hold its copied
    constants next to the originals.
    """
    spec = importlib.util.spec_from_file_location("harness_shape_under_test", TOOL)
    assert spec is not None and spec.loader is not None, TOOL
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_constants_copied_from_memkit_are_the_ones_memkit_holds(
    monkeypatch,
) -> None:
    """Every constant copied into the capture tool, against the one memkit holds.

    The walk comparison does not stand in for this: it agrees whenever both
    sides read one tree the same way, whatever they call things. And it reaches
    only what `harness_memory` owns — `CONSOLIDATE_LOCK` and
    `_managed_dir()` are copied from `cli_doctor`, which the equivalence test
    never imports — so a rename there desynchronises this tool with nothing
    going red. All four agree today; this is what says so tomorrow.
    """
    # The seam OFF, or this compares the test's own override against the real
    # platform path and fails for the one reason that is not drift.
    monkeypatch.delenv(MANAGED_DIR_ENV, raising=False)
    module = _tool_module()
    assert module.MANAGED_DIR_ENV == MANAGED_DIR_ENV
    assert module.INDEX_NAME == harness_memory.INDEX_NAME
    assert module.DIRECTORY_KEY == harness_memory.DIRECTORY_KEY
    assert module.MEMORY_KEYS == (
        harness_memory.ENABLED_KEY,
        harness_memory.DIRECTORY_KEY,
        harness_memory.DREAM_KEY,
    )
    assert module.CONSOLIDATE_LOCK == cli_doctor.CONSOLIDATE_LOCK
    assert module._managed_dir() == cli_doctor._managed_dir()
    # And the two the ARTIFACT GATE holds a fixture to. Re-typed above like
    # every other vocabulary in this file, so the gate cannot be widened by
    # widening the thing it is checking; memkit owns the names, and a scope or
    # a switch it stops reading is one this gate should stop admitting.
    assert set(harness_memory.SCOPE_ORDER) == SCOPE_OK
    assert {
        harness_memory.ENABLED_KEY,
        harness_memory.DIRECTORY_KEY,
        harness_memory.DREAM_KEY,
    } == MEMORY_KEY_OK


def test_the_lock_the_doctor_would_read_is_the_one_recorded(tmp_path) -> None:
    """Both spellings at once, which is the state the two-place lookup exists
    for and the only one in which the ORDER is observable.

    `cli_doctor` takes the project directory's lock first and this took the
    memory directory's, so a tree rebuilt from `lock_age_s` reports an age
    doctor never would. The case above cannot see it: its two projects hold
    one lock each, so either order answers the same.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-both")
    _write(memory / "one.md", "x\n")
    now = time.time()
    for parent, age in ((memory, 900), (memory.parent, 7200)):
        lock = parent / cli_doctor.CONSOLIDATE_LOCK
        lock.write_text("", encoding="utf-8")
        os.utime(lock, (now - age, now - age))
    entry = _by_key(_shape("--config-dir", str(config), "--raw"))["-both"]
    assert 7100 < entry["lock_age_s"] < 7300, "the memory directory's lock won"


def test_a_description_is_measured_the_way_the_checker_measures_it(tmp_path) -> None:
    """The number is only worth carrying if it is the CHECKER's number.

    `>155 characters` is decided by `memory_integrity._scalar`, which strips a
    matching pair of quotes and undoes their escapes before measuring — and 50
    of the 72 harness-written memories on the capture host are quoted, so this
    was two characters long over most of a real machine, with one file sitting
    at 154 by the checker and 156 here. The expected lengths come from
    `_scalar` itself rather than from literals: a copy pinned to a number
    somebody typed is a copy that drifts from what it is a copy of.
    """
    config = tmp_path / "config"
    memory = _memory_dir(config, "-a")
    raws = (
        "a plain one",
        '"about memory: a fact"',
        '"say \\"hi\\""',
        "'it''s here'",
    )
    for number, raw in enumerate(raws):
        _write(memory / f"m{number}.md", f"---\ndescription: {raw}\n---\n\nbody\n")
    # A continued description, which NEITHER reader folds: the checker's
    # frontmatter parser skips the indented line as a nested key, and the
    # recall hook's regex is `(.+)$` with no DOTALL.
    _write(
        memory / "continued.md",
        "---\ndescription: first line\n  and a continuation\n---\n\nbody\n",
    )
    # And a block scalar, which the checker refuses as DESC-BAD before
    # measuring anything and the hook reads as one character.
    _write(
        memory / "block.md",
        "---\ndescription: >\n  one line\n  and another\n---\n\nbody\n",
    )
    files = {
        item["name"]: item
        for item in _by_key(_shape("--config-dir", str(config), "--raw"))["-a"]["files"]
    }
    for number, raw in enumerate(raws):
        value, error = memory_integrity._scalar(raw)
        assert error is None, (raw, error)
        assert value is not None
        assert files[f"m{number}.md"]["description_len"] == len(value), raw
    assert files["continued.md"]["description_len"] == len("first line")
    assert files["block.md"]["description_len"] == 1

    # AND THE FIVE SPELLINGS THE CHECKER REFUSES OUTRIGHT. `_scalar` answers
    # an error rather than a value for each, and the shape answers the length
    # of what was written — which is the narrower claim the docstring makes,
    # and it is here so that claim is watched rather than asserted in prose.
    refused = ("", '"never closed', ">", "a plain one: with a colon", "hash #here")
    rejects = _memory_dir(config, "-b")
    for number, raw in enumerate(refused):
        value, error = memory_integrity._scalar(raw)
        assert value is None and error, (raw, value, error)
        _write(rejects / f"r{number}.md", f"---\ndescription: {raw}\n---\n\nbody\n")
    written = {
        item["name"]: item
        for item in _by_key(_shape("--config-dir", str(config), "--raw"))["-b"]["files"]
    }
    for number, raw in enumerate(refused):
        assert written[f"r{number}.md"]["description_len"] == len(raw), raw


# The 3.9 spellings a 3.8 grammar accepts and a 3.8 interpreter does not.
# Attribute names first, then the (module, attribute) pairs where a bare name
# would false-positive on somebody's own attribute, then whole modules.
THREE_NINE_METHODS = ("removeprefix", "removesuffix")
THREE_NINE_CALLS = (
    ("functools", "cache"),
    ("ast", "unparse"),
    ("random", "randbytes"),
    ("math", "nextafter"),
    ("os", "pidfd_open"),
)
THREE_NINE_MODULES = ("zoneinfo", "graphlib")


def _three_nine_offences(tree: ast.AST) -> list:
    """Every 3.9-only spelling this walk knows how to see, with its line.

    `|` is the one that needs care in both directions. Bitwise OR on integers
    is ordinary 3.8 — `os.O_WRONLY | os.O_CREAT` is in this very tool — so a
    blanket ban on `BitOr` is a guard that fires on correct code, which is the
    fastest way to get a guard deleted. What it looks for instead is the two
    spellings that are not integers: a dict DISPLAY on either side of the
    operator, and any `|` inside an annotation, which is PEP 604. A merge of
    two dict-valued NAMES is invisible here, and pyright cannot see it either
    now that typeshed has dropped 3.8; that one is on review.
    """
    found = []
    annotated = set()
    for node in ast.walk(tree):
        for holder in (
            getattr(node, "annotation", None), getattr(node, "returns", None)
        ):
            if holder is not None:
                annotated.update(id(inner) for inner in ast.walk(holder))
    dicts = (ast.Dict, ast.DictComp)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if id(node) in annotated:
                found.append((node.lineno, "`|` in an annotation is PEP 604, 3.10"))
            elif isinstance(node.left, dicts) or isinstance(node.right, dicts):
                found.append((node.lineno, "`dict | dict` merge is 3.9"))
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.BitOr)
            and isinstance(node.value, dicts)
        ):
            found.append((node.lineno, "`dict |= dict` merge is 3.9"))
        elif isinstance(node, (ast.With, ast.AsyncWith)) and len(node.items) > 1:
            # Conservative on purpose: an unparenthesised multi-item `with` is
            # legal 3.8, but the parenthesised spelling is 3.9 and
            # `feature_version` does not reject it — so the two are
            # indistinguishable here and one file can afford the stricter rule.
            found.append((node.lineno, "a multi-item `with` may be the 3.9 spelling"))
        elif isinstance(node, ast.Attribute):
            if node.attr in THREE_NINE_METHODS:
                found.append((node.lineno, f"str.{node.attr} is 3.9"))
            elif isinstance(node.value, ast.Name):
                for module, attr in THREE_NINE_CALLS:
                    if node.value.id == module and node.attr == attr:
                        found.append((node.lineno, f"{module}.{attr} is 3.9"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in THREE_NINE_MODULES:
                    found.append((node.lineno, f"{alias.name} is 3.9"))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in THREE_NINE_MODULES
        ):
            found.append((node.lineno, f"{node.module} is 3.9"))
    return found


def test_the_38_walk_sees_the_spellings_it_claims_to() -> None:
    """The guard above is an assertion that a list is empty, which is what an
    empty list says whether or not the walk can see anything at all.

    So the walk is pointed at each spelling first. A guard nobody has watched
    fire is the same object as a guard that does not.
    """
    for source in (
        "x = {'a': 1} | {'b': 2}\n",
        "x = {'a': 1}\ny = x | {'b': 2}\n",
        "x = {}\nx |= {'b': 2}\n",
        "def f(a: int | None) -> str | None:\n    return None\n",
        "with open('a') as a, open('b') as b:\n    pass\n",
        "x = 'ab'.removeprefix('a')\n",
        "x = 'ab'.removesuffix('b')\n",
        "import functools\n@functools.cache\ndef f():\n    pass\n",
        "import ast\nx = ast.unparse(None)\n",
        "import random\nx = random.randbytes(1)\n",
        "import math\nx = math.nextafter(1.0, 2.0)\n",
        "import os\nx = os.pidfd_open(1)\n",
        "import zoneinfo\n",
        "from graphlib import TopologicalSorter\n",
    ):
        # Every one of these PARSES at 3.8, which is the whole problem.
        tree = ast.parse(source, feature_version=(3, 8))
        assert _three_nine_offences(tree), source
    # And the 3.8 spellings it must NOT complain about — a guard that fires on
    # correct code is a guard somebody deletes rather than reads.
    for clean in (
        "with open('a') as a:\n    x = a.read()\n",
        "import os\nx = os.O_WRONLY | os.O_CREAT | os.O_TRUNC\n",
        "def f(a, b):\n    return a | b\n",
        "from typing import Optional\ndef f(a: Optional[int]) -> Optional[str]:\n"
        "    return None\n",
    ):
        tree = ast.parse(clean, feature_version=(3, 8))
        assert _three_nine_offences(tree) == [], clean


def test_harness_shape_parses_as_python_38() -> None:
    """The floor is 3.8, not this repository's 3.9: the capture host it was
    first piped to runs 3.8.18, and it ran there unmodified.

    TWO GUARDS, because neither is sufficient. `feature_version=(3, 8)` gates
    the grammar the PEG parser applies — a walrus is allowed, `match` and
    `except*` are rejected — and it is the only check that survives the ssh
    pipe. It cannot see a parenthesised context manager, and it cannot see a
    3.9+ stdlib call at all, because those parse fine at every feature
    version. So the walk below greps the parse tree for the spellings a
    reviewer was otherwise asked to grep for by hand.

    `pyrightconfig-shape38.json` is the third guard and catches a different
    thing again — a PEP-585 subscript evaluated at runtime, which is an error
    at 3.8 and legal at 3.9. It cannot cover the stdlib half either: typeshed
    dropped Python 3.8, so its stubs no longer carry the version guards that
    would make `str.removeprefix` an error. Hence this list.
    """
    source = TOOL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TOOL), feature_version=(3, 8))
    assert _three_nine_offences(tree) == []


def test_the_tool_imports_nothing_it_could_not_find_on_a_stranger_s_machine(
) -> None:
    """It is piped over ssh to a host with no uv, no memkit and no copy of this
    repository. A third-party import is a capture that dies on the machine
    worth capturing, and a `memkit` import is the same failure wearing a name
    that looks safe from in here."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert "memkit" not in imported
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, sorted(
        imported - set(sys.stdlib_module_names)
    )


def _fixture_files(directory: Path) -> list:
    """Every file below `directory`, whatever it is called.

    RECURSIVE and suffix-agnostic. `glob("*.json")` covered the top level of a
    directory whose natural organisation is one subdirectory per host, so an
    un-anonymised capture one level down went through the whole file green,
    and so did one at the top level under any other suffix. Everything under
    here is a fixture and has to pass; nothing here is skipped for being
    called something the gate did not expect.
    """
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _fixture_shape(path: Path) -> dict:
    """The shape in `path`, or a failure — never a skip.

    A file under the fixtures directory that is not a shape is the thing this
    gate is for: it is either a capture nobody finished or one nobody meant to
    commit, and both are worth a red test.
    """
    try:
        shape = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AssertionError(f"{path}: not a shape this gate can read: {exc}") from exc
    assert isinstance(shape, dict), path
    return shape


def _tracked_fixtures(repo: Path = REPO, directory: Path = SHAPES):
    """The files git holds under the fixtures directory, or None outside a
    checkout.

    The walk above says every file present is gated; this says the files
    present are the ones that were reviewed. A shape that arrived some other
    way — copied in while debugging, or written by a capture aimed at the
    wrong directory — is then a red test rather than a fixture nobody chose.
    An unpacked archive is not a checkout and has no answer to give, which is
    why this is allowed to have none.

    The two arguments default to this repository's own and are here so the
    rule below can be run against a checkout a test builds: the real fixtures
    directory is not somewhere a case may plant a file to see the guard fire.
    """
    found = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", str(directory)],
        capture_output=True, text=True, timeout=300,
    )
    if found.returncode != 0:
        return None
    return sorted(repo / name for name in found.stdout.split("\0") if name)


def _fixtures_are_the_files_that_were_committed(
    repo: Path = REPO, directory: Path = SHAPES
) -> None:
    """Every file under `directory` is one `repo` was asked to keep.

    The other half of the artifact gate. The walk says every file present is
    checked for names; this says the files present are the ones somebody
    reviewed — and outside a checkout there is no answer, so nothing is
    claimed.
    """
    tracked = _tracked_fixtures(repo, directory)
    if tracked is None:
        return
    assert _fixture_files(directory) == tracked, (
        "a file under the fixtures directory nobody committed"
    )


def test_the_artifact_gate_finds_a_fixture_wherever_it_was_put(tmp_path) -> None:
    """What the gate enumerates is the gate.

    An un-anonymised shape at `harness_shapes/hosts/prod-laptop.json` passed
    every case in this file, and so did one at the top level under a suffix
    nobody globbed for — the gate was reading its own expectations rather than
    the directory. Held here on a planted tree rather than on the real one,
    because a case that writes into `tests/data/` to prove a point is a case
    that leaves a fixture behind when it fails.
    """
    planted = {
        "top.json",
        "hosts/prod-laptop.json",
        "hosts/archive/older.shape",
        "notes.txt",
    }
    for name in planted:
        _write(tmp_path / name, "{}\n")
    assert {
        str(path.relative_to(tmp_path)) for path in _fixture_files(tmp_path)
    } == planted
    # And a file that is not a shape fails rather than being passed over.
    with pytest.raises(AssertionError):
        _fixture_shape(_write(tmp_path / "notes.txt", "not a shape at all\n"))
    with pytest.raises(AssertionError):
        _fixture_shape(_write(tmp_path / "list.json", "[]\n"))


def _typed(record: dict, types: dict, where) -> None:
    """Every field in `types` is the kind of thing the schema says it is."""
    for field, allowed in types.items():
        kinds = allowed if isinstance(allowed, tuple) else (allowed,)
        value = record[field]
        assert isinstance(value, kinds), (where, field, value)
        if int in kinds:
            # `isinstance(True, int)`, so a plain int check admits a flag.
            assert not isinstance(value, bool), (where, field, value)


def _gated(shape: dict, where) -> None:
    """Everything one committed shape has to satisfy.

    A function rather than a loop body so the rules can be run over a planted
    shape too: what an artifact gate is worth is what it REJECTS, and there is
    no other way to watch that without committing the thing it must reject.
    """
    assert shape["schema"] == 1, where
    assert shape["tool"] == "harness_shape", where
    assert shape["anonymised"] is True, where
    assert shape["memory_dirs"], (where, "a shape of nothing tests nothing")
    # A committed fixture is a COMPLETE capture. A half-failed one is a
    # tree nobody can rebuild, and it looks exactly like a small machine.
    assert shape["skipped"] == 0, where
    # Subscripted, never `.get`: a fixture without this counter predates
    # the field set the tool emits, and a default that stands in for it
    # reads as a complete capture. The KeyError is the right failure.
    assert shape["read_errors"] == 0, where
    _typed(shape, SHAPE_TYPES, where)
    # The two harness rows, which are adopter-controlled strings and were
    # the only free text in either committed fixture. Typed BEFORE they are
    # read: a record addressed by key is gated by its key set (`_field_sets`)
    # and by its value types, the way every other record here is.
    _typed(shape["harness"], HARNESS_TYPES, where)
    version = shape["harness"]["version_hint"]
    assert version is None or VERSION_RE.fullmatch(version), (where, version)
    install = shape["harness"]["install"]
    assert install is None or install in INSTALL_OK, (where, install)
    for entry in shape["memory_dirs"]:
        assert KEY_RE.match(entry["key"]), (where, entry["key"])
        _typed(entry, DIR_TYPES, where)
        for item in entry["files"]:
            assert FILE_RE.match(item["name"]), (where, item["name"])
            _typed(item, FILE_TYPES, where)
            _typed(
                item, UNREAD_TYPES if item["unreadable"] else READ_TYPES, where,
            )
        if entry["index"] is not None:
            _typed(entry["index"], INDEX_TYPES, where)
    for scope_name, scope in shape["settings"].items():
        # The KEY, not just what hangs off it: a scope named after an
        # employer's policy set carries the name in the one place every rule
        # below was looking past.
        assert scope_name in SCOPE_OK, (where, scope_name)
        _typed(scope, SCOPE_TYPES, where)
        for key, value in scope["memory_keys"].items():
            assert key in MEMORY_KEY_OK, (where, key)
            if key == harness_memory.DIRECTORY_KEY:
                assert value in (None, "<path>"), (where, value)
            else:
                # A switch, the placeholder that says it was set to something
                # else, or the null the settings file actually declared —
                # which the harness reads as absent, and which a shape may
                # therefore carry. Never the something else.
                assert value is None or value == "<set>" or isinstance(
                    value, bool
                ), (where, key, value)
        for event in scope["hooks"]:
            assert event in HOOK_OK or HOOK_PSEUDONYM_RE.fullmatch(event), (
                where, event,
            )
        for plugin in scope["plugins"]:
            assert PLUGIN_RE.match(plugin), (where, plugin)


def test_the_committed_shapes_carry_no_names() -> None:
    """The artifact gate: every shape checked into this repository, whatever it
    is called and wherever under the fixtures directory it sits.

    Read off the directory rather than from a list of names, so a fixture is
    covered by the commit that adds it and nothing has to be remembered. It
    skips only while there are no fixtures at all, which is the window between
    the tool landing and the first capture being reviewed.
    """
    fixtures = _fixture_files(SHAPES) if SHAPES.is_dir() else []
    if not fixtures:
        pytest.skip(
            "no shapes captured yet — "
            "`python3 tools/harness_shape.py --out tests/data/harness_shapes/<name>.json`"
        )
    _fixtures_are_the_files_that_were_committed()
    for path in fixtures:
        _gated(_fixture_shape(path), path.name)


def test_a_fixture_nobody_committed_is_a_failure_of_the_gate(tmp_path) -> None:
    """The half of the artifact gate that says these files were REVIEWED, held
    to its own rule.

    It has been correct and unwatched: nothing turned red if the comparison
    went away, so the guard against a shape that arrived some other way — a
    capture aimed at the wrong directory, a copy left behind while debugging —
    was one edit from being decoration. Run on a checkout this case builds,
    never on `tests/data/`: a case that plants a file there to prove a point
    is a case that leaves a fixture behind when it fails.
    """
    repo = tmp_path / "checkout"
    shapes = repo / "tests" / "data" / "harness_shapes"
    _write(shapes / "one.json", "{}\n")

    # Not a checkout: no answer to give, and nothing claimed either way.
    assert _tracked_fixtures(repo, shapes) is None
    _fixtures_are_the_files_that_were_committed(repo, shapes)

    def git(*args: str) -> None:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=300,
        )
        assert done.returncode == 0, done.stdout + done.stderr

    git("init", "-q")
    git("add", "--", str(shapes))
    assert _tracked_fixtures(repo, shapes) == [shapes / "one.json"]
    _fixtures_are_the_files_that_were_committed(repo, shapes)

    _write(shapes / "two.json", "{}\n")
    with pytest.raises(AssertionError, match="nobody committed"):
        _fixtures_are_the_files_that_were_committed(repo, shapes)


def _with_files(shape: dict) -> dict:
    """The first memory directory in `shape` that has a file to injure."""
    return next(entry for entry in shape["memory_dirs"] if entry["files"])


def _injected(shape: dict, which: str) -> dict:
    """A copy of `shape` carrying one real name where the gate was not looking."""
    planted = json.loads(json.dumps(shape))
    if which == "a scope named after an organisation":
        planted["settings"]["AcmeCorp-Internal-Policy"] = planted["settings"]["user"]
    elif which == "a switch key named after an organisation":
        planted["settings"]["user"]["memory_keys"]["AcmeCorpPolicyGate"] = True
    elif which == "a path where a file size belongs":
        _with_files(planted)["files"][0]["size"] = "/Users/alice/src/acme/secret.md"
    elif which == "a path where a description length belongs":
        _with_files(planted)["files"][0]["description_len"] = "/opt/acme/bin"
    elif which == "a hostname where a lock age belongs":
        _with_files(planted)["lock_age_s"] = "acme-corp-host-01"
    elif which == "a flag where a count belongs":
        _with_files(planted)["key_len"] = True
    elif which == "a path under the harness record":
        planted["harness"]["captured_from"] = "/Users/alice/.claude"
    elif which == "a number where a version hint belongs":
        planted["harness"]["version_hint"] = 20250908
    elif which == "an extra field in a file record":
        _with_files(planted)["files"][0]["symlink_target_kind"] = "socket"
    else:
        raise AssertionError(which)
    return planted


@pytest.mark.parametrize(
    "injection",
    [
        "a scope named after an organisation",
        "a switch key named after an organisation",
        "a path where a file size belongs",
        "a path where a description length belongs",
        "a hostname where a lock age belongs",
        "a flag where a count belongs",
        "a number where a version hint belongs",
    ],
)
def test_the_artifact_gate_rejects_a_name_it_was_not_looking_at(injection) -> None:
    """The gate read every VALUE it knew the shape of and nothing else.

    So a settings scope keyed `AcmeCorp-Internal-Policy`, carrying the exact
    field set the tool emits, passed both gates — and so did a `/Users` path
    sitting in `files[].size`, because nothing said a size is a number. Both
    are unreachable from the tool as it stands; the gate's job is the fixture
    that is stale or hand-edited, which is how the field-set gate beside it
    came to be written.
    """
    fixtures = _fixture_files(SHAPES) if SHAPES.is_dir() else []
    if not fixtures:
        pytest.skip("no shapes captured yet")
    shape = _fixture_shape(fixtures[0])
    _gated(shape, fixtures[0].name)
    with pytest.raises(AssertionError):
        _gated(_injected(shape, injection), "planted")


def _field_sets(shape: dict) -> dict:
    """The field names a shape uses, one set per kind of record.

    A set of frozensets rather than one union, so a shape whose records
    disagree among themselves shows up here as two entries rather than as a
    single wider set.
    """
    dirs, files, indexes, scopes = set(), set(), set(), set()
    for entry in shape["memory_dirs"]:
        dirs.add(frozenset(entry))
        files.update(frozenset(item) for item in entry["files"])
        if entry["index"] is not None:
            indexes.add(frozenset(entry["index"]))
    for scope in shape["settings"].values():
        scopes.add(frozenset(scope))
    return {
        "shape": {frozenset(shape)},
        "harness": {frozenset(shape["harness"])},
        "memory_dir": dirs,
        "file": files,
        "index": indexes,
        "scope": scopes,
    }


def test_a_committed_shape_carries_the_field_set_the_tool_emits_today(
    tmp_path,
) -> None:
    """One `schema` number, one document. Both fixtures declared schema 1 while
    one of them predated five fields and still carried a sixth the tool had
    stopped emitting — so a consumer that branched on the number got a
    `KeyError` off the older file.

    The expectation is TAKEN FROM A LIVE CAPTURE rather than typed out here:
    what makes a committed shape usable is that it is the document this tool
    produces, and a list maintained beside the tool is another copy to drift.
    Subset over a set of frozensets is equality for every record kind a
    fixture has any of, and vacuous for one it has none of — a capture with no
    index is a machine, not a defect.
    """
    fixtures = _fixture_files(SHAPES) if SHAPES.is_dir() else []
    if not fixtures:
        pytest.skip("no shapes captured yet")
    emitted = _field_sets(_shape("--config-dir", str(_tree(tmp_path / "plain"))))
    # A second tree for the settings scope, which the first one has none of.
    settings = _field_sets(
        _shape(
            "--config-dir",
            str(_sentinel_tree(tmp_path / "settings")),
            env=_managed_env(tmp_path / "no-managed"),
        )
    )
    for kind, found in settings.items():
        emitted[kind] |= found
    for path in fixtures:
        shape = _fixture_shape(path)
        for kind, found in _field_sets(shape).items():
            stale = [sorted(record) for record in found - emitted[kind]]
            assert not stale, (path.name, kind, stale)


@pytest.mark.parametrize(
    "injection,kind",
    [
        ("a path under the harness record", "harness"),
        ("an extra field in a file record", "file"),
    ],
)
def test_the_field_set_gate_rejects_a_key_the_tool_does_not_emit(
    injection, kind, tmp_path,
) -> None:
    """A record read by key is gated by its key set too.

    The gate above read `harness.version_hint` and `harness.install` by name
    and built no field set for the record holding them, so a third key there —
    `captured_from`, an absolute path — passed both gates, while the same
    planting one level up was caught. The set of kinds `_field_sets` returns
    is the list of records anything may address by key.
    """
    fixtures = _fixture_files(SHAPES) if SHAPES.is_dir() else []
    if not fixtures:
        pytest.skip("no shapes captured yet")
    emitted = _field_sets(_shape("--config-dir", str(_tree(tmp_path / "plain"))))
    shape = _fixture_shape(fixtures[0])
    # The same comparison over the untouched fixture, so a kind the live
    # capture has none of cannot pass this test by being empty.
    assert not _field_sets(shape)[kind] - emitted[kind], kind
    planted = _injected(shape, injection)
    assert _field_sets(planted)[kind] - emitted[kind], injection


# The real entry point, reached in a child that first makes the machine
# hostile. Two of the states below cannot be handed to a process from outside
# it: an argument holding a NUL byte does not fit through `argv`, and a
# descriptor table with one entry left cannot be inherited across an `exec`
# that still has its own imports to finish — measured, at every limit from one
# free descriptor upward, the interpreter either starts and the walk has room
# or neither happens.
_HOSTILE_DOOR = '''import importlib.util
import os
import sys

tool, mode, tree = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("harness_shape_hostile", tool)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
argv = ["--config-dir", tree]
if mode == "nul":
    argv += ["--out", os.path.join(tree, "nul\\0dir", "shape.json")]
elif mode == "squeeze":
    # One whole capture first, so every import and every lazily read locale
    # file this run needs is taken while there are descriptors to take them
    # with: what is being measured is the walk under exhaustion, not the
    # import system under it.
    module.capture(tree)
    held = []
    while True:
        try:
            held.append(os.open(os.devnull, os.O_RDONLY))
        except OSError:
            break
    os.close(held.pop())
sys.exit(module.main(argv))
'''

# Every hostile tree, and the door it must leave by. The expected code is in
# the id because a table that allowed either door for every case would pass
# with the contract gone.
_HOSTILE_TREES = {
    "a NUL byte in the destination path -> exit 2": 2,
    "a fifo where a memory file goes -> exit 0": 0,
    "a symlink loop at a project and at a memory file -> exit 0": 0,
    "an unreadable projects directory -> exit 2": 2,
    "one free descriptor for the whole walk -> exit 0": 0,
    "a projects directory that is a dangling link -> exit 2": 2,
    "a memory file whose name is not UTF-8 -> exit 0": 0,
    "an index row deeper than the recursion limit -> exit 0": 0,
    "--raw with the shape going down a pipe -> exit 0": 0,
    "--out into a directory that refuses the write -> exit 2": 2,
}


def _hostile_tree(kind, tmp_path):
    """One hostile machine, and the command that reads it.

    Returns the argv, and the modes to put back afterwards — a directory left
    at 0o000 takes the rest of the session's temporary directory with it.
    """
    config = tmp_path / "config"
    if kind.startswith("a projects directory that is a dangling link"):
        config.mkdir()
        os.symlink(str(tmp_path / "never-here"), str(config / "projects"))
        return [sys.executable, str(TOOL), "--config-dir", str(config)], []
    memory = _memory_dir(config, "-a")
    _write(memory / "MEMORY.md", "- [a](hot/deep.md)\n")
    _write(memory / "hot" / "deep.md", "---\nname: d\n---\nx\n")
    plain = [sys.executable, str(TOOL), "--config-dir", str(config)]
    if kind.startswith("a NUL byte") or kind.startswith("one free descriptor"):
        door = _write(tmp_path / "door.py", _HOSTILE_DOOR)
        mode = "nul" if kind.startswith("a NUL byte") else "squeeze"
        return [sys.executable, str(door), str(TOOL), mode, str(config)], []
    if kind.startswith("a fifo"):
        os.mkfifo(str(memory / "note.md"))
        return plain, []
    if kind.startswith("a symlink loop"):
        loop = config / "projects" / "-loop"
        os.symlink(str(loop), str(loop))
        ring = memory / "ring.md"
        os.symlink(str(ring), str(ring))
        return plain, []
    if kind.startswith("an unreadable projects directory"):
        projects = config / "projects"
        projects.chmod(0o000)
        return plain, [(projects, 0o755)]
    if kind.startswith("a memory file whose name is not UTF-8"):
        try:
            with open(os.path.join(os.fsencode(str(memory)), b"\xff.md"), "wb") as h:
                h.write(b"x\n")
        except OSError as exc:
            pytest.skip(f"this filesystem refuses the name: {exc.strerror}")
        return plain, []
    if kind.startswith("an index row deeper"):
        rows = "a/" * 4000
        _write(memory / "MEMORY.md", f"- [a]({rows}x.md)\n")
        return plain, []
    if kind.startswith("--raw"):
        return plain + ["--raw"], []
    refuses = tmp_path / "refuses"
    refuses.mkdir()
    refuses.chmod(0o500)
    return plain + ["--out", str(refuses / "shape.json")], [(refuses, 0o755)]


@pytest.mark.skipif(ROOT, reason="root reads and writes what nobody else can")
@pytest.mark.parametrize("kind", sorted(_HOSTILE_TREES))
def test_hostile_trees_exit_zero_or_two_with_no_traceback(kind, tmp_path) -> None:
    """The exit contract, held against the machines nobody enumerated.

    Every closure of this class so far guarded one member of the syscall
    family at one site — a NUL byte reaching a stat, a descriptor taken above
    the handler that releases it, an unguarded `..` open, an unguarded write
    to a pipe — and the next site was open again. What a wrapper on the far
    end of an ssh pipe is owed does not depend on which site: a shape and a 0,
    or one line naming this tool and a 2, for anything a config directory and
    the machine under it can be.

    A traceback is the failure this is written against, because it means the
    number the wrapper read was chosen by the interpreter rather than by this
    tool.
    """
    doors = list(_HOSTILE_TREES.values())
    # A table that allowed either door for every case would pass with no
    # contract at all, so both doors are named and both are populated.
    assert doors.count(2) >= 3 and doors.count(0) >= 2, doors
    argv, restore = _hostile_tree(kind, tmp_path)
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    finally:
        for path, mode in restore:
            path.chmod(mode)
    assert run.returncode == _HOSTILE_TREES[kind], run.stdout + run.stderr
    assert "Traceback" not in run.stderr, run.stderr
    if run.returncode == 2:
        lines = run.stderr.strip().splitlines()
        assert len(lines) == 1, run.stderr
        assert lines[0].startswith("harness_shape:"), run.stderr
        assert run.stdout == "", "it failed and emitted a shape anyway"
    else:
        assert json.loads(run.stdout)["schema"] >= 1, run.stdout


# Every helper here that JUDGES — a path, a descriptor, a repository — and the
# one-line reason its answer has to be read. The judgement and the act on it
# are one question asked once: a call whose value is dropped has asked it and
# then done something else, which is the shape three rounds of this file closed
# one instance at a time.
_GUARDS = (
    ("_regular_fd", "the descriptor IS the answer: an fstat judged what was opened"),
    ("_resolves_inside", "whether a link reaches bytes this capture is reading"),
    ("_outside", "whether a row target escapes the directory being walked"),
    ("_row_present", "whether the row names something in the directory"),
    ("_open_dir", "a descriptor for the level, refusing a link or a file at it"),
    ("_landing_dir", "where a write actually lands, which is not its name"),
    ("_occupant", "which way a destination name is already taken"),
    ("_git_says_worktree", "git's own answer about a directory"),
    ("_worktree_above", "whether a checkout stands above the descriptor"),
    ("_inside_worktree", "the same question asked about a directory name"),
    ("_stdout_is_a_file", "whether fd 1 is a file rather than a pipe"),
    ("_stdout_destination", "the path fd 1 writes to, or no answer"),
    ("_is_own_config_dir", "whether the tree named is this machine's own"),
    ("create", "_Landing: the descriptor of the destination it just made"),
    ("inside_worktree", "_Landing: the checkout question asked of the descriptor"),
)


def test_every_guard_call_uses_the_value_it_returns() -> None:
    """A judgement nobody reads is a judgement that decided nothing.

    Each name above answers a question, and the caller acts on the answer; a
    call standing alone as a statement has asked and then gone on to act on
    something else — the path re-derived from a string, the directory reopened
    by name, the descriptor taken twice. Every instance of that class in this
    file was a real defect, and the cheap catch for the next one is that no
    call to any of them may be an expression statement or be assigned to `_`.
    """
    source = TOOL.read_text(encoding="utf-8")
    tree = ast.parse(source, str(TOOL))
    named = frozenset(name for name, _ in _GUARDS)

    def called(node):
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Name) and node.func.id in named:
            return node.func.id
        if isinstance(node.func, ast.Attribute) and node.func.attr in named:
            return node.func.attr
        return None

    # The lint is worth nothing if the names have moved: it would pass over a
    # file that calls none of them.
    reached = {
        answer
        for answer in (called(node) for node in ast.walk(tree))
        if answer is not None
    }
    assert reached == named, sorted(named - reached)
    dropped = []
    for node in ast.walk(tree):
        # A statement, or an assignment to the name that means "not read":
        # `_ = guard(...)` drops the answer with a comment's worth of ceremony
        # and nothing else.
        if isinstance(node, ast.Expr) or (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id == "_"
                for target in node.targets
            )
        ):
            answer = called(node.value)
            if answer is not None:
                dropped.append((answer, node.lineno))
    assert dropped == [], dropped
