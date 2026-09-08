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
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
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
    "has_frontmatter": bool,
    "has_description": bool,
    "has_name": bool,
    "has_type": bool,
    "frontmatter_truncated": bool,
    "description_len": (int, type(None)),
}
INDEX_TYPES = {"rows": int, "dangling_rows": int, "truncated": bool}
SCOPE_TYPES = {"unreadable": bool}


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
    # AND NEVER OPENED. The target carries frontmatter and a description; the
    # link records neither, and the size is the link's rather than the file's.
    assert by_name["m5.md"]["has_frontmatter"] is False
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
    assert files["linked.md"]["has_name"] is False

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
            _run("--config-dir", str(config), "--raw", "--out", str(dest / "leak.json"))
            assert not (checkout / "leak.json").exists(), (
                "real names landed in a checkout the run never named"
            )
    finally:
        stop.set()
        flipper.join()


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
    """
    config = tmp_path / "config"
    config.mkdir()
    os.symlink(tmp_path / "moved-away", config / "projects")
    refused = _run("--config-dir", str(config))
    assert refused.returncode == 2, refused.stdout
    assert refused.stdout == "", "it failed and emitted a shape anyway"
    assert "harness_shape:" in refused.stderr

    # The control the handler is there for: nothing written yet really is a
    # machine, and it still captures.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _shape("--config-dir", str(empty))["projects_total"] == 0


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
    # with its real size and no frontmatter — the same four flags an empty
    # file gets, which is why the count above is the thing that tells them
    # apart. (A file whose `lstat` fails records `size: null`, never 0.)
    assert files["two.md"]["size"] == unreadable_file.stat().st_size
    assert files["two.md"]["has_frontmatter"] is False
    assert files["one.md"]["has_frontmatter"] is False


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
    """The module docstring says running this tool against
    `harness_memory.inventory` is what keeps the copies in step.

    True only for the constants `harness_memory` owns. `CONSOLIDATE_LOCK` and
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


def _tracked_fixtures():
    """The files git holds under the fixtures directory, or None outside a
    checkout.

    The walk above says every file present is gated; this says the files
    present are the ones that were reviewed. A shape that arrived some other
    way — copied in while debugging, or written by a capture aimed at the
    wrong directory — is then a red test rather than a fixture nobody chose.
    An unpacked archive is not a checkout and has no answer to give, which is
    why this is allowed to have none.
    """
    found = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", str(SHAPES)],
        capture_output=True, text=True, timeout=300,
    )
    if found.returncode != 0:
        return None
    return sorted(REPO / name for name in found.stdout.split("\0") if name)


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
    # the only free text in either committed fixture.
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
    tracked = _tracked_fixtures()
    if tracked is not None:
        assert fixtures == tracked, "a file under the fixtures directory nobody committed"
    for path in fixtures:
        _gated(_fixture_shape(path), path.name)


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
