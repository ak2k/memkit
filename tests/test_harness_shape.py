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
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from memkit import harness_memory

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
HOOK_RE = re.compile(r"[A-Za-z]+|h\d+")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _shape(*args: str) -> dict:
    out = _run(*args)
    assert out.returncode == 0, out.stdout + out.stderr
    return json.loads(out.stdout)


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
FOLDED = "one line and another"


def _tree(tmp_path: Path) -> Path:
    """One config directory holding every state the shape distinguishes.

    Four project directories: a normal one with an index and two memories, an
    index-only one, one whose `memory` is empty, and one whose `memory` is a
    symlink to somewhere else — plus a project directory with no `memory` at
    all, which is what 3,900 of the 3,923 entries on a real machine are.
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
        "---\ndescription: >\n  one line\n  and another\ntype: user\n---\n\nbody\n",
    )
    linked = config / "projects" / "-h-u-git-linked"
    linked.mkdir(parents=True)
    os.symlink(outside, linked / "memory")
    (config / "projects" / "-h-u-git-none").mkdir(parents=True)
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
    written = _run("--config-dir", str(config), "--out", str(out))
    assert written.returncode == 0, written.stdout + written.stderr
    shape = json.loads(out.read_text(encoding="utf-8"))
    # The same capture through the other exit: a shape is a file that gets
    # committed and a stream that gets piped over ssh, and the two have to be
    # the same document.
    assert shape == _shape("--config-dir", str(config))

    assert shape["schema"] == 1
    assert shape["tool"] == "harness_shape"
    assert shape["anonymised"] is True
    # No `$HOME` lookup anywhere: a temporary config directory has no version
    # and no install method, and a null here is what proves the capture did not
    # reach for the operator's own.
    assert shape["harness"] == {"version_hint": None, "install": None}
    assert shape["projects_total"] == 5
    assert shape["memory_dirs_total"] == 4
    assert shape["skipped"] == 0
    # The bare `memory` directory holds no `*.md`, so it is counted and not
    # listed; the other three are listed, index-only included.
    assert len(shape["memory_dirs"]) == 3

    listed = _by_key(shape)
    assert set(listed) == {"-s1-s2-s3-s4", "-s1-s2-s3-s5", "-s1-s2-s3-s6"}
    normal = listed["-s1-s2-s3-s4"]
    assert normal["key_len"] == len("-h-u-git-app")
    assert normal["is_symlink"] is False
    assert normal["symlink_target_kind"] is None
    assert normal["lock_age_s"] is None
    assert normal["index"] == {"lines": 4, "rows": 2, "dangling_rows": 1}
    source = config / "projects" / "-h-u-git-app" / "memory"
    files = {entry["name"]: entry for entry in normal["files"]}
    assert set(files) == {"MEMORY.md", "m1.md", "m2.md"}
    assert files["m1.md"] == {
        "name": "m1.md",
        "size": (source / "kept.md").stat().st_size,
        "has_frontmatter": True,
        "has_description": True,
        "description_len": len(DESCRIPTION),
        "has_name": True,
        "has_type": True,
    }
    assert files["m2.md"]["has_frontmatter"] is False
    assert files["m2.md"]["description_len"] is None
    assert files["m2.md"]["size"] == (source / "plain.md").stat().st_size

    index_only = listed["-s1-s2-s3-s5"]
    assert [entry["name"] for entry in index_only["files"]] == ["MEMORY.md"]
    assert index_only["index"] == {"lines": 3, "rows": 1, "dangling_rows": 1}

    linked = listed["-s1-s2-s3-s6"]
    assert linked["is_symlink"] is True
    assert linked["symlink_target_kind"] == "external"
    assert linked["index"] is None
    assert len(linked["files"]) == 1
    # A folded scalar is one description, and its length is the folded length —
    # the number the >155-character rule is decided on.
    assert linked["files"][0]["description_len"] == len(FOLDED)
    assert linked["files"][0]["has_type"] is True
    assert linked["files"][0]["has_name"] is False


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
    lock = inner / ".consolidate-lock"
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (stale, stale))
    outer = _memory_dir(config, "-b")
    _write(outer / "one.md", "x\n")
    beside = outer.parent / ".consolidate-lock"
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
        harness_memory.ENABLED_KEY: None,
        harness_memory.DIRECTORY_KEY: "<path>",
        harness_memory.DREAM_KEY: True,
    }
    # A pseudonym and a real event name, sorted as OUTPUT: `Gate--<sentinel>`
    # sorts first among the real keys and last here, which is the point.
    assert user["hooks"] == ["UserPromptSubmit", "h1"]
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
    adoptable = {
        entry["key"]: entry
        for entry in shape["memory_dirs"]
        if any(item["name"] != harness_memory.INDEX_NAME for item in entry["files"])
    }
    found = harness_memory.inventory(str(config))
    assert set(adoptable) == {project.key for project in found}
    assert {project.key: project.is_symlink for project in found} == {
        key: entry["is_symlink"] for key, entry in adoptable.items()
    }
    # The index-only directory is the difference between the two questions, and
    # it is a directory the tool reports and the inventory does not.
    assert "-h-u-git-empty" not in adoptable
    assert any(
        entry["key"] == "-h-u-git-empty" for entry in shape["memory_dirs"]
    )


def test_harness_shape_parses_as_python_38() -> None:
    """The floor is 3.8, not this repository's 3.9: the capture host it was
    first piped to runs 3.8.18, and it ran there unmodified.

    This is a SYNTAX guard and nothing more, with two holes worth naming. A
    walrus is allowed, `match` and `except*` are rejected — but parenthesised
    context managers are NOT rejected, because `feature_version` gates the
    grammar the PEG parser applies and not that spelling. And it cannot see a
    3.9+ STDLIB call at all, since those parse fine at every feature version;
    the two a reviewer greps for instead are `str.removeprefix` and dict `|`
    merge, which read as ordinary 3.8 syntax and die at run time on the host.
    """
    source = TOOL.read_text(encoding="utf-8")
    ast.parse(source, filename=str(TOOL), feature_version=(3, 8))


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


def test_the_committed_shapes_carry_no_names() -> None:
    """The artifact gate: every shape checked into this repository, whatever it
    is called.

    Read off the directory rather than from a list of names, so a fixture is
    covered by the commit that adds it and nothing has to be remembered. It
    skips only while there are no fixtures at all, which is the window between
    the tool landing and the first capture being reviewed.
    """
    fixtures = sorted(SHAPES.glob("*.json")) if SHAPES.is_dir() else []
    if not fixtures:
        pytest.skip(
            "no shapes captured yet — "
            "`python3 tools/harness_shape.py --out tests/data/harness_shapes/<name>.json`"
        )
    for path in fixtures:
        shape = json.loads(path.read_text(encoding="utf-8"))
        assert shape["schema"] == 1, path.name
        assert shape["tool"] == "harness_shape", path.name
        assert shape["anonymised"] is True, path.name
        assert shape["memory_dirs"], (path.name, "a shape of nothing tests nothing")
        # The two harness rows, which are adopter-controlled strings and were
        # the only free text in either committed fixture.
        version = shape["harness"]["version_hint"]
        assert version is None or VERSION_RE.fullmatch(version), (path.name, version)
        install = shape["harness"]["install"]
        assert install is None or install in INSTALL_OK, (path.name, install)
        for entry in shape["memory_dirs"]:
            assert KEY_RE.match(entry["key"]), (path.name, entry["key"])
            for item in entry["files"]:
                assert FILE_RE.match(item["name"]), (path.name, item["name"])
        for scope in shape["settings"].values():
            for key, value in scope["memory_keys"].items():
                if key == harness_memory.DIRECTORY_KEY:
                    assert value in (None, "<path>"), (path.name, value)
                else:
                    # A switch, or the null that says it was set to something
                    # else. Never the something else.
                    assert value is None or isinstance(value, bool), (path.name, key)
            for event in scope["hooks"]:
                assert HOOK_RE.fullmatch(event), (path.name, event)
            for plugin in scope["plugins"]:
                assert PLUGIN_RE.match(plugin), (path.name, plugin)
