"""Unit tests for `memkit.harness_memory`.

Three answers about somebody else's feature — where it writes, what it has
written, and which settings file decides — and each of them is a fact about the
harness rather than about memkit. So the cases here script the harness's own
shapes on disk: a linked worktree that must key to its main checkout, a project
directory holding nothing but an index, a `~/` that has to expand before any
path comparison is worth anything.

The one shape that is RECORDED rather than required is the submodule. Nothing
here measured what the harness does with one, so the case says what memkit
yields — a key that is at least the submodule's own — and refuses to dress it
up as the harness's rule.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest

from memkit import cli_doctor as doctor
from memkit import harness_memory


def _git() -> str:
    return shutil.which("git") or ""


def _init(root: pathlib.Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    (root / "a.md").write_text("x\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "a.md"],
        cwd=root, check=True, timeout=60,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x"],
        cwd=root, check=True, timeout=60,
    )


def _key(path) -> str:
    """The sanitised spelling of a path, derived the way the harness does."""
    return str(path).replace("/", "-").replace(".", "-")


@pytest.fixture
def config_dir(tmp_path, monkeypatch) -> pathlib.Path:
    """A scratch harness config dir, with `HOME` and the managed scope inside it.

    Every scope `settings_scopes` reads has to be under the fixture or the
    developer's own `~/.claude/settings.json` answers for the precedence cases —
    and the managed one is a fixed absolute path, so it is redirected rather
    than written.
    """
    home = tmp_path / "home"
    config = tmp_path / "claude-config"
    project = tmp_path / "project"
    for path in (home, config, project):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(doctor.CONFIG_DIR_ENV, str(config))
    monkeypatch.setattr(doctor, "_managed_dir", lambda: str(tmp_path / "managed"))
    (tmp_path / "managed").mkdir(exist_ok=True)
    monkeypatch.chdir(project)
    return config


def _memories(config: pathlib.Path, key: str, *names: str) -> pathlib.Path:
    directory = config / "projects" / key / "memory"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(
            f"---\ndescription: {name}\n---\n\nbody\n", encoding="utf-8"
        )
    return directory


def _settings(path: pathlib.Path, **blob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")


# --- what the harness has written --------------------------------------------


def test_a_config_dir_with_nothing_written_yields_an_empty_inventory(config_dir):
    """Two spellings of "nothing", because they arrive by different routes: a
    machine that has never run the feature has no `projects/` at all, and one
    whose projects hold no memories has the directory and no answer in it."""
    assert harness_memory.inventory(str(config_dir)) == []
    (config_dir / "projects").mkdir()
    assert harness_memory.inventory(str(config_dir)) == []
    (config_dir / "projects" / "-home-u" / "memory").mkdir(parents=True)
    assert harness_memory.inventory(str(config_dir)) == []


def test_the_inventory_counts_memories_and_not_the_index(config_dir) -> None:
    """Three project directories, one of each shape the field has.

    The index-only directory is the one that decides this function's contract:
    `MEMORY.md` is written by the harness whether or not anything else is, so
    counting it would report memories to an adopter who has none — and it is
    still carried in `files`, because anything that moves the directory has to
    move it too.
    """
    _memories(config_dir, "-home-u-git-app", "one.md", "two.md", "MEMORY.md")
    _memories(config_dir, "-home-u-index-only", "MEMORY.md")
    elsewhere = config_dir / "linked-store"
    elsewhere.mkdir()
    (elsewhere / "note.md").write_text("x\n", encoding="utf-8")
    linked = config_dir / "projects" / "-home-u-linked"
    linked.mkdir(parents=True)
    (linked / "memory").symlink_to(elsewhere)

    found = harness_memory.inventory(str(config_dir))
    assert [p.key for p in found] == ["-home-u-git-app", "-home-u-linked"]

    app, link = found
    assert app.files == ["MEMORY.md", "one.md", "two.md"]
    assert app.memories == 2
    assert app.is_symlink is False
    assert app.path == str(config_dir / "projects" / "-home-u-git-app" / "memory")
    # A SYMLINKED directory is listed and flagged, never silently skipped: it
    # is the shape an adopter who already wired the harness at their store by
    # hand is in, and it is the one adoption must not move.
    assert link.is_symlink is True
    assert link.memories == 1


def test_the_inventory_skips_what_it_cannot_read_rather_than_raising(config_dir):
    """One unreadable directory must not take the other 3917 with it: this runs
    on a real config dir where a stale mount or a root-owned entry is ordinary,
    and a diagnostic that dies there reports nothing at all."""
    if os.geteuid() == 0:
        pytest.skip("root reads everything, so this cannot be staged")
    _memories(config_dir, "-home-u-git-app", "one.md")
    blocked = _memories(config_dir, "-home-u-blocked", "one.md")
    blocked.chmod(0o000)
    try:
        assert [p.key for p in harness_memory.inventory(str(config_dir))] == [
            "-home-u-git-app"
        ]
    finally:
        blocked.chmod(0o700)


def test_a_file_is_not_a_project_directory(config_dir) -> None:
    """`projects/` holds directories, and a file sitting in it is skipped on
    the same rule as an unreadable one rather than raising NotADirectoryError."""
    (config_dir / "projects").mkdir()
    (config_dir / "projects" / "stray.json").write_text("{}", encoding="utf-8")
    _memories(config_dir, "-home-u-git-app", "one.md")
    assert [p.key for p in harness_memory.inventory(str(config_dir))] == [
        "-home-u-git-app"
    ]


def test_every_shape_a_link_can_take_is_recorded_separately(config_dir) -> None:
    """THREE flags, because whatever copies one of these owes each a different
    answer: a linked memory directory is already wired somewhere else and must
    not be moved, a linked project directory is reached through somebody's
    link, and a linked file inside is a memory whose bytes live outside the
    directory being copied.

    One flag over the memory directory alone reported the other two as
    ordinary, which is the reading that makes a copy follow a link out of the
    config directory and move a file it does not own.
    """
    _memories(config_dir, "-p-plain", "a.md")

    elsewhere = config_dir / "target"
    elsewhere.mkdir()
    (elsewhere / "b.md").write_text("x\n", encoding="utf-8")
    (config_dir / "projects" / "-p-linked-memory").mkdir(parents=True)
    (config_dir / "projects" / "-p-linked-memory" / "memory").symlink_to(elsewhere)

    real = config_dir / "real" / "memory"
    real.mkdir(parents=True)
    (real / "c.md").write_text("x\n", encoding="utf-8")
    (config_dir / "projects" / "-p-linked-project").symlink_to(config_dir / "real")

    directory = _memories(config_dir, "-p-linked-file")
    outside = config_dir / "outside.md"
    outside.write_text("x\n", encoding="utf-8")
    (directory / "d.md").symlink_to(outside)

    found = {project.key: project for project in harness_memory.inventory(config_dir)}
    assert sorted(found) == [
        "-p-linked-file", "-p-linked-memory", "-p-linked-project", "-p-plain",
    ]

    plain = found["-p-plain"]
    assert (plain.is_symlink, plain.linked_project, plain.linked_files) == (
        False, False, (),
    )
    assert plain.linked is False

    memory = found["-p-linked-memory"]
    assert memory.is_symlink is True and memory.linked is True

    project = found["-p-linked-project"]
    assert project.linked_project is True and project.is_symlink is False
    assert project.linked is True

    linked_file = found["-p-linked-file"]
    assert linked_file.linked_files == ("d.md",)
    assert linked_file.is_symlink is False and linked_file.linked is True


def test_a_directory_named_like_a_memory_is_not_one(config_dir) -> None:
    """`memory/notes.md/` is a directory whose name ends in `.md`, and counted
    as a memory it offers an adopter something to move that no move can carry.
    The file test is what separates them, and it is the only thing that does.
    """
    (config_dir / "projects" / "-p-dir" / "memory" / "notes.md").mkdir(parents=True)
    assert harness_memory.inventory(str(config_dir)) == []
    _memories(config_dir, "-p-dir", "real.md")
    found = harness_memory.inventory(str(config_dir))
    assert [(p.key, p.files) for p in found] == [("-p-dir", ["real.md"])]


# --- which project this is ---------------------------------------------------


def test_the_project_key_comes_from_the_repository_not_the_directory(tmp_path):
    """Every worktree and every subdirectory of one checkout keys to ONE
    directory, which is what makes "this project's memories" a single answer.

    The linked worktree is the case that cannot be got right by path prefix: it
    lives outside the main checkout's path entirely, and only its git common
    dir says which repository it belongs to.
    """
    if not _git():
        pytest.skip("no git")
    home = pathlib.Path(os.path.realpath(tmp_path))
    root = home / "main"
    root.mkdir()
    _init(root)
    (root / "src" / "deep").mkdir(parents=True)
    linked = home / "linked"
    subprocess.run(
        ["git", "worktree", "add", "-q", "--detach", str(linked)],
        cwd=root, check=True, timeout=60,
    )
    assert harness_memory.project_key(str(root)) == _key(root)
    assert harness_memory.project_key(str(root / "src" / "deep")) == _key(root)
    assert harness_memory.project_key(str(linked)) == _key(root)


def test_outside_a_repository_the_key_is_the_directory(tmp_path) -> None:
    """No repository above the cwd is an ANSWER about the world, not a failure:
    the harness keys on the directory there, and so does this."""
    plain = pathlib.Path(os.path.realpath(tmp_path)) / "no.repo.here"
    plain.mkdir()
    assert harness_memory.project_key(str(plain)) == _key(plain)
    # And `.` is sanitised as well as `/`, which is the half a path with a
    # dotted directory name is the only witness for.
    assert "no-repo-here" in harness_memory.project_key(str(plain))


def test_a_key_is_produced_for_a_directory_that_is_not_there(tmp_path) -> None:
    """`project_key` never raises. What hangs off that is a doctor run in a
    session whose workdir was removed underneath it — the report has to carry
    its other twenty-five checks rather than a traceback."""
    gone = str(tmp_path / "removed")
    assert harness_memory.project_key(gone) == _key(gone)


def test_two_submodules_of_one_superproject_do_not_share_a_key(tmp_path):
    """TWO of them, because one cannot show the defect this case exists for.

    A submodule's `.git` is a file naming a directory under the SUPERPROJECT's
    git dir — `<super>/.git/modules/<name>` — and no `commondir` sits beside
    it, so the common-dir walk lands on `<super>/.git/modules`: the same answer
    for every submodule of that superproject. Two of them keyed to one
    directory is two projects' memories in one pile, and the later migration
    reading that key moves the wrong ones.

    What memkit answers instead is RECORDED rather than required: the
    submodule's own worktree root, which is at minimum unique. What the harness
    itself keys a submodule to was never measured, and if it ever is, this is
    the case that says what changed.
    """
    if not _git():
        pytest.skip("no git")
    home = pathlib.Path(os.path.realpath(tmp_path))
    super_root = home / "super"
    for path in (super_root, home / "one", home / "two"):
        path.mkdir()
        _init(path)
    for name in ("suba", "subb"):
        added = subprocess.run(
            [
                "git", "-c", "protocol.file.allow=always",
                "-c", "user.email=t@t", "-c", "user.name=t",
                "submodule", "add", "-q",
                str(home / ("one" if name == "suba" else "two")), name,
            ],
            cwd=super_root, capture_output=True, text=True, timeout=120,
        )
        if added.returncode != 0:
            pytest.skip(f"this git refuses a file-transport submodule: {added.stderr}")
    assert (super_root / "suba" / ".git").is_file()
    first = harness_memory.project_key(str(super_root / "suba"))
    second = harness_memory.project_key(str(super_root / "subb"))
    assert first != second
    assert first == _key(super_root / "suba")
    assert second == _key(super_root / "subb")
    assert first != _key(super_root / ".git" / "modules")
    # And the shapes that are NOT submodules keep the common-dir answer, which
    # is the whole reason the walk goes through it.
    assert harness_memory.project_key(str(super_root)) == _key(super_root)


def test_the_default_directory_is_the_key_under_the_config_dir(config_dir):
    """The path an adopter is told the harness writes to.

    ONE SPELLING, with no trailing separator. It carried one, to read as a
    directory rather than a file, and every surface prints it through
    `_display_path` — which re-spells a path under `$HOME` relative to it and
    normalises the separator away in doing so. So the separator survived for a
    config directory outside HOME and nowhere else, and `~/.claude` is where
    every real install keeps this.
    """
    cwd = os.getcwd()
    default = harness_memory.default_dir(str(config_dir), cwd)
    assert default == os.path.join(
        str(config_dir), "projects", harness_memory.project_key(cwd), "memory"
    )
    assert not default.endswith(os.sep)


def test_a_cwd_that_is_not_an_absolute_path_is_refused(config_dir) -> None:
    """`""` is what a caller holds once `os.getcwd()` has failed, and keyed it
    disappears: `os.path.join` swallows the empty component and the answer is
    `<config dir>/projects/memory`, a path that reads as derived and is
    nowhere. A relative one is the same defect arriving by a different route —
    it would be resolved against whatever directory the PROCESS stands in,
    which is the second walk this module's contract refuses.
    """
    for cwd in ("", "sub", "./sub", os.curdir):
        with pytest.raises(ValueError):
            harness_memory.project_key(cwd)
        with pytest.raises(ValueError):
            harness_memory.default_dir(str(config_dir), cwd)


def test_a_path_the_os_will_not_resolve_is_answered_rather_than_raised():
    """An embedded NUL is legal in JSON and legal in a settings file, and
    `realpath` raises `ValueError` on one rather than `OSError` — so the
    never-raises contract is only kept by catching both. What hangs off it is a
    doctor row: an exception escaping this walk demotes the whole check to
    UNKNOWN.
    """
    assert harness_memory.project_key("/tmp/a\x00b") == "-tmp-a\x00b"


# --- which settings file decides ---------------------------------------------


def test_the_directory_is_read_in_the_order_the_harness_reads_it(config_dir):
    """Managed over local over user, and the answer names the scope — a remedy
    that said "your settings" over four candidate files is one nobody can act
    on."""
    scopes = doctor.settings_scopes
    assert harness_memory.configured_dir(scopes()) == (None, None)

    _settings(config_dir / "settings.json", autoMemoryDirectory="/u/user")
    assert harness_memory.configured_dir(scopes()) == ("/u/user", "user")

    _settings(
        pathlib.Path.cwd() / ".claude" / "settings.local.json",
        autoMemoryDirectory="/u/local",
    )
    assert harness_memory.configured_dir(scopes()) == ("/u/local", "local")

    _settings(
        pathlib.Path(doctor._managed_dir()) / doctor.MANAGED_SETTINGS_NAME,
        autoMemoryDirectory="/u/managed",
    )
    assert harness_memory.configured_dir(scopes()) == ("/u/managed", "managed")


def test_the_checked_in_project_scope_is_not_read_for_this_key(config_dir):
    """The harness IGNORES `autoMemoryDirectory` in a checked-in
    `.claude/settings.json`, so that cloning a repository cannot redirect where
    an agent writes its memories.

    Reading it here would make the report name a directory nothing writes to —
    and would quietly hand a repository the answer to "where are this adopter's
    memories", which is the thing the harness's own rule exists to refuse.
    """
    _settings(
        pathlib.Path.cwd() / ".claude" / doctor.SETTINGS_NAME,
        autoMemoryDirectory="/u/from-the-repository",
    )
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (None, None)
    assert "project" not in harness_memory.DIRECTORY_SCOPES

    # And it does not mask a scope that IS read.
    _settings(config_dir / "settings.json", autoMemoryDirectory="/u/user")
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (
        "/u/user", "user",
    )


def test_a_tilde_is_expanded_before_anything_compares_the_path(config_dir):
    """`~/notes/search` and the directory it names are one directory. Compared
    unexpanded against a store's corpus root, a setting pointing straight at
    the store reports as outside every store — the exact wrong answer, with a
    remedy telling the adopter to change a setting that is already right."""
    _settings(config_dir / "settings.json", autoMemoryDirectory="~/notes/search")
    directory, where = harness_memory.configured_dir(doctor.settings_scopes())
    assert directory == os.path.join(os.environ["HOME"], "notes", "search")
    assert where == "user"


def test_a_value_that_is_not_a_path_is_not_an_answer(config_dir) -> None:
    """An empty string and a non-string are two ways a settings file says
    nothing, and both have to read as "unset" rather than as a directory: what
    hangs off the difference is whether doctor names the derived default."""
    for value in ("", 17, None, ["/u/list"]):
        _settings(config_dir / "settings.json", autoMemoryDirectory=value)
        assert harness_memory.configured_dir(doctor.settings_scopes()) == (None, None)
