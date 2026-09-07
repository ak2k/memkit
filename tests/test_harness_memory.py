"""Unit tests for `memkit.harness_memory`.

Three answers about somebody else's feature — where it writes, what it has
written, and which settings file decides — and each of them is a fact about the
harness rather than about memkit. So the cases here script the harness's own
shapes on disk: a linked worktree that must key to its main checkout, a project
directory holding nothing but an index, a `~/` that has to expand before any
path comparison is worth anything.

The one shape that is RECORDED rather than required is the submodule. Nothing
here measured what the harness does with one, so the case says what memkit
yields and refuses to dress it up as a rule.
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


def test_what_a_submodule_checkout_yields_is_recorded_not_required(tmp_path):
    """A RECORDING. Nothing here measured what the harness keys a submodule to,
    so this states what memkit answers and nothing more.

    A submodule's `.git` is a file naming a directory under the SUPERPROJECT's
    git dir, and no `commondir` sits beside it — so the walk lands on
    `<super>/.git/modules`, which is neither the submodule's root nor the
    superproject's. If the harness is ever measured on one, this is the case
    that says what changed.
    """
    if not _git():
        pytest.skip("no git")
    home = pathlib.Path(os.path.realpath(tmp_path))
    super_root = home / "super"
    inner = home / "inner"
    for path in (super_root, inner):
        path.mkdir()
        _init(path)
    added = subprocess.run(
        [
            "git", "-c", "protocol.file.allow=always",
            "-c", "user.email=t@t", "-c", "user.name=t",
            "submodule", "add", "-q", str(inner), "sub",
        ],
        cwd=super_root, capture_output=True, text=True, timeout=120,
    )
    if added.returncode != 0:
        pytest.skip(f"this git refuses a file-transport submodule: {added.stderr}")
    assert (super_root / "sub" / ".git").is_file()
    yielded = harness_memory.project_key(str(super_root / "sub"))
    assert yielded == _key(super_root / ".git" / "modules")
    assert yielded != _key(super_root / "sub")
    assert yielded != _key(super_root)


def test_the_default_directory_is_the_key_under_the_config_dir(config_dir):
    """The path an adopter is told the harness writes to, spelled as a
    directory: it is printed far more often than it is opened."""
    cwd = os.getcwd()
    default = harness_memory.default_dir(str(config_dir), cwd)
    assert default == os.path.join(
        str(config_dir), "projects", harness_memory.project_key(cwd), "memory", ""
    )
    assert default.endswith(os.sep)


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
