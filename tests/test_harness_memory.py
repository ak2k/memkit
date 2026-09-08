"""Unit tests for `memkit.harness_memory`.

Three answers about somebody else's feature — where it writes, what it has
written, and which settings file decides — and each of them is a fact about the
harness rather than about memkit. So the cases here script the harness's own
shapes on disk: a linked worktree that must key to its main checkout, a project
directory holding nothing but an index, a `~/` that has to expand before any
path comparison is worth anything.

The submodule and the linked worktree were RECORDED here before they were
measured, and they are measured now: 2.1.258 keys a submodule on itself and a
linked worktree on its main checkout, which is what these cases assert.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
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
    """The sanitised spelling of a path, derived the way the harness does.

    Measured on 2.1.258: every character outside `[A-Za-z0-9]` becomes `-`,
    one for one, with no run collapsing. Resolved first, because the harness
    keys on the process's own cwd and the kernel has already resolved that.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(str(path)))


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
    # The harness surfaces that are NOT a settings file. A developer running
    # the suite with one of these exported would otherwise get a different
    # answer from the fixture than CI does.
    for name in (harness_memory.DISABLE_ENV, *harness_memory.OVERRIDE_ENV):
        monkeypatch.delenv(name, raising=False)
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
    assert harness_memory.inventory(str(config_dir)) == ([], True, "")
    (config_dir / "projects").mkdir()
    assert harness_memory.inventory(str(config_dir)) == ([], True, "")
    (config_dir / "projects" / "-home-u" / "memory").mkdir(parents=True)
    assert harness_memory.inventory(str(config_dir)) == ([], True, "")


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

    found, _read_ok, _unreadable = harness_memory.inventory(str(config_dir))
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
        assert [p.key for p in harness_memory.inventory(str(config_dir))[0]] == [
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
    assert [p.key for p in harness_memory.inventory(str(config_dir))[0]] == [
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

    found = {
        project.key: project
        for project in harness_memory.inventory(config_dir)[0]
    }
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
    assert harness_memory.inventory(str(config_dir)) == ([], True, "")
    _memories(config_dir, "-p-dir", "real.md")
    found, _read_ok, _unreadable = harness_memory.inventory(str(config_dir))
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
    assert harness_memory.project_key("/tmp/a\x00b") == "-tmp-a-b"


def test_the_sanitiser_replaces_every_character_that_is_not_alphanumeric():
    """Measured on 2.1.258, and wider than the `/` and `.` an earlier reading
    of it had: `[^A-Za-z0-9]` maps ONE FOR ONE to `-`, with no run collapsing,
    so a space and an underscore go the same way a separator does and two
    adjacent ones stay two dashes.

    What hangs off the exact rule is a directory NAME, so a rule that is close
    is a report naming a directory that is not there.
    """
    key = harness_memory.project_key("/tmp/a b_c.d/e+f")
    assert key.endswith("-a-b-c-d-e-f")
    # `/tmp` resolves to `/private/tmp` on a mac and stays `/tmp` on Linux;
    # what this case is about is the characters, not the prefix.
    assert key.startswith("-")
    assert "_" not in key and " " not in key


def test_a_key_over_the_harnesss_cap_is_refused_rather_than_guessed(tmp_path):
    """The harness truncates a key at 200 characters and appends `-<base36>`
    derived from the untruncated path. That hash was not measured, so the two
    names this could return are one the harness does not use and a prefix that
    is no directory at all — and naming either is the defect every other
    refusal in this module exists to avoid.
    """
    deep = pathlib.Path(os.path.realpath(tmp_path))
    while len(str(deep)) <= harness_memory.KEY_MAX:
        deep = deep / "a-directory-with-a-long-enough-name"
    assert len(str(deep)) > harness_memory.KEY_MAX
    with pytest.raises(ValueError) as raised:
        harness_memory.project_key(str(deep))
    assert str(harness_memory.KEY_MAX) in str(raised.value)

    # And one character under the cap is answered normally.
    ok = "/" + "a" * (harness_memory.KEY_MAX - 1)
    assert len(harness_memory.project_key(ok)) == harness_memory.KEY_MAX


def test_the_key_is_the_physical_path_and_not_the_spelling_used(tmp_path):
    """Measured on 2.1.258: a session standing in a symlinked directory outside
    any repository wrote to the TARGET's key, not the link's — the harness keys
    on the process's own cwd, which the kernel resolved before the process ever
    saw it.

    Inside a repository the walk already resolves before it climbs; this is the
    case that does not, and it is the one an adopter with a symlinked scratch
    directory is in.
    """
    home = pathlib.Path(os.path.realpath(tmp_path))
    real = home / "real"
    real.mkdir()
    link = home / "link"
    link.symlink_to(real)
    assert harness_memory.project_key(str(link)) == harness_memory.project_key(
        str(real)
    )
    assert harness_memory.project_key(str(link)) == _key(real)

    # A CHECKOUT reached through a symlink was already right, because the walk
    # resolves before it climbs. Pinned so that stays true: only the two
    # give-up paths out of the derivation ever returned the caller's spelling.
    if not _git():
        return
    checkout = home / "checkout"
    checkout.mkdir()
    _init(checkout)
    (home / "through").symlink_to(checkout)
    assert harness_memory.project_key(str(home / "through")) == _key(checkout)


# --- which settings file decides ---------------------------------------------


def test_the_directory_is_read_in_the_order_the_harness_reads_it(config_dir):
    """Managed over local over project over user, measured on 2.1.258 out of
    the resolver, and the answer names the scope — a remedy that said "your
    settings" over four candidate files is one nobody can act on.

    LOCAL OVER USER is the half that had it backwards. The scopes arrive from
    `settings_scopes` in the order doctor reports them, which puts `user`
    second, and read in arrival order a `settings.local.json` redirecting the
    harness is invisible behind a `settings.json` that does not.
    """
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


def test_a_checked_in_project_scope_does_redirect_the_directory(config_dir):
    """MEASURED ON 2.1.258, and it overturns the rule this module was built on.

    The shipped schema text says the key is "Ignored if set in projectSettings
    (checked-in .claude/settings.json) for security". It is not: the resolver
    consults that file whenever its trust gate passes, and the gate returns
    true unconditionally for every NON-INTERACTIVE invocation — `claude -p`, a
    hook, the SDK, a subagent — whatever the folder trust state. A live probe
    with no trust recorded anywhere had the harness both announce and write to
    the directory a checked-in file named.

    So a clone decides where an agent writes its memories, and reading this
    scope is what lets the report say so. `local` still outranks it.
    """
    _settings(
        pathlib.Path.cwd() / ".claude" / doctor.SETTINGS_NAME,
        autoMemoryDirectory="/u/from-the-repository",
    )
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (
        "/u/from-the-repository", "project",
    )
    assert harness_memory.CHECKOUT_SCOPE in harness_memory.SCOPE_ORDER

    # It outranks user settings, and `settings.local.json` outranks it.
    _settings(config_dir / "settings.json", autoMemoryDirectory="/u/user")
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (
        "/u/from-the-repository", "project",
    )
    _settings(
        pathlib.Path.cwd() / ".claude" / doctor.LOCAL_SETTINGS_NAME,
        autoMemoryDirectory="/u/local",
    )
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (
        "/u/local", "local",
    )


def test_the_switches_are_read_in_the_same_order_as_the_directory(config_dir):
    """One order for all three keys. The two booleans were read in the order
    the scopes arrived rather than in the harness's, so a
    `settings.local.json` turning auto-memory back on sat behind a
    `settings.json` turning it off — and the row said memkit was the only
    memory system on the machine while the harness wrote.
    """
    scopes = doctor.settings_scopes
    assert harness_memory.switch(scopes(), harness_memory.ENABLED_KEY) == (None, None)

    _settings(config_dir / "settings.json", autoMemoryEnabled=False)
    assert harness_memory.switch(scopes(), harness_memory.ENABLED_KEY) == (
        False, "user",
    )

    _settings(
        pathlib.Path.cwd() / ".claude" / doctor.LOCAL_SETTINGS_NAME,
        autoMemoryEnabled=True,
    )
    assert harness_memory.switch(scopes(), harness_memory.ENABLED_KEY) == (
        True, "local",
    )

    # An explicit `null` is absence, which is what the harness's own `!= null`
    # makes it: the scope below decides.
    _settings(
        pathlib.Path.cwd() / ".claude" / doctor.LOCAL_SETTINGS_NAME,
        autoMemoryEnabled=None,
    )
    assert harness_memory.switch(scopes(), harness_memory.ENABLED_KEY) == (
        False, "user",
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
    for value in ("", 17, None, ["/u/list"], "relative/path", "/a"):
        _settings(config_dir / "settings.json", autoMemoryDirectory=value)
        assert harness_memory.configured_dir(doctor.settings_scopes()) == (None, None)


def test_a_value_the_harness_rejects_masks_the_scope_below_it(config_dir):
    """MEASURED: the resolver takes the first NON-NULL value it finds and
    validates afterwards, so a scope holding a value the harness refuses does
    not fall through to the next scope — it falls through to the DEFAULT
    directory.

    Read the other way, this reports a lower scope's good value as the answer
    and names a directory the harness does not write to, which is the exact
    shape of the defect this module exists to close. An explicit `null` is the
    one value that is absence rather than a bad answer.
    """
    _settings(config_dir / "settings.json", autoMemoryDirectory="/u/user")
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (
        "/u/user", "user",
    )

    local = pathlib.Path.cwd() / ".claude" / doctor.LOCAL_SETTINGS_NAME
    _settings(local, autoMemoryDirectory="")
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (None, None)

    _settings(local, autoMemoryDirectory=None)
    assert harness_memory.configured_dir(doctor.settings_scopes()) == (
        "/u/user", "user",
    )


# --- the surfaces that are not a settings file -------------------------------


def test_the_environment_decides_the_switch_above_every_settings_scope(
    config_dir, monkeypatch
) -> None:
    """`CLAUDE_CODE_DISABLE_AUTO_MEMORY=0` turns the feature ON before the
    harness reads a settings file at all, which is the direction that matters:
    an adopter who set `autoMemoryEnabled: false` has a second memory system
    running and every settings scope says otherwise.

    Read from the 2.1.258 code: the value is lower-cased and trimmed, one word
    list turns the feature off and the other forces it on, and a word in
    neither list falls through to the settings.
    """
    _settings(config_dir / "settings.json", autoMemoryEnabled=False)
    for spelling in ("0", "false", "no", "off", " OFF ", "False"):
        monkeypatch.setenv(harness_memory.DISABLE_ENV, spelling)
        assert harness_memory.env_switch() == (True, spelling), spelling
    for spelling in ("1", "true", "yes", "on", " ON ", "TRUE"):
        monkeypatch.setenv(harness_memory.DISABLE_ENV, spelling)
        assert harness_memory.env_switch() == (False, spelling), spelling
    # Unset, empty, and a word in neither list are all "the settings decide".
    for spelling in ("", "banana", "2"):
        monkeypatch.setenv(harness_memory.DISABLE_ENV, spelling)
        assert harness_memory.env_switch() == (None, ""), spelling
    monkeypatch.delenv(harness_memory.DISABLE_ENV)
    assert harness_memory.env_switch() == (None, "")


def test_an_environment_override_is_named_rather_than_resolved(
    config_dir, monkeypatch
) -> None:
    """Three variables outrank every settings scope for the DIRECTORY, and
    memkit resolves none of them. What it can do is say one is in effect, so
    the row stops claiming to know where the harness writes."""
    assert harness_memory.overrides() == ()
    monkeypatch.setenv("CLAUDE_CODE_PROJECT_DIR_NAME", "elsewhere")
    assert harness_memory.overrides() == ("CLAUDE_CODE_PROJECT_DIR_NAME",)
    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", "/u/somewhere")
    assert harness_memory.overrides() == (
        "CLAUDE_COWORK_MEMORY_PATH_OVERRIDE",
        "CLAUDE_CODE_PROJECT_DIR_NAME",
    )
    # An empty value is not an override: the harness's own reader refuses a
    # falsy one before it looks at anything.
    monkeypatch.setenv("CLAUDE_CODE_REMOTE_MEMORY_DIR", "")
    assert "CLAUDE_CODE_REMOTE_MEMORY_DIR" not in harness_memory.overrides()


def test_the_directory_is_the_one_the_harness_normalises_to(config_dir) -> None:
    """The validator memkit mirrors, read out of the 2.1.258 code: only a
    LEADING `~/` expands, a remainder that normalises to `.` or above the home
    directory is refused outright, and the value is normalised and stripped of
    trailing separators BEFORE the absolute / three-character / NUL tests.

    Validating the expanded string instead accepted four values the harness
    replaces with the default — and naming a directory the harness does not
    write to is the whole of what this module exists to prevent.
    """
    home = os.environ["HOME"]
    # `/a/` is the value the ORDER decides: three characters as written and two
    # once the separator is gone, so testing the string before it is stripped
    # calls it usable and names `/a` as a directory the harness never writes to.
    refused = (
        "~", "~/", "~/.", "~/..", "~/../elsewhere", ".", "..", "/a", "/a/", "/"
    )
    for value in refused:
        assert not harness_memory.usable_dir(value), value
    # Normalised and de-separated, so what is named is what the harness uses.
    for value, want in (
        ("/a/b/", "/a/b"),
        ("/a/b//", "/a/b"),
        ("/a/b/../c", "/a/c"),
        ("~/notes/", os.path.join(home, "notes")),
        ("~/notes/../notes/x", os.path.join(home, "notes", "x")),
    ):
        assert harness_memory.usable_dir(value), value
        _settings(config_dir / "settings.json", autoMemoryDirectory=value)
        assert harness_memory.configured_dir(doctor.settings_scopes()) == (
            want, "user",
        ), value


def test_the_inventory_survives_a_config_dir_that_is_not_a_path(tmp_path) -> None:
    """`scandir` raises `ValueError` rather than `OSError` on an embedded NUL,
    and this value comes from the environment. Unreachable through a POSIX
    environment variable, closed for the same reason the walk above it is: an
    exception escaping here demotes a whole doctor row to UNKNOWN."""
    # The path it names is the one it tried to list, spelled as it was built:
    # a caller that prints it prints what this process actually asked for.
    assert harness_memory.inventory("/c\x00d") == ([], False, "/c\x00d/projects")
    (tmp_path / "projects" / "-p" / "memory").mkdir(parents=True)
    assert harness_memory.inventory(str(tmp_path)) == ([], True, "")


def test_a_projects_directory_that_cannot_be_read_is_not_an_empty_one(
    tmp_path,
) -> None:
    """The enumeration failure is a state of its own, and has to be.

    Swallowed into `[]` it is byte-identical to "nothing has been written
    here", and a caller that reads that emptiness as fact then asserts an
    absence it never observed.
    """
    if os.geteuid() == 0:
        pytest.skip("root reads a directory whatever its mode says")
    projects = tmp_path / "projects"
    (projects / "-home-u" / "memory").mkdir(parents=True)
    (projects / "-home-u" / "memory" / "one.md").write_text("x\n", encoding="utf-8")
    found, read_ok, unreadable = harness_memory.inventory(str(tmp_path))
    assert [project.key for project in found] == ["-home-u"]
    assert (read_ok, unreadable) == (True, "")

    projects.chmod(0o000)
    try:
        found, read_ok, unreadable = harness_memory.inventory(str(tmp_path))
    finally:
        projects.chmod(0o755)
    assert found == []
    assert read_ok is False
    # AND WHICH DIRECTORY, because `read_ok` false no longer means this one:
    # a caller that names `projects/` over a failing project directory sends
    # its reader to a directory that is already readable.
    assert unreadable == str(projects)

    # A config directory with no `projects/` at all is a walk that FOUND
    # nothing, not one that failed: the difference between the two is the
    # whole of what this flag carries, and a machine that has never run the
    # feature is the commonest state there is.
    assert harness_memory.inventory(str(tmp_path / "nowhere")) == ([], True, "")


def test_a_memory_directory_that_will_not_list_is_named_by_the_walk(
    tmp_path,
) -> None:
    """Three ways a project holds no listing, and only one of them failed.

    No `memory/` at all, and a `memory` that is a file, are both "this project
    wrote nothing": the harness creates that directory with the first memory
    it puts there. A `memory/` that refuses to list is a walk that did not
    happen, and read as the first two it becomes an absence nothing observed.
    """
    if os.geteuid() == 0:
        pytest.skip("root reads a directory whatever its mode says")
    projects = tmp_path / "projects"
    (projects / "-p-none").mkdir(parents=True)
    (projects / "-p-file").mkdir()
    (projects / "-p-file" / "memory").write_text("x\n", encoding="utf-8")
    kept = projects / "-p-kept" / "memory"
    kept.mkdir(parents=True)
    (kept / "one.md").write_text("x\n", encoding="utf-8")

    found, read_ok, unreadable = harness_memory.inventory(str(tmp_path))
    assert [project.key for project in found] == ["-p-kept"]
    assert (read_ok, unreadable) == (True, "")

    shut = projects / "-p-shut" / "memory"
    shut.mkdir(parents=True)
    (shut / "one.md").write_text("x\n", encoding="utf-8")
    shut.chmod(0o000)
    try:
        found, read_ok, unreadable = harness_memory.inventory(str(tmp_path))
    finally:
        shut.chmod(0o755)
    assert [project.key for project in found] == ["-p-kept"]
    assert (read_ok, unreadable) == (False, str(shut))


def test_one_unanswerable_name_in_the_inventory_does_not_drop_its_siblings(
    config_dir,
) -> None:
    """A `.md` whose file test raises costs that name and nothing else.

    Guarded per directory, one symlink loop took the whole project out of the
    walk, and the count an adopter reads was a count of the directories that
    happened to answer. The guard is per NAME, so what a loop costs is the row
    it is on.
    """
    _memories(config_dir, "-p-loop", "one.md", "two.md")
    memory = config_dir / "projects" / "-p-loop" / "memory"
    (memory / "a.md").symlink_to(memory / "b.md")
    (memory / "b.md").symlink_to(memory / "a.md")

    found, read_ok, unreadable = harness_memory.inventory(str(config_dir))
    assert [(p.key, p.files) for p in found] == [("-p-loop", ["one.md", "two.md"])]
    # The DIRECTORY listed, which is what this flag is about: the name that
    # would not answer is off the list rather than counted as read.
    assert (read_ok, unreadable) == (True, "")


def test_a_project_that_will_not_answer_does_not_empty_the_inventory_walk(
    config_dir, monkeypatch
) -> None:
    """One entry raising on its link test loses that project, not the walk.

    No filesystem here refuses a link test on demand — `scandir` answers it
    from the directory entry — so the refusal is scripted. What it stands for
    is real: a config directory holds thousands of these, and an enumeration
    that dies on one of them reports nothing about the rest.
    """
    _memories(config_dir, "-p-one", "one.md")
    _memories(config_dir, "-p-two", "two.md")
    projects = str(config_dir / "projects")
    real = os.scandir

    class _Refusing:
        def __init__(self, entry) -> None:
            self.name = entry.name
            self.path = entry.path

        def is_symlink(self) -> bool:
            raise OSError(5, "input/output error")

    def scandir(path):
        if str(path) != projects:
            return real(path)
        with real(path) as entries:
            listed = [
                _Refusing(entry) if entry.name == "-p-two" else entry
                for entry in entries
            ]
        return contextlib.nullcontext(listed)

    monkeypatch.setattr(os, "scandir", scandir)
    found, read_ok, unreadable = harness_memory.inventory(str(config_dir))
    assert [project.key for project in found] == ["-p-one"]
    assert read_ok is False
    assert unreadable == str(config_dir / "projects" / "-p-two")
