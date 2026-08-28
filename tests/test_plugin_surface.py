"""The plugin surface: manifests, registration, and the three wrappers.

What every case here has in common is that the thing it pins lives in two
files that no compiler, importer or type checker connects. A manifest key and
the environment variable a shell script reads. A `timeout` in a JSON
registration and a constant in the hook. A python floor in the checker and the
probe that avoids it. Each pair is one edit away from disagreeing, and every
disagreement is silent at runtime — a hook that no longer receives its config
still exits 0 and still prints nothing, which is also what a corpus with
nothing to say looks like.

The wrappers are driven as real processes against a SHIM interpreter: a script
on PATH called `python3` that records the environment it was handed instead of
running anything. That is the only way to see what the wrapper decided, since
what it decides is what it exports into the process it replaces itself with.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from memkit import cli_doctor as doctor
from memkit import memory_prompt_recall as hook

REPO = Path(__file__).resolve().parent.parent
PLUGIN_MANIFEST = REPO / ".claude-plugin" / "plugin.json"
MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
HOOKS_JSON = REPO / "hooks" / "hooks.json"
BIN = REPO / "bin"
COMMON_SH = BIN / "lib" / "common.sh"

# The payload an adopter receives. A plugin installed from a github source is a
# clone, so a file that is not tracked is a file that is not there — for
# everyone except the person who wrote it.
PAYLOAD = [
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
    "hooks/hooks.json",
    "bin/memkit",
    "bin/memkit-hook",
    "bin/memkit-recall",
    "bin/lib/common.sh",
    "src/memkit/memory_prompt_recall.py",
    "src/memkit/common-words.txt",
    # The package's executor. Not reached from the hook — that path starts no
    # process and imports nothing from here — but the dispatcher's two
    # subcommands both import it at module scope, so it is on the 3.9 floor
    # and in the payload for the same reason `cli_doctor.py` is.
    "src/memkit/_exec.py",
    "src/memkit/cli.py",
    # Imported by the dispatcher at module scope, so it is on the 3.9 floor and
    # in the payload for the same reason `cli.py` is: `bin/memkit` runs the
    # dispatcher, and a dispatcher whose import fails is a plugin that installs
    # and answers nothing.
    "src/memkit/cli_doctor.py",
    "src/memkit/cli_init.py",
    "src/memkit/__init__.py",
    # The checker `bin/memkit` routes to when a local python meets the 3.12
    # floor: `MEMKIT_CHECKER_CMD` is `<python> -m memkit.memory_integrity`, run
    # against THIS tree, so leaving it out of the payload made that route name
    # a module the adopter does not have. Safe to add — its only first-party
    # import is `memkit.memory_prompt_recall`, already here.
    "src/memkit/memory_integrity.py",
    # The two skills. Not reached by any import, so the closure assertion below
    # cannot find them; they are payload in the sense that matters — a plugin
    # install without them registers `Skills (0)` and every command an agent
    # was told to reach for is unreachable.
    "skills/doctor/SKILL.md",
    "skills/init/SKILL.md",
]


# The real `sh`, located in THIS environment rather than assumed onto the
# child's PATH. The cases below hand a wrapper to a shell deliberately — it is
# the only way to leave `$0` bare, and the only way to get a trace — and the
# child's PATH is a claim those same cases are making about what the wrapper
# can reach. Resolving the binary here keeps the two apart: a case can say
# "nothing but a shim is reachable" and still be run by a shell.
SH = shutil.which("sh") or "/bin/sh"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# CI's ubuntu runner is a non-root user, which is the environment these
# adopters have. Under root nothing is unreadable, so a mode-000 case measures
# the ordinary path and fails for a reason that is about the runner.
needs_permissions = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads mode-000 files, so nothing is unreadable"
)


def _readme_section(heading: str) -> str:
    """One `## ` section of the README, heading to next heading of that level."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start : nxt if nxt != -1 else len(text)]


def _pinned_file_count() -> int | None:
    """How many files the marketplace pin names, or None where that cannot be
    read here.

    `_git("ls-tree", ...)` against a sha this checkout does not carry exits
    non-zero and prints nothing, and `len([])` is 0 — a number that reads like
    an answer. Callers need the difference, because "the pin has 0 files" and
    "this tree cannot see the pin" say opposite things about the note.
    """
    out = _git("ls-tree", "-r", "--name-only",
               _json(MARKETPLACE)["plugins"][0]["source"]["sha"])
    return len(out.stdout.split()) if out.returncode == 0 else None


def _needs_checkout() -> None:
    """Skip only where a checkout genuinely cannot exist, and FAIL elsewhere.

    These cases read the index and the commit graph, so the packaged nix leg —
    which builds from a store copy with no `.git` — cannot run them. That leg
    sets `MEMKIT_NO_CHECKOUT`, and everywhere else a missing checkout is a
    broken environment rather than a licence to pass: the plain-python job is
    where these are the gate, and a skip there reports green under the same
    check name as a run.
    """
    if os.environ.get("MEMKIT_NO_CHECKOUT") == "1":
        pytest.skip("packaged build — no .git in the store copy, by construction")
    assert (REPO / ".git").exists(), (
        "no .git here, and this context did not declare itself packaged. These "
        "cases read the index and the commit graph; skipping them silently is "
        "what makes a green check name mean nothing."
    )
    assert shutil.which("git"), "git is not on PATH"


# --- the manifests ------------------------------------------------------------


def test_the_option_key_mangles_to_the_variable_the_wrapper_reads() -> None:
    """The one pin that makes config delivery work at all.

    The harness builds the variable by uppercasing the option key with
    non-alphanumerics replaced by `_` (read out of the 2.1.238 bundle, and
    confirmed end to end: `memkitConfig` arrived as
    `CLAUDE_PLUGIN_OPTION_MEMKITCONFIG`). Nothing connects the key in the
    manifest to the name in the shell script, and a rename on either side
    leaves a plugin that installs, loads, reports healthy, and serves nothing.

    The mangling is applied here rather than asserted as a literal, so a key
    with an underscore or a digit is still checked against the real rule.
    """
    options = _json(PLUGIN_MANIFEST)["userConfig"]
    assert list(options) == ["memkitConfig"], options
    key = next(iter(options))
    expected = "CLAUDE_PLUGIN_OPTION_" + re.sub(r"[^A-Za-z0-9_]", "_", key).upper()
    assert expected in COMMON_SH.read_text(encoding="utf-8"), expected


def test_the_option_key_is_one_the_harness_will_accept() -> None:
    """Keys are identifier-shaped; a hyphen or a dot is not a warning.

    Measured on 2.1.238: `userConfig.some.dotted-key: Invalid key in record`,
    and a plugin carrying one installs and then reports `failed to load` — a
    state visible only in `claude plugin list`, not at install time.
    """
    for key in _json(PLUGIN_MANIFEST)["userConfig"]:
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key), key


def test_the_option_is_optional_and_says_what_it_is_for() -> None:
    """`required: false`, and it is a trade rather than a relaxation.

    A required option produces a typed prompt at enable, which is friction on
    exactly the no-TTY path the install story promises — and it was never a
    guarantee: a declared default is NOT exported to hook processes when the
    option is unset (measured), so a required-but-skipped install reached the
    hook with nothing on either rung anyway. What it bought was a warning; what
    it cost was a prompt on every scripted install.

    MEASURED on 2.1.241, and the whole decision rests on it: with
    `required: false`, a value passed as `--config memkitConfig=<path>` still
    arrives as `CLAUDE_PLUGIN_OPTION_MEMKITCONFIG` in the hook process. If that
    ever stops being true this option has to go back to required, because the
    unset case is silent by design and there would be nothing left to make the
    set case loud.
    """
    option = _json(PLUGIN_MANIFEST)["userConfig"]["memkitConfig"]
    assert option["required"] is False
    assert option["type"] == "string"
    assert option["default"].startswith("~/"), option["default"]
    for field in ("title", "description"):
        assert option[field].strip(), field
    # The description is rendered by the harness during `/plugin install`, so
    # it is the first screen a cold adopter reads — and it named
    # `/memkit:init` while `cli.py` still listed init in `_PENDING`. The
    # adopter ran it, got nothing, had no config, and the plugin stayed
    # silently inert.
    #
    # So any command it names in the present tense must exist.
    from memkit.cli import _HANDLERS, _PENDING

    described = option["description"]
    named = {n for n in (*_PENDING, *_HANDLERS) if f"/memkit:{n}" in described}
    assert named, described
    assert named <= set(_HANDLERS), sorted(named - set(_HANDLERS))
    # And the sentence that told an adopter to write it by hand is gone, since
    # a command now does it.
    assert "manual in this build" not in described, described


def test_the_marketplace_entry_pins_a_commit_rather_than_a_branch() -> None:
    """An unpinned same-repo source means every commit to main becomes hook
    code in an adopter's next session, on a surface that runs before every
    prompt. The schema accepts `ref` and `sha`, and `sha` wins.

    The sha is checked for SHAPE here and for EXISTENCE below, because a
    plausible-looking placeholder is the failure this is guarding: a 40-hex
    string that names no commit fails at `git clone`, on the adopter's machine,
    after they have accepted a trust dialog.
    """
    entry = _json(MARKETPLACE)["plugins"][0]
    source = entry["source"]
    assert re.fullmatch(r"[0-9a-f]{40}", source["sha"]), source
    assert "ref" not in source, "a ref beside a sha is dead config — sha wins"


def test_the_source_is_one_an_adopter_without_ssh_keys_can_clone() -> None:
    """`{"source": "github"}` clones over SSH with no HTTPS fallback, so an
    adopter without GitHub SSH keys gets `Permission denied (publickey)` from
    `install` — after `marketplace add` has already succeeded, because THAT
    fetch does fall back to HTTPS. Measured on 2.1.x from a scratch profile
    with no credentials.

    The `url` type takes the clone URL verbatim, so an `https://` one is
    anonymous. It is the same repository and the same pinned sha; only the
    transport changes.

    An `ssh://` or `git@` url here would reintroduce the failure while
    satisfying every other assertion in this file, which is why the scheme is
    asserted rather than the host.
    """
    source = _json(MARKETPLACE)["plugins"][0]["source"]
    assert source["source"] == "url", (
        "the `github` source type is SSH-first and has no HTTPS fallback", source
    )
    url = source["url"]
    assert url.startswith("https://"), (
        "an adopter with no SSH keys cannot clone this", url
    )
    assert "@" not in url.split("//", 1)[1].split("/", 1)[0], (
        "userinfo in the host means SSH or a credential nobody has", url
    )
    # Still this repository, and still the pin the rest of the file reasons
    # about — the transport is the only thing that changed.
    assert url.rstrip("/").endswith("/ak2k/memkit.git"), url
    # And the README says so, because "do I need a GitHub account for this"
    # is the question the old form answered wrongly and silently. The claim is
    # only true while the source type above holds it up.
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "anonymously over HTTPS" in readme, "the README does not say it"


# The paragraph that tells an adopter the marketplace route is not live yet.
# It is load-bearing rather than decorative: at a pin whose commit carries no
# payload the install SUCCEEDS and registers nothing, so nothing an adopter is
# told to run distinguishes it from a correct install waiting for a config.
#
# The BOLD MARKERS are part of the match, and they are what keeps this from
# firing on the release-wrinkle paragraph, which carries the same words
# unbolded to describe what the INSTALLED copy still says — so emphasising that
# sentence would fail this case with a message about the marketplace pin.
NOT_YET_INSTALLABLE = "**Not yet installable from this marketplace.**"
# The other half of the same convention, for the state a repo that HAS shipped
# is normally in: the pin serves a working plugin and `main` has grown payload
# since. The README's own Status section defines this marker and uses it.
FROM_THE_NEXT_RELEASE = "*(from the next release)*"
# What has to be at the pinned sha for the marketplace to serve an install at
# all. A payload file `main` added afterwards makes the pin INCOMPLETE, which
# is a different claim from the pin carrying no plugin.
INSTALLABLE_CORE = (
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
    "hooks/hooks.json",
    "bin/memkit-hook",
    "bin/lib/common.sh",
    "src/memkit/memory_prompt_recall.py",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60
    )


def test_the_pinned_sha_is_a_commit_in_this_history() -> None:
    _needs_checkout()
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    assert _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, (
        f"{sha} is not a commit in this repository"
    )
    # Existence is not enough: a sha from an unrelated branch, or from a fork,
    # satisfies `cat-file` and is not what an adopter would be served.
    assert _git("merge-base", "--is-ancestor", sha, "HEAD").returncode == 0, (
        f"{sha} is not an ancestor of HEAD"
    )


def test_the_readme_and_the_pinned_payload_say_the_same_thing() -> None:
    """Both directions, because both have happened in one repo or another: a
    pin moved without the README edit, and a README edit without the pin.

    Read out of the object store rather than by checking anything out —
    `git cat-file -e <sha>:<path>` per payload entry is the same question the
    harness's clone answers, and it costs no worktree.

    Green today by naming the state we are actually in, and the release commit
    that moves the pin is the one whose own test forces the paragraph out.

    THREE states, not two, because a repo that has shipped once is normally in
    the third and the two-state version reads it as the first. A pin carrying
    no plugin at all is "not yet installable"; a pin carrying a working plugin
    plus a `main` that has grown payload since is the ordinary between-releases
    state, and the README marks that one *(from the next release)* rather than
    telling an adopter the marketplace cannot serve them.
    """
    _needs_checkout()
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    at_sha = {
        path: _git("cat-file", "-e", f"{sha}:{path}").returncode == 0
        for path in PAYLOAD
    }
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    missing = sorted(path for path, ok in at_sha.items() if not ok)
    if not missing:
        assert NOT_YET_INSTALLABLE not in readme, (
            "the pin now carries the whole payload — the README still says it "
            "does not"
        )
    elif all(at_sha[path] for path in INSTALLABLE_CORE):
        assert NOT_YET_INSTALLABLE not in readme, (
            f"the pin serves a working plugin; {missing} is main moving ahead "
            "of it, which is not the same claim"
        )
        assert FROM_THE_NEXT_RELEASE in readme, (
            f"main carries payload the pin does not ({missing}) and the README "
            "no longer marks the divergence"
        )
    else:
        assert NOT_YET_INSTALLABLE in readme, (
            "the pin carries no plugin payload and the README no longer says "
            f"so: {missing}"
        )


def test_the_docs_quote_the_release_this_payload_is() -> None:
    """The no-install recipes an adopter TYPES, held to the version this tree
    ships.

    Unpinned, `uvx --from git+…/memkit` resolves whatever `main` holds, so
    somebody checking their store survives an uninstall gets a different build
    than the one they were running. The rev used to be pinned against a shell
    constant, which was one of four copies of a spec that had already drifted
    once. There is no constant now — nothing in `bin/` or `src/` names a rev,
    because nothing memkit runs is fetched — so the manifest's own version is
    what the docs are held to, and it is already the thing a release bumps.

    NOT a route this package takes: `--from` names the source and the trailing
    word names a console script inside it, so no name is resolved from any
    public index by either line.
    """
    manifest = json.loads(
        (REPO / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    rev = "v" + manifest["version"]
    quoted = {}
    for rel in ("README.md", "docs/STORE.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        quoted[rel] = set(
            re.findall(r"git\+https://github\.com/ak2k/memkit@([\w.]+)", text)
        )
    assert quoted["README.md"], "the README does not quote the spec at all"
    for rel, revs in quoted.items():
        assert revs <= {rev}, (rel, sorted(revs), rev)
    # And no `uvx --from` call site is UNPINNED, which is how the uninstall
    # recipe shipped for two releases: somebody checking their store survives an
    # uninstall got a different build than the one they were running. Scoped to
    # `uvx`, because the pip channel's `pip install git+…/memkit` is deliberately
    # unpinned — that channel installs the CLIs from main and says so.
    for rel in ("README.md", "docs/STORE.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert not re.search(
            r"uvx --from git\+https://github\.com/ak2k/memkit(?![@\w.])", text
        ), rel


def test_every_relative_link_in_the_readme_resolves() -> None:
    """A README that points at a file nobody shipped is worse than one that
    points nowhere: it reads as an assurance that the detail exists somewhere.

    `docs/ADMISSION.md` is the reason this is here — the plugin section sends
    an adopter to it before they install a hook that runs on every prompt.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    targets = [
        target
        for target in re.findall(r"\]\(([^)\s]+)\)", readme)
        if not target.startswith(("http://", "https://", "#", "mailto:"))
    ]
    missing = [t for t in targets if not (REPO / t.split("#", 1)[0]).exists()]
    assert not missing, missing
    # Non-vacuity: there ARE relative links, and the one this exists for is
    # among them.
    assert "docs/ADMISSION.md" in targets, targets

    # And every in-page link resolves to a heading that exists. The README now
    # carries a table of contents and cross-links between sections, and a
    # heading rename breaks those silently — GitHub renders a dead anchor as a
    # jump to the top of the page, which reads as the link working.
    anchors = {
        re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
        # `####` too: GitHub anchors every heading level, and the nav block
        # links to one.
        for heading in re.findall(r"^#{2,6} (.+)$", readme, re.M)
    }
    inpage = re.findall(r"\]\(#([^)]+)\)", readme)
    assert inpage, "no in-page links at all"
    assert not [a for a in inpage if a not in anchors], (
        sorted({a for a in inpage if a not in anchors}), sorted(anchors)
    )


def test_the_admission_note_answers_what_it_claims_to() -> None:
    """What an adopter receives and where the trust boundary sits — the two
    questions the plugin section defers to it.

    Pinned by SUBJECT rather than by prose, so the note can be rewritten
    without this becoming a spelling test: what may not happen is the note
    quietly losing the half about admission while keeping the inventory.
    """
    note = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    # IN THE SECTION THAT OWES THE ANSWER, not anywhere in the file. A subject
    # pinned against the whole document is defeated by a second mention of the
    # word somewhere else — which is exactly what happened: a sentence about
    # `memkitConfig` added two sections earlier let the trust-boundary half
    # drop its own.
    start = note.index("## Where the trust boundary sits")
    boundary = note[start : note.index("\n## ", start + 10)]
    for subject in ("memkitConfig", "$CLAUDE_PLUGIN_DATA", "MEMKIT_CONFIG",
                    "interpreter"):
        assert subject in boundary, subject
    # The COUNT, which is the claim rather than the wording: rung 3 was deleted
    # because a file in the payload tree is a file the repository can ship, and
    # a note saying the config is admitted from "a number of places" is a note
    # that has stopped making the promise this section exists to make.
    assert "exactly two places" in boundary, boundary[:400]
    # And the inventory half keeps the one that is its own.
    # And the inventory half keeps the ones that are its own. `PreToolUse` is
    # one of them: it is a row in the `## What runs, and when` table, not a
    # statement about the trust boundary.
    inventory = note[: note.index("## Where the trust")]
    for subject in ("UserPromptSubmit", "PreToolUse"):
        assert subject in inventory, subject
    # The count it states is the count of THIS TREE, not of the currently
    # pinned sha, and the difference is the whole of how a release works here.
    #
    # A release is two pull requests (docs/RELEASING.md): the first carries the
    # release state, the second moves the pin to the first one's squash commit.
    # So while the release-state PR is open, the pin still names the PREVIOUS
    # release — and this file is describing the tree it is sitting in, which is
    # the tree the next pin will name and the one an adopter will install.
    # Anchoring on the pin instead would demand the previous release's numbers
    # in the commit whose job is to update them.
    #
    # It is also the command the note's own "Reproducing these numbers" block
    # tells a reader to run, which is where the stale figure was found: the doc
    # said 57 while `git ls-tree -r HEAD` returned 62.
    # `ls-files` rather than `ls-tree HEAD`: the two agree in a clean checkout —
    # which is what an adopter's installed copy is, and what the recipe runs
    # there — and only `ls-files` sees a file the release commit has staged but
    # not yet committed, which is the state this case runs in while the release
    # is being written.
    _needs_checkout()
    listed = _git("ls-files")
    assert listed.returncode == 0, listed.stderr
    count = len(listed.stdout.split())
    assert f"**{count} files" in note, (count, "not the number the note states")
    # ONCE PER TREE. A second copy of the SAME number is a second thing to keep
    # in step, and the paragraph below the table says the two agree "by
    # construction" — a claim a hard-coded repeat makes false the first time
    # either moves. Two trees are described here, so two counts are expected
    # and a third, or either one twice, is the drift this convicts.
    pinned = _pinned_file_count()
    stated = re.findall(r"\b(\d+) files\b", note)
    if pinned is None:
        # The pin cannot be read here, so the second count cannot be checked
        # against anything — but its EXISTENCE still can. Two trees are
        # described, so: this tree's count exactly once, and at most two
        # counts in the note. A third is the drift this rule is for.
        assert stated.count(str(count)) == 1, stated
        assert len(stated) <= 2, stated
    else:
        expected = [str(count)] if pinned == count else [str(count), str(pinned)]
        assert stated == expected, (stated, expected)
    # Stated ONCE. Both stale figures the final review found were second copies
    # of the total — one in the `.git` row ("on top of the 57"), one closing the
    # reproduce recipe ("returns the same 57") — sitting far from the headline
    # somebody had updated. A number that appears twice is a number that will
    # disagree with itself; the other places that need it now refer to it
    # instead of repeating it.
    assert note.count(f"{count} files") == 1, (
        count, [ln for ln in note.splitlines() if f"{count} files" in ln]
    )


def test_the_admission_notes_breakdown_sums_to_the_total_it_states() -> None:
    """The rows are the argument. This page's opening line says the case
    "rests on the exact ones", and a reader who does what it asks — count the
    categories, check the arithmetic — got 89 against a stated 90, because a
    commit updated the headline and left the `tests/` row two lines below it.

    Every row is read off the tree here rather than compared to a literal, so
    the drift that matters is the row disagreeing with what an install puts on
    the machine, not with a number somebody typed twice.
    """
    _needs_checkout()
    note = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    tracked = _git("ls-files").stdout.split()

    def under(*prefixes: str) -> int:
        return sum(1 for f in tracked if f.startswith(prefixes))

    payload = under("bin/", "src/memkit/", "hooks/", ".claude-plugin/", "skills/")
    tests = under("tests/")
    prose = under("docs/") + sum(
        1 for f in tracked if f in ("README.md", "LICENSE", "NOTICE")
    )
    rest = len(tracked) - payload - tests - prose
    rows = {
        "`bin/`, `src/memkit/`, `hooks/`, `.claude-plugin/`, `skills/`": payload,
        "`tests/`": tests,
        "`.github/`, `nix/`, `tools/`, `flake.*`, `pyproject.toml`, config files":
            rest,
        "`README.md`, `LICENSE`, `NOTICE`, and all of `docs/`": prose,
    }
    for label, count in rows.items():
        assert f"| {label} | {count} |" in note, (label, count, "row is stale")
    # And the rows are the total, which is the arithmetic the page asks for.
    assert sum(rows.values()) == len(tracked), rows
    assert f"**{len(tracked)} files" in note, len(tracked)


def test_the_admission_notes_recipe_returns_the_number_it_states() -> None:
    """This page's whole claim on a reader is checkability: "every number here
    is read out of the tree" plus a command to run.

    The command resolved `marketplace.json` while the number described this
    tree, so an adopter doing exactly what the trust document asks — in order
    to decide whether to trust it — got a different number back from the
    document's own recipe. The two describe two different trees, and both are
    worth stating; what they cannot do is share one sentence.

    Self-retiring, like the release markers: the moment the pin names this
    tree, the two counts agree and the marked sentence about the pinned tree
    is asserted to be gone.
    """
    _needs_checkout()
    note = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    here = len(_git("ls-files").stdout.split())
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    pinned = len(_git("ls-tree", "-r", "--name-only", sha).stdout.split())

    def mib(ref: str) -> str:
        rows = _git("ls-tree", "-r", "-l", ref).stdout.splitlines()
        total = sum(int(row.split()[3]) for row in rows)
        return f"{total / 1048576:.1f} MiB"

    # The size is a number of the same kind and had drifted the same way: the
    # tree was 1.67 MiB while the page said 1.5, and nothing looked.
    assert mib("HEAD") in note, (mib("HEAD"), "not the size the note states")

    # The recipe that reproduces the headline runs against this tree, which is
    # what the headline is about.
    recipe = note.split("## Reproducing these numbers", 1)[1]
    reproduces_here = recipe.split("```", 2)[1]
    assert "ls-files" in reproduces_here or "HEAD" in reproduces_here, reproduces_here
    assert "marketplace.json" not in reproduces_here, reproduces_here

    assert mib(sha) in note, (mib(sha), "not the size the note states for the pin")
    if here == pinned:
        # The pin has caught up: one tree, one number, and the paragraph that
        # existed to explain the gap has to go with it.
        assert "still names" not in note, note
        return
    # While it has not: both numbers stated, the pinned one marked as what an
    # install gives you today, and the recipe for it kept.
    assert f"**{here} files" in note, here
    assert f"{pinned} files" in note, pinned
    assert "from the next release" in note, "the gap between the two trees is unmarked"
    assert "marketplace.json" in recipe, "no recipe for the tree an install gets"




def test_the_manifest_and_the_marketplace_entry_agree_on_the_version() -> None:
    """`claude plugin tag` refuses to tag a release when they disagree, which
    is late: by then the version in the entry is what adopters resolve."""
    entry = _json(MARKETPLACE)["plugins"][0]
    manifest = _json(PLUGIN_MANIFEST)
    assert entry["name"] == manifest["name"] == "memkit"
    assert entry["version"] == manifest["version"]


def test_the_manifests_carry_the_metadata_an_adopter_is_shown() -> None:
    """Name, description, version, author — what `/plugin` renders in a list
    and what a directory submission is judged on. Absent metadata installs
    perfectly and reads as an unattributed blob.

    `--strict` demands these too, but only through the invocation that points
    at the plugin manifest itself: pointed at the repo root the validator
    checks the MARKETPLACE, and raises schema errors from the plugin manifests
    it lists while passing over their metadata warnings. Measured on 2.1.238 by
    deleting `author` and watching the single-invocation CI step stay green.
    CI now runs both; this fails in the suite rather than three minutes into a
    workflow.
    """
    manifest = _json(PLUGIN_MANIFEST)
    for field in ("name", "description", "version", "author"):
        assert manifest.get(field), field
    assert _json(MARKETPLACE).get("description")
    assert isinstance(_json(MARKETPLACE)["owner"], dict), "owner must be an object"


def test_ci_validates_both_manifests_and_not_only_the_marketplace() -> None:
    """The step is what makes the assertion above a release gate rather than a
    unit test, and it was checking half of what its name claimed."""
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text()
    assert "claude plugin validate . --strict" in workflow
    assert "claude plugin validate .claude-plugin/plugin.json --strict" in workflow


def test_ci_runs_the_rigs_harness_tier_as_a_gate_rather_than_a_courtesy() -> None:
    """The one required context in which the HARNESS produces the config
    option, rather than a test setting it.

    Two halves, and the second is what makes it a gate. The `python` job has to
    install the pinned binary before the suite — which it does, for the CLI
    tier — and it has to declare the tier required, or a job whose install step
    quietly stopped working reports green while every scenario that watches the
    harness skips. A skipped scenario and a passing one look identical in a
    check name.
    """
    from rig import REQUIRED_ENV

    # The `python:` job's own text, not the whole file: a whole-file grep
    # cannot tell which job carries a line, cannot tell a live line from a
    # commented one, and would pass with the declaration moved to the `nix`
    # job — where `tests/rig` is not even collected.
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text()
    jobs = re.split(r"^  (?=\w[\w-]*:$)", workflow, flags=re.MULTILINE)
    python_job = [j for j in jobs if j.startswith("python:")]
    assert len(python_job) == 1, [j.split(":", 1)[0] for j in jobs]
    body = python_job[0]
    live = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    assert f'{REQUIRED_ENV}: "1"' in live, REQUIRED_ENV
    assert "npm install -g @anthropic-ai/claude-code@" in live
    assert "-m pytest" in live
    # And nowhere else, or the declaration can drift into a job that never
    # runs the rig while this stays green.
    elsewhere = workflow.replace(body, "")
    assert REQUIRED_ENV not in elsewhere, "declared outside the job that runs it"


# --- the workflows as a set --------------------------------------------------
#
# Three files now install the same pinned Claude Code, and the pin is the
# repository's statement about which harness build its measured claims were
# measured against. Nothing in yaml connects three copies of a version string,
# so the connection is here.

WORKFLOWS = REPO / ".github" / "workflows"
# `renovate.json`'s custom manager matches this exact shape: a marker comment
# and then the assignment on the next line. A pin without the marker is a pin
# renovate cannot see.
PINNED_HARNESS = re.compile(
    r"#\s*renovate:\s*datasource=npm\s+depName=@anthropic-ai/claude-code\s*\n"
    r'\s*CLAUDE_CODE_VERSION:\s*"([^"]+)"'
)
ANY_HARNESS_PIN = re.compile(r'^\s*CLAUDE_CODE_VERSION:\s*"([^"]+)"', re.MULTILINE)


def test_every_workflow_pins_the_same_claude_code_build() -> None:
    """One harness version across every workflow, and every pin visible to
    renovate.

    Both halves are the gate, and the second is what makes the first
    maintainable. `live.yml` carried an unmarked copy of the pin, so renovate's
    custom manager never saw it: a bump moved `check.yml` alone and the live
    tier went on measuring a build nothing else in the repository used. An
    equality assertion without the markers would just turn that drift into a
    red build somebody has to fix by hand every time.

    The claims this pin carries — the option-name mangling, the trailing slash
    on the plugin root, exit 2 blocking a turn — were each measured against one
    build. Two workflows on two builds is two different sets of claims wearing
    one repository's name.
    """
    marked = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        pins = ANY_HARNESS_PIN.findall(text)
        if not pins:
            continue
        assert len(pins) == 1, (path.name, pins)
        assert PINNED_HARNESS.findall(text) == pins, (
            f"{path.name} pins CLAUDE_CODE_VERSION with no renovate marker "
            "directly above it, so renovate cannot move it and it will drift",
            pins,
        )
        marked[path.name] = pins[0]

    # Not a vacuous pass: the rig runs in more than one workflow, and a version
    # of this test that found one pin would agree with itself forever.
    assert len(marked) >= 2, marked
    assert len(set(marked.values())) == 1, marked


def _job_names(workflow: str) -> set[str]:
    """Every context a workflow reports, as branch protection sees them.

    The `name:` a job declares wins, and a job without one reports its key.
    Deliberately text rather than yaml: nothing in this repository's test
    dependencies parses yaml, and the shape being read is two levels deep and
    regular.
    """
    body = workflow.split("\njobs:\n", 1)[1]
    names = set()
    for block in re.split(r"^  (?=\w[\w-]*:$)", body, flags=re.MULTILINE)[1:]:
        key = block.split(":", 1)[0]
        declared = re.search(r"^    name:\s*(.+)$", block, re.MULTILINE)
        names.add(declared.group(1).strip() if declared else key)
    return names


def test_the_remote_install_workflow_touches_no_required_context() -> None:
    """A scheduled workflow must not report a name branch protection waits for.

    Sharing one would make merges gate on a nightly clone from github — the
    check would go red on github's bad afternoon and block every PR, and a
    stale success would satisfy a context nobody had re-run. `automerge.yml`
    is where the required list actually lives, so it is read rather than
    restated here.
    """
    automerge = (WORKFLOWS / "automerge.yml").read_text(encoding="utf-8")
    listed = re.search(r"const requiredChecks = \[(.*?)\];", automerge, re.S)
    assert listed, "automerge.yml no longer declares requiredChecks"
    required = set(re.findall(r'"([^"]+)"', listed.group(1)))
    assert "python" in required, required  # the list was found, not an empty match

    for name in ("remote-install.yml", "live.yml"):
        jobs = _job_names((WORKFLOWS / name).read_text(encoding="utf-8"))
        assert jobs, name
        assert not (jobs & required), (name, sorted(jobs & required))


def test_the_remote_install_workflow_runs_the_tier_on_a_schedule_and_no_secret(
) -> None:
    """The four properties that make this workflow the gate it claims to be.

    Each one fails silently if it regresses: a commented-out schedule reports
    nothing, a missing opt-in skips every scenario, a missing required
    declaration turns a broken install step into a green run, and a secret in
    scope quietly retires the no-credential claim the whole tier is about.
    """
    from rig import REMOTE_ENV, REQUIRED_ENV

    text = (WORKFLOWS / "remote-install.yml").read_text(encoding="utf-8")
    live = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )

    # A schedule that is present but commented out is exactly what `live.yml`
    # ships, and reading the file with comments left in cannot tell them apart.
    assert re.search(r"^  schedule:\n    - cron:", live, re.MULTILINE), live
    assert "workflow_dispatch:" in live
    # The manifest is what it guards, and the path filter is what makes a
    # change to it report the same day rather than the next.
    assert '- ".claude-plugin/**"' in live, live

    assert f'{REMOTE_ENV}: "1"' in live, REMOTE_ENV
    assert f'{REQUIRED_ENV}: "1"' in live, REQUIRED_ENV
    assert "tests/rig/test_remote_install.py" in live

    # The claim under test is that a machine with NO credential can install
    # this plugin. A `secrets.` reference is how that claim stops being true
    # without anything failing.
    assert "secrets." not in live, "the no-credential tier reached for a secret"


def test_every_payload_file_is_tracked() -> None:
    """A github install is a clone. An untracked wrapper works perfectly on the
    machine it was written on and is missing for every adopter — and the
    failure it produces there is the wrapper's own "payload is incomplete"
    refusal, i.e. a plugin that installs and never speaks again.
    """
    _needs_checkout()
    out = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *PAYLOAD],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr


def test_a_git_gated_case_fails_rather_than_skips_where_a_checkout_is_expected(
    monkeypatch, tmp_path
) -> None:
    """The skip is the thing being guarded, not the tool.

    Three of these cases skipped whenever `.git` was absent, so the packaged
    leg and a broken plain-python job produced the same green under the same
    check name. Only a context that DECLARES itself packaged may skip.
    """
    monkeypatch.delenv("MEMKIT_NO_CHECKOUT", raising=False)
    monkeypatch.setattr("test_plugin_surface.REPO", tmp_path, raising=False)
    # A skip here would SKIP THIS CASE rather than fail it, which is the whole
    # defect wearing the test's own clothes — so it is caught by name.
    try:
        _needs_checkout()
    except pytest.skip.Exception as skipped:
        raise AssertionError(f"skipped where it must fail: {skipped}") from None
    except AssertionError as failed:
        assert "did not declare itself packaged" in str(failed), failed
    else:
        raise AssertionError("a missing checkout was accepted")

    # `Skipped` derives from BaseException, so it has to be named: catching
    # `Exception` here let the assertion skip the case it is asserting about.
    monkeypatch.setenv("MEMKIT_NO_CHECKOUT", "1")
    with pytest.raises(pytest.skip.Exception, match="packaged build"):
        _needs_checkout()


def test_the_packaged_leg_is_the_only_context_that_declares_itself_packaged() -> None:
    """The marker's one producer, pinned where the skip is read.

    A test that skips on an environment variable is a test anybody can turn
    off; what makes it honest is that exactly one build sets it, and that build
    is the one whose source really has no checkout.
    """
    flake = (REPO / "flake.nix").read_text(encoding="utf-8")
    assert flake.count('MEMKIT_NO_CHECKOUT = "1"') == 1, flake.count(
        'MEMKIT_NO_CHECKOUT = "1"'
    )
    workflow = (REPO / ".github" / "workflows" / "check.yml").read_text()
    assert "MEMKIT_NO_CHECKOUT" not in workflow


def test_the_payload_carries_every_file_its_own_entry_points_import() -> None:
    """PAYLOAD is hand-kept, and the failure it produces is the wrapper's own
    "the plugin payload is incomplete" refusal — a plugin that installs and
    never speaks again.

    A SUBSET assertion, not equality: the closure of the two 3.9 entry points
    does not reach `memory_integrity.py`, which the dispatcher routes to only
    when a 3.12 interpreter is available, so equality would make that entry
    red. What this catches is the direction that matters — a module a shipped
    entry point imports that nobody remembered to list.
    """
    import sys

    sys.path.insert(0, str(REPO / "tests"))
    from test_packaging import _floor_39_closure

    reachable = {
        str(path.relative_to(REPO)) for path in _floor_39_closure()
    }
    listed = set(PAYLOAD)
    assert reachable <= listed, sorted(reachable - listed)


def test_the_payload_root_carries_no_config_of_its_own() -> None:
    """The other direction of the tracking assertion above, and the one the
    wrappers' admission rule rests on.

    That test asks whether each payload entry is tracked; nothing asked what
    ELSE the payload carries. A `memkit.json` at the root — committed, or
    merely sitting in the checkout of whoever runs the rig — used to be a
    config rung, and a config decides both which directories the every-prompt
    hook reads and which binary it exec's.

    Against the REAL repository root rather than the `root` fixture, which is
    built from PAYLOAD and can never contain one. Both states are checked: the
    index, because a clone delivers that, and the working tree, because the rig
    stages from it.
    """
    _needs_checkout()
    out = subprocess.run(
        ["git", "ls-files", "--", ":(top)memkit.json"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert out.stdout.strip() == "", out.stdout
    assert not (REPO / "memkit.json").exists(), "a config at the payload root"


def test_the_wrappers_are_executable_in_the_index() -> None:
    """Mode 100755 in git, not merely on this filesystem. A clone restores the
    executable bit from the index, and a wrapper checked in as 644 is a hook
    the harness cannot run at all."""
    _needs_checkout()
    out = subprocess.run(
        ["git", "ls-files", "-s", "bin/"], cwd=REPO,
        capture_output=True, text=True, timeout=60,
    )
    modes = {
        line.split("\t")[1]: line.split()[0] for line in out.stdout.splitlines()
    }
    for wrapper in ("bin/memkit", "bin/memkit-hook", "bin/memkit-recall"):
        assert modes[wrapper] == "100755", (wrapper, modes[wrapper])
    # And the sourced library is NOT executable: it sits in bin/lib precisely
    # so that nothing on the agent's PATH can invoke it.
    assert modes["bin/lib/common.sh"] == "100644", modes["bin/lib/common.sh"]


# --- the registration ---------------------------------------------------------


def _entries() -> list[tuple[str, dict]]:
    """(event, handler) for every hook this plugin registers."""
    out = []
    for event, groups in _json(HOOKS_JSON)["hooks"].items():
        for group in groups:
            for handler in group["hooks"]:
                out.append((event, handler))
    return out


def test_the_registration_passes_zero_arguments_on_every_entry() -> None:
    """The hook file is dual-mode: no arguments reads a payload off stdin, ANY
    argument is the search CLI — where argparse answers an unrecognised flag
    with exit 2. On UserPromptSubmit, exit 2 does not merely fail: the turn is
    blocked and the user gets their prompt handed back unanswered (measured on
    2.1.238). So a stray argument in this file costs every prompt in every
    session, on every machine that installed the plugin.

    Enumerating EVERY entry rather than the one that exists today is the point.
    A later unit adds PreToolUse to this same file, and a pin that names one
    event would go on passing while the new entry carried the defect.
    """
    assert _entries(), "hooks.json registers nothing — this pin would be vacuous"
    for event, handler in _entries():
        assert handler.get("args", []) == [], (event, handler)
        # DECLARED rather than omitted, because `args: []` is what selects the
        # exec form: without the key the command is a shell string, and the
        # hook file's dual mode turns any stray word into the search CLI.
        assert "args" in handler, (event, "declare args: [] rather than omitting it")
        assert handler["type"] == "command"
        # And no space in the command itself — under a shell that would be two
        # arguments, which is the same defect arriving through the path.
        assert " " not in handler["command"], handler["command"]


def test_every_registered_timeout_matches_the_constant_it_is_paired_with() -> None:
    """`timeout` is the harness's kill, and the hook's own budget is set
    beneath it so that an overrun leaves a record instead of being killed
    mid-write. Nothing connects the number in this JSON to the constant in the
    module; before the plugin existed, the consumer's own suite carried this
    assertion because the registration lived in its settings file.

    Scoped per event, because a later unit's PreToolUse entry gets its OWN
    constant pair — sharing the module's single budget would put an internal
    deadline above the harness's kill point.
    """
    expected = {
        "UserPromptSubmit": (hook.HARNESS_TIMEOUT, hook.BUDGET_SECONDS),
        "PreToolUse": (hook.TASK_HARNESS_TIMEOUT, hook.TASK_BUDGET_SECONDS),
    }
    for event, handler in _entries():
        assert event in expected, f"{event} has no declared constant pair"
        assert handler["timeout"] == expected[event][0], (event, handler["timeout"])
    # Both halves of the relation, per event, so neither file can be edited into
    # agreement with a budget that no longer sits beneath it — and so that a
    # second event cannot be registered against the first event's budget, which
    # is the failure the pair exists to prevent: an internal deadline above the
    # harness's kill point never fires, and a killed hook writes no record.
    for event, (timeout, budget) in expected.items():
        assert budget < timeout, (event, budget, timeout)
    assert len({budget for _, budget in expected.values()}) == len(expected), (
        "two events sharing one budget constant is the sharing this pins against"
    )


def test_the_registration_runs_the_wrapper_and_not_the_hook_directly() -> None:
    """A registration naming the hook file would work — and would inherit
    whatever MEMKIT_CONFIG the launching context carried, which is ambient
    configuration arriving by the back door."""
    for _event, handler in _entries():
        assert handler["command"] == "${CLAUDE_PLUGIN_ROOT}/bin/memkit-hook", handler


def test_the_checker_floor_lives_in_one_file_now_that_the_shell_holds_none(
) -> None:
    """The floor used to live in two files, because a POSIX-sh probe cannot
    import python — so the number was written twice and a test held the copies
    equal.

    The shell no longer probes anything: which interpreter runs the checker is
    python's own question, answered where the floor already is. So the second
    copy is gone rather than kept in agreement, and what is pinned now is that
    it did not come back.
    """
    guard = (REPO / "src" / "memkit" / "memory_integrity.py").read_text()
    match = re.search(r"sys\.version_info < \((\d+), (\d+)\)", guard)
    assert match, "the checker's version guard moved — this pin cannot see it"
    floor = (int(match.group(1)), int(match.group(2)))
    assert floor == doctor.CHECKER_FLOOR, (doctor.CHECKER_FLOOR, floor)
    shell = COMMON_SH.read_text(encoding="utf-8")
    for gone in ("MEMKIT_CHECKER_FLOOR", "MEMKIT_UVX_SPEC", "MEMKIT_CHECKER_CMD",
                 "MEMKIT_CHECKER_ROUTE", "memkit_resolve_checker",
                 "memkit_trusted_path", "uvx"):
        assert gone not in shell, gone


# --- the wrappers, as processes ----------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A plugin root whose files are the repo's, reached through symlinks.

    Symlinked FILES rather than directories, deliberately. The wrappers resolve
    their own tree with `cd "$(dirname $0)/.." && pwd -P`, and `pwd -P` walks
    through a symlinked *directory* component — so a `bin` symlink would send
    every wrapper back to the real repo and quietly test the wrong tree.

    THE PINNED INTERPRETER LIST IS THE CASE'S, not the machine's. The shipped
    list names five absolute system paths, and whether any of them exists is a
    fact about the host: a mac has `/usr/bin/python3`, a Linux nix build
    sandbox has none of the five, and the whole fallback rung therefore
    refused there while every case that reaches it passed here. Repinned to
    the python running this suite, an assertion that the fallback answered
    with a pinned path says the RUNG was consulted rather than that the runner
    happened to be a mac. Cases that need another pinned state — none at all,
    or a directory ahead of a file — repin again over this.

    The SHIPPED list is not left unread: `_pinned_pythons()` with no argument
    is it, and `test_the_shipped_pinned_list_is_absolute_paths_only` holds its
    shape.
    """
    plugin = tmp_path / "plugin"
    for rel in PAYLOAD:
        dest = plugin / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(REPO / rel)
    return _repinned(plugin, [sys.executable])


def _repinned(root: Path, pythons: list) -> Path:
    """`root` with the wrapper's pinned interpreter list replaced.

    A REAL EDIT to a copy of `bin/lib/common.sh`, not an environment override,
    because the list is deliberately not readable from the environment: it is
    assigned unconditionally when the library is sourced, so an exported value
    of the same name is overwritten before anything reads it. That is the
    property the whole change rests on, and a test knob that punched through it
    would be the ambient channel back under another name.

    What this buys is the two states a machine with `/usr/bin/python3` on it
    cannot otherwise be put into: no interpreter anywhere, and an interpreter
    that is the case's own.
    """
    real = (root / "bin" / "lib" / "common.sh").resolve()
    text = real.read_text(encoding="utf-8")
    body = "\n".join(str(p) for p in pythons)
    swapped = re.sub(
        r'MEMKIT_SYSTEM_PYTHONS="[^"]+"',
        f'MEMKIT_SYSTEM_PYTHONS="{body}"',
        text,
        count=1,
    )
    assert swapped != text, "the pinned list was not found to replace"
    copy = root / "bin" / "lib" / "common.sh"
    copy.unlink()
    copy.write_text(swapped, encoding="utf-8")
    return root


def _shim(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


# Records what the wrapper handed it, instead of being an interpreter. `-` as a
# default marks unset, which is a different fact from empty: the hook wrapper
# must UNSET MEMKIT_CONFIG when no rung answers, not blank it.
SHIM_BODY = """
{
  echo "argv=$*"
  echo "MEMKIT_CONFIG=${MEMKIT_CONFIG-<unset>}"
  echo "MEMKIT_PLUGIN=${MEMKIT_PLUGIN-<unset>}"
  echo "MEMKIT_CHECKER_ROUTE=${MEMKIT_CHECKER_ROUTE-<unset>}"
  echo "MEMKIT_CHECKER_CMD=${MEMKIT_CHECKER_CMD-<unset>}"
  echo "PYTHONPATH=${PYTHONPATH-<unset>}"
} > "$SHIM_OUT"
# The same argv again, losslessly. `$*` joins on a space, so it cannot tell
# `--search "flange torque"` from `--search flange torque` — and neither can
# `sh -x`, on the shell that matters (see `Shim.argv`). NUL-separated because
# it is the one byte an argument cannot contain. The backslash is DOUBLED
# because this body is a python string before it is a script: a lone one is an
# octal escape, and what reached the shim was a real NUL where the format
# specifier should have been, so the loop wrote nothing at all.
#
# Guarded on the variable being SET, unlike the block above, and the
# difference is one this file paid for: `> "$SHIM_OUT"` with nothing in it
# fails and writes nothing, while `> "$SHIM_OUT.argv"` succeeds against a
# file called `.argv` in whatever directory the case put the shim's cwd —
# which was the repository. A shim reached by a case that sets no SHIM_OUT
# records nothing rather than leaving a file behind.
if [ -n "${SHIM_OUT:-}" ]; then
  : > "$SHIM_OUT.argv"
  for _arg do printf '%s\\0' "$_arg" >> "$SHIM_OUT.argv"; done
fi
exit 0
"""


def _run(
    wrapper: Path, *args: str, env: dict, cwd: Path | None = None, shell_trace=False
) -> subprocess.CompletedProcess:
    argv = [SH, "-x", str(wrapper)] if shell_trace else [str(wrapper)]
    return subprocess.run(
        [*argv, *args],
        capture_output=True, text=True, timeout=60, env=env,
        cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL,
    )


class Shim:
    """One fake `python3`, and a reader for what it saw.

    Reached the way a real install reaches its own interpreter — recorded in
    the config, which is what `memkit init` writes — because the wrappers no
    longer search the session's PATH for one. `_config_file` puts the record
    in for every case that has a shim beside it. A case with no config
    therefore cannot observe through this at all, and reads the wrapper's own
    decision off `sh -x` instead (`_decided_config`).

    Its directory is still the whole of PATH in every case, which keeps the
    other claim honest: the wrappers need no external command.

    A class rather than attributes bolted onto a returned function: the shim's
    directory and its output file are things a case reaches for, and hanging
    them off a callable cost three `type: ignore`s in a file whose pyright pass
    is now a gate.

    Callable, so the call sites that build an environment read as they did.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "shimbin"
        self.out = tmp_path / "shim-out.txt"
        self._home = tmp_path / "home"
        _shim(self.dir, "python3", SHIM_BODY)

    def __call__(self, **extra: str) -> dict[str, str]:
        env = {
            # The shim directory and NOTHING else. A PATH with `/usr/bin` on
            # it makes two claims false at once: that the wrappers need no
            # external command, and that the only `python3` a case can reach is
            # the one it wrote. Both went unnoticed until a runner whose
            # `/usr/bin/python3` meets the checker floor answered for a shim
            # written to refuse, and a sandbox with no coreutils failed on a
            # `head` nobody knew was there.
            "PATH": str(self.dir),
            "HOME": str(self._home),
            "SHIM_OUT": str(self.out),
        }
        env.update(extra)
        return env

    def read(self) -> dict[str, str]:
        return dict(
            line.split("=", 1)
            for line in self.out.read_text().splitlines()
            if "=" in line
        )

    def argv(self) -> list[str]:
        """The argv the shim was handed, with its word boundaries intact.

        Read from the CHILD rather than from `sh -x`, and the difference is
        not stylistic: the trace cannot carry word boundaries on every shell.
        bash quotes a traced word containing a space, dash does not, and dash
        is `/bin/sh` on the Linux runners — so `--search "flange torque"` and
        `--search flange torque` trace identically there, and a parser reading
        either one back reports three words. MEASURED both ways: the child's
        argv is the same two words under both shells.
        """
        raw = self.out.with_name(self.out.name + ".argv").read_bytes()
        return raw.decode().split("\0")[:-1]


@pytest.fixture
def shimmed(tmp_path: Path) -> Shim:
    return Shim(tmp_path)


def _pinned_pythons(root: Path | None = None) -> list[str]:
    """The wrapper's fallback list, read out of the wrapper rather than typed
    a second time here.

    With a `root`, the list of the tree the case actually ran — which the
    `root` fixture repins to this build's own python, so `in _pinned_pythons(
    root)` means the fallback rung answered rather than that the host owns one
    of the five shipped paths. With none, the shipped list an adopter gets.
    """
    common = COMMON_SH if root is None else root / "bin" / "lib" / "common.sh"
    text = common.read_text(encoding="utf-8")
    body = re.search(r'MEMKIT_SYSTEM_PYTHONS="([^"]+)"', text)
    assert body, "the pinned interpreter list is not assigned a literal"
    found = [line.strip() for line in body.group(1).splitlines() if line.strip()]
    assert all(p.startswith("/") for p in found), found
    return found


def test_the_shipped_pinned_list_is_absolute_paths_only() -> None:
    """What an adopter's install falls back to, read from the shipped file
    rather than from the copy the cases repin.

    Absolute on every entry is the property the whole rung rests on: a
    slashless word sends `exec` to the session's PATH and a relative one to
    the session's directory, which is the lookup this list exists to replace.
    """
    shipped = _pinned_pythons()
    assert len(shipped) >= 2, shipped
    # And the refusal advertises them, so an adopter can see whether their
    # python is simply somewhere this list does not reach.
    assert all(p in COMMON_SH.read_text(encoding="utf-8") for p in shipped)


class Decided:
    """What a wrapper DECIDED, read off `sh -x`.

    `config` is the value it settled `MEMKIT_CONFIG` to (or `<unset>`),
    `interpreter` the path it settled on, `handoff` the argv it exec'd.
    """

    __slots__ = ("config", "interpreter", "handoff", "returncode", "stderr")

    def __init__(self, out) -> None:
        self.returncode = out.returncode
        self.stderr = out.stderr
        self.config = "<unset>"
        self.interpreter = ""
        self.handoff: list = []
        for line in out.stderr.splitlines():
            stripped = line.lstrip("+ ")
            if stripped == "unset MEMKIT_CONFIG":
                self.config = "<unset>"
            elif stripped.startswith("MEMKIT_CONFIG="):
                self.config = stripped.split("=", 1)[1]
            elif stripped.startswith("PY="):
                self.interpreter = stripped[3:]
            elif stripped.startswith("exec "):
                # Split the way a shell would, which is right only as far as
                # the trace is quoted — and that is the shell's choice, not
                # this file's. bash quotes a traced word containing a space;
                # dash, which is `/bin/sh` on the Linux runners, prints it
                # bare. So `handoff` reads word boundaries a multi-word
                # argument does not survive, and a case that needs them exact
                # observes the child instead (`Shim.argv`). Every case reading
                # `handoff` therefore passes single-word arguments.
                self.handoff = shlex.split(stripped[5:])


def _decide(
    root: Path, wrapper: str, env: dict, *args, cwd=None, expect_rc: int | None = 0
) -> Decided:
    """Run one wrapper under `sh -x` and read the decisions it made.

    The shim cannot answer this question any more when NO config resolves. The
    wrappers stopped searching the session's PATH for an interpreter — a lookup
    a checkout steers, whose answer was exec'd on every prompt — and a config
    is what records one, so a case with no config has no way to put its own
    python in front of the wrapper. That is the point of the change, not a gap
    in it.

    So the decision is read where it is MADE. `sh -x` traces every command the
    shell evaluated, `export` and `unset` alike, which is a stronger reading
    than the shim's: it sees the wrapper's own act rather than the environment
    a child happened to receive.
    """
    out = _run(root / "bin" / wrapper, *args, env=env, shell_trace=True, cwd=cwd)
    if expect_rc is not None:
        assert out.returncode == expect_rc, out.stderr
    assert "MEMKIT_CONFIG" in out.stderr, (
        "the trace never mentions the variable, so this reads nothing"
    )
    return Decided(out)


def _decided_config(
    root: Path, wrapper: str, env: dict, *args, cwd=None, expect_rc: int | None = 0
) -> str:
    decided = _decide(
        root, wrapper, env, *args, cwd=cwd, expect_rc=expect_rc
    )
    # The wrapper reached the hand-off, so the decision is the one it acted on
    # rather than one it made before refusing for another reason.
    assert decided.handoff, decided.stderr[-600:]
    return decided.config


def _config_file(path: Path, **extra) -> Path:
    """A config the wrappers can act on, shaped like one `memkit init` wrote.

    That means it RECORDS AN INTERPRETER. The wrappers no longer resolve one
    over the session's PATH — a lookup a checkout steers, whose answer was
    exec'd on every prompt — so the shim fixture's python is reachable only
    the way a real install reaches its own: written into the config. Found by
    walking up from the config's own directory rather than passed at
    thirty-odd call sites, because every case that has a shim keeps it in the
    same place under `tmp_path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"schema": hook.SCHEMA, "roots": {}, "stores": []}
    if "interpreter" not in extra:
        for parent in path.parents:
            shim = parent / "shimbin" / "python3"
            if shim.is_file():
                blob["interpreter"] = str(shim)
                break
    blob.update(extra)
    path.write_text(json.dumps(blob))
    return path


def test_rung_one_is_the_manifest_option(root, tmp_path, shimmed) -> None:
    config = _config_file(tmp_path / "opt.json")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    seen = shimmed.read()
    assert seen["MEMKIT_CONFIG"] == str(config)
    assert seen["MEMKIT_PLUGIN"] == "1"
    # Zero arguments to the hook: anything else is the search CLI.
    assert seen["argv"] == str(root / "src" / "memkit" / "memory_prompt_recall.py")


def test_rung_one_expands_a_tilde_the_way_a_person_types_it(
    root, tmp_path, shimmed
) -> None:
    """The option value is a string typed into an install command, not a word
    any shell expanded. `~/.cache/...` — which is what the manifest's own
    default says — arrives with a literal tilde, and every rung would miss it.
    """
    home = tmp_path / "home"
    config = _config_file(home / ".cache" / "memory-recall" / "memkit.json")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG="~/.cache/memory-recall/memkit.json")
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_rung_two_is_the_plugin_data_dir(root, tmp_path, shimmed) -> None:
    data = tmp_path / "plugindata"
    config = _config_file(data / "memkit.json")
    env = shimmed(CLAUDE_PLUGIN_DATA=str(data))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_a_config_inside_the_payload_is_not_a_rung(root, tmp_path, shimmed) -> None:
    """The admission rule, from the side that has no environment to check.

    A plugin install is a clone of a pinned commit, so a `memkit.json` at the
    payload root is a file the repository can ship — and a config names the
    directories the every-prompt hook reads and the binary it exec's. Both
    halves are asserted here because they fail differently: the config half
    poisons what is served, and the interpreter half decides what runs at all,
    before anything has parsed a byte of JSON.
    """
    evil = tmp_path / "evil"
    evil.write_text("#!/bin/sh\ntouch " + str(tmp_path / "EVIL-RAN") + "\n")
    evil.chmod(0o755)
    _config_file(root / "memkit.json", interpreter=str(evil))
    env = shimmed()
    assert "CLAUDE_PLUGIN_ROOT" not in env and "CLAUDE_PLUGIN_DATA" not in env
    assert _decided_config(root, "memkit-hook", env) == "<unset>"
    # And the interpreter half: the payload's file named one and it did not
    # run, so the file was not read as a config at all.
    assert not (tmp_path / "EVIL-RAN").exists()


def test_the_rungs_are_tried_in_order(root, tmp_path, shimmed) -> None:
    option = _config_file(tmp_path / "one.json")
    data = tmp_path / "two"
    _config_file(data / "memkit.json")
    env = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(option), CLAUDE_PLUGIN_DATA=str(data)
    )
    _run(root / "bin" / "memkit-hook", env=env)
    assert shimmed.read()["MEMKIT_CONFIG"] == str(option)

    # Rung 1 naming a file that is not there yet is the NORMAL state between
    # install and init, and it must fall through rather than stopping.
    env = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(tmp_path / "absent.json"),
        CLAUDE_PLUGIN_DATA=str(data),
    )
    _run(root / "bin" / "memkit-hook", env=env)
    assert shimmed.read()["MEMKIT_CONFIG"] == str(data / "memkit.json")


@pytest.mark.parametrize(
    "wrapper,args",
    [
        ("memkit-hook", ()),
        ("memkit-recall", ("--search", "x")),
        ("memkit", ("doctor",)),
    ],
)
def test_every_wrapper_answers_the_config_question_identically(
    root, tmp_path, shimmed, wrapper, args
) -> None:
    """The config-delivery policy is hand-duplicated in all three wrappers and
    every rung and override case drove one of them.

    Both directions of the override, because the second is the one that
    matters: setting `MEMKIT_CONFIG` when a rung answered is obvious, and
    UNSETTING it when none did is what stops an adopter's other memkit
    installation handing this one a corpus nobody pointed it at. A wrapper that
    dropped the `unset` would serve the ambient config and look identical from
    outside.
    """
    config = _config_file(tmp_path / "shared.json")
    answered = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config),
        MEMKIT_CONFIG=str(tmp_path / "ambient.json"),
    )
    # No return code here: two of the three wrappers reach a real subcommand
    # under a real interpreter now, and `memkit doctor` on a fixture profile
    # legitimately exits 1. What is asserted is the wrapper's own decision and
    # that it got as far as making the hand-off.
    assert _decided_config(
        root, wrapper, answered, *args, expect_rc=None
    ) == str(config), wrapper

    ambient = shimmed(MEMKIT_CONFIG=str(_config_file(tmp_path / "ambient.json")))
    assert _decided_config(
        root, wrapper, ambient, *args, expect_rc=None
    ) == "<unset>", wrapper


def test_no_rung_leaves_the_config_unset_rather_than_inherited(
    root, tmp_path, shimmed
) -> None:
    """A hard override in BOTH directions, and the unsetting half is the one
    that matters. An adopter with a nix or pip memkit on the same machine may
    have MEMKIT_CONFIG exported in the shell that launched the harness;
    inheriting it would make the plugin serve stores nobody pointed it at.
    """
    env = shimmed(MEMKIT_CONFIG=str(_config_file(tmp_path / "ambient.json")))
    assert _decided_config(root, "memkit-hook", env) == "<unset>"


def test_an_unset_data_dir_never_becomes_a_root_level_path(
    root, tmp_path, shimmed
) -> None:
    """`${CLAUDE_PLUGIN_DATA}/memkit.json` with the variable unset is
    `/memkit.json`, and a hook that reads every prompt must never stat a
    root-level path it did not mean to name.

    Read off `sh -x`, which traces every command the shell evaluated — the only
    portable way to see a test the wrapper *made* rather than a path it
    returned. The cwd-relative variant of the same bug is covered too: a
    `memkit.json` sitting in the directory the session happens to stand in must
    not be picked up either.
    """
    cwd = tmp_path / "somewhere"
    cwd.mkdir()
    _config_file(cwd / "memkit.json")
    for env in (shimmed(), shimmed(CLAUDE_PLUGIN_DATA="")):
        out = _run(root / "bin" / "memkit-hook", env=env, cwd=cwd, shell_trace=True)
        assert out.returncode == 0
        # `/memkit.json` as a WHOLE word: every legitimate candidate this
        # builds is prefixed by an absolute directory, so the bare root-level
        # path can only appear through the empty expansion.
        assert not re.search(r"(?<![\w/])/memkit\.json\b", out.stderr), out.stderr
        assert _decided_config(root, "memkit-hook", env, cwd=cwd) == "<unset>"


def test_the_interpreter_the_hook_runs_is_never_found_by_searching(
    root, tmp_path, shimmed
) -> None:
    """The every-prompt path's LAST process start, and who chooses it.

    With no config, or with one that records no interpreter, the wrapper used
    to exec whatever `command -v python3` returned over the session's own
    unfiltered PATH — on every prompt, before any rule in this package existed
    to have an opinion. A checkout that exports a `.direnv/bin` or ships a
    `node_modules/.bin` therefore named the program that read every prompt.

    Three claims, and the third is the one that makes the other two hold:
    a python on PATH is not consulted; a pinned absolute path is; and the
    pinned list itself is not readable from the environment.
    """
    hostile = _shim(tmp_path / "hostile", "python3", "echo pwned")
    env = shimmed()
    env["PATH"] = os.pathsep.join([str(hostile.parent), str(shimmed.dir)])
    decided = _decide(root, "memkit-hook", env)
    assert decided.interpreter in _pinned_pythons(root), decided.interpreter
    assert str(hostile) != decided.interpreter

    # And the list is not an environment input. Exporting the name is the
    # obvious way to try to steer it, and the assignment in the library runs
    # unconditionally when it is sourced, so an inherited value never survives
    # to be read.
    steered = dict(env, MEMKIT_SYSTEM_PYTHONS=str(hostile))
    assert _decide(root, "memkit-hook", steered).interpreter == decided.interpreter

    # And a pinned candidate that is a DIRECTORY is skipped, not exec'd.
    # `[ -x ]` alone is true of one — the execute bit means "searchable" — and
    # `exec` on it dies 126, which on UserPromptSubmit hands the user their
    # prompt back. The list is a list of files.
    a_directory = tmp_path / "libexec-bin"
    a_directory.mkdir()
    real = _shim(tmp_path / "real", "python3", "exit 0")
    skipped = _decide(
        _repinned(root, [a_directory, real]), "memkit-hook", shimmed()
    )
    assert skipped.interpreter == str(real), skipped.interpreter


def test_a_recorded_interpreter_wins_over_the_path(root, tmp_path, shimmed) -> None:
    """PATH probing alone hands the process that reads every prompt to whatever
    direnv/mise/venv shim the launching shell carried. The nix channel pins its
    interpreter absolutely; the plugin channel must not silently drop that.
    """
    recorded = _shim(tmp_path / "recorded", "python3", SHIM_BODY)
    marker = tmp_path / "recorded-ran.txt"
    recorded.write_text(
        f'#!/bin/sh\necho "$*" > "{marker}"\n' + SHIM_BODY
    )
    recorded.chmod(0o755)
    config = _config_file(tmp_path / "rec.json", interpreter=str(recorded))
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert marker.is_file(), "the PATH interpreter answered instead of the recorded one"


def test_an_unusable_recorded_interpreter_falls_back_to_a_pinned_python(
    root, tmp_path, shimmed
) -> None:
    """An interpreter recorded at init and gone by now — a venv deleted, a
    homebrew python upgraded out from under its path — must not take retrieval
    down with it.

    What it falls back TO is a fixed list of absolute system paths, not a
    lookup: the fallback used to be `command -v python3` over the session's own
    PATH, so an install whose recorded interpreter had moved handed the process
    that reads every prompt to whatever the checkout put in front.
    """
    config = _config_file(
        tmp_path / "gone.json", interpreter=str(tmp_path / "no" / "such" / "python3")
    )
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    decided = _decide(root, "memkit-hook", env)
    assert decided.config == str(config)
    assert decided.interpreter in _pinned_pythons(root), decided.interpreter
    # The shim is on PATH and is NOT what answered, which is the whole point.
    assert str(shimmed.dir) not in decided.interpreter, decided.interpreter
    assert not shimmed.out.exists()


def test_a_relative_recorded_interpreter_is_not_a_path_into_the_session_dir(
    root, tmp_path, shimmed
) -> None:
    """`[ -x ]` and `exec` resolve a non-absolute value against different
    things, and only one of them is the wrapper's to choose.

    The test is against the wrapper's CWD, which under the harness is whatever
    directory the session stands in. `exec` on a value with a slash resolves
    against the same CWD and runs it; on a slashless word it searches PATH
    instead. So a recorded `./interp/python3` used to hand the process that
    reads every prompt to an executable sitting in the directory somebody
    happened to open — chosen by the session, not by the install.
    """
    session = tmp_path / "session"
    marker = tmp_path / "relative-ran.txt"
    _shim(session / "interp", "python3", f'echo ran > "{marker}"')
    config = _config_file(tmp_path / "rel.json", interpreter="./interp/python3")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    decided = _decide(root, "memkit-hook", env, cwd=session)
    assert not marker.exists(), "a config named an interpreter inside the cwd"
    assert decided.interpreter in _pinned_pythons(root), decided.interpreter
    assert decided.handoff[1:] == [
        str(root / "src" / "memkit" / "memory_prompt_recall.py")
    ], decided.handoff


def test_a_recorded_interpreter_the_path_cannot_answer_still_exits_zero(
    root, tmp_path
) -> None:
    """The exit contract, at the config field rather than at an empty PATH.

    A slashless `"interpreter": "python3"` passed the executable test against
    a CWD that happened to hold a file of that name, and then `exec` — which
    searches PATH for a slashless word — found nothing and left the
    every-prompt hook exiting 127. On UserPromptSubmit that is a blocked turn,
    produced by a config field on a machine where the fallback would have
    worked had it been consulted.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("sed", "head"):
        found = shutil.which(name)
        assert found, name
        (tools / name).symlink_to(found)
    session = tmp_path / "session"
    marker = tmp_path / "session-python-ran.txt"
    _shim(session, "python3", f'echo ran > "{marker}"')
    config = _config_file(tmp_path / "bare.json", interpreter="python3")
    env = {
        "PATH": str(tools),
        "HOME": str(tmp_path / "home"),
        "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(config),
    }
    decided = _decide(root, "memkit-hook", env, cwd=session)
    # The field is refused BY SHAPE and the pinned fallback answers, so the
    # slashless word never reaches `exec` at all — which is where the 127 came
    # from, and where a `python3` sitting in the session's own directory would
    # have been found instead.
    assert not marker.exists(), "the session directory's python3 answered"
    assert decided.interpreter in _pinned_pythons(root), decided.interpreter
    assert "is not an absolute path" in decided.stderr, decided.stderr[-400:]
    assert "pinned system python" in decided.stderr, decided.stderr[-400:]


def test_a_directory_recorded_as_the_interpreter_is_not_exec_d(
    root, tmp_path, shimmed
) -> None:
    """`[ -x ]` is true of a DIRECTORY — the execute bit means "searchable" —
    so a value like `/opt/homebrew/opt/python@3.12/libexec/bin`, which is how
    the field gets written by hand from a PATH entry with the last segment
    dropped, passed the guard and skipped the PATH probe.

    `exec` then died 126, on every prompt of every session, with no fallback:
    the hook wrapper's whole contract is that every path exits 0, and 126 on
    UserPromptSubmit hands the user their prompt back unanswered. The status
    comes from `exec` rather than from a literal, so no scrape can see it —
    only a run can.
    """
    a_directory = tmp_path / "libexec-bin"
    a_directory.mkdir()
    config = _config_file(tmp_path / "dir.json", interpreter=str(a_directory))
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    decided = _decide(root, "memkit-hook", env)
    # The pinned fallback answered, which is what the refusal promises.
    assert decided.interpreter in _pinned_pythons(root), decided.interpreter
    assert decided.handoff[1:] == [
        str(root / "src" / "memkit" / "memory_prompt_recall.py")
    ], decided.handoff
    # And every wrapper, because each one has its own exit vocabulary and 126
    # is in none of them.
    for wrapper, args in (
        ("memkit-recall", ("--search", "flange torque")),
        ("memkit", ("doctor",)),
    ):
        other = _run(root / "bin" / wrapper, *args, env=shimmed(
            CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)
        ))
        assert other.returncode != 126, (wrapper, other.returncode, other.stderr)


def test_a_relative_config_path_is_not_a_path_into_the_session_dir(
    root, tmp_path, shimmed
) -> None:
    """The same rule as the interpreter field, on the rung above it.

    An adopter who typed `--config memkitConfig=memkit.json` at install has
    every repository they later open handing the every-prompt hook its own
    `memkit.json` — which names both the store roots whose file contents are
    injected into the model's context and the absolute binary exec'd on every
    prompt. The manifest asks for an absolute path and nothing enforced it.
    """
    session = tmp_path / "someone-elses-repo"
    session.mkdir()
    marker = tmp_path / "session-config-used.txt"
    _shim(session, "python3", f'echo used > "{marker}"')
    _config_file(session / "memkit.json", interpreter=str(session / "python3"))
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG="memkit.json")
    assert _decided_config(root, "memkit-hook", env, cwd=session) == "<unset>"
    assert not marker.exists(), "the session directory named the interpreter"


# --- the dependency contract -------------------------------------------------

# Words that are shell syntax rather than a command, and the builtins the
# wrappers are allowed to spend. Everything else a scrape finds is something
# that has to be FOUND on a PATH the harness composed, which is the failure
# this pins: a `head` nobody had noticed took the whole interpreter resolution
# out inside a Linux nix sandbox, and the wrapper still exited 0.
SHELL_SYNTAX = frozenset({
    "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done",
    "case", "esac", "in", "function", "select", "time",
})
ALLOWED_BUILTINS = frozenset({
    "command", "printf", "echo", "read", "cd", "pwd", "export", "unset", "set",
    "shift", "exec", "eval", "trap", "exit", "return", "break", "continue",
    "local", "true", "false", "test",
})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_REDIRECT = re.compile(r"^\d?[<>]+")
_WORD = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)")


def _command_words(text: str) -> set[str]:
    """Every word this text uses in command position.

    Deliberately crude and deliberately DEFAULT-DENY: it over-collects rather
    than under-collects, and the test subtracts the names that are allowed
    instead of listing the ones that are not. A denylist of `sed`, `head`,
    `grep` would have to be extended by the same person who added the command.
    """
    words: set[str] = set()
    for raw in text.splitlines():
        # A `#` only opens a comment at the start of a word — `${x#pat}` is a
        # parameter expansion, and eating the rest of that line would hide any
        # command after it.
        line = re.sub(r"(^|\s)#.*$", r"\1", raw)
        # Parameter expansions hold no command position, and their contents are
        # full of the characters everything below splits on.
        for _ in range(4):
            line = re.sub(r"\$\{[^{}]*\}", "$X", line)
        line = re.sub(r"'[^']*'", "''", line)
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        # `$(` and a backtick OPEN a command position; the rest merely end one.
        for fragment in re.split(r"[;|&]+|\$\(|\)|`", line):
            piece = fragment.strip()
            while piece:
                head, _, rest = piece.partition(" ")
                # `for x in …` and `case x in …` name a variable and a subject,
                # neither of which is ever run.
                if head in ("for", "case", "select"):
                    piece = ""
                    break
                if (
                    head in SHELL_SYNTAX
                    or _ASSIGNMENT.match(head)
                    or _REDIRECT.match(head)
                    or head in ("!", "{", "}", "(")
                ):
                    piece = rest.strip()
                    continue
                break
            match = _WORD.match(piece)
            if match:
                words.add(match.group(1))
    return words


def test_the_wrappers_run_no_command_they_would_have_to_find() -> None:
    """The dependency contract stated in `bin/lib/common.sh`, held to.

    The PATH these run on belongs to the harness, and nothing obliges it to
    carry coreutils. Reading one config field through `sed | head` was enough
    to make the recorded interpreter unreadable inside a Linux nix sandbox
    while the wrapper went on reporting healthy.
    """
    defined = set()
    sources = {}
    for rel in ("bin/lib/common.sh", "bin/memkit", "bin/memkit-hook", "bin/memkit-recall"):
        sources[rel] = (REPO / rel).read_text(encoding="utf-8")
        defined |= set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\)", sources[rel], re.M))
    external = {
        rel: sorted(_command_words(text) - SHELL_SYNTAX - ALLOWED_BUILTINS - defined)
        for rel, text in sources.items()
    }
    assert not any(external.values()), external
    # Non-vacuity: the scrape really does see command words, and it really does
    # see the ones that are allowed. A parser that collected nothing would pass
    # the assertion above for the wrong reason.
    seen = set().union(*(_command_words(t) for t in sources.values()))
    assert {"command", "printf", "exec"} <= seen, sorted(seen)
    assert defined >= {"memkit_resolve_config", "memkit_resolve_interpreter"}, defined


@pytest.mark.parametrize(
    "wrapper,args",
    [("memkit-hook", ()), ("memkit-recall", ("--search", "x")), ("memkit", ("doctor",))],
)
def test_a_wrapper_still_works_with_nothing_on_its_path(
    root, tmp_path, shimmed, wrapper, args
) -> None:
    """The same contract, driven rather than read — with an EMPTY PATH, which
    is every missing coreutils at once.

    The marker is the half that matters: an assertion that only checked stderr
    would pass on a wrapper that read no config and exec'd nothing, which is
    exactly what the missing `head` produced.
    """
    marker = tmp_path / "interpreter-ran.txt"
    interpreter = _shim(tmp_path / "elsewhere", "python3", f'echo ran > "{marker}"')
    config = _config_file(tmp_path / "memkit.json", interpreter=str(interpreter))
    env = dict(
        shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)), PATH=""
    )
    out = _run(root / "bin" / wrapper, *args, env=env)
    assert "not found" not in out.stderr, out.stderr
    assert marker.is_file(), (
        "the interpreter recorded in the config was never read or never run",
        out.stderr,
    )
    # And the field really was parsed out of the file rather than guessed: the
    # only python3 that could have answered is the one the config names, since
    # the PATH holding the shim is gone too.
    assert marker.read_text().strip() == "ran"


# `interpreter` written the ways a config really carries it. The reader is
# hand-rolled shell rather than a JSON parser — it has to be, since it may not
# spend a command — so the shapes are driven rather than reasoned about.
CONFIG_SHAPES = {
    "compact": '{{"schema":1,"interpreter":"{py}"}}\n',
    "pretty": '{{\n  "schema": 1,\n  "interpreter": "{py}"\n}}\n',
    "padded": '{{\n  "schema": 1,\n  "interpreter"   :    "{py}"\n}}\n',
    "tabbed": '{{\n\t"interpreter"\t:\t"{py}"\n}}\n',
    # No trailing newline: `read` reports failure on that last line, and a loop
    # that trusted its status would read every config except the ones an editor
    # did not finish.
    "unterminated": '{{"schema":1,"interpreter":"{py}"}}',
    # The word as a plain string, before the real key. It is a complete
    # `"interpreter"` with a comma after it rather than a colon, so a reader
    # that scanned to the first occurrence and gave up would take this
    # config's interpreter to be nothing at all.
    "decoyed": '{{"tags":["interpreter","python"],"interpreter":"{py}"}}\n',
}


@pytest.mark.parametrize("shape", sorted(CONFIG_SHAPES))
def test_the_interpreter_field_is_read_out_of_the_shapes_a_config_takes(
    root, tmp_path, shimmed, shape
) -> None:
    """Every shape must reach the same interpreter, with an empty PATH so the
    only thing that could have answered is the recorded one."""
    marker = tmp_path / f"{shape}-ran.txt"
    interpreter = _shim(tmp_path / "elsewhere", "python3", f'echo ran > "{marker}"')
    config = tmp_path / "memkit.json"
    config.write_text(CONFIG_SHAPES[shape].format(py=interpreter), encoding="utf-8")
    env = dict(shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)), PATH="")
    out = _run(root / "bin" / "memkit-hook", env=env)
    assert out.returncode == 0, out.stderr
    assert marker.is_file(), (shape, out.stderr, config.read_text())


def test_the_shim_fixture_can_reach_nothing_it_did_not_write(shimmed) -> None:
    """The fixture's PATH is a CLAIM the cases using it depend on, so it is
    asserted here rather than left to be true by accident.

    It was not: with `/usr/bin:/bin` appended, a runner whose system python3
    meets the checker floor answered for a shim written to refuse it, and two
    route cases asserted `uvx` and `none` against a machine that had neither
    question. The failure was invisible on the author's platform, where the
    system python is too old to qualify — which is the whole reason this line
    is a test and not a comment.
    """
    assert shimmed()["PATH"] == str(shimmed.dir)
    assert [p.name for p in shimmed.dir.iterdir()] == ["python3"]


def test_a_config_path_that_is_merely_wrong_is_not_the_same_as_no_config(
    root, tmp_path, shimmed
) -> None:
    """A typo in `memkitConfig` must reach the adopter, and only the shape
    rules reached them.

    `/home/them/memkti.json` is absolute, canonical, and outside every
    respelling the refusal helper knows — so the rung falls through in silence
    and every diagnostic this build has says `config: none`, byte for byte what
    an install that never set the option says. The one person who can be sure a
    config exists is the one who typed the path, and this was the state that
    told them nothing.

    The two failures are separated because the remedies are: a path that is not
    there is a typo or an uninstalled file, and a path that is there and shut
    is a permission on a file the adopter believes they own.
    """
    missing = tmp_path / "not-installed" / "memkit.json"
    said = {}
    for label, path in (("missing", missing), ("present", _config_file(tmp_path / "real.json"))):
        out = _run(
            root / "bin" / "memkit-hook",
            env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(path)),
        )
        # Still zero. This runs on UserPromptSubmit, where any other status
        # takes the user's prompt away from them over a config typo.
        assert out.returncode == 0, (label, out.returncode, out.stderr)
        said[label] = out.stderr
    assert str(missing) in said["missing"], said["missing"]
    assert "does not exist" in said["missing"], said["missing"]
    # The control that makes the assertion above mean something: a config that
    # IS there says nothing at all, so the line is about this state and not
    # about the option being set.
    assert said["present"] == "", said["present"]

    # And no option at all stays silent too — the state the message exists to
    # be distinguishable from.
    quiet = _run(root / "bin" / "memkit-hook", env=shimmed())
    assert quiet.returncode == 0, quiet.stderr
    assert quiet.stderr == "", quiet.stderr
    assert said["missing"] != quiet.stderr

    # Rung 2 deliberately does NOT do this. `$CLAUDE_PLUGIN_DATA/memkit.json`
    # is absent on every plugin install until something writes it, so a line
    # about it would fire on every prompt of the normal pre-init state.
    data = tmp_path / "plugindata"
    data.mkdir()
    pre_init = _run(root / "bin" / "memkit-hook", env=shimmed(CLAUDE_PLUGIN_DATA=str(data)))
    assert pre_init.returncode == 0, pre_init.stderr
    assert pre_init.stderr == "", pre_init.stderr

    # The README teaches the pair, because reading `config: none` correctly
    # depends on knowing whether anything was said beside it.
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "`config: none` with nothing on\nstderr means the option was never set" in readme


@needs_permissions
def test_a_config_this_process_cannot_open_says_so_rather_than_going_quiet(
    root, tmp_path, shimmed
) -> None:
    """The other half of the rung's readability check, and a different remedy.

    `[ -r ]` answers false for a file that is not there and for one that is
    there behind a mode, and an adopter chasing the second while being told the
    first goes looking in the wrong place.
    """
    shut = _config_file(tmp_path / "shut.json")
    shut.chmod(0o000)
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(shut))
    try:
        out = _run(root / "bin" / "memkit-hook", env=env)
        # The hook did not go on to use it — the rung was abandoned, not
        # retried.
        decided = _decided_config(root, "memkit-hook", env)
    finally:
        shut.chmod(0o644)
    assert out.returncode == 0, (out.returncode, out.stderr)
    assert "cannot be read" in out.stderr, out.stderr
    assert "does not exist" not in out.stderr, out.stderr
    assert decided == "<unset>"


def test_a_relative_plugin_data_dir_is_not_a_path_into_the_session_dir(
    root, tmp_path, shimmed
) -> None:
    """The same rule as rung 1, on the rung beside it.

    `$CLAUDE_PLUGIN_DATA` is the harness's variable and the wrappers do not vet
    where it came from, which is recorded above the resolver. What they CAN do
    is refuse to resolve it against the session's directory: a relative value
    made the every-prompt hook read `<cwd>/<value>/memkit.json`, a config
    naming both the store roots whose contents are injected and the binary that
    is exec'd. The comment claiming "ABSOLUTE, on every rung" was written by
    the commit that guarded one of the two.
    """
    session = tmp_path / "someone-elses-repo"
    (session / "reldata").mkdir(parents=True)
    marker = tmp_path / "session-data-used.txt"
    _shim(session, "python3", f'echo used > "{marker}"')
    _config_file(
        session / "reldata" / "memkit.json", interpreter=str(session / "python3")
    )
    env = shimmed(CLAUDE_PLUGIN_DATA="reldata")
    assert _decided_config(root, "memkit-hook", env, cwd=session) == "<unset>"
    assert not marker.exists(), "the session directory named the interpreter"

    # And a tilde is expanded rather than refused, exactly as rung 1 does it —
    # the value is typed by a person or written by a harness, not expanded by
    # any shell.
    home = tmp_path / "home"
    data = home / "plugindata"
    config = _config_file(data / "memkit.json")
    tilde = shimmed(CLAUDE_PLUGIN_DATA="~/plugindata")
    assert _run(root / "bin" / "memkit-hook", env=tilde).returncode == 0
    assert shimmed.read()["MEMKIT_CONFIG"] == str(config)


def test_every_wrapper_declares_the_name_it_answers_to() -> None:
    """`MEMKIT_SELF` decides the binary named in every message the shared
    library emits, and the library's fallback means a wrapper that forgot to
    set it fails silently — naming the wrong binary beside its own exit code,
    which is the defect the variable was added to fix.

    Nothing in the library can enforce it: a hard failure there would sit on
    the every-prompt path for a diagnostic. So the enforcement is here.
    """
    for wrapper in ("memkit", "memkit-hook", "memkit-recall"):
        text = (BIN / wrapper).read_text()
        assert f"\nMEMKIT_SELF={wrapper}\n" in text, wrapper
        # Before the library is SOURCED — not merely before it is mentioned —
        # or the first message it emits on this wrapper's behalf carries the
        # fallback instead of the name.
        sourced = text.index('. "$MEMKIT_ROOT/bin/lib/common.sh"')
        assert text.index(f"MEMKIT_SELF={wrapper}") < sourced, wrapper


@pytest.mark.parametrize(
    "value",
    [
        "/proc/self/cwd/memkit.json",
        "//etc/memkit.json",
        "/./etc/memkit.json",
        "/tmp/../etc/memkit.json",
        "/dev/fd/3/memkit.json",
        "relative/memkit.json",
    ],
)
def test_a_config_rung_admits_only_what_the_interpreter_rule_admits(
    root, tmp_path, shimmed, value
) -> None:
    """One admission rule, applied to the whole class it names.

    The interpreter field refused non-canonical and process-relative paths and
    the config rungs tested only for a leading slash — the wider blast radius
    guarded more weakly, since a config decides which directories the
    every-prompt hook reads AND which binary it execs. On Linux, which is this
    repo's CI and the nix channel, `/proc/self/cwd/memkit.json` is absolute,
    passes a leading-slash test, and resolves through the running process.

    Driven with a file that really is there at the non-canonical spelling, so
    a refusal cannot be confused with the path not existing.
    """
    data = tmp_path / "data"
    data.mkdir()
    _config_file(data / "memkit.json")
    noncanonical = f"{tmp_path}/./data/memkit.json"
    for rung, env in (
        ("option", shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=noncanonical)),
        ("data dir", shimmed(CLAUDE_PLUGIN_DATA=f"{tmp_path}/./data")),
    ):
        assert _decided_config(root, "memkit-hook", env) == "<unset>", rung

    # The class itself, one spelling per run, through the option.
    assert _decided_config(
        root, "memkit-hook", shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=value)
    ) == "<unset>", value

    # And a canonical absolute path is still served, or this is "refuse
    # everything" wearing a rule's clothes.
    good = _config_file(tmp_path / "good" / "memkit.json")
    assert _decided_config(
        root, "memkit-hook", shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(good))
    ) == str(good)


def test_a_process_relative_interpreter_is_refused(root, tmp_path, shimmed) -> None:
    """Absolute is not the same as fixed. `/proc/self/cwd/python3` is resolved
    by the kernel through the RUNNING process, so it names an executable in
    whatever directory the session stands in — the outcome the absoluteness
    rule exists to prevent, restored on Linux, which is what the nix channel
    and this repo's CI run.

    `/dev/fd/*` is the same class and is reachable on this platform, which is
    what makes the case runnable here rather than only reasoned about.
    """
    # Every respelling, because the guard is a literal prefix test and the
    # kernel normalises before it resolves — six of these walked past it while
    # naming exactly what it refuses.
    for value in (
        "/proc/self/cwd/python3",
        "//proc/self/cwd/python3",
        "/./proc/self/cwd/python3",
        "/tmp/../proc/self/cwd/python3",
        "/proc//self/cwd/python3",
        "/usr/../proc/self/cwd/python3",
        "/a/./b/../proc/self/cwd/python3",
        "/dev/fd/3/python3",
        "/dev//fd/3/python3",
        "/./dev/fd/3/python3",
        "/proc/self/cwd/.",
        "/proc/self/cwd/..",
    ):
        config = _config_file(tmp_path / "proc.json", interpreter=value)
        env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
        decided = _decide(root, "memkit-hook", env)
        assert value in decided.stderr, (value, decided.stderr)
        assert (
            "session stands in" in decided.stderr or "canonical" in decided.stderr
        ), (value, decided.stderr)
        assert decided.interpreter in _pinned_pythons(root), decided.interpreter
    # And a canonical absolute path is still honoured, or the guard above is
    # just "refuse everything".
    honoured = _shim(tmp_path / "real", "python3", "exit 0")
    marker = tmp_path / "canonical-ran.txt"
    honoured.write_text(f'#!/bin/sh\necho ran > "{marker}"\n')
    honoured.chmod(0o755)
    config = _config_file(tmp_path / "ok.json", interpreter=str(honoured))
    assert _run(
        root / "bin" / "memkit-hook",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)),
    ).returncode == 0
    assert marker.is_file(), "a canonical absolute interpreter was refused"


def test_a_recorded_interpreter_expands_a_tilde_and_says_when_it_is_refused(
    root, tmp_path, shimmed
) -> None:
    """Two halves of one complaint: an adopter records an interpreter, keeps a
    working install, and runs every prompt under a python they did not choose.

    The tilde half is a trap of this file's own making — the config PATH one
    rung above is tilde-expanded, so the file teaches that `~` works — and the
    silence half is what made it undiagnosable: exit 0, nothing on stderr, and
    no surface in this build reports the resolved interpreter.
    """
    home = tmp_path / "home"
    recorded = _shim(home / "venv" / "bin", "python3", "exit 0")
    marker = tmp_path / "recorded-ran.txt"
    recorded.write_text(f'#!/bin/sh\necho ran > "{marker}"\n')
    recorded.chmod(0o755)
    config = _config_file(tmp_path / "tilde.json", interpreter="~/venv/bin/python3")
    env = shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config))
    assert _run(root / "bin" / "memkit-hook", env=env).returncode == 0
    assert marker.is_file(), "a tilde interpreter was refused, as if it were relative"

    # And a value that really is unusable is refused OUT LOUD.
    gone = _config_file(tmp_path / "gone.json", interpreter=str(tmp_path / "nope"))
    out = _run(
        root / "bin" / "memkit-hook",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(gone)),
    )
    assert out.returncode == 0
    assert "is not an executable file" in out.stderr, out.stderr
    assert "Falling back" in out.stderr, out.stderr


def test_every_shared_message_names_the_wrapper_that_is_running(
    root, tmp_path
) -> None:
    """The messages are shared between the three wrappers and the exit codes
    are not.

    `memkit-recall` exits 4 when nothing can run it. An agent that read
    `memkit:` on that line looks 4 up in the `memkit` table, where it means
    "the subcommand exists and is not in this build" — a wrong diagnosis
    produced by the name in the message rather than by the code.
    """
    empty = {"PATH": str(tmp_path / "nothing"), "HOME": str(tmp_path)}
    bare = _repinned(root, [tmp_path / "nowhere"])
    for wrapper, args in (
        ("memkit-hook", ()),
        ("memkit-recall", ("--search", "x")),
        ("memkit", ("doctor",)),
    ):
        out = _run(bare / "bin" / wrapper, *args, env=empty)
        assert out.stderr.startswith(f"{wrapper}: "), (wrapper, out.stderr)


def test_no_interpreter_is_a_named_refusal_that_still_exits_zero(
    root, tmp_path
) -> None:
    """The exit contract, at the one failure that cannot be fixed by fixing the
    store. Exit 2 on UserPromptSubmit blocks the turn and hands the prompt back
    unanswered; any non-zero exit puts an error in front of the user on every
    prompt. So the refusal exits 0 and speaks on stderr, where doctor — which
    runs this wrapper directly — reads it.
    """
    env = {"PATH": str(tmp_path / "empty"), "HOME": str(tmp_path)}
    out = _run(_repinned(root, [tmp_path / "nowhere"]) / "bin" / "memkit-hook", env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""
    assert "no interpreter is recorded" in out.stderr, out.stderr
    assert "3.9" in out.stderr and "memkit init" in out.stderr, out.stderr
    # The refusal names the paths it tried, so an adopter can see whether their
    # python is simply somewhere this list does not reach.
    assert str(tmp_path / "nowhere") in out.stderr, out.stderr


def _code_only(line: str) -> str:
    r"""One shell line with its comment and its quoted spans removed.

    Both removals matter for finding a real `exit`. A trailing `# why` must not
    hide the statement in front of it, and the word inside
    `echo "exit 98 in a string"` must not be read as one. Quoted spans are
    blanked rather than dropped so column-shaped assertions stay honest.

    Only a `#` that starts a word is a comment: `"$#"` is an argument count and
    `${1#\~/}` is a parameter expansion, and both occur in these wrappers.
    """
    out: list[str] = []
    quote = None
    for char in line:
        if quote is not None:
            out.append(" " if char != quote else char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "#" and (not out or out[-1].isspace()):
            break
        else:
            out.append(char)
    return "".join(out)


EXIT_TOKEN = re.compile(r"(?<![\w-])exit(?![\w-])")


def _exit_statuses(text: str, where: str) -> set[int]:
    r"""Every status a shell file can `exit` with, DEFAULT-DENY.

    The rule is inverted from the obvious one: rather than matching the exit
    forms we expect and ignoring the rest, this finds every `exit` token in
    code and fails on any whose form it does not recognise. The previous shape
    — `^\s*exit (\d+)$` — required end-of-line immediately after the digits,
    so `exit 1  # why`, `cmd && exit 1` (the idiomatic POSIX form) and a bare
    `exit` were all invisible, and a paired "no computed exits" guard did not
    fire when the next character was a digit. Measured: an unreached
    `exit 1  # comment` planted in the hook wrapper left the whole suite green.

    Necessary and NOT sufficient, in both directions, which is why every case
    using this pairs it with real runs. A scrape cannot see `set -u` aborting
    on an unbound variable — non-zero, with no literal to find — and it cannot
    see the final `exec`, which hands the python side's own status through as
    the wrapper's. What it does see is the thing a runtime case cannot: a
    branch nobody remembered to reach.
    """
    statuses = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        code = _code_only(raw).rstrip()
        if not EXIT_TOKEN.search(code):
            continue
        at = f"{where}:{lineno}"
        assert len(EXIT_TOKEN.findall(code)) == 1, f"{at}: two exits on one line"
        # Whatever precedes it may be a `&&`/`||` chain or a `case` arm; what
        # FOLLOWS it must be a literal status and nothing else.
        match = re.search(EXIT_TOKEN.pattern + r"\s*(\S*)\s*(;;)?$", code)
        assert match, f"{at}: unrecognised exit form: {code.strip()!r}"
        assert match.group(1).isdigit(), (
            f"{at}: exit without a literal status — a bare `exit` propagates "
            f"$? and a computed one cannot be read here: {code.strip()!r}"
        )
        statuses.add(int(match.group(1)))
    assert statuses, f"{where} has no exit literals — this pin would be vacuous"
    return statuses


def _exit_literals(wrapper: str) -> set[int]:
    return _exit_statuses((REPO / "bin" / wrapper).read_text(), wrapper)


def test_the_sourced_library_can_end_no_wrapper() -> None:
    """`bin/lib/common.sh` is SOURCED, so an `exit` in it exits the wrapper —
    and every scrape read `bin/<wrapper>` only.

    Measured: a top-level `[ … ] && exit 1` planted in the library made
    `bin/memkit-hook` return 1 — the non-zero `UserPromptSubmit` exit the hook
    wrapper's whole contract is about, which blocks the turn — and the entire
    suite stayed green. The library is also where the refusal paths live, so
    it is exactly the file that keeps gaining reasons to want an exit.

    A NEGATIVE pin rather than `_exit_literals`, which asserts a non-empty set:
    the right number of exits here is none.
    """
    text = COMMON_SH.read_text(encoding="utf-8")
    found = [
        (n, line.strip())
        for n, line in enumerate(text.splitlines(), 1)
        if EXIT_TOKEN.search(_code_only(line))
    ]
    assert not found, found
    # `return` is how this file ends a function, and it must have some, or the
    # assertion above is about a file that stopped being a resolver.
    assert re.search(r"^\s*return \d+$", text, re.MULTILINE), "no returns either"


def test_the_exit_scrape_sees_the_forms_it_would_otherwise_miss() -> None:
    """The anti-vacuity control for the helper above, which is the static half
    of the hook's fail-open contract.

    Each line here is a form the previous regex could not see. A scrape that
    silently stops matching is a green test about nothing, and this one guards
    a property whose failure mode is an error in front of every prompt.
    """
    seen = _exit_statuses(
        "\n".join(
            (
                "exit 0",
                "exit 2  # trailing comment",
                "[ -n \"$x\" ] && exit 3",
                "foo || exit 4",
                "    exit 5 ;;",
                "# exit 99 in prose is not an exit",
                'echo "exit 98 in a string is not one either"',
            )
        ),
        "<control>",
    )
    assert seen == {0, 2, 3, 4, 5}, seen
    for hostile in ("exit", "exit $rc", "exit 1 || true", "exit 0; exit 1"):
        with pytest.raises(AssertionError):
            _exit_statuses(hostile, "<control>")


def _half_delivered(tmp_path: Path, wrapper: str, *, library: bool) -> Path:
    """A payload root missing the hook file, with or without the library.

    The distinction is the whole point: copying only the wrapper leaves control
    at the `common.sh` guard, so the branch that answers for an incomplete
    payload is never executed and a wrong exit code there survives every test
    that thinks it covers it.
    """
    root = tmp_path / f"{wrapper}-{'half' if library else 'bare'}"
    if root.exists():
        # Built once per shape and reused. Re-copying is not merely wasteful:
        # the source may be read-only — it is under /nix/store in the packaged
        # check — and `shutil.copy` carries the mode across, so the second
        # write to the same destination fails with EACCES.
        return root
    (root / "bin" / "lib").mkdir(parents=True)
    shutil.copy(REPO / "bin" / wrapper, root / "bin" / wrapper)
    if library:
        shutil.copy(COMMON_SH, root / "bin" / "lib" / "common.sh")
    return root


def test_the_hook_wrapper_never_exits_non_zero(root, tmp_path, shimmed) -> None:
    """Every reachable refusal, in one place, because the property is about the
    SET of them: a new branch that exits 1 is invisible until it is the branch
    an adopter is on, and by then it is a message in front of every prompt.
    """
    assert _exit_literals("memkit-hook") == {0}

    cases = [
        # no interpreter anywhere
        (root / "bin" / "memkit-hook", {"PATH": str(tmp_path / "nothing")}, ()),
        # cannot locate the tree at all: no library to source
        (_half_delivered(tmp_path, "memkit-hook", library=False) / "bin"
         / "memkit-hook", shimmed(), ()),
        # the library is there and the hook file is not — the branch the case
        # above cannot reach, because control leaves at the library guard
        (_half_delivered(tmp_path, "memkit-hook", library=True) / "bin"
         / "memkit-hook", shimmed(), ()),
        # arguments that should never arrive, and are ignored if they do
        (root / "bin" / "memkit-hook", shimmed(), ("--search", "x")),
    ]
    for wrapper, env, args in cases:
        out = _run(wrapper, *args, env={"HOME": str(tmp_path), **env})
        assert out.returncode == 0, (wrapper, args, out.returncode, out.stderr)
    # And the branches really are distinct, or two of the cases above are one
    # case run twice.
    messages = {
        _run(
            _half_delivered(tmp_path, "memkit-hook", library=lib) / "bin"
            / "memkit-hook",
            env={"HOME": str(tmp_path), **shimmed()},
        ).stderr.split("\n")[0]
        for lib in (False, True)
    }
    assert len(messages) == 2, messages


@needs_permissions
@pytest.mark.parametrize(
    "wrapper,args,allowed",
    [
        ("memkit-hook", (), {0}),
        ("memkit-recall", ("--search", "x"), {2, 4}),
        ("memkit", ("doctor",), {1}),
    ],
)
def test_a_payload_that_cannot_be_READ_is_refused_not_exec_d(
    tmp_path, shimmed, wrapper, args, allowed
) -> None:
    """`[ -f ]` admits a file this process cannot open, and the guard's whole
    job is to stop before something else fails on it.

    Measured on the hook wrapper: with the payload at mode 000 the guard passed
    and `exec` reached CPython, which exits **2** when it cannot open the
    script — the one status the file's own header says must never be returned,
    because on UserPromptSubmit it hands the user their prompt back unanswered.
    An unreadable `common.sh` is the same shape one line up: sourcing it is
    fatal on some `/bin/sh`.

    Each wrapper against ITS OWN documented set, because they do not share one:
    the hook may only ever exit 0, and the other two have codes that mean
    "could not start".
    """
    root = tmp_path / "payload"
    for rel in PAYLOAD:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dest)
        dest.chmod(0o755 if rel.startswith("bin/") else 0o644)

    for unreadable in ("src/memkit/memory_prompt_recall.py", "src/memkit/cli.py"):
        (root / unreadable).chmod(0o000)
    out = _run(root / "bin" / wrapper, *args, env=shimmed())
    assert out.returncode in allowed, (wrapper, out.returncode, out.stderr)
    assert "incomplete" in out.stderr, out.stderr

    # And the library one rung up, which is sourced rather than exec'd.
    (root / "bin" / "lib" / "common.sh").chmod(0o000)
    out = _run(root / "bin" / wrapper, *args, env=shimmed())
    assert out.returncode in allowed, (wrapper, out.returncode, out.stderr)
    assert "cannot locate the plugin tree" in out.stderr, out.stderr


# The reachable set differs by ENTRY POINT SHAPE, and that is the whole
# finding: the hook and the search wrapper `exec` the hook module as a loose
# file, which imports nothing from the package around it, while the dispatcher
# runs `python -m memkit.cli` and so pulls in `__init__` and everything
# `__init__` imports. Guarding the file named on the command line covered one
# shape and not the other.
@needs_permissions
@pytest.mark.parametrize(
    "wrapper,args,allowed,reachable",
    [
        ("memkit-hook", (), {0}, ("src/memkit/memory_prompt_recall.py",)),
        ("memkit-recall", ("--search", "x"), {2, 4}, ("src/memkit/memory_prompt_recall.py",)),
        (
            "memkit",
            ("--help",),
            {1},
            (
                "src/memkit/__init__.py",
                "src/memkit/memory_prompt_recall.py",
                "src/memkit/cli.py",
            ),
        ),
    ],
)
def test_each_payload_file_is_refused_on_its_own_not_only_as_a_set(
    tmp_path, shimmed, wrapper, args, allowed, reachable
) -> None:
    """One unreadable file at a time, because shutting two at once lets either
    guard answer for both.

    The case above shuts `memory_prompt_recall.py` and `cli.py` together, so
    the dispatcher's guard on `cli.py` fired and the run looked correct while
    an unreadable HOOK MODULE went to `python -m` and came back as an import
    traceback — the package's `__init__` imports it, and a guard that names
    only the module on the `-m` line does not cover what that import reaches.

    Each wrapper against every python file ITS OWN entry point reaches — see
    the table above, which is where that difference is written down.
    """
    root = tmp_path / "payload"
    for rel in PAYLOAD:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dest)
        dest.chmod(0o755 if rel.startswith("bin/") else 0o644)

    for rel in reachable:
        (root / rel).chmod(0o000)
        try:
            out = _run(root / "bin" / wrapper, *args, env=shimmed())
        finally:
            (root / rel).chmod(0o644)
        assert out.returncode in allowed, (rel, out.returncode, out.stderr)
        assert "Traceback" not in out.stderr, (rel, out.stderr)
        assert out.stderr.startswith(f"{wrapper}: "), (rel, out.stderr)
        assert "incomplete" in out.stderr, (rel, out.stderr)

    # Non-vacuity: with nothing shut, the same invocation gets through to the
    # payload — so the refusals above are the guard and not the environment.
    ok = _run(root / "bin" / wrapper, *args, env=shimmed())
    assert "incomplete" not in ok.stderr, ok.stderr


def test_the_search_wrapper_says_it_could_not_start_rather_than_that_you_erred(
    root, tmp_path, shimmed
) -> None:
    """The opposite assertion to the hook wrapper's, and it is opposite on
    purpose: here there is no prompt to get out of the way of, and the caller
    is an agent choosing a next move from the code.

    2 already means "what you asked for is wrong" — an unparseable config, a
    `--dir` that is not there, arguments that make no sense — and all three of
    those send an agent to fix its own request. A machine with no python on it
    answers none of them, so it gets a code of its own; otherwise the one
    failure no query can survive is reported as the one class of failure a
    different query might.
    """
    assert _exit_literals("memkit-recall") == {hook.EXIT_ERROR, hook.EXIT_CANNOT_START}
    assert hook.EXIT_CANNOT_START not in (
        hook.EXIT_OK, hook.EXIT_NO_MATCH, hook.EXIT_ERROR, hook.EXIT_INERT
    )
    # A code an agent branches on and cannot look up is a code it has to guess
    # at, so the row is part of the change rather than a follow-up — and this
    # one has to warn about the collision, because `memkit`'s own table gives
    # the same number a different meaning.
    from memkit.cli import EXIT_NOT_IN_BUILD

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    row = next(
        (
            line for line in readme.splitlines()
            if line.startswith(f"| {hook.EXIT_CANNOT_START} |")
            and "memkit-recall" in line
        ),
        None,
    )
    assert row, "no README row for the search CLI's start-failure code"
    assert EXIT_NOT_IN_BUILD == hook.EXIT_CANNOT_START and "different meaning" in row

    empty = {"PATH": str(tmp_path / "nothing"), "HOME": str(tmp_path)}
    cannot_start = [
        # no interpreter anywhere
        (
            _repinned(root, [tmp_path / "nowhere"]) / "bin" / "memkit-recall",
            empty,
            ("--search", "x"),
        ),
        # cannot locate the tree
        (_half_delivered(tmp_path, "memkit-recall", library=False) / "bin"
         / "memkit-recall", shimmed(), ("--search", "x")),
        # the library present, the hook file absent
        (_half_delivered(tmp_path, "memkit-recall", library=True) / "bin"
         / "memkit-recall", shimmed(), ("--search", "x")),
    ]
    for wrapper, env, args in cannot_start:
        out = _run(wrapper, *args, env={"HOME": str(tmp_path), **env})
        assert out.returncode == hook.EXIT_CANNOT_START, (
            wrapper, out.returncode, out.stderr
        )
    # And the one branch that really is a wrong invocation keeps saying so.
    # A config, because the wrapper reaches the shim through the field an
    # install records rather than through a lookup — and without one, the real
    # interpreter would answer and report inertness, which is a third thing.
    config = _config_file(tmp_path / "recall.json")
    bare = _run(
        root / "bin" / "memkit-recall",
        env=shimmed(CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config)),
    )
    assert bare.returncode == hook.EXIT_ERROR, bare.stderr


def test_arguments_to_the_hook_wrapper_are_ignored_not_forwarded(
    root, shimmed
) -> None:
    """The second of two independent guards on the same failure. hooks.json
    passing zero arguments is pinned above; this is what holds if that pin is
    ever edited away, because a forwarded `--search` turns the every-prompt
    hook into a CLI whose argparse exits 2 — a blocked turn, every turn.
    """
    env = shimmed()
    decided = _decide(root, "memkit-hook", env, "--search", "anything")
    assert decided.handoff[1:] == [
        str(root / "src" / "memkit" / "memory_prompt_recall.py")
    ], decided.handoff
    assert "ignoring 2 argument" in decided.stderr


def test_the_wrapper_resolves_its_tree_through_a_doubled_separator(
    root, tmp_path, shimmed
) -> None:
    """The harness expands `${CLAUDE_PLUGIN_ROOT}` WITH a trailing slash
    (measured on 2.1.238), so the registration hands the wrapper
    `<root>//bin/memkit-hook`. Everything the wrapper builds is derived from
    that string.

    Read off the hook path the wrapper hands its interpreter, because that is
    the derived value the whole tree is built from — equality rather than a
    "//" search, so a derivation that normalized the separator and landed in
    the wrong tree is caught too.
    """
    doubled = f"{root}//bin/memkit-hook"
    out = subprocess.run(
        [SH, "-x", doubled], capture_output=True, text=True, timeout=60,
        env=shimmed(), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr
    handoff = Decided(out).handoff
    assert handoff[1:] == [
        str(root / "src" / "memkit" / "memory_prompt_recall.py")
    ], handoff


def test_what_argv0_a_shebang_script_receives_is_measured_not_assumed(
    tmp_path,
) -> None:
    """The `$0` derivation rests on a kernel behaviour, and the measurement
    behind it was taken on darwin.

    The adopters are on Linux, and this repo's CI runs there — so the claim is
    re-measured in the environment that matters rather than carried over. If
    Linux ever passed argv[0] instead of execve's pathname, the wrappers'
    slashed branch would stop being the one a PATH lookup takes and the
    `command -v` fallback would become load-bearing without anything saying so.

    Stated as an EITHER-OR rather than a platform assertion: both answers are
    survivable — the wrappers handle a bare `$0` too — and what must not happen
    is the code believing one while the kernel does the other.
    """
    probe = tmp_path / "bin"
    probe.mkdir()
    _shim(probe, "argv0-probe", 'printf "%s" "$0"')
    env = {"PATH": f"{probe}:{os.environ['PATH']}", "HOME": str(tmp_path)}

    by_path_lookup = subprocess.run(
        ["argv0-probe"], capture_output=True, text=True, timeout=60, env=env
    ).stdout
    handed_to_sh = subprocess.run(
        [SH, "argv0-probe"], capture_output=True, text=True, timeout=60,
        env=env, cwd=str(probe),
    ).stdout

    # A PATH lookup: whichever the kernel does, the wrappers cover it — the
    # slashed branch walks up from the directory, and the slashless one goes
    # through `command -v`. What this pins is that ONE of them is exercised.
    assert by_path_lookup in (str(probe / "argv0-probe"), "argv0-probe"), (
        by_path_lookup
    )
    # Handed to `sh` by name, there is no pathname to substitute, so this is
    # the branch the bare-argv[0] cases below drive on every platform.
    assert handed_to_sh == "argv0-probe", handed_to_sh
    # And the docstring's claim about a PATH lookup, recorded where a change
    # would be read rather than inferred.
    assert ("/" in by_path_lookup) == (
        by_path_lookup == str(probe / "argv0-probe")
    )


@pytest.mark.parametrize(
    "wrapper,args",
    [
        ("memkit-hook", ()),
        ("memkit-recall", ("--search", "flange torque")),
        ("memkit", ("doctor",)),
    ],
)
def test_a_wrapper_invoked_by_name_from_the_path_still_finds_its_tree(
    root, tmp_path, shimmed, wrapper, args
) -> None:
    """`bin/` is on the agent's PATH while the plugin is enabled, so a bare
    `memkit …` has to find its own tree with no directory in argv[0] to walk up
    from.

    All THREE wrappers, because the derivation is hand-duplicated in each and
    covering one covered one: measured, deleting the `command -v` branch from
    `bin/memkit-hook` and from `bin/memkit` left the whole suite green. It
    matters most for `bin/memkit`, which is the one an agent really does type
    bare, and where a broken derivation is an exit 1 with "cannot locate the
    plugin tree" rather than anything about memkit.

    Run through `sh <name>` from the wrapper's own directory, and that is the
    only way to produce the case rather than a convenience. MEASURED on this
    platform: a shebang script executed through a PATH lookup receives the
    RESOLVED path as `$0` — the kernel passes execve's pathname to the
    interpreter, not argv[0] — so `subprocess.run(["memkit-recall", …])`
    exercises the slashed branch and says nothing about this one. Handing the
    name to `sh` directly is what leaves `$0` bare.

    The tree the wrapper resolved is the whole assertion: a `command -v` that
    answered with a different install on the same PATH would run that install's
    files, which is the wrong-tree failure this derivation exists to avoid, and
    it is invisible in an exit code.

    Read off the SHIM, through a config, rather than off `sh -x`. Two things
    the trace cannot do: it cannot carry the word boundaries of
    `--search "flange torque"` on a shell whose xtrace does not quote, and it
    left the case reaching for whichever pinned system python the host
    happened to have — which is none of them inside a Linux build sandbox.
    Recording an interpreter is what an install does, so the case reaches its
    own python the way an adopter's does.
    """
    env = shimmed(
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(_config_file(tmp_path / "byname.json"))
    )
    env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
    out = subprocess.run(
        [SH, wrapper, *args],
        capture_output=True, text=True, timeout=60, env=env,
        cwd=str(root / "bin"), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == 0, out.stderr
    handed = shimmed.argv()
    if wrapper == "memkit":
        # The dispatcher runs the package rather than the hook file, so the
        # tree shows up as the PYTHONPATH it prepends.
        assert handed == ["-m", "memkit.cli", "doctor"], handed
        assert shimmed.read()["PYTHONPATH"] == str(root / "src"), shimmed.read()
    else:
        hook_file = root / "src" / "memkit" / "memory_prompt_recall.py"
        assert handed == [str(hook_file), *args], handed


def test_the_search_wrapper_refuses_rather_than_blocking_on_stdin(root, shimmed):
    """No arguments is not "search for nothing" — it is the hook's payload
    mode, where the file blocks on stdin for a JSON payload that is never
    coming. An agent that ran it bare would hang until something killed it,
    with no output to explain why."""
    out = _run(root / "bin" / "memkit-recall", env=shimmed())
    assert out.returncode == 2
    assert "--search" in out.stderr
    assert not shimmed.out.exists(), "the interpreter should not have been reached"


# --- the dispatcher's checker route ------------------------------------------


def test_the_dispatcher_runs_the_package_from_this_tree(root, shimmed) -> None:
    """PREPENDED, not assigned — and the difference is only visible when there
    is something to prepend to.

    With no `PYTHONPATH` in the environment the two are the same string, so the
    assertion held for a wrapper that had stopped preserving what the caller
    set. An adopter with a pip-installed memkit of another version is exactly
    who that costs: their `PYTHONPATH` disappearing takes their own packages
    with it.
    """
    inherited = "/opt/an/adopters/own/packages"
    out = _run(
        root / "bin" / "memkit", "doctor",
        env=shimmed(PYTHONPATH=inherited), shell_trace=True,
    )
    assert Decided(out).handoff[1:] == ["-m", "memkit.cli", "doctor"], out.stderr[-400:]
    assert f"PYTHONPATH={root / 'src'}:{inherited}" in out.stderr, out.stderr[-400:]
    assert "MEMKIT_PLUGIN=1" in out.stderr

    # And with nothing inherited it is just this tree, or the wrapper is
    # prepending to an empty string and leaving a stray separator.
    plain = _run(root / "bin" / "memkit", "doctor", env=shimmed(), shell_trace=True)
    assert f"\n+ PYTHONPATH={root / 'src'}\n" in plain.stderr, plain.stderr[-400:]


def test_the_dispatcher_exports_no_checker_route_for_anything_to_read(
    root, tmp_path, shimmed
) -> None:
    """The three cases this replaces asserted the VALUES of two variables the
    dispatcher exported — a route name and a whitespace-joined argv — and the
    second of those was a command a subcommand then ran. An environment
    variable choosing the code that runs is not a thing to validate; it is a
    channel, and this asserts it is closed.

    The probe that produced them is gone too: it EXECUTED each candidate python
    it found, over a PATH the session steers, before any python-side rule
    existed to have an opinion.
    """
    shim = _shim(shimmed.dir, "python3", SHIM_BODY)
    out = _run(_repinned(root, [shim]) / "bin" / "memkit", "doctor", env=shimmed())
    assert out.returncode == 0, out.stderr
    seen = shimmed.read()
    for gone in ("MEMKIT_CHECKER_ROUTE", "MEMKIT_CHECKER_CMD"):
        assert seen[gone] == "<unset>", (gone, seen[gone])
    # ANTI-VACUITY: the recorder really does report a variable that IS set, so
    # `<unset>` above is an observation and not a broken shim.
    assert seen["MEMKIT_PLUGIN"] == "1", seen["MEMKIT_PLUGIN"]


def test_the_dispatcher_refuses_by_name_when_nothing_can_run_it(
    root, tmp_path
) -> None:
    """Unlike the hook wrapper, this one exits non-zero: there is no prompt to
    get out of the way of, and a CLI that reports success while running nothing
    is the false green the whole observation surface exists to prevent. Not 2,
    which means "you invoked this wrongly" — an agent reading that retries with
    different arguments against a machine that cannot run memkit at all.
    """
    from memkit.cli import EXIT_NO_RUNTIME, EXIT_NOT_IN_BUILD, EXIT_USAGE

    bare = _repinned(root, [tmp_path / "nowhere"])
    out = _run(bare / "bin" / "memkit", "doctor", env={"PATH": str(tmp_path / "none")})
    assert out.returncode == EXIT_NO_RUNTIME
    assert "no interpreter is recorded" in out.stderr, out.stderr

    # Every non-zero code this wrapper can produce, against the table an agent
    # reads. A shell script is the one place a new exit code can appear with
    # nothing to look it up in, and the two codes it must never borrow are
    # already spoken for by the dispatcher it fronts. Through the same
    # default-deny helper as the other two wrappers, so this copy cannot go on
    # believing a narrower regex.
    codes = _exit_literals("memkit") - {0}
    assert codes == {EXIT_NO_RUNTIME}, codes
    assert EXIT_NO_RUNTIME not in (EXIT_USAGE, EXIT_NOT_IN_BUILD)


# --- the commands this channel tells an agent to run -------------------------

# A token that could be a command an agent types. Path-shaped and
# dotted-filename tokens are dropped on purpose: `~/.cache/memory-recall/` is a
# directory this channel really does use and `memkit.json` is a file, and
# neither is something anybody runs.
# A token that could be a command an agent types. Every command any channel
# ships is either the bare word `memkit` or a hyphenated `mem…-…`, so requiring
# that shape drops the English words this text is full of (`memory`,
# `memories`) STRUCTURALLY rather than by naming them. Path-shaped and
# dotted-filename tokens are dropped too: `~/.cache/memory-recall/` is a
# directory this channel really uses and `memkit.json` is a file.
COMMANDISH = re.compile(r"(?<![\w./-])(memkit|mem[a-z0-9]+-[a-z0-9-]+)(?![\w./-])")

# The one hyphenated `mem…` token that is not a command, DERIVED rather than
# listed: it is the frame's XML tag, and the emitter is where that fact lives.
#
# Derived because a hand-kept exception list is the cheapest way to silence a
# real failure here — demonstrated: making the dispatcher advertise
# `memkit-init` turns the case below red, and one line added to a list turns it
# green with the bad advice still printed. There is nothing to add a line to
# now, and the equality below says so out loud.
# The frame's delimiter, which is not a command. Matched by STEM because both
# frames now suffix it with a per-run nonce, so the exception cannot be a
# literal without silently ceasing to excuse the thing it is for.
NOT_A_COMMAND = re.compile(rf"{re.escape(hook.FRAME_TAG)}(-[0-9a-f]+)?")


def _corpus(tmp_path: Path, **extra) -> Path:
    """A store with more matching memories than one pointer block can carry, so
    the truncation notice — the one actionable line memkit emits — is rendered.
    """
    corpus = tmp_path / "store" / "search"
    corpus.mkdir(parents=True)
    for n in range(hook.MAX_HITS + 3):
        (corpus / f"flange_torque_{n}.md").write_text(
            f"---\ndescription: Flange fastener {n} tightens in a star pattern, "
            "in three passes, to the torque the table gives.\ntype: reference\n"
            f"---\n\n# Flange torque {n}\n\nThree passes, star pattern.\n"
        )
    return _config_file(
        tmp_path / "memkit.json",
        roots={"home": {"kind": "path", "path": str(tmp_path)}},
        stores=[{"id": "s", "role": "personal", "dir": "store", "live_root": "home"}],
        **extra,
    )


def _surfaces(
    root: Path, tmp_path: Path, config: Path | None, broken: Path
) -> dict[str, str]:
    """Every surface this channel renders, driven as the agent would reach it.

    Real processes through the real wrappers, with a real interpreter: what is
    under test is the name a plugin adopter is handed, and the wrappers are what
    make this channel a channel at all.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        # `None` is the state between install and init: the option names a
        # path that is not there yet, which is what an adopter's install
        # command produces before anything has written the file.
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config or tmp_path / "not-yet.json"),
    )
    env.pop("MEMKIT_CONFIG", None)
    recall = root / "bin" / "memkit-recall"
    dispatcher = root / "bin" / "memkit"
    query = "flange fastener tightening star pattern passes torque"

    def run(wrapper: Path, *args: str, **extra: str) -> str:
        out = subprocess.run(
            [str(wrapper), *args], capture_output=True, text=True, timeout=120,
            env={**env, **extra}, stdin=subprocess.DEVNULL, cwd=str(tmp_path),
        )
        return out.stdout + out.stderr

    # The truncation notice lives on the HOOK path, not the search CLI: the CLI
    # returns everything it found, and the notice exists because the block a
    # prompt gets is capped.
    hook_out = subprocess.run(
        [str(root / "bin" / "memkit-hook")],
        input=json.dumps({"session_id": "surfaces", "prompt": query}),
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path),
    )

    return {
        # The only line in an injected block that tells the agent to do
        # something.
        "truncation": hook_out.stdout + hook_out.stderr,
        "help": run(recall, "--help"),
        "usage error": run(recall, "--nope"),
        "debug-config": run(recall, "--debug-config"),
        "config error": run(
            recall, "--search", query,
            CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(broken),
        ),
        "inert": run(
            recall, "--search", query,
            CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(tmp_path / "absent.json"),
        ),
        "dispatcher help": run(dispatcher, "--help"),
        "dispatcher init manifest": run(dispatcher, "init", "--dry-run"),
    }


def test_every_command_this_channel_prints_is_one_it_ships(root, tmp_path) -> None:
    """The invariant, over the SET of surfaces rather than over the one that
    was wrong.

    A command memkit prints as a next step has to resolve on the caller's PATH
    and has to resolve to THIS install. A plugin install ships no
    `memory-recall`, so an agent following that advice got exit 127 — and on a
    machine that also has a pip or nix memkit it got the other install's
    stores, which is the collision the distinct name exists to prevent.

    Scraped rather than compared against a list of strings, so a surface that
    starts printing a command name later is covered without an edit here.
    """
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    shipped = {
        entry.name
        for entry in (root / "bin").iterdir()
        if entry.is_file() and os.access(entry, os.X_OK)
    }
    assert "memkit-recall" in shipped, shipped
    # The exception set is exactly the frame tag, and nothing may be added to
    # it by hand: an allowlist is the cheapest way to silence a real failure
    # here, and the case would then go on passing by no longer looking at the
    # one thing it is for. Anything else that needs excusing is a defect in the
    # scrape's SHAPE, which is a change somebody has to argue for.
    assert NOT_A_COMMAND.fullmatch(hook.FRAME_TAG), NOT_A_COMMAND.pattern
    assert NOT_A_COMMAND.fullmatch(hook._PROMPT_FRAME_TAG), hook._PROMPT_FRAME_TAG
    excused = {n for n in shipped | {hook.SEARCH_BINARY} if NOT_A_COMMAND.fullmatch(n)}
    assert not excused, excused

    # Three config states, because they reach the name through three different
    # routes and a fix can cover one without the others: a config that omits
    # `search_cli` takes the default applied in `Config.__init__`, one that
    # names it takes the field itself — this is the value the README's own
    # worked example produces — and NO CONFIG AT ALL takes the default applied
    # where the config would have been read.
    #
    # The third is the state the original defect was reported in — a freshly
    # installed plugin, before init — and leaving it out let a channel-aware
    # fix applied at one of the two application points pass: with the override
    # written `if _plugin_install() and cfg is not None`, an unconfigured
    # plugin goes back to advertising the binary it does not ship and the whole
    # suite stays green.
    # A config that raises something `load_config` does NOT convert to
    # ConfigError: `json.load` on a deeply nested document raises
    # RecursionError, and `_config()` catches only ConfigError. That is the
    # state the dispatcher's `except` fallback is reached in, and it was
    # handing plugin adopters the binary their channel does not ship.
    raising = tmp_path / "raising.json"
    raising.parent.mkdir(parents=True, exist_ok=True)
    raising.write_text("[" * 200_000 + "]" * 200_000)
    states = {
        "omitted": _corpus(tmp_path / "omitted"),
        "named": _corpus(tmp_path / "named", search_cli="memory-recall --search"),
        "absent": None,
        "raising outside ConfigError": raising,
    }
    for state, config in states.items():
        surfaces = _surfaces(root, tmp_path / state.split()[0], config, broken)
        named: set[str] = set()
        for surface, text in surfaces.items():
            found = {
                name
                for name in COMMANDISH.findall(text)
                if not NOT_A_COMMAND.fullmatch(name)
            }
            assert found <= shipped, (state, surface, sorted(found - shipped), text)
            named |= found
        # Anti-vacuity at the STATE rather than at each surface: with no
        # config the hook is inert by construction and `--debug-config` reports
        # routes rather than commands, so two surfaces legitimately name none —
        # but a state in which NOTHING names a command is a state this case is
        # not measuring.
        assert named & shipped, (state, sorted(named))
        # The truncation notice specifically: the line whose whole purpose is
        # to be run, and the one a config value used to be able to break. With
        # no config there is no corpus to truncate and no path to name, so what
        # that state pins instead is the dispatcher's fallback — which is where
        # a pre-init adopter actually meets a command name.
        if config is not None and config.name != "raising.json":
            assert "memkit-recall --config " in surfaces["truncation"], state
        else:
            # PRE-INIT, and the command must still name `--config`. A bare
            # `memkit-recall --search` answers `inert`, exit 3, in the shell
            # the dispatcher runs in — and the exit table beside it says exit 3
            # means "no config", which is the one conclusion the `--config`
            # interpolation exists to prevent. There is no path to fill in
            # yet, so it carries the placeholder the README uses.
            #
            # The DISPATCHER HELP is where that lands now. It used to be a
            # pending subcommand's refusal; with both subcommands landed, the
            # description `--help` prints is the surface a pre-init adopter
            # meets a command name on, and it is built by the same helper.
            refusal = surfaces["dispatcher help"]
            assert "memkit-recall" in refusal, refusal
            if config is None:
                # The pre-init state specifically. A config that RAISES cannot
                # know a path to name — `_meanwhile`'s fallback is reached
                # precisely because resolving it failed — so the placeholder is
                # claimed only where there is a path the adopter could supply.
                assert "memkit-recall --config " in refusal, refusal
                assert "memkitConfig" in refusal, refusal
                bare = f"`{hook.PLUGIN_SEARCH_BINARY} --search"
                assert bare not in refusal, refusal


def test_the_advertised_command_runs_from_the_agents_bash_tool(
    root, tmp_path
) -> None:
    """The invariant's second clause, on the channel the command is typed into.

    MEASURED in a live session: a Bash-tool process gets the plugin's `bin/` on
    PATH and NONE of `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA` or
    `CLAUDE_PLUGIN_OPTION_*` — four plugin bin directories were on PATH and no
    plugin variable was set. Both surviving config rungs are plugin env, so a
    bare `memkit-recall --search` there resolves nothing and answers `inert`,
    telling the agent a serving installation is unconfigured. That is the one
    conclusion exit 3 exists to prevent.

    So the command is taken out of the block the hook injected and RUN, in that
    environment, rather than compared to a string. Quoting is part of the
    claim: a config path can contain a space, and a command an agent cannot
    paste is not a command.
    """
    # A space, because quoting is part of the claim — and an INVISIBLE
    # codepoint, because the emission pass strips those: a path quoted here
    # and rewritten there names a file that does not exist, which is a command
    # worse than none because it looks runnable.
    corpus = tmp_path / "spaced dir"
    config = _corpus(corpus)
    hidden = _corpus(tmp_path / "hidden\u200bdir")
    bare = _run(
        root / "bin" / "memkit-recall", "--debug-config",
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path / "home"),
            "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(hidden),
        },
    )
    assert bare.returncode == 0, bare.stderr
    advertised_for_hidden = [
        x for x in bare.stdout.splitlines() if x.startswith("search_cli:")
    ]
    assert advertised_for_hidden == ["search_cli: memkit-recall --search"], (
        advertised_for_hidden
    )
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config),
    )
    env.pop("MEMKIT_CONFIG", None)
    query = "flange fastener tightening star pattern passes torque"
    injected = subprocess.run(
        [str(root / "bin" / "memkit-hook")],
        input=json.dumps({"session_id": "bashtool", "prompt": query}),
        capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path),
    )
    assert injected.returncode == 0, injected.stderr
    # The notice line specifically, found by the reserved prefix that makes it
    # memkit's own — a retrieved description cannot start a line, which is what
    # the frame's carve-out rests on.
    advertised = [
        line.split("search: ", 1)[1]
        for line in injected.stdout.splitlines()
        if line.startswith(hook.NOTICE_PREFIX) and "search: " in line
    ]
    assert advertised, injected.stdout

    # The Bash tool's shape: the plugin's bin on PATH, no plugin environment.
    bash_tool = {
        "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
    }
    assert not [k for k in bash_tool if k.startswith("CLAUDE_PLUGIN_")]
    out = subprocess.run(
        shlex.split(advertised[0]), capture_output=True, text=True, timeout=120,
        env=bash_tool, cwd=str(tmp_path), stdin=subprocess.DEVNULL,
    )
    assert out.returncode == hook.EXIT_OK, (out.returncode, out.stderr, advertised[0])
    assert "flange_torque_" in out.stdout, out.stdout

    # And EVERY backticked command the dispatcher hands out — not the one that
    # was fixed. `--help` is the surface: it is the cheapest probe an agent
    # makes of a fresh install, and every command it prints is one that will be
    # pasted. The pending-subcommand refusals used to be a second such surface
    # and are gone with the last pending name.
    for args in (("--help",),):
        surface = _run(
            root / "bin" / "memkit", *args, env={**env, "PATH": os.environ["PATH"]}
        )
        printed = re.findall(r"`([^`]+)`", surface.stdout + surface.stderr)
        commands = [c for c in printed if c.split()[0].startswith("memkit")]
        assert commands, (args, surface.stdout, surface.stderr)
        for command in commands:
            runnable = command.replace('"<terms>"', '"flange torque"')
            probed = subprocess.run(
                shlex.split(runnable), capture_output=True, text=True, timeout=120,
                env=bash_tool, cwd=str(tmp_path), stdin=subprocess.DEVNULL,
            )
            assert probed.returncode == hook.EXIT_OK, (
                args, runnable, probed.returncode, probed.stdout, probed.stderr
            )


def test_the_scrape_can_see_a_command_this_channel_does_not_ship(tmp_path) -> None:
    """The control for the case above, which would otherwise pass by finding
    nothing. Off the plugin channel the same surfaces name `memory-recall` —
    correctly, since pip and nix install that console script — and the scrape
    sees it."""
    out = subprocess.run(
        ["python3", hook.__file__, "--help"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert "memory-recall" in set(COMMANDISH.findall(out.stdout + out.stderr))


# --- the store guidance ------------------------------------------------------

STORE_DOC = REPO / "docs" / "STORE.md"


def _worked_memory_block(text: str) -> str:
    """The ```markdown fence holding the worked memory FILE.

    Identified by its frontmatter rather than by position: the document has
    other markdown fences — the `CLAUDE.md` import line, the agent block — and
    "the first one" silently became one of those the moment another was added
    above it, which is a test that goes on passing about the wrong text.
    """
    blocks = [
        b for b in re.findall(r"```markdown\n(.*?)```", text, re.S)
        if b.lstrip().startswith("---")
    ]
    assert len(blocks) == 1, f"{len(blocks)} markdown blocks carry frontmatter"
    return blocks[0]


def test_the_worked_memory_in_the_docs_really_surfaces(tmp_path) -> None:
    """The example is executed, not illustrated.

    A worked example is the first thing an adopter copies and the first thing
    to rot: the description cap, the frontmatter keys and the pointer's shape
    are all things this repository changes, and a README that demonstrates a
    memory nobody ever retrieved is worse than none — it fails on their
    machine, where they have no way to tell their store from our example.

    So the file is taken out of the doc, dropped into a scratch store, and the
    real hook is asked the question the doc says to ask it.
    """
    memory = _worked_memory_block(STORE_DOC.read_text(encoding="utf-8"))
    assert memory.lstrip().startswith("---"), memory[:80]

    store = tmp_path / "notes"
    (store / "search").mkdir(parents=True)
    (store / "search" / "postgres-connection-pool.md").write_text(
        memory, encoding="utf-8"
    )
    config = _config_file(
        tmp_path / "memkit.json",
        roots={"notes": {"kind": "path", "path": str(store)}},
        stores=[{
            "id": "notes", "role": "project", "dir": ".",
            "live_root": "notes", "edit_root": "notes",
        }],
    )
    prompt = "why do prepared statements break under pgbouncer transaction pooling"
    out = subprocess.run(
        ["python3", str(REPO / "src" / "memkit" / "memory_prompt_recall.py")],
        input=json.dumps({"session_id": "storedoc", "prompt": prompt}),
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path),
             "MEMKIT_CONFIG": str(config)},
    )
    assert out.returncode == 0, out.stderr
    pointers = [ln for ln in out.stdout.splitlines() if ln.startswith("- ")]
    assert len(pointers) == 1, out.stdout

    # The description the doc shows is the description the agent gets.
    described = re.search(r"^description:\s*(.+)$", memory, re.M)
    assert described, memory[:200]
    assert described.group(1).strip() in pointers[0], (described.group(1), pointers[0])

    # The pointer the doc PRINTS quotes the description the doc SHOWS. Without
    # this the two halves of the example drift apart silently: the file and the
    # assertion both come from the doc, so editing the frontmatter alone keeps
    # this case green while the rendered line goes on quoting the old text.
    shown = STORE_DOC.read_text(encoding="utf-8")
    rendered = next(
        ln for ln in shown.splitlines()
        if ln.startswith("- ~/notes/search/postgres-connection-pool.md")
    )
    assert described.group(1).strip() in rendered, (described.group(1), rendered)

    # And the doc's rendered pointer is not a hand-drawn picture of one: the
    # terms it claims matched are the terms that matched.
    claimed = re.search(r"\[matches (\d+)/(\d+) prompt terms: ([^\]]+)\]", STORE_DOC.read_text(encoding="utf-8"))
    assert claimed, "the doc shows no pointer"
    actual = re.search(r"\[matches (\d+)/(\d+) prompt terms: ([^\]]+)\]", pointers[0])
    assert actual, pointers[0]
    assert claimed.groups() == actual.groups(), (claimed.groups(), actual.groups())


def test_the_release_procedure_is_written_down_and_reachable() -> None:
    """The mechanics are two PRs in an order that is not guessable, and the
    reasoning survived only in review threads until now.

    Pinned by SUBJECT, not by prose: what may not happen is the file losing the
    half that explains WHY two, or the half that says which commit to tag —
    both of which have been got wrong once each.
    """
    note = (REPO / "docs" / "RELEASING.md").read_text(encoding="utf-8")
    for subject in ("two pull requests", "cannot name its own sha", "squash",
                    "marketplace.json", "plugin.json", "from the next release",
                    "gate:shape", "hatch-vcs"):
        assert subject in note, subject
    # The tag goes on the release-state commit, which is the instruction the
    # whole document exists to make unmissable.
    assert re.search(r"[Tt]ag .{0,40}\bS1\b", note), "the tag target is not named"
    # And it is reachable: a procedure nobody can find is one nobody follows.
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "docs/RELEASING.md" in readme, "the README does not link the procedure"


def test_the_store_docs_name_only_commands_this_channel_ships(root) -> None:
    """Every `memkit…` command the store guidance hands out has to exist where
    the reader is standing.

    The guidance is written for a plugin adopter, whose `PATH` carries the
    plugin's `bin/` and nothing else of memkit's — so a command borrowed from
    the pip channel reads as instruction and answers `command not found`. The
    checker is the live trap: it is a console script pip and nix install and
    the plugin does NOT ship, which is why the doc routes it through `uvx`.
    """
    shipped = {
        entry.name
        for entry in (root / "bin").iterdir()
        if entry.is_file() and os.access(entry, os.X_OK)
    }
    assert "memkit-recall" in shipped, shipped
    # A code span never spans a line, and saying so is what keeps a ``` fence
    # from pairing with the inline backticks below it and swallowing the
    # document into one match — which finds nothing and passes.
    def inline(text: str) -> list[str]:
        return re.findall(r"`([^`\n]+)`", text)

    surfaces = {
        "docs/STORE.md": STORE_DOC.read_text(encoding="utf-8"),
        "README.md#your-store": _readme_section("## Your store"),
    }
    for where, text in surfaces.items():
        for command in inline(text):
            head = command.split()[0] if command.split() else ""
            if not COMMANDISH.fullmatch(head):
                continue
            assert head in shipped, (where, command, sorted(shipped))
    # Non-vacuity: the scrape sees the command the guidance really does hand
    # out, so a section that named nothing could not pass quietly.
    found = {
        c.split()[0]
        for c in inline(surfaces["docs/STORE.md"])
        if c.split() and COMMANDISH.fullmatch(c.split()[0])
    }
    assert "memkit-recall" in found, found
    # The checker is named, and only ever behind `uvx --from`.
    doc = surfaces["docs/STORE.md"]
    for match in re.finditer(r"memory-integrity", doc):
        line = doc[doc.rfind("\n", 0, match.start()) + 1 : match.end()]
        assert "uvx --from" in line, line


def test_the_minimal_config_in_the_readme_is_a_working_config() -> None:
    """The four lines the Quick start tells a cold adopter to save.

    It is the first thing they type that can be wrong, and it is printed twice
    — once in Quick start and once leading `## Config`. So it is parsed out of
    the README, written to disk, pointed at a real store, and searched.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    # Two copies: the one Quick start writes with a heredoc, and the one
    # `## Config` prints as JSON. They are the same four lines and must stay
    # so — an adopter who reads both and finds them different has to work out
    # which is current.
    written = re.findall(
        r"cat > [^\n]*memkit\.json <<'EOF'\n(.*?)EOF", readme, re.S
    )
    printed = [
        b for b in re.findall(r"```json\n(.*?)```", readme, re.S)
        if '"schema"' in b and len(b.splitlines()) <= 5
    ]
    assert len(written) == 1, f"{len(written)} heredoc configs in the README"
    assert len(printed) == 1, f"{len(printed)} printed minimal configs"
    assert written[0].strip() == printed[0].strip(), (written[0], printed[0])
    spec = json.loads(written[0])

    with tempfile.TemporaryDirectory() as tmp:
        notes = Path(tmp) / "notes"
        notes.mkdir()
        (notes / "pgbouncer.md").write_text(
            "---\ndescription: PgBouncer in transaction mode breaks "
            "session-scoped features.\n---\n\n# PgBouncer\n\nbody\n"
        )
        # Only the root's path is swapped — every other field is the README's.
        spec["roots"]["notes"]["path"] = str(notes)
        config = Path(tmp) / "memkit.json"
        config.write_text(json.dumps(spec))
        out = subprocess.run(
            ["python3", str(REPO / "src" / "memkit" / "memory_prompt_recall.py"),
             "--config", str(config), "--search", "pgbouncer transaction pooling"],
            capture_output=True, text=True, timeout=60,
            env={"PATH": os.environ["PATH"], "HOME": tmp},
        )
        assert out.returncode == hook.EXIT_OK, (out.returncode, out.stderr)
        assert "pgbouncer.md" in out.stdout, out.stdout
        # And no `/./` in the paths this config makes memkit print. The
        # pointer path is `~`-relative and normalises on its way through that,
        # so the diagnostic is where the raw join shows: it prints the store
        # directory as resolved, and `"dir": "."` joined raw put a `/./`
        # through the middle of it.
        diag = subprocess.run(
            ["python3", str(REPO / "src" / "memkit" / "memory_prompt_recall.py"),
             "--config", str(config), "--debug-config"],
            capture_output=True, text=True, timeout=60,
            env={"PATH": os.environ["PATH"], "HOME": tmp},
        )
        assert diag.returncode == hook.EXIT_OK, diag.stderr
        # The exact directory, with nothing appended: a raw join of `.` prints
        # `<store>/.`, and asserting on a substring like `/./` misses it.
        assert f"store notes: {notes} [" in diag.stdout, diag.stdout
        assert f"corpus:  {notes} —" in diag.stdout, diag.stdout
        assert "/./" not in out.stdout, out.stdout


def test_the_description_limit_the_docs_teach_is_the_one_the_checker_enforces() -> None:
    """Three numbers stand behind this and only one is the author's.

    `docs/STORE.md` tells a person — and the paste-able block tells their agent
    — what length to write to. If that drifts above the checker's cap, every
    memory either fails the check or is written to a length that will be
    rejected, and the doc is the last place anyone would look for the cause.
    """
    from memkit import memory_integrity as checker

    doc = STORE_DOC.read_text(encoding="utf-8")
    assert f"under {checker.MAX_DESC_CHARS} characters" in doc, checker.MAX_DESC_CHARS
    assert f"over **{checker.MAX_DESC_CHARS}**" in doc
    assert f"at most **{hook.DESC_KEEP_CHARS}**" in doc
    # The agent is told the author's number, not the hook's ceiling.
    assert f"under {checker.MAX_DESC_CHARS} characters, that would make me open" in doc
    assert "under 160 characters" not in doc, "the old number survives somewhere"


def test_the_silent_gates_section_names_every_gate_the_hook_applies() -> None:
    """`## Why nothing appeared` is the answer to the most common adopter
    question, and it is only worth having if it is complete.

    The three-word floor is pinned to the constant rather than to prose: it is
    the gate an adopter meets first, because two distinctive words is the
    natural smoke test.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Why nothing appeared")
    section = readme[start : readme.index("\n## ", start + 10)]
    assert f"under {_number_word(hook.MIN_PROMPT_WORDS)} words" in section, (
        hook.MIN_PROMPT_WORDS, section[:400]
    )
    assert f"over {hook.PROMPT_MAX_CHARS} characters" in section, (
        hook.PROMPT_MAX_CHARS, "the paste ceiling has no row"
    )
    for gate in ("already fired this session", "envelope", "disabled",
                 "config: none", "corpus", "session budget",
                 "began with `/`", "all common words", "ran out of time",
                 "installed mid-session"):
        assert gate in section, gate
    # The cross-check command is reachable where the reader is standing, and
    # the section says plainly that the CLI is not subject to the prompt gates.
    assert '"$RECALL" --config <your config> --search' in section
    assert "applies\nfewer gates than the hook" in section


def _number_word(n: int) -> str:
    return {2: "two", 3: "three", 4: "four"}.get(n, str(n))


# Sections that show a reader commands to type in THEIR OWN shell. The plugin
# puts `bin/` on the agent's PATH and nothing on the user's, so a bare plugin
# binary here is a `command not found` handed to somebody who is already
# checking whether the install worked. It has now been written twice.
TERMINAL_SECTIONS = ("## Quick start", "## Why nothing appeared")


def _fenced_command_lines(text: str) -> list[str]:
    """Every line inside a ``` fence that looks like a command being run."""
    out = []
    for block in re.findall(r"\n```[a-z]*\n(.*?)```", text, re.S):
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "{", "}", "-", "|")):
                continue
            # JSON bodies sit in these fences too (the heredoc'd config). A
            # leading quote does NOT disqualify a line — `"$RECALL" …` is the
            # correct form this check exists to require.
            if '":' in stripped or stripped.startswith('"schema"'):
                continue
            out.append(stripped)
    return out


def test_no_reader_facing_section_tells_a_terminal_to_run_a_plugin_binary(
    root,
) -> None:
    """The channel-correct-command invariant, on the surfaces a reader pastes
    from.

    `memkit-recall` exists — it is in the plugin's `bin/` — so the scrape that
    checks a command against what the plugin SHIPS passes it happily. The
    failure is about WHERE: on a plugin install that directory is added to the
    agent's `PATH` and to nothing else, so the same command typed into a
    terminal exits 127. The README says so, hundreds of lines below the paste.

    This is the check that shape needs: in the sections written for someone at
    their own prompt, a plugin binary may only appear reached by path.
    """
    plugin_binaries = {
        entry.name
        for entry in (root / "bin").iterdir()
        if entry.is_file() and os.access(entry, os.X_OK)
    }
    assert "memkit-recall" in plugin_binaries, plugin_binaries

    offenders = []
    for heading in TERMINAL_SECTIONS:
        section = _readme_section(heading)
        for line in _fenced_command_lines(section):
            head = line.split()[0].strip('"')
            if head in plugin_binaries:
                offenders.append((heading, line))
    assert not offenders, offenders

    # Non-vacuity, in both directions. The scrape really does read these
    # sections' command lines, and each section really does hand out a way to
    # run the search — reached by path, through the derivation.
    for heading in TERMINAL_SECTIONS:
        section = _readme_section(heading)
        lines = _fenced_command_lines(section)
        assert lines, heading
        assert any('"$RECALL"' in line for line in lines), heading
        assert 'plugins/cache/memkit/memkit' in section, heading


def test_the_quick_start_sequence_runs_as_printed(tmp_path) -> None:
    """Steps 3 and 4, executed in order in a scratch HOME.

    Both have failed on paste before: step 3 wrote into
    `~/.cache/memory-recall/`, which an unconfigured install deliberately does
    not create, and step 4 put the memory where a later step would strand it.
    A quick start is the one part of a document that has to run.
    """
    home = tmp_path / "home"
    home.mkdir()
    section = _readme_section("## Quick start")
    # The two heredoc blocks are the steps that touch the filesystem; the
    # install and the read-backs need Claude Code and a real install.
    blocks = re.findall(r"\n```\n(mkdir -p .*?)```", section, re.S)
    assert len(blocks) == 2, f"{len(blocks)} filesystem steps in Quick start"
    script = "set -e\n" + "\n".join(blocks)
    ran = subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(home)},
    )
    assert ran.returncode == 0, (ran.returncode, ran.stderr, script)

    # And the store it just built answers the question the step says to ask.
    # Wherever the section says to put it — read out of the heredoc rather than
    # hardcoded, so moving the recommended location cannot leave this asserting
    # about the old one.
    written = re.search(r"cat > (\S*memkit\.json) <<'EOF'", section)
    assert written, section[:400]
    config = Path(written.group(1).replace("~", str(home), 1))
    assert config.is_file(), sorted(str(x) for x in home.rglob("*.json"))
    out = subprocess.run(
        ["python3", str(REPO / "src" / "memkit" / "memory_prompt_recall.py"),
         "--config", str(config), "--search",
         "why do prepared statements break under pgbouncer transaction pooling"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(home)},
    )
    assert out.returncode == hook.EXIT_OK, (out.returncode, out.stdout, out.stderr)
    # The pointer the top of the README promises — same file, same section tag.
    assert "postgres-connection-pool.md" in out.stdout, out.stdout
    assert "[section: PgBouncer transaction mode]" in out.stdout, out.stdout
    # The memory lands where an agent following STORE.md would also write, so
    # the two recipes compose without taking retrieval away.
    assert (home / "notes" / "search" / "postgres-connection-pool.md").is_file()


@pytest.mark.parametrize(
    "module,prog",
    [("memkit.memory_integrity", "memory-integrity"),
     ("memkit.eval_memory_recall", "memory-eval")],
)
def test_the_two_checker_help_surfaces_are_readable(module, prog) -> None:
    """`--help` is the only documentation these two commands have.

    Both passed the module docstring to argparse's default formatter, which
    reflows it: a layout table became one run-on paragraph, and the design
    rationale written for the next maintainer was printed to whoever asked for
    help. The other two entry points already use the raw formatter.
    """
    out = subprocess.run(
        ["python3", "-m", module, "--help"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert f"usage: {prog}" in text, text[:200]
    # Structure survived. The default formatter collapses every authored line
    # break, so a description that still has paragraphs is the whole check.
    described = text[text.index("\n\n") : text.index("options:")]
    assert described.count("\n\n") >= 2, described
    # It says what the exit codes mean, which is what "errors teach" needs from
    # a command whose whole output is a verdict.
    assert "exit codes:" in text, text
    assert "\n  0  " in text and "\n  1  " in text, text
    # And it is not the module docstring: that text is for the next maintainer.
    for maintainer_only in ("uv run --script", "deliberate one-file break",
                            "KTD", "shebang the rest of this author"):
        assert maintainer_only not in text, maintainer_only


def test_the_verification_block_checks_the_installed_path() -> None:
    """The check has to exercise the config the HOOK reads.

    Checking the path the reader meant to install reports a healthy store while
    the hook is inert, and a one-character typo in `memkitConfig` is the
    likeliest install mistake there is — the one state where a green light is
    worse than no light.
    """
    for heading in ("## Quick start", "## Why nothing appeared"):
        section = _readme_section(heading)
        if '"$RECALL"' not in section:
            continue
        for line in _fenced_command_lines(section):
            if not line.startswith('"$RECALL"'):
                continue
            # A literal path here is the defect: it is the reader's intention,
            # not the installed option.
            assert "/.config/" not in line and "/.cache/" not in line, line
            assert "--config" in line, line
    quick = _readme_section("## Quick start")
    assert "pluginConfigs" in quick, "the block never reads settings.json back"
    assert '--config "$MEMKIT_CFG"' in quick, quick[-900:]


def test_the_config_location_is_not_a_cache_directory() -> None:
    """The config is the one file in this design that nothing regenerates.

    `memkit init` can write one again, but only where the adopter consents to
    it a second time — and the README tells the reader, two sections from where
    the file used to live, that everything under `~/.cache/memory-recall/` is
    disposable. A config under a purged cache directory is an install that goes
    inert on a cache clear with nothing on screen to say why.
    """
    default = _json(PLUGIN_MANIFEST)["userConfig"]["memkitConfig"]["default"]
    assert "/.cache/" not in default, default
    quick = _readme_section("## Quick start")
    written = re.search(r"cat > (\S*memkit\.json) <<'EOF'", quick)
    assert written, quick[:400]
    assert "/.cache/" not in written.group(1), written.group(1)
    # The manifest offers what the page tells you to create, so an adopter who
    # takes the default and one who follows the page land in the same place.
    assert written.group(1).replace("~", "") == default.replace("~", ""), (
        written.group(1), default
    )


# --- what the inert message says a config can arrive by ----------------------

# One phrase per rung `memkit_resolve_config` really tries. The mapping is the
# only handwritten link in the chain: the rungs are scraped from the shell and
# the phrases are read out of the module, so the two ends cannot be edited into
# agreement through this table without someone editing this table too.
# One phrase per CANDIDATE PATH `memkit_resolve_config` really builds, keyed on
# the shell expression that builds it — and the scrape below additionally pins
# that `_candidate` is the function's only sink, so a rung that skipped the
# variable entirely cannot serve a config unseen.
#
# The mapping is the only handwritten link in the chain: the expressions are
# scraped from the shell and the phrases are read out of the module.
ROUTE_FOR_RUNG = {
    '$(memkit_expand_home "$CLAUDE_PLUGIN_OPTION_MEMKITCONFIG")':
        "the `memkitConfig` install option",
    '$(memkit_expand_home "$CLAUDE_PLUGIN_DATA")/memkit.json':
        "$CLAUDE_PLUGIN_DATA/memkit.json",
}


def _resolver_rungs() -> set[str]:
    """Every candidate path `memkit_resolve_config` tests, as written.

    `_candidate=` is the resolver's one shape for "a path this rung might
    serve": each rung assigns it and then `[ -f ]`s it. Anything assigned there
    and not in ROUTE_FOR_RUNG is an admission route nobody has classified, and
    the message that enumerates the routes is stale the moment one appears.
    """
    text = COMMON_SH.read_text(encoding="utf-8")
    match = re.search(r"^memkit_resolve_config\(\) \{$(.*?)^\}$", text, re.S | re.M)
    assert match, "memkit_resolve_config moved — this pin cannot see it"
    body = match.group(1)
    # Every value the function can PRINT is a route it serves, and the scrape
    # has to start there rather than at the assignments: a rung written
    # `if [ -f "$HOME/.memkit.json" ]; then printf '%s\\n' "$HOME/.memkit.json";
    # return 0; fi` assigns nothing and is served all the same. Measured — the
    # suite stayed green with exactly that rung in place.
    #
    # So `_candidate` must be the only sink, and then classifying the
    # assignments classifies the routes.
    # EQUALITY, not subset, which is what anchors this on a non-empty match:
    # the regex is keyed on one spelling, so a rung written with `echo` printed
    # nothing this could see and the subset held vacuously. A rung spelled
    # `if [ -f "$HOME/.memkit.json" ]; then echo "$HOME/.memkit.json"; fi` is a
    # live admission route reachable from any home directory, and it left the
    # whole file green.
    # Statements redirected to STDERR are not sinks: the resolver's answer is
    # what it writes to stdout, and a refusal message written the same way is
    # not an admission route. Removed before scraping, continuations included.
    to_stdout = re.sub(r"printf(?:\\\n|[^\n])*?>&2", "", body)
    printed = set(re.findall(r"""printf\s+'%s\\n'\s+(\S+)""", to_stdout))
    assert printed == {'"$_candidate"'}, sorted(printed)
    # And no OTHER way of writing to stdout, since the equality above only
    # constrains the spelling it can see.
    for other in ("echo ", "printf '%s'", "cat ", "tee ", ">&1"):
        assert other not in to_stdout, (other, to_stdout)
    # An assignment counts wherever the line puts it — a rung written
    # `… || _candidate=<expr>` is a rung.
    candidates = set(
        re.findall(r"^\s*(?:\|\||&&)?\s*_candidate=(\S.*?)\s*$", body, re.M)
    )
    # An empty assignment is the rejection arm of the absoluteness guard, not a
    # route: `_candidate=""` is how a non-absolute value is dropped.
    return {c for c in candidates if c not in ('""', "''")}


def test_the_inert_message_names_the_rungs_the_resolver_actually_tries() -> None:
    """The rungs live in POSIX sh and the sentence that describes them lives in
    Python, with nothing between them.

    A rung deleted there used to leave a confident sentence here — telling an
    agent to configure an install through a route the code no longer has, on
    the one surface whose whole job is to say why nothing is happening. Set
    equality in both directions, so a rung added is as red as a rung removed.
    """
    rungs = _resolver_rungs()
    assert rungs == set(ROUTE_FOR_RUNG), (sorted(rungs), sorted(ROUTE_FOR_RUNG))
    # The basename is part of the route, not decoration: the message tells an
    # adopter which FILE to create.
    assert "memkit.json" in "".join(rungs), rungs
    expected = {"--config PATH"} | {ROUTE_FOR_RUNG[rung] for rung in rungs}
    assert set(hook.PLUGIN_CONFIG_ROUTES) == expected, hook.PLUGIN_CONFIG_ROUTES


# Phrases that described the config rung this repo deleted. A tombstone rather
# than a derivation: the rung is gone from the shell, so nothing can scrape it
# out of the resolver, and the only way a document naming it goes red is a list
# of the words it was described with. Both documents that enumerate routes
# carried one of these after the code stopped honouring it.
RETIRED_ROUTE_PHRASES = (
    "beside the plugin",
    "one beside",
    "in the plugin's own directory",
    "memkit.json beside",
)

ROUTE_DOCS = ("README.md", "docs/ROLLOUT.md")


def test_the_soak_logs_growth_rule_is_published_where_a_consumer_reads_it() -> None:
    """The outcome vocabulary is now a cross-repo contract: another
    repository's analyzers compute injection rates from `log.jsonl`, and its
    suite asserts every outcome memkit can emit has been classified.

    So the rule a consumer codes against has to be in writing, not implied by
    the producer's behaviour — and the discriminator in particular, since
    without it the only way to exclude a non-prompt record from a per-prompt
    population is to learn each new outcome's NAME, which is the coupling the
    static enumeration exists to remove.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert '"concludes": false' in readme, "the discriminator is undocumented"
    assert "grows without a version bump" in readme
    assert "prompt_sha" in readme


def test_the_rollout_runbook_verifies_both_channels() -> None:
    """The per-host checks read `~/.claude/hooks`, a `/nix/store` symlink and a
    consumer checkout — none of which a plugin install has — so on a host that
    installed memkit as a plugin every one of them fails or passes vacuously.

    That is the silent-failure mode the runbook's own opening says it exists to
    prevent, and README sends every second-host adopter there with no channel
    caveat.
    """
    rollout = (REPO / "docs" / "ROLLOUT.md").read_text(encoding="utf-8")
    assert "## Per-host verify, plugin channel" in rollout
    # The commands that block was written from, each run against a real
    # install before it was written down.
    for command in (
        "claude plugin list",
        "claude plugin details memkit@memkit",
        "pluginConfigs",
        "--debug-config",
        # Where derived state lands, in the form that is right on a Linux
        # workstation as well as a mac.
        "${XDG_CACHE_HOME:-$HOME/.cache}/memory-recall/",
    ):
        assert command in rollout, command
    # And the plugin block says the nix sections are not for this adopter, who
    # has no darwin-rebuild, no flake input and no consumer checkout.
    plugin_block = rollout.split("## Per-host verify, plugin channel", 1)[1]
    assert "NIX-FLEET rollout" in plugin_block
    # And the nix block says which channel it is for, so a plugin adopter does
    # not work through four checks that cannot apply.
    assert "nix-channel checks" in rollout


def test_no_document_still_offers_the_config_route_the_code_dropped() -> None:
    """An operator who follows a runbook naming a deleted route drops a
    `memkit.json` into the payload root and gets a plugin that installs,
    reports enabled and serves nothing — with no error anywhere, which is the
    silent failure the runbook exists to prevent.

    Both documents that enumerate the routes are checked, because the round
    that deleted the rung rewrote one of them and missed the other, leaving the
    repo shipping two answers to "which paths will an every-prompt hook read".
    """
    for name in ROUTE_DOCS:
        text = (REPO / name).read_text(encoding="utf-8").lower()
        for phrase in RETIRED_ROUTE_PHRASES:
            assert phrase not in text, f"{name} still offers: {phrase!r}"
        # And each still names the rung that IS there, or the tombstone above
        # would pass on a document that stopped describing routes at all.
        assert "$claude_plugin_data" in text or "plugin's own data directory" in text, name


def test_both_channels_inert_messages_name_only_their_own_routes(
    root, tmp_path
) -> None:
    """Both branches, explicitly, because the suite runs under one of them.

    With `MEMKIT_PLUGIN` unset the plugin wording is the untested branch and
    vice versa, and the defect this replaced was exactly a message that was
    right for the channel the tests happened to run on: a plugin install told
    an agent to set `$MEMKIT_CONFIG`, which both wrappers strip before the hook
    sees it, so following the advice measurably changed nothing.
    """
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    through_plugin = subprocess.run(
        [str(root / "bin" / "memkit-recall"), "--search", "flange torque"],
        capture_output=True, text=True, timeout=120, env=env,
        stdin=subprocess.DEVNULL,
    )
    assert through_plugin.returncode == hook.EXIT_INERT, through_plugin.stderr
    assert hook.CONFIG_ENV not in through_plugin.stderr, through_plugin.stderr
    for route in hook.PLUGIN_CONFIG_ROUTES:
        assert route in through_plugin.stderr, route

    direct = subprocess.run(
        ["python3", hook.__file__, "--search", "flange torque"],
        capture_output=True, text=True, timeout=120, env=env,
        stdin=subprocess.DEVNULL,
    )
    assert direct.returncode == hook.EXIT_INERT, direct.stderr
    assert "CLAUDE_PLUGIN" not in direct.stderr, direct.stderr
    for route in hook.CONFIG_ROUTES:
        assert route in direct.stderr, route

    # And `--help`, which is the cheapest probe an agent runs and therefore the
    # first place it learns what to try. It told a plugin adopter the config
    # default was `$MEMKIT_CONFIG` and, on `--dir`, to unset it — a variable
    # both wrappers strip, so the first claim is false and following the second
    # changes nothing.
    plugin_help = _run(root / "bin" / "memkit-recall", "--help", env=env)
    assert plugin_help.returncode == 0, plugin_help.stderr
    assert hook.CONFIG_ENV not in plugin_help.stdout, plugin_help.stdout
    for route in hook.PLUGIN_CONFIG_ROUTES:
        assert route in plugin_help.stdout, route
    direct_help = subprocess.run(
        ["python3", hook.__file__, "--help"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert f"${hook.CONFIG_ENV}" in direct_help.stdout, direct_help.stdout


def test_a_whitespace_only_search_cli_does_not_take_down_the_dispatcher(
    root, tmp_path
) -> None:
    """`_parser()` calls `_meanwhile` while BUILDING its description, so
    anything that raises there takes down every `memkit` invocation — including
    `--help`, the cheapest probe an adopter or an agent runs, and the one the
    dispatcher's docstring says nothing about the config may break.

    A whitespace-only `search_cli` is truthy, so `Config` keeps it and
    `split()` returns nothing to index.
    """
    real = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    for value in ("   ", "\t", " \n "):
        config = _config_file(tmp_path / "ws.json", search_cli=value)
        for args in (("--help",), ("doctor", "--check", "platform")):
            out = _run(
                root / "bin" / "memkit", *args,
                env={**real, "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(config)},
            )
            assert "IndexError" not in out.stderr, (value, args, out.stderr)
        # The command NAME is rendered by the description, which is what every
        # invocation builds and what `--help` prints. A whitespace-only value
        # is truthy, so the config keeps it and the default is never applied —
        # and a description that interpolated it would tell an agent to run
        # nothing at all, which is worse than naming the wrong binary.
            # And OFF the plugin channel, which is where the value is
            # actually honoured: on the plugin channel the advertised command
            # is this channel's own, so the config's whitespace never reaches
            # the split that raised.
            direct = subprocess.run(
                ["python3", "-m", "memkit.cli", *args],
                capture_output=True, text=True, timeout=120,
                env={
                    **real,
                    "MEMKIT_CONFIG": str(config),
                    "PYTHONPATH": str(REPO / "src"),
                },
            )
            assert "IndexError" not in direct.stderr, (value, args, direct.stderr)

        # The command NAME is rendered by the description, which every
        # invocation builds and which `--help` prints. A whitespace-only value
        # is truthy, so the config keeps it and the default is never applied —
        # and a description that interpolated it would tell an agent to run
        # nothing at all, which is worse than naming the wrong binary.
        helped = _run(
            root / "bin" / "memkit", "--help",
            env={**real, "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(config)},
        )
        assert "memkit-recall" in helped.stdout, (value, helped.stdout)
        off_channel = subprocess.run(
            ["python3", "-m", "memkit.cli", "--help"],
            capture_output=True, text=True, timeout=120,
            env={**real, "MEMKIT_CONFIG": str(config), "PYTHONPATH": str(REPO / "src")},
        )
        assert "memory-recall" in off_channel.stdout, (value, off_channel.stdout)


def test_the_help_epilog_carries_every_exit_code_this_binary_can_produce(
    root, tmp_path
) -> None:
    """The epilog's own comment says it is built from the constants so the help
    and the README cannot drift from what the code returns. It was complete
    before the start-failure code existed and stopped being complete when it
    landed — an agent meeting an undocumented 4 falls back to the nearest
    neighbour or to shell convention, and both readings are wrong in the unsafe
    direction.

    Over the CONSTANTS, so the next code added is covered without an edit here.
    """
    real = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    rendered = _run(root / "bin" / "memkit-recall", "--help", env=real).stdout
    # OVER the constants, which is what the epilog's own comment claims of
    # itself. A hand-written list is the drift it says it prevents, and it is
    # how a code an agent branches on comes to have no row to look up.
    codes = {
        value
        for name, value in vars(hook).items()
        if name.startswith("EXIT_") and isinstance(value, int)
    }
    assert len(codes) >= 5, codes
    listed = {int(m) for m in re.findall(r"^  (\d+)  ", rendered, re.MULTILINE)}
    assert listed == codes, (listed, codes)
    # And the collision with the dispatcher's table is stated on both sides,
    # in both directions: the two tables swap 1 and 4, and 1 is the dangerous
    # one — on this table it means "nothing matched", which tells an agent to
    # stop looking.
    from memkit.cli import EXIT_NO_RUNTIME, EXIT_NOT_IN_BUILD

    assert "dispatcher's table is its own" in rendered, rendered
    dispatcher = _run(root / "bin" / "memkit", "--help", env=real).stdout
    assert "swaps these two" in dispatcher, dispatcher
    assert EXIT_NO_RUNTIME == hook.EXIT_NO_MATCH
    assert EXIT_NOT_IN_BUILD == hook.EXIT_CANNOT_START


def test_debug_config_says_when_it_overrode_the_field_it_is_labelled_with(
    root, tmp_path
) -> None:
    """`--debug-config` is the command the README and the rollout runbook both
    name as *the* verification surface, and every line of it reports the file —
    except this one, whose label is the config key verbatim and whose value is
    not from the config. An operator cannot tell the two apart.
    """
    real = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    overridden = _corpus(tmp_path / "over", search_cli="memory-recall --search")
    out = _run(
        root / "bin" / "memkit-recall", "--debug-config",
        env={**real, "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(overridden)},
    )
    assert out.returncode == 0, out.stderr
    assert "! the config's own `search_cli` is not in effect" in out.stdout, out.stdout

    # And NOT where the config never declared the field: there is no value to
    # have been overridden, and the line asserts something false about "the
    # name it records". Measured byte-identical on two configs differing only
    # in whether the key is present.
    undeclared = _corpus(tmp_path / "undeclared")
    silent = _run(
        root / "bin" / "memkit-recall", "--debug-config",
        env={**real, "CLAUDE_PLUGIN_OPTION_MEMKITCONFIG": str(undeclared)},
    )
    assert silent.returncode == 0, silent.stderr
    assert "! the config's own" not in silent.stdout, silent.stdout

    # And no divergence line where there is no divergence, or the note is
    # decoration rather than a report. Off the plugin channel the field IS the
    # advertised command — which is also why the note can never be silent on
    # the plugin channel: the `--config <path>` prefix that makes the command
    # runnable in the agent's Bash tool is not something a config file can
    # carry, so the two always differ there.
    same = subprocess.run(
        ["python3", hook.__file__, "--debug-config"],
        capture_output=True, text=True, timeout=120,
        env={**real, "MEMKIT_CONFIG": str(overridden)},
    )
    assert same.returncode == 0, same.stderr
    assert "search_cli: memory-recall --search" in same.stdout, same.stdout
    assert "! the config's own" not in same.stdout, same.stdout


# --- the hook file the wrapper actually runs ---------------------------------


def test_the_wrapper_execs_the_byte_identical_hook(root, tmp_path) -> None:
    """`_VERSION` is a sha256 of the hook's own bytes and is stamped on every
    soak record, so it is what makes records comparable across install
    channels. A wrapper that copied, patched or wrapped the file — to bake a
    config in, say — would fork the log into halves that no analyzer can join,
    silently: the field would still be there, still eight hex characters.
    """
    corpus = tmp_path / "store" / "search"
    corpus.mkdir(parents=True)
    (corpus / "flange_torque.md").write_text(
        "---\ndescription: Flange fasteners tighten in a star pattern, in three "
        "passes.\ntype: reference\n---\n\n# Flange torque\n\nThree passes.\n"
    )
    config = _config_file(
        tmp_path / "memkit.json",
        roots={"home": {"kind": "path", "path": str(tmp_path)}},
        stores=[{"id": "s", "role": "personal", "dir": "store", "live_root": "home"}],
    )
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        CLAUDE_PLUGIN_OPTION_MEMKITCONFIG=str(config),
    )
    env.pop("MEMKIT_CONFIG", None)
    payload = json.dumps(
        {"session_id": "wrapv", "prompt": "flange fastener tightening passes"}
    )

    through_wrapper = subprocess.run(
        [str(root / "bin" / "memkit-hook")], input=payload, capture_output=True,
        text=True, timeout=60, env=env,
    )
    assert through_wrapper.returncode == 0
    # A pointer, not an exit code: an inert hook and a wired one both exit 0
    # and print nothing on a prompt with no answer.
    assert "flange_torque.md" in through_wrapper.stdout, through_wrapper.stdout

    direct = subprocess.run(
        ["python3", hook.__file__], input=payload, capture_output=True, text=True,
        timeout=60, env=dict(env, MEMKIT_CONFIG=str(config), session="x"),
    )
    assert direct.returncode == 0

    records = [
        json.loads(line)
        for line in (tmp_path / ".cache" / "memory-recall" / "log.jsonl")
        .read_text()
        .splitlines()
    ]
    versions = {r["v"] for r in records}
    assert len(versions) == 1, versions
    assert versions != {"?"}, "the hook could not read itself — the pin is vacuous"


def test_the_plugin_marker_is_absent_without_the_wrapper(tmp_path) -> None:
    """R6's non-degradation premise, from the other side: nothing about a
    plugin install may reach a nix or pip one. The marker is exported by the
    wrapper and by nothing else, so a hook run any other way cannot take a
    plugin-only branch.
    """
    out = subprocess.run(
        ["python3", hook.__file__, "--debug-config"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert out.returncode == hook.EXIT_INERT
    assert "MEMKIT_PLUGIN" not in out.stdout


def test_every_outcome_the_readme_publishes_has_a_reason_doctor_can_render(
) -> None:
    """The two halves of the *Why nothing appeared* triage, pinned together.

    The prose table is the best-tested writing in the project and two
    walkthroughs verified every row; doctor's `gate-outcomes` renders the same
    names as counts, with the same reasons, out of the adopter's own log. A
    name that arrived in one and not the other is a histogram row nobody can
    read, or a documented outcome the mechanized table silently omits — and the
    vocabulary grows without a version bump, so the drift is the normal case
    rather than the exceptional one.

    `dup-registration` is in both, and the two `trust:` outcomes are in
    neither: they are the marker's vocabulary, not the log's.
    """
    from memkit.cli_doctor import OUTCOME_REASONS

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("**The outcome vocabulary.**")
    table = readme[start : readme.index("\n## ", start)]
    published = {
        name
        for row in re.findall(r"^\| (.+?) \|", table, re.M)
        for name in re.findall(r"`([a-z][a-z:-]*)`", row)
        if not name.startswith("trust:")
    }
    assert "injected" in published and "gate:short" in published, sorted(published)
    # EQUALITY, in both directions. A name in the prose and not in the
    # histogram is a documented outcome doctor silently omits; a name in the
    # histogram and not in the prose is a row an adopter meets with no
    # explanation anywhere.
    assert published == set(OUTCOME_REASONS), (
        sorted(published - set(OUTCOME_REASONS)),
        sorted(set(OUTCOME_REASONS) - published),
    )


def test_every_index_outcome_the_emitter_defines_has_a_row_in_the_readme() -> None:
    """The index-state vocabulary's third pin, and the one an adopter reads.

    Same defect as the one above, in a second vocabulary: `truncated` shipped
    with a doctor arm missing and the README carrying it, so the README was
    the superset and the machine-readable half the stale one. Equality in both
    directions, derived from the emitter's constants — a name in the code and
    not in the prose is an outcome an adopter meets with no explanation
    anywhere; a name in the prose and not in the code is a state nothing can
    produce.

    `BUILD_SCHEMA` is the record's version rather than an outcome, and it is
    an int, which is the discriminator: an outcome is a string.
    """
    from memkit import memory_prompt_recall as hook

    defined = {
        value
        for name, value in vars(hook).items()
        if name.startswith("BUILD_") and isinstance(value, str)
    }
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("Today it is", readme.index("**Two rules for anything"))
    para = readme[start : readme.index("\n\n", start)]
    published = set(re.findall(r"`([a-z]+)`", para))
    assert defined == published, (
        sorted(defined - published),
        sorted(published - defined),
    )

    # And the one outcome that means "indexed INCOMPLETELY" names BOTH of the
    # causes the emitter can raise it for. It builds two different reason
    # strings under this one name and they send a reader to different places:
    # out of budget converges over the following runs, over the per-file cap
    # never does. A gloss naming only the budget sends the owner of a single
    # oversize memory hunting a corpus that is too large.
    gloss = para[para.index("`truncated`") :]
    gloss = gloss[: gloss.index("`busy`")].lower()
    assert "budget" in gloss, gloss
    assert "cap" in gloss, gloss


# --- the two skills ----------------------------------------------------------

SKILLS = REPO / "skills"


def _frontmatter(path: Path) -> dict:
    """The SKILL.md frontmatter as a flat mapping. Deliberately not a YAML
    parser: these files are hand-written and small, and a dependency to read
    two of them is a dependency in the payload."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    block = text.split("---\n", 2)[1]
    out: dict = {}
    key = ""
    for line in block.splitlines():
        if line.startswith(" ") and key:
            out[key] += " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            out[key] = value.strip()
    return out


def test_both_skills_are_there_and_declare_what_the_harness_reads(root) -> None:
    """`Skills (2)` on a real install, measured on 2.1.241. A skill the harness
    does not register is a command an agent was told to reach for and cannot.

    Against the STAGED payload rather than the repository, because that is what
    an adopter receives: a skill in the working tree and not in the payload
    list is a skill the install does not carry, and `Skills (0)` is what the
    harness then reports.
    """
    for name in ("doctor", "init"):
        assert (root / "skills" / name / "SKILL.md").is_file(), name
        path = SKILLS / name / "SKILL.md"
        assert path.is_file(), path
        front = _frontmatter(path)
        assert front["name"] == name
        assert len(front["description"]) > 80, front["description"]
        # The description is what a model matches on, so it has to say when to
        # reach for it and not only what it is.
        assert "Use " in front["description"], front["description"]


def test_init_is_not_model_invocable_and_doctor_is() -> None:
    """init writes files. A model may not decide to run it, and the harness key
    that enforces that is the one thing standing between a two-turn consent
    handshake and a one-turn mutation."""
    assert _frontmatter(SKILLS / "init" / "SKILL.md")[
        "disable-model-invocation"
    ] == "true"
    assert "disable-model-invocation" not in _frontmatter(
        SKILLS / "doctor" / "SKILL.md"
    )


def test_every_allowed_tools_entry_pins_an_exact_argument_shape() -> None:
    """An open-ended prefix over `memkit` pre-approves `init --confirm` from a
    doctor grant. `:*` is the narrowest form that admits a digest and nothing
    else, and it is used on exactly the one entry that needs it.
    """
    grants = {}
    for name in ("doctor", "init"):
        raw = _frontmatter(SKILLS / name / "SKILL.md")["allowed-tools"]
        grants[name] = [entry.strip() for entry in raw.split("), Bash(")]
    doctor_entries = _frontmatter(SKILLS / "doctor" / "SKILL.md")["allowed-tools"]
    # A prefix over `memkit doctor` and not over `memkit`: it admits `--json`,
    # `--config <path>` and `--check <id>` — the two follow-ups the report's
    # own remedies ask for — and cannot reach `init`, which is the subcommand
    # that writes. An exact-string grant left the one agent-actor remedy that
    # names a next command naming one this skill could not issue.
    assert doctor_entries == (
        "Bash(${CLAUDE_PLUGIN_ROOT}/bin/memkit doctor:*)"
    ), doctor_entries
    assert "init" not in doctor_entries

    init_entries = _frontmatter(SKILLS / "init" / "SKILL.md")["allowed-tools"]
    # ONE entry, and it is the turn that writes nothing.
    #
    # `--confirm` is deliberately absent. A prefix grant over it let the whole
    # handshake happen inside one turn — run the dry-run, read the digest out
    # of the model's own tool result, and apply it with `--wire-claude-md` and
    # `--auto-dream-off` attached — writing to the user's `CLAUDE.md` and
    # `settings.json` with no message they ever saw. The permission prompt on
    # the writing call is the only part of this consent the harness enforces
    # rather than the model observing.
    #
    # The read-only turn IS a prefix match, and the asymmetry used to run the
    # other way: the body tells the agent to pass four flags, and an exact
    # grant dropped it into a prompt for using one of them on the turn that
    # writes nothing.
    assert init_entries == (
        "Bash(${CLAUDE_PLUGIN_ROOT}/bin/memkit init --dry-run:*)"
    ), init_entries
    assert "--confirm" not in init_entries, init_entries
    # No bare prefix anywhere: `Bash(.../memkit:*)` or `Bash(.../memkit *)`
    # would make the doctor grant cover every subcommand this binary has.
    for name, entries in grants.items():
        for entry in entries:
            assert "/bin/memkit doctor" in entry or "/bin/memkit init" in entry, (
                name, entry
            )
            assert not entry.rstrip(")").endswith("/bin/memkit"), (name, entry)


def test_the_doctor_skill_says_to_relay_the_report_rather_than_re_derive_it():
    """The most reliable way to produce a confident wrong answer about an
    install is to attach a plausible summary to a correct report: the reader
    then has two accounts and no way to tell which was measured."""
    body = (SKILLS / "doctor" / "SKILL.md").read_text(encoding="utf-8")
    assert "verbatim" in body
    assert "summarise it" in body
    assert "re-derive" in body
    # The branching rule, in the fields an agent actually reads.
    assert "actor" in body and "terminal" in body
    assert "zero `FAIL`" in body
    for status in doctor.STATUSES:
        assert status in body, status


def test_every_flag_the_init_skill_documents_is_inside_a_grant() -> None:
    """A skill that tells the agent to pass a flag and then leaves it outside
    the pre-approval is a handshake with a permission prompt in the middle of
    it — on the one skill where the two turns are the whole of the consent."""
    body = (SKILLS / "init" / "SKILL.md").read_text(encoding="utf-8")
    grants = [
        entry.strip().removeprefix("Bash(").rstrip(")")
        for entry in _frontmatter(SKILLS / "init" / "SKILL.md")[
            "allowed-tools"
        ].split("), Bash(")
    ]
    prefixes = [g[: -len(":*")] for g in grants if g.endswith(":*")]
    assert prefixes, grants
    documented = set(re.findall(r"^- `(--[a-z-]+)(?: [A-Z]+)?`", body, re.M))
    assert documented >= {"--store", "--config", "--wire-claude-md"}, documented
    for flag in documented:
        # Every documented flag has to be reachable from at least one prefix
        # grant: appended to it, the command is still inside the pattern.
        assert any(
            f"{prefix} {flag}".startswith(prefix) for prefix in prefixes
        ), flag
    # And the read-only turn is one of the prefixes, which is the half that was
    # missing.
    assert any(p.endswith("init --dry-run") for p in prefixes), prefixes


def test_the_init_skill_describes_both_turns_and_the_codes_it_can_return():
    """A two-turn handshake a skill does not describe is one an agent will
    collapse into a single turn."""
    body = (SKILLS / "init" / "SKILL.md").read_text(encoding="utf-8")
    assert "--dry-run" in body and "--confirm" in body
    assert "new message" in body or "new turn" in body or "in a new" in body
    assert "writes nothing" in body
    # And the writing turn says why it will ask, so an agent meeting the
    # prompt does not read it as a misconfiguration to route around.
    assert "not pre-approved" in body
    assert "Do not work around the prompt" in body
    from memkit import cli_init

    for code in (cli_init.EXIT_OK, cli_init.EXIT_USAGE, cli_init.EXIT_REFUSED,
                 cli_init.EXIT_INCOMPLETE):
        assert f"| {code} |" in body, code


def _doctor_remedies() -> list[str]:
    """Every `remedy` string doctor can emit, read out of the source.

    Statically rather than by running it, because a run only reaches the
    states that machine is in — and the remedies that matter most belong to
    the states an adopter's machine is in and this one is not.
    """
    import ast

    tree = ast.parse((REPO / "src" / "memkit" / "cli_doctor.py").read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name != "Check":
            continue
        remedy = None
        if len(node.args) >= 4:
            remedy = node.args[3]
        for keyword in node.keywords:
            if keyword.arg == "remedy":
                remedy = keyword.value
        if remedy is None:
            continue
        # Only the literal parts. An f-string's interpolations are paths and
        # counts, not command names, and this is a check about command names.
        pieces = []
        for part in (
            remedy.values if isinstance(remedy, ast.JoinedStr) else [remedy]
        ):
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces.append(part.value)
        if pieces:
            out.append("".join(pieces))
    return out


def test_no_doctor_remedy_tells_a_terminal_to_run_a_plugin_binary(root) -> None:
    """The channel-correct-command invariant, on the third reader-facing
    surface.

    `memkit-recall` exists — it is in the plugin's `bin/` — so a scrape that
    checks a command against what the plugin SHIPS passes it happily. The
    failure is about WHERE: that directory is added to the agent's PATH and to
    nothing else, so the same command typed into a terminal exits 127. Doctor's
    remedies are relayed to a person, which puts them under the same rule as
    the README's own terminal-facing sections.
    """
    plugin_binaries = {
        entry.name
        for entry in (root / "bin").iterdir()
        if entry.is_file() and os.access(entry, os.X_OK)
    }
    assert "memkit-recall" in plugin_binaries

    remedies = _doctor_remedies()
    # Non-vacuity: there ARE remedies, and enough of them that a scrape reading
    # none would be visibly wrong.
    assert len(remedies) >= 10, len(remedies)
    offenders = []
    for remedy in remedies:
        for command in re.findall(r"`([^`\n]+)`", remedy):
            head = command.split()[0] if command.split() else ""
            if head in plugin_binaries:
                offenders.append((command, remedy))
    assert not offenders, offenders
    # And the scrape really does see backticked commands in remedy text.
    assert any("`" in remedy for remedy in remedies), remedies


def test_every_doctor_check_id_appears_in_the_readmes_triage_table() -> None:
    """The mechanized table and the prose table, pinned to each other.

    `## Why nothing appeared` is the best-tested writing in this project — two
    walkthroughs verified every row — and doctor is the thing that runs it for
    you. A check id renamed on one side and not the other leaves an adopter
    reading a report whose rows the page does not explain, or a page citing a
    check that no longer exists.

    Not every id has a triage row and that is deliberate: `channel`, `build`
    and `uninstall-story` answer questions the table is not about. What may
    never happen is a row citing an id that is not real.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Why nothing appeared")
    section = readme[start : readme.index("\n## ", start + 10)]
    cited = set(re.findall(r"`([a-z][a-z-]*)`", section)) & set(doctor.CHECK_IDS)
    assert cited, section[:400]
    # Every id the table cites is one doctor really emits.
    bogus = {
        name
        for name in re.findall(r"^\| \*\*[^|]+\| `([a-z-]+)` \|", section, re.M)
    } - set(doctor.CHECK_IDS)
    assert not bogus, sorted(bogus)
    # And the rows that carry an id are all of them: a row with none is a
    # silent state the report cannot name.
    rows = re.findall(r"^\| \*\*[^|]+\|([^|]*)\|", section, re.M)
    assert len(rows) >= 12, len(rows)
    assert all(row.strip().startswith("`") for row in rows), [
        row for row in rows if not row.strip().startswith("`")
    ]
    # The line that told a reader doctor was not in this build is gone.
    assert "would run this list for you, is not in this build" not in readme


def test_the_block_the_docs_show_is_the_block_the_hook_writes(tmp_path) -> None:
    """The 559-byte thing that enters every prompt was described three times
    across these pages and never shown once.

    Regenerated here rather than trusted: a pasted block is prose the moment
    the emitter moves, and this one is quoted as evidence about what an install
    puts in front of a model. `<store>` stands in for the absolute path, and
    `XXXXXXXX` for the frame's nonce — the two substitutions, and the second
    is forced: the delimiter carries eight hex digits drawn per RUN, so a
    byte-for-byte quote of a real one is a block no later run can reproduce.
    Normalising it keeps the comparison exact everywhere the emitter is
    deterministic, which is everything the block says including the `lines=`
    count the opener declares.
    """
    home = tmp_path / "home"
    home.mkdir()
    out = subprocess.run(
        ["python3", str(REPO / "src" / "memkit" / "memory_prompt_recall.py")],
        input=json.dumps(
            {
                "session_id": "docblock",
                "prompt": "why do sprocket backlash and flange torque matter "
                "after a gearbox rebuild",
            }
        ),
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "MEMKIT_CONFIG": str(REPO / "tests" / "fixtures" / "memkit.json"),
        },
    )
    assert out.returncode == 0, out.stderr
    corpus = str(REPO / "tests" / "fixtures" / "corpus" / "project")
    emitted = out.stdout.replace(corpus, "<store>").strip()
    drawn = re.search(r"<(" + hook.FRAME_TAG + r"-[0-9a-f]+)[ >]", emitted)
    assert drawn, emitted[:80]
    # Non-vacuity: the nonce really was drawn, so normalising it is not
    # quietly erasing a delimiter that had stopped carrying one.
    assert len(drawn.group(1)) == len(hook.FRAME_TAG) + 1 + hook.FRAME_NONCE_BYTES * 2
    placeholder = hook.FRAME_TAG + "-" + "X" * (hook.FRAME_NONCE_BYTES * 2)
    emitted = emitted.replace(drawn.group(1), placeholder)
    assert emitted.startswith("<" + placeholder + " "), emitted[:80]

    admission = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    shown = re.search(
        r"```\n(<" + re.escape(placeholder) + r"[^\n]*>\n.*?</"
        + re.escape(placeholder) + r">)\n```",
        admission,
        re.S,
    )
    assert shown, "no pointer block in the admission note"
    assert shown.group(1) == emitted, (
        "the block in docs/ADMISSION.md is not what the hook writes",
        shown.group(1),
        emitted,
    )


def test_a_claim_about_a_command_the_pin_cannot_serve_carries_the_marker() -> None:
    """The `## Status` convention, enforced where it matters most.

    This page describes `main`; the marketplace installs a release. An adopter
    who reads "run /memkit:init" and installs the pin gets a plugin with no
    such skill and no way to know why — which is the exact failure the marker
    convention exists to prevent, on the one command the quick start now leads
    with.

    Self-retiring: once the pin carries the skill, the marker must go, and this
    is what says so.
    """
    _needs_checkout()
    sha = _json(MARKETPLACE)["plugins"][0]["source"]["sha"]
    at_pin = (
        _git("cat-file", "-e", f"{sha}:skills/init/SKILL.md").returncode == 0
    )
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("**3. Run `/memkit:init`")
    first_mention = readme[start : start + 200]
    if at_pin:
        assert FROM_THE_NEXT_RELEASE not in first_mention, (
            "the pin now carries the skill — the marker is stale"
        )
    else:
        assert FROM_THE_NEXT_RELEASE in first_mention, first_mention


def test_no_doctor_remedy_names_a_slash_command_off_the_plugin_channel() -> None:
    """Skills ship only in the plugin payload, so `/memkit:init` is a command a
    nix or pip adopter's harness does not have — and the rollout runbook sends
    the nix operator to doctor first.

    The sibling scrape above matches command HEADS that are plugin binary
    names, so it never saw a slash command. This is the same invariant on the
    surface the channel split was built for: a remedy that guessed would send
    an adopter to a command their channel cannot run.

    Over the REMEDY strings, like its sibling — a docstring naming the command
    it is explaining is prose, and a scrape that could not tell the two apart
    would be one somebody silences.
    """
    hardcoded = [text for text in _doctor_remedies() if "/memkit:" in text]
    # Exactly one literal survives, and it is inside the branch that has
    # already tested the channel: `_config_route`'s plugin arm returns before
    # the non-plugin one is reached. Everything else interpolates
    # `_init_command`, which asks.
    unguarded = [
        text for text in hardcoded
        if "On this channel it writes the config" not in text
    ]
    assert not unguarded, unguarded
    assert hardcoded, "the scrape sees nothing, so it proves nothing"


def test_the_init_remedy_names_a_command_each_channel_has(root) -> None:
    """Both halves, run rather than read: on the plugin channel the slash
    command, off it the binary the channel really ships."""
    from memkit import cli_doctor as doc

    plugin = doc.Machine()
    off = doc.Machine()
    saved = os.environ.get(hook.PLUGIN_ENV)
    try:
        os.environ[hook.PLUGIN_ENV] = "1"
        assert doc._init_command(plugin) == "/memkit:init"
        os.environ.pop(hook.PLUGIN_ENV, None)
        rendered = doc._init_command(off)
    finally:
        if saved is None:
            os.environ.pop(hook.PLUGIN_ENV, None)
        else:
            os.environ[hook.PLUGIN_ENV] = saved
    assert "/memkit:" not in rendered, rendered
    assert "memkit init --dry-run" in rendered
    # And it is a command this channel ships: `memkit` is a console script the
    # pip and nix installs both put on the adopter's own PATH.
    assert rendered.split("`")[1].split()[0] == "memkit"


def test_every_outcome_the_hook_emits_has_a_reason_and_a_row() -> None:
    """The EMITTER, pinned to the two readers that were only pinned to each
    other.

    The README table and doctor's `OUTCOME_REASONS` agree by an equality
    assertion, and nothing tied either to the code that writes the names. A new
    outcome could therefore ship and reach an adopter's histogram as `(an
    outcome this build does not know)` with both readers green — which is the
    drift the vocabulary's own growth rule says to expect.

    Scraped as literals, which the log's contract already requires of them:
    "the outcome arrives as a string LITERAL at each call site, because that is
    what lets the consumer enumerate the vocabulary statically".
    """
    from memkit.cli_doctor import OUTCOME_REASONS

    source = (REPO / "src" / "memkit" / "memory_prompt_recall.py").read_text()
    emitted = set(re.findall(r'done\(\s*"([a-z:-]+)"', source))
    emitted |= set(re.findall(r'return "(gate:[a-z]+)"', source))
    # The one conditional call site, whose two names are the delivery split.
    emitted |= {"injected", "output-lost"}
    assert len(emitted) >= 12, sorted(emitted)

    # The CLI's own records are not prompt outcomes — they carry
    # `"concludes": false`, which is the log's published discriminator and what
    # `_prompt_records` filters on — so they need no histogram row.
    prompt_outcomes = {name for name in emitted if not name.startswith("cli")}
    assert prompt_outcomes <= set(OUTCOME_REASONS), sorted(
        prompt_outcomes - set(OUTCOME_REASONS)
    )


def test_the_search_cli_marks_its_records_as_not_prompt_outcomes(tmp_path) -> None:
    """The discriminator, measured rather than read.

    `gate-outcomes` counts the per-prompt population, and an adopter who
    followed the README's own instruction to run the search command would
    otherwise see their command-line runs inflating a line labelled
    "last N prompts".
    """
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "MEMKIT_CONFIG": str(REPO / "tests" / "fixtures" / "memkit.json"),
    }
    for query in ("flange torque sequence", "zzz nothing matches zzz"):
        subprocess.run(
            ["python3", str(REPO / "src" / "memkit" / "memory_prompt_recall.py"),
             "--search", query],
            capture_output=True, text=True, timeout=120, env=env,
        )
    log = home / ".cache" / "memory-recall" / "log.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records, "the search wrote no record, so this proves nothing"
    for record in records:
        assert str(record.get("outcome", "")).startswith("cli"), record
        assert record.get("concludes") is False, record
        # `cwd` is a PROMPT-path key. The README says a new top-level key may
        # arrive and does not promise it on every shape, so what makes the
        # shapes readable is that each one is consistent: a key present on some
        # command-line records and not others is worse for a downstream reader
        # than one that is never there.
        assert "cwd" not in record, record

    from memkit.cli_doctor import _prompt_records

    assert _prompt_records(records) == []


def test_the_admission_numbers_reproduce_from_its_own_recipe() -> None:
    """The one document written to be checkable has to check out.

    The table said 70 files and 1.3 MiB while its own prose insisted the
    numbers counted the PIN — where the recipe prints 63 and 1.3 MiB — and
    `git ls-files` at HEAD prints 70 and 1.8. The headline was right under
    neither reading, so a reader who did what the document told them to got a
    different answer from the document.
    """
    _needs_checkout()
    note = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    listed = _git("ls-files").stdout.split()
    assert f"**{len(listed)} files" in note, (len(listed), "not the stated count")

    sizes = _git("ls-tree", "-r", "-l", "HEAD").stdout.splitlines()
    total = sum(int(line.split()[3]) for line in sizes if line.split()[3] != "-")
    stated = re.search(r"about ([\d.]+) MiB\*\*", note)
    assert stated, note[:400]
    assert abs(float(stated.group(1)) - total / 1048576) < 0.1, (
        stated.group(1), total / 1048576
    )
    # And the recipe names the tree the table counts, rather than one that
    # answers differently.
    assert "git ls-files | wc -l" in note
    # ONCE PER TREE. A second count in prose is a number nobody updates with
    # the first, and this document's whole claim is that a reader can check it.
    # The pinned tree is the second one described, and it is a different tree
    # rather than a repeat — see the note above the same rule in
    # `test_the_admission_note_answers_what_it_claims_to`.
    pinned = _pinned_file_count()
    counts = {int(n) for n in re.findall(r"\b(\d+) files\b", note)}
    assert len(listed) in counts, counts
    if pinned is None:
        # Same reasoning as the sibling rule: the pinned tree's figure is a
        # second TREE, not a second copy, and it cannot be verified from here.
        assert len(counts) <= 2, counts
    else:
        expected = {len(listed)} if pinned == len(listed) else {len(listed), pinned}
        assert counts == expected, (counts, expected)

    # The shell line count, which was the one number in this file that was
    # never re-derived: it said "about 550" while the two files held 658, and
    # the figure is what somebody decides how much shell to read before
    # installing. Its recipe is in the same block as the others.
    shell = sum(
        len((REPO / rel).read_text(encoding="utf-8").splitlines())
        for rel in ("bin/memkit-hook", "bin/lib/common.sh")
    )
    assert f"**{shell} lines of POSIX" in note, (shell, "not the stated count")
    assert "wc -l bin/memkit-hook bin/lib/common.sh" in note


def test_the_published_sweep_budget_is_the_one_the_code_holds() -> None:
    """Both documents that publish it, against the live constants.

    ADMISSION.md's first line promises "every number here is read out of the
    tree at the pinned sha rather than remembered", and this pair was
    remembered: a commit raised the caps six-fold to shrink convergence and
    left the sentence saying 500 and 100, where it stayed through two review
    rounds while README.md next door already said 3000 and 1000. Asserted
    against `SWEEP_MAX_STATS`/`SWEEP_MAX_UNLINKS` rather than against a
    literal, so the next bump moves the documents or fails here.
    """
    stats, unlinks = hook.SWEEP_MAX_STATS, hook.SWEEP_MAX_UNLINKS
    note = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for name, text in (("ADMISSION.md", note), ("README.md", readme)):
        found = re.findall(r"(\d+) stats and\s+(\d+)\s+unlinks", text)
        assert found, (name, "no sweep-budget sentence to check")
        for pair in found:
            assert (int(pair[0]), int(pair[1])) == (stats, unlinks), (name, pair)


# --- the one path-admission rule, proved over the PAIR ------------------------


def _shell_answers(function: str, corpus: list) -> list:
    """`function` run over `corpus` inside ONE shell, one answer per line.

    One process rather than one per path: the point of a differential test is
    a corpus wide enough to find the case a reading would miss, and a fork per
    case puts a ceiling on how wide that can be.
    """
    driver = (
        f'. "{COMMON_SH}"\n'
        "while IFS= read -r line; do\n"
        f"  {function}\n"
        "done\n"
    )
    out = subprocess.run(
        ["sh", "-c", driver],
        input="\n".join(corpus) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    answers = out.stdout.split("\n")[: len(corpus)]
    assert len(answers) == len(corpus), (len(answers), len(corpus))
    return answers


def _path_corpus() -> list:
    """Every short arrangement of the tokens the rule turns on, plus the
    literals the field produced.

    Generated rather than listed: the failures this exists to catch are the
    ones a person reading both implementations agrees with themselves about.
    Newline-free by construction — the driver is line-oriented, and a path
    with a newline in it is a case neither implementation was written for.
    """
    tokens = ("", "/", "a", ".", "..", "~", "proc", "dev", "fd", " ")
    corpus = [
        "",
        "~",
        "~/",
        "~/x",
        "~root/x",
        "~/../x",
        "/proc",
        "/proc/",
        "/proc/self/cwd/memkit.json",
        "/procx/a",
        "/dev/fd",
        "/dev/fd/",
        "/dev/fd/3",
        "/dev/fdx/a",
        "/a/b",
        "//",
        "/.",
        "/..",
        "a//b",
        "/a/./b",
        "/a/../b",
        "/a/.b",
        "/a/..b",
        "/a/b/.",
        "/a/b/..",
        "relative/path",
        "./relative",
        "../relative",
    ]
    for one in tokens:
        for two in tokens:
            for three in tokens:
                corpus.append(one + two + three)
                corpus.append("/" + one + two + three)
    seen = set()
    unique = []
    for path in corpus:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def test_the_shell_and_the_python_admit_exactly_the_same_paths() -> None:
    """ONE admission rule, proved over the pair rather than read twice.

    `bin/lib/common.sh` decides what the hook will READ and what it will EXEC;
    `memkit init` decides what to WRITE. When those disagree, init writes a
    config the wrapper then refuses, and the adopter gets a store, a clean
    integrity check, exit 0 and silence on every prompt — reachable through the
    option rung, which init trusted and the shell vetted.

    The rule exists twice because the shell cannot import Python and the hook
    path may not fork a shell. What makes it one rule is this: the same corpus
    through both, with the same verdict AND the same sentence, or the sweep is
    red.
    """
    from memkit.memory_prompt_recall import path_refusal

    corpus = _path_corpus()
    answers = _shell_answers(
        'if _why=$(memkit_path_refusal "$line"); then '
        "printf 'R\\t%s\\n' \"$_why\"; else printf 'A\\t\\n'; fi",
        corpus,
    )
    disagreements = []
    for path, answer in zip(corpus, answers):
        verdict, _, why = answer.partition("\t")
        mine = path_refusal(path)
        theirs = why if verdict == "R" else ""
        if mine != theirs:
            disagreements.append((path, theirs, mine))
    assert not disagreements, disagreements[:10]
    # Non-vacuous in both directions: the corpus really does contain paths the
    # rule admits and paths it refuses for each of its three reasons.
    refusals = {path_refusal(p) for p in corpus}
    assert "" in refusals
    # One admission plus each of the rule's four refusals.
    assert len(refusals) == 5, refusals


def test_the_shell_and_the_python_expand_home_the_same_way(monkeypatch) -> None:
    """`os.path.expanduser` is not this rule.

    It expands `~someone/x`, which the shell leaves alone — so the two would
    admit different paths, and the one that admits more is the one that writes
    the config.
    """
    from memkit.memory_prompt_recall import expand_home

    corpus = [p for p in _path_corpus() if p]
    home = "/tmp/memkit-home-fixture"
    env = dict(os.environ, HOME=home)
    driver = (
        f'. "{COMMON_SH}"\n'
        "while IFS= read -r line; do memkit_expand_home \"$line\"; done\n"
    )
    out = subprocess.run(
        ["sh", "-c", driver],
        input="\n".join(corpus) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env=env,
    )
    answers = out.stdout.split("\n")[: len(corpus)]
    monkeypatch.setenv("HOME", home)
    disagreements = [
        (path, theirs, expand_home(path))
        for path, theirs in zip(corpus, answers)
        if expand_home(path) != theirs
    ]
    assert not disagreements, disagreements[:10]
    assert expand_home("~/x") == home + "/x"
    assert expand_home("~root/x") == "~root/x"


def test_the_checker_probe_never_executes_what_the_session_path_supplied(
    tmp_path,
) -> None:
    """The probe runs each candidate to ask its version, so the lookup that
    finds it is a lookup that executes it.

    A checkout with a `.direnv/bin/python3.12` therefore got its own program
    run as the user on every `memkit` invocation — `bin/memkit` calls
    `memkit_resolve_checker` unconditionally, including for `doctor` and for
    `init --dry-run`, the two commands the skills pre-approve.
    """
    session = tmp_path / "project"
    (session / ".direnv" / "bin").mkdir(parents=True)
    marker = tmp_path / "PWNED-probe.txt"
    shim = session / ".direnv" / "bin" / "python3.12"
    shim.write_text(f"#!/bin/sh\necho pwned >> {marker}\nexit 0\n", encoding="utf-8")
    shim.chmod(0o755)
    env = dict(
        os.environ,
        PATH=os.pathsep.join([str(shim.parent), "/usr/bin", "/bin"]),
        PWD=str(session),
    )
    env.pop("CLAUDE_PLUGIN_ROOT", None)
    out = subprocess.run(
        [
            "sh",
            "-c",
            f'. "{COMMON_SH}"\n'
            "memkit_resolve_checker /usr/bin/false\n"
            'printf "%s\\n" "$MEMKIT_CHECKER_CMD"\n'
            'printf "%s\\n" "$PATH"\n',
        ],
        capture_output=True, text=True, timeout=120, check=True,
        cwd=str(session), env=env,
    )
    command, path_after = out.stdout.splitlines()[:2]
    assert not marker.exists(), marker.read_text()
    assert str(shim) not in command, command
    # And the filter is the probe's own rule, not a change to what the
    # subcommand inherits: doctor reports on the PATH the install really has.
    assert path_after == env["PATH"], (path_after, env["PATH"])


def test_the_shell_has_no_second_implementation_of_the_path_rule(
    tmp_path,
) -> None:
    """The parity case this replaces held two implementations of one rule in
    agreement. There is one now, and the second one is deleted rather than
    kept honest.

    It could not be made correct in POSIX sh under this project's own
    zero-external-command rule: no `realpath`, so its filter was a string
    prefix test; it compared against the LOGICAL `$PWD` where python compares
    `realpath(getcwd())`; and its success path printed `""`, which POSIX reads
    as the current directory. Its only consumer was a probe that executed each
    candidate python it found, and that is gone too.
    """
    shell = COMMON_SH.read_text(encoding="utf-8")
    assert "memkit_trusted_path" not in shell
    # And nothing else in `bin/` resolves a program by NAME any more, which is
    # the property the deleted parity case was standing in for. `command -v` on
    # the wrapper's own `$0` is not that: it locates this file, not a program
    # to run.
    for wrapper in ("memkit", "memkit-hook", "memkit-recall"):
        text = (BIN / wrapper).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "command -v" not in line or line.lstrip().startswith("#"):
                continue
            assert '"$_self"' in line, (wrapper, line)
def test_one_home_expansion_reaches_the_reader_the_commands_hand_paths_to(
    tmp_path, monkeypatch
) -> None:
    """"ONE PREDICATE, because the failure it prevents is two of them" — and
    `load_config`, ~470 lines below where that is written, still had the other
    one.

    It is the function `Machine.config()` hands `--config` to on both new
    commands, so the two disagreed on the same flag: `cli_init._resolve_config`
    routed the value through `expand_home` while doctor handed the raw value
    to `load_config`, which called `os.path.expanduser` and so accepted a
    `~someone/x` the wrapper refuses to read. A doctor reporting on a config
    the install cannot load is the shape this pair exists to prevent.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "memkit.json").write_text(
        json.dumps({"schema": hook.SCHEMA, "roots": {}, "stores": []}),
        encoding="utf-8",
    )
    assert hook.load_config("~/memkit.json") is not None
    # `~someone/x` is the case that separates the two rules. `expanduser`
    # turns it into an absolute path; the shell, and this reader, leave it
    # alone — so the file is simply not found.
    with pytest.raises(hook.ConfigError):
        hook.load_config("~nobody-here/memkit.json")
    source = pathlib.Path(hook.__file__).read_text(encoding="utf-8")
    assert "os.path.expanduser(path)" not in source, (
        "a second home expansion is back in the module that declares it has one"
    )


def test_the_init_skill_says_where_dry_run_goes_in_the_argv() -> None:
    """The turn-one grant is a literal prefix.

    `Bash(... init --dry-run:*)` admits an invocation only where `--dry-run`
    immediately follows `init`, so a model that wrote `init --store PATH
    --dry-run` — which reads as equally correct — falls outside the grant and
    raises a permission prompt on the turn this page promises is pre-approved.
    It fails toward MORE prompting rather than less, which is the safe
    direction; what it costs is a handshake the user has been told is two
    turns turning into three.
    """
    skill = (REPO / "skills" / "init" / "SKILL.md").read_text(encoding="utf-8")
    grant = re.search(r"^allowed-tools: (.+)$", skill, re.M)
    assert grant, skill[:200]
    assert grant.group(1).strip().endswith("init --dry-run:*)"), grant.group(1)
    assert "`--dry-run` goes first" in skill


def test_the_task_registration_matches_the_subagent_tool_and_nothing_else() -> None:
    """One `PreToolUse` entry, matched on the Agent tool by name.

    `"Agent"` rather than `"^Agent$"`, and the difference is not cosmetic. The
    harness picks its matching strategy from the CHARACTERS in the matcher
    (measured on 2.1.238): a matcher of word characters, `|`, `,`, spaces and
    hyphens takes an exact-equality branch that first canonicalizes the token
    through the tool alias table, while anything carrying a regex metacharacter
    compiles to a RegExp and is tested unanchored. So the plain form is the
    exact one AND the one that survives a rename — `Task` still dispatches to
    `Agent` through that table — whereas the anchored form is a literal string
    that a rename leaves matching nothing, silently.

    Measured both ways on the pinned binary: `Read` and `^Read$` fire, `Rea`
    and `ead` do not, `Read.*` fires on NotebookEdit-shaped names too.
    """
    entries = [(event, h) for event, h in _entries() if event == "PreToolUse"]
    assert len(entries) == 1, entries
    groups = _json(HOOKS_JSON)["hooks"]["PreToolUse"]
    assert len(groups) == 1, groups
    assert groups[0]["matcher"] == hook.TASK_TOOL, groups[0]
    assert hook.TASK_TOOL == "Agent"
    # No metacharacter, or the harness takes the regex branch and the alias
    # canonicalization that makes this survive a rename never runs.
    assert re.fullmatch(r"[A-Za-z0-9_|, -]+", groups[0]["matcher"]), groups[0]
    # And the prompt path's entry stays unmatched — a matcher there would scope
    # a hook that must see every prompt.
    for group in _json(HOOKS_JSON)["hooks"]["UserPromptSubmit"]:
        assert "matcher" not in group, group


def test_the_docs_count_the_hooks_the_registration_actually_declares() -> None:
    """`plugin details` is the only surface that tells an adopter whether
    registration took, and six places across README.md and docs/ROLLOUT.md
    certify a number for it. The number moved with this registration and the
    prose did not, so a correct install failed the reader's first verification
    step while the install that half-failed passed it.

    Pinned against `hooks.json` rather than against a literal, so the next
    registration change cannot land without the documentation.
    """
    handlers = len(_entries())
    assert handlers == 2, handlers
    for path in (REPO / "README.md", REPO / "docs" / "ROLLOUT.md"):
        text = path.read_text(encoding="utf-8")
        assert f"Hooks ({handlers})" in text, path
        # Every PRESCRIPTIVE statement — the ones a reader checks their own
        # install against — names the live count. A stale one turns a correct
        # install into a reported failure at the reader's first verification
        # step, and certifies the half-registered one as healthy.
        for stated in re.findall(r"must report Hooks \((\d+)\)", text):
            assert int(stated) == handlers, (path, stated)
        for stated in re.findall(r"Hooks \((\d+)\)` is a working install", text):
            assert int(stated) == handlers, (path, stated)
        # Every OTHER count that appears has to be talking about a failure —
        # or about the RELEASE that is currently pinned, which is the second
        # legitimate thing `Hooks (1)` can mean. `.claude-plugin/marketplace.json`
        # pins v0.2.1, whose tree registers one hook, and the pin moves in its
        # own release PR: for the whole window between that merge and this one,
        # a healthy install reports `Hooks (1)` and this page used to call that
        # its own failure, sending a correct install to `Reinstall` at the
        # reader's first verification step. Both readings are here now, marked
        # per `## Status`.
        #
        # Read over a window rather than a line, because the sentence that
        # names either one wraps and a line-scoped check is a test of the line
        # breaks.
        for wrong in (f"Hooks ({n})" for n in range(4) if n != handlers):
            start = 0
            while (at := text.find(wrong, start)) != -1:
                # Tight, because it has to be a claim about THIS mention: the
                # words have to sit in the same clause, not merely on the same
                # screen as some other count's explanation.
                window = text[max(0, at - 90) : at + 90].lower()
                assert "failure" in window or "pinned release" in window, (
                    path, wrong, window
                )
                start = at + 1
        # And the marker convention is actually used, or "from the next
        # release" is a rule the page states and does not follow — which is
        # how the six sites above came to describe a release that has not
        # shipped as if it had.


def test_the_docs_name_every_build_outcome_and_every_state_file() -> None:
    """Two inventories a reader builds a sweeper and a dashboard against.

    The `.build` vocabulary is a documented contract — "these are all of them",
    with a stated rule for the unrecognised ones — and a new outcome that never
    reaches the list leaves the reader classifying it by the rule instead of by
    name. The derived-state list is the one an external sweeper is written
    from: it enumerated one non-index file while this tree writes two, so a
    sweeper built from it globbed the session ledgers and left every per-spawn
    one behind.
    """
    text = (REPO / "README.md").read_text(encoding="utf-8")
    outcomes = {
        value
        for name, value in vars(hook).items()
        if name.startswith("BUILD_") and isinstance(value, str)
    }
    assert len(outcomes) >= 6, outcomes
    for outcome in outcomes:
        assert f"`{outcome}`" in text, outcome
    # The per-spawn ledger, by the prefix a sweeper would glob for.
    assert f"`{hook.TASK_STATE_PREFIX}<tool-use-id>.json`" in text
    assert "`<session-uuid>.json`" in text


def test_the_docs_state_the_frame_sizes_the_frames_actually_are() -> None:
    """Both frames' fixed overhead is a documented number a reader subtracts
    from the 16 KiB refusal bound to work out which of their briefs still get
    served — and neither figure was pinned by anything, so the two sites
    describing the subagent block drifted 226 bytes apart inside one commit
    range while both stayed plausible.

    Fixed part only: `_framed([])` and `_task_framed([])` are the block with no
    pointer lines, which is what "plus the pointer lines" in each sentence
    means.
    """
    text = (REPO / "README.md").read_text(encoding="utf-8")
    prompt_bytes = len(hook._framed([]).encode())
    task_bytes = len(hook._task_framed([]).encode())
    assert f"**{prompt_bytes} bytes fixed**" in text, prompt_bytes
    assert f"**{task_bytes} bytes fixed**" in text, task_bytes
    # The subagent figure is stated twice, in the section a reader is pointed
    # at and in the disclosures, and it is the pair that drifted.
    assert text.count(str(task_bytes)) >= 2, task_bytes
    # Anti-vacuity: the two frames are not the same size, so a check that
    # matched one figure against both would not pass.
    assert prompt_bytes != task_bytes, (prompt_bytes, task_bytes)


def test_the_docs_mark_what_the_pinned_release_does_not_carry_yet() -> None:
    """The window between merging a registration and shipping it.

    `.claude-plugin/marketplace.json` pins the sha `/plugin install` clones,
    and that pin moves in its own release PR — so for the whole window between
    this merge and that one, every adopter who runs Quick start against a
    marketplace install sees one fewer hook than this page describes. The page
    defines a convention for exactly that (`## Status`: a behaviour that has
    landed here and not in a release is marked *(from the next release)*) and
    the subagent docs shipped without it, calling a correct install a failure
    and sending the reader to `Reinstall`, which reinstalls the same sha.

    Derived from the pin rather than asserted as a state: the pinned tree is in
    this repo's own history, so its registration can be counted offline. When
    the release PR moves the pin the counts agree, this case stops requiring
    markers, and RELEASING.md item 4's sweep is free to remove them.
    """
    pinned = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    sha = pinned["plugins"][0]["source"]["sha"]
    shown = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{sha}:hooks/hooks.json"],
        capture_output=True, text=True,
    )
    if shown.returncode != 0:
        pytest.skip(f"the pinned sha {sha[:12]} is not in this clone's history")
    registered = sum(
        len(group["hooks"])
        for groups in json.loads(shown.stdout)["hooks"].values()
        for group in groups
    )
    if registered == len(_entries()):
        return  # the pin carries what this tree registers; nothing to mark
    for path in (REPO / "README.md", REPO / "docs" / "ROLLOUT.md"):
        text = path.read_text(encoding="utf-8")
        assert "from the next release" in text, (path, registered)


def test_the_admission_note_names_both_events_it_registers() -> None:
    """The note is what the README points at for 'what runs on my machine'.
    Its subject list is asserted by name above; this is the half that a new
    event silently escapes — the higher-consequence of the two, since it
    rewrites a tool call rather than printing to a transcript."""
    note = (REPO / "docs" / "ADMISSION.md").read_text(encoding="utf-8")
    # The INVENTORY block, not the prose around it: a paragraph can mention an
    # event while the list a reader counts stays one short, which is the shape
    # this went wrong in.
    after = note[note.index("## What runs, and when") :]
    block = after[after.index("```") + 3 : after.index("```", after.index("```") + 3)]
    for event, handler in _entries():
        assert event in block, (event, block)
        assert f"{handler['timeout']}s" in block, (event, block)
    assert hook.TASK_TOOL in block, block


def test_the_remote_tier_counts_hooks_from_the_manifest_it_installed() -> None:
    """The remote tier asserts a hook count against a clone of the sha in
    `.claude-plugin/marketplace.json`, and that sha is whichever release is
    current — so a literal there is a snapshot that goes red on the release
    that moves the pin, not in the review that changed the registration.

    The tier itself is opt-in and needs the network, so what is checked here is
    the derivation it now uses: over this tree, the manifest and the
    registration agree.
    """
    from rig.test_remote_install import _registered_hooks

    assert _registered_hooks(REPO) == len(_entries())
