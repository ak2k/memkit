#!/usr/bin/env python3
"""Unit tests for memory-integrity.py's checks, over fixture stores.

Run: `pytest tests/test_memory_integrity.py -q`, or `python3 -m unittest` —
these are unittest cases and either runner works.

Every case builds its own store under a tmpdir and hands the checker a fixture
CONFIG, because the checker takes its trees and its citation roots from one:
a store built without them would be checked by a citation regex that matches
nothing, and pass for the wrong reason.

Every shape here occurs in a real store. The false-positive ones are
load-bearing: a link check that flags quoted examples gets switched off, and
then the dead links it exists to catch come back.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from memkit import eval_memory_recall as ev
from memkit import memory_integrity as mi
from memkit import memory_prompt_recall as hook

MEMORY_HEAD = "# Fixture hot ledger\n\n## Index\n"
SEARCH_HEAD = "# Fixture search ledger\n\n## Index\n"

# The store dir every fixture uses, and the top-level trees a fixture citation
# may name. Both are CONFIG in the shipped tool, so the fixtures have to supply
# them: `docs/plans/x.md` is only a citation because `docs` is on this list.
STORE_DIR = "docs/memories"
FIXTURE_CITED_ROOTS = (".github", "docs", "modules", "scripts", "users")
FIXTURE_CITED_SUFFIXES = (".conf", ".lock")


def _memory(name: str, body: str = "") -> str:
    return f"---\nname: {name}\ndescription: fixture memory {name}\n---\n\n{body}\n"


def _config(
    path: Path, stores: list[dict], *, cited: bool = True, **extra
) -> Path:
    """Write a fixture config and return its path.

    Roots are `~`-rooted so the whole tree can be redirected with HOME, which
    is what the sandboxed runs and the subprocess cases rest on.
    """
    body = {
        "schema": hook.SCHEMA,
        "roots": {"home": {"kind": "path", "path": "~"}},
        "stores": stores,
        "citations": (
            {
                "roots": list(FIXTURE_CITED_ROOTS),
                "extra_suffixes": list(FIXTURE_CITED_SUFFIXES),
            }
            if cited
            else {}
        ),
    }
    body.update(extra)
    path.write_text(json.dumps(body))
    return path


def _loaded(path: Path, **kw):
    """`load_config`, narrowed to a Config.

    The reader answers None for ABSENCE, which is a legitimate state with a
    case of its own below. It is never the state a test that just wrote the
    file is in, so narrowing once here keeps every other call site from
    carrying a branch it cannot reach.
    """
    cfg = hook.load_config(str(path), **kw)
    assert cfg is not None, f"{path} did not load"
    return cfg


class LinkCase(unittest.TestCase):
    """One fixture store per test, settled by a --write pass so the only
    findings left are the ones the case is about."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, files: dict[str, str], memory_md: str = MEMORY_HEAD) -> dict:
        d = self.root / STORE_DIR
        for sub in ("hot", "search", "archive"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "MEMORY.md").write_text(memory_md)
        (d / "SEARCH.md").write_text(SEARCH_HEAD)
        for rel, text in files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        store = mi._store(
            self.root,
            STORE_DIR,
            cited_roots=FIXTURE_CITED_ROOTS,
            cited_suffixes=FIXTURE_CITED_SUFFIXES,
        )
        mi.check(store, True, self.names(store))  # generate SEARCH.md rows
        return store

    def names(self, store: dict) -> set[str]:
        return mi._memory_names((store,))

    def run_check(self, store: dict, write: bool = False) -> tuple[list[str], ...]:
        errors, warnings = mi.check(store, write, self.names(store))
        return (
            [e for e in errors if e.startswith("DEAD-LINK")],
            [w for w in warnings if w.startswith("DANGLING-WIKILINK")],
            [e for e in errors if not e.startswith("DEAD-LINK")],
        )


class DeadLinks(LinkCase):
    def test_dead_relative_link_is_an_error(self) -> None:
        store = self.build({"search/a.md": _memory("a", "see [b](../hot/gone.md)")})
        dead, _, other = self.run_check(store)
        self.assertEqual(other, [])
        self.assertEqual(len(dead), 1, dead)
        self.assertIn("docs/memories/search/a.md:6", dead[0])
        self.assertIn("../hot/gone.md", dead[0])

    def test_valid_links_across_tiers_and_out_of_store_are_clean(self) -> None:
        (self.root / "docs" / "plans").mkdir(parents=True)
        (self.root / "docs" / "plans" / "p.md").write_text("plan\n")
        store = self.build(
            {
                "search/a.md": _memory(
                    "a",
                    "[hot](../hot/h.md), [plan](../../plans/p.md), "
                    "[sub](domain/k.md), [self](#a-heading)",
                ),
                "search/domain/k.md": _memory("k", "[back](../a.md)"),
                "hot/h.md": _memory("h", "[down](../search/a.md)"),
                "archive/old.md": _memory("old", "[live](../search/a.md)"),
            },
            MEMORY_HEAD + "\n- [h](hot/h.md) — a hot fixture\n",
        )
        dead, dangling, other = self.run_check(store)
        self.assertEqual((dead, dangling, other), ([], [], []))

    def test_dead_link_in_a_ledger_preamble_is_an_error(self) -> None:
        store = self.build({}, MEMORY_HEAD.replace("ledger", "[ledger](nope.md)"))
        dead, _, _ = self.run_check(store)
        self.assertEqual(len(dead), 1, dead)
        self.assertIn("MEMORY.md:1", dead[0])

    def test_external_urls_and_anchors_are_ignored(self) -> None:
        store = self.build(
            {
                "search/a.md": _memory(
                    "a",
                    "[web](https://example.invalid/x.md) [mail](mailto:x@example.com)\n"
                    "[anchor](#somewhere) [word](notapath)",
                )
            }
        )
        self.assertEqual(self.run_check(store), ([], [], []))

    def test_home_absolute_link_into_the_store_is_checked(self) -> None:
        # `~`-rooted links appear in the ledger preambles; they must resolve
        # like any other, while an absolute path elsewhere on the filesystem
        # (/nix/store, /etc) is not this script's business.
        store = self.build({"search/a.md": _memory("a", "x")})
        roots = (store["root"],)
        src = store["dir"] / "search" / "a.md"
        self.assertIsNone(mi._link_path("/nix/store/deadbeef-x/x.md", src, roots))
        inside = store["root"] / "docs/memories/search/missing.md"
        self.assertEqual(mi._link_path(str(inside), src, roots), inside.resolve())
        self.assertEqual(
            mi._link_path("~/gone/x.md", src, (Path.home(),)),
            (Path.home() / "gone/x.md").resolve(),
        )

    def test_write_mode_reports_dead_links_too(self) -> None:
        store = self.build({"search/a.md": _memory("a", "[b](../hot/gone.md)")})
        self.assertEqual(len(self.run_check(store, write=True)[0]), 1)


class CodeContexts(LinkCase):
    """The two shapes that made a naive checker unusable, plus their fenced
    cousins. Both were found in real memories, not invented here: a pandoc
    image example inside a code span, and a regex with a bracket class."""

    def test_pandoc_image_example_in_a_code_span_is_clean(self) -> None:
        store = self.build(
            {
                "search/a.md": _memory(
                    "a", "Embed via `![alt](pic.png){width=6.5in}` and run pandoc."
                )
            }
        )
        self.assertEqual(self.run_check(store), ([], [], []))

    def test_regex_example_in_a_code_span_is_clean(self) -> None:
        store = self.build(
            {"search/a.md": _memory("a", "the slug gate is `^\\d+[hd](?:_[a-z_]+)?$`.")}
        )
        self.assertEqual(self.run_check(store), ([], [], []))

    def test_links_inside_fenced_blocks_are_clean(self) -> None:
        store = self.build(
            {
                "search/a.md": _memory(
                    "a",
                    "```markdown\n[row](../hot/gone.md)\n```\n\n"
                    "~~~\n[row](../hot/also-gone.md)\n~~~\n",
                )
            }
        )
        self.assertEqual(self.run_check(store), ([], [], []))

    def test_bare_image_in_prose_is_still_checked(self) -> None:
        store = self.build({"search/a.md": _memory("a", "![diagram](pic.png)")})
        dead, _, _ = self.run_check(store)
        self.assertEqual(len(dead), 1, dead)

    def test_a_dead_link_after_a_closed_fence_is_still_found(self) -> None:
        # Fence tracking that never closes would swallow the rest of the file,
        # turning the check into a no-op nobody notices.
        store = self.build(
            {"search/a.md": _memory("a", "```\ncode\n```\n\n[b](../hot/gone.md)")}
        )
        self.assertEqual(len(self.run_check(store)[0]), 1)

    def test_mask_code_keeps_line_numbers_and_closes_only_on_a_bare_fence(
        self,
    ) -> None:
        text = "a\n```py\nb\n```\nc\n"
        self.assertEqual(mi._mask_code(text), ["a", "", "", "", "c"])
        # An info string never closes a block, and ~~~ does not close ```.
        self.assertEqual(mi._mask_code("```\n~~~\n```sh\n```\nx\n")[-1], "x")

    def test_mask_spans_pairs_equal_length_backtick_runs(self) -> None:
        self.assertEqual(mi._mask_spans("a `b` c"), "a     c")
        self.assertEqual(mi._mask_spans("a ``b ` c`` d"), "a           d")
        self.assertEqual(mi._mask_spans("unmatched ` tick"), "unmatched ` tick")


class Wikilinks(LinkCase):
    def test_dangling_wikilink_warns_and_does_not_fail(self) -> None:
        store = self.build({"search/a.md": _memory("a", "see [[not_written_yet]]")})
        dead, dangling, other = self.run_check(store)
        self.assertEqual((dead, other), ([], []))
        self.assertEqual(len(dangling), 1, dangling)
        self.assertIn("not_written_yet", dangling[0])

    def test_wikilink_resolves_by_stem_frontmatter_name_or_archive(self) -> None:
        store = self.build(
            {
                "search/a.md": _memory("renamed_in_frontmatter", "x"),
                "search/b.md": _memory("b", "[[a]] [[renamed_in_frontmatter]] [[old]]"),
                "archive/old.md": _memory("old", "x"),
            }
        )
        self.assertEqual(self.run_check(store), ([], [], []))

    def test_bash_double_bracket_and_char_classes_in_code_are_clean(self) -> None:
        store = self.build(
            {
                "search/a.md": _memory(
                    "a",
                    "guard with `[[ -f $f ]]` and strip `[[:space:]]`\n\n"
                    "```bash\nif [[ -n $x ]]; then tr -d '[[:space:]]'; fi\n```\n",
                )
            }
        )
        self.assertEqual(self.run_check(store), ([], [], []))

    def test_wikilinks_pointing_at_the_other_store_resolve(self) -> None:
        # The index is built over both stores, so a project memory may name a
        # personal one. Passing a two-store index is what main() does.
        store = self.build({"search/a.md": _memory("a", "see [[elsewhere]]")})
        errors, warnings = mi.check(store, False, self.names(store) | {"elsewhere"})
        self.assertEqual(errors, [])
        self.assertEqual([w for w in warnings if w.startswith("DANGLING")], [])


class CitedPaths(LinkCase):
    """Repo paths named in prose. The check is blame-aligned — only memories
    the commit touches are read — so every case here lands a fixture repo at
    HEAD and then edits it. Without the commit the tests would pass on a check
    that never ran."""

    def git(self, *args: str) -> str:
        # Identity, signing and hooks are forced per-command: the fixture must
        # not depend on (or trip over) whatever global gitconfig the machine
        # running the tests happens to have.
        return subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "user.name=fixture",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def landed(self, files: dict[str, str], repo_files: tuple[str, ...] = ()) -> dict:
        """A fixture store committed to a fresh repo, plus any real repo files
        a citation is allowed to resolve to. Nothing is 'changed' on return."""
        for rel in repo_files:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture\n")
        store = self.build(files)
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "fixture")
        return store

    def touch(self, store: dict, rel: str, body: str) -> None:
        """Rewrite a memory AFTER the commit — i.e. make it a file this commit
        touches, which is the only kind the check reads."""
        (store["dir"] / rel).write_text(_memory(Path(rel).stem, body))

    def findings(self, store: dict) -> tuple[list[str], list[str], list[str]]:
        errors, warnings = mi.check(store, False, self.names(store))
        return (
            [e for e in errors if e.startswith("DEAD-PATH")],
            [e for e in errors if not e.startswith("DEAD-PATH")],
            [w for w in warnings if w.startswith("CITED-PATHS-SKIPPED")],
        )

    def test_cited_path_that_exists_is_clean(self) -> None:
        store = self.landed(
            {"search/a.md": _memory("a", "x")}, ("modules/common/example.nix",)
        )
        self.touch(store, "search/a.md", "the fleet list is modules/common/example.nix")
        self.assertEqual(self.findings(store), ([], [], []))

    def test_cited_path_that_is_gone_is_an_error(self) -> None:
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(store, "search/a.md", "the fleet list is modules/common/example.nix")
        dead, other, _ = self.findings(store)
        self.assertEqual(other, [])
        self.assertEqual(len(dead), 1, dead)
        self.assertIn("docs/memories/search/a.md:6", dead[0])
        self.assertIn("modules/common/example.nix", dead[0])

    def test_unchanged_file_with_a_broken_citation_is_not_flagged(self) -> None:
        # The whole design: rot you did not cause must not fail your commit.
        store = self.landed(
            {
                "search/a.md": _memory("a", "the fleet list is modules/gone.nix"),
                "search/b.md": _memory("b", "x"),
            }
        )
        self.touch(store, "search/b.md", "nothing cited here")
        self.assertEqual(self.findings(store), ([], [], []))

    def test_an_untracked_memory_is_checked(self) -> None:
        store = self.landed({"search/a.md": _memory("a", "x")})
        (store["dir"] / "search" / "new.md").write_text(
            _memory("new", "see scripts/gone.py")
        )
        mi.check(store, True, self.names(store))  # row the new file
        dead, _, _ = self.findings(store)
        self.assertEqual(len(dead), 1, dead)
        self.assertIn("scripts/gone.py", dead[0])

    def test_line_number_suffix_is_not_part_of_the_path(self) -> None:
        store = self.landed(
            {"search/a.md": _memory("a", "x")}, ("users/example/host.nix",)
        )
        self.touch(store, "search/a.md", "the alias is users/example/host.nix:1470")
        self.assertEqual(self.findings(store), ([], [], []))
        self.touch(store, "search/a.md", "the alias is users/example/gone.nix:1470")
        dead, _, _ = self.findings(store)
        self.assertEqual(len(dead), 1, dead)
        self.assertIn("users/example/gone.nix names no file", dead[0])

    def test_trailing_punctuation_is_prose_not_filename(self) -> None:
        store = self.landed(
            {"search/a.md": _memory("a", "x")}, ("modules/example/base.nix",)
        )
        self.touch(store, "search/a.md", "set it in modules/example/base.nix.")
        self.assertEqual(self.findings(store), ([], [], []))

    def test_citations_in_code_spans_and_fences_are_clean(self) -> None:
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(
            store,
            "search/a.md",
            "run `scripts/gone.py --write`\n\n```sh\nvim modules/also-gone.nix\n```\n",
        )
        self.assertEqual(self.findings(store), ([], [], []))

    def test_a_url_containing_a_repo_tree_is_not_a_citation(self) -> None:
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(
            store,
            "search/a.md",
            "upstream https://github.com/o/r/blob/main/modules/gone.nix has it",
        )
        self.assertEqual(self.findings(store), ([], [], []))

    def test_link_destinations_are_left_to_the_dead_link_check(self) -> None:
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(store, "search/a.md", "see [plan](docs/plans/gone.md)")
        dead, other, _ = self.findings(store)
        self.assertEqual(dead, [])
        self.assertEqual(len(other), 1, other)
        self.assertTrue(other[0].startswith("DEAD-LINK"), other[0])

    def test_a_bare_topic_is_not_a_citation_but_a_trailing_slash_is(self) -> None:
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(store, "search/a.md", "the docs/plans convention")
        self.assertEqual(self.findings(store), ([], [], []))
        self.touch(store, "search/a.md", "drop it in docs/plans/")
        self.assertEqual(len(self.findings(store)[0]), 1)

    def test_without_git_the_check_is_skipped_and_says_so(self) -> None:
        # No `landed()`, so the tmpdir is not a repo. Silence here is only
        # acceptable because the run announces it.
        store = self.build({"search/a.md": _memory("a", "see modules/gone.nix")})
        dead, other, skipped = self.findings(store)
        self.assertEqual((dead, other), ([], []))
        self.assertEqual(len(skipped), 1, skipped)

    def test_a_clean_tree_says_it_checked_nothing(self) -> None:
        # The silent no-op: nothing changed, so no citation is read, and the
        # run used to be indistinguishable from one that checked and passed.
        store = self.landed(
            {"search/a.md": _memory("a", "the fleet list is modules/gone.nix")}
        )
        dead, other, skipped = self.findings(store)
        self.assertEqual((dead, other), ([], []))
        self.assertEqual(len(skipped), 1, skipped)
        self.assertIn("clean", skipped[0])

    def test_the_audit_switch_reads_unchanged_files_too(self) -> None:
        # The env var is the same switch as --all and must land in the same
        # place: the unchanged file IS read, and its drift is reported — as a
        # warning, because nobody in this commit touched it.
        store = self.landed(
            {"search/a.md": _memory("a", "the fleet list is modules/gone.nix")}
        )
        with unittest.mock.patch.dict(
            mi.os.environ, {"MEMORY_INTEGRITY_ALL_CITATIONS": "1"}
        ):
            errors, warnings = mi.check(store, False, self.names(store))
        self.assertEqual([e for e in errors if e.startswith("DEAD-PATH")], [])
        self.assertEqual(len([w for w in warnings if w.startswith("DEAD-PATH")]), 1)

    def test_all_reports_world_drift_as_a_warning_never_an_error(self) -> None:
        # --all audits memories this commit never touched, so its findings
        # must not fail anybody's commit — same finding, warning side.
        store = self.landed(
            {"search/a.md": _memory("a", "the fleet list is modules/gone.nix")}
        )
        errors, warnings = mi.check(store, False, self.names(store), True)
        self.assertEqual([e for e in errors if e.startswith("DEAD-PATH")], [])
        dead = [w for w in warnings if w.startswith("DEAD-PATH")]
        self.assertEqual(len(dead), 1, warnings)
        self.assertIn("modules/gone.nix", dead[0])

    def test_all_still_blocks_on_a_citation_this_commit_broke(self) -> None:
        # The N1 hole: --all is what check-all runs, so demoting EVERY finding
        # to a warning left the blocking path with no caller at all — an
        # author's own dead citation passed in the one place authors run this.
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(store, "search/a.md", "the roster is modules/i_made_it_up.nix")
        errors, _ = mi.check(store, False, self.names(store), True)
        dead = [e for e in errors if e.startswith("DEAD-PATH")]
        self.assertEqual(len(dead), 1, errors)
        self.assertIn("modules/i_made_it_up.nix", dead[0])

    def test_a_dirty_non_memory_file_does_not_silence_the_skip_notice(self) -> None:
        # The N2 hole: the notice keyed on an EMPTY changed set, so any
        # unrelated dirty file — the everyday state of a working tree — put the
        # run back to passing without reading a single citation.
        store = self.landed(
            {"search/a.md": _memory("a", "the fleet list is modules/gone.nix")}
        )
        (self.root / "README.md").write_text("unrelated\n")
        dead, other, skipped = self.findings(store)
        self.assertEqual((dead, other), ([], []))
        self.assertEqual(len(skipped), 1, skipped)
        self.assertIn("changed set", skipped[0])

    def test_indented_sample_output_is_not_a_citation(self) -> None:
        # Citations mask indented code; links deliberately do not. A path in a
        # pasted transcript belongs to whatever tree the transcript came from.
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(
            store,
            "search/a.md",
            "transcript:\n\n    $ cat modules/not_here.nix\n\nend",
        )
        dead, _, _ = self.findings(store)
        self.assertEqual(dead, [])

    def test_a_tab_reaches_the_indent_whatever_precedes_it(self) -> None:
        # CommonMark lays tabs on a four-column grid, so one, two or three
        # spaces and then a tab is an indented code block exactly as four
        # spaces is. Matching the literal shapes let all three through, and
        # what came through was a transcript path reported against the commit
        # that pasted it.
        store = self.landed({"search/a.md": _memory("a", "x")})
        for indent in ("\t", " \t", "  \t", "   \t"):
            with self.subTest(indent=repr(indent)):
                self.touch(
                    store,
                    "search/a.md",
                    f"transcript:\n\n{indent}$ cat modules/not_here.nix\n\nend",
                )
                dead, _, _ = self.findings(store)
                self.assertEqual(dead, [])

    def test_an_indented_block_right_after_a_fence_is_still_code(self) -> None:
        # A fenced block ends what came before it, so the next line starts a
        # fresh block and an indented run there opens code. Carrying the
        # pre-fence paragraph state across left this one line unmasked.
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(
            store,
            "search/a.md",
            "prose\n\n```\nfenced\n```\n    $ cat modules/not_here.nix\n\nend",
        )
        dead, _, _ = self.findings(store)
        self.assertEqual(dead, [])

    def test_a_citation_broken_in_a_commit_still_blocks(self) -> None:
        # Committing is not verifying. With only the working tree in the blame
        # set, an author who committed the break before running the check had
        # it reclassified as world drift — a warning about somebody else's rot,
        # earned by having committed it — and the check passed on the very
        # change that broke it.
        store = self.landed({"search/a.md": _memory("a", "x")})
        # The real BLAME_BASE, planted rather than patched, so this exercises
        # the default the checker ships with.
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.touch(store, "search/a.md", "the fleet list is modules/common/example.nix")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "break it")
        # The tree really is clean, so nothing but the commit can be blamed.
        self.assertEqual(self.git("status", "--porcelain").strip(), "")

        dead, other, _ = self.findings(store)
        self.assertEqual(other, [])
        self.assertEqual(len(dead), 1, dead)
        self.assertIn("docs/memories/search/a.md", dead[0])

    def test_without_a_base_ref_a_clean_tree_says_it_checked_nothing(self) -> None:
        # The documented degrade: no merge base — an unfetched remote, a repo
        # that has no such ref — leaves the working tree as the only answer.
        # That is the pre-existing behaviour, and it must stay LOUD rather than
        # quietly passing.
        # No origin/main is planted, so BLAME_BASE resolves to nothing.
        store = self.landed({"search/a.md": _memory("a", "x")})
        self.touch(store, "search/a.md", "the fleet list is modules/common/example.nix")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "break it")

        dead, other, skipped = self.findings(store)
        self.assertEqual((dead, other), ([], []))
        self.assertEqual(len(skipped), 1, skipped)


class Layout(LinkCase):
    """Files the tier layout does not account for, and the fence that hides
    them. Both are false-GREENS: the checker reports nothing and the memory
    is unreachable anyway."""

    def test_a_memory_loose_at_the_store_root_is_an_error(self) -> None:
        store = self.build({})
        (store["dir"] / "loose.md").write_text(_memory("loose"))
        _, _, other = self.run_check(store)
        self.assertEqual(len(other), 1, other)
        self.assertTrue(other[0].startswith("STRAY-ROOT"), other[0])

    def test_a_memory_under_a_non_tier_directory_is_an_error(self) -> None:
        # hot/, search/ and archive/ are the tiers; a file under anything
        # else is read by no check and searched by no hook.
        store = self.build({})
        (store["dir"] / "drafts").mkdir()
        (store["dir"] / "drafts" / "wip.md").write_text(_memory("wip"))
        _, _, other = self.run_check(store)
        self.assertEqual(len(other), 1, other)
        self.assertTrue(other[0].startswith("STRAY-DIR"), other[0])
        self.assertIn("drafts/wip.md", other[0])

    def test_tier_directories_and_search_subdirs_are_not_stray(self) -> None:
        store = self.build(
            {
                "hot/h.md": _memory("h"),
                "search/a.md": _memory("a"),
                "search/domain/k.md": _memory("k"),
                "archive/old.md": _memory("old"),
            },
            MEMORY_HEAD + "\n- [h](hot/h.md) — a hot fixture\n",
        )
        _, _, other = self.run_check(store)
        self.assertEqual([e for e in other if e.startswith("STRAY")], [])

    def test_an_unterminated_fence_is_reported_not_silently_masked(self) -> None:
        # Everything below the fence is masked, so the dead link inside it is
        # invisible; the fence itself is the finding.
        store = self.build(
            {"search/a.md": _memory("a", "```sh\nvim x\n\n[dead](../hot/gone.md)\n")}
        )
        dead, _, other = self.run_check(store)
        self.assertEqual(dead, [])
        self.assertEqual(len(other), 1, other)
        self.assertTrue(other[0].startswith("UNCLOSED-FENCE"), other[0])
        self.assertIn("search/a.md:6", other[0])

    def test_a_closed_fence_is_not_reported(self) -> None:
        store = self.build(
            {"search/a.md": _memory("a", "```sh\nvim [x](../hot/gone.md)\n```\n")}
        )
        dead, _, other = self.run_check(store)
        self.assertEqual((dead, other), ([], []))


class SubIndexCounts(LinkCase):
    def store_with_sub_index(self, present: bool) -> dict:
        d = self.root / "docs" / "memories"
        for sub in ("hot", "search/domain", "archive"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "SEARCH.md").write_text(SEARCH_HEAD)
        (d / "search" / "domain" / "k.md").write_text(_memory("k"))
        (d / "MEMORY.md").write_text(
            MEMORY_HEAD + "\n- [domain](search/domain/INDEX.md) — sub (7 memories)\n"
        )
        if present:
            # Sub-index membership is read from the sub-index's own rows.
            (d / "search" / "domain" / "INDEX.md").write_text(
                SEARCH_HEAD + "\n- [k](k.md) — fixture memory k\n"
            )
        return mi._store(
            self.root,
            STORE_DIR,
            ("search/domain/INDEX.md",),
            cited_roots=FIXTURE_CITED_ROOTS,
            cited_suffixes=FIXTURE_CITED_SUFFIXES,
        )

    def test_a_missing_sub_index_leaves_its_count_untouched(self) -> None:
        # --write must not "correct" the row to (0 memories): the number it
        # would write is an artifact of the file being gone, and rewriting it
        # destroys the only evidence of what was there.
        store = self.store_with_sub_index(present=False)
        errors, _ = mi.check(store, True, self.names(store))
        self.assertTrue(
            any(e.startswith("ERROR: sub-index not found") for e in errors), errors
        )
        self.assertIn("(7 memories)", store["hot_ledger"].read_text())

    def test_a_present_sub_index_still_gets_its_count_fixed(self) -> None:
        store = self.store_with_sub_index(present=True)
        mi.check(store, True, self.names(store))
        self.assertIn("(1 memories)", store["hot_ledger"].read_text())


class Descriptions(LinkCase):
    """The `description:` line is the retrieval surface — the hook indexes it
    and the ledgers are generated from it — so a memory without a usable one
    is unreachable rather than merely untidy. memory-integrity.py has always
    rejected those, and nothing held the rejection in place; these are the
    tests for a check that was written and never exercised.

    Measured against the corpus on 2026-08-13, both stores were already clean
    on every axis here: 0 of 248 memories missing a description, 0 missing a
    `type:`, 0 over the cap. That is what makes these ratchets rather than
    repairs — they keep a clean tree clean.
    """

    def desc_errors(self, text: str) -> list[str]:
        store = self.build({"search/a.md": text})
        errors, _ = mi.check(store, False, self.names(store))
        return [e for e in errors if e.startswith("DESC-")]

    def test_a_memory_with_no_description_is_an_error(self) -> None:
        errs = self.desc_errors("---\nname: a\n---\n\nbody\n")
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("DESC-BAD", errs[0])
        self.assertIn("empty", errs[0])

    def test_a_description_over_the_cap_is_an_error_naming_the_length(self) -> None:
        # The cap is below the hook's DESC_KEEP_CHARS so an authored
        # description is never the thing that gets truncated mid-sentence.
        over = "x" * (mi.MAX_DESC_CHARS + 1)
        errs = self.desc_errors(f"---\nname: a\ndescription: {over}\n---\n\nbody\n")
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("DESC-LONG", errs[0])
        self.assertIn(f"{mi.MAX_DESC_CHARS + 1}c", errs[0])

    def test_a_description_exactly_at_the_cap_is_clean(self) -> None:
        at = "x" * mi.MAX_DESC_CHARS
        self.assertEqual(
            self.desc_errors(f"---\nname: a\ndescription: {at}\n---\n\nb\n"), []
        )

    def test_a_quoted_description_is_measured_without_its_quotes(self) -> None:
        # The quotes are YAML syntax, not description text, and every long
        # description in the corpus is quoted — counting them turns 13 files
        # that are inside the cap into 13 spurious failures.
        at = "x" * mi.MAX_DESC_CHARS
        self.assertEqual(
            self.desc_errors(f'---\nname: a\ndescription: "{at}"\n---\n\nb\n'), []
        )


class ConfigDrivenStores(unittest.TestCase):
    """What the checker and the eval read out of one config file.

    These are the seam the extraction created: before it, the trees and the
    citation roots were constants in the source, so nothing could disagree with
    them. Now they are data, and every one of these cases is a way that data
    can be wrong quietly.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def load(self, stores: list[dict], **extra):
        path = _config(self.tmp / "memkit.json", stores, **extra)
        return _loaded(path, honor_env_overrides=True)

    def test_n_stores_come_back_in_config_order(self) -> None:
        # KTD10's whole property: the tools take a LIST, so a one-store
        # deployment and a two-store one differ only in this file. Order is a
        # contract as well — retrieval interleaves in it — so it is asserted,
        # not just membership.
        cfg = self.load(
            [
                {"id": "a", "dir": "one", "live_root": "home"},
                {"id": "b", "dir": "two", "live_root": "home"},
            ]
        )
        root = self.tmp / "repo"
        self.assertEqual(
            ev.store_roots(cfg, root), [root / "one", root / "two"]
        )

    def test_a_single_store_config_is_not_a_special_case(self) -> None:
        # The reason the original test existed — an analyzer run from a
        # worktree measuring the MAIN checkout — belongs to the consumer's
        # inverse test, where the analyzers live. What survives here is the
        # config plumbing: one store is just a shorter list.
        cfg = self.load([{"id": "only", "dir": "notes", "live_root": "home"}])
        root = self.tmp / "repo"
        self.assertEqual(ev.store_roots(cfg, root), [root / "notes"])

    def test_the_checker_reports_which_tree_answered_for_each_store(self) -> None:
        # The wrong-tree bug was invisible precisely because the output named
        # no repo: `[OK] docs/memories/ (139 files)` reads the same whether it
        # inspected your worktree or somebody else's checkout.
        cfg = self.load([{"id": "project", "dir": STORE_DIR, "live_root": "home"}])
        stores, notes = mi.stores_from_config(cfg)
        self.assertEqual(len(stores), 1)
        self.assertIn("project store:", notes[0])
        self.assertIn("configured path", notes[0])

    def test_a_store_with_no_cited_roots_says_so_instead_of_passing(self) -> None:
        # An empty citations.roots makes the prose-citation regex match
        # nothing, so every memory passes without a path being looked at. That
        # green means "not configured" and reads as "verified".
        cfg = self.load(
            [{"id": "s", "dir": STORE_DIR, "live_root": "home"}], cited=False
        )
        stores, _ = mi.stores_from_config(cfg)
        self.assertEqual(stores[0]["cited_roots"], ())
        store = mi._store(self.tmp, STORE_DIR, cited_roots=())
        for sub in ("hot", "search"):
            (store["dir"] / sub).mkdir(parents=True)
        (store["dir"] / "MEMORY.md").write_text(MEMORY_HEAD)
        (store["dir"] / "SEARCH.md").write_text(SEARCH_HEAD)
        _, warnings = mi.check(store, False, set())
        self.assertTrue(
            any(w.startswith("CITED-PATHS-UNCONFIGURED") for w in warnings), warnings
        )

    def test_a_store_that_never_declared_citations_is_not_told_off_for_it(self):
        # The other half, and it is the commonest store there is: a config that
        # never mentions `citations` has opted OUT of the feature. Both
        # findings above were the first thing a fresh adopter's checker run
        # said, about a check they never asked for — and a report whose first
        # two lines are noise is a report they learn to skim.
        #
        # Declared-and-empty keeps the warning, because that IS a citation
        # check configured to match nothing.
        store = mi._store(self.tmp, STORE_DIR, cited_roots=(), citations_declared=False)
        for sub in ("hot", "search"):
            (store["dir"] / sub).mkdir(parents=True)
        (store["dir"] / "MEMORY.md").write_text(MEMORY_HEAD)
        (store["dir"] / "SEARCH.md").write_text(SEARCH_HEAD)
        _, warnings = mi.check(store, False, set())
        self.assertEqual([w for w in warnings if w.startswith("CITED-PATHS")], [])

    def test_whether_citations_were_declared_survives_the_config_read(self):
        # Absent-or-empty collapses the two states away, and the checker needs
        # the difference. Read off the config rather than asserted about the
        # store dict alone, because the store is built FROM it.
        declared = self.load(
            [{"id": "s", "dir": STORE_DIR, "live_root": "home"}], cited=False
        )
        self.assertTrue(declared.citations_declared)
        path = self.tmp / "nocit.json"
        path.write_text(
            json.dumps(
                {
                    "schema": hook.SCHEMA,
                    "roots": {"home": {"kind": "path", "path": str(self.tmp)}},
                    "stores": [{"id": "s", "dir": STORE_DIR, "live_root": "home"}],
                }
            )
        )
        silent = hook.load_config(str(path))
        assert silent is not None
        self.assertFalse(silent.citations_declared)
        stores, _ = mi.stores_from_config(silent)
        self.assertFalse(stores[0]["citations_declared"])

    def test_a_newer_schema_is_refused_rather_than_half_read(self) -> None:
        # A reader that met a higher number and carried on would be reading
        # half a config, which for a fail-open hook is a silent retrieval
        # outage rather than an error anybody sees.
        path = self.tmp / "future.json"
        path.write_text(json.dumps({"schema": hook.SCHEMA + 1, "stores": []}))
        with self.assertRaises(hook.ConfigError) as caught:
            hook.load_config(str(path))
        self.assertIn("schema", str(caught.exception))

    def test_no_config_at_all_is_absence_not_an_error(self) -> None:
        # Inert is a legitimate state and the shipped default: no config means
        # no stores, which means zero pointers and exit 0.
        with unittest.mock.patch.dict(mi.os.environ, {}, clear=True):
            self.assertIsNone(hook.load_config())

    def test_a_config_that_is_there_and_unreadable_is_an_error(self) -> None:
        # The other half: "no config" and "a config I could not honour" are
        # different states, and only the first is allowed to be silent.
        path = self.tmp / "broken.json"
        path.write_text("{not json")
        with self.assertRaises(hook.ConfigError):
            hook.load_config(str(path))

    def test_a_root_expands_home_when_it_is_read_not_when_it_is_written(self) -> None:
        # 13 subprocess cases in the hook suite and every sandboxed check
        # redirect an entire corpus with HOME and nothing else. Pre-resolving
        # a `~` root at write time would work in-process and silently score the
        # developer's real stores in the child.
        cfg = self.load([{"id": "s", "dir": "notes", "live_root": "home"}])
        with unittest.mock.patch.dict(mi.os.environ, {"HOME": "/somewhere/else"}):
            fresh = _loaded(self.tmp / "memkit.json")
            self.assertEqual(fresh.root("home"), "/somewhere/else")
        self.assertEqual(cfg.root("home"), mi.os.path.expanduser("~"))


class VerifiesTheEditTree(unittest.TestCase):
    """Which of a store's two trees the checker reads, blames and rewrites.

    The roots answer different questions, and this is the seam where the two
    were confused. `live_root` is what the recall HOOK serves, so a session in
    any checkout reads the canonical memories. `edit_root` is the tree whose
    change is being verified, which is the only tree a checker can honestly
    report on: rooted at the live copy instead, a run from a worktree printed a
    confident `[OK]` about a tree nobody was editing, and `--write` regenerated
    ledgers over there while the drift stayed here.

    Every case below builds the two trees with DIFFERENT memories in them, so
    no finding, generated row or rewritten byte is ambiguous about which tree
    it came from. A store whose two roots agree — the fixtures, and any
    single-checkout deployment — cannot tell the two behaviours apart at all,
    which is why this suite has to spend a second tree to see the difference.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.live = self.lay_out("live", "live_only")
        self.edit = self.lay_out("edit", "edit_only")

    def lay_out(self, name: str, memory: str) -> Path:
        """A whole store under `<tmp>/<name>`, holding one search memory.

        Deliberately left UNSETTLED — SEARCH.md carries its preamble and no
        rows — so both trees are drifted, and a run that reports no drift is
        reporting about neither of them.
        """
        root = self.tmp / name
        d = root / STORE_DIR
        for sub in ("hot", "search", "archive"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "MEMORY.md").write_text(MEMORY_HEAD)
        (d / "SEARCH.md").write_text(SEARCH_HEAD)
        (d / "search" / f"{memory}.md").write_text(
            _memory(memory, f"a dead pointer only {name} has: [x](../hot/{name}.md)")
        )
        return root

    def config(self):
        """One store, live in one tree and edited in the other."""
        path = _config(
            self.tmp / "memkit.json",
            [
                {
                    "id": "project",
                    "dir": STORE_DIR,
                    "live_root": "live",
                    "edit_root": "edit",
                }
            ],
            roots={
                "live": {"kind": "path", "path": str(self.live)},
                "edit": {"kind": "path", "path": str(self.edit)},
            },
        )
        return _loaded(path, honor_env_overrides=True)

    def built(self) -> dict:
        stores, _ = mi.stores_from_config(self.config())
        self.assertEqual(len(stores), 1)
        return stores[0]

    def test_the_store_the_checker_builds_is_rooted_at_the_edit_tree(self) -> None:
        store = self.built()
        self.assertEqual(store["root"], self.edit)
        self.assertEqual(store["dir"], self.edit / STORE_DIR)

    def test_the_live_tree_is_still_named_so_the_ok_cannot_be_misread(self) -> None:
        # Both trees on the line, because the fact that makes this run's [OK]
        # narrow — a live copy nobody just verified — is invisible otherwise.
        _, notes = mi.stores_from_config(self.config())
        self.assertIn(str(self.edit), notes[0])
        self.assertIn(str(self.live), notes[0])
        self.assertLess(notes[0].index(str(self.edit)), notes[0].index(str(self.live)))

    def test_findings_name_the_edit_trees_memories_and_never_the_live_ones(
        self,
    ) -> None:
        # Both trees hold a memory with a dead link. Only one of them is the
        # tree this run is answerable for.
        store = self.built()
        errors, _ = mi.check(store, False, mi._memory_names((store,)))
        report = "\n".join(errors)
        self.assertIn("edit_only.md", report)
        self.assertNotIn("live_only.md", report)
        self.assertTrue([e for e in errors if e.startswith("DEAD-LINK")], errors)
        self.assertTrue([e for e in errors if e.startswith("LEDGER-DRIFT")], errors)

    def test_write_regenerates_the_edit_tree_and_does_not_touch_the_live_one(
        self,
    ) -> None:
        # The property the old rooting was defended on — `--write` only ever
        # regenerates one tree — is kept; which tree it is, is what moved.
        live_ledger = self.live / STORE_DIR / "SEARCH.md"
        before = live_ledger.read_bytes()
        store = self.built()
        mi.check(store, True, mi._memory_names((store,)))
        edited = (self.edit / STORE_DIR / "SEARCH.md").read_text()
        self.assertIn("edit_only.md", edited)
        self.assertNotIn("live_only.md", edited)
        self.assertEqual(live_ledger.read_bytes(), before)

    def test_the_blamed_tree_is_the_verified_tree(self) -> None:
        # Blame used to be the one thing that followed the edit tree, through a
        # separate pass, while everything else read the live one. There is no
        # second tree to reconcile now: the store carries one root and the
        # citation scan is handed that.
        store = self.built()
        self.assertEqual(store["root"], self.edit)
        self.assertNotIn("edit_root", store)
        _, why = mi._changed_files(store["root"], store["blame_base"])
        self.assertIn(str(self.edit), why)


class RemediationText(unittest.TestCase):
    """What a finding actually tells its reader to type.

    The remediation is the only part of a check anybody acts on, and it rots in
    a way nothing else in this suite would notice: it named
    `scripts/memory-integrity.py --write` for as long as this tool WAS that
    file, and it went on looking right through the extraction that deleted the
    file. Two properties outlive any one layout, so both are pinned here — the
    recipe names the installed entry point, and it is BARE.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def load(self, stores: list[dict], **extra):
        path = _config(self.tmp / "memkit.json", stores, **extra)
        return _loaded(path, honor_env_overrides=True)

    def drifted(self, **kw) -> dict:
        """A store drifted BOTH ways at once: a search memory with no row in
        the generated ledger, and a hot row whose count disagrees."""
        d = self.tmp / STORE_DIR
        for sub in ("hot", "search", "archive"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "MEMORY.md").write_text(
            MEMORY_HEAD + "\n- [search](SEARCH.md) — the ledger (7 memories)\n"
        )
        (d / "SEARCH.md").write_text(SEARCH_HEAD)
        (d / "search" / "k.md").write_text(_memory("k"))
        return mi._store(
            self.tmp,
            STORE_DIR,
            cited_roots=FIXTURE_CITED_ROOTS,
            cited_suffixes=FIXTURE_CITED_SUFFIXES,
            **kw,
        )

    def finding(self, store: dict, kind: str) -> str:
        errors, _ = mi.check(store, False, mi._memory_names((store,)))
        hits = [e for e in errors if e.startswith(kind)]
        self.assertEqual(len(hits), 1, errors)
        return hits[0]

    def test_ledger_drift_names_the_installed_command(self) -> None:
        found = self.finding(self.drifted(), "LEDGER-DRIFT")
        self.assertIn("run `memory-integrity --write`", found)
        self.assertNotIn("scripts/", found)

    def test_count_drift_names_the_installed_command(self) -> None:
        found = self.finding(self.drifted(), "COUNT-DRIFT")
        self.assertIn("run `memory-integrity --write`", found)
        self.assertNotIn("scripts/", found)

    def test_the_recipe_carries_no_environment_prefix_to_remember(self) -> None:
        # The bare command is correct BECAUSE the run verified the tree you are
        # standing in: `--write` from here rewrites the ledgers this run just
        # complained about. While the checker verified the live tree instead,
        # the recipe grew a config-derived `VAR=$PWD` prefix to compensate, and
        # a store whose live root is a pinned path with a declared override is
        # exactly the shape that grew it — so that is the shape asserted bare.
        self.drifted()  # for the layout it lays down under self.tmp
        cfg = self.load(
            [{"id": "s", "dir": STORE_DIR, "live_root": "pinned"}],
            roots={
                "pinned": {
                    "kind": "path",
                    "path": str(self.tmp),
                    "env": "FIXTURE_STORE_REPO",
                }
            },
        )
        stores, _ = mi.stores_from_config(cfg)
        errors, _ = mi.check(stores[0], False, set())
        drifts = [e for e in errors if "DRIFT" in e]
        self.assertEqual(len(drifts), 2, errors)  # LEDGER-DRIFT and COUNT-DRIFT
        for finding in drifts:
            self.assertIn("run `memory-integrity --write`", finding)
            self.assertNotIn("$PWD", finding)
            self.assertNotIn("FIXTURE_STORE_REPO", finding)

    def test_nothing_still_names_the_pre_extraction_script_path(self) -> None:
        # The string this class exists for. A grep, not a behaviour: a future
        # finding can reintroduce the dead path without going through any of
        # the paths exercised above.
        self.assertNotIn(
            "scripts/memory-integrity.py", Path(mi.__file__).read_text()
        )


class ToolAgreements(unittest.TestCase):
    """Numbers and behaviours that must agree across files in this repo.

    The hook is a standalone script the harness runs by path and the checker is
    a different tool with a different python floor, so every shared number is
    written down more than once. These are the tripwires for that.
    """

    def test_the_description_cap_stays_under_the_hooks_truncation(self) -> None:
        # A description longer than the hook shows reaches no reader. The cap
        # here has to stay below the hook's cut, and the error message quotes
        # that number, so it is asserted too.
        self.assertLess(mi.MAX_DESC_CHARS, hook.DESC_KEEP_CHARS)
        self.assertLess(hook.DESC_KEEP_CHARS, hook.DESC_MAX_CHARS)
        source = Path(mi.__file__).read_text()
        self.assertIn(f"truncates at {hook.DESC_KEEP_CHARS}", source)

    def test_every_eval_retrieval_honors_the_repo_it_was_pointed_at(self) -> None:
        # --repo exists because the hook resolves its stores to the LIVE roots
        # on purpose, so an eval run from a worktree otherwise scores that
        # worktree's edited descriptions against the live copies. The override
        # reaches retrieval through recall(dirs=...) and nowhere else: a call
        # site that omits it silently measures the default stores, and the
        # scoreboard then covers two checkouts without naming either. That is
        # what the vocab slice did — three call sites passed dirs and the
        # fourth did not, and nothing failed, because the two corpora agree
        # whenever --repo is not used.
        #
        # By AST over every call, not by running one: the defect is an
        # omission at a call site nobody is looking at, so the assertion has
        # to be about all of them.
        src = Path(ev.__file__).read_text()
        calls = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "recall"
        ]
        self.assertTrue(calls, "found no recall() call sites to check")
        for call in calls:
            self.assertIn(
                "dirs",
                {kw.arg for kw in call.keywords},
                f"recall() at line {call.lineno} ignores --repo",
            )

    def test_a_hook_copied_without_its_data_files_will_not_load(self) -> None:
        # COMMON_WORDS_FILE resolves from the hook's own __file__, and a
        # missing file leaves _common_words() an EMPTY frozenset rather than
        # raising: every stopword stays a search term and retrieval shifts
        # wholesale. `--hook` exists to make exactly these copies, so the
        # failure has to be loud — a measurement that is silently wrong is
        # worse than one that refuses to run.
        #
        # The consumer's inverse test asserts the same of ITS loader. Two
        # separate implementations, and the second grew its --hook after the
        # first did, so the guard is exactly the kind of thing written once and
        # forgotten on the copy — now literally across two repos.
        with tempfile.TemporaryDirectory() as tmp:
            lone = Path(tmp) / "memory_prompt_recall.py"
            lone.write_bytes(Path(hook.__file__).read_bytes())
            with self.assertRaises(RuntimeError) as caught:
                ev.load_hook(lone)
        self.assertIn("common-words.txt", str(caught.exception))

    def test_the_shipped_wordlist_is_armed(self) -> None:
        # A wordlist that went missing once — untracked, so a sealed build
        # dropped it — leaves the Zipf floor open and every gate passes.
        self.assertTrue(len(hook._common_words()) > 10_000)


class EvalGateDecisionRules(unittest.TestCase):
    """The retrieval eval's gate, exercised rather than described.

    eval-memory-recall.py decides which of a scoreboard's moved lines the TOOL
    is answerable for, and that decision is four pure functions and one
    constant: verdict(), case_record(), read_snapshot()/write_snapshot() and
    GATING. Nothing runs them on a prompt's path, so before this class the only
    thing that exercised them was a full run — a real corpus, the hook, an FTS5
    index and the committed snapshot — which reports that the number moved and
    not which rule moved it. Every case here is one rule, hermetically.

    Two seams are deliberately absent because they are not callable. The
    aggregation (`against_snapshot`) is a closure over main()'s locals, and the
    vacuity floor is a list comprehension in main()'s tail; both end in
    sys.exit, and reaching either means running the whole harness against a
    fixture checkout with a hook, a store pair and an index in it. What is
    asserted instead is the pair the closure joins — the kind verdict() returns
    and the slices GATING names — plus a structural read of the one branch that
    joins them.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = ev

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def record(
        self, status: str, file: str | None = None, position: str | None = None
    ) -> dict:
        """A snapshot row, built by the script's own constructor.

        Never by hand: case_record is the one place the row's shape is decided,
        and a literal dict here would keep passing after that shape moved.
        """
        return self.eval.case_record(status, file, position)

    # --- verdict(): which of the four kinds, and whose fault it is ---

    def test_a_case_that_matches_its_record_passes_with_nothing_to_say(self) -> None:
        seen = self.record("PASS", "a.md", "search")
        self.assertEqual(self.eval.verdict(seen, dict(seen)), ("ok", ""))

    def test_a_moved_status_on_the_baselined_corpus_is_a_regression(self) -> None:
        # The whole point of the fingerprint: same corpus, moved outcome, so
        # the retriever moved. One of the two kinds that can fail the build.
        kind, why = self.eval.verdict(
            self.record("MISS", "a.md", "search"),
            self.record("PASS", "a.md", "search"),
        )
        self.assertEqual(kind, "regression")
        self.assertIn("PASS", why)

    def test_a_case_the_snapshot_never_heard_of_is_new(self) -> None:
        # `new` gates alongside `regression` in a gating slice: letting an
        # unrecorded case pass would make adding one the way to add an ungated
        # case, with --update-snapshot the sanctioned path instead.
        kind, _ = self.eval.verdict(self.record("PASS", "a.md", "search"), None)
        self.assertEqual(kind, "new")

    def test_a_target_that_changed_tier_is_drift_because_the_question_did(
        self,
    ) -> None:
        # search asserts injection and hot asserts abstention, so a promoted
        # target makes the recorded status an answer to a different question —
        # not the same question answered worse.
        kind, why = self.eval.verdict(
            self.record("ABSTAIN-OK", "a.md", "hot"),
            self.record("PASS", "a.md", "search"),
        )
        self.assertEqual(kind, "drift")
        self.assertIn("search", why)
        self.assertIn("hot", why)

    def test_a_case_repointed_at_another_memory_is_drift_not_a_regression(self) -> None:
        # Pointing a case at a different memory changes what is asserted, not
        # how well the assertion held, and the reason names both files because
        # the reader's next move is to decide which one the case is about.
        kind, why = self.eval.verdict(
            self.record("PASS", "b.md", "search"),
            self.record("PASS", "a.md", "search"),
        )
        self.assertEqual(kind, "drift")
        self.assertIn("a.md", why)
        self.assertIn("b.md", why)

    def test_the_target_is_compared_before_its_tier_and_its_outcome(self) -> None:
        # All three moved at once. The reason has to name the outermost, since
        # a retargeted case's tier and status are facts about a different file.
        _, why = self.eval.verdict(
            self.record("MISS", "b.md", "hot"),
            self.record("PASS", "a.md", "search"),
        )
        self.assertIn("a.md", why)
        self.assertNotIn("PASS", why)

    def test_the_tier_is_compared_before_the_outcome(self) -> None:
        kind, why = self.eval.verdict(
            self.record("MISS", "a.md", "hot"),
            self.record("PASS", "a.md", "search"),
        )
        self.assertEqual(kind, "drift")
        self.assertNotIn("PASS", why)

    def test_a_moved_corpus_demotes_every_mismatch_to_drift(self) -> None:
        # The non-gating regime. These stores are not the ones baselined, so
        # nothing measured is attributable to the tool — including a case the
        # snapshot never recorded, which under a moved corpus is usually a
        # memory somebody just wrote.
        regressed, _ = self.eval.verdict(
            self.record("MISS", "a.md", "search"),
            self.record("PASS", "a.md", "search"),
            corpus_matches=False,
        )
        self.assertEqual(regressed, "drift")
        unrecorded, why = self.eval.verdict(
            self.record("PASS", "a.md", "search"), None, corpus_matches=False
        )
        self.assertEqual(unrecorded, "drift")
        self.assertIn("corpus changed", why)

    def test_drift_keeps_its_own_reason_when_the_corpus_also_moved(self) -> None:
        # A retargeted case is drift for a reason of its own in both regimes.
        # Appending "corpus changed" would point the reader at
        # --update-snapshot for a case somebody rewrote by hand.
        _, why = self.eval.verdict(
            self.record("PASS", "b.md", "search"),
            self.record("PASS", "a.md", "search"),
            corpus_matches=False,
        )
        self.assertNotIn("corpus changed", why)

    def test_a_recorded_skip_compares_equal_to_the_same_skip(self) -> None:
        # Skips carry a status like every other row. Recording position only
        # left their status comparing None to None, so a case whose target had
        # been retired passed forever after.
        seen = self.record("SKIP", "a.md", "archive")
        self.assertEqual(seen["status"], "SKIP")
        self.assertEqual(self.eval.verdict(seen, dict(seen)), ("ok", ""))

    def test_a_target_retired_since_the_baseline_reports_as_drift(self) -> None:
        # Retirement moves the position first, so it is reported and does not
        # gate — which is right: the memory was retired on purpose, and the
        # answer is to drop the case and re-baseline, not to fail the build.
        kind, _ = self.eval.verdict(
            self.record("SKIP", "a.md", "archive"),
            self.record("PASS", "a.md", "search"),
        )
        self.assertEqual(kind, "drift")

    def test_a_class_that_names_no_file_compares_on_its_status_alone(self) -> None:
        # The abstention cases assert about no target at all, so both sides
        # read None for file and position and the comparison falls through.
        seen = self.record("NOINJECT-OK")
        self.assertEqual(set(seen), {"status"})
        self.assertEqual(self.eval.verdict(seen, dict(seen)), ("ok", ""))
        kind, _ = self.eval.verdict(seen, self.record("NOINJECT-FAIL"))
        self.assertEqual(kind, "regression")

    def test_an_explicit_null_target_compares_equal_to_an_omitted_one(self) -> None:
        # A hand-edited snapshot can carry `"file": null` where a written one
        # omits the key; reading those as different targets would report drift
        # on every abstention case at once and gate nothing.
        self.assertEqual(
            self.eval.verdict(
                self.record("NOINJECT-OK"),
                {"file": None, "position": None, "status": "NOINJECT-OK"},
            ),
            ("ok", ""),
        )

    def test_a_record_reads_target_then_tier_then_outcome(self) -> None:
        # json.dumps preserves insertion order and this file is written to be
        # read in a diff, so the key order is the reading order.
        self.assertEqual(
            list(self.record("PASS", "a.md", "search")), ["file", "position", "status"]
        )

    # --- which failures reach the exit code ---

    def test_the_default_gate_is_the_suite_and_only_the_suite(self) -> None:
        # Which slices gate is config now, and the DEFAULT is what a consumer
        # who never names one gets. `vocab` must not be in it: those prompts
        # were built to defeat term matching, so their misses diagnose
        # `description:` wording, and folding them in would fail CI on prose.
        path = self.tmp / "memkit.json"
        path.write_text(json.dumps({"schema": hook.SCHEMA, "stores": []}))
        cfg = _loaded(path)
        self.assertEqual(cfg.eval_gating, frozenset({"suite"}))

    def test_a_config_can_widen_the_gate_but_must_say_so(self) -> None:
        # The widening is the consumer's call — a corpus whose abstention
        # slice is meant to be clean gates on it — and it is only ever explicit.
        path = _config(
            self.tmp / "wide.json", [], eval={"gating_slices": ["suite", "noinject"]}
        )
        self.assertEqual(_loaded(path).eval_gating, frozenset({"suite", "noinject"}))

    def test_nothing_outside_a_gating_slice_can_raise_the_failure_count(self) -> None:
        # The aggregation is a closure over main()'s locals and cannot be
        # called from here, so its one load-bearing branch is read instead:
        # every site that raises gate_fails sits under a test that consults
        # the gating set, and that test admits exactly the two attributable
        # kinds. An increment written outside one — on drift, or on any
        # regression whatever the slice — is how a corpus edit starts failing
        # CI, and no case line in the output would say so.
        tree = ast.parse(Path(ev.__file__).read_text())
        bumps = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AugAssign)
            and getattr(n.target, "id", "") == "gate_fails"
        ]
        self.assertEqual(len(bumps), 1, "the gate grew a second failure counter site")
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(child is bumps[0] for child in ast.walk(node))
            and "gating"
            in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        ]
        self.assertTrue(
            guards, "gate_fails is raised without consulting the gating set"
        )
        kinds = {
            c.value for c in ast.walk(guards[0].test) if isinstance(c, ast.Constant)
        }
        self.assertTrue(
            {"regression", "new"}.issubset(kinds),
            f"the gating branch admits {kinds}, not both attributable kinds",
        )

    # --- the snapshot file itself ---

    def cases(self) -> dict:
        return {
            "suite": {
                "how do I retire a fixture memory": self.record("PASS", "a.md", "search")
            },
            "noinject": {"thanks, that makes sense now": self.record("NOINJECT-OK")},
            "vocab": {},
        }

    def test_a_snapshot_survives_a_write_and_a_read_unchanged(self) -> None:
        path = self.tmp / "expect.json"
        cases = self.cases()
        self.eval.write_snapshot(path, cases, "deadbeef")
        self.assertEqual(
            self.eval.read_snapshot(path), {"corpus": "deadbeef", "cases": cases}
        )

    def test_re_baselining_a_run_that_moved_nothing_rewrites_the_same_bytes(
        self,
    ) -> None:
        # The file exists to be read in a review diff, so an --update-snapshot
        # over an unmoved run has to produce no diff at all — a reformat, a
        # reordering or a dropped field would bury the one line that did move.
        path = self.tmp / "expect.json"
        self.eval.write_snapshot(path, self.cases(), "deadbeef")
        first = path.read_bytes()
        again = self.eval.read_snapshot(path)
        assert again is not None, "the file this case just wrote reads as absent"
        self.eval.write_snapshot(path, again["cases"], again["corpus"])
        self.assertEqual(path.read_bytes(), first)

    def test_a_gating_read_refuses_a_snapshot_with_no_fingerprint(self) -> None:
        # Unattributable is the NON-gating regime, so a snapshot predating the
        # fingerprint would otherwise read as "the corpus moved" and buy a
        # permanently green check that never says it stopped looking.
        path = self.tmp / "expect.json"
        path.write_text(json.dumps({"cases": {"suite": {}}}))
        with self.assertRaises(RuntimeError) as caught:
            self.eval.read_snapshot(path)
        self.assertIn("--update-snapshot", str(caught.exception))

    def test_the_run_that_is_about_to_overwrite_it_reads_it_leniently(self) -> None:
        # The refusal above names --update-snapshot as the fix, so that run
        # cannot be the one it refuses.
        path = self.tmp / "expect.json"
        path.write_text(json.dumps({"cases": {"suite": {}}}))
        prior = self.eval.read_snapshot(path, require_fingerprint=False)
        assert prior is not None, "the file this case just wrote reads as absent"
        self.assertIsNone(prior["corpus"])

    def test_a_missing_snapshot_reads_as_nothing_recorded(self) -> None:
        # Distinct from an unreadable one: main() turns this into "run
        # --update-snapshot and commit the result" rather than a green run.
        self.assertIsNone(self.eval.read_snapshot(self.tmp / "absent.json"))

    def test_a_snapshot_with_no_cases_object_is_refused_either_way(self) -> None:
        # A truncated or hand-mangled file is not "no expectations recorded".
        # Read as one, the run would compare nothing and exit like a clean
        # one, so it has to raise in both modes — including the update mode,
        # which is otherwise the lenient one.
        path = self.tmp / "expect.json"
        path.write_text(json.dumps({"corpus": "deadbeef"}))
        for require in (True, False):
            with (
                self.subTest(require_fingerprint=require),
                self.assertRaises(RuntimeError),
            ):
                self.eval.read_snapshot(path, require_fingerprint=require)

    def test_the_written_file_keeps_its_prompts_readable(self) -> None:
        # ensure_ascii=False and no key sort, both for the same reason: a
        # prompt whose em dash came back as an escape, sorted away from the
        # neighbours it was written beside, is a file only a machine can
        # review — and this one is read by a human deciding whether the
        # movement it shows was meant.
        path = self.tmp / "expect.json"
        self.eval.write_snapshot(
            path,
            {
                "suite": {
                    "zeroth — an em dash": self.record("PASS", "z.md", "search"),
                    "first": self.record("MISS", "a.md", "search"),
                }
            },
            "deadbeef",
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("zeroth — an em dash", text)
        self.assertNotIn("\\u2014", text)
        self.assertLess(text.index("zeroth"), text.index("first"))
        self.assertTrue(text.endswith("\n"), "no trailing newline to diff against")

    # --- the fingerprint that picks the regime ---

    def fixture_store(self, name: str, files: dict[str, str]) -> Path:
        """A checkout holding only memories, laid out where the eval looks."""
        repo = self.tmp / name
        for rel, text in files.items():
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return repo

    def one_store_config(self):
        """A config whose single store is where fixture_store puts things."""
        path = _config(
            self.tmp / "fp.json",
            [{"id": "project", "dir": STORE_DIR, "live_root": "home"}],
        )
        return _loaded(path)

    def test_a_memory_retiered_without_an_edit_hashes_differently(self) -> None:
        # The tier is the directory and the assertion follows it, so a
        # hot/->search/ move flips what every case about that file asserts
        # while changing not one byte of it. Hashing contents alone would call
        # that corpus the baselined one and gate on the flip.
        body = _memory("a", "body")
        before = self.fixture_store("before", {"docs/memories/search/a.md": body})
        after = self.fixture_store("after", {"docs/memories/hot/a.md": body})
        cfg = self.one_store_config()
        self.assertNotEqual(
            self.eval.corpus_fingerprint(cfg, before),
            self.eval.corpus_fingerprint(cfg, after),
        )

    def test_a_stray_non_markdown_file_cannot_switch_the_gate_off(self) -> None:
        # Only *.md is indexed, so a .DS_Store or an editor swapfile landing in
        # a store must not be able to move the run into the regime where every
        # mismatch is drift and nothing gates.
        body = _memory("a", "body")
        clean = self.fixture_store("clean", {"docs/memories/search/a.md": body})
        littered = self.fixture_store(
            "littered",
            {"docs/memories/search/a.md": body, "docs/memories/search/.DS_Store": "x"},
        )
        cfg = self.one_store_config()
        self.assertEqual(
            self.eval.corpus_fingerprint(cfg, clean),
            self.eval.corpus_fingerprint(cfg, littered),
        )


class FixtureEvalSensitivity(unittest.TestCase):
    """Whether the fixture eval CI gates on can go red at all.

    `memory-eval` over `tests/fixtures/` is one of the repo's five checks, and
    all CI reads of it is the exit code. That makes it the check most able to
    fail open: a run that compared nothing and a run that compared everything
    and liked it print the same 0, and if the invented corpus, the cases and
    the committed snapshot ever stop being able to disagree, the check goes on
    passing while asserting nothing. Nothing else in this repo would notice —
    the eval's unit tests below exercise `verdict()` and the snapshot reader
    directly, which is precisely the layer that would still look right.

    So this drives the whole harness as CI drives it, over a COPY of the
    fixtures, and asserts that the two failures it exists to catch reach the
    exit code. The same probe was run once out of tree — a hook copy with
    MAX_HITS=0, pointed at CI, watched to redden — and the evidence lived in a
    closed pull request rather than in the suite.

    A third mutation deliberately has no case here: re-pointing a fixture case
    at a different memory exits 0, because `verdict()` compares the target
    before the outcome and a retargeted case asserts a different thing (see
    test_a_case_repointed_at_another_memory_is_drift_not_a_regression). It is
    the wrong probe for this property, and reads as insensitivity if you try
    it.
    """

    FIXTURES = Path(__file__).parent / "fixtures"
    # One suite case, quoted because both mutations key on it. A prompt that
    # falls out of the fixture config fails these tests loudly rather than
    # quietly mutating nothing.
    CASE = "recalibrate a widget after a firmware flash"

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.fixtures = self.copy_writable(self.FIXTURES, self.tmp / "fixtures")

    def copy_writable(self, src: Path, dst: Path) -> Path:
        """`copytree`, then restore write permission on every copy.

        The nix checks run these suites against a checkout in the store, where
        every file is mode 444 and a copy of one inherits that — so the two
        mutations below opened a read-only file and died, on the one leg where
        the corpus is not a working tree. The flake's own fixture checks say
        the same thing as `chmod -R u+w`.
        """
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        for path in (dst, *dst.rglob("*")):
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        return dst

    def run_eval(self, *args: str) -> subprocess.CompletedProcess[str]:
        """The eval as CI runs it, on this interpreter.

        `-m` rather than the console script: the entry point is only on PATH
        when the package was installed with one, and the nix checks run the
        suites from a python environment where that is not the shape.
        """
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "memkit.eval_memory_recall",
                "--config",
                str(self.fixtures / "memkit.json"),
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

    def snapshot(self) -> dict:
        return json.loads((self.fixtures / "eval-expectations.json").read_text())

    def test_the_unmutated_fixtures_are_green(self) -> None:
        # The control, and not a formality: without it every assertion below is
        # satisfied by an eval that fails on everything, which is the other way
        # this check could be worthless.
        done = self.run_eval()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("matches the snapshot", done.stdout)

    def test_an_outcome_that_moved_off_the_snapshot_reaches_the_exit_code(
        self,
    ) -> None:
        # The corpus is untouched, so the run is in the attributable regime and
        # a recorded outcome that no longer holds is the tool's to answer for.
        path = self.fixtures / "eval-expectations.json"
        data = self.snapshot()
        recorded = data["cases"]["suite"][self.CASE]
        self.assertEqual(recorded["status"], "PASS", "the fixture case moved")
        recorded["status"] = "MISS"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

        done = self.run_eval()
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("REGRESSION", done.stdout)
        self.assertIn("1 gating failure(s)", done.stdout)

    def test_a_retrieval_change_the_corpus_did_not_ask_for_reddens_the_gate(
        self,
    ) -> None:
        # The end the gate exists for: the corpus is what was baselined and the
        # RETRIEVER moved. MAX_HITS is the constant that does it most bluntly —
        # at 0 the hook retrieves as before and points at none of it, so every
        # search case misses while the abstention cases still pass, which is
        # also what proves the mutation reached retrieval and not the harness.
        #
        # A whole-directory copy because the hook resolves common-words.txt
        # beside __file__, and the eval refuses a lone .py for that reason.
        stock = Path(hook.__file__).parent
        copy = self.copy_writable(stock, self.tmp / "hookcopy")
        target = copy / Path(hook.__file__).name
        source = target.read_text()
        blinded = re.sub(r"^MAX_HITS = \d+", "MAX_HITS = 0", source, count=1, flags=re.M)
        self.assertNotEqual(blinded, source, "MAX_HITS is no longer assigned plainly")
        target.write_text(blinded)

        done = self.run_eval("--hook", str(target))
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("MAX_HITS=0", done.stdout)
        self.assertIn("REGRESSION", done.stdout)
        self.assertIn("5/5 retrieved", self.run_eval().stdout)  # and it is the copy
        self.assertIn("search tier: 0/5 retrieved", done.stdout)
