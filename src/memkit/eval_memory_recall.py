#!/usr/bin/env python3
"""Eval harness for memory recall — does the hook surface the right memory?

Runs the consumer's suite of realistic mid-task prompts through the SAME
retrieval path as the memory-recall hook (imported, not reimplemented) and
scores what the hook would have injected.

The harness ships CASE-FREE. Cases pair a prompt with the basename of a real
memory, so they are the consumer's data and live in the consumer's config
(`eval.cases`), never in this file — a shipped case list would be a public
inventory of somebody's private corpus, and it would score every adopter
against a corpus they do not have.

Scoring is TIER-AWARE, because the two tiers want opposite outcomes and a
suite that cannot tell them apart cannot tell a correct abstention from a
wrong miss:

  search/  -> PASS when the expected file IS injected (retrieval).
  hot/     -> PASS when the expected file is NOT injected (abstention). Hot
              memories are already in context via the auto-loaded MEMORY.md,
              so the hook excludes them by design; pointing at one spends the
              injection budget on something the model has already read.

A case's tier is resolved from where its file lives RIGHT NOW — tier is the
directory (`<store>/{hot,search,archive}/…`), so promoting or demoting a
memory flips its assertion with no edit to the case. Cases name a file, never
a path, for the same reason, and there is deliberately no second `hot` case
list to keep in agreement with the corpus.

A third class, `noinject`, names no file at all: prompts that must inject
NOTHING (NOINJECT-OK / NOINJECT-FAIL). Precision has no other probe here —
every suite case asks "did the right thing surface", none asks "did anything
surface that should not have", and a retrieval harness that only measures
recall rewards a hook that injects on everything.

The suite is the regression test for `description:` quality: when a search
case fails, the fix is normally to sharpen the expected file's description
line (the retrieval surface), then re-run. An ABSTAIN-FAIL is the opposite
signal — a hot file leaking into the pointers — and is a hook bug, not a
description bug. Add a case whenever a real session needed a memory and the
hook missed it.

Some `noinject` cases are expected to be RED on a given corpus, and that is
the point: they measure the relevance floor against prompts the corpus has
nothing to say about. Do not delete or reword a case to get the number up — a
case only earns a rewrite if it stopped being structurally outside the corpus.
The snapshot records WHICH of them fail, so the by-design reds gate anyway: one
MORE failing is the floor being loosened, which is what buying recall with
precision looks like from here.

A fourth class, `vocab`, paraphrases suite cases into the vocabulary a session
uses BEFORE it knows the right nouns: same target, symptom words in place of
the target's own. It is reported outside the exit code by default.

Two words worth holding apart, because everything below uses both. A CLASS is
what a result line reports; a SLICE is how the snapshot files and gates it. The
four classes group into three slices: `suite` holds search + hot (one case
list, its assertion resolved per target's tier), plus `noinject` and `vocab`.
Which slices gate is config (`eval.gating_slices`).

The EXIT CODE is a diff against a committed snapshot of these outcomes rather
than the raw fail count, because a raw count cannot gate anything here: some
cases are red by design, and all of them are scored against a corpus that
changes whenever somebody writes a memory. The snapshot absorbs the first —
it records the by-design reds as the expected reds, so what gates is
MOVEMENT. A CORPUS FINGERPRINT, sha256 over every store's contents, absorbs
the second, and it is what makes a red here mean one thing:

  fingerprint MATCHES the snapshot's — the corpus is the one that was
      baselined, so an outcome that moved in a gating slice moved because the
      TOOL moved. It gates.
  fingerprint DIFFERS — a memory was written, edited, retired or retiered
      since the baseline, so nothing measured here is attributable to the
      tool. EVERY mismatch reports as DRIFT and nothing gates — and the run
      REFUSES, non-zero, pointing at --update-snapshot for a human who has
      looked at what moved. It exited 0 until 2026-08-21, which made "this
      run gated nothing" and "this run gated everything and found nothing
      wrong" the same answer to CI; on the consumer being measured then, the
      first was the commoner state by an order of magnitude.

So a red on a bump PR (corpus untouched) is always the tool, and a memory
edit is never falsely red. Position — the tier a target sits in today — is
recorded and reported, but it does NOT decide attributability: "the target
did not move, therefore the tool did" is false, since three new memories can
outrank a target that never budged.

Usage:
  memory-eval                      # run the configured suite, score, gate
  memory-eval -v                   # + show what WAS retrieved on misses
  memory-eval --hook X             # score a copy with one constant changed
  memory-eval --repo Y             # score another checkout's stores
  memory-eval --all-stores         # every store whatever the cwd
  memory-eval --snapshot F         # gate against F, not the configured one
  memory-eval --update-snapshot    # re-baseline, deliberately
Exit code = failures in the gating slices, or a refusal (a corpus that moved
under the snapshot, a gating slice that compared nothing, an unreadable or
absent snapshot); 0 = gated and clean, so it can gate CI. Every way of NOT
gating is non-zero, which is the property that makes a green here mean
something.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

from memkit._exec import Untrusted, _execute, enforce_execution_boundary
from memkit.memory_prompt_recall import (
    CONFIG_ENV,
    ConfigError,
    _utf8,
    load_config,
)

# The packaged hook, and the default for --hook. `--repo` moves the STORES;
# the hook that scores them is memkit's own, because a tree's stores scored by
# some other tree's hook is a pair no snapshot can describe.
STOCK_HOOK = pathlib.Path(__file__).with_name("memory_prompt_recall.py")


def store_roots(cfg, repo: pathlib.Path) -> list[pathlib.Path]:
    """Every configured store's directory under one root.

    N stores, in config order, which is the order retrieval interleaves them
    in — the KTD10 property: nothing here knows how many stores there are or
    what they are called.

    All of them resolve under `repo` rather than under their own configured
    roots, and that is the eval's whole `--repo` contract: a run that edits
    descriptions in one checkout and scores them against another checkout's
    copies reports every corpus change as a no-op. Production is right to
    read the live copy; a gate is not.
    """
    return [repo / store.dir for store in cfg.stores]


def corpus_fingerprint(cfg, repo: pathlib.Path) -> str:
    """One digest over every one of `repo`'s stores — the fact that decides
    whether a mismatch is the tool's or the corpus's.

    Content-addressed (relative path plus the sha256 of the bytes, sorted, each
    store labelled and folded together), so a clean checkout and the read-only
    copy a CI check runs from hash the same, while any memory written, edited,
    renamed, retiered or retired hashes differently. Tiers are inside the
    paths, so a hot/->search/ move registers even though the bytes did not
    change.

    Only `*.md` is hashed, because only `*.md` is indexed: a stray .DS_Store
    must not be able to switch the gate into its non-gating regime.

    Untracked memories are the one asymmetry to know about. They count here and
    not in a sealed source snapshot, so re-baselining with one sitting in a
    store leaves CI hashing a different corpus — green, and gating nothing.

    Both encodes below are the hook's total one, and neither is hygiene. A
    store id comes out of `json.load`, which turns an escaped `\\udXXX` in the
    config into a lone surrogate; a relative path comes out of `rglob`, which
    turns a filename the filesystem holds as undecodable bytes into one. A
    strict `.encode()` raises `UnicodeEncodeError` on either, and this function
    runs before the gate does anything — so one such name anywhere under a
    store root would end the run with a traceback and no eval, rather than with
    a fingerprint that separates that corpus from every other. Digesting
    `surrogatepass` bytes separates exactly the corpora a strict encode would
    have separated; it simply also answers for the ones it dies on.
    """
    digest = hashlib.sha256()
    for store, root in zip(cfg.stores, store_roots(cfg, repo), strict=True):
        digest.update(_utf8(f"{store.id}\0"))
        if not root.is_dir():
            continue
        files = sorted(
            ((p.relative_to(root).as_posix(), p) for p in root.rglob("*.md")),
            key=lambda entry: entry[0],
        )
        for rel, path in files:
            content = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update(_utf8(f"{rel}\0{content}\0"))
    return digest.hexdigest()


def case_record(
    status: str, file: str | None = None, position: str | None = None
) -> dict:
    """One snapshot row, built in the one place so every site agrees on shape.

    Keys in the order the diff is read: which target, where it lives, what
    happened. A class whose cases name no file omits both of the first two
    rather than writing nulls — a null compares equal to a missing key anyway,
    and reads as a target that was looked for and not found.

    Every row carries a `status`, including the skips: a target retired to
    archive/ used to record position only, so its status compared None to None
    and passed forever after. Retirement is a transition worth seeing.
    """
    row: dict = {}
    if file is not None:
        row["file"] = file
    if position is not None:
        row["position"] = position
    row["status"] = status
    return row


def cases_from_config(cfg) -> dict:
    """The consumer's case lists, normalised to {slice: [ {prompt, file?} ]}.

    Three slices, no more: `suite` (each case a prompt plus the BASENAME of the
    memory it is about — the tier resolver finds where that file lives now),
    `noinject` (prompts that must inject nothing; no file, no position), and
    `vocab` (paraphrases of suite cases, same targets). A slice the config
    omits is simply empty, which the vacuity check below then refuses to
    accept as a gating pass.
    """
    out: dict[str, list[dict]] = {"suite": [], "noinject": [], "vocab": []}
    for slice_, raw in (cfg.eval_cases or {}).items():
        if slice_ not in out:
            raise ConfigError(f"{cfg.path}: eval.cases has unknown slice {slice_!r}")
        for case in raw:
            prompt = case.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ConfigError(f"{cfg.path}: a {slice_} case has no prompt")
            if slice_ != "noinject" and not case.get("file"):
                raise ConfigError(
                    f"{cfg.path}: {slice_} case {prompt[:40]!r} names no file"
                )
            out[slice_].append(case)
    return out


def load_hook(path: pathlib.Path):
    """Import a hook file as a module, refusing one copied without its data.

    The `--hook` A/B is normally run against a copy of the hook with one
    constant changed, and the copy is the whole point: it is the only way to
    hold everything else fixed. But the hook resolves common-words.txt beside
    __file__ and _common_words() falls back to an EMPTY stopword set when the
    file is missing instead of raising, so a lone .py silently keeps every
    stopword as a search term and scores a different retriever than the one
    you meant to test. That cost three bucket runs on 2026-08-13; it is a
    hard error here rather than a caveat in a docstring.
    """
    spec = importlib.util.spec_from_file_location("recall_hook", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    words = getattr(mod, "COMMON_WORDS_FILE", None)
    if words and not pathlib.Path(words).exists():
        raise RuntimeError(
            f"{path} has no {pathlib.Path(words).name} beside it — copy the "
            "hook's whole directory, not just the .py"
        )
    return mod


def stores(cfg, repo: pathlib.Path) -> list[pathlib.Path]:
    """The store roots this run WILL ACTUALLY SEARCH from this cwd.

    A store can be cwd-gated: run from outside the repo that gates it, the hook
    never searches it, and every case targeting it then fails for a reason that
    has nothing to do with retrieval quality. Reading every configured store
    unconditionally resolved files the run could never return and reported them
    as MISSes — a suite whose result depended on which directory you were
    standing in, and which said nothing about that.

    The gate is asked of the CONFIG (`searched_stores`, which counts worktrees
    via the hook's own predicate) and then applied to `repo`'s copies, rather
    than reading `_search_dirs()` back, because that function answers for the
    live roots by construction.
    """
    searched = {store.id for store in cfg.searched_stores()}
    roots = [
        repo / store.dir for store in cfg.stores if store.id in searched
    ]
    return [p for p in roots if p.is_dir()]


def search_dirs(hook, roots: list[pathlib.Path]) -> list[str]:
    """What to hand recall(dirs=...): each store's `search/` subtree.

    recall() takes overridden dirs verbatim — it does not apply _search_root
    itself — so the tier pruning has to happen here or the run indexes hot/
    and scores the abstention cases against a corpus production never sees.
    """
    return [hook._search_root(str(p)) for p in roots]


def all_stores(cfg, repo: pathlib.Path) -> list[pathlib.Path]:
    """Every store that exists, searched from here or not — so a case can be
    told apart from a case this cwd cannot serve."""
    return [p for p in store_roots(cfg, repo) if p.is_dir()]


def locate(roots: list[pathlib.Path], name: str) -> tuple[str, pathlib.Path] | None:
    """(tier, path) for a case's file, from where it lives now.

    The first path component under the store IS the tier, so a file moved
    between hot/ and search/ since the case was written is followed without
    editing the suite. A store not yet laid out by tier (file directly in the
    root) reads as `search`, matching the hook's own untiered fallback.
    """
    for root in roots:
        for path in sorted(root.rglob(name)):
            rel = path.relative_to(root).parts
            return (rel[0] if len(rel) > 1 else "search"), path
    return None


def pointers(hook, prompt: str, hits: list[str]) -> tuple[list[str], list[str]]:
    """(everything that survives the floor, what the hook would POINT AT).

    Both are basenames, best-first; the second is the first cut to MAX_HITS.

    recall() returns PRE-floor hits; the injection path then drops whatever
    _passes_floor rejects. Scoring the post-floor set is what makes both
    assertions honest: a search case cannot pass on a hit that production
    would floor, and a hot case cannot pass on an abstention that was really
    a floor away from leaking.

    Then the CAP, which is the difference between "retrieved" and "injected":
    the hook's main() stops filling its pointer list at MAX_HITS, so a file
    sitting at post-floor rank 3 is retrieved and never shown. Uncapped, this
    function scored post-floor lists up to 18 entries long and a case could
    pass on a memory no session would ever see — one case was passing on a
    hit sitting at post-floor rank 4, four lines below anything injected.

    This is what MRR@2 reduces to here. A ranked metric like MRR only earns
    its arithmetic when the consumer sees a ranked LIST; this consumer sees
    two lines and no more, so reciprocal rank has exactly two nonzero values
    and the honest question collapses to membership of the top MAX_HITS.
    Widening MAX_HITS widens this automatically — the cap is read from the
    hook, not restated.

    The cap costs the hot class some sensitivity, knowingly: re-running the
    teeth check that first justified the hot cases (drop `hot` from the hook's
    EXCLUDE_DIRS, force _search_root back to the store root) flipped 7 of 9
    hot cases rather than 9 on the corpus it was measured against — the other
    two sat at post-floor rank 5 and 3, retrieved but never shown. That is
    the correct weaker claim: production prints two lines, so "would not be
    injected" is the only abstention the eval can honestly assert.

    Session dedup is deliberately not modelled: production also subtracts
    paths already injected this session, which only ever makes it inject
    LESS, so a search case that passes here could still be deduped away in a
    long session, and an abstention case that passes here passes a fortiori.
    """
    terms = list(dict.fromkeys((hook.build_query(prompt) or "").split()))
    passed = [
        pathlib.Path(h).name
        for h in hits
        if hook._passes_floor(*hook._relevance(terms, h, hook._LEX_ROOT.get(h, "")))
    ]
    return passed, passed[: hook.MAX_HITS]


# The weakest this gate may be, whatever the brief set says, and the smallest
# population a rate over it may be taken from.
#
# The thresholds live beside the briefs so a number travels with what it was
# measured over — and that is exactly why they cannot be the only thing
# deciding how strict the gate is. A fixture edit setting `min_served: 0.0` and
# `max_injected: 1.0` leaves both comparisons unable to fail, and the run still
# prints two rates and exits 0, which reads identically to a gate that held.
# Same for the populations: delete the negative half and precision is a rate
# over nothing, which the arithmetic reports as zero leakage.
#
# So the file may be STRICTER than these and never looser, and the bounds are
# in code where loosening them is a diff somebody reads rather than a fixture
# edit nobody does.
LONG_BRIEF_MIN_SERVED_FLOOR = 0.6
LONG_BRIEF_MAX_INJECTED_CEILING = 0.2
LONG_BRIEF_MIN_CASES = 6
# The slice's name in `eval.gating_slices` and in the snapshot. One spelling,
# because a config naming it is what makes the per-case rows gate AND what
# makes a missing brief directory a refusal rather than a note — two facts that
# must not be able to disagree about which slice they are about.
LONG_BRIEF_SLICE = "longbrief"


def long_brief_set(root: pathlib.Path) -> dict:
    """The paired brief set at `root`: briefs read off disk, plus the two rates
    they gate on.

    Files rather than config entries because a brief is kilobytes of prose, and
    the rates sit beside them rather than in the config for the same reason the
    corpus fingerprint sits in the snapshot: a number is only worth what it was
    measured over, so it travels with the thing it was measured over.

    Refuses rather than warns on a set that cannot gate. Every check below has
    the same shape as the vacuity check further down — a run that gated nothing
    and a run that gated everything and found nothing wrong must not print the
    same exit code.
    """
    where = root / "index.json"
    index = json.loads(where.read_text(encoding="utf-8"))
    for key in ("min_served", "max_injected"):
        value = index.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"{where}: no {key} rate to gate on")
        if value != value or value in (float("inf"), float("-inf")):
            raise RuntimeError(f"{where}: {key} is not a finite rate")
    if index["min_served"] < LONG_BRIEF_MIN_SERVED_FLOOR:
        raise RuntimeError(
            f"{where}: min_served {index['min_served']} is below the "
            f"{LONG_BRIEF_MIN_SERVED_FLOOR} floor this gate is allowed to be "
            "set at — a coverage bar this low cannot fail"
        )
    if index["max_injected"] > LONG_BRIEF_MAX_INJECTED_CEILING:
        raise RuntimeError(
            f"{where}: max_injected {index['max_injected']} is above the "
            f"{LONG_BRIEF_MAX_INJECTED_CEILING} ceiling this gate is allowed "
            "to be set at — an injection bar this high cannot fail"
        )
    # A CASE is a distinct brief, and the floors below count cases. Entries
    # were counted instead, so twelve copies of one row satisfied a bar written
    # to mean twelve briefs — a rate re-measuring one passing case, at full
    # strength, while every regression in the rest of the corpus went unscored.
    # Same for one brief in both halves: a case asserting two opposite outcomes
    # and scoring whichever half asks.
    seen: dict[str, str] = {}
    texts: dict[str, tuple[str, str]] = {}
    bodies: dict[str, str] = {}
    for half in ("served", "unserved"):
        entries = index.get(half, [])
        if not isinstance(entries, list):
            raise RuntimeError(f"{where}: the {half} half is not a list of cases")
        for case in entries:
            if not isinstance(case, dict) or not isinstance(case.get("brief"), str):
                raise RuntimeError(
                    f"{where}: a {half} case has no `brief` path: {case!r:.80}"
                )
            # Resolved against the root and required to stay under it: `brief`
            # is joined onto this directory, so `..` or an absolute path reads
            # a file nobody reviewing this directory can see.
            rel = case["brief"]
            full = (root / rel).resolve()
            if not full.is_relative_to(root.resolve()):
                raise RuntimeError(
                    f"{where}: {half} case `{rel}` resolves outside {root}"
                )
            if not full.is_file():
                raise RuntimeError(f"{where}: {half} case `{rel}` is not a file")
            key = str(full)
            if key in seen:
                where_first = seen[key]
                raise RuntimeError(
                    f"{where}: `{rel}` is in both halves"
                    if where_first != half
                    else f"{where}: the {half} half names the same brief twice: {rel}"
                )
            seen[key] = half
            # And by CONTENT, which is what the invariant above actually says.
            # Uniqueness by resolved path counts two filenames holding one
            # brief as two cases — the same population inflation the path check
            # exists to prevent, one copy command away.
            # VERBATIM. Production receives a brief exactly as the harness
            # sent it, and `_task_emission` measures the whole payload against
            # the hook's write bound — so a loader that quietly trimmed the
            # fixture measured a brief the fixture does not contain, and near
            # that bound could score served where production refuses. The
            # committed file has to BE those bytes, which is a thing a fixture
            # author can fix once rather than a difference the gate hides on
            # every run.
            bodies[key] = full.read_text(encoding="utf-8")
            if bodies[key] != bodies[key].strip():
                raise RuntimeError(
                    f"{where}: `{rel}` has leading or trailing whitespace. The "
                    "gate measures the bytes on disk, so the file has to be "
                    "exactly the brief — strip it in the fixture, not here"
                )
            digest = hashlib.sha256(_utf8(bodies[key])).hexdigest()[:12]
            if digest in texts:
                first_rel, first_half = texts[digest]
                raise RuntimeError(
                    f"{where}: `{rel}` and `{first_rel}` are the same brief "
                    f"under two names ({half}/{first_half})"
                )
            texts[digest] = (rel, half)
        if len(entries) < LONG_BRIEF_MIN_CASES:
            raise RuntimeError(
                f"{where}: the {half} half has "
                f"{len(entries)} case(s), under the "
                f"{LONG_BRIEF_MIN_CASES} a rate can be taken over. A coverage "
                "floor with no served briefs, or an injection ceiling with no "
                "unserved ones, is a rate over an empty population"
            )
    def _read(case: dict) -> dict:
        brief = bodies[str((root / case["brief"]).resolve())]
        # The snapshot key carries a digest of the brief, so an EDITED brief
        # reads as a new case and its old row as a stale one rather than
        # quietly inheriting a recorded outcome. The config's own cases get
        # this for free — their key is the prompt text — and a case keyed on a
        # filename alone would be the one kind of drift nothing reports: the
        # corpus fingerprint does not cover this directory, because these are
        # the queries and not the corpus.
        digest = hashlib.sha256(_utf8(brief)).hexdigest()[:12]
        return {
            "name": f"{case['brief']}#{digest}",
            "file": case.get("file"),
            "brief": brief,
            # Whether this brief is the one that drives the cap. Carried
            # through rather than dropped: the slice reads it to decide
            # whether to assert the cap's consequence, and a key silently lost
            # here is a gate that runs on nothing and says so nowhere.
            "over_cap": bool(case.get("over_cap")),
        }
    return {
        "min_served": float(index["min_served"]),
        "max_injected": float(index["max_injected"]),
        "served": [_read(c) for c in index.get("served", [])],
        "unserved": [_read(c) for c in index.get("unserved", [])],
    }


# What the long-brief slice reaches for on the hook module. Named here rather
# than probed for with one symbol, which is what the probe used to be: the
# surface below it grew to ten names and a keyword, so a copy carrying
# `task_gate` and nothing else — the immediately preceding commit of this
# branch qualified — passed the probe and then died mid-run with an uncaught
# AttributeError, after the suite slice had already printed its PASS lines.
# Exit 1 is reserved for a gate failing, so a crash and a real regression were
# the same signal to CI.
#
# A HAND-WRITTEN LIST IS THE FAILURE MODE, so it is not trusted to be one:
# the pin walks what the guarded block actually reaches and requires this
# tuple to cover it. Two names here are reached in ways that walk cannot see
# and are deliberate — `_pointer_line`, checked below for its `over_brief`
# keyword rather than called, and `_task_framed`, which the hook's own
# `_task_block` calls — so the relation is coverage rather than equality.
TASK_SURFACE = (
    "task_gate",
    "build_task_query",
    "recall",
    "_eligible",
    "_task_floor",
    "_task_framed",
    "_pointer_line",
    "_task_block",
    "_task_emission",
    # The one this list had already drifted past. `_task_delivery` calls it
    # while reconciling what the emission carried against what it picked, so
    # a hook whose copy renamed it took the gap check's "no gap" answer and
    # died one call later with the AttributeError this constant is here to
    # turn into a skip. Derived from the walk rather than trusted, by
    # test_the_task_surface_declares_every_hook_name_the_slice_reaches.
    "_display_path",
    "TASK_MAX_HITS",
    "TASK_BUDGET_SECONDS",
)


def task_surface_gap(hook) -> str | None:
    """The first thing `task_delivery` needs and this hook has not got.

    None when the whole surface is there. A keyword is checked as well as a
    name: a copy can carry `_pointer_line` without the `over_brief` argument
    this slice passes it, and the failure is the same uncaught AttributeError
    one call later.
    """
    for name in TASK_SURFACE:
        if getattr(hook, name, None) is None:
            return name
    params = inspect.signature(hook._pointer_line).parameters
    if "over_brief" not in params:
        return "`over_brief` argument to _pointer_line"
    return None


# How much the non-brief part of a real Agent payload is assumed to weigh, in
# characters. `_task_emission` measures the WHOLE serialized object against the
# hook's write bound, and production echoes back whatever keys the harness
# sent — so a gate whose payload is smaller than production's scores a brief as
# served at a size production would refuse. The three keys below are the ones
# the Agent tool requires; this pads them out so the gate's payload is no
# SMALLER than a real one, which is the direction that keeps the gate
# conservative. It cannot be exact — the harness owns that shape — so the
# assumption is named here rather than left in the shape of a stub.
TASK_INPUT_ASSUMED_OVERHEAD = 1024


# A pointer line, up to the first field memkit renders AFTER the path.
# Non-greedy, so it stops at the first of them: the path field is memkit's own
# rendering of a filesystem path and everything past it is file content.
#
# TWO SEPARATORS, because the em-dash one is CONDITIONAL. `_pointer_line`
# renders ` — {desc}` only when there is a description, and `_description`
# returns "" for a memory with neither `description:` frontmatter nor a `# `
# heading, and on OSError. Anchored on the em-dash alone such a line did not
# parse and its name was dropped from the delivered set — which on the SERVED
# half undercounts and fails loudly, and on the UNSERVED half, whose test is
# `ok = not shown`, scored a pointer that really did reach an unattended
# subagent as BRIEF-QUIET. A leak certified as clean. The ` [` evidence tag is
# unconditional, so it is the anchor that always exists.
_POINTER_PATH = re.compile("^- (?P<path>.+?) \u2014 ")
# The fallback, tried only when the line has no em-dash at all. Separate
# patterns rather than one alternation because a non-greedy `.+?` stops at
# whichever alternative appears EARLIEST, so a path that itself contains
# ` [` — a legal filename — would be cut short on a line that also has a
# description. Anchored on `[matches `, which `_pointer_line` renders
# unconditionally, rather than on a bare bracket.
_POINTER_BARE = re.compile("^- (?P<path>.+?) \\[matches ")


def _delivered_names(appended: str) -> set[str]:
    """The basenames the block's pointer lines actually POINT AT.

    Off the path field alone. Taking every whitespace token's basename made
    any word of a surviving DESCRIPTION able to vouch for a pointer that was
    shed or never emitted, and descriptions here are file contents — a memory
    that names its neighbour is ordinary, not contrived. The gate would then
    report subagent coverage for a pointer the subagent never received, which
    is the single thing this slice exists to measure.

    A line whose separator was consumed does not parse, and so does not count
    as a delivery. That is the answer rather than a gap: the reader of such a
    line is handed a filename with somebody's sentence welded to it, and
    scoring it as delivered would be scoring the failure as a success.

    STORE-RELATIVE, not the bare basename. Two configured stores may hold
    different files under one name, and a basename comparison then lets a
    pointer to the wrong store's file satisfy a case the target was never
    delivered for — an incorrect memory selection scored as a correct one.
    The name is kept alongside it because the expectations are written as
    basenames; the caller decides which identity it is asking about.
    """
    return {pathlib.Path(p).name for p in _delivered_paths(appended)}


def _delivered_paths(appended: str) -> set[str]:
    """The PATH field of every pointer line, as rendered.

    The identity the gate compares on, because a basename is not one: two
    configured stores may hold different files under the same name, and a
    pointer to the wrong store's file then satisfies a case whose target was
    never delivered — an incorrect memory selection scored as a correct one.
    """
    paths = set()
    for line in appended.splitlines():
        match = _POINTER_PATH.match(line) or _POINTER_BARE.match(line)
        if match is not None:
            paths.add(match.group("path"))
    return paths


def _pad_to_overhead(tool_input: dict) -> dict:
    """Weigh the non-brief part up to `TASK_INPUT_ASSUMED_OVERHEAD`, or raise.

    The pad lands on the WHOLE serialized object rather than on one key's
    value, because the whole object is what production weighs: an assumption
    spelled as the length of `description` stops being the assumption the
    moment this input grows a key.

    RAISES rather than clipping, which is the half that was missing. The
    clip's silence is the failure mode: `max(spare, 0)` turns "this shape has
    outgrown the constant" into "no padding today", and the gate goes on
    running, green, with a payload SMALLER than production's — which is
    exactly the direction that lets a brief score `served` at a size
    production refuses with `task:oversize`. This invariant has already
    drifted once, and what caught it was a person reading the diff.

    A gate is allowed to fail loudly; the hook it measures is not. Nothing
    here runs on the every-prompt path.
    """
    weight = len(json.dumps(dict(tool_input, prompt=""), ensure_ascii=False))
    spare = TASK_INPUT_ASSUMED_OVERHEAD - weight
    if spare < 0:
        raise ValueError(
            f"the assumed Agent payload weighs {weight} characters, past "
            f"TASK_INPUT_ASSUMED_OVERHEAD={TASK_INPUT_ASSUMED_OVERHEAD}; the "
            "gate's payload would be smaller than production's, so it would "
            "score briefs served at sizes production refuses"
        )
    tool_input["description"] += "." * spare
    return tool_input


def task_delivery(hook, brief: str, dirs: list[str]) -> dict:
    """Everything one brief's trip through the task path produced, as a
    record — which is what a subagent WOULD ACTUALLY RECEIVE for this brief.

    Every stage is the task path's own — its gate, its query builder, its floor
    bars — because that is the whole subject: a slice scored through
    `build_query` and the prompt path's bars would measure a retriever no
    subagent ever meets, and would report the shape of the population this
    exists to prove is served.

    And it runs to the EMISSION rather than stopping at retrieval, which is the
    difference between "the ranker found it" and "the subagent got it". The
    task path can retrieve perfectly and deliver nothing: an `updatedInput`
    that fails the output-shape allowlist is refused whole, and so is a brief
    whose emission crosses the write bound — and a slice that stopped at the
    floor would score both as served. Reading the names back OUT of the
    emitted bytes rather than off the picks is what makes that true rather
    than merely intended.

    A gated brief returns nothing, which is the same answer as no hits and is
    correct here: the question this slice asks is what reaches the subagent.

    ONE RECORD RATHER THAN A LIST OF NAMES, and both halves of the slice
    read it. The names are what was delivered; `unanswerable` is whether
    the corpus could answer at all, and dropping it is how a broken index
    came to satisfy the injection ceiling — on the leakage half an index
    that could not answer and a brief that was correctly quiet produce the
    same empty list. The cap's consequence is here too: how many hits
    cleared the floor, how many the cap kept, and the bytes the emission
    decided to write. All from ONE trip, because two trips is two
    populations.
    """
    return _task_delivery(hook, brief, dirs)


def entrypoint_delivery(
    hook_file: pathlib.Path,
    config: pathlib.Path | str,
    cwd: pathlib.Path,
    brief: str,
) -> tuple[set[str], str]:
    """What the REAL hook PROCESS delivers for one brief: (names, why-not).

    Every other stage this slice drives is the task path's own function, which
    is right and is not enough — none of them is the registered entry point.
    `main`'s event dispatch, the tool-name check, the ledger write, the signal
    handlers and the stdout delivery all sit between a correct `_task_block`
    and a subagent that actually receives it, and a break in any of them
    leaves the real hook emitting no `updatedInput` while this slice goes on
    reporting served coverage.

    ONE brief per run rather than all of them: the question is whether the
    entry point still reaches the pipeline the rest of the slice measures, and
    that is answered by one trip. The cost is one subprocess.

    Isolated state and a fresh `tool_use_id`, so the run leaves no ledger
    behind and cannot be deduped by one.
    """
    payload = {
        "session_id": "memkit-eval-entrypoint",
        "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_use_id": f"toolu_eval{os.getpid()}",
        "tool_input": {
            "prompt": brief,
            "description": "score this brief",
            "subagent_type": "general-purpose",
        },
    }
    with tempfile.TemporaryDirectory() as state:
        # Through `_exec._execute`, the package's one process start. The
        # environment is BUILT there rather than inherited — `CHILD_ENV_KEEP`
        # plus these three — which is what the boundary is for and costs this
        # call nothing: the hook reads its config from `MEMKIT_CONFIG`, its
        # state directory from `XDG_CACHE_HOME`, and `$HOME` is kept.
        try:
            out = _execute(
                [sys.executable, "-B", str(hook_file)],
                input=json.dumps(payload),
                timeout=180,
                cwd=str(cwd),
                env_extra={
                    "XDG_CACHE_HOME": state,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    CONFIG_ENV: str(config),
                },
            )
        except (OSError, subprocess.SubprocessError, Untrusted) as exc:
            return set(), f"the hook process did not run ({type(exc).__name__})"
    if out.returncode != 0:
        return set(), f"the hook process exited {out.returncode}"
    if not out.stdout.strip():
        return set(), "the hook process emitted nothing"
    try:
        updated = json.loads(out.stdout)["hookSpecificOutput"]["updatedInput"]
        delivered = updated["prompt"]
    except (ValueError, KeyError, TypeError) as exc:
        return set(), f"the emission was not an updatedInput ({exc})"
    appended = (
        delivered[len(brief) :] if delivered.startswith(brief) else delivered
    )
    return _delivered_names(appended), ""


def over_cap_faults(hook, case: dict, got: dict) -> list[str]:
    """What is wrong with the way the cap was applied to this brief, if
    anything.

    The one thing about the task path that only a brief of this shape can
    exercise. `TASK_MAX_HITS` is 3 and every other fixture brief clears the
    floor on one memory, so until this case existed the eval never drove the
    cap at all: the truncation sentence and the block that carries it were
    covered by production's own tests and by the eval and production sharing a
    spelling, which is a source tripwire rather than a case. A future change
    that keeps the spelling and stops writing the sentence — or writes the
    wrong count into it — would pass everything else here.

    Three claims, all read out of the bytes the emission decided to write
    rather than off the picks: the cap bound, the count of what it dropped is
    the difference it actually dropped, and the sentence saying so reached the
    subagent.
    """
    faults = []
    if got["unanswerable"]:
        # The index could not answer, which is already refused above by its own
        # name. Reporting it here as well would say the corpus or the brief has
        # moved, which is a wrong diagnosis of a right refusal.
        return faults
    if got["eligible"] <= hook.TASK_MAX_HITS:
        faults.append(
            f"{case['name']} cleared the floor on {got['eligible']} "
            f"memories, at or under the cap of {hook.TASK_MAX_HITS} — the case "
            "exists to drive the cap and no longer does, so the corpus or the "
            "brief has moved and the truncation path is ungated again"
        )
        return faults
    if got["picks"] != hook.TASK_MAX_HITS:
        faults.append(
            f"{case['name']} showed {got['picks']} pointers with "
            f"{got['eligible']} eligible; the cap is {hook.TASK_MAX_HITS}"
        )
    dropped = got["eligible"] - got["picks"]
    if got["truncated"] != dropped:
        faults.append(
            f"{case['name']} reported {got['truncated']} truncated with "
            f"{dropped} actually dropped"
        )
    plural = "match" if dropped == 1 else "matches"
    sentence = f"{dropped} further {plural} not shown"
    if sentence not in got["delivered"]:
        faults.append(
            f"{case['name']} delivered a block that does not say "
            f"`{sentence}` — the cap bound and the subagent was not told"
        )
    return faults


def _task_delivery(hook, brief: str, dirs: list[str]) -> dict:
    """The one trip. See `task_delivery` above for why each stage is the task
    path's own."""
    empty = {
        "names": [],
        "eligible": 0,
        "picks": 0,
        "truncated": 0,
        "delivered": "",
        "unanswerable": 0,
    }
    # WHERE PRODUCTION'S CLOCK STARTS, which is before the gate and the query
    # builder rather than at the search. `main` stamps `t0` and hands it down,
    # so what those two stages spend comes out of the budget the search then
    # runs under; a clock started at the search hands retrieval a budget
    # production has already spent part of, and reports pointers production
    # abandons. The measured divergence on the fixture corpus is 1.4-3.2 ms of
    # 7,000, which is the number this stops depending on.
    t0 = time.monotonic()
    if hook.task_gate(brief) is not None:
        return empty
    query = hook.build_task_query(brief)
    # THE BUDGET PRODUCTION RUNS UNDER. `recall`'s `deadline` defaults to None,
    # which is unlimited — so against a consumer's own store under `--repo` or
    # `--all-stores` the gate could wait for pointers production abandons and
    # report them as served. The fixture corpus is far too small for the two to
    # differ, which is why this only ever showed up by reading it.
    # `stats`, for the reason production reads it. `recall` suppresses a
    # per-dir failure and returns the other dirs' hits, so a corpus that could
    # not ANSWER and a corpus with nothing to say arrive here identically — and
    # production splits them, into `task:index-unavailable` and `task:nomatch`.
    # A gate that does not split them scores an index that could not be read as
    # a retriever that found nothing, which is a quality number reporting an
    # infrastructure failure. That window is the normal case on this path:
    # parallel spawns share one index, and every contender that loses a cold
    # build's write-lock race meets an index with no committed rows.
    rec: dict = {}
    hits = hook.recall(
        brief,
        stats=rec,
        dirs=dirs,
        query=query,
        deadline=t0 + hook.TASK_BUDGET_SECONDS,
    )
    unanswerable = int(rec.get("errs_lex") or 0)
    empty = dict(empty, unanswerable=unanswerable)
    terms = list(dict.fromkeys((query or "").split()))
    # `_eligible` with the hook's own bars, not a comprehension with a copy of
    # them: this slice is the only automated gate over the task path's
    # relevance, so a second spelling of the floor here is a gate that can
    # silently score a retriever no subagent meets.
    eligible, _floored = hook._eligible(hits, terms, **hook._task_floor())
    if not eligible:
        return empty
    # THE HOOK'S OWN CAP AND ITS CONSEQUENCE, not a second copy. This slice
    # took its own slice of the eligible list and built the frame with the
    # truncation count defaulted to zero, so on any brief the cap binds on it
    # scored a block SMALLER than the one production writes — missing the
    # truncation sentence, against the write bound this slice exists to gate.
    block, picks, truncated = hook._task_block(eligible)
    if not picks:
        return empty
    # The tool input a spawn actually carries: every key the Agent tool
    # requires, so the allowlist is exercised on a realistic shape rather than
    # on a one-key stub that could never fail it — and weighing at least what a
    # real one weighs, so the byte budget this measures is not smaller than the
    # budget production measures. See TASK_INPUT_ASSUMED_OVERHEAD.
    tool_input = {
        "prompt": brief,
        "description": "score this brief",
        "subagent_type": "general-purpose",
    }
    _pad_to_overhead(tool_input)
    # THE HOOK'S OWN EMISSION DECISION, not a second copy of it. This slice
    # re-derived it — `_task_payload`, then its own size test, and no
    # encodability test at all — so a divergence between the two was invisible
    # to the one gate over subagent delivery, and a brief the hook would refuse
    # to write scored as served.
    text, _verdict, _size = hook._task_emission(tool_input, block)
    if text is None:
        return empty
    delivered = json.loads(text)["hookSpecificOutput"]["updatedInput"]["prompt"]
    # ONLY THE APPENDED PART, and only its pointer lines. `updatedInput.prompt`
    # is the brief this slice supplied plus the block, so `name in delivered`
    # could be satisfied by the brief's own text — a gate its own input can
    # pass. No shipped fixture names a corpus file today, which made that
    # latent rather than wrong, and one edit to a fixture away from a gate that
    # answers yes to an empty block.
    appended = delivered[len(brief) :] if delivered.startswith(brief) else delivered
    carried = _delivered_paths(appended)
    return {
        # Matched on the RENDERED path, which is the identity, and reported as
        # a basename, which is how the expectations are written.
        "names": [
            pathlib.Path(path).name
            for path, _, _ in picks
            if hook._display_path(path) in carried
        ],
        "eligible": len(eligible),
        "picks": len(picks),
        "truncated": truncated,
        "delivered": appended,
        "unanswerable": unanswerable,
    }


def read_snapshot(path: pathlib.Path, require_fingerprint: bool = True) -> dict | None:
    """The committed expectations — {"corpus": digest, "cases": slice -> prompt
    -> record}; None if absent.

    Cases are keyed by the prompt itself rather than an index or a hash: the
    file is read in a review diff, and a case reordered or reworded should
    show up there as the case it is.

    A gating run demands the fingerprint rather than shrugging at a file that
    predates it, because unattributable is the NON-gating regime: read
    leniently, a snapshot with no digest is a permanently green check that
    never says it stopped looking. A run that is about to overwrite the file
    passes require_fingerprint=False — the refusal would otherwise name
    --update-snapshot as the fix and then reject it.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, dict):
        raise RuntimeError(f"{path} has no `cases` object — regenerate it")
    corpus = data.get("corpus")
    if not isinstance(corpus, str):
        if require_fingerprint:
            raise RuntimeError(
                f"{path} has no `corpus` fingerprint — regenerate it with "
                "--update-snapshot"
            )
        corpus = None
    return {"corpus": corpus, "cases": cases}


def write_snapshot(
    path: pathlib.Path, cases: dict[str, dict[str, dict]], corpus: str
) -> None:
    """Rewrite the snapshot from a run, in suite order and readably.

    ensure_ascii=False and no key sort: the point of this file is that a
    human reads its diff, and \\u2014-escaped prompts sorted away from their
    neighbours are a file that only a machine can review.
    """
    body = {
        "note": (
            "Expected outcomes of `memory-eval` on this checkout's memory "
            "stores. Regenerate with --update-snapshot, "
            "deliberately, after reading what moved; a diff here is either a "
            "corpus edit you meant or a retrieval regression you did not."
        ),
        "corpus_note": (
            "sha256 over every store's *.md contents when these outcomes were "
            "recorded. A run that hashes the same corpus can attribute a "
            "moved outcome to the retriever, and gates on it; a run that "
            "hashes a different one reports every mismatch as drift."
        ),
        "corpus": corpus,
        "cases": cases,
    }
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", "utf-8")


def verdict(
    seen: dict, want: dict | None, corpus_matches: bool = True
) -> tuple[str, str]:
    """One case against its snapshot: (ok|new|drift|regression, why).

    `corpus_matches` is the attribution rule, and the only thing that decides
    whether a failure can be pinned on the tool: the stores this run measured
    either hash to what the snapshot was written from or they do not. If they
    do not, every mismatch — a moved status, a case the snapshot never heard
    of — demotes to drift for a human to re-baseline.

    `file` and `position` are compared before status, and each answers a
    question the case's assertion silently rests on: which target the case
    names at all, and which tier that target sits in today (search asserts
    injection, hot asserts abstention). Either one moving makes the recorded
    status an answer to a different question, so it is drift in both regimes
    — a retargeted case is a change in what is being asserted, not in the
    thing asserted about.

    Cases whose class names no file carry neither field, so both sides read
    None and the comparison falls through to status.
    """
    if want is None:
        kind, why = "new", "no expectation recorded"
    elif seen.get("file") != want.get("file"):
        kind, why = (
            "drift",
            f"snapshot says {want.get('file')}, case now names {seen.get('file')}",
        )
    elif seen.get("position") != want.get("position"):
        kind, why = (
            "drift",
            f"snapshot says {want.get('position')}, now {seen.get('position')}",
        )
    elif seen.get("status") != want.get("status"):
        kind, why = "regression", f"snapshot says {want.get('status')}"
    else:
        return "ok", ""
    if kind != "drift" and not corpus_matches:
        return "drift", f"{why}; corpus changed since the baseline"
    return kind, why


def main() -> None:
    # See the note in memory_integrity.main: the module docstring is a
    # maintainer's document and argparse's default formatter reflows its usage
    # list into one run-on line.
    ap = argparse.ArgumentParser(
        prog="memory-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Score retrieval against your own cases and gate on a snapshot.\n"
            "\n"
            "The cases live in your config under `eval`; memkit ships none,\n"
            "because a case pairs a prompt with the filenames it should\n"
            "surface and those are your memories. Each run scores the\n"
            "configured slices and compares them against the committed\n"
            "snapshot, so what fails is a CHANGE in behaviour rather than an\n"
            "absolute score."
        ),
        epilog=(
            "typical use:\n"
            "  memory-eval                      run the configured suite and gate\n"
            "  memory-eval -v                   + per-case detail\n"
            "  memory-eval --update-snapshot    accept this run as the baseline\n"
            "\n"
            "exit codes:\n"
            "  0  the gating slices held — or a snapshot was written, which is\n"
            "     an acceptance and exits 0 even on a red run\n"
            "  1  a gating slice regressed, or the run could not start. The\n"
            "     message names which, and what to do about it."
        ),
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=f"config naming the stores and the cases (default: ${CONFIG_ENV})",
    )
    ap.add_argument(
        "--repo",
        type=pathlib.Path,
        default=None,
        help="checkout whose memory stores to score (default: the config's "
        "eval.root)",
    )
    ap.add_argument(
        "--hook",
        type=pathlib.Path,
        default=None,
        help="hook file to score; a copy with one constant changed is the "
        f"only honest way to A/B a constant (default: {STOCK_HOOK.name})",
    )
    ap.add_argument(
        "--all-stores",
        action="store_true",
        help="score every one of --repo's stores whatever the cwd — what a CI "
        "check passes, since a build sandbox stands outside every checkout",
    )
    ap.add_argument(
        "--snapshot",
        type=pathlib.Path,
        default=None,
        help="expectations to gate against (default: the config's eval.snapshot "
        "under --repo)",
    )
    ap.add_argument(
        "--update-snapshot",
        action="store_true",
        help="rewrite the expectations from this run — the only thing that "
        "ever writes them, so a re-baseline is always somebody's decision",
    )
    args = ap.parse_args()

    # The eval is an operator's instrument, not the prompt path: it honours the
    # per-root environment overrides the hook refuses, and it refuses to run
    # with no config at all rather than scoring an empty corpus green.
    try:
        cfg = load_config(args.config, honor_env_overrides=True)
        if cfg is None:
            sys.exit(
                "memory-eval: no config — pass --config PATH or set "
                f"${CONFIG_ENV}. The cases and the stores both live there."
            )
        cases = cases_from_config(cfg)
    except ConfigError as exc:
        sys.exit(f"memory-eval: {exc}")
    if cfg.eval_snapshot is None:
        sys.exit(f"memory-eval: {cfg.path} names no eval.snapshot to gate against")

    repo = (args.repo or pathlib.Path(cfg.root(cfg.eval_root or "self"))).resolve()
    hook_file = (args.hook or STOCK_HOOK).resolve()
    hook = load_hook(hook_file)
    # --all-stores bypasses the cwd gate rather than emulating it. That gate
    # answers which SESSIONS are served a gated store, which is not a fact
    # about retrieval, and the run already reports a store it cannot reach
    # rather than scoring it. A build sandbox stands outside every checkout, so
    # a gating run without this scores only the ungated cases and files the
    # rest as drift — a green earned by not looking.
    every = all_stores(cfg, repo)
    permitted = stores(cfg, repo)
    roots = every if args.all_stores else permitted
    if args.all_stores and len(roots) != len(cfg.stores):
        sys.exit(
            f"--all-stores found {len(roots)} store(s) under {repo}, "
            f"wanted {len(cfg.stores)}"
        )
    dirs = search_dirs(hook, roots)
    unsearched = [p for p in every if p not in roots]
    # Stores this run reads that PRODUCTION would refuse from this cwd. The
    # long-brief slice is the only gate over subagent delivery, and it hands
    # `dirs` straight to `recall` — so a target sitting in a cwd-gated store
    # makes the served floor pass on a memory the real `_task_main` would
    # answer `task:nodirs` for in the same environment. `--all-stores` is a
    # reporting mode; it may not also be the thing that gates.
    ungated = [p for p in roots if p not in permitted]
    snap_path = args.snapshot or repo / cfg.eval_snapshot
    corpus = corpus_fingerprint(cfg, repo)
    prior = read_snapshot(snap_path, require_fingerprint=not args.update_snapshot)
    # Whether this run can attribute anything to the tool. No snapshot at all
    # reads as "cannot": --update-snapshot is then the only legal next move.
    corpus_matches = prior is not None and prior["corpus"] == corpus
    # Annotated, because the long-brief slice narrows it below and the
    # inferred type is a frozenset of whatever literals the default happened to
    # carry.
    gating: frozenset[str] = cfg.eval_gating
    if ungated and LONG_BRIEF_SLICE in gating:
        # Said out loud, because every way of not having this gate is
        # otherwise silent and a green run has to name the gates it ran. The
        # slice still RUNS and still prints its rates; what it stops doing is
        # deciding the exit code, since it would be deciding it on a delivery
        # production refuses from this cwd.
        print(
            "long briefs: --all-stores is reading "
            + ", ".join(str(p) for p in ungated)
            + ", which this cwd is gated out of — reporting only, not gating"
        )
        gating = frozenset(s for s in gating if s != LONG_BRIEF_SLICE)
    # Say what this run measured. Four of these lines are the difference
    # between "the hook missed" and "you ran the suite from somewhere the hook
    # does not look" or "you scored a corpus nobody baselined".
    print(f"config: {cfg.path}")
    print(f"hook:   {hook_file}  (MAX_HITS={hook.MAX_HITS})")
    print(f"cwd:    {pathlib.Path.cwd()}")
    print(f"stores: {', '.join(dirs) or '(none)'}")
    if unsearched:
        print(
            "        not searched from this cwd: "
            + ", ".join(str(p) for p in unsearched)
        )
    print(f"snap:   {snap_path}")
    if prior is None:
        regime = "no snapshot yet"
    elif corpus_matches:
        regime = "matches the snapshot — gating slices answer for the tool"
    elif prior["corpus"] is None:
        regime = "the snapshot records none — nothing gates"
    else:
        regime = (
            f"DIFFERS from the snapshot's {prior['corpus'][:12]} — "
            "the corpus moved, so nothing gates"
        )
    print(f"corpus: {corpus[:12]} ({regime})")
    print()
    # Both refusals guard the same failure: a snapshot is only worth what the
    # run that wrote it measured. A run that cannot reach a store records its
    # cases as unsearched and permanently narrows the gate to whatever the cwd
    # could see, and a run against a modified hook records the CANDIDATE's
    # behaviour as the baseline the candidate is supposed to be judged against.
    if args.update_snapshot and unsearched:
        sys.exit(
            "refusing to write a snapshot from a run that cannot search "
            + ", ".join(str(p) for p in unsearched)
            + " — rerun from inside the checkout being scored"
        )
    if args.update_snapshot and hook_file != STOCK_HOOK.resolve():
        sys.exit("refusing to write a snapshot scored against a --hook copy")
    if prior is None and not args.update_snapshot:
        sys.exit(
            f"no expectations at {snap_path} — nothing to gate against; "
            "run --update-snapshot and commit the result"
        )

    # class -> [passed, total]. `noinject` is not a tier: its cases name no
    # file, so nothing about them is resolved from the stores.
    scored = {"search": [0, 0], "hot": [0, 0], "noinject": [0, 0]}
    skipped = 0
    seen_cases: dict[str, dict[str, dict]] = {
        "suite": {}, "noinject": {}, "vocab": {}, LONG_BRIEF_SLICE: {}
    }
    tally = {"regression": 0, "drift": 0, "new": 0}
    # A name nobody has is a refusal, not a KeyError. `gating_slices` is
    # hand-typed — README tells an adopter to add `longbrief` to it — and the
    # vacuity check below indexes `compared` with whatever the config carries,
    # so a typo ended a fully green run with a traceback and exit 1, which CI
    # reads as the regression that did not happen. Refused here rather than
    # made lenient there: `compared.get(s, 0)` would stop the crash by counting
    # a typo as a satisfied gate, which is the failure the vacuity check exists
    # to prevent.
    unknown = sorted(gating - set(seen_cases))
    if unknown:
        sys.exit(
            f"{cfg.path}: eval.gating_slices names {', '.join(unknown)} — "
            f"no such slice; the slices are {', '.join(sorted(seen_cases))}"
        )
    # Per SLICE, how many cases actually met a recorded expectation. Counted
    # because "0 failures" and "0 comparisons" print the same exit code, and
    # the second is a gate that stopped looking — see the vacuity check below.
    compared = dict.fromkeys(seen_cases, 0)
    gate_fails = 0
    prior_cases = prior["cases"] if prior else {}

    def against_snapshot(slice_: str, prompt: str, seen: dict) -> str:
        """Record one case's outcome, diff it, and return the line's tail.

        The tail rides on the case's own line rather than in a block at the
        end because the two facts are read together: which case moved, and
        which of the four ways it moved.
        """
        nonlocal gate_fails
        seen_cases[slice_][prompt] = seen
        want = prior_cases.get(slice_, {}).get(prompt)
        kind, why = verdict(seen, want, corpus_matches)
        # Only these two answered the snapshot's question — the rest are the
        # snapshot declining to answer, and the vacuity check counts them as
        # such however they exit.
        if kind in ("ok", "regression"):
            compared[slice_] += 1
        if kind == "ok":
            return ""
        tally[kind] += 1
        # `new` gates alongside `regression`: an unrecorded case in a gating
        # slice is a case nobody baselined, and letting it pass makes adding
        # one the way to add an ungated case. The sanctioned path is
        # --update-snapshot in the same change. Under a moved corpus neither
        # ever reaches here — verdict() has already demoted them to drift.
        if kind in ("regression", "new") and slice_ in gating:
            gate_fails += 1
            return f"  <- {kind.upper()} ({why})"
        return f"  <- {kind.upper()} ({why}; not gating)"

    for case in cases["suite"]:
        prompt, expected = case["prompt"], case["file"]
        found = locate(roots, expected)
        if found is None or found[0] == "archive":
            # An unfound case is one of two different things, and scoring them
            # alike hid the difference: the memory is gone (fix the suite), or
            # it lives in a store this cwd does not search (fix nothing —
            # rerun from the repo).
            elsewhere = locate(unsearched, expected) if found is None else None
            if elsewhere:
                why = f"in {elsewhere[1].parent}, not searched from this cwd"
                tail = ""
                position = "unsearched"
            else:
                why = "archived" if found else "no such file in either store"
                tail = "; drop the case"
                position = "archive" if found else "absent"
            moved = against_snapshot(
                "suite", prompt, case_record("SKIP", expected, position)
            )
            print(f"[SKIP] {prompt[:58]:<58} -> {expected} ({why}{tail}){moved}")
            skipped += 1
            continue
        tier = "hot" if found[0] == "hot" else "search"
        try:
            hits = hook.recall(prompt, dirs=dirs)  # abs paths, best-first
        except AttributeError:
            sys.exit(
                "hook has no recall(prompt) entrypoint — expose one "
                "(the __main__ path should call it too)"
            )
        names = [pathlib.Path(h).name for h in hits]
        passed, shown = pointers(hook, prompt, hits)
        if tier == "hot":
            ok = expected not in shown
            mark = "ABSTAIN-OK" if ok else "ABSTAIN-FAIL"
        else:
            ok = expected in shown
            mark = "PASS" if ok else "MISS"
        # Three ways to miss, and they call for different fixes: never
        # retrieved (query/description), retrieved then floored (floor), or
        # retrieved and above the floor but ranked out of the pointer slots
        # (rank — usually a description competing badly against neighbours).
        note = ""
        if not ok and tier == "search":
            if expected in passed:
                note = f" (post-floor rank {passed.index(expected) + 1}, past MAX_HITS)"
            elif expected in names:
                note = " (retrieved but FLOORED)"
        scored[tier][0] += int(ok)
        scored[tier][1] += 1
        moved = against_snapshot("suite", prompt, case_record(mark, expected, found[0]))
        print(f"[{mark:<12}] {prompt[:58]:<58} -> {expected}{note}{moved}")
        if args.verbose and not ok and not note:
            print(f"       got: {shown or '(nothing)'}")

    for case in cases["noinject"]:
        prompt = case["prompt"]
        _, shown = pointers(hook, prompt, hook.recall(prompt, dirs=dirs))
        ok = not shown
        mark = "NOINJECT-OK" if ok else "NOINJECT-FAIL"
        scored["noinject"][0] += int(ok)
        scored["noinject"][1] += 1
        moved = against_snapshot("noinject", prompt, case_record(mark))
        # A leak names what leaked, always — not behind -v. "must inject
        # nothing" fails identically for every prompt, so the line as it
        # stood ("-> (nothing)" on a FAIL) reported the expectation and
        # withheld the only fact that tells one leak from another, which is
        # the fact you need to decide whether a floor is wrong or a
        # description is.
        got = f"injected {shown}" if shown else "(nothing)"
        print(f"[{mark:<12}] {prompt[:58]:<58} -> {got}{moved}")

    served, vocab_tot = 0, 0
    for case in cases["vocab"]:
        prompt, expected = case["prompt"], case["file"]
        found = locate(roots, expected)
        if found is None or found[0] != "search":
            # A hot or archived twin makes the case meaningless rather than
            # failing: the hook excludes those by design, so "not injected"
            # would say nothing about retrieval.
            moved = against_snapshot(
                "vocab",
                prompt,
                case_record("SKIP", expected, found[0] if found else "absent"),
            )
            print(
                f"[VOCAB-SKIP  ] {prompt[:58]:<58} -> {expected} "
                f"(not in search/){moved}"
            )
            continue
        # dirs=dirs, like every other call site here. Without it this slice
        # measured the DEFAULT stores while the three above measured --repo's,
        # so a --repo run printed one scoreboard over two different corpora and
        # named neither.
        hits = hook.recall(prompt, dirs=dirs)
        _, shown = pointers(hook, prompt, hits)
        got = expected in shown
        vocab_tot += 1
        served += int(got)
        # Overlap against the target is printed, not asserted in a comment: a
        # description rewrite can hand a case the vocabulary it was written to
        # withhold, and then it silently stops testing anything. Zero is the
        # DESIGNED value — these prompts are worded as symptoms, so every term
        # they share with the target ("reboot", "pool", "write") is common —
        # and a case that drifts above zero has stopped being a vocabulary-gap
        # case whether or not it still passes.
        terms = list(dict.fromkeys((hook.build_query(prompt) or "").split()))
        matched, _, _ = hook._relevance(terms, str(found[1]))
        common = hook._common_words()
        n = len([t for t in matched if t.lower() not in common])
        mark = "VOCAB-FOUND" if got else "VOCAB-MISS"
        moved = against_snapshot("vocab", prompt, case_record(mark, expected, found[0]))
        print(
            f"[{mark:<12}] {prompt[:58]:<58} -> {expected} "
            f"(retrieved {len(hits)}, distinctive overlap {n}){moved}"
        )

    # The long-brief slice. Its cases are FILES rather than config entries, its
    # scoring is the task path's rather than the prompt path's, and it gates on
    # two RATES rather than on the snapshot — see below for why it needs both.
    served_hit = leaked = 0
    cap_fail: list[str] = []
    unanswered: list[str] = []
    briefs = None
    if not cfg.eval_long_briefs:
        if LONG_BRIEF_SLICE in gating:
            # The config has SAID it wants subagent delivery gated, and there
            # is nothing to run. Printing a line and exiting 0 made that state
            # — a config predating the key, a typo in it, or a newer config
            # read by an older memkit that drops what it does not know — a
            # green eval over a task path that could be completely broken.
            # Asking for a gate that cannot run is a refusal, not a note.
            sys.exit(
                f"{cfg.path}: eval.gating_slices names `{LONG_BRIEF_SLICE}` "
                "and eval.long_briefs names no brief directory, so the only "
                "gate over subagent delivery cannot run"
            )
        # Otherwise said out loud, because every way of not having this gate is
        # otherwise silent, and a green run has to say which gates it ran.
        print("\nlong briefs: eval.long_briefs is not configured — slice skipped")
    else:
        brief_root = repo / cfg.eval_long_briefs
        if not (brief_root / "index.json").is_file():
            sys.exit(
                f"eval.long_briefs names {brief_root}, which has no index.json"
            )
        missing = task_surface_gap(hook)
        if missing is not None:
            # A hook with no task path, or with an older shape of it. That is a
            # legitimate A/B subject — scoring an older build as "served
            # nothing" would report the absence of a feature as a quality
            # regression — and it is NOT a legitimate state for the hook this
            # repo ships: a regression that deletes or renames any of it would
            # otherwise skip the only gate over it and exit 0.
            if hook_file == STOCK_HOOK.resolve():
                sys.exit(
                    f"{hook_file} has no `{missing}`, so the long-brief slice "
                    "cannot run — and this is the shipped hook rather than a "
                    "--hook copy, so the feature the slice gates is missing "
                    "rather than merely older"
                )
            print(
                f"\nlong briefs: this hook has no {missing} — slice skipped"
            )
            # An A/B subject that cannot RUN this slice is not a run that
            # failed to gate it. The operator named another hook on purpose,
            # and the vacuity check at the end would otherwise refuse every
            # A/B against a build older than the task path — turning the
            # documented `--hook` workflow into an error on the exact class of
            # copy it exists for. The SHIPPED hook took the `sys.exit` above
            # and never reaches this.
            gating = frozenset(s for s in gating if s != LONG_BRIEF_SLICE)
        else:
            try:
                briefs = long_brief_set(brief_root)
            except RuntimeError as exc:
                # A refusal rather than a traceback: exit 1 is documented as a
                # gate failing or a refusal, and a stack trace makes a
                # malformed fixture index look like a crash in the tool.
                sys.exit(f"memory-eval: {exc}")
            print()
            entrypoint_checked = False
            for case in briefs["served"]:
                got = task_delivery(hook, case["brief"], dirs)
                shown = got["names"]
                ok = case["file"] in shown
                if shown and not entrypoint_checked:
                    # ONCE, on the first brief this slice actually delivered
                    # for: everything above is the task path's own functions,
                    # and none of them is the registered ENTRY POINT. A break
                    # in the dispatch, the tool-name check or the stdout
                    # delivery leaves the real hook emitting nothing while
                    # this slice reports coverage.
                    entrypoint_checked = True
                    live, why = entrypoint_delivery(
                        hook_file,
                        # RESOLVED, because the child runs from `repo` and
                        # `cfg.path` is as the operator typed it: a relative
                        # `--config` names nothing from there, and the hook
                        # answers that by being inert — which reads here as
                        # the entry point being broken.
                        pathlib.Path(cfg.path).resolve(),
                        repo,
                        case["brief"],
                    )
                    if why or not (live & set(shown)):
                        unanswered.append(
                            f"{case['name']}: the hook PROCESS delivered "
                            f"{sorted(live) or '(nothing)'} where this slice's "
                            f"own pipeline delivered {sorted(shown)} — "
                            + (why or "the entry point and the pipeline it "
                               "measures do not agree")
                        )
                if got["unanswerable"] and not ok:
                    # NOT a miss. An index that could not answer says nothing
                    # about the retriever, and scoring it as a miss puts an
                    # infrastructure failure into a coverage rate — quietly,
                    # since the row and the rate look exactly like a gate that
                    # stopped serving. Refused instead: the run says which
                    # corpus could not answer and exits non-zero, which is what
                    # production does with the same fact under another name.
                    unanswered.append(
                        f"{case['name']} scored against an index that could not "
                        f"answer ({got['unanswerable']} dir(s) failed to search) "
                        "— an unanswerable corpus is not a retrieval miss, and "
                        "this run cannot say anything about coverage"
                    )
                served_hit += ok
                mark = "BRIEF-SERVED" if ok else "BRIEF-MISS"
                if got["unanswerable"] and not ok:
                    mark = "BRIEF-NOINDEX"
                moved = against_snapshot(
                    LONG_BRIEF_SLICE, case["name"], case_record(mark, case["file"])
                )
                print(
                    f"[{mark:<12}] {case['name'][:58]:<58} -> "
                    f"{case['file']} (got {shown or '(nothing)'}){moved}"
                )
                if case.get("over_cap"):
                    cap_fail.extend(over_cap_faults(hook, case, got))
            for case in briefs["unserved"]:
                got = task_delivery(hook, case["brief"], dirs)
                shown = got["names"]
                ok = not shown
                leaked += not ok
                mark = "BRIEF-QUIET" if ok else "BRIEF-LEAK"
                if got["unanswerable"] and ok:
                    # The same refusal as the served half, on the OPPOSITE
                    # outcome. An index that could not answer injects nothing,
                    # and injecting nothing is exactly what a correctly quiet
                    # brief looks like — so on this half the unattributable
                    # result is the CLEAN one, and counting it certifies the
                    # injection ceiling against a corpus that was never
                    # searched. That ceiling is this suite's only bound on what
                    # the task path says to an unattended subagent.
                    unanswered.append(
                        f"{case['name']} scored against an index that could not "
                        f"answer ({got['unanswerable']} dir(s) failed to search) "
                        "— an unanswerable corpus is not a quiet one, and this "
                        "run cannot say anything about leakage"
                    )
                    mark = "BRIEF-NOINDEX"
                moved = against_snapshot(
                    LONG_BRIEF_SLICE, case["name"], case_record(mark)
                )
                saw = f"injected {shown}" if shown else "(nothing)"
                print(f"[{mark:<12}] {case['name'][:58]:<58} -> {saw}{moved}")

    # A case deleted from a list up there leaves its expectation behind, and a
    # stale expectation is the one kind of drift no case line can report —
    # nothing iterates it any more.
    for slice_, want in prior_cases.items():
        for prompt in want:
            if prompt not in seen_cases.get(slice_, {}):
                tally["drift"] += 1
                print(
                    f"[DRIFT       ] {prompt[:58]:<58} -> in the snapshot's "
                    f"{slice_} slice, not in the suite; not gating"
                )

    ret, tot = scored["search"]
    abst, atot = scored["hot"]
    quiet, qtot = scored["noinject"]
    print(f"\nsearch tier: {ret}/{tot} retrieved")
    print(f"hot tier:    {abst}/{atot} correctly not injected")
    print(f"no-inject:   {quiet}/{qtot} correctly injected nothing")
    print(f"combined:    {ret + abst + quiet}/{tot + atot + qtot} passed")
    if skipped:
        print(f"skipped:     {skipped} (unscored)")
    print(
        f"vocab slice: {served}/{vocab_tot} symptom-worded prompts served by the "
        "lexical stage — an instrument, gated only if eval.gating_slices says so"
    )
    # The rates, and whether they hold. Two of them, because either alone is
    # met by a gate that does nothing: a coverage floor on its own is satisfied
    # by a path that serves every brief, and an injection ceiling by one that
    # serves none. The pair is what makes the numbers a calibration rather than
    # a count — "non-vacuous" bounds neither a gate too strict for real briefs
    # nor one too loose for irrelevant ones.
    rate_fail: list[str] = list(cap_fail) + list(unanswered)
    if briefs is not None:
        # No `or 1` denominator guard: `long_brief_set` refuses a half that
        # cannot carry a rate, so an empty population is a refusal rather than
        # a division this has to survive. The guard was the bug — it turned
        # "there are no negative briefs" into "nothing leaked".
        served_rate = served_hit / len(briefs["served"])
        leak_rate = leaked / len(briefs["unserved"])
        print(
            f"long briefs: {served_hit}/{len(briefs['served'])} served "
            f"({served_rate:.3f}, floor {briefs['min_served']:.3f}); "
            f"{leaked}/{len(briefs['unserved'])} leaked "
            f"({leak_rate:.3f}, ceiling {briefs['max_injected']:.3f})"
        )
        if served_rate < briefs["min_served"]:
            rate_fail.append(
                f"long-brief coverage {served_rate:.3f} is under the "
                f"{briefs['min_served']:.3f} floor — the task gate is refusing "
                "briefs it was calibrated to serve"
            )
        if leak_rate > briefs["max_injected"]:
            rate_fail.append(
                f"long-brief injection {leak_rate:.3f} is over the "
                f"{briefs['max_injected']:.3f} ceiling — the task gate is "
                "rewriting spawns the corpus has nothing to say about"
            )
    loose = tally["regression"] + tally["new"] - gate_fails
    parts = [f"{gate_fails} gating failure(s) in {'/'.join(sorted(gating))}"]
    if loose:
        parts.append(f"{loose} outside the gate")
    if tally["drift"]:
        parts.append(f"{tally['drift']} drifted (the corpus moved under the case)")
    if tally["new"]:
        parts.append(f"{tally['new']} unrecorded (newer than the snapshot)")
    print("vs snapshot: " + ", ".join(parts))
    if rate_fail and not args.update_snapshot:
        # Ahead of the corpus-moved refusal, and deliberately: that refusal
        # says nothing was attributable, which is true of every SNAPSHOT
        # comparison and false of these. A rate is an absolute measurement of
        # the corpus in front of it, so a moved corpus is exactly when it still
        # answers — and exactly when a coverage collapse would otherwise be
        # filed as drift and re-baselined away.
        sys.exit("; ".join(rate_fail))
    if prior is not None and not corpus_matches and not args.update_snapshot:
        print(
            "             these stores are not the ones baselined, so every "
            "line above is drift and nothing was gated"
        )
        # And that is a REFUSAL, not a pass. Everything above is right — under
        # a moved corpus nothing measured here is attributable to the tool, so
        # nothing may gate — but exiting 0 on it made "this run gated nothing"
        # and "this run gated everything and found nothing wrong" the same
        # answer to CI, and the first one is by far the commoner. Measured on
        # the consumer at the time this changed: 88 memory-touching commits in
        # 30 days against 3 re-baselines ever, so the check spent most of its
        # life inert while reporting green.
        #
        # Consumer impact, deliberately: nix-config's `memory-eval` check and
        # `check-all` will now fail on any memory edit that does not re-baseline
        # in the same change. That is the contract its own MEMORY.md already
        # states ("re-baseline with --update-snapshot and commit the snapshot in
        # the same change") being enforced rather than waived, and the remedy is
        # one command that historically moves only the fingerprint line.
        #
        # A consumer BUMPING to this build should carry a fresh
        # --update-snapshot in the same change as the bump. Whatever drift
        # accumulated while this exited 0 is standing and invisible, and the
        # first run on the new build surfaces all of it at once — on whichever
        # PR happens to move the input, which is rarely the one expecting it.
        sys.exit(
            "corpus moved — re-baseline with `--update-snapshot` and commit "
            "the snapshot in the same change"
        )
    # The pointer fires on ANY unclean run, regressions included: a regression
    # is sometimes the outcome you meant (a floor deliberately loosened), and
    # the re-baseline is how you say so — leaving it off the failing case made
    # the fix look like it had no sanctioned path.
    if any(tally.values()) and not args.update_snapshot:
        print("             --update-snapshot accepts these, once you know why")
    if args.update_snapshot:
        write_snapshot(snap_path, seen_cases, corpus)
        # Exit 0 even on a red run: re-baselining is the act of accepting what
        # the run reported, and a nonzero exit here would make the accepted
        # state indistinguishable from a refusal to write.
        print(f"wrote {snap_path}")
        # Except the rate floors, which a re-baseline may not accept. The
        # snapshot records WHAT HAPPENED and accepting it is the whole point;
        # the rates record what has to be true whatever happened, and a floor
        # that `--update-snapshot` can silence is not a floor. The write still
        # lands first, so the remedy for a moved corpus is not blocked by this
        # — the run just does not report success.
        sys.exit("; ".join(rate_fail) if rate_fail else 0)
    # A gating slice that compared nothing is the failure mode a green cannot
    # show: zero failures and zero comparisons print the same exit code, and
    # an empty or missing slice would otherwise buy a pass by having no
    # expectations to fail. Only asked in the attributable regime — under a
    # moved corpus every slice compares nothing BY DESIGN.
    vacuous = [s for s in sorted(gating) if not compared[s]]
    if vacuous and corpus_matches:
        sys.exit(
            "nothing was gated: "
            + "; ".join(
                f"the {s} slice compared 0 of {len(seen_cases[s])} case(s) "
                "against the snapshot"
                for s in vacuous
            )
            + " — re-baseline with --update-snapshot and commit the result"
        )
    sys.exit(gate_fails)


def cli() -> None:
    """The console script, which is where this process's own rules belong.

    `memory-eval` is on the adopter's PATH — a consumer's CI check runs it
    from the installed package — and this module starts a process through
    `_execute`, so it is a memkit command in exactly the sense the other two
    are. It shipped without this call because the pin that installs the
    guarantee named its subjects by hand; it derives them from
    `[project.scripts]` now.

    Not in `main()`: an audit hook cannot be removed once installed, so an
    in-process `main()` would put this process's rules on every later caller
    in the same interpreter.
    """
    enforce_execution_boundary()
    main()


if __name__ == "__main__":
    cli()
