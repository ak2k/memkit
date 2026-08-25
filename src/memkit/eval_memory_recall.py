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
import json
import pathlib
import sys

from memkit.memory_prompt_recall import CONFIG_ENV, ConfigError, load_config

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
    """
    digest = hashlib.sha256()
    for store, root in zip(cfg.stores, store_roots(cfg, repo), strict=True):
        digest.update(f"{store.id}\0".encode())
        if not root.is_dir():
            continue
        files = sorted(
            ((p.relative_to(root).as_posix(), p) for p in root.rglob("*.md")),
            key=lambda entry: entry[0],
        )
        for rel, path in files:
            content = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update(f"{rel}\0{content}\0".encode())
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
        if hook._passes_floor(*hook._relevance(terms, h))
    ]
    return passed, passed[: hook.MAX_HITS]


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
    roots = every if args.all_stores else stores(cfg, repo)
    if args.all_stores and len(roots) != len(cfg.stores):
        sys.exit(
            f"--all-stores found {len(roots)} store(s) under {repo}, "
            f"wanted {len(cfg.stores)}"
        )
    dirs = search_dirs(hook, roots)
    unsearched = [p for p in every if p not in roots]
    snap_path = args.snapshot or repo / cfg.eval_snapshot
    corpus = corpus_fingerprint(cfg, repo)
    prior = read_snapshot(snap_path, require_fingerprint=not args.update_snapshot)
    # Whether this run can attribute anything to the tool. No snapshot at all
    # reads as "cannot": --update-snapshot is then the only legal next move.
    corpus_matches = prior is not None and prior["corpus"] == corpus
    gating = cfg.eval_gating
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
    seen_cases: dict[str, dict[str, dict]] = {"suite": {}, "noinject": {}, "vocab": {}}
    tally = {"regression": 0, "drift": 0, "new": 0}
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
    loose = tally["regression"] + tally["new"] - gate_fails
    parts = [f"{gate_fails} gating failure(s) in {'/'.join(sorted(gating))}"]
    if loose:
        parts.append(f"{loose} outside the gate")
    if tally["drift"]:
        parts.append(f"{tally['drift']} drifted (the corpus moved under the case)")
    if tally["new"]:
        parts.append(f"{tally['new']} unrecorded (newer than the snapshot)")
    print("vs snapshot: " + ", ".join(parts))
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
        sys.exit(0)
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
    main()


if __name__ == "__main__":
    cli()
