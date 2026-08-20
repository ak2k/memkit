#!/usr/bin/env python3
"""UserPromptSubmit hook: inject pointers to relevant memory files.

Lexical retrieval over the git-tracked memory corpora named by the config
file (see load_config below): an in-process SQLite FTS5 index (BM25 over
heading-delimited markdown sections, ~3 ms warm) over every configured store,
minus the ones whose `cwd_gate` says this session is standing outside them.

There was a second stage, `ck --sem` over bge-small embeddings, on the theory
that embeddings rescue the vocabulary-mismatch prompts term overlap cannot
reach. Deleted 2026-08-13 after a three-arm experiment measured it rescuing
nothing: under the shipped trigger it never fired on a prompt with a real
subject, 21 of 23 known-good targets turned out to be lexically retrieved
already and lost on RANK rather than on vocabulary, and stubbing it out left
the eval bit-identical. The lever it was reached for is lexical rank and
MAX_HITS. That verdict was measured on ONE single-author corpus and carries
reopen conditions; the README's disclosure section states both.

Injects POINTERS, not content: up to MAX_HITS lines of `path — description`,
where description comes from the file's frontmatter. The model Reads the full
file if relevant. Content injection is the documented context-pollution
failure mode (claude-mem v3); pointers cost ~40 tokens per hit.

The same retrieval is available on demand — `memory-recall --search
"<terms>"` — because MAX_HITS is a budget, not a verdict: a prompt gets the
best MAX_HITS pointers, and everything the floor let through beyond them is
named only as a count the agent can go collect. It is also the answer to the
question the per-prompt hook cannot serve, since it sees the user's prompt
and never the task the agent drifted into three tool calls later.

Searches the SEARCH tier only. Hot-tier memories live in `<store>/hot/` and
are already in context via the auto-loaded MEMORY.md, so pointing at them
spends the injection budget on something the model has already read — `hot`
is in EXCLUDE_DIRS for that reason.

Session dedup: paths already injected in this session are suppressed via a
state file keyed on the hook payload's session_id, which also caps the
session's pointer LEDGER at POINTER_BUDGET — past it a stronger hit displaces
the weakest already spent (see _replace), so the count of lines rendered into
a session is not itself bounded. Trigger-driven injection exists
because model-discretion recall does not happen in practice: left to its own
judgement the model simply does not go looking.

Fail-open by construction: any error, missing binary, or over-budget path
exits 0 with no output — a broken hook must never block a prompt. The hook
never writes memory CONTENT — a writer on the read path is how these systems
teach themselves their own output. What it
does write is disposable cache, rebuildable from the corpus at any time: the
lexical index under ~/.cache/memory-recall/. It no longer invokes ck at all.

**Must import under python 3.9.** The harness runs this file with whatever
`python3` the PATH resolves to, which on a stock macOS is /usr/bin/python3
(3.9.6) — not the flake's interpreter and not a version this repo controls.
A module-scope PEP 604 annotation (`frozenset[str] | None`) is EVALUATED at
import time there and raises TypeError, which the harness reports as nothing
at all: the hook is fail-open, so a hook that cannot even be imported looks
exactly like a corpus with nothing to say. `from __future__ import
annotations` makes every annotation in this file a string, so 3.10+ syntax
is free below this line — but a `|` union in a RUNTIME position (a cast, a
`get_type_hints` call, an isinstance) would still break, and no test on the
flake's python 3.13 would notice.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable

# --- configuration -----------------------------------------------------------
#
# ONE json file names every tree this tool reads, and all three tools —
# this hook, the integrity checker, the eval — read that one file. The reader
# lives HERE, in the file with the hardest constraints (stdlib only, importable
# under 3.9, usable as a loose script with nothing but its wordlist beside it),
# and the other two import it, so there is exactly one implementation of what a
# store is.
#
# The path arrives in MEMKIT_CONFIG, and only the path. Which directories an
# every-prompt hook reads and injects from is the memory-poisoning surface of
# the whole design, so the installed hook gets that variable baked in by a hard
# wrapper override and the VALUES inside the file are never overridable from the
# environment on the hook path: `honor_env_overrides` defaults to False here and
# is turned on only by the checker, the eval, and `--debug-config`.
#
# The CLIs additionally take the path as `--config`, which is the same one fact
# arriving by argument instead of by environment. That is an addition to how a
# PERSON or an agent points a tool at a tree, never to what the hook will read:
# the hook parses no arguments at all (see cli), so nothing an argument can say
# reaches the every-prompt path.
#
# Shipped defaults are inert. No MEMKIT_CONFIG and no file means no stores,
# which means zero pointers and exit 0 — a memkit that has not been configured
# says nothing rather than guessing at a corpus.
#
# The config is never hashed into _VERSION (see _version): a config-only change
# must not fork the soak log into two incomparable halves, and the shipped
# source stays byte-stable for the same reason.
SCHEMA = 1
CONFIG_ENV = "MEMKIT_CONFIG"
# Advertised to agents when a truncation notice names the on-demand search.
# Overridable per-config, never from the environment: it is a command string
# handed to an agent.
DEFAULT_SEARCH_CLI = "memory-recall --search"


class ConfigError(Exception):
    """A config file that is present and cannot be honoured.

    Distinct from absence, which is a legitimate state (inert). The hook is
    fail-open, so it degrades to inert and says why in the soak record; the
    checker and the eval exit non-zero with the message.
    """


class Store:
    """One memory store: where it lives, where it is edited, how it is gated."""

    __slots__ = (
        "id",
        "role",
        "dir",
        "live_root",
        "edit_root",
        "sub_indexes",
        "cwd_gate",
    )

    def __init__(self, raw: dict) -> None:
        self.id = _require_str(raw, "id", "stores[]")
        self.role = raw.get("role", "project")
        if self.role not in ("project", "personal"):
            raise ConfigError(f"stores[{self.id}].role must be project or personal")
        self.dir = _require_str(raw, "dir", f"stores[{self.id}]")
        self.live_root = _require_str(raw, "live_root", f"stores[{self.id}]")
        self.edit_root = raw.get("edit_root") or self.live_root
        self.sub_indexes = tuple(raw.get("sub_indexes") or ())
        gate = raw.get("cwd_gate")
        self.cwd_gate = gate.get("root") if isinstance(gate, dict) else None


class Config:
    """The parsed config file, with roots resolved lazily and once.

    `stores` is ORDERED and the order is a contract, not cosmetics: recall()
    interleaves hits round-robin across store dirs in this order, and the eval
    takes the first store containing a case's file. Reordering the list changes
    retrieval.
    """

    def __init__(self, path: str, raw: dict, honor_env_overrides: bool) -> None:
        schema = raw.get("schema")
        if schema != SCHEMA:
            # A reader that met a higher number and carried on would be reading
            # half a config, which for a hook that fails open is a silent
            # retrieval outage rather than an error anybody sees.
            raise ConfigError(
                f"{path}: schema {schema!r}, this build speaks {SCHEMA}"
            )
        self.path = path
        self.honor_env_overrides = honor_env_overrides
        self._roots_raw = raw.get("roots") or {}
        self._resolved: dict = {}
        self.stores = [Store(s) for s in (raw.get("stores") or [])]
        citations = raw.get("citations") or {}
        self.cited_roots = tuple(citations.get("roots") or ())
        self.extra_suffixes = tuple(citations.get("extra_suffixes") or ())
        self.blame_base = citations.get("blame_base") or "origin/main"
        self.search_cli = raw.get("search_cli") or DEFAULT_SEARCH_CLI
        ev = raw.get("eval") or {}
        self.eval_root = ev.get("root")
        self.eval_snapshot = ev.get("snapshot")
        self.eval_gating = frozenset(ev.get("gating_slices") or ("suite",))
        self.eval_cases = ev.get("cases") or {}

    def root(self, name: str) -> str:
        """Absolute path of a named root, resolved once and reported with it.

        Returns just the path; `root_source` says which route answered, which
        is the line that made a wrong-tree run visible the one time it happened.
        """
        return self.root_with_source(name)[0]

    def root_with_source(self, name: str) -> tuple:
        if name in self._resolved:
            return self._resolved[name]
        spec = self._roots_raw.get(name)
        if not isinstance(spec, dict):
            raise ConfigError(f"{self.path}: no root named {name!r}")
        answer = self._resolve(name, spec)
        self._resolved[name] = answer
        return answer

    def _resolve(self, name: str, spec: dict) -> tuple:
        # A per-root env override is declared IN the config — one variable can
        # therefore never mean two trees — and is honoured only by the tools
        # that opt in. The hook never does.
        env = spec.get("env")
        if env and self.honor_env_overrides:
            value = os.environ.get(env)
            if value:
                return os.path.realpath(os.path.expanduser(value)), env
        kind = spec.get("kind")
        if kind == "path":
            # expanduser at READ time, never pre-resolved by whoever wrote the
            # file: redirecting HOME is how the test suite and the build sandbox
            # point the whole tool at a fixture corpus.
            raw = spec.get("path")
            if not isinstance(raw, str):
                raise ConfigError(f"{self.path}: root {name!r} has no path")
            return os.path.expanduser(raw), "configured path"
        if kind == "config_relative":
            up = spec.get("up", 0)
            base = os.path.dirname(os.path.abspath(self.path))
            for _ in range(int(up)):
                base = os.path.dirname(base)
            return base, f"{up} up from the config file"
        if kind == "git_toplevel":
            try:
                out = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    return (
                        os.path.realpath(out.stdout.strip()),
                        "git toplevel of cwd",
                    )
            except (OSError, subprocess.SubprocessError):
                pass
            fallback = spec.get("fallback")
            if not fallback:
                raise ConfigError(
                    f"{self.path}: root {name!r} is git_toplevel outside a repo "
                    "and declares no fallback"
                )
            path, _ = self.root_with_source(fallback)
            return path, f"fallback to {fallback} (git unavailable)"
        raise ConfigError(f"{self.path}: root {name!r} has unknown kind {kind!r}")

    def store_dir(self, store: Store, which: str = "live") -> str:
        root = store.live_root if which == "live" else store.edit_root
        return os.path.join(self.root(root), store.dir)

    def searched_stores(self) -> list:
        """Stores this session may read, in config order.

        A store with a `cwd_gate` is searched only from inside the named root —
        including that root's git worktrees, which live outside its path prefix
        and share its git common dir.
        """
        out = []
        for store in self.stores:
            if store.cwd_gate is None or _cwd_in_root(self.root(store.cwd_gate)):
                out.append(store)
        return out


def _require_str(raw: dict, key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} is missing a {key!r} string")
    return value


@functools.lru_cache(maxsize=None)
def _cwd_in_root(root: str) -> bool:
    """True when the session cwd is inside `root` — including its git
    worktrees, which live outside the path prefix but share the git common dir.

    Cached: this forks git, and several callers ask. The cwd cannot change
    inside one hook invocation, so the cache is per-process by construction —
    but a test that chdirs between calls has to clear it
    (`_cwd_in_root.cache_clear()`).
    """
    cwd = os.getcwd()
    if cwd == root or cwd.startswith(root + os.sep):
        return True
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        common = out.stdout.strip()
        return out.returncode == 0 and os.path.realpath(common) == os.path.realpath(
            os.path.join(root, ".git")
        )
    except OSError:
        return False
    except subprocess.SubprocessError:
        return False


def load_config(path: str | None = None, honor_env_overrides: bool = False):
    """Parse the config file, or None when there is none to parse.

    `path` wins; otherwise MEMKIT_CONFIG. The environment carries the PATH and
    nothing else — that is what the module's wrapper `--set` makes
    non-ambient — while the values inside stay off-limits to the environment
    unless a caller asks for `honor_env_overrides`.

    Absence returns None (inert). A file that is present and unreadable,
    unparseable or of a schema this build does not speak raises ConfigError:
    "no config" and "a config I could not honour" are different states and only
    the first one is allowed to be silent.
    """
    if path is None:
        path = os.environ.get(CONFIG_ENV) or None
    if path is None:
        return None
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise ConfigError(f"{path}: no such config file")
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level is not an object")
    return Config(path, raw, honor_env_overrides)


@functools.lru_cache(maxsize=None)
def _config(honor_env_overrides: bool = False):
    """The hook's own config, loaded at most once per process.

    Returns None when there is none, and None when there is one this build
    cannot honour — the hook is fail-open, so a bad config degrades to inert
    and says so in the soak record rather than blocking a prompt.
    `_config_error` carries the reason for that record.
    """
    global _CONFIG_ERROR
    try:
        return load_config(_CONFIG_PATH, honor_env_overrides=honor_env_overrides)
    except ConfigError as exc:
        _CONFIG_ERROR = str(exc)
        return None


_CONFIG_ERROR: str | None = None
# The config path a CLI caller named, when one did. Left None on the hook path,
# where absence is the whole point: the installed hook reads MEMKIT_CONFIG and
# only MEMKIT_CONFIG, because that is the variable the wrapper bakes in.
_CONFIG_PATH: str | None = None


def _use_config(path: str | None) -> None:
    """Point every config read in this process at `path` (None: the env again).

    `--config` has to reach `_search_dirs` and `_search_cli`, which take no
    arguments and are reached from inside `recall()`. Threading a path through
    them would put a CLI-only parameter into two functions on the every-prompt
    path and into every test that calls them, for the sake of a value the hook
    never supplies — the same trade `_LEX_COUNTS` is a module global for.

    Setting it is what makes `--config` an alternative to exporting
    MEMKIT_CONFIG rather than a second way to spell it: the highest-traffic
    verification path stops having to mutate the environment of whatever ran
    it. Both caches are cleared because a second call in one process — the
    suite, and a doctor checking two configs in a row — must not answer from
    the first one's parse.
    """
    global _CONFIG_PATH, _CONFIG_ERROR
    _CONFIG_PATH = path
    _CONFIG_ERROR = None
    _config.cache_clear()


# Low: just a typo/accident guard. The REAL junk gate is the stopword
# filter (>=2 content words) — short prompts are where users compress to
# exactly the load-bearing tokens (a five-word question naming one host,
# was wrongly gated at the previous minimum of 6).
MIN_PROMPT_WORDS = 3
# Harness envelopes: scaffolding the harness addresses to the agent, not a
# question the user asked. On the author's corpus 14.9% of search-reaching
# traffic is one of these, and the shipped hook injects on 100% of them at the
# full MAX_HITS — mean 3.00 pointers — so a session drains POINTER_BUDGET in
# ten notifications before the user has asked anything. The terms carrying
# those matches are `output` `file` `tool` `code` `claude` `id` `task` `src`
# `summary` `tmp` `exit` `user`: the vocabulary of the scaffolding itself,
# which is why no ranking or floor change can reach this — the matches are
# real, the reader is not. Gated rather than ranked for that reason.
#
# Markers only, anchored at the start of the stripped prompt. This is the
# zero-false-positive subset of the provenance stratifier's rules; its
# heuristic cues (alpha-ratio, glyph and fence detection) are deliberately NOT
# adopted, because those exist to classify a corpus after the fact and can fire
# on an ordinary question dense in identifiers.
#
# Anchoring alone is not enough, and the claim that used to stand here — that a
# human prompt merely MENTIONING a task notification does not begin with one —
# was wrong in one shape: leading with the tag is the natural way to ask ABOUT
# it (`<some-tool> keeps timing out on the login page, can you look?`).
# 8 of 12 constructed prompts of that shape gated. Hence the completeness check
# in _is_envelope: a real envelope's opening tag owns its line or its block
# closes, and a person leading with a tag continues in prose on the same line.
#
# The marker list is a closed allowlist over a vocabulary Claude Code releases
# own, so it is extended from MEASURED transcript presence rather than from
# imagination. Counted in the author's own transcripts: `<command-message`
# (68 occurrences), `<bash-input`/`<bash-stdout` (22), `<!-- Generated by
# ce-lite converter` (175) and `[Request interrupted by user]` (1683) were all
# found in that vocabulary while slipping the predicate; `<command-message` in
# particular is the same slash-command family as the already-gated
# `<command-name`, differing only in which tag the harness emits first. The
# exhaustive replay confirms the bar holds — 34645/34645 both directions, 0 of
# 27,902 human-or-mixed prompts touched — but only after the stratifier it
# compares against was given the same five markers, which it had not been.
# Adding a marker here without adding it there does not fail: it makes the
# replay report the addition as a false positive of the hook. The tripwire for
# that is test_the_stratifier_knows_every_marker_the_hook_gates_on.
_ENVELOPE = re.compile(
    r"^(?:<(?:task-notification|teammate-message|system-reminder|command-name"
    r"|command-message|bash-input|bash-stdout"
    r"|local-command-\w+|user-prompt-submit-hook|agent-\w+|background-task\w*"
    r"|hook-\w+)\b"
    r"|<!--\s*Generated by ce-lite converter"
    r"|\[SYSTEM NOTIFICATION"
    r"|\[Request interrupted by user)",
    re.I,
)
# The opening tag of a well-formed envelope, used only for the completeness
# check below. Deliberately narrower than _ENVELOPE: the bracket markers
# (`[SYSTEM NOTIFICATION`, `[Request interrupted by user`) and the ce-lite HTML
# comment have no tag to close, so they never reach it.
_ENVELOPE_TAG = re.compile(r"^<([A-Za-z0-9_-]+)(?:\s[^>]*)?>")


def envelope_probes() -> list[str]:
    """One concrete envelope string per alternative of the marker pattern,
    synthesized FROM that pattern rather than typed out anywhere.

    Public API, and deliberately so — `# consumer:` a downstream project's
    inverse test, which owns a transcript stratifier that has to classify every
    marker this hook gates on. The alternative is that consumer re-implementing
    this synthesizer, which is the "tests that re-implement the predicate"
    anti-pattern: a hand-written list is a second copy of the marker set and
    goes stale in exactly the way the check exists to catch, by continuing to
    pass for markers nobody added it to.

    Raises on any construct it has not been taught, because a silently-skipped
    alternative is the same untested marker with a cleaner-looking suite.
    """
    pattern = _ENVELOPE.pattern
    m = re.search(r"<\(\?:(.*?)\)\\b", pattern, re.S)
    if m is None:
        raise RuntimeError(f"the marker pattern's tag group changed shape: {pattern!r}")
    alts = m.group(1).split("|")
    rest = pattern[m.end() :].lstrip("|").rstrip(")")
    alts += [a for a in rest.split("|") if a]

    probes = []
    for alt in alts:
        lit = alt.replace(r"\w+", "x").replace(r"\w*", "")
        lit = lit.replace(r"\s*", " ").replace(r"\[", "[")
        if "\\" in lit:
            raise RuntimeError(f"unhandled regex construct in {alt!r}")
        probes.append(lit if lit.startswith(("<", "[")) else f"<{lit}>")
    return probes


def _is_envelope(stripped: str) -> bool:
    """True for a harness envelope, and the only place that decision is made.

    A function rather than an inline `_ENVELOPE.match(...)` because the
    anchoring is half the contract and it is carried by TWO redundant
    mechanisms — the leading `^` and `.match` rather than `.search`. A test
    that reaches past this and applies `_ENVELOPE` itself supplies its own
    anchoring and therefore cannot observe the loss of either one; both
    mutants survive such a test while a real unanchored predicate would start
    suppressing questions ABOUT envelopes.

    The other half is completeness. A marker at the front says the prompt may
    be scaffolding; what distinguishes scaffolding from a question about
    scaffolding is that the harness emits a whole block — the opening tag owns
    its line, or the matching close appears — while a person who opens with a
    tag keeps writing prose after it on the same line. Measured against 34,570
    real messages of the author's, the two predicates gate the identical 4,735
    records, so this costs no real envelope; it recovers the paste-then-ask
    shape that anchoring alone suppressed.
    """
    if not _ENVELOPE.match(stripped):
        return False
    m = _ENVELOPE_TAG.match(stripped)
    if not m:
        return True
    first_line = stripped.split("\n", 1)[0].rstrip()
    close = f"</{m.group(1)}>"
    return first_line == m.group(0) or close.lower() in stripped.lower()


# Pointers one prompt may inject. Back to 3, the value shipped until the
# 2026-08-12 restructure dropped it to 2, on two independent measurements
# that both land past the second slot: a bucket of labelled pairs whose answer
# cleared the floor and then ranked past the cap, and the sem experiment's 23
# known-good targets, 21/23 of which were lexically retrieved and stranded by
# the cap rather than missed by vocabulary. The cap only bounds how many
# eligible hits fit; the floor still decides eligibility, so a third slot
# cannot admit anything the floor rejected. Costs ~40 tokens per pointer, so
# up to ~120 per injecting prompt.
#
# The size of that bucket has been restated twice and both earlier figures are
# wrong; see the plan's 2026-08-13 amendment for the current table. "745
# labelled follow-through misses" counted the whole labelled set, of which a
# quarter were SHOWN — successes, not misses — and the set itself was drawn by
# an oracle that recursed into subagent transcripts, harvesting agent-to-agent
# briefs the hook never runs on. What survives the correction is the ordering,
# not the magnitude: the A/B that actually justifies this constant moved 47
# pairs from TRUNCATED to SHOWN with every other bucket bit-identical.
MAX_HITS = 3
# Pointers a single session may accumulate. Dedup already stops any one path
# recurring, but a long session drifts across topics and kept paying the
# per-prompt cost; past ~30 pointers the marginal one is noise against the
# session's own context. Counted over the state file's `spent` LEDGER, not
# over everything ever shown: past the budget a hit with strictly better
# evidence displaces the weakest pointer already spent (see _replace), so the
# two numbers separate — the session keeps its best 30 rather than its first
# 30, and the ledger is what the bar is measured against.
POINTER_BUDGET = 30
# Floored candidates named in the soak record. Bounded because a two-store
# window can floor ~20 files and the log is a line-per-prompt file, but the
# cap only ever hides the tail: the reason to name them is "why was X not
# recalled", and the near-misses that prompts is rank-ordered at the front.
FLOORED_LOG_MAX = 6
FLOOR_LEX = 0.3  # drop weak-tail lex chunks (scores are top-normalized to 1.0)
# FILES the lexical stage considers per dir, taken at the file level (chunks
# are folded to files in SQL before the limit applies). The relevance floor
# was calibrated against a window this size, so widening it silently re-tunes
# the floor: more candidates move the best score the floor is measured
# against.
CANDIDATE_LIMIT = 10
# The harness's own kill, restated here so the arithmetic below can be
# checked against it. It is the harness's registered `timeout` for this hook
# (seconds); the consumer's own test asserts the two still agree, because
# nothing else connects a number in a settings file to a number in this one.
HARNESS_TIMEOUT = 15
# Wall-clock this hook may spend before it must produce whatever it has. Set
# BELOW the harness timeout on purpose: at the harness's number the hook is
# killed mid-flight and the soak log — the only instrument that would show
# the overrun — records nothing, so the failure this budget exists to catch
# is invisible exactly when it happens. Three seconds of headroom covers the
# interpreter start, the FTS sync, and the write of the record itself.
#
# A whole-hook budget rather than per-call timeouts because per-call timeouts
# bound one call and the hook makes one per corpus, so the count grows with
# every corpus added. Retrieval no longer spawns anything, so what the
# deadline now covers is interpreter start, the one `git rev-parse`, the FTS
# sync (unbounded on a cold corpus) and the record write — and it stays
# because it is the only instrument that would show an overrun at all.
BUDGET_SECONDS = 12

# Relevance floor (soak review 2026-07-12 + Zipf extension 2026-07-19).
# Coincidence injections match on terms that are COMMON
# ENGLISH but rare in the corpus ("see", "fix", "improvements") — so
# corpus statistics can't catch them (BM25's IDF actually RANKS them
# high) and ck's top-normalized scores can't either. The signal that
# works: English word frequency. A hit passes if any matched term is
# distinctive (NOT in common-words.txt — Zipf < 3.5 English) or if >= 3
# terms matched (multi-term overlap is evidence even when each word is
# common: "media write permission denied" is four common words and still
# names one filesystem-permissions memory and nothing else).
# Additionally, `type: feedback` memories keep the stricter original bars
# (>=2 terms AND >=0.12 ratio) — behavior memories coincide more.
# Calibrated by replay of the author's own transcripts (7926 tagged
# injections): floors ~35%, top matched terms among floored = "s, fix, see,
# use, yes, sure" (Zipf 4.8-6.4); known-good technical hits unaffected.
MIN_MATCHED_TERMS = 3  # all-common-word hits need this many matches...
# ...AND that many matches must be a real share of the prompt. Added
# 2026-08-12 after `yet/use/project` (three common words, in a long prompt)
# injected on its own. Term FREQUENCY cannot separate that trio from the
# calibrated good hit "media write permission denied":
# media 5.30 / write 5.03 are as common in English as yet 5.54 / project 5.22,
# so no Zipf cut and no per-term weighting splits them (measured, not
# assumed). What does separate them is share-of-prompt: three incidental
# common matches out of forty terms is coincidence, three out of eight is the
# prompt's subject. Function words that are never content-bearing (`use`,
# `yet`, `like`, `via`) went into _STOPWORDS in the same change, which is
# what kills that specific trio before it ever reaches ck.
ALL_COMMON_MIN_RATIO = 0.20
COMMON_WORDS_FILE = os.path.join(os.path.dirname(__file__), "common-words.txt")
FEEDBACK_MIN_TERMS = 2  # feedback additionally: >= this many terms
FEEDBACK_MIN_RATIO = 0.12  # ...AND matched/total >= this

# A pointer line is `path — description`, and a description long enough to
# wrap costs more context than the pointer buys. Named because
# the integrity checker caps authored descriptions below DESC_KEEP_CHARS and
# quotes the number in its error: as bare literals here, raising the cut
# would silently make that cap and that message wrong.
DESC_KEEP_CHARS = 157
DESC_MAX_CHARS = 160

# Ledgers, sub-indexes, and dead memories must not surface as pointers.
EXCLUDE_BASENAMES = {"MEMORY.md", "SEARCH.md", "INDEX.md"}
# `hot` is excluded, not because hot memories are irrelevant, but because
# they are already in context (MEMORY.md auto-loads them). Written as an
# exclusion rather than an allow-list of search roots so a new domain
# sub-directory under `search/` is discoverable without touching this file.
EXCLUDE_DIRS = {"archive", "hot"}
# Markdown heading, i.e. a lexical chunk boundary (see _md_sections).
_HEADING = re.compile(r"^#{1,6}\s")

# Function words are stripped from the QUERY before either stage runs. BM25
# happily matches on "on the of", and both stages' scores are relative to
# the query's own best hit, so a score floor cannot filter stopword-only
# junk — a query with no real content words must instead produce no query at
# all — the same fix other prompt-context tools arrive at. Conservative
# list: only words
# never content-bearing in an engineering prompt.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being am do does did have has had will
    would could should may might can must to of in on at by for with from as
    into about and or but if then than so not no nor i you we he she it they
    me my your our his her their this that these those there here what which
    when where why how who whom some any all each every both few more most
    other such only own same just also very too still please help want need
    trying try get got make makes doing keep keeps use uses using yet like
    via
    """.split()  # noqa: SIM905 — compact beats a 110-line literal
)


def _search_root(store: str) -> str:
    """`<store>/search` once the store is laid out by tier, else the store.

    Hot memories are already in the agent's context, so pointing at them
    again is pure cost; the index never holds them either way, because
    _fts_scan prunes EXCLUDE_DIRS as it walks. Rooting here rather than at
    the store keeps that a matter of what gets indexed rather than what gets
    discarded after ranking. Domain sub-directories (search/<domain>/) are
    inside this root, so they stay searchable without enumerating them here.
    The fallback keeps the hook working on a store that has not been
    migrated yet.
    """
    tiered = os.path.join(store, "search")
    return tiered if os.path.isdir(tiered) else store


def _live_dirs(cfg) -> list[str]:
    """The store directories `cfg` offers this session, in config order.

    Split out from _search_dirs so that a caller holding a config parsed under
    different rules — `--debug-config` honours per-root env overrides, the
    search path does not — asks the same question of it. The two surfaces
    disagreeing about which stores exist is exactly the defect _config_state
    exists to prevent, and they can only agree if the predicate is one
    function.

    Order feeds _interleave's tie-breaking, so the config's list order is what
    decides which store wins a tie — most-specific-first is the intended
    shape. Each store resolves through its LIVE root, so a session standing in
    a worktree still reads the copy that is actually live.
    """
    dirs = [cfg.store_dir(s, "live") for s in cfg.searched_stores()]
    return [_search_root(d) for d in dirs if os.path.isdir(d)]


def _search_dirs() -> list[str]:
    """The stores the HOOK may search: _live_dirs over the hook's own config.

    Empty without a config, which is the inert default: no stores, no
    pointers.
    """
    cfg = _config()
    return _live_dirs(cfg) if cfg is not None else []


def _config_state(honor_env_overrides: bool = False) -> tuple:
    """Whether this installation has anything to search — decided once.

    Returns `(config, error, inert)`, where at most one of the last two is
    set. `error` is a config that is present and cannot be honoured. `inert`
    is the reason there is nothing to search, phrased for a person. Both None
    means the config resolved and at least one store is on disk and in scope.

    One derivation because there are two surfaces that answer this question and
    they disagreed. `--search` called an installation inert while
    `--debug-config` printed `searched` beside every store and exited 0 — and
    `--debug-config` is where the dispatcher's own refusal message sends an
    agent first, so the surface that says "this machine is fine" was the one
    reached by anybody following the instructions. The predicate is the
    same one _search_dirs uses (`searched_stores` then `isdir`), so a store
    this returns as searchable is a store retrieval will actually open.

    `honor_env_overrides` is a parameter rather than a constant because it is
    the one thing the two surfaces legitimately differ on: `--debug-config`
    honours the per-root overrides so a developer can point a session at a
    fixture tree, and the every-prompt path never may. Reading the state
    through one function does not mean reading it through one config.

    load_config directly rather than through _config(): this is the CLI's
    question, and _config folds the error into None and parks it in a global
    so the fail-open hook can degrade quietly. Out here the error is half the
    answer.
    """
    try:
        cfg = load_config(_CONFIG_PATH, honor_env_overrides=honor_env_overrides)
    except ConfigError as exc:
        return None, str(exc), None
    if cfg is None:
        return None, None, (
            f"no config (no --config, ${CONFIG_ENV} unset), so no stores to search"
        )
    if not _live_dirs(cfg):
        return cfg, None, (
            f"{cfg.path} configures no store this session can search "
            "(missing on disk, or gated to another tree)"
        )
    return cfg, None, None


def _search_cli() -> str:
    """The command string the truncation notice tells the agent to run."""
    cfg = _config()
    return cfg.search_cli if cfg is not None else DEFAULT_SEARCH_CLI


def _excluded(path: str) -> bool:
    if os.path.basename(path) in EXCLUDE_BASENAMES:
        return True
    parts = set(path.split(os.sep))
    return bool(parts & EXCLUDE_DIRS)


def _md_sections(text: str) -> list[str]:
    """Heading-delimited chunks of a markdown file; the whole file when it
    has no headings.

    Chunking is what keeps a long topic file from diluting its own score:
    BM25 divides by document length, so a query that matches one section of
    a 400-line memory should compete on that section's length, not the
    file's. The span ahead of the first heading is a chunk in its own right
    — it carries the frontmatter `description:`, the line memories are
    written to be found by.
    """
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if _HEADING.match(ln)]
    if not starts:
        return [text]
    sections = ["\n".join(lines[: starts[0]])] if starts[0] else []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(lines)
        sections.append("\n".join(lines[s:e]))
    return sections


def _section_label(chunk: str) -> str:
    """The heading a chunk starts with, as a short display label; '' if it
    does not start with one.

    _md_sections cuts at headings, so every chunk but one begins with the
    heading that owns it — the exception is a file's preamble, which is
    frontmatter and a description, not a place in the document worth naming.
    An empty label is therefore a real answer ("this matched the file's own
    summary"), not a failure to find one.
    """
    first = chunk.split("\n", 1)[0]
    if not _HEADING.match(first):
        return ""
    label = first.lstrip("#").strip()
    return label[:57] + "..." if len(label) > 60 else label


def _fts_db(root: str) -> str:
    """Index path for one corpus root, named by a digest of that root.

    One DB per root, not one shared DB, because _fts_sync treats its walk as
    authoritative: an invocation from outside a cwd-gated store's repo
    searches the ungated stores alone, and against a shared index that walk
    would sweep the gated store. Scoping the DB to a root makes "not walked" and
    "no longer exists" the same statement.
    """
    digest = hashlib.sha256(root.encode()).hexdigest()[:12]
    return os.path.join(_state_dir(), f"fts5-{digest}.db")


def _fts_note_root(db: str, root: str) -> str:
    """Record which corpus a DB holds, as a sidecar next to it.

    sha256 is one-way, so nothing in the cache dir otherwise says what
    `fts5-9d0e2c1b4a77.db` is an index OF — and "why was this memory not
    recalled" starts by looking at the index that should have held it. That
    made the first debugging step a digest recomputation over candidate roots,
    which is a thing to know rather than a thing to see.

    Advisory only: the engine never reads it, a stale one costs nothing, and
    every failure is suppressed. Written where the root-to-digest mapping is
    defined rather than inside _fts_connect, which has no business knowing
    what the file it opens is for.
    """
    sidecar = db.removesuffix(".db") + ".root"
    with contextlib.suppress(OSError):
        if not os.path.exists(sidecar):
            with open(sidecar, "w", encoding="utf-8") as f:
                f.write(root + "\n")
    return sidecar


# What a `.build` record's `outcome` may say, and the order they outrank each
# other in when a run is more than one of them: BUSY beats REBUILT beats
# PARTIAL beats OK. Named for the same reason the EXIT_* codes are — the reader
# is doctor, in another module and eventually another repo, and a bare literal
# in two places is a vocabulary that drifts.
#
# BUSY   the sync never ran (another session held the write lock), so `files`
#        is unknown and the query answered from the index as it stood.
# REBUILT the index was damaged, unlinked and built from the corpus again. Run
#        after run, this is the self-healing loop that otherwise reads exactly
#        like a healthy cache.
# PARTIAL the walk could not read part of the corpus, so `files` undercounts
#        and a low number is not evidence the corpus is small.
# OK     a complete sync over a fully readable corpus.
BUILD_OK = "ok"
BUILD_BUSY = "busy"
BUILD_REBUILT = "rebuilt"
BUILD_PARTIAL = "partial"
# Bumped when the record's shape changes. Nothing reads these yet, which is
# precisely when a version key is free to add and impossible to retrofit.
BUILD_SCHEMA = 1


def _fts_note_build(db: str, outcome: str, files: int | None) -> str:
    """Record how this corpus was last indexed, as a second sidecar.

    "Never indexed" and "indexed, and the corpus turned out to be empty" are
    the same absence from outside this file — no pointers, and an index file
    that may or may not be there — and the only way to tell them apart was to
    open the index, which syncs, which rebuilds whatever the walk finds stale.
    A diagnostic that repairs the state it is diagnosing cannot report on it.

    `files` is what the walk read INTO the index: not a row count, not the
    corpus's size on disk, and not a count of what the index now holds — a
    file the walk could not read keeps its existing rows and is excluded here,
    because this number's job is to say what this run established. `files: 0`
    with `outcome: ok` therefore means the corpus really was empty, and that
    claim is only safe because every way of failing to read the corpus lands
    on an outcome other than OK. `None` means the run never got far enough to
    count.

    Same posture as the `.root` sidecar: advisory, never read by the engine, a
    stale one costs nothing, every failure suppressed. Rewritten every run,
    because the question is what happened LAST. Written beside and renamed
    over, unlike `.root`, because two sessions index the same corpus at once
    and a reader that met half a line could not tell a torn write from an
    index holding nothing — which is the one distinction this file exists to
    make.
    """
    sidecar = db.removesuffix(".db") + ".build"
    tmp = f"{sidecar}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "v": BUILD_SCHEMA,
                    "ts": int(time.time()),
                    "outcome": outcome,
                    "files": files,
                },
                f,
                separators=(",", ":"),
            )
        os.replace(tmp, sidecar)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    return sidecar


def _fts_connect(db: str) -> sqlite3.Connection:
    """Open (creating if absent) one corpus root's index.

    The index is a cache that can be rebuilt from the corpus at any time, so
    durability buys nothing and costs latency on the interactive path: WAL
    plus synchronous=NORMAL trade fsyncs away. busy_timeout keeps two
    concurrent sessions' hooks from turning a brief write lock into a
    failure, and losing that race anyway is just another rebuild.
    """
    con = sqlite3.connect(db)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=250")
        # IF NOT EXISTS means a column change here does NOT migrate an index
        # built by an older version of this file — it leaves the old table in
        # place, and the first identity SELECT then fails on the missing
        # column. That failure is the migration: it is a plain sqlite3 error,
        # so _fts_dir routes it to the damage path, which unlinks and rebuilds
        # from the corpus. One rebuild per corpus on upgrade, ~55 ms, silent.
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING"
            " fts5(path UNINDEXED, mtime_ns UNINDEXED, ctime_ns UNINDEXED,"
            " size UNINDEXED, text)"
        )
    except BaseException:
        # A statement can fail after connect() has already handed back an open
        # handle. Closing it here is what keeps the caller's recovery honest:
        # an abandoned WAL handle is finalized whenever the interpreter gets
        # round to it, and it removes -wal/-shm BY PATH — which by then may
        # belong to the replacement index.
        con.close()
        raise
    return con


def _fts_identity(con: sqlite3.Connection) -> dict[str, tuple[int, int, int]]:
    """What the index believes about each file it holds."""
    return {
        path: (mtime_ns, ctime_ns, size)
        for path, mtime_ns, ctime_ns, size in con.execute(
            "SELECT DISTINCT path, mtime_ns, ctime_ns, size FROM chunks"
        )
    }


def _fts_answerable(con: sqlite3.Connection) -> bool:
    """Whether this index holds anything at all to answer with."""
    return con.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None


# Lexical-stage health, summed across corpus dirs and surfaced in the soak
# log only when nonzero. The first three describe an index that is answering
# from less than the whole corpus, which is otherwise indistinguishable in the
# log from a corpus with nothing to say: files held out of the identity
# comparison (unstattable, unreadable), dirs the walk could not enter, and
# syncs skipped because another session held the write lock. The fourth is the
# damage path — an index that self-heals leaves no other trace, so a cache
# being destroyed and rebuilt on every prompt reads exactly like a healthy one
# until this count is nonzero on line after line.
#
# Module-level rather than threaded through _fts_dir/_fts_sync/_fts_scan,
# which would put an observability argument in the signature of every
# index-maintenance function and in the tests that call them directly. The
# process is one-shot, but recall() zeroes them first so a harness that calls
# it in a loop (the eval, this suite) still reports per-call.
_LEX_COUNTS: dict[str, int] = {
    "lex_spared": 0,
    "lex_unwalked": 0,
    "lex_busy_skip": 0,
    "lex_rebuilds": 0,
}

# Where each hit came from INSIDE its file: path -> the heading of the
# best-ranked chunk, for the pointer line's `[section: ...]` tag. Kept beside
# the hits rather than returned with them because every path in this hook
# travels as a bare string — through _interleave's rank merge, the session
# dedup set — and threading a tuple through all of that to carry a display
# label would rewrite each of those to unpack something they have no use for.
_LEX_SECTIONS: dict[str, str] = {}

# Which query terms the INDEX matched in that same best-ranked chunk: path ->
# terms, in query order. Travels beside the hits for the reason above, and it
# is the ONLY source for the relevance floor and the `[matches n/m]` tag — a
# path missing from here has no evidence and the floor drops it.
_LEX_MATCHED: dict[str, list[str]] = {}

# What the ranker actually scored each hit: path -> rank/best_rank, the same
# top-normalized number FLOOR_LEX is compared against, so 1.0 is that dir's
# best chunk and FLOOR_LEX is the weakest thing kept. Logged, never acted on.
#
# The soak log records WHICH files were injected but not how close the call
# was, and those are different questions: a pointer at 0.98 and a pointer at
# 0.31 look identical in the log, so "is the cap costing real answers or tail
# noise" could only be answered by re-running the prompt against a corpus that
# had since moved. Cross-dir comparison is not one of the things this supports
# — each dir normalizes against its own best hit, so the merge is by rank
# (_interleave) and these numbers are only ever meaningful within one dir.
_LEX_SCORES: dict[str, float] = {}


def _fts_scan(root: str) -> tuple[dict[str, tuple[int, int, int]], set[str], set[str]]:
    """Walk one corpus root: (stat-able identities, spared paths, unwalked).

    A file this walk cannot establish anything about is SPARED rather than
    indexed: it is held out of the identity comparison, so a mismatch it can
    never resolve does not make every prompt take the write lock, and held
    out of the sweep, so whatever rows it already has keep answering.

    Only ENOENT from the stat is evidence the file is gone. Any other stat
    failure says the walk could not find out, and unproven absence must not
    reach the sweep as if it were a deletion — one EACCES would otherwise
    drop a memory that is sitting right there.

    Readability is NOT decided here — it is settled by the read in _fts_sync,
    which has to handle a failed read anyway and only reads the files whose
    identity moved. A probe pass over every file on every prompt (os.access,
    or an open/close) only predicts what that read is about to report: it
    cannot change an outcome, and costs ~1.9 ms of a ~10 ms warm path.

    `unwalked` is the directories the walk could not read. The walk is only
    authoritative where it reached — an unreadable subtree is invisible, not
    empty, and treating it as empty would sweep every memory in it out of
    the index.
    """
    disk: dict[str, tuple[int, int, int]] = {}
    spared: set[str] = set()
    unwalked: set[str] = set()

    def unreadable(exc: OSError) -> None:
        unwalked.add(exc.filename or root)

    for dirpath, dirnames, filenames in os.walk(root, onerror=unreadable):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not name.endswith(".md") or _excluded(path):
                continue
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue  # vanished mid-walk; it simply stays unindexed
            except OSError:
                spared.add(path)
                continue
            disk[path] = (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    return disk, spared, unwalked


def _fts_sync(con: sqlite3.Connection, root: str) -> tuple[int, int, int]:
    """Bring the index in line with the corpus. The walk is authoritative.

    Returns `(files, spared, unwalked)` for `_fts_note_build`: how many files
    this walk actually read into the index, and how much of the corpus it could
    not account for. All three, not just the first, because `files` alone
    cannot be trusted without them — a corpus nobody can read walks to zero
    files and no error, which is byte-identical to a corpus that is genuinely
    empty. The counts are what let the record say `partial` instead of
    claiming an empty corpus it never saw.

    A file's identity is (mtime_ns, ctime_ns, size), carried on every one of
    its chunk rows: FTS5 has no unique constraints and no update-by-key, so
    idempotence lives in comparing that triple, not in the INSERT. Files whose
    triple moved are deleted and re-inserted whole; stored paths a COMPLETE
    walk did not see are swept, which is the only GC this index has (a deleted
    memory would otherwise stay answerable forever).

    ctime is in the triple because mtime and size together are forgeable: a
    writer that replaces a file with same-sized content and os.utime()s the old
    mtime back leaves an identity this index cannot tell from the version it
    already holds, and the stale text answers forever. ctime moves on that
    write and utime cannot set it. The cost is that ctime also moves on chmod
    and rename, so a file whose bytes did not change is occasionally re-read
    and re-indexed — the harmless direction of the trade.

    The steady state reads the identity once and touches nothing else: no
    lock, no file contents. Changed files are read BEFORE the lock, both to
    keep IO out of the transaction and because a read that fails has to be
    classified where the failure can convert into "spared" rather than into
    another attempt next prompt. The identity snapshot the work is planned
    from is then re-read INSIDE the transaction. Sessions run concurrently, so
    between one session's plan and its lock another may have already done the
    whole re-index; planning from the pre-lock snapshot makes every waiter redo
    it, turning one changed memory into an N-session re-index queue that grows
    past busy_timeout. One transaction also means an interrupted sync leaves
    the previous index rather than a half-updated one.

    Unreadable files are skipped, never indexed as empty — but an index that
    holds NOTHING and skipped something raises, because an empty index is
    indistinguishable from a corpus with nothing to say once the rows are
    gone, and "no hits" is an answer the caller trusts.
    """
    disk, spared, unwalked = _fts_scan(root)
    _LEX_COUNTS["lex_unwalked"] += len(unwalked)
    snapshot = _fts_identity(con)
    if unwalked:
        # An incomplete walk cannot tell "deleted" from "in the part I could
        # not read", so nothing it failed to see is evidence of anything:
        # sparing all of it is both why those rows survive and why a corpus
        # with an unreadable subtree still converges instead of taking the
        # write lock on every prompt for as long as the mode persists. This
        # one line is the whole of that guarantee — it empties `sweep` below,
        # so an incomplete walk cannot delete anything.
        spared |= snapshot.keys() - disk.keys()
    # Read the stale files' contents before deciding to take the lock. This is
    # where readability is established at all, so a read that fails has to
    # convert into "spared" HERE: a path left in a comparison it can never
    # satisfy re-enters BEGIN IMMEDIATE on every prompt, forever, against
    # every session at once.
    staged: dict[str, list[str]] = {}
    for path, ident in list(disk.items()):
        if snapshot.get(path) == ident:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                staged[path] = _md_sections(f.read())
        except OSError:
            spared.add(path)
            del disk[path]  # unreadable is not indexable; its old rows stand
    # What to sweep is decided HERE, from the walk's own snapshot, and carries
    # each candidate's identity with it. The re-read under the lock is a better
    # picture of the index but a worse one of the corpus: it names rows written
    # after this walk, for files that did exist, and "my walk never saw it" is
    # not evidence about a file that came into being after the walk.
    sweep = {p: i for p, i in snapshot.items() if p not in disk and p not in spared}
    if {p: i for p, i in snapshot.items() if p not in spared} != disk:
        con.execute("BEGIN IMMEDIATE")
        try:
            stored = _fts_identity(con)
            for path, ident in disk.items():
                if stored.get(path) == ident:
                    continue
                if path in staged:
                    sections = staged[path]
                else:
                    # Nothing was staged, so this path matched the pre-lock
                    # snapshot and stopped matching under the lock: a racing
                    # writer left a different version behind. Read it BEFORE
                    # the DELETE — deleting first would empty the index of a
                    # file that is about to be readable again — and let a
                    # failure here be the race it is; if it turns out to be a
                    # mode rather than a moment, the next scan classifies it.
                    try:
                        with open(path, encoding="utf-8", errors="replace") as f:
                            sections = _md_sections(f.read())
                    except OSError:
                        spared.add(path)
                        continue
                con.execute("DELETE FROM chunks WHERE path = ?", (path,))
                con.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    [(path, *ident, section) for section in sections],
                )
            for path, ident in sweep.items():
                if stored.get(path) != ident:
                    # Its row changed, or is gone, since the walk. Either way
                    # another session has been here with a fresher view of this
                    # file than the walk had, and deleting on the strength of
                    # the older view would undo work that supersedes it.
                    continue
                con.execute("DELETE FROM chunks WHERE path = ?", (path,))
            con.commit()
        except BaseException:
            # This connection is REUSED for the query — a caller that swallows
            # a lost write-lock race still reads from it — so it must never be
            # left holding a stale write transaction. close() would roll back
            # too, but not before that read.
            con.rollback()
            raise
    # Counted here, where `spared` is final: the staging loop and the
    # in-transaction backstop both add to it after _fts_scan returned.
    _LEX_COUNTS["lex_spared"] += len(spared)
    if (spared or unwalked) and not _fts_answerable(con):
        raise OSError(f"index empty and part of {root} unreadable")
    # `disk` still holds any path the in-transaction backstop failed to reopen
    # — it cannot be deleted there, since that loop iterates the live dict —
    # so the subtraction happens here instead. Every other spared path was
    # either never in `disk` or already removed by the staging loop, so one
    # difference covers all of them.
    return len(disk.keys() - spared), len(spared), len(unwalked)


def _fts_search(con: sqlite3.Connection, query: str) -> list[str]:
    """Query one index; return file paths best-first.

    Terms are OR-ed quoted phrases. build_query has already reduced the
    prompt to word-char terms, and quoting each one is what keeps a term
    that collides with FTS5's query syntax (AND, OR, NOT, NEAR) from being
    read as an operator.

    The fold to best-chunk-per-file is GROUP BY, so CANDIDATE_LIMIT bounds
    FILES and a long memory split across many chunks cannot crowd the window
    with its own text. Doing it in SQL rather than over-fetching chunks and
    folding here is both exact and the fastest of the three shapes measured
    (1.18 ms vs 1.60 over-fetching 8x, 2.01 folding every matching chunk).
    bm25 `rank` is negative and unbounded (lower is better), so each file is
    normalized against the best file in the window before FLOOR_LEX applies —
    the floor is a claim about the weak tail relative to this query's best
    hit, never about an absolute score.

    `text` and `rowid` are selected bare alongside min(rank), which SQLite
    defines as the values of the row the minimum came from (the documented
    bare-column special case for a single min/max aggregate — verified to hold
    on an fts5 table, sqlite 3.53.4). Under any other aggregate those columns
    would describe an arbitrary chunk of the file, and both the section label
    and the term evidence built from them would be confidently wrong rather
    than absent.
    """
    terms = list(dict.fromkeys(query.split()))
    match = " OR ".join(f'"{t}"' for t in terms)
    if not match:
        return []
    rows = con.execute(
        "SELECT path, min(rank) AS r, text, rowid FROM chunks WHERE chunks MATCH ?"
        " GROUP BY path ORDER BY r LIMIT ?",
        (match, CANDIDATE_LIMIT),
    ).fetchall()
    if not rows:
        return []
    best_rank = rows[0][1]
    if not best_rank:
        return []  # divisor guard: sqlite clamps idf, so a matched row is < 0
    hits = []
    ranked: dict[str, int] = {}
    for path, rank, text, rowid in rows:
        # The index can legitimately be a sweep behind — a sync that lost the
        # write-lock race answers from rows that still name a deleted memory —
        # and the caller turns a hit into a pointer the user is told to read.
        score = rank / best_rank
        if score < FLOOR_LEX or _excluded(path) or not os.path.exists(path):
            continue
        _LEX_SCORES[path] = score
        label = _section_label(text)
        if label:
            _LEX_SECTIONS[path] = label
        ranked[path] = rowid
        hits.append(path)
    if ranked:
        _record_matched(con, terms, ranked)
    return hits


def _record_matched(
    con: sqlite3.Connection, terms: list[str], ranked: dict[str, int]
) -> None:
    """Ask the index which of `terms` it matched in each ranked FILE.

    The evidence the floor judges and the pointer line advertises has to be
    the SAME evidence the ranker acted on, and re-deriving it outside the
    index got that wrong. A `\\b` regex is not the unicode61 tokenizer:
    unicode61 splits on `_`, `\\b` does not, so `LATEST_REPLY` matched the
    query term `reply` in FTS5 and then matched nothing here, and
    `memory_required_gb` likewise for `gb`. Those hits reached the floor
    claiming zero matched terms, which was then read as "matched by embedding"
    and waved through — a precision hole and a false provenance claim, both
    out of one regex. One MATCH per term cannot drift from the tokenizer
    again, because it IS the tokenizer. Measured at 0.52-0.60 ms per store for
    a 7-term query over a full 10-candidate window, against 1.9-5.1 ms for the
    file reads and regexes it replaces.

    Counting is scoped to the FILE, not to the chunk that ranked, and that is
    a deliberate choice against the tidier one. Chunk-scoping is more truthful
    per pointer — 23% of advertised matched terms come from regions of the file
    that did not rank — but it shrinks every count by a term or two, and
    MIN_MATCHED_TERMS was calibrated against file-wide counts. Measured over
    200 prompts replayed from the author's corpus — as is every count in this
    paragraph — scoping to the chunk took the hook from 18 silent
    prompts to 32, going newly silent on 14 that the shipped code served, and
    both hand-classified genuine losses were among them: two follow-up
    questions whose only distinctive terms sat in a part of the target file
    that had not itself ranked, which file-scoping answers exactly as the
    shipped code does. File-scoping costs one prompt,
    and that one was injecting on zero term overlap. Both scopes kill all three
    zero-overlap pointers, so the precision win belongs to the tokenizer-exact
    half of this function and not to the scope.

    Re-deriving the floor for chunk-scoped counts is a live calibration
    question, not a settled one — dropping MIN_MATCHED_TERMS 3 -> 2 alongside
    it recovers the volume but changes WHICH memories are injected, so it needs
    its own evidence rather than a constant nudged until the totals look right.
    `[section: ...]` still names the chunk that ranked, because that is a
    display claim about provenance and remains exactly true.
    """
    paths = list(ranked)
    # The only thing interpolated is a run of `?`, one per path — SQL cannot
    # parameterize the length of an IN list. Both the term and every path are
    # still bound, so no value reaches the parser as text.
    ph = ",".join("?" * len(paths))
    for p in paths:
        _LEX_MATCHED[p] = []
    for t in terms:
        for (p,) in con.execute(
            f"SELECT DISTINCT path FROM chunks WHERE chunks MATCH ? AND path IN ({ph})",  # noqa: S608
            (f'"{t}"', *paths),
        ):
            _LEX_MATCHED[p].append(t)


def _fts_busy(exc: BaseException) -> bool:
    """True for "someone else holds the write lock", false for damage.

    The two have to be told apart because the recovery for damage — unlink
    the DB and rebuild — is destructive when applied to contention: it
    deletes a healthy file out from under a live writer, whose committed
    work then goes to an orphaned inode, and unlinking the DB without its
    -wal is SQLite's own documented way to corrupt a database.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)  # 3.11+
    if code is not None:
        # Masked to the primary code: SQLite's extended results pack a reason
        # into the high bits (a SETLK build reports contention as
        # SQLITE_BUSY_TIMEOUT, 773), and an unmasked comparison would read
        # those as damage and delete the index.
        #
        # sqlite3.SQLITE_* landed in 3.11, the same release as
        # `sqlite_errorcode` — so this branch is unreachable on the 3.9 floor
        # and the names cannot raise there. The floor pass (`pyright -p
        # pyrightconfig-hook39.json`) resolves them against 3.9 stubs and
        # cannot see that guard, so it is told here rather than by widening
        # the config, which would blind the whole file.
        return code & 0xFF in (
            sqlite3.SQLITE_BUSY,  # pyright: ignore[reportAttributeAccessIssue]
            sqlite3.SQLITE_LOCKED,  # pyright: ignore[reportAttributeAccessIssue]
        )
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _fts_dir(query: str, d: str) -> list[str]:
    """The lexical stage over ONE dir; return file paths best-first.

    Sync then query, every invocation: a memory written a minute ago is
    exactly the one the next prompt needs, and nothing else in this hook's
    lifecycle gets to run first, so freshness has to be established at query
    time.

    Contention is handled at two layers, differently. A sync that loses the
    write-lock race is skipped and the query runs anyway: another session is
    mid-update, its index is intact and at worst one edit stale, which beats
    spending the stage on a lock — but only if that index holds rows at all,
    since an empty one would answer "no hits" and be believed. Contention
    anywhere else (the connect, the recovery path) propagates into the
    caller's per-dir suppression and is counted as errs_lex.

    Only SQLite's own errors route to recovery, and recovery is one unlink
    and rebuild (~55 ms for the larger store, against ~2 ms for a warm sync):
    rebuild-from-corpus is all a disposable cache needs. An unreadable corpus
    or a bug in this file is not something deleting the cache diagnoses or
    repairs, so those propagate untouched.
    """
    if not os.path.isdir(d):
        return []
    db = _fts_db(d)
    _fts_note_root(db, d)

    def attempt(base: str) -> list[str]:
        con = _fts_connect(db)
        try:
            outcome, files = base, None
            try:
                files, spared, unwalked = _fts_sync(con, d)
                # A corpus nobody can read walks to zero files without raising,
                # and `ok` over zero files is the claim that the corpus is
                # empty — the exact confusion this sidecar exists to break. The
                # walk's own account of what it could not reach is the only
                # thing that separates them.
                if (spared or unwalked) and outcome == BUILD_OK:
                    outcome = BUILD_PARTIAL
            except sqlite3.OperationalError as exc:
                if not _fts_busy(exc) or not _fts_answerable(con):
                    raise
                _LEX_COUNTS["lex_busy_skip"] += 1
                outcome = BUILD_BUSY
            # Noted between the sync and the query, so a search that raises
            # still leaves the record of how the index it was about to read
            # got there.
            _fts_note_build(db, outcome, files)
            return _fts_search(con, query)
        finally:
            con.close()

    try:
        return attempt(BUILD_OK)
    except sqlite3.Error as exc:
        if _fts_busy(exc):
            raise
        # Concurrent recoveries are deliberately not serialized. Two processes
        # can interleave here badly enough that one deletes the -wal of a
        # rebuild the other is still writing, and the loser then fails open —
        # but its next invocation finds either a healthy index or an unreadable
        # one it rebuilds again, so the cost is one wasted rebuild, never a
        # wrong answer. A lock to prevent that would add staleness and cleanup
        # failure modes of its own to a cache whose repair is "delete it".
        _LEX_COUNTS["lex_rebuilds"] += 1
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                os.unlink(db + suffix)
        # Noted BEFORE the retry, and again after it if the retry gets that
        # far. The index this record describes has just been deleted, so the
        # window between the unlink and a rebuild that does not complete is one
        # where the previous record — very possibly `ok`, with a file count —
        # outlives every row it was describing. A reader arriving then is told
        # the corpus is indexed and healthy when there is no index at all.
        _fts_note_build(db, BUILD_REBUILT, None)
        return attempt(BUILD_REBUILT)


def _interleave(ranked_lists: list[list[str]]) -> list[str]:
    """Round-robin merge of per-dir rankings (rank 1s first, then rank 2s…).

    Cross-dir scores are incomparable — the lexical stage normalizes each
    dir's bm25 ranks against that dir's own best chunk, so a dir's
    best-of-a-bad-lot also scores 1.0 — so merge by rank instead. The caller
    orders dirs most-specific-first (project before personal inside the
    repo) so ties break toward the more specific corpus.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for tier in range(max((len(lst) for lst in ranked_lists), default=0)):
        for lst in ranked_lists:
            if tier < len(lst) and lst[tier] not in seen:
                seen.add(lst[tier])
                merged.append(lst[tier])
    return merged


def _description(path: str) -> str:
    """Frontmatter `description:` line, else first heading, else ''."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return ""
    m = re.search(r"^description:\s*(.+)$", head, re.MULTILINE)
    if not m:
        m = re.search(r"^#\s+(.+)$", head, re.MULTILINE)
    if not m:
        return ""
    desc = m.group(1).strip().strip("\"'")
    return desc[:DESC_KEEP_CHARS] + "..." if len(desc) > DESC_MAX_CHARS else desc


def _relevance(terms: list[str], path: str) -> tuple[list[str], int, str]:
    """Read a memory file once and return (matched query terms in query
    order, total terms, frontmatter `type:`).

    The matched terms are the agent-facing relevance evidence: a stage's
    scores are normalized against its own best hit, so a score can't tell a
    strong hit from best-of-a-bad-lot — but 'matched 6/8 terms incl. the
    load-bearing ones' vs 'matched 1/8' is honest signal. The `type:` drives
    the feedback relevance floor (see FEEDBACK_MIN_*). One read serves both.

    The terms come from the index (_record_matched), never from a scan of this
    text: they must agree with what the ranker matched, and be scoped to the
    chunk that actually ranked. A `\\b` regex over the file does neither — it
    disagrees with FTS5's unicode61 tokenizer on every identifier (`reply`
    inside LATEST_REPLY, `gb` inside memory_required_gb), which is what put
    pointers on screen claiming term overlap the index had never found.

    A path with no index evidence therefore yields none, and the floor drops
    it. That case is unreachable while _record_matched is the only writer, and
    it is meant to stay unreachable: it fails a future divergence closed.

    The side channel is intersected with `terms` rather than trusted whole, so
    a caller judging a hit against a different query than the one that
    retrieved it gets its own terms answered, never the search's.

    Only the frontmatter is read, since that is all `type:` needs.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return [], len(terms), "?"
    mtype = "?"
    m = re.search(r"^\s*type:\s*(\w+)", head, re.MULTILINE)
    if m:
        mtype = m.group(1)
    hit = set(_LEX_MATCHED.get(path, ()))
    return [t for t in terms if t in hit], len(terms), mtype


def _common_words() -> frozenset[str]:
    """Lazy-load the English-common wordlist (generated by
    tools/generate-common-words.py). Missing/unreadable file fails OPEN:
    empty set -> every term counts as distinctive -> the Zipf branch never
    floors -> behavior degrades to the pre-Zipf floor, never breaks."""
    global _COMMON
    if _COMMON is None:
        try:
            with open(COMMON_WORDS_FILE, encoding="utf-8") as f:
                _COMMON = frozenset(
                    w for line in f if (w := line.strip()) and not w.startswith("#")
                )
        except OSError:
            _COMMON = frozenset()
    return _COMMON


_COMMON: frozenset[str] | None = None


def _passes_floor(matched: list[str], n_total: int, mtype: str) -> bool:
    """Relevance floor over the matched query terms. See the MIN_MATCHED /
    COMMON_WORDS comment block for the rationale + calibration.

    - 0 matched terms -> reject.
    - Any distinctive matched term (not common English) -> pass.
    - All-common matches -> pass only on >= MIN_MATCHED_TERMS matches AND
      >= ALL_COMMON_MIN_RATIO share of the query's terms.
    - type: feedback additionally keeps the stricter original bars.

    Term evidence is now required, full stop. There was an exemption for zero
    matches, written for semantic hits that by construction matched no term;
    with that stage deleted the exemption had no legitimate claimant left, and
    it had never checked for one — for a year it fired on the COUNT, so 47 of
    the 50 pointers it ever waved through came from prompts where the semantic
    stage never ran. A hit whose own index reports no matched term is a
    contradiction between claim and evidence, and this is the last place that
    can say so.
    """
    n_matched = len(matched)
    if n_matched == 0:
        return False
    ratio = n_matched / n_total if n_total else 0.0
    if mtype == "feedback" and (
        n_matched < FEEDBACK_MIN_TERMS or ratio < FEEDBACK_MIN_RATIO
    ):
        return False
    common = _common_words()
    if any(t.lower() not in common for t in matched):
        return True
    return n_matched >= MIN_MATCHED_TERMS and ratio >= ALL_COMMON_MIN_RATIO


def _display_path(path: str) -> str:
    """~-relative — unambiguous from any cwd. A repo-relative form would
    only resolve when the session cwd is exactly the repo root (not a
    subdirectory or worktree)."""
    home = os.path.expanduser("~")
    if path.startswith(home + os.sep):
        return "~/" + os.path.relpath(path, home)
    return path


def _state_dir() -> str:
    """Per-user 0700 cache dir, not world-writable /tmp: filenames are
    predictable, so a shared /tmp would allow symlink pre-planting. Also
    stable across launch contexts, unlike macOS's per-context TMPDIR."""
    d = os.path.expanduser("~/.cache/memory-recall")
    try:
        os.makedirs(d, mode=0o700, exist_ok=True)
    except OSError:
        d = tempfile.gettempdir()
    return d


def _session_state_path(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:80]
    return os.path.join(_state_dir(), f"{safe}.json")


def _evidence(matched: list[str], total: int) -> float:
    """How much this hit's slot was bought with: the share of the prompt's
    terms it matched.

    The same statistic the floor calibrates on (ALL_COMMON_MIN_RATIO) and the
    pointer line renders, so the budget cannot rank hits by one measure while
    the floor admits them by another. A ratio rather than a count because the
    comparison is across PROMPTS of different lengths, where a raw count means
    different things. It is a coarse scalar and known to be: 1 of 2 terms
    outranks 4 of 9 here, which is arguable. It decides only replacement
    order, never admission — the floor already did that.
    """
    return len(matched) / total if total else 0.0


def _load_session(path: str) -> tuple[set[str], dict[str, float | None]]:
    """(every path shown this session, the budget ledger).

    Two sets, deliberately not one. `shown` is the dedup set and never
    shrinks; the ledger is what POINTER_BUDGET counts and can lose entries to
    replacement. Evicting from a single combined set would let the evicted
    memory be injected a second time — the budget would buy the same pointer
    twice, and the session would see it repeated.

    A file written by the pre-ledger schema is a bare JSON list of paths:
    pointers that were spent with no record of what bought them. They load
    with evidence None, meaning "not comparable", and a budget holding any of
    those stays terminal. Scoring them 0.0 instead would read "no evidence"
    as "worthless evidence" and let the first hit of any strength evict them,
    which would make precisely the oldest sessions unbounded.

    Nothing here trusts the file's VALUES either. This hook writes only floats,
    but the file is on disk under a name a person can guess, survives across
    sessions, and has already carried one other schema; a `spent` whose values
    are strings would send _replace's sort into a TypeError, and because the
    file persists, every prompt of that session for the rest of its life. So a
    value that is not a real number loads as None — the same "not comparable"
    bucket the legacy schema uses — which degrades that session to a terminal
    budget instead of killing it.
    """
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return set(), {}
    if isinstance(state, list):
        return set(state), dict.fromkeys(state)
    shown = {p for p in (state.get("shown") or []) if isinstance(p, str)}
    spent = state.get("spent")
    if not isinstance(spent, dict):
        return shown, dict.fromkeys(shown)
    return shown, {
        p: (e if isinstance(e, (int, float)) and not isinstance(e, bool) else None)
        for p, e in spent.items()
        if isinstance(p, str)
    }


def _replace(
    spent: dict[str, float | None], offered: list[tuple[str, list[str], int]]
) -> tuple[list[tuple[str, list[str], int]], dict[str, float | None], list[str]]:
    """Past the budget: keep the POINTER_BUDGET strongest of what is already
    spent plus what is offered. Returns (injectable, new ledger, evicted).

    A terminal budget makes a long session's later prompts strictly worse
    served than its first, whatever they ask: the 31st prompt's perfect match
    loses to the 3rd prompt's incidental one purely by arriving later. This
    makes the budget a bar instead of a deadline — the cost of a pointer is
    displacing the weakest one already paid for, so a session keeps its best
    30 rather than its first 30.

    The sort is stable over spent-then-offered, so a newcomer that only TIES
    the weakest incumbent sorts behind it and is the one truncated away:
    strictly exceeds, never merely equals. Note what this does NOT bound —
    the number of pointers RENDERED into a session, which grows by one on
    every replacement. It is self-damping, since each replacement raises the
    weakest evidence in the ledger and so the bar for the next one, but the
    ledger caps spend, not lines.
    """
    ranked: list[tuple[str, float]] = [
        (p, e) for p, e in spent.items() if e is not None
    ]
    ranked += [(p, _evidence(m, t)) for p, m, t in offered]
    keep = sorted(ranked, key=lambda pe: -pe[1])[:POINTER_BUDGET]
    kept = {p for p, _ in keep}
    return (
        [o for o in offered if o[0] in kept],
        dict(keep),
        [p for p in spent if p not in kept],
    )


def _version() -> str:
    """Short content hash of this file, stamped on every soak record.

    Every question the log is asked is a comparison across time — did the
    floor get stricter, did the truncation rate move, is this rate the one
    that was measured — and none of them is answerable unless a record says
    which code wrote it. Before this, a behavior change silently split the log
    into two incomparable halves with nothing to separate them by but a
    timestamp somebody had to remember.

    Hashing the TUNING CONSTANTS instead would be cheaper and looks equivalent.
    It is not: the retired semantic stage's "fires on 1.1% of prompts" was a
    code defect — its trigger counted raw hits where it meant floor survivors
    — with every constant in this file unchanged, so a constants hash would
    have declared those records comparable to today's. Behavior lives in the
    whole file.
    """
    global _VERSION
    if _VERSION is None:
        try:
            with open(__file__, "rb") as f:
                _VERSION = hashlib.sha256(f.read()).hexdigest()[:8]
        except OSError:
            _VERSION = "?"
    return _VERSION


_VERSION: str | None = None


def _log_session(session_id: object) -> str:
    """The session id as the soak log stores it: the first 12 characters.

    Truncated because the log is joined to transcripts by filename prefix and
    the full id buys nothing there, and because a shorter field is a smaller
    thing to leak from a file whose contract (see _soak_log) admits only
    hashes, counts, basenames, and the sanitized query terms.

    A named function rather than a slice at the call site so the analyzers can
    assert against the real writer: both of them separate real records from
    harness ones by the SHAPE of this field, and a change to the width here
    would otherwise silently empty their windows.
    """
    return str(session_id)[:12]


def _soak_log(record: dict) -> None:
    """Append one JSONL line per invocation (plan step 3's metrics: misses
    are otherwise invisible — a silent hook is indistinguishable from a
    gated, failed, or no-match prompt). Best-effort; failures never affect
    injection.

    What a record may carry is a contract, and it is not "no prompt text": a
    record that reached retrieval also holds `query`, the terms build_query
    kept — stopwords dropped, non-word characters replaced, capped at 40 terms
    and 160 characters. That field is prompt-DERIVED and the only one; the raw
    prompt is never written, and everything else here is a hash, a count, or a
    basename. It stays because the offline shadow harness replays it: the
    field is that harness's entire corpus, so dropping it is the same decision
    as retiring the harness, and three docstrings claiming the log held no
    prompt text was how it nearly got dropped as dead weight.
    """
    with contextlib.suppress(Exception):
        record["ts"] = int(time.time())
        record["v"] = _version()
        with open(os.path.join(_state_dir(), "log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def build_query(stripped: str) -> str | None:
    """Prompt -> sanitized search query (None if nothing content-bearing).

    Shared by main() and the eval harness so evals exercise the real query
    construction.
    """
    words = [
        w
        for w in stripped.split()[:80]
        if w.lower().strip(".,!?;:'\"()") not in _STOPWORDS
    ]
    if len(words) < 2:
        return None

    # Sanitize per word, because both stages read the query as SYNTAX before
    # they read it as words: a leading '-' is a clap flag to ck (and an
    # exclusion operator to its parser), apostrophes/quotes/parens hard-error
    # a ck search ("what's" -> Syntax Error, verified 0.7.11), and a bare
    # quote would terminate the phrase _fts_search wraps each term in. Keep
    # only word chars, then re-check.
    terms = [t for w in words[:40] if (t := re.sub(r"[^\w]", " ", w).strip())]

    # Compound splitting: user shorthand fuses identifiers ("node1" for a host
    # written down as node-4310-1). The corpus tokenizes hyphenated names into
    # parts, but BM25 matches whole tokens — "node1" hits nothing while
    # "node 1" hits the right memories (verified). Keep the original AND
    # the alpha split parts (bare digits are noise across a corpus full of
    # numbers).
    extra = []
    for w in terms:
        parts = re.split(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", w, flags=re.I)
        if len(parts) > 1:
            extra.extend(p for p in parts if len(p) > 1 and not p.isdigit())
    return " ".join(terms + extra) or None


def prompt_gate(stripped: str) -> str | None:
    """The outcome main() declines this prompt with, or None if it reaches
    retrieval. Pure: reads the prompt and nothing else, writes nothing.

    It exists so that "would production have declined this?" has ONE answer.
    The follow-through analyzer's inverted join has to ask exactly that —
    its GATED bucket is the share of known-correct answers retrieval never got
    a chance at — and it used to ask by testing `build_query(...) is None`,
    which is only the last of three gates. Every prompt production refuses for
    its SHAPE (an envelope, a slash command, a two-word question, a 4000-plus
    character paste) was therefore scored as a retrieval failure, and the
    NO-MATCH and FLOOR/WINDOW buckets carried the difference: the analyzer
    blamed the ranker for prompts the ranker never saw.

    The prompt must be the UNtruncated one. The join's own prompt field is
    capped for display, and a long paste cut to fit passes a length gate the
    real prompt failed — the truncation manufacturing the opposite error.

    `gate:nodirs` is deliberately not here. It is a fact about the machine's
    corpora, not about the prompt, so it would answer for the operator's
    stores rather than for the --dir the caller passed, and no analyzer wants
    that. main() checks it between this function's two halves.
    """
    # An envelope is an envelope at any length, so this precedes the shape
    # gate; in main() it also precedes build_query/recall, which is what keeps
    # scaffolding vocabulary out of the index entirely.
    if _is_envelope(stripped):
        return "gate:envelope"
    # Nothing to do on short prompts, slash commands, pasted blobs.
    if (
        not stripped
        or stripped.startswith("/")
        or len(stripped.split()) < MIN_PROMPT_WORDS
        or len(stripped) > 4000
    ):
        return "gate:shape"
    if build_query(stripped) is None:
        return "gate:stopwords"
    return None


def recall(
    prompt: str,
    stats: dict | None = None,
    dirs: list[str] | None = None,
    deadline: float | None = None,
) -> list[str]:
    """Full retrieval path (query build + lexical search), no session dedup,
    no soak log, no stdout. Returns hit paths best-first. main() wraps this;
    the eval harness and --search call it directly.

    `dirs` overrides the corpora. The default is the two memory stores, and
    they are the only corpora the per-prompt hook may search — but the stages
    themselves are not specific to memories, and --search over some other
    corpus of markdown is a thing a caller legitimately wants. Overridden
    dirs are used verbatim: a
    caller naming a directory means that directory, not _search_root's
    search/ subtree of it.

    `deadline` is a time.monotonic() instant the retrieval must not run past.
    None (the eval, --search, the tests) means no clock: those callers are not
    on a prompt's critical path and would rather have the answer.
    """
    rec = stats if stats is not None else {}
    query = build_query(prompt.strip())
    if not query:
        return []
    dirs = [d for d in dirs if os.path.isdir(d)] if dirs else _search_dirs()
    if not dirs:
        return []

    def _stage(name: str, search: Callable[..., list[str]]) -> list[str]:
        # Per-dir isolation: one dir failing (an index that would not rebuild)
        # must not discard the other dir's results — and a failed dir
        # contributes nothing rather than reading as 'no hits'. A dir SKIPPED
        # for want of budget is neither, and is counted separately: an error
        # means the corpus could not answer, a skip means it was never asked.
        #
        # The deadline still has something to bound now that no subprocess is
        # spawned: a cold or invalidated index makes the FIRST dir's sync
        # re-chunk the whole corpus, and the second dir would then start its
        # own with the budget already gone.
        ranked = []
        skipped = 0
        for d in dirs:
            if deadline is not None and time.monotonic() >= deadline:
                skipped += 1
                continue
            with contextlib.suppress(Exception):
                ranked.append(search(query, d))
        rec[f"errs_{name}"] = len(dirs) - len(ranked) - skipped
        if skipped:
            rec[f"skipped_{name}"] = skipped
        return _interleave(ranked)

    for key in _LEX_COUNTS:
        _LEX_COUNTS[key] = 0
    _LEX_SECTIONS.clear()
    _LEX_MATCHED.clear()
    _LEX_SCORES.clear()
    hits = _stage("lex", _fts_dir)
    rec["lex_hits"] = len(hits)
    # The built query, kept for the offline shadow harness — the instrument
    # that decides whether the deleted semantic stage ever comes back.
    # It replays these against `ck --sem`, so this line IS its corpus; with no
    # consumer named here it reads as dead weight, which is how it came to be
    # deleted along with the LEX_THIN branch that used to gate it. Now
    # unconditional, because that gate is gone and the shadow re-derives
    # "thin" from `lex_hits` offline — which is the point of an offline
    # instrument: the threshold stays re-choosable after the fact.
    rec["query"] = query[:160]
    # Only when they fire: these are exceptions, and a key present on every
    # line is a key nobody greps for.
    rec.update({k: v for k, v in _LEX_COUNTS.items() if v})
    return hits


def _eligible(
    paths: list[str], terms: list[str]
) -> tuple[list[tuple[str, list[str], int]], list[str]]:
    """Hits that clear the relevance floor, in rank order, and which did not.

    The rejects are returned by PATH rather than counted because "why was that
    memory not recalled" is the question the soak log gets asked, and a count
    cannot answer it: the file that was found and dropped and the file that
    was never retrieved at all both appear as absence.

    Every candidate is judged, not only the ones that will fit: the floor
    decides what deserves a pointer at all, and MAX_HITS then decides how many
    of those a single prompt can afford. Stopping at the cap would leave the
    surplus unjudged, and a truncation notice offering `3 further matches`
    that turn out to be three coincidence hits is worse than no notice. The
    cost is one read per surplus candidate; _description, the other read, is
    still paid only for the lines actually rendered.
    """
    kept: list[tuple[str, list[str], int]] = []
    floored: list[str] = []
    for path in paths:
        matched, total, mtype = _relevance(terms, path)
        if _passes_floor(matched, total, mtype):
            kept.append((path, matched, total))
        else:
            floored.append(path)
    return kept, floored


def _floored_stat(floored: list[str]) -> dict:
    """Soak-log keys for what the relevance floor dropped: the count, and the
    files themselves.

    "Why wasn't memory X recalled" has three different answers — never
    retrieved, retrieved and floored, retrieved and capped — and the count
    alone distinguishes none of them, so answering it meant re-running the
    prompt against a corpus that had since changed. Basenames rather than
    contents: they are enough to tell the three apart, and they stay inside
    what _soak_log's contract admits into the log.

    The scores go alongside because floored and RANKED-HIGH is a different
    finding from floored and bottom-of-window: the first says the floor is
    overruling the ranker on this query shape, which is worth knowing before
    touching either.
    """
    stat: dict = {"floored": len(floored)}
    if floored:
        head = floored[:FLOORED_LOG_MAX]
        stat["floored_files"] = [os.path.basename(p) for p in head]
        stat["floored_scores"] = _scores(head)
    return stat


def _scores(paths: list[str]) -> list[float]:
    """Ranker scores for `paths`, positionally aligned with the name list they
    accompany in the log — 0.0 for a path the index has no score for.

    Three digits: these exist to be bucketed and compared against FLOOR_LEX
    (0.3), not to be reproduced exactly, and the full float would be the
    widest thing on the line.
    """
    return [round(_LEX_SCORES.get(p, 0.0), 3) for p in paths]


def _pointer_line(path: str, matched: list[str], total: int) -> str:
    """One pointer: where the file is, what it says it is, and the evidence
    for surfacing it — the matched terms, plus the section that matched, which
    is where to start reading a 400-line memory.

    Every pointer now carries term evidence, because every pointer comes from
    the term index and the floor rejects a hit that has none. The alternative
    tag this used to render — `[no direct term match — semantic-stage hit]` —
    went with the stage it named.
    """
    desc = _description(path)
    shown = ", ".join(matched[:6]) + (", …" if len(matched) > 6 else "")
    section = _LEX_SECTIONS.get(path)
    return (
        f"- {_display_path(path)}"
        + (f" — {desc}" if desc else "")
        + f" [matches {len(matched)}/{total} prompt terms: {shown}]"
        + (f" [section: {section}]" if section else "")
    )


@contextlib.contextmanager
def _sigterm_masked():
    """Hold SIGTERM for the duration of a block, then let it through.

    A pending signal is delivered on unblock, so the handler still runs and
    the process still leaves promptly — it just runs AFTER the block finished
    and can see the state the block established. That is the only way to make
    a write and the flag describing it indivisible; without it, every ordering
    of the two leaves some window where a kill produces either a duplicate
    record or none. Anything masked here must be short and non-blocking, or
    the mask is a way to outlast the harness's SIGTERM instead of a way to
    survive it.
    """
    try:
        prev = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    except (AttributeError, ValueError, OSError):
        yield  # no mask available: fall back to the unprotected ordering
        return
    try:
        yield
    finally:
        # Restore the mask we found rather than unblocking outright: these
        # nest (the injected path masks a block that calls done(), which masks
        # again), and an inner UNBLOCK would lift the outer block early.
        with contextlib.suppress(ValueError, OSError):
            signal.pthread_sigmask(signal.SIG_SETMASK, prev)


def main() -> None:
    t0 = time.monotonic()

    # Fail-open starts before the first read, not after it. json.load blocks
    # until the harness writes the payload, and a SIGTERM arriving in that
    # window used to find Python's default disposition and kill the process —
    # rc=-15 with no output, measured, which is the one status a hook on every
    # prompt must never return. There is no record to write yet (rec does not
    # exist), so the early handler only has to leave quietly; the
    # record-writing handler replaces it as soon as there is something to say.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    payload = json.load(sys.stdin)
    prompt = payload.get("prompt", "") or ""

    stripped = prompt.strip()
    rec: dict = {
        "prompt_sha": hashlib.sha256(stripped.encode()).hexdigest()[:12],
        "words": len(stripped.split()),
        "session": _log_session(payload.get("session_id", "")),
    }

    logged = False

    def done(outcome: str, **kw) -> None:
        nonlocal logged
        rec.update(outcome=outcome, ms=int((time.monotonic() - t0) * 1000), **kw)
        with _sigterm_masked():
            _soak_log(rec)
            logged = True

    # A killed hook used to leave NO record: the harness SIGTERMs at
    # HARNESS_TIMEOUT and the process died between the last stage and the
    # write, so the one outcome the soak log exists to expose — retrieval
    # taking longer than the prompt can wait — was the one outcome it could
    # not show. Write what we have, then leave immediately: os._exit skips
    # interpreter teardown, which is the right call inside a signal handler
    # holding a half-finished run, and 0 because a killed hook must still
    # look like a hook that had nothing to say.
    #
    # `logged` closes the window on the other side: SIGTERM arriving after a
    # terminal record was written would append a SECOND record for the same
    # prompt, one `injected` and one `killed`, and every rate the analyzers
    # compute is a count over records.
    #
    # The flag is set AFTER _soak_log returns, and both halves run under
    # _sigterm_masked so no signal can land between them. Order still matters
    # if the mask is ever unavailable: setting the flag first turns a mid-write
    # signal from a rare duplicate into a rare TOTAL LOSS — the handler sees
    # the flag, declines to write, and os._exit drops the buffered line without
    # flushing it, measured at zero records, zero bytes, and no malformed line
    # either, so nothing downstream can even count the loss. A duplicate is
    # visible in the log and can be reconciled; a silent hole cannot.
    #
    # SIGKILL is outside all of this. Nothing in a process can run on SIGKILL,
    # so a hard kill still loses whatever has not reached the file — the
    # harness sends SIGTERM, which is what these windows are about.
    def _flush_on_kill(signum, frame) -> None:
        if not logged:
            with contextlib.suppress(Exception):
                done("killed")
        os._exit(0)

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, _flush_on_kill)

    # The prompt-only gates come from prompt_gate() so that an analyzer asking
    # "would production have declined this?" gets the same answer this line
    # does. `gate:nodirs` stays here, between the two halves of that answer: it
    # is a fact about the machine rather than the prompt, and it outranks
    # gate:stopwords because a machine with no corpora could not have answered
    # the prompt whatever its vocabulary. Splitting the call is what preserves
    # that order without restating any condition.
    gate = prompt_gate(stripped)
    if gate in ("gate:envelope", "gate:shape"):
        return done(gate)
    if not _search_dirs():
        # No config, or one this build could not honour. Both leave the hook
        # with nothing to search; only the second has something to say, and it
        # says it here because a fail-open hook has no other voice — an
        # unhonourable config would otherwise look exactly like a corpus with
        # nothing in it.
        why = {"config": _CONFIG_ERROR} if _CONFIG_ERROR else {}
        return done("gate:nodirs", **why)
    if gate:
        return done(gate)

    # Session dedup: never re-inject a path already surfaced this session.
    # Read BEFORE retrieval, because one budget outcome is decidable without
    # it: a ledger holding pointers whose evidence was never recorded has
    # nothing to compare a new hit against, so the answer is "no" whatever
    # retrieval returns, and retrieving anyway would spend the full lexical
    # stage on a result already known to be discarded. Every other budget
    # state needs the hits in hand — the whole point of a replaceable budget
    # is that the answer depends on what was found.
    #
    # Everything from here on runs inside a recorder of last resort. The
    # `killed` outcome exists because a hook that leaves NO record is the one
    # result the soak log cannot count, and SIGTERM was only one way to reach
    # it: __main__ suppresses every exception to keep the hook fail-open, so
    # any raise past this point also exits 0, silent, with nothing logged.
    # Reproduced with a hand-written ledger whose evidence values were strings
    # — _replace's sort raised TypeError, and because the state file persists,
    # every prompt of that session was silently dead for the rest of its life.
    # The re-raise preserves fail-open (__main__ still swallows it); `logged`
    # keeps this from appending a second record for a prompt already recorded.
    try:
        session_id = str(payload.get("session_id", "") or "nosession")
        state_path = _session_state_path(session_id)
        shown, spent = _load_session(state_path)
        if len(spent) >= POINTER_BUDGET and any(e is None for e in spent.values()):
            return done("gate:budget", pointers=len(spent), ledger="legacy")

        hits = recall(stripped, stats=rec, deadline=t0 + BUDGET_SECONDS)
        if not hits:
            return done("nomatch")

        candidates = [p for p in hits if p not in shown]
        if not candidates:
            return done("deduped", hits=len(hits))

        # Per-hit relevance: which query terms the file contains + its type.
        # Filter feedback coincidence (soak review) BEFORE the MAX_HITS cap so a
        # strong lower-ranked hit surfaces when a floored one sits above it. The
        # overlap evidence also becomes the agent-facing [matches n/m] tag.
        query_terms = list(dict.fromkeys((build_query(stripped) or "").split()))
        eligible, floored = _eligible(candidates, query_terms)
        if not eligible:
            return done("floored", hits=len(hits), **_floored_stat(floored))

        room = POINTER_BUDGET - len(spent)
        if room > 0:
            picks = eligible[: min(MAX_HITS, room)]
            ledger = dict(spent)
            ledger.update({p: _evidence(m, t) for p, m, t in picks})
            evicted: list[str] = []
        else:
            offered = eligible[:MAX_HITS]
            picks, ledger, evicted = _replace(spent, offered)
            if not picks:
                # `best` is the strongest thing the BAR REFUSED, so it is taken
                # over what _replace was offered rather than over all eligible.
                # Quoting the best of everything the floor let through would
                # attribute to the budget a rejection it never made: past MAX_HITS
                # the cap dropped those, and the two decisions have to stay legible
                # apart in the log or "raise the budget" and "raise the cap" cannot
                # be told from each other afterwards.
                return done(
                    "gate:budget:weak",
                    hits=len(hits),
                    pointers=len(spent),
                    best=round(max(_evidence(m, t) for _, m, t in offered), 3),
                )

        lines = [_pointer_line(*e) for e in picks]
        fresh = [p for p, _, _ in picks]
        overlaps = [f"{len(m)}/{t}" for _, m, t in picks]

        # What the cap cost, named rather than dropped. The budget is a claim
        # about what a PROMPT can afford, not about what was worth finding, and
        # until this line the difference was invisible: two pointers looked the
        # same whether they were the whole answer or the top of a pile.
        #
        # The query goes in whole. Truncating it would produce a command that
        # searches for something other than what the count was taken over, i.e. a
        # number the agent cannot reproduce — and build_query has already bounded
        # it at 40 terms of word characters, which is also why it needs no shell
        # quoting beyond the surrounding pair.
        truncated = len(eligible) - len(picks)
        if truncated:
            rec["truncated"] = truncated
            # Named, for the same reason floored_files are: a count says the cap
            # bound, not what it cost. This is the TRUNCATED bucket of the
            # rank-oracle join, and answering "should the cap move again" needs the
            # identities — the join is against what the session went on to READ,
            # which is a filename. The share is deliberately not quoted here: it
            # has been restated three times as the oracle was corrected, and a
            # number duplicated into a comment is one that goes stale silently.
            #
            # By IDENTITY, not by position. `picks` is a prefix of `eligible` only
            # on the room > 0 branch; past the budget _replace filters the offered
            # window by evidence and returns a subsequence, so a positional
            # `eligible[len(picks):]` slides by however many it dropped from the
            # middle and names injected files as cut while the candidate the budget
            # actually refused appears nowhere. The count stayed right and every
            # identity was wrong, which is the worse failure — these names feed the
            # report's most-cut table and the cut-score percentiles, i.e. the
            # evidence the next MAX_HITS decision is argued from.
            picked = {p for p, _, _ in picks}
            cut = [e for e in eligible if e[0] not in picked][:FLOORED_LOG_MAX]
            rec["truncated_files"] = [os.path.basename(p) for p, _, _ in cut]
            rec["truncated_scores"] = _scores([p for p, _, _ in cut])
            lines.append(
                f"- …{truncated} further match{'es' if truncated > 1 else ''} — "
                f'search: {_search_cli()} "{" ".join(query_terms)}"'
            )

        # Deliver, spend the dedup state, record — in that order and indivisibly.
        # Each of the three is a claim about the others, and a kill between any
        # two of them makes the surviving pair a lie. Recording first said
        # "injected" for pointers still sitting in a buffer that os._exit would
        # drop; persisting dedup first burned those paths for the rest of the
        # session, so the prompt lost its pointers AND could never be offered them
        # again. Delivery is what the other two are about, so it goes first, and
        # the flush is the delivery — stdout is a pipe here, hence block-buffered,
        # and an unflushed write has reached nobody.
        #
        # A closed reader is therefore not "injected" at all: nothing arrived, so
        # nothing is spent — neither the dedup set nor a budget eviction — and the
        # record says so. `injected` keeps its name in that record because the
        # field means "which pointers this run produced", and the outcome is what
        # tells the analyzers whether they landed. `evicted` cannot keep its name
        # the same way: it does not describe this run's output but a mutation of
        # the ledger, so it is reported only when that mutation actually reached
        # the disk. Delivery is not enough to know that — the state write has its
        # own failure, swallowed here so a read-only cache dir cannot cost the user
        # a prompt — and a run that delivered while the write failed would
        # otherwise claim displacements no later session can observe. Worse than
        # the log lie: with the write lost, `shown` does not advance either, so the
        # session re-offers the same pointer on every prompt and the budget stops
        # bounding anything. `state: "unwritten"` is how that becomes visible in
        # the log rather than reading as a run of legitimate injections.
        #
        # Masking SIGTERM across a write is only safe because this write cannot
        # block. MAX_HITS caps it at three pointer lines plus a truncation notice,
        # each line a path and a description already truncated to DESC_MAX_CHARS,
        # so the payload is ~2KB against a pipe buffer of 16KiB at its smallest —
        # the flush returns without waiting for a reader to drain anything. Were
        # the payload ever to approach the buffer, a slow reader would park this
        # section with SIGTERM held and the harness's timeout would stop being
        # able to stop the hook. Raising MAX_HITS or lifting the description cap
        # is what would do it.
        delivered = True
        with _sigterm_masked():
            try:
                sys.stdout.write(
                    "Possibly relevant memories — the [matches n/m] tag shows "
                    "which of your prompt's terms each file contains, and "
                    "[section: ...] the part of the file that matched; read the "
                    "ones whose matched terms are load-bearing for the task, skip "
                    "incidental overlaps:\n" + "\n".join(lines) + "\n"
                )
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                delivered = False

            # Written beside and renamed over, never truncated in place. The ledger
            # is the one piece of state here that cannot be rebuilt — the FTS index
            # regenerates from the corpus, this does not — and `open(path, "w")`
            # destroys it before writing the replacement, so anything that stops
            # the write in between (ENOSPC, a SIGKILL after the harness's grace,
            # a full volume) leaves a valid prefix of invalid JSON. _load_session
            # reads that as a fresh session, so the budget silently resets to zero
            # and the session spends it again on paths it has already shown, with
            # no record that it happened. os.replace is atomic within the cache
            # dir, so a reader sees the old ledger or the new one and never a torn
            # one; it also turns two same-session hook processes from a possible
            # corrupt file into a plain lost update. _fts_sync takes exactly this
            # care with the index, which is the rebuildable one.
            persisted = False
            if delivered:
                tmp_path = f"{state_path}.{os.getpid()}.tmp"
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"shown": sorted(shown | set(fresh)), "spent": ledger}, f
                        )
                    os.replace(tmp_path, state_path)
                    persisted = True
                except OSError:
                    # dedup degrades to per-prompt; injection still capped
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)

            done(
                "injected" if delivered else "output-lost",
                injected=[os.path.basename(p) for p in fresh],
                overlap=overlaps,
                scores=_scores(fresh),
                **_floored_stat(floored),
                **({} if persisted or not delivered else {"state": "unwritten"}),
                **(
                    {"evicted": [os.path.basename(p) for p in evicted]}
                    if persisted and evicted
                    else {}
                ),
            )
    except Exception as exc:
        if not logged:
            with contextlib.suppress(Exception):
                done("error", err=type(exc).__name__)
        raise


# --- exit codes, and the state each one names --------------------------------
#
# grep's three, plus one. 0/1/2 stay exactly grep's — found, found nothing, the
# search itself failed — because the caller is an agent deciding whether to go
# read something, and those three are the moves it already knows. The fourth
# exists because none of them can say "this installation has nothing to search":
# an inert memkit answers every query with the same empty result an exhaustive
# search of a real corpus produces, and an agent reading that as absence
# concludes the memory is not there rather than that it never looked. That
# collapse is why the state gets a code of its own rather than a caveat in the
# message. Named constants because doctor and the skills branch on these from
# outside this file, and a number in two repos is a number that drifts.
EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_ERROR = 2
EXIT_INERT = 3


def _print_config() -> int:
    """`--debug-config`: what this installation resolved, and from where.

    The operator-facing answer to "why did the hook say nothing" — which is
    otherwise indistinguishable from a corpus with nothing to say, because the
    hook is fail-open by construction. Honours the per-root environment
    overrides the hook path refuses, so a developer can point a session at a
    fixture tree without the every-prompt path ever growing that ability.

    An inert installation exits EXIT_INERT, not 0 — and that verdict comes from
    _config_state, the same one `--search` reads. Printing "inert" and exiting
    successfully asks the reader to parse prose to learn that nothing is wired
    up, and the reader here is usually an agent that checked the status and
    moved on; printing `searched` beside a store directory that is not there
    tells it something false about the store as well.
    """
    cfg, error, inert = _config_state(honor_env_overrides=True)
    if error:
        print(f"memory-recall: {error}", file=sys.stderr)
        return EXIT_ERROR
    if cfg is None:
        print(
            "config:     none — inert: no stores, no pointers "
            f"(no --config, ${CONFIG_ENV} unset)"
        )
        return EXIT_INERT
    print(f"config:     {cfg.path} (schema {SCHEMA})")
    print(f"search_cli: {cfg.search_cli}")
    searched = cfg.searched_stores()
    for store in cfg.stores:
        live = cfg.store_dir(store, "live")
        gated = "always" if store.cwd_gate is None else f"cwd under {store.cwd_gate}"
        # Three states, not two. A store this session is allowed to read and
        # whose directory does not exist was reported as `searched`, which is
        # the single most misleading line this command can print: it names the
        # path AND asserts the path is being read.
        if store not in searched:
            state = "NOT searched here"
        elif not os.path.isdir(live):
            state = "NOT on disk"
        else:
            state = "searched"
        print(f"store {store.id}: {live} [{store.role}; {gated}; {state}]")
    if inert:
        print(f"inert:      {inert}")
        return EXIT_INERT
    return EXIT_OK


def search_cli(argv: list[str]) -> int:
    """`--search "<terms>" [--dir PATH ...]` — the hook's own retrieval, on
    demand, printing the pointer lines it would have injected.

    grep's exit codes plus EXIT_INERT (see that block), because the caller is
    an agent deciding whether to go read something and each case wants a
    different next move. Deliberately NOT fail-open: that contract exists so a
    broken hook cannot block a prompt, and there is no prompt here — exiting 0
    and silent on a broken index would present it as a corpus with nothing to
    say.

    No cap and no session state. MAX_HITS and POINTER_BUDGET are claims about
    what may be PUSHED into a session's context unasked; a search the agent
    ran is the opposite of that, and suppressing a file because the hook
    already mentioned it would answer a different question than the one asked.
    The soak record is still written, under its own outcome, so the log keeps
    counting every use of the retrieval path.

    It is the same retrieval path the hook runs, so what it returns is what a
    prompt in the same words would have been offered — with the cap and the
    dedup removed, not the floor. A query the corpus has no words for returns
    nothing and exits EXIT_NO_MATCH; there is no nearest-neighbour guess behind
    that any more.
    """
    # Local import: this is the only caller, and the hook path — which runs on
    # every prompt and parses no arguments — need not pay for a mode it never
    # enters. Small (0.54 ms warm, 2.7 ms cold, measured) but not nothing, and
    # the same trade the scan's rejected probe pass was decided on.
    import argparse

    t0 = time.monotonic()
    ap = argparse.ArgumentParser(
        prog="memory-recall",
        description="Search the memory corpora the way the recall hook does.",
    )
    ap.add_argument("--search", metavar="TERMS")
    ap.add_argument(
        "--dir",
        action="append",
        metavar="PATH",
        help="corpus to search instead of the memory stores (repeatable)",
    )
    ap.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=f"config file naming the stores to search (default: ${CONFIG_ENV}). "
        "An alternative to exporting the variable, not a second spelling of it: "
        "verifying a config you just wrote should not mean mutating the "
        "environment of whatever ran this",
    )
    ap.add_argument(
        "--debug-envelope-probes",
        action="store_true",
        help="emit one synthesized envelope string per gated marker, as JSON. "
        "For a downstream stratifier that must classify the same markers",
    )
    ap.add_argument(
        "--debug-config",
        action="store_true",
        help="print the resolved configuration and exit. The one place per-root "
        "environment overrides are honoured — the hook path ignores them, because "
        "which trees an every-prompt hook reads is not the ambient environment's "
        "decision to make",
    )
    args = ap.parse_args(argv)
    _use_config(args.config)

    if args.debug_envelope_probes:
        print(json.dumps(envelope_probes()))
        return EXIT_OK
    if args.debug_config:
        return _print_config()
    if not args.search:
        ap.error("--search is required")

    stripped = args.search.strip()
    rec: dict = {
        "prompt_sha": hashlib.sha256(stripped.encode()).hexdigest()[:12],
        "words": len(stripped.split()),
        "session": "cli",
    }
    dirs = [os.path.expanduser(d) for d in args.dir] if args.dir else None
    # recall() drops dirs that are not there, because the hook's own two
    # corpora legitimately come and go with the cwd. A --dir the caller
    # NAMED is different: a typo would search whatever is left, or nothing,
    # and report "no matches" for a corpus never opened.
    missing = [d for d in dirs or [] if not os.path.isdir(d)]
    if missing:
        print(
            f"memory-recall: not a directory: {', '.join(missing)}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    # The config is consulted whenever the stores are what will be searched,
    # and ALSO whenever the caller named one — even under --dir, where it
    # decides nothing. `memory-recall --config <the one just written> --dir
    # <corpus>` is a verification invocation, and letting it pass on a config
    # that does not parse is the one way for that check to come back green
    # about a file it never opened. Same reasoning as the --dir typo above.
    if args.config is not None or dirs is None:
        # Whether there is anything to search is settled BEFORE retrieval,
        # because recall() cannot answer it: an unconfigured machine and an
        # unanswerable query both come back as an empty list, and the caller
        # would read the second meaning of the first. The hook is right to be
        # silent about this — it has a prompt to get out of the way of — which
        # is exactly why the CLI has to say it instead.
        #
        # Inertness is not asked under --dir: the caller named the corpus, so
        # the config has no say in what gets searched and no standing to make
        # this run inert. That is the shape of the zero-config trial an adopter
        # runs before there is a config to be inert. A config that cannot be
        # PARSED is a different matter, and is refused either way.
        _, error, inert = _config_state()
        if error or (dirs is None and inert):
            if dirs is None:
                # Counted for the same reason the hook counts gate:nodirs — a
                # run of the retrieval path that never reached a corpus is
                # still a use of it. A --dir run that failed on its config is
                # not: nothing was searched and nothing was going to be.
                why = {"config": error} if error else {}
                rec.update(
                    outcome="cli:nodirs",
                    ms=int((time.monotonic() - t0) * 1000),
                    **why,
                )
                _soak_log(rec)
            if error:
                print(f"memory-recall: {error}", file=sys.stderr)
                return EXIT_ERROR
            print(
                f"memory-recall: inert — {inert}; this is not a claim of "
                "absence",
                file=sys.stderr,
            )
            return EXIT_INERT
    hits = recall(stripped, stats=rec, dirs=dirs)
    terms = list(dict.fromkeys((build_query(stripped) or "").split()))
    eligible, floored = _eligible(hits, terms)
    lines = [_pointer_line(*e) for e in eligible]
    rec.update(
        outcome="cli",
        ms=int((time.monotonic() - t0) * 1000),
        shown=len(lines),
        **_floored_stat(floored),
    )
    _soak_log(rec)

    if not lines:
        # recall() suppresses per-dir failures so one dead corpus cannot cost
        # the other its results — which means an empty answer and a failed
        # search look identical from out here unless the error count is
        # consulted. With nothing found, any failed dir makes "no matches"
        # a claim this run did not establish.
        if rec.get("errs_lex"):
            print(
                "memory-recall: no matches, but "
                f"{rec['errs_lex']} dir(s) failed to search"
                " — result is not a claim of absence",
                file=sys.stderr,
            )
            return EXIT_ERROR
        return EXIT_NO_MATCH
    print("\n".join(lines))
    return EXIT_OK


def cli() -> None:
    """Entry point for the `memory-recall` console script AND for running this
    file directly as a hook command.

    Two modes in one binary because the harness invokes the FILE (a path in a
    settings entry) while an agent invokes the COMMAND, and both have to reach
    the same retrieval or the recipe the hook advertises is a different tool
    from the hook. Arguments mean --search; no arguments means read a hook
    payload off stdin.
    """
    if len(sys.argv) > 1:
        try:
            sys.exit(search_cli(sys.argv[1:]))
        except SystemExit:
            raise
        except Exception as exc:
            # Never exit EXIT_NO_MATCH or EXIT_INERT on a failure: both codes
            # are spoken for, and an agent reading either as "there is no such
            # memory" or "nothing is set up here" would stop looking.
            print(f"memory-recall: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
    # Fail-open: no output, exit 0 — never block the prompt.
    with contextlib.suppress(Exception):
        main()
    # suppress() above does NOT cover a reader that closed early, and that is
    # an ordinary thing for the harness to do. stdout here is a pipe, so it is
    # block-buffered and the pointer block can still be sitting unwritten at
    # this point; CPython flushes it during interpreter shutdown, long after
    # the last handler in this file has run, and turns the BrokenPipeError
    # into exit status 120 — measured, on main as well. Flush while the
    # failure is still catchable, and on failure point fd 1 at /dev/null so
    # the shutdown flush finds somewhere harmless to land. Losing the pointers
    # is already certain by then; the only question is whether the hook also
    # reports a failure the prompt will be blamed for.
    try:
        sys.stdout.flush()
    except Exception:
        with contextlib.suppress(Exception):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(0)


if __name__ == "__main__":
    cli()
