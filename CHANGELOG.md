# Changelog

Notable changes to memkit, newest first, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Dates are the day a release's state was merged. What `/plugin install` puts on
your machine is the tree at the sha in `.claude-plugin/marketplace.json`, which
moves one commit later — [docs/RELEASING.md](docs/RELEASING.md) explains the
ordering.

## [Unreleased]

### Added

- **"Keep your store in git" in `docs/STORE.md`** — the store as a private
  repository, which findings need history to say anything at all, and where
  Claude Code's own agent-written memories land. They default to a per-project
  directory outside every store; `memoryDir` redirects them, and pointing it at
  `<store>/search` rather than the store root is what keeps the harness's flat
  writes inside the corpus root.

## [0.4.0] — 2026-08-31

### Added

- **A mutation sweep that ships with the tree it audits**, at
  `tools/mutation_sweep.py` with its 365 probes beside it. Each breaks one
  load-bearing rule and requires the paired tests to fail, so a green suite is
  evidence that a broken tree would have gone red rather than an assumption
  that it would. Nothing an adopter runs, but it arrives in the payload like
  everything else the repository tracks — `docs/ADMISSION.md`'s counts move
  with it.
- **`task-outcomes`, a doctor check.** The subagent population as its own
  histogram, counted and glossed the way `gate-outcomes` counts the prompt
  one. The two are never summed: one record per prompt against one per spawn,
  a 15-second budget against a 7-second one, so a rate over the union is a
  rate over an unknown mixture. `gate-outcomes` now says which population it
  counts.

### Fixed

- **HOT-STALE can see a ledger row change.** The walk that dates a row reads
  `git log -p`, and it skipped the `---`/`+++` file headers by excluding any
  line whose second character was another `+` or `-`. That is every row in the
  file: a row is a markdown list item, so an edit to one arrives as
  `+- [name](file.md)` and `-- [name](file.md)`, and both went out with the
  headers. The date fell back to whatever older commit happened to name the
  file on an indented line, so the warning outlived the author doing exactly
  what it asked. Over one consuming store's hot ledger the walk saw 4 of 131
  rows; it now sees 131, and a header is recognised by its own shape.
- **Doctor renders what a `task:` outcome means.** Every one of the twenty
  `task:*` reasons in doctor's map was unreachable: the histogram counted the
  per-prompt population, which excludes every record the subagent path writes,
  and `subagent-delivery` printed the raw name. An adopter met `task:killed`
  and a full stop. Every check that names an outcome now goes through one
  gloss, and a name from a newer build says so rather than appearing as a word
  the reader should recognise.
- **`plugin-diagnostics` explains both trust refusals.** It defined
  `trust:unconfigured` in its remedy and printed `trust:config-error` bare, so
  of a two-name vocabulary the name an adopter with a broken config is looking
  at was the one explained nowhere in the report.
- **`memory-eval` and `memkit init` survive a lone surrogate.** The corpus
  fingerprint and init's digest helper both encoded strictly, so a store id
  carrying an escaped `\udXXX` from the config, or a path arriving through
  `sys.argv`'s `surrogateescape`, ended the run with a `UnicodeEncodeError`
  out of a hashing loop — before the eval scored anything or `--dry-run`
  printed its manifest.

## [0.3.0] — 2026-08-29

### Added

- **Subagent pointer delivery.** A second hook, on `PreToolUse` matched to the
  `Agent` tool, appends pointer lines to a subagent's brief before it is
  spawned. It rewrites the tool call rather than printing to the transcript,
  echoes the brief back verbatim, and marks the appended block as retrieved
  data. Registered only on the plugin channel; nix and pip installs register
  what their own `settings.json` names. A healthy plugin install now reports
  `Hooks (2)`.
- **`memkit init`.** Declared but not shipped in 0.2.1, and implemented here.
  It prints a manifest of every path it would write plus a digest, and writes
  nothing until you approve it: the state directory, the config at the path you
  named, the store skeleton, one canary memory, and a journal record written at
  each mutation. `--wire-claude-md` and `--auto-dream-off` are the only writes
  outside that set and each has its own flag.
- **`memkit doctor`.** Also declared in 0.2.1 and implemented here. One
  envelope, one line per check, `--json` for a machine — among them the config
  route and who authored the config, the corpus root, the index state, the
  registration count, subagent delivery, the gate outcomes and the state
  directory. It runs the installed hook once with a query it seeds itself,
  because a store that answers proves nothing about the path that serves
  prompts, and the report says so.
- **`/memkit:init` and `/memkit:doctor`** as skills, so the two commands are
  reachable from a session on the plugin channel.
- **A long-brief slice in `memory-eval`**, gating what reaches a subagent on
  served and leak rates over paired fixtures rather than on the prompt path's
  numbers.
- **A `task:` outcome vocabulary** for the subagent path, kept separate from
  the prompt path's so a rate computed over either is a rate over one
  population. `README.md`'s outcome table lists all of them.

### Changed

- **The pointer block's delimiters carry a per-run nonce** and are held to
  whole-line positions, replacing the previous approach of rewriting delimiter
  lookalikes out of retrieved text. The block declares its own line count and
  states, in the text a model reads, that a tag appearing inside it is file
  content.
- **Every program memkit starts is an absolute path to an executable file.**
  The interpreter comes from the `interpreter` field in your config or from a
  fixed list of absolute system paths, and never from a `PATH` search, so a
  checked-in `node_modules/.bin` or a directory-exported venv cannot supply it.
  Git runs one of a closed set of templates with configuration keys that name a
  program overridden. The per-prompt path starts no process at all beyond its
  own `exec`.
- **Derived state is collected on an allowlist of names memkit writes**, at
  most hourly and bounded per run, so the default for anything else in the
  state directory is keep.
- **A record is bound before the dispatch chooses a path**, carrying the working
  directory and the session, so a hook killed in that window still leaves a line
  you can attribute.

### Fixed

- A payload whose `prompt` was present but not a string raised inside the hook
  before it had a record to write. It is recorded as `main:badpayload`, naming
  the type. Present since 0.1.0.
- A store whose index build was truncated — by the budget, or by a single file
  over the per-file cap — reported as though retrieval were complete. `doctor`
  names that state and distinguishes the two causes.
- Two `doctor` rows reported more than they had measured: the subagent-delivery
  row said a delivery had happened where it had only read the registration, and
  the hook-probe row quoted a timeout that was not the one governing the run.
- `memory-eval`'s guard for an incomplete hook copy checked a hand-written list
  of names that had drifted, so a copy missing a function passed the guard and
  failed mid-run instead of skipping. The list is derived from what the guarded
  code reaches.
- The state sweep left `<name>.<pid>.tmp` files behind whatever their age.
  They are collectible after an hour, which is well past any window in which
  the rename that would claim them is still coming.

## [0.2.1] — 2026-08-25

### Added

- `docs/RELEASING.md`: the two-pull-request release procedure, which commit to
  tag, and the checklist.

### Fixed

- `.claude-plugin/plugin.json`, the version in the marketplace entry, the rev
  the `uvx` recipes name, and `docs/ADMISSION.md`'s file counts all described
  the release before them. They describe the tree they ship in.
- The outcome table said it was exhaustive while two live `trust:*` values sat
  outside it. Both are rows, scraped from the source rather than transcribed.
- The uninstall recipe was the one `uvx --from` call site with no rev on it, so
  it ran a different build than the one being checked.

## [0.2.0] — 2026-08-25

### Added

- `docs/STORE.md`: what a memory file is, how a pointer gets chosen, and a
  worked example executed by a test rather than transcribed.
- `docs/ADMISSION.md`: what a plugin install puts on the machine, with the
  recipe for reproducing every count in it.
- A CI tier that installs from GitHub, which nothing had exercised before.

### Changed

- The marketplace entry serves the payload over a transport an adopter can
  actually install from.
- `--debug-config` prints the corpus root, its file count and any files
  stranded outside it; `--search` names what it searched when nothing matches.
- The four prompt-shape gates get their own outcome names. `gate:shape`, which
  0.1.x writes for all four, stays in the outcome table as the value that
  decodes a log from that release.

## [0.1.0] — 2026-08-23

### Added

- First release: the retrieval hook, the integrity checker and the eval, over
  one config file.
