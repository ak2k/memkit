# memkit

Injects pointers to your own memory files into every Claude Code prompt:
lexical retrieval over a directory of markdown, plus a checker and an eval for
it. It never injects file contents — the model decides what to open.

When it fires, this is what lands in the prompt:

```
- ~/notes/search/postgres-connection-pool.md — PgBouncer in transaction mode breaks session-scoped features — prepared statements, advisory locks, and SET LOCAL do not survive. [matches 5/7 prompt terms: prepared, statements, pgbouncer, transaction, pooling] [section: PgBouncer transaction mode]
```

- [Quick start](#quick-start) — install to first pointer
- [Your store](#your-store) — what a memory file is · [docs/STORE.md](docs/STORE.md) for the rest
- [Why nothing appeared](#why-nothing-appeared) — every silent gate, in the order you hit them
- [The four commands](#the-four-commands) · [Config](#config) · [Exit codes](#exit-codes)
- [Install (details)](#install-details) — the other two channels, and every caveat
- [Leaving](#leaving) — disable, uninstall, and what survives either
- [Retrieval disclosures](#retrieval-disclosures) — what was measured, and on what
- [docs/ADMISSION.md](docs/ADMISSION.md) — what an install puts on your machine
- [docs/ROLLOUT.md](docs/ROLLOUT.md) — fleet rollout and rollback

## Status

Pre-1.0 and shaped by one deployment. The interfaces below are the
ones its own consumer uses; treat them as unstable until this repo has a
second adopter. See [Retrieval disclosures](#retrieval-disclosures) before
assuming any measured claim generalises to your corpus.

**This page describes `main`; the marketplace installs a release.** What
`/plugin install` puts on your machine is the tree at the sha pinned in
[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json), so between
releases this page can describe behaviour your copy does not have yet. Releases
follow merged behaviour changes rather than a calendar, and each one re-aligns
the two. Where a behaviour has landed here and not in a release, this page marks
it *(from the next release)*.

Bugs, questions, and the second adopter's experience:
[github.com/ak2k/memkit/issues](https://github.com/ak2k/memkit/issues). If you
have a security concern, that tracker is the only channel this project has —
say so in the title and I will move it somewhere private.

## Quick start

Four steps, on the Claude Code plugin channel. Nothing here needs a GitHub
account or SSH keys — both clones are anonymous over HTTPS.

**1. Install.**

```
claude plugin marketplace add ak2k/memkit
claude plugin install memkit@memkit --yes \
  --config memkitConfig="$HOME/.config/memkit/memkit.json"
```

**2. Check it registered, and that the option arrived.** A *wrong* `--config`
value installs exactly as quietly as a right one — the omitted case now warns,
the mistyped case does not — so read the value back:

```
claude plugin list                                   # memkit@memkit must be ✔ enabled
claude plugin details memkit@memkit                  # must report Hooks (1)
python3 -c 'import json,os,sys;print(json.load(open(os.path.expanduser(
  sys.argv[1])))["pluginConfigs"]["memkit@memkit"]["options"])' \
  "${CLAUDE_CONFIG_DIR:-~/.claude}/settings.json"
```

The last one prints `{'memkitConfig': '<the path you passed>'}`. An empty
result, or a `KeyError`, is an install that skipped or mistyped `--config` —
which is inert by design and says nothing at runtime, so this is the only place
it surfaces.

`Hooks (1)` says the hook is registered; it does **not** say the plugin is
enabled — a disabled plugin still reports it — which is why `plugin list` comes
first.

**3. Write the config.** Four lines is the whole minimum. Put it somewhere you
would keep a dotfile — **not** under `~/.cache/`, which this page tells you
elsewhere is disposable and which the platform may purge; the config is the one
file here that nothing regenerates. Pass this path to `memkitConfig` in step 1:

```
mkdir -p ~/.config/memkit
cat > ~/.config/memkit/memkit.json <<'EOF'
{ "schema": 1,
  "roots": { "notes": { "kind": "path", "path": "~/notes" } },
  "stores": [ { "id": "notes", "dir": ".", "live_root": "notes" } ] }
EOF
```

Everything else on the [Config](#config) page is optional.

**4. Write one memory and ask for it.** Memories go in `search/` — make it now
even for your first file, so nothing has to move later. (Creating `search/`
under a store that already has memories above it takes those out of retrieval:
[Your store](#your-store).)

```
mkdir -p ~/notes/search
cat > ~/notes/search/postgres-connection-pool.md <<'EOF'
---
description: PgBouncer in transaction mode breaks session-scoped features — prepared statements, advisory locks, and SET LOCAL do not survive.
---

# PgBouncer transaction mode

Transaction pooling hands a different backend to every transaction.
EOF
```

Now ask Claude Code *"why do prepared statements break under pgbouncer
transaction pooling"*, and the pointer at the top of this page is what arrives.

**Checking it by hand.** `memkit-recall` is on the *agent's* `PATH`, not your
shell's — nothing is added to your terminal — so from your own shell reach the
installed copy by path:

```
CFG="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
PLUGIN="$CFG/plugins/cache/memkit/memkit"
RECALL="$(ls -d "$PLUGIN"/*/bin/memkit-recall | tail -1)"

# the config the HOOK will use — read back out of settings.json, not retyped
MEMKIT_CFG="$(python3 -c 'import json,os,sys;print(json.load(open(os.path.expanduser(
  sys.argv[1])))["pluginConfigs"]["memkit@memkit"]["options"]["memkitConfig"])' \
  "$CFG/settings.json")"
test -f "$MEMKIT_CFG" || echo "the installed option names a file that is not there: $MEMKIT_CFG"

"$RECALL" --config "$MEMKIT_CFG" --debug-config
"$RECALL" --config "$MEMKIT_CFG" --search "pgbouncer pooling"
```

Feeding the read-back into `--debug-config` is the point: checking the path
*you* meant to install will report a healthy store while the hook reads a
different one, and a one-character typo in `memkitConfig` is the likeliest
install mistake there is.

**When a change takes effect.** Editing the config needs no restart — the hook
re-reads it on every prompt, measured: a session whose config named no stores
started returning pointers on the next prompt after the file gained one, same
session, nothing relaunched. The same goes for adding, editing or archiving a
memory. Installing, enabling or disabling the plugin is different: Claude Code
reads its hook registrations when a session starts, so **start a new session
after an install** — that is also why step 2 checks `plugin list` rather than
trusting the install's success line.

`--debug-config` prints the config it resolved and, per store, the directory
retrieval will read *(from the next release: also its file count, and a warning
naming any memories stranded outside it)*. `--search` applies fewer gates than
the hook — see [Why nothing appeared](#why-nothing-appeared) — so it answering
while a session stays quiet is information, not a contradiction.

## Your store

A store is a directory of markdown files. Point a config at it and the next
prompt searches it — you do not need ledgers, a particular layout, or even
frontmatter to get pointers.

A memory is one file, one claim:

```markdown
---
name: postgres-connection-pool
description: PgBouncer in transaction mode breaks session-scoped features — prepared statements, advisory locks, and SET LOCAL do not survive.
type: reference
---

# PgBouncer transaction mode

Transaction pooling hands a different backend to every transaction, so
anything the client thinks is session state is gone by the next statement.
```

Ask *"why do prepared statements break under pgbouncer transaction pooling"*
and that file arrives as a pointer:

```
- ~/notes/search/postgres-connection-pool.md — PgBouncer in transaction mode breaks session-scoped features — … [matches 5/7 prompt terms: prepared, statements, pgbouncer, transaction, pooling] [section: PgBouncer transaction mode]
```

`description:` is the whole of what the agent sees before deciding to open the
file, so write it as the claim rather than a title. Everything else about
stores — what is searched and what is skipped, how a pointer gets chosen, the
160-character cap, git, archiving, the optional checker, and a paste-able
`CLAUDE.md` block for letting your agent write the memories — is in
[docs/STORE.md](docs/STORE.md).

## Why nothing appeared

You typed something and no pointer came back. Every gate below is silent by
design — the hook exits 0 whatever happens, because a hook that fails any other
way blocks your prompt — so this is the list, in the order a prompt meets them.

The fastest triage is to ask the CLI the same question. **It applies fewer
gates than the hook**, so a `--search` that answers while the session stays
quiet localises the problem to this list rather than to your store. On a plugin
install `memkit-recall` is on the agent's `PATH` and not your shell's, so from
your own terminal reach it by path:

```
PLUGIN="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins/cache/memkit/memkit"
RECALL="$(ls -d "$PLUGIN"/*/bin/memkit-recall | tail -1)"

"$RECALL" --config <your config> --debug-config      # what resolved, and from where
"$RECALL" --config <your config> --search "<terms>"  # would anything match at all
```

Inside Claude Code's Bash tool the bare name works, since that is the `PATH`
the plugin adds to. On a pip or nix install the command is `memory-recall` and
is on your own `PATH`.

| what stopped it | how you can tell | what to do |
|---|---|---|
| **The plugin was installed mid-session** | `plugin details` says `Hooks (1)` and `--search` answers, but no prompt injects and no record is written | Claude Code registers hooks when a session starts — start a new session |
| **The plugin is disabled** | `claude plugin list` shows `✘ disabled`; `plugin details` still says `Hooks (1)` | re-enable it |
| **No config reached the hook** | `--debug-config` prints `config: none`, exit 3 | read the option back out of `settings.json` — [Quick start](#quick-start) step 2 |
| **The config path is wrong** | `--debug-config` says the path does not exist | fix the path and re-run the install command |
| **The prompt was under three words** | `gate:short` *(`gate:shape` on the current release)*; `--search` with the same words answers | deliberate — a two-word prompt has no subject to retrieve on |
| **The prompt began with `/`** | no record at all — Claude Code resolves slash commands before the hook runs, so an empty `tail -1` is the tell | deliberate: a slash command is an instruction to Claude Code, not a question about your work |
| **The prompt was over 4000 characters** | `gate:long` *(`gate:shape` on the current release)* | deliberate, and the one most people meet: a pasted stack trace or log excerpt retrieves on the paste's vocabulary rather than on your question. Ask in your own words, then paste |
| **The prompt was all common words** | `gate:stopwords` | "is it the" leaves no term to search on |
| **The same prompt already fired this session** | `deduped`; the first identical prompt got pointers | deliberate: a memory is offered once per session |
| **The prompt began with an editor or tool envelope** | `gate:envelope`; the prompt started with something like `<system-reminder>` | deliberate — that text is not what you asked |
| **The hook ran out of time** | `killed` | it is registered with a 15-second timeout and gives up rather than delaying your prompt. A first run on a large store builds the index; the next one is fast |
| **The corpus is not where you think** | `--debug-config` prints the corpus root *(from the next release: its file count, and a line naming files stranded outside it)*. On an older release: a pip or nix install has the checker, which reports `STRAY-ROOT: ./<file>` for a memory left above `search/`; a plugin install does not ship it, so `ls` your store root for markdown sitting above `search/` | move them, or point `dir` at the right directory |
| **Nothing matched well enough** | `--search` exits 1 *(from the next release: and prints `no match in N files under <root>`)* | see [How a pointer gets chosen](docs/STORE.md#how-a-pointer-gets-chosen) — a match on a common English word alone will not carry a pointer |
| **The session budget is spent** | 30 pointers already delivered this session | deliberate; a stronger match still displaces a weaker one |

The prompt-shape gates — the three-word floor, the slash prefix, the paste
ceiling, the all-stopword case and the envelope prefix — are the reason the hook
and `--search` can honestly disagree about the same words: `--search` applies
none of them. So do the once-per-session rule and the session budget, which are
about the session rather than the words. `memkit doctor`, which would run this
list for you, is not in this build.

**The outcome vocabulary.** Each record's `outcome` names what happened. These
are all of them on `main`; a release before the prompt-shape split writes
`gate:shape` where the table below names four:

| outcome | meaning |
|---|---|
| `injected` | pointers were written into the prompt |
| `gate:envelope` · `gate:empty` · `gate:slash` · `gate:short` · `gate:long` · `gate:stopwords` | the prompt's shape, per the table above. The middle four are *(from the next release)* — releases before the split record all of them as **`gate:shape`**, which is what a log shows today |
| `gate:nodirs` | nothing to search: no config, or no store on disk and in scope here |
| `nomatch` | the stores were searched and nothing came back |
| `deduped` | every match had already been offered this session |
| `floored` | matches existed and none cleared the relevance bar |
| `gate:budget:weak` | the session's 30 pointers are spent and nothing beat the weakest |
| `gate:budget` | the same, on a session ledger written before this build recorded per-pointer evidence — that budget cannot be reasoned about, so it is terminal |
| `dup-registration` | two installs on one machine registered the same hook. Not a prompt outcome — it carries `"concludes": false` and is written beside the record the prompt makes for itself |
| `killed` | the hook was stopped — timeout, or the session ended |
| `output-lost` | pointers were built and the write did not land |
| `error` | an unexpected failure; the record names the exception type |
| `cli:*` | written by `--search`, not by a prompt. `"concludes": false` marks these |

One record per prompt, so when two gates could apply the record names one — a
repeated prompt whose other match was below the bar records `floored`, not
`deduped`.

To see the hook working at all, submit a prompt and read the last soak record:

```
tail -1 "${XDG_CACHE_HOME:-$HOME/.cache}/memory-recall/log.jsonl"
```

`"outcome":"injected"` is a served prompt. Every other outcome names which gate
it hit — but check the record is *yours*: if your prompt never reached the hook
(a wrong config path, a mid-session install, a slash command) nothing is
appended and `tail -1` hands back whichever prompt ran before. A plugin install that has never been configured writes no log at all —
creating the shared state directory is a mutation nobody asked for — and
records its refusals in `"${CLAUDE_CONFIG_DIR:-$HOME/.claude}"/plugins/data/memkit-memkit/trust.json`
instead. That path is derived, not a stable interface, but `cat` it and the
`"outcome"` is the answer.

One limit worth knowing before you read it: a `memkitConfig` that is **set but
wrong** records `trust:unconfigured`, the same value as never having been
configured. The wrapper refuses the path before the hook runs, so the hook never
learns a config was named. `--debug-config` separates the two — it names the
path and says it does not exist — which is why step 2 of the
[Quick start](#quick-start) reads the option back out of `settings.json`.

## The four commands

- **`memory-recall`** — the `UserPromptSubmit` hook, and the same retrieval on
  demand: `memory-recall --search "<terms>"`. A plugin install ships this as
  `memkit-recall`; see [Both names, once](#install-details).
- **`memory-integrity`** — the store's checker. Layout, ledgers, frontmatter,
  dead links, dangling wikilinks, and prose path citations. `--write`
  regenerates the search ledgers from frontmatter. Optional — retrieval needs
  none of it, and a plugin install does not ship it
  ([docs/STORE.md](docs/STORE.md#the-ledgers-and-whether-you-need-them)).
- **`memory-eval`** — a snapshot-gated retrieval eval. The cases are *your*
  data, supplied in config; memkit ships none.
- **`memkit`** — the dispatcher the setup and diagnosis subcommands hang off.
  A skeleton in this build: `memkit --help` lists `doctor` and `init`, says
  they have not landed, and names what to reach for meanwhile.

## Config

One JSON file, read by all of them. `--config PATH` where a tool takes one,
otherwise `$MEMKIT_CONFIG` — **except on a plugin install**, which never reads
that variable and takes the path from the two routes in
[Install (details)](#install-details) instead.

The minimum is four lines, and it is a complete working config:

```json
{ "schema": 1,
  "roots": { "notes": { "kind": "path", "path": "~/notes" } },
  "stores": [ { "id": "notes", "dir": ".", "live_root": "notes" } ] }
```

Per store only `id`, `dir` and `live_root` are required: `role` defaults to
`project`, `edit_root` to `live_root`, and there is no `cwd_gate` unless you
write one. `citations`, `search_cli` and `eval` are optional in the file.
Everything below is the shape a mature install grows into, not a starting
point.

```json
{
  "schema": 1,
  "roots": {
    "canonical": { "kind": "path", "path": "~/notes" },
    "checkout": { "kind": "git_toplevel", "fallback": "self" },
    "self": { "kind": "config_relative", "up": 2 }
  },
  "stores": [
    {
      "id": "project",
      "role": "project",
      "dir": "docs/memories",
      "live_root": "canonical",
      "edit_root": "checkout",
      "sub_indexes": ["search/domain/INDEX.md"],
      "cwd_gate": { "root": "canonical" }
    }
  ],
  "citations": {
    "roots": ["docs", "scripts"],
    "extra_suffixes": [".conf"],
    "blame_base": "origin/main"
  },
  "search_cli": "memory-recall --search",
  "eval": {
    "root": "checkout",
    "snapshot": "eval-expectations.json",
    "gating_slices": ["suite", "noinject"],
    "cases": { "suite": [], "noinject": [], "vocab": [] }
  }
}
```

- **`schema`** — a reader that meets a higher number fails rather than reading
  half a config. For the hook, "fails" means degrading to inert and recording
  why: it is fail-open and must never block a prompt.
- **`roots`** — named, each with a resolution `kind`. `path` is `~`-expanded
  when the config is *read*, so redirecting `HOME` redirects the whole tool.
  `git_toplevel` follows the checkout you are standing in. `config_relative`
  walks up from the config file, which is how the same file works inside a
  build sandbox with no `$HOME` and no `.git`. A root may declare an `env`
  override *by name*. The checker, the eval and `memory-recall --debug-config`
  resolve it; the hook never does, and `--debug-config` uses it for its display
  only — its exit code always describes the tree the hook will serve.
- **`role`** — `project` or `personal`, defaulting to `project`. A label: this
  build validates it and prints it in `--debug-config`, and nothing else reads
  it. It does not affect retrieval or ranking.
- **`stores`** — a list of N stores, **ordered**, and the order is a contract:
  retrieval interleaves hits across store directories in this order. Each
  store names two roots, because the tools want different trees:
  `live_root` is the tree the **hook serves** — the copy every session reads,
  whichever checkout it is standing in — while `edit_root` is the tree
  **`memory-integrity` verifies**, blames, and rewrites under `--write`.
  Retrieval wants the canonical copy; verifying a change means reading the tree
  that change is in, which is why `git_toplevel` is the useful setting for
  `edit_root` and why a check run from a worktree needs no redirection.
  `edit_root` defaults to `live_root`, which is the right answer whenever one
  tree is both. A `cwd_gate` restricts a store to sessions inside the named
  root, including that root's git worktrees.
- **`citations`** — which top-level trees a prose path may name, extra
  suffixes to treat as filenames, and the base ref a change is blamed against.
- **`search_cli`** — the command memkit prints when a pointer block truncates
  its matches, so an agent can run the rest of the search itself. The default
  is the one this channel ships: `memory-recall --search` for pip and nix,
  `memkit-recall --config <path> --search` for a plugin install. **On a plugin
  install the VALUE is ignored** — deliberately, because one config file is
  read by every channel, so a value written for a pip install would otherwise
  travel to a plugin one and name a binary that is either absent (the agent
  gets exit 127) or, on a machine carrying both, the *other* install, searching
  stores nobody pointed it at. `--debug-config` prints a `!` line when it has
  overridden the field — on a plugin install that is
  `memkit-recall --config <path> --debug-config`, since the override only
  happens on that channel and `memory-recall` is not the binary it ships.

  The **type check is not** ignored: a `search_cli` that is not a string is a
  `ConfigError` on every channel, which for the hook means degrading to inert
  and for the CLIs means exit 2. That is deliberate too — one file travels
  between channels, so a config that is broken for one of them is broken.
- **`interpreter`** — an absolute path to the python that runs the hook, and
  the plugin channel is where it matters. There the wrapper resolves an
  interpreter itself, preferring this value and falling back to whatever
  `python3` the launching shell's `PATH` gives it — which on a machine with
  direnv, mise or an activated venv is not a python you chose. The nix channel
  bakes its interpreter in and ignores this. It must be **absolute and
  canonical**: a value with `..`, `//` or `/./` in it, or under `/proc` or
  `/dev/fd`, names a different file depending on which directory the session
  stands in, so it is refused with a line on stderr and the `PATH` probe
  answers instead. `~` is expanded.
- **`eval.cases`** — three slices. `suite` pairs a prompt with the *basename*
  of the memory it is about; the tier is resolved at run time from where the
  file lives now, so promoting a memory from `search/` to `hot/` flips its
  assertion (`must be injected` becomes `must not be`) with no edit to the
  case. `noinject` prompts must inject nothing. `vocab` paraphrases suite
  cases in symptom words and is an instrument, not usually a gate.

`tests/fixtures/` holds a small working example of all of it: an invented
two-store corpus, a config, and the eval snapshot it produces.

### Derived state

The lexical index and its sidecars live under `$XDG_CACHE_HOME/memory-recall/`,
or `~/.cache/memory-recall/` where that variable is unset — the XDG default, and
what a mac gets. A relative `$XDG_CACHE_HOME` is ignored: the directory an
every-prompt hook writes into is not the session's to choose. Keyed
by a digest of the corpus root. Everything memkit writes here is disposable —
delete any of it and the next run rebuilds from the corpus.

Your **config** is not, and it may be living here: earlier versions of this page
put it at `~/.cache/memory-recall/memkit.json`, and the plugin offered that as
its default. Nothing regenerates it — `memkit init` is not in this build — so if
yours is under `~/.cache/`, move it somewhere a cache sweep will not reach and
re-run the install command with the new path.

- `fts5-<digest>.db` — the SQLite FTS5 index.
- `fts5-<digest>.root` — which corpus root that digest is for. Advisory; the
  engine never reads it. It exists because sha256 is one-way and "why was this
  memory not recalled" starts by finding the index that should have held it.
- `fts5-<digest>.build` — how the last index build went, as one JSON object:
  `{"v": 1, "ts": <unix seconds>, "outcome": "...", "files": <int|null>}`.

The `.build` record exists so that "never indexed" (no file) and "indexed, and
the corpus is empty" (`files: 0`) can be told apart without opening the index —
opening it syncs it, and a diagnostic that repairs what it measures cannot
report on it.

**Two rules for anything reading it.** `v` is bumped only when the record's
*shape* changes — a key added, removed or retyped — never for a new `outcome`
value. And **an `outcome` you do not recognise must be treated as not-OK**:
only `ok` licenses reading `files` as the size of the corpus. Under every other
outcome the count is a floor or absent (`null` when the run never got far
enough to count). Those two together are what let the vocabulary grow without
older readers mistaking a new failure state for a healthy one. Today it is
`ok`, `partial` (part of the corpus was unreadable), `busy` (another session
held the write lock, so nothing was counted), `unreadable` (the corpus could
not be read at all) and `rebuilt` (the index was damaged and built again).

One more file per session: `<session-uuid>.json`, holding the once-per-session
dedup ledger. It is disposable — deleting it lets that session offer a memory
again — and one is written per session, so a long-lived cache directory
accumulates them.

A sweep that collects these files must take the index, its `.root` and its
`.build` together: an orphaned `.build` outliving its index reads as a real
record of a corpus that is no longer there.

#### `log.jsonl` — the soak log, and what a reader may assume of it

One JSON object per line, appended by the hook and by the search CLI. Not on
every invocation: a plugin install that has not been configured refuses before
it would write anything, deliberately, because creating the shared state
directory is a mutation nobody asked for. It is read outside this repository —
the nix consumer's analyzers compute injection rates from it and its test suite
asserts that every outcome memkit can emit has been classified — so the growth
rule is a contract rather than an implementation note.

- **The `outcome` vocabulary grows without a version bump.** `v` is a hash of
  the hook's own bytes, not a schema version; a new outcome is a normal change
  and will arrive in one. A reader that partitions outcomes into populations
  must therefore fail loudly on one it does not recognise rather than dropping
  it from both halves — a silently unclassified outcome is a rate computed over
  a denominator nobody checked.
- **`"concludes": false` marks a record that is not a prompt outcome**, and it
  is the ONLY filter that isolates the per-prompt population. Two kinds carry
  it: duplicate-registration detection, which is about the machine and is
  written beside the record the prompt produces for itself, and every record
  the search CLI writes, since an agent running a command is not a prompt
  anyone typed. Exclude both from any per-prompt population.
- **Do not filter on `prompt_sha` or `ms`.** The CLI's records carry both — it
  hashes the query the same way — so a consumer keying on them pulls a
  command-line search into the denominator of every injection rate. Keying on
  the outcome's NAME instead means learning each new name as it arrives; the
  discriminator is there so you do not have to.
- **What a record may contain is bounded**: hashes, counts, basenames, and the
  sanitized query terms. Never raw prompt text, and never file contents.

## Exit codes

`memory-recall` exit codes — grep's three, plus two. `memkit-recall`, the name
a plugin install puts on the agent's PATH, shares this table:

| code | means | for the caller |
|---|---|---|
| 0 | pointers found, printed to stdout | read them |
| 1 | the stores were searched and nothing matched | there is no such memory |
| 2 | the search itself failed, wholly or in part — an unparseable config, a `--dir`/`--config` that is not there, some corpus that could not be opened, or arguments that make no sense | fix what stderr names; never read as absence |
| 3 | **inert**: nothing to search — no config, or no store on disk and in scope for this directory | stderr names which; this is *not* a claim of absence |
| 4 | the search never started — no plugin tree found, an incomplete payload, or no interpreter to run it with. Only the plugin's `memkit-recall` wrapper emits this; stderr names what is missing | nothing about the query or the config will change it. `memkit`'s own table below gives 4 a different meaning |

Why 4 rather than 2 for that last one: 2 says "what you asked for is wrong",
and all three of the states it names send a caller to fix its own request —
against, in this case, a machine that cannot run memkit at all.

`memory-recall --debug-config` reports the resolved config and shares those
codes: 3 when nothing is searchable, so it cannot come back green about an
installation `--search` calls inert. **What that shared code covers is config
and store resolution, not retrieval health** — this command never opens an
index, so a corrupt index and a healthy one both exit 0 here while `--search`
separates them. Read a green from it as "the config resolves and the stores are
where it says", never as "retrieval works".

It is the one `memory-recall` mode that resolves a root's `env` override — the
checker and the eval honour those too; the hook never does — and it resolves it
in the *display* only. The exit code is always taken from the tree the hook
will actually serve, since that is what the code is a claim about. Where an
override sends this command somewhere the hook will not look, it prints the
divergence per store: what did the resolving, what this run resolved, and what
the hook will read.

`memkit`'s codes are its own, because it is a different command with a
different job:

| code | means |
|---|---|
| 0 | the subcommand ran |
| 1 | memkit could not start at all — no interpreter, or an incomplete plugin payload. Only the plugin's `bin/memkit` wrapper emits this; stderr names what is missing |
| 2 | usage error, or a subcommand that does not exist |
| 4 | the subcommand exists and is not in this build — stderr names the fallback. **Not** the 4 in the table above: these are different commands and neither borrows the other's vocabulary |

## Install (details)

### Nix (flake)

```nix
{
  inputs.memkit.url = "github:ak2k/memkit";
}
```

The home-manager module installs the package and writes the hook entries into
Claude Code's hooks directory:

```nix
{
  imports = [ inputs.memkit.homeManagerModules.default ];

  programs.memkit = {
    enable = true;
    configFile = ./memkit.json;   # or set `roots` / `stores` / `citations`
  };
}
```

It writes two files into `~/.claude/hooks/` (configurable via `hooksDir`):
`memory-prompt-recall.py`, a wrapper that execs the packaged hook with the
config path **baked in**, and `common-words.txt` beside it. It also adds the
package to `home.packages`, because the search recipe the hook advertises to
the model has to actually be on `PATH`.

The config path is baked with a hard wrapper override, not a default. Which
directories an every-prompt hook reads and injects from is the whole
memory-poisoning surface of this design, and an ambient environment variable
would hand that decision to whatever repository the session happens to be
standing in.

### Claude Code plugin

```
/plugin marketplace add ak2k/memkit
/plugin install memkit@memkit
```

Installed this way, the hook is not registered in the session you are sitting
in — Claude Code reads hook registrations at session start. Start a new one.

or, from a shell, in one non-interactive command:

```
claude plugin marketplace add ak2k/memkit
claude plugin install memkit@memkit --yes \
  --config memkitConfig="$HOME/.config/memkit/memkit.json"
```

That registers the `UserPromptSubmit` hook and puts the plugin's `bin/` on the
agent's `PATH`. It reads nothing and says nothing until you give it a config —
see below.

**No GitHub credentials needed.** Both steps clone anonymously over HTTPS, so
neither a GitHub account nor SSH keys are required to install this.

#### What you are installing

**What the marketplace serves you.** The entry pins a released commit sha, so
`marketplace add` does not mean "whatever is on main". A release moves the pin,
and until it does, updating the marketplace changes nothing about the code in
your sessions. This matters more here than for most plugins: the payload is a
hook that runs before every prompt you type. What arrives is the whole tracked
tree at that sha — more than the hook needs, and worth knowing about before you
install a prompt hook: [docs/ADMISSION.md](docs/ADMISSION.md) says what is in
it and where the trust boundary sits.

The `uvx` fallback used for checker work on a machine with no Python 3.12
resolves `git+https://github.com/ak2k/memkit@v0.1.0` — the release tag, not
`main`. Nothing on the every-prompt path uses it; it is reached only by a
subcommand that regenerates a ledger.

**Check that it took.** `claude plugin details memkit@memkit` reports the hooks
it registered: `Hooks (1)` is a working install and `Hooks (0)` is not. Worth
running once, because a memkit that installed correctly and has not been given
a config is *also* silent by design (see below) — so from the outside it looks
exactly like one that installed nothing.

**What the installed copy is.** A release commit cannot name itself, so the pin
names an earlier commit, and what you install is the tree as it stood then —
including that tree's documentation. Its README still says memkit is not yet
installable from this marketplace (it is; this is the corrected copy), it
presents the tiered store layout as required when it is not, and its links to
`docs/STORE.md` and `docs/ADMISSION.md` dangle because neither file existed at
the pin.

Every difference between that copy and this page is a change made after the pin.
The ones that alter behaviour are the *(from the next release)* markers on this
page, so it does not enumerate them again in prose that would go stale on the
next merge. What is worth checking is the thing that does not move: that the
tree on your machine is exactly the commit the entry names.
[docs/ADMISSION.md](docs/ADMISSION.md) gives the two commands.

**Installing from a clone**, which is what you want while developing against
your own checkout: `claude plugin marketplace add <path to your checkout>`,
with the entry's `source` set to `"./"`. Both halves of that are what
`tests/rig/` does to exercise the real install path.

#### Configuring it

**Setting it up is manual in this build.** `/memkit:init` — the consented,
journalled setup this is designed around — has not landed yet, so for now write
the config by hand (schema and a worked example under [Config](#config)) at the
path you passed to `--config`. Until that file exists the plugin is **inert**:
the hook exits 0, prints nothing, reads no directory of yours, and records the
refusal in the plugin's own data directory, where a future `memkit doctor` can
report it. You can read it now: the directory is derived rather than secret —
`"${CLAUDE_CONFIG_DIR:-$HOME/.claude}"/plugins/data/memkit-memkit/trust.json` —
and [Why nothing appeared](#why-nothing-appeared) says what its `outcome`
means: `trust:unconfigured` for no config, `trust:config-error` for one that is
there and unreadable or unparseable.
It is not a stable interface. That is the intended state, not a failure — but
nothing will surface pointers until you write the config.

The config path reaches the hook through the `memkitConfig` option above, or
through `$CLAUDE_PLUGIN_DATA/memkit.json`, in that order — and through no other
route this build reads. Both are variables Claude Code sets, so the claim is
independence from *your shell's* `$MEMKIT_CONFIG`, not from anything that can
write Claude Code's environment; [docs/ADMISSION.md](docs/ADMISSION.md) states
the residual. Both are environment variables Claude Code exports into the hook process
(`CLAUDE_PLUGIN_OPTION_MEMKITCONFIG` and `CLAUDE_PLUGIN_DATA`), and both must
name an absolute path **after `~` expansion** — a leading `~/` is expanded and
nothing else is, which is why the plugin's own default value is a `~/` path. `$MEMKIT_CONFIG` is not among them: an every-prompt hook's list of
directories is not the ambient environment's decision to make, which is the
same rule the nix module follows by baking the path in. It is also never taken
from the plugin's own directory, for the sharper version of the same reason —
the payload is a clone of a pinned commit, so a config shipped inside it would
let the code you installed choose which of your directories the hook reads and
which binary runs it.

A `memkitConfig` path that is set but **wrong** — not there, or there and
unreadable — is named on stderr rather than quietly ignored, and the two are
reported apart because the remedies are. So `config: none` with nothing on
stderr means the option was never set; `config: none` with a line about it
means the path was.

#### Using it from a shell

**`bin/` is the agent's PATH, not your terminal's.** While the plugin is
enabled, `memkit-recall` is on the agent's Bash tool's `PATH`. It is not on your
own — nothing is added to your shell — so the same command typed into a terminal
will not be found.

**It needs `--config` there.** A Bash-tool process gets the plugin's `bin/` and
none of the plugin's environment (measured: `CLAUDE_PLUGIN_ROOT`,
`CLAUDE_PLUGIN_DATA` and every `CLAUDE_PLUGIN_OPTION_*` unset), and both config
routes above are environment variables — so a bare
`memkit-recall --search "…"` there resolves no config and answers `inert`,
exit 3. Run `memkit-recall --config <the path you passed to memkitConfig>
--search "…"` instead. That is the form the hook's own pointer block prints, so
following what memkit hands you is already right.

The plugin's directory is **appended**, so any name already on your `PATH`
wins (measured, not assumed). That is why the two names that matter are ones
nothing else ships: a second `memory-recall` from a pip or nix install would
have resolved first and searched the wrong stores without saying so, which is a
wrong answer wearing a right one's clothes. The exception is `memkit` itself,
which collides with this project's own console script — if you also have memkit
installed via pip or nix, a bare `memkit` in the agent's shell is that one.
Invoke the plugin's copy by path when it has to be that copy. **Not through
`$CLAUDE_PLUGIN_ROOT`** — that variable is set only inside processes the plugin
itself spawns, so in your terminal and in the agent's Bash tool it is empty and
the command expands to `/bin/memkit`. Build the path from the cache instead:

```
PLUGIN="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins/cache/memkit/memkit"
"$(ls -d "$PLUGIN"/*/bin/memkit | tail -1)" --help
```

**Debugging what Claude Code told the hook.** `tests/rig/hookdump.py` is a
standalone recorder: register it on any hook event in a scratch
`CLAUDE_CONFIG_DIR` and it writes one JSON file per invocation with the argv,
the environment, the cwd and the payload. It is how the claims in this section
were measured. Its records hold the WHOLE environment — `ANTHROPIC_API_KEY`
included — and the whole prompt, so keep them inside the scratch profile they
came from.

#### Leaving

**If you disable the plugin**, the hook stops and the store is untouched — it
lives outside every plugin-managed path by design. Retrieval still works
without Claude Code, and with no install:

```
uvx --from git+https://github.com/ak2k/memkit memory-recall \
  --search "<terms>" --config ~/.config/memkit/memkit.json
```

`claude plugin uninstall memkit` additionally removes the plugin's data
directory — which holds only the refusal records above — unless you pass
`--keep-data`. Your config lives wherever you put it — step 3 recommends
outside the cache — and is never touched. Your index and your soak log live in
`$XDG_CACHE_HOME/memory-recall/` — `~/.cache/memory-recall/` when that variable
is unset, which is the case on a mac — and your memories live wherever your
config says; none of that is removed.

### Plain Python

```
pip install git+https://github.com/ak2k/memkit
export MEMKIT_CONFIG=/path/to/memkit.json
```

**This installs the four commands and registers no hook.** Nothing on this
channel writes a `UserPromptSubmit` entry, so prompts are unaffected until you
add one yourself — the nix module and the plugin are the two channels that wire
it up. Use pip when you want the CLIs (searching a store by hand, running the
checker or the eval); pick the plugin channel if what you want is pointers in
your prompts.

### Where this runs, and what it needs

A Linux workstation is the ordinary case,
and the plugin channel is written for it: the hook, the wrappers and the rig
scenarios all run on Linux in CI, on the same `ubuntu-latest` an adopter's
machine resembles. macOS is supported and is where the FLOORS come from — a
stock mac ships Python 3.9.6 and a `/bin/sh` that is bash 3.2 in POSIX mode, so
the hook is stdlib-only and imports under **Python 3.9** and the wrappers are
POSIX `sh` with no bashisms. Every Linux distribution worth installing on
clears both, which is why those constraints are invisible there rather than
absent.

The hook takes whatever `python3` the `PATH` resolves to, so the floor is a
floor rather than a target. `memory-integrity` and `memory-eval` require
**3.12**, and the checker says so by name on an older interpreter rather than
dying on a syntax error.

Without a config, memkit is **inert**: no stores, zero pointers. That is a
deliberate default, not an oversight — there is no ambient search path to guess
at. The *hook* stays silent and exits 0 whatever happens, because a hook that
fails any other way blocks a prompt; the CLIs say which state they are in.

**Both names, once.** The commands below are spelled `memory-recall`, which is
what pip and nix install. A plugin install ships `memkit-recall` instead and no
`memory-recall` at all — the names differ on purpose, because plugin `bin/` is
appended to the agent's `PATH` and a second `memory-recall` would win the
collision and search another install's stores without saying so. Everything
that follows applies to both; substitute the name your channel ships. On a
plugin install, add `--config <the path you passed to --config memkitConfig>`:
the agent's Bash tool receives the plugin's `bin/` and none of the plugin's
environment, so that flag is the route that survives into it — which is why the
hook's own truncation notice now prints the command that way.

`memory-recall --search "<terms>" --dir <a directory of your own notes>` works
with no config at all — the caller named the corpus, so nothing has to be
configured for it to answer. One caveat worth knowing before you reach for it
as a smoke test: a `$MEMKIT_CONFIG` that is *set and broken* is refused even
here, because a config that cannot be parsed is somebody's mistake on any
branch and silently ignoring it is how a typo survives. Unset the variable or
fix the file first; an unset one is not an error.

Rolling this out across more than one machine, verifying a host afterwards, and
rolling it back: [docs/ROLLOUT.md](docs/ROLLOUT.md). Read it before the second
host — the hook fails open, so a broken rollout is silent. It carries a verify
block per channel; the nix one reads paths a plugin install does not have.

## Retrieval disclosures

Everything below was measured on **one single-author corpus of a few hundred
memories**. The mechanisms are general; the verdicts are not, and this section
exists so nobody adopts a negative result that was never tested against their
data.

**The retrieval stage is lexical only, and that is a measured choice rather
than a simplification.** An embeddings-based second stage shipped and was
removed in August 2026 after a three-arm experiment: under its shipped trigger
it never fired on a prompt with a real subject, most known-good targets turned
out to be lexically retrieved already and lost on *rank* rather than on
vocabulary, and stubbing it out left the eval bit-identical. The lever that
actually moved results was lexical rank and the pointer cap.

*Reopen if:* your corpus has genuine vocabulary mismatch — users describing
symptoms in words the documents never use — and you can show a measurable
number of targets that lexical retrieval never retrieves *at all*, as opposed
to retrieving and ranking below the cap. Measure at the injection surface (what
the hook would actually show), never against the raw candidate list; against
the candidate list the removed stage looked useful.

**Query expansion is dead here, three separate ways, for one reason.** Porter
stemming, a prefix-match rule, and vocabulary typo correction were each built,
measured against real prompts, and dropped. They fail by a single mechanism:
expansion harms retrieval when the expanded target is corpus-frequent, and no
downstream guard can see it. A term present in most chunks contributes almost
no score anywhere and matches almost everywhere, so it displaces one strong
specific hit with several weak diffuse ones — while every available guard
(relevance floors, co-occurrence windows, edit-distance budgets) tests the
*candidate*, not its breadth.

The last guard anyone had left, a document-frequency ceiling on expansion
targets, was measured and points the wrong way (AUC 0.33): harm is bimodal in
frequency, arriving both from mid-frequency discourse words that match broadly
and weakly, and from rare targets that dominate by inverse document frequency
precisely because they are rare. No frequency statistic separates the classes.

*Reopen if:* (1) you can demonstrate upside first — six configurations across
two surfaces produced none here; (2) you have a technical lexicon, because
correctly-typed domain vocabulary is invisible to English word frequency; and
(3) your proposal does not rest on a frequency statistic of the expansion
target, since corpus document frequency, English word frequency, and pointwise
mutual information were each measured dead or inverted.

**A relevance floor keyed on English word frequency does the work instead.** A
hit whose matched terms are all common English is treated as conversational
coincidence unless at least three terms matched *and* they are a real share of
the prompt. The wordlist behind that floor is a committed artifact
(`src/memkit/common-words.txt`), regenerated by `tools/generate-common-words.py`
against a pinned corpus-frequency dataset. It is a floor calibrated on one
corpus; the shape of the rule is likely to transfer, the exact thresholds are
not.

**What lands in the prompt, and what it costs.** The pointers arrive inside a
`<memkit-pointers>` block whose preamble tells the model the lines are DATA and
not instructions — paths and descriptions are file contents, and every one of
them is sanitized before it is rendered, so a memory cannot close the block or
smuggle control characters through it. The block is the frame plus one line per
pointer: **559 bytes fixed** on any prompt that fires, plus the pointer lines
themselves, which are as long as your descriptions. When the per-prompt cap cuts
matches, a truncation notice is added carrying the search command to see the
rest — measured at **another ~460 bytes**, since it quotes your config path and
the whole query. A prompt that retrieves nothing costs nothing: the hook writes
no block at all.

**Injection is pointers, never content.** Content injection is the recorded
context-pollution failure of comparable tools. A pointer costs tens of tokens
and lets the model decline.

## Development

```
pytest -q                                       # both suites
memory-eval --config tests/fixtures/memkit.json # the fixture retrieval gate
nix flake check                                 # everything, four platforms
```

`nix build .#memkit` builds the package. The version comes from git tags via
hatch-vcs; the Nix build passes it in explicitly, because a flake input
materialises without a `.git` directory, and a drift check asserts the built
tool reports what was pinned.

`src/memkit/common-words.txt` is a committed artifact, not a source file. It is
what `tools/generate-common-words.py` produces under the pinned wordfreq, and
CI regenerates and diffs it — so edit the generator, never the wordlist:

```
uv run tools/generate-common-words.py         # regenerate
uv run tools/check-wordlist-reproducible.py   # what CI asserts
```

CI runs the same things twice over, once per install story: a plain-python leg
(`uv venv` + editable install, then the suites, the fixture eval, ruff, two
pyright passes — the package at its 3.12 floor, then the 3.9 entry points — and
the plugin manifests through `claude plugin validate --strict` at a pinned
Claude Code) and a nix leg (`nix flake check` on x86_64-linux, aarch64-linux and
aarch64-darwin). Neither trigger is path-filtered, so a docs-only change still
reports every context. Two more workflows report nothing a merge waits for:
`remote-install.yml` installs from github daily (and on any change to
`.claude-plugin/`), and `live.yml` runs the model-backed tier by hand.

`tests/rig/` drives the real `claude` binary against a scratch profile, because
what the plugin claims — that a manifest option reaches a hook's environment,
that an installed wrapper emits pointers — are claims about Claude Code, which
this repo does not own, and every one of them fails silently. Four tiers, and
each names its own dependency: the CLI and harness tiers need only the binary
and run in CI, the live tier needs a model, and the remote tier needs the
network.

```
pytest tests/rig                          # CLI + harness tiers
MEMKIT_RIG_LIVE=1 pytest tests/rig        # + the scenarios that need a model
MEMKIT_RIG_REMOTE=1 pytest tests/rig      # + the install from github itself
```

The remote tier is the only one that leaves the machine, which is why it is
opt-in and why it runs on a daily workflow rather than on a pull request. Every
other tier stages the working tree as a marketplace serving itself in place,
so none of them can see the TRANSPORT — and a marketplace entry that could
only be cloned with SSH keys is how v0.1.0 shipped an install that failed for
everyone while every check was green. It installs from `ak2k/memkit` at the sha
main's manifest pins, with no credential of any kind.

The live tier expects an Anthropic-compatible endpoint in `ANTHROPIC_BASE_URL`
(it never touches your real credentials or your real `~/.claude` — every
scenario asserts its config dir is scratch before running anything that
writes).

**Which pyright config a new file belongs in.** `pyrightconfig.json` includes
`src/`, `tests/` and `tools/` by directory, so a new file is covered there with
no edit. `pyrightconfig-hook39.json` is an explicit file list, and it must name
every file a **3.9 interpreter can execute**. That is two entry points: the
recall hook, which Claude Code runs with whatever `python3` the `PATH`
resolves to, and `memkit.cli`, because the plugin's `bin/memkit` runs the
dispatcher on that same interpreter — only checker-backed work routes to 3.12,
and sending the whole dispatcher there would put `memkit doctor` out of reach
on any machine whose `python3` is older than 3.12 — a stock mac is the case
that forced it, and it is the machine that most needs to ask whether its
install works. Add a module either entry point imports and it belongs in that
list on the same commit; its 3.9 floor is otherwise unchecked, and the failure
surfaces as a hook that silently retrieves nothing. The direction is easy to
invert: a module that merely *imports* one of those two does not belong there,
since nothing puts it in front of the 3.9 interpreter.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
