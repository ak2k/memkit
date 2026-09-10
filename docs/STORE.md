# Your store

A store is a directory of markdown files. memkit reads it; nothing writes to it
but you and your agent. Everything below was run against the shipped code —
where a claim says "surfaces", a test in this repository drives the hook and
checks it.

## What retrieval actually requires

Almost nothing. Point a config at a directory of `.md` files and you get
pointers. In particular you do **not** need ledgers, a `hot/` directory, or
frontmatter to be retrievable.

- **Which directory is searched.** `<store>/search` when that directory
  exists, otherwise the store directory itself. So a flat folder of notes
  works.

  **Start in `search/`, or migrate in one step.** The moment `search/` exists
  it becomes the corpus root, and every file still above it stops being
  retrieved — the files are untouched on disk, `--search` still answers for
  whatever moved, and nothing else changes.

  This is easiest to trip without meaning to: the agent block at the bottom of
  this page writes memories to `<store>/search/`, so on a flat store the first
  memory your agent writes takes every earlier one out of retrieval. The
  [Quick start](../README.md#quick-start) therefore creates `search/` from the
  first file, and so should you. `/memkit:init` starts you there — it lays the
  store out with `search/` and `hot/` before it writes anything into either, so
  the layout never has to change — and it refuses to adopt a flat store that
  already holds memories, naming the one-step migration instead of performing
  it silently. If you already have a flat store, move all of it in one step.

  Both diagnostics catch the stranded state: `--debug-config` names the files,
  and on a pip or nix install the checker reports `STRAY-ROOT: ./<file>` for a
  memory above `search/`.
  `--debug-config` prints the corpus root, its file count, and a line naming
  any files stranded outside it:

  ```
  store notes: ~/notes [project; always; searched]
    corpus:  ~/notes/search — 1 file
    ! 2 markdown files under ~/notes are outside the corpus root and
      will not be retrieved — move them into search/
  ```
- **What is skipped.** `archive/` and `hot/` are pruned while walking —
  `hot/` because those memories are meant to be in the session's context
  already, so pointing at them again costs tokens and adds nothing.

  **Nothing in a plugin install puts them there.** memkit registers one
  `UserPromptSubmit` hook and loads no files; auto-loading `hot/` is something
  you wire up, with one line in your `CLAUDE.md`:

  ```markdown
  @~/notes/MEMORY.md
  ```

  Until you do, a file under `hot/` is reachable by neither route — not
  retrieved, not loaded. **If you are not wiring that up, keep everything in
  `search/`.** `MEMORY.md`,
  `SEARCH.md` and `INDEX.md` never surface as pointers anywhere.
- **What is indexed.** The whole file, split into chunks at markdown headings,
  so one section of a long memory competes on its own length rather than the
  file's. The span before the first heading is its own chunk — that is where
  the frontmatter sits.

## A memory file

```markdown
---
name: postgres-connection-pool
description: PgBouncer in transaction mode breaks session-scoped features — prepared statements, advisory locks, and SET LOCAL do not survive.
type: reference
---

# PgBouncer transaction mode

Transaction pooling hands a different backend to every transaction, so
anything the client thinks is session state is gone by the next statement.

## What breaks

- Prepared statements: the protocol-level ones. Use `prepare_threshold=0`.
- Advisory locks taken with `pg_advisory_lock` — they outlive the transaction
  and so leak onto a random backend.
```

Dropped into a store and asked *"why do prepared statements break under
pgbouncer transaction pooling"*, that file surfaces as:

```
- ~/notes/search/postgres-connection-pool.md — PgBouncer in transaction mode breaks session-scoped features — prepared statements, advisory locks, and SET LOCAL do not survive. [matches 5/7 prompt terms: prepared, statements, pgbouncer, transaction, pooling] [section: PgBouncer transaction mode]
```

### The fields that change behaviour

**`description:` is the line the agent reads.** It is the whole of what a
pointer shows about the file, so write it as the sentence that decides whether
to open the file — a claim, not a title. "PgBouncer transaction mode" tells the
agent nothing it did not get from the filename.

It is also indexed like the rest of the file, so the words in it are words the
memory can be found by. That is a side effect worth using, not a substitute for
a body: matching runs over the whole file.

If there is no `description:`, the first `# heading` is used instead. If there
is neither, the pointer renders as the path and the matched terms alone — still
retrievable, just mute about itself.

**Write descriptions under 155 characters.** Three numbers sit behind that one:
the checker rejects a description over **155**, the hook renders at most **157**
before adding `...`, and its hard ceiling is 160. The ladder is deliberate —
the checker's cap is below the hook's cut so an authored description is never
truncated — and 155 is the only one you need. The long form belongs in the
body.

**`type:` has one behaviour, `type: feedback`.** Those memories must clear a
stricter relevance bar before they surface, because a standing instruction that
appears on a loosely related prompt is worse than one that stays quiet. Every
other value — `reference`, `project`, whatever you invent — behaves the same.

**`name:`** is used by the checker, not by retrieval.

### How a pointer gets chosen

A file must match at least one of the prompt's terms — term evidence is
required, and the pointer prints which terms matched so you can see why it was
offered. A match on a distinctive word is enough; matches only on common
English words have to clear a count and a share of the prompt.

Two caps bound the cost: at most **3 pointers per prompt**, and at most **30
per session**, after which a new pointer has to displace the weakest one
already spent. When the cap cuts something, the block says so and prints the
search command to see the rest.

**The hook applies gates `--search` does not**, which is why the two can
honestly disagree about the same words: a prompt under three words, a prompt
identical to one already served this session, and a prompt that opens with an
editor or tool envelope all return nothing from the hook and answer normally
from the CLI. The full list, in the order a prompt meets them, is
[Why nothing appeared](../README.md#why-nothing-appeared).

## One store or two

Most people want one: a personal store in `~/notes`, always searched, private to
you. Start there.

A second store earns its place when the memories belong to a **project** rather
than to you — things a teammate cloning the repository should get, and that you
do not want surfacing while you work on something else. Keep it inside that
checkout and gate it:

```json
{ "id": "project", "role": "project", "dir": "docs/memories",
  "live_root": "canonical", "cwd_gate": { "root": "canonical" } }
```

`cwd_gate` is what makes it a project store: the store is searched only from
sessions inside that root, including its git worktrees. Without it, a project's
memories follow you into every unrelated session.

Two things about the list itself. It is **ordered**, and the order is a
contract — retrieval interleaves hits across stores in the order you write them,
so the store you put first is the one that wins a tie. And `role` is a label:
this build validates it and prints it in `--debug-config`, and nothing reads it.
The behaviour comes from `cwd_gate` and the ordering, not from the word.

## Git is the management layer

memkit does not manage your memories. Retrieval never runs git — it walks the
directory — so a store does not have to be a repository at all.

Make it one anyway. The store is prose your agent will act on, which makes the
useful questions historical ones: when did this become true, who changed it,
what did it say before. `git log` answers those and memkit has no reason to
reimplement it, and `git pull` is how a store shared between machines or people
stays shared.

**Sharing a store with other people is a trust decision.** Every description in
it is rendered into your prompts, so anyone who can push to that repository can
put text in front of your agent. memkit's frame says the block is data and
sanitizes every line, and its delimiter carries a random per-run suffix a
description written earlier cannot spell — so a description cannot close the
frame or smuggle control characters — but it cannot make the content true, and a plausible wrong memory
is the thing it does not defend against. Review pulls into a shared store the
way you would review code.

The checker is the one part that uses git, and only when the store is inside a
repository: it dates memories against a base ref to find citations that have
gone stale. Outside a repository it says so and skips that pass.

## Keep your store in git

The concrete form of that: **make the store a private git repository and point
`memkitConfig` at it.** Nothing else is required. memkit reads a directory, the
config names it, and no part of retrieval asks how the directory got there or
whether anything tracks it.

**Private, and the word carries weight.** The paragraph above is about text
arriving *from* a repository; this is the other direction. What makes a memory
worth keeping is that it is specific — this codebase's trap, what that incident
turned out to be, the constraint a customer will not move on — and specific is
the same property as disclosing. Pushing the store publishes it to wherever the
remote is, so the default should be a repository whose readers you chose.

### What a repository turns on

The stale-citation pass is not the only part that needs history. Two more
findings are dated ones, and both are inert without it:

- **A hot memory edited after the row that describes it.** The file's date is
  the newest commit touching it; the row's is the newest commit whose diff to
  `MEMORY.md` touched that row. The finding is the comparison of the two, so it
  needs both — a file no commit has touched dates to zero and is skipped. Outside
  a repository this does not warn; it has nothing to say.
- **A row a rewrite took away.** `SEARCH.md` is generated, so a memory that
  stops producing a row loses its only pointer silently. Whether that happened is
  decided against the copy of the ledger at the blame base, read out of git. With
  no such ref the answer is withheld rather than guessed, because "the ledger had
  no rows then" and "every row was lost" are the same observation.

Retrieval itself is unchanged — a plain directory answers prompts exactly as a
repository does. What the repository adds is something for the hygiene machinery
to date against.

### Where your agent's own memories land

Claude Code keeps a memory of its own, and by default none of it reaches your
store. Measured on 2.1.238, the version CI installs: it writes agent-curated
memories to `<config dir>/projects/<project key>/memory/`. Measured on 2.1.258:
it creates that directory at startup even when it writes nothing into it, and
the key, matching the documentation page, is the git repository root with every
character that is not a letter or digit replaced by `-`, so every subdirectory
of one repository shares one directory; outside a repository the cwd is used
instead. The path is the physical one, so a checkout reached through a symlink
keys on the symlink's target. A linked worktree maps to its main checkout's
root, so a repository's worktrees share that directory too; a submodule keys on
itself. Measured on 2.1.258: the config dir is `$CLAUDE_CONFIG_DIR` when that
is set. Read from the code on 2.1.258 and not exercised: it is `~/.claude`
otherwise, and a key past 200 characters is truncated there and given a base36
hash suffix.

```bash
# needs git 2.31+; a "fatal:" here means the key fell back to the cwd
# ASCII, newline-free paths only: tr maps bytes, the harness maps characters
# `|| root=` keeps the fallback reachable where the paste runs under `set -e`
root=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || root=
case $root in */.git) root=${root%/.git} ;;
  *) root=$(git rev-parse --path-format=absolute --show-toplevel || pwd -P) ;; esac
key=$(printf '%s\n' "$root" | tr -c 'A-Za-z0-9\n' '-')
printf '%s\n' "$key"
```

The common dir is the main checkout's `.git` from a linked worktree, which is
why it is read first; in a submodule it sits under the superproject, and the
`case` falls through to the submodule's own root.

Left there they are outside every store: nothing retrieves them and nothing
curates them. Point the harness at the store instead. The setting is
`autoMemoryDirectory`, and the value worth giving it is a directory of the
harness's own **under the corpus root**, so that what it writes is retrievable
the moment it lands:

```json
{ "autoMemoryDirectory": "~/notes/search/auto-memory" }
```

Retrieval walks the corpus recursively, so a subdirectory of `search/` is read
like any other. Give the harness one of its own rather than the corpus root:
measured on 2.1.258, a Write or Edit of a `.md` file that already carries
frontmatter, whose path starts with the configured directory, rewrites that
frontmatter — `name` slugified, a session id and a timestamp appended — so a
directory that also holds your own memory files gets them rewritten too. Read
from the code, not exercised: every other top-level key is moved under
`metadata:` at the same time. Measured on 2.1.258: a leading `~/` is expanded,
and the value is used as it stands for every project, with no per-project
directory under it.

In your user settings — `<config dir>/settings.json` — that sends every
project's memories to your personal store, which is the one to set once and
forget about. In a checkout's
`.claude/settings.local.json` it sends that project's memories to that project's
store. So does its **checked-in `.claude/settings.json`**: measured on 2.1.258,
`claude -p` reads `autoMemoryDirectory` from that file with no trust check. Read
from the code, not exercised: in an interactive session the checked-in value is
gated on folder trust. A clone's checked-in settings can redirect where your
agent writes. The settings schema's own description says that file is ignored,
and is wrong. Read from the code, not exercised: the precedence among settings
scopes, highest first, is managed policy, the `--settings` flag,
`.claude/settings.local.json`, `.claude/settings.json`, then user settings.

A symlink does the same job where the setting is not yours to set and no
settings file of yours outranks the one that carries it. The shape is one link
per project: the harness's own memory directory for this repository —
`<config dir>/projects/<key>/memory`, for the key the block above prints —
becomes a symlink into the store. Retrieval then
reads what the harness writes, and the harness goes on writing to the path it
already knows. Point the link at a directory of the harness's own, for the
reason the setting has one: a corpus directory that also holds your own memory
files gets them rewritten. Where a checkout's checked-in
`.claude/settings.json` declares `autoMemoryDirectory` already, the flag below
refuses (`auto-memory-redirected`) and this link reaches nothing: the harness
writes where that file sends it, not to the directory the key names. The route
left there is that checkout's own `.claude/settings.local.json`, which the
harness reads above it. Under managed policy the value is not yours to
override.

**What the shape costs.** The link is per project, so the next repository needs
its own. It is tied to the physical path the key derives from, so a checkout
that moves keys somewhere else and needs a new link. And whatever is already
written has to move into the store before the swap or it is orphaned, and it
lands there as new files: they carry no ledger row until the checker's
`--write` pass adds one.

**`/memkit:init --adopt-auto-memory` does all of that for every project at
once.** It copies what the harness has written into the store and redirects the
harness there, listing every path in one manifest you approve before anything
is written. That is the route this page recommends, and the reason it no longer
prints a chain of shell to do the same work by hand. It writes the setting into
your user settings, the bottom of the precedence list above, so where a checkout
carries a checked-in `.claude/settings.json` that sets it too, set
`.claude/settings.local.json` in that checkout instead: untracked, and above
both. Its refusal is order-dependent, too — it reads the cwd when it runs, so a
clone made afterwards is checked by nothing.

**One case still wants a hand.** An earlier revision of this page pointed that
directory at the corpus root itself. With `$key` still set by the block above,
this names the harness's memory directory for this repository and shows what
is there:

`dir=${CLAUDE_CONFIG_DIR:-$HOME/.claude}/projects/$key/memory; ls -ld "$dir"`

Where `ls -ld "$dir"` shows a symlink, repoint it at the harness's own
directory. Quit the harness first — it recreates `$dir` at startup, and a
recreation between `rm` and `ln` leaves the link inside `$dir` rather than in
its place.

`[ -L "$dir" ] && [ -d "$store" ] && mkdir -p "$target" && rm "$dir" && ln -sn "$target" "$dir"`

That race ends at rc 0 all the same, and `ls -ld "$dir"` then shows a directory
where it showed a link. Where it shows a directory holding one link named for
`$target`, the harness recreated `$dir` and the link landed one level down
inside it. Quit the harness and move that link up into `$dir`'s own place: it
already points where the line above was taking it. This page prints no command
for the move — `ls -ld` prints one line about a directory and never its
contents, so what else is in there is yours to read first.

`$target` is the harness's directory under the corpus root:
`$store/search/auto-memory` where `search/` exists, `$store/auto-memory` where
it does not — a store with no `search/` has `$store` itself for a corpus root.
Creating `search/` afterwards takes that directory back out of retrieval, so
making `search/` first is the simpler order. `rm` removes the link and never
what it points at, so memories already lying flat in the corpus root stay
where they are and stay retrievable. Where `$dir` is not a link the first test
fails and nothing after it runs. `$store` is your store's root, and all three
are yours to set before the line runs: an unset `$store` or `$dir` fails a
test rather than a command, so the line stops with a status and nothing on
stderr, while an unset `$target` fails `mkdir`, which does say so on stderr.
Under `set -u` all three are the shell's own message instead where the line
reaches that name, and it stops there rather than on a test. An earlier test
that fails first never expands the later name, so that line stops exactly as it
does without the option: a status, and nothing to read.

`memkit doctor` reads `autoMemoryDirectory` from the settings scopes the harness
honours and reports which file declares it — by the role that file plays, never
by its path — and how the directory it names stands to a corpus root: inside one,
so what the harness writes there is retrieved, or in one of the placements that
keeps it out of retrieval, from a name retrieval prunes to outside every store.
Could not look is a third answer and not a quieter version of the second: where
a store this run had to read would not resolve, the row places the directory
against nothing and sends you to that store rather than to the directory. Where
the key is unset it says the harness writes to the directory it derives from the
git root, under its own config directory, and counts the memories already
outside every store — or, where a store would not resolve, the memories nothing
was compared with. It renders no path on any branch.

### Before you wire it up

- **The default path is derived from the repository root**, so a symlink into
  it is tied to one checkout path. Clone the project to `~/work/app` on one
  machine and `~/src/app` on another, and only the machine whose path you linked
  is wired up. `autoMemoryDirectory` carries no such coupling, which is the
  better reason to prefer it.
- **The layout rule does not relax for a repository.** The harness writes into
  one directory — `MEMORY.md` and one file per memory, no `search/`. Point it at
  the store root
  and every one of those files sits above the corpus root and is not retrieved:
  [the same trap as any other file left there](#what-retrieval-actually-requires),
  now arriving on its own. Pointing at `<store>/search/auto-memory` avoids it,
  and the `MEMORY.md` that lands there alongside them is ignored by retrieval
  and by the checker alike, wherever it sits.
- **The lighter alternative**, for a repository that already keeps its memories
  in its own tree: leave the harness where it is and put a stub `MEMORY.md` in its
  directory naming the real store.

  ```markdown
  # Project memory

  This project's memories live in `~/src/app/docs/memories` and are indexed by
  the `MEMORY.md` there. Read that one.
  ```

  Writes still land outside the store, but the agent reads its way in — curation
  without relocating anything.

## Writing and retiring

**New memory.** Write the file. That is the whole of it — the next prompt
searches an index rebuilt from the directory.

**Retire one.** `git mv` it into `archive/`. It stops being retrievable and
stays readable, which is what you want for something that was true once.

**The index is disposable.** memkit keeps a SQLite FTS index beside your cache
so it does not re-read the corpus on every prompt. Delete it and it rebuilds.
The markdown is the source of truth; the index never is.

## The ledgers, and whether you need them

`MEMORY.md` and `SEARCH.md` at the store root are for **readers**, not for
retrieval — a hand-curated index of the memories worth loading every session,
and a generated index of everything else. Retrieval ignores both.

They exist for the layout the checker enforces, and the checker is optional.
Two things about it are worth knowing before you reach for it:

- **It does not bootstrap a store.** On a directory with none of this it
  reports the first thing missing and stops. To get a clean run you need
  `MEMORY.md` and `SEARCH.md` to already exist — empty is fine — and both
  `hot/` and `search/` present. `--write` fills SEARCH.md's rows from your
  frontmatter; it does not create the files.
- **A plugin install does not ship it.** The plugin's `bin/` carries
  `memkit`, `memkit-hook` and `memkit-recall` and no checker. If you want one,
  run it out of band:

  ```
  uvx --from git+https://github.com/ak2k/memkit@v0.4.0 memory-integrity --config <your config>
  ```

If you are not maintaining a curated hot tier, skipping all of it is a
reasonable choice. Retrieval does not care.

## Letting your agent write the memories

The point of a memory store is that it accumulates without you sitting down to
write it. This is a **suggestion**, not something memkit installs or enforces:
paste something like it into your own `CLAUDE.md` and edit it to taste.

```markdown
## Memory

When we settle something worth not re-deriving — a root cause, a decision and
why, a trap in this codebase — write it to `~/notes/search/<slug>.md`:

---
name: <slug>
description: <one sentence, under 155 characters, that would make me open this file>
type: reference
---

then the finding, and how it was established.

Write the memory when the thing is settled, not when it is still a hypothesis.
One claim per file.

Editing a memory in place is right when you are sharpening or correcting the
SAME claim — a number that was wrong, a cause you now understand better. Check
the `description:` still describes what the file now says, since that line is
what retrieval matches on.

Supersession is the other case, and it is a new file: when the claim itself has
been replaced, write the new memory and `git mv` the old one into
`~/notes/archive/`. Do not rewrite a memory into a record of its own history.

To check a memory can be found: `memkit-recall --config <your config> --search "<terms>"`.
```

The only memkit command in that block is the search, and it is the one a plugin
install puts on the agent's `PATH`. Writing a memory needs no memkit command at
all — it is a file.
