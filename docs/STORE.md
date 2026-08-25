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
  first file, and so should you. If you already have a flat store, move all of
  it in one step.

  The checker catches the stranded state today — it reports
  `STRAY-ROOT: ./<file>` for a memory above `search/` — and `--debug-config`
  will name them too from the next release.
  From the next release, `--debug-config` prints the corpus root, its file
  count, and a line naming any files stranded outside it:

  ```
  store notes: /home/you/notes [project; always; searched]
    corpus:  /home/you/notes/search — 1 file
    ! 2 markdown files under /home/you/notes are outside the corpus root and
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
sanitizes every line — a description cannot close the frame or smuggle control
characters — but it cannot make the content true, and a plausible wrong memory
is the thing it does not defend against. Review pulls into a shared store the
way you would review code.

The checker is the one part that uses git, and only when the store is inside a
repository: it dates memories against a base ref to find citations that have
gone stale. Outside a repository it says so and skips that pass.

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
  uvx --from git+https://github.com/ak2k/memkit@v0.1.0 memory-integrity --config <your config>
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
One claim per file. Do not edit a memory to record that it changed — write the
new one and `git mv` the old into `~/notes/archive/`.

To check a memory can be found: `memkit-recall --config <your config> --search "<terms>"`.
```

The only memkit command in that block is the search, and it is the one a plugin
install puts on the agent's `PATH`. Writing a memory needs no memkit command at
all — it is a file.
