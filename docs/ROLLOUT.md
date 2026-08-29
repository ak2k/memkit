# Rollout and rollback

memkit is not an ordinary dependency of the repository that consumes it. Its
hook runs on **every prompt, on every host**, and it is fail-open by contract:
when it cannot run it prints nothing and exits 0. A rollout that breaks it
therefore produces no error, no failed unit and no red check — only pointers
that stop arriving, which nobody notices until they go looking for a memory
that used to surface. Everything below exists because that failure is silent.

Written from one real rollout — a Nix flake consumer with a mixed darwin and
NixOS fleet — and amended after the first live darwin conversion and rollback
drill, which found the layout defect below. Host names and aliases here are
placeholders (`$CONSUMER` is the consumer checkout); the shapes are what
transfers.

**If you installed through the Claude Code plugin**, the section you want is
[Per-host verify, plugin channel](#per-host-verify-plugin-channel), and it is
the only one here that applies to you. Everything before it — the unsafe
window, the layout conversion, the nix per-host checks — is about a fleet
managed by the home-manager module, and its commands read paths a plugin
install does not have.

## The one unsafe window

Claude Code invokes the hook by path — `~/.claude/hooks/memory-prompt-recall.py`
in a `settings.json` it reads, not something memkit controls. Before
cutover the consumer carries that file in-tree; after cutover memkit's
home-manager module writes it. So there is exactly one interval where the path
names nothing:

    git pull            # in-tree copy gone, module not yet activated
    <no rebuild yet>    # <-- hook dangling: fail-open, zero pointers, no error
    darwin-rebuild switch

**Rule: on any host where the checkout is pulled by hand, pull and rebuild in
one command.** Not two commands in sequence — one, so an interrupted rollout
cannot leave a host parked in the window:

    cd $CONSUMER && git pull && <rebuild command>

NixOS hosts never see this window: their deploy aliases fetch, hard-reset to
`origin/main` and `nixos-rebuild switch` inside a single remote invocation, and
unattended auto-upgrade does the same. Pull and switch are already one step
there. They are **not** exempt from the layout defect below.

## The layout conversion defect

Reproduced deterministically — twice on fresh generations and once on a stale
one. It applies to exactly one transition: a consumer converting
`~/.claude/hooks` from a single whole-directory symlink into the checkout
(`mkOutOfStoreSymlink`) into per-file entries beneath that same path. That is
the cutover, and only the cutover.

home-manager's orphan cleanup does not remove the old directory symlink.
`linkGeneration` then writes the new per-file entries *through* it, into the
symlink's target — the consumer's git working tree. What lands there:

- tracked hook files replaced by symlinks that resolve back through
  `~/.claude/hooks` to themselves — **resolution loops**, `ELOOP`. On the
  measured host, four of six hooks were dead this way.
- `*.backup` copies of the originals and other untracked debris, in the
  checkout.

**Activation reports success throughout.** Nothing fails and nothing is logged,
and a hook that resolves looks exactly like one that loops until you try to
resolve it. Worth knowing which ones survive: memkit's two files are *new* at
cutover, so they can point at real content and answer correctly while the hooks
that were already tracked are dead. A memkit-focused verify therefore passes on
a host whose Claude Code hooks are broken — which is why check 1 tests every entry
rather than the two this repository owns.

### Converting a host

Remove the directory symlink *before* the switch:

    cd $CONSUMER && git pull && rm ~/.claude/hooks && <rebuild command>

With nothing at the path there is nothing to write through: home-manager
creates a real directory and populates it. This widens the dangling-hook gap to
cover the rebuild rather than just the pull, which is an acceptable trade — the
hook is fail-open, and it is one host at a time.

A NixOS host's deploy is one remote command and sees no pull-without-rebuild
window, but the defect is identical for the user's own checkout there. Run the
same `rm ~/.claude/hooks` over ssh before the deploy.

### Repairing a host converted in the wrong order

Idempotent from then on:

    readlink ~/.claude/hooks              # the directory that got written to
    git -C $CONSUMER checkout -- <that directory>/
    # then delete the untracked debris left in it: the two memkit files
    # (memory-prompt-recall.py, common-words.txt), every *.backup, __pycache__
    rm ~/.claude/hooks
    <rebuild command>                     # or re-run this generation's activate

Read `readlink` before anything else — once the symlink is gone you have lost
the pointer to the directory that needs restoring.

## Rollout order

1. **Pre-flight.** `git -C $CONSUMER status --short` is clean on every host you
   are about to touch. A dirty consumer checkout is not cosmetic here: the
   conversion defect deposits build output into the working tree, and you want
   to be able to tell that debris apart from your own edits.
2. **Merge** the cutover PR (first adoption) or the input-pin bump PR
   (every rollout after that).
3. **First host: one darwin machine, and only one.** At the cutover, use the
   conversion command — `git pull && rm ~/.claude/hooks && <rebuild>` as a
   single command; for a later bump, `git pull && <rebuild>` is enough. Verify
   it (below). **Then drill the cutover rollback on this host, before any
   second host rebuilds** — a rollback path that has only been read is not a
   rollback path. Roll forward again (with the `rm` again), verify again.
4. **Remaining darwin hosts**, same command, same verification.
5. **NixOS hosts** via their deploy aliases, or by letting unattended
   auto-upgrade pick the change up on its own schedule — preceded at the
   cutover by `rm ~/.claude/hooks` over ssh.

Steps 3 and 4 are separated for the first adoption because the cutover is the
only change that converts the hooks directory's *layout*, and that conversion
is the one that needs the `rm`. Routine input-pin bumps do not touch the
layout; for those, step 3's drill is optional and step 4 can be a sweep.

## Per-host verify

Four checks. The middle two prove the tool is installed and answers; the first
and last are the ones that catch the conversion defect.

**These four are nix-channel checks.** They read `~/.claude/hooks`, a
`/nix/store` symlink and a consumer checkout, none of which a plugin install
has — so on a host that installed memkit as a Claude Code plugin every one of
them either fails or passes vacuously. Use [the plugin-channel
block](#per-host-verify-plugin-channel) below instead; the two are alternatives,
not stages.

**1. The hooks path is a real directory, and every entry resolves.**

    test -L ~/.claude/hooks && echo "FAIL: still a symlink"
    for f in ~/.claude/hooks/*; do readlink -f "$f" >/dev/null || echo "LOOP: $f"; done
    ls -l ~/.claude/hooks/

`test -L` must **fail**. A symlink still at that path means the conversion did
not complete, and the per-file entries went into the consumer checkout.

`readlink -f` exiting non-zero on an entry means a resolution loop — the
write-through signature. Test every entry, not just memkit's two: those two are
new at cutover and can answer correctly while the pre-existing hooks are dead.

`ls` should then show `memory-prompt-recall.py` and `common-words.txt` as
symlinks into `/nix/store`, beside whatever other per-file hook entries your
consumer keeps.

**2. On-demand search returns hits.**

    cd $CONSUMER && memory-recall --search "<a term you know is in the store>"

Run it **from inside the consumer checkout**, not from `$HOME`. A store with a
`cwd_gate` is searched only from inside the named root (its git worktrees
count), so the same command run from elsewhere does not search it.

**Use `--debug-config` as the check, not the exit code of a search.** What a
gated-out store costs you depends on the rest of the config: if EVERY store is
gated out the run is inert — exit **3**, with the per-store reason on stderr —
but the reference config has an ungated personal store, so the gated one simply
drops out and a search from the wrong directory returns an ordinary exit **1**
that looks exactly like a term with no matches.

    cd $CONSUMER && memory-recall --debug-config

names the config that resolved and, per store, whether it is `searched`, `NOT
on disk` (nobody created it on this host) or `NOT searched here` (gated to a
tree you are standing outside of). That distinction is the thing worth
verifying, and it is the only surface that shows it per store. Note what it
does **not** cover: it never opens an index, so it says nothing about whether
retrieval would return anything — step 3 is what establishes that.

For the search itself: exit **1** means the stores really were opened and
nothing matched; **2** is a failure to search at all; **3** is nothing to
search. Pick a term you know matches a `search/`-tier memory: `hot/` memories
are excluded from retrieval by design, because they are in context already.

**3. The hook injects pointers.** Without starting a session:

    echo '{"session_id":"verify-'$(date +%s)'","transcript_path":"/dev/null",
           "cwd":"'$PWD'","hook_event_name":"UserPromptSubmit",
           "prompt":"<a prompt you expect pointers for>"}' \
      | ~/.claude/hooks/memory-prompt-recall.py

Expect the pointer block on stdout and exit 0. **Use a fresh `session_id` every
probe.** Pointers already served to a session are suppressed for it, so a
second probe reusing the same id returns *different, lower-ranked* pointers, and
a repeated identical prompt returns **zero bytes**. That is the deduplication
working, not a fault, and it is the easiest way to talk yourself into
diagnosing a healthy host.

This probe bypasses Claude Code's own wiring, so finish with a real session on
a prompt you know is good: only that exercises the `settings.json` entry, and
a settings-side breakage looks identical to a healthy host from the shell.

**4. The consumer checkout is still clean.**

    git -C $CONSUMER status --short

Must be as clean as it was at pre-flight. Typechanges (`T`) on tracked hook
files, or untracked `*.backup` entries beside them, are the conversion defect
written into your working tree: stop rolling out and repair this host first.
Do not `git clean` that directory before you have looked — the `.backup` files
are the only remaining copies of what those tracked files used to hold.

## Per-host verify, plugin channel

The checks above read paths a plugin install does not have. Run
`memkit doctor` first — it is the machine reader for every one of the five
below, and it additionally exercises the installed hook, which none of them
do:

```
CFG="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
PLUGIN="$CFG/plugins/cache/memkit/memkit"
MEMKIT="$(ls -d "$PLUGIN"/*/bin/memkit 2>/dev/null | sort -V | tail -1)"
"$MEMKIT" doctor --json
```

Zero `FAIL` is the bar; `INFO`, `ASSUMPTIONS-UNVERIFIED` and `UNKNOWN` do not
block, and the harness version stamp mismatches on almost every host. The five
manual checks stay below it as the fallback for a host where doctor cannot run
— no interpreter, no payload — which is precisely the state its own
`interpreter` check would have reported if it could.

These five were run against a real scratch install before being written here,
which is the whole point of a verify procedure.

The sections before this one are a NIX-FLEET rollout, written from a mixed
darwin and NixOS estate. A colleague adopting through the plugin channel — the
ordinary case, and normally a Linux workstation — needs this block and none of
the rest: there is no `darwin-rebuild`, no flake input to pin, and no consumer
checkout to keep clean. Paths below are written with `${CLAUDE_CONFIG_DIR}`
rather than a hardcoded `~/.claude`, since that is what an adopter who has
moved it will have.

**1. The plugin is installed and enabled.**

    claude plugin list

`memkit@memkit` must appear with `Status: ✔ enabled`. Absent means the install
did not land; present-and-disabled means somebody turned it off.

**2. It registered its hooks.**

    claude plugin details memkit@memkit

`Hooks (2)`: `UserPromptSubmit` and `PreToolUse`, which is what 0.3.0 and later
register.
**`Hooks (0)` is the failure this check exists for** — it is what an install
from a marketplace pin whose commit carries no payload looks like, and every
other signal on that host says success.
**`Hooks (1)` is a second, quieter failure**: one registered and one did not,
so the host serves prompts and silently stops serving subagent briefs. On a
0.2.x host it is instead the whole registration and healthy. Record
which number each host reported, not just that the command ran, and record the
release the host is on beside it — the same number means different things
either side of 0.3.0.

**3. The option reached Claude Code.**

    python3 -c 'import json,os,sys;print(json.load(open(os.path.expanduser(
      sys.argv[1])))["pluginConfigs"]["memkit@memkit"]["options"])' \
      "${CLAUDE_CONFIG_DIR:-~/.claude}/settings.json"

`{'memkitConfig': '<absolute path>'}`. An empty result is an install that
skipped `--config`, which is inert by design and says nothing at runtime.

**4. Retrieval resolves and the stores answer.**

    PLUGIN="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins/cache/memkit/memkit"
    RECALL="$(ls -d "$PLUGIN"/*/bin/memkit-recall | tail -1)"
    "$RECALL" --config <the path from check 3> --debug-config
    "$RECALL" --config <the path from check 3> --search "<a term in your store>"

`--config` is not optional here and is the reason this block exists: a shell —
yours or the agent's Bash tool — receives none of the plugin's environment, so
without it both commands answer `inert` (exit 3) on a host that is serving
correctly. `--debug-config` exits 0 and names the config and each store;
`--search` exits 0 with pointer lines, 1 for no match, 3 for inert.

**5. Derived state is where you expect it.**

    ls "${XDG_CACHE_HOME:-$HOME/.cache}/memory-recall/"

The index, the soak log and the session ledgers land under `$XDG_CACHE_HOME`
when it is set — which on a Linux workstation it often is — and under
`~/.cache` otherwise. An empty directory after a served prompt means the hook
wrote somewhere else, which is worth knowing before you read the log as
evidence of anything.

**A silent exit 0 from the hook is not a healthy host.** The wrapper exits 0 on
every path by design, including every refusal, so quiet means nothing on its
own — check 2 and check 4 are what distinguish a serving install from a refused
one.

## Rollback

Two units. They are not interchangeable, and which one applies depends only on
whether the consumer had a pinned input *before* the change you are undoing.

### Cutover rollback (first adoption)

At cutover no prior pin exists, so there is nothing to pin backwards to. Revert
the whole thing:

    cd $CONSUMER
    git rev-list --parents -n 1 <cutover commit>   # how many parents?
    git revert <cutover commit>                    # one parent: squash merge
    git revert -m 1 <cutover commit>               # two parents: merge commit
    git push
    # then on each rebuilt host, as ONE command:
    git pull && <rebuild command>

Check the parents first rather than reaching for `-m 1` out of habit: a
squash-merged PR is one ordinary commit and `-m 1` fails on it, which is a
confusing error to meet while rolling something back.

This restores the directory symlink and the in-tree hook files **together**,
which is the property that makes it safe: the layout conversion and the file
removals were one commit, so undoing it cannot leave Claude Code pointing at a
path that neither side provides. The same single-command discipline applies —
a revert pulled but not rebuilt parks the host in the same dangling window,
just from the other direction.

**Drilled for real, and it works.** Revert plus rebuild restored the
whole-directory symlink layout cleanly, with no `rm` needed and no debris. The
reverse transition is safe because home-manager owns the per-file entries it is
removing — the defect is specific to writing new entries beneath a path that is
still somebody else's symlink. Rolling *forward* again re-triggers it, so the
`rm ~/.claude/hooks` pre-step applies to every re-conversion, not just the
first.

### Bump rollback (every rollout after the first)

The consumer pins memkit by rev. Rolling back is a normal PR that moves the pin
and the lock entry back:

    # in $CONSUMER/flake.nix
    memkit.url = "github:<owner>/memkit/<older rev>";

    nix flake update memkit     # re-lock just this input
    # commit flake.nix + flake.lock, open the PR, let CI gate it, merge

Two caveats worth naming before you reach for the fast path:

- **Lock divergence.** `nix flake update memkit` updates one input. Bare
  `nix flake update` updates *everything*, and a rollback PR that quietly
  carries a dozen unrelated input moves is no longer a rollback — it is a
  fleet-wide upgrade with a rollback's commit message. Name the input.
- **Who owns the lock.** If your consumer treats an automated dependency bot as
  the sole `flake.lock` writer, a hand-rolled lock edit fights it: the next
  bot run may re-open the very bump you reverted. Pin the rev in `flake.url`
  (not just the lock) and say in the PR body why, or the rollback will not
  outlast the next bot run.

An emergency local rollback — `--override-input memkit <older-path-or-rev>` on
one host's rebuild — is legitimate to stop the bleeding on that host, but it
does not survive the next rebuild and it is invisible to every other host. Use
it to buy the time to open the PR, never instead of opening it.

## Detection

**Pre-merge is where the gate is.** The consumer's `checks.memory-eval` runs
the eval against the real corpus with the tool from the pinned input, and fails
on any drift from a committed expectations snapshot. That is what stands
between an automated bump and every host's every-prompt hook, and it is why the
gating slices are chosen deliberately rather than left at the default: a slice
that is not gating reports without blocking.

**Runtime is not gated, and that is a known posture, not an oversight.** The
hook fails open by contract; a host whose hook has been silently inert for a
week emits nothing that distinguishes it from a quiet week.

`memkit doctor --json` is the agent-invocable detector that answers "is
retrieval actually working here", and it is what the per-host verify above
leads with. It changes what a check COSTS, not when one happens: it still runs
when something runs it. What it removes is the reason a per-host check used to
be skipped — five commands whose output somebody had to read — and what it adds
is a run of the installed hook, which no manual check in this document ever
did.

Nothing in this milestone polls it. Until something does, treat the verify as
mandatory per host rather than as a spot check; `--json` and the exit code make
it a thing a cron or a deploy step can gate on when somebody wants that.

### The soak log's degradation keys, and the reader that drops them

The hook omits a `lex_*` key when nothing happened, so absence in the log means
none of it happened — which is a contract only as good as the reader holding up
its end. The known external reader,
`~/.config/nix/scripts/memory-recall-report.py`, sums a fixed four-key tuple
(`lex_spared`, `lex_unwalked`, `lex_busy_skip`, `lex_rebuilds`) written before
this branch, so every key added since is silently dropped from its health
summary — `lex_note_unwritten`, which predates this branch, and `lex_deadline`,
`lex_oversize` and `lex_unswept`, which say a sync hit the budget, refused a
file for size, or ran out of budget mid-sweep, plus four that name a file the
store owner can see in their tree and retrieval is not searching:
`lex_outside` (a `*.md` symlink whose target leaves the store),
`lex_unnameable` (a filename this hook cannot render, so a pointer to it would
name a path that does not exist), `lex_undecodable` (a filename the filesystem
holds as bytes that are not UTF-8) and `lex_linkdir` (a symlinked
subdirectory, which `os.walk` never descends).

The consequence is an operator reading "no lex degradation" while stores are
actively hitting those paths, which is exactly the observability an oversize
file's permanent staleness depends on. The fix is in another repository and
belongs there: widen that tuple to the full `_LEX_COUNTS` key set, or derive it
from `hook._LEX_COUNTS.keys()` if the collector can import memkit, so the next
key added here does not need a matching hand-edit there.

## The edit-to-live loop

Two different things travel at two very different speeds, and confusing them is
the most common way to conclude the system is broken.

**Memory-store edits are live at the next prompt.** No rebuild, no bump, no
restart. The hook reads the *live root's working tree* — the actual directory
on disk that the config's root resolves to — so editing a memory file makes it
retrievable immediately. This is deliberate: the store is data, and data that
needed a deploy would not get written.

The one place this bites: if your config pins a store's live root to a fixed
canonical path, then edits made in a *worktree* of that repo are not live —
the hook goes on serving the canonical copy until the change lands there. No
environment variable moves it, and that is the design: which directories an
every-prompt hook reads from is the memory-poisoning surface, so the root
overrides a config declares are honoured by the operator tools and never on
the prompt path. To see what a worktree's copy *would* retrieve, use the tool
that takes the tree as an argument — `memory-eval --repo "$PWD"` scores that
checkout's stores through the same retrieval path the hook uses.

The **checker** is the other half of the same story and needs no redirection
at all: `memory-integrity` verifies, blames and rewrites `edit_root`, which
follows the checkout you are standing in, so a run from a worktree checks that
worktree. It names both trees on every run — the one it verified and the one
the hook serves — so a green run is never mistakable for a claim about the
live copy.

**Tooling changes travel the long way**, and there is no shortcut that is also
safe:

    memkit main
      -> dependency-bot bump PR against the consumer
      -> consumer CI (checks.memory-eval gates retrieval behaviour)
      -> merge
      -> host rebuild

No stage is redundant. The eval gate is the only pre-merge check that
sees the real corpus, and the rebuild is the only thing that puts new bytes on
a host.

**Measured latency: not yet recorded.** This loop has not been exercised on a
real tooling fix at the time of writing. The first one to go through it should
record the wall-clock end to end here, split into bump-PR latency (bot cadence,
or dashboard approval if the input is gated) and merge-to-host latency.

### Escape hatch, and when it stops being one

For a live debug session — you are iterating on the hook itself and a bump per
iteration is absurd — run the hook from a local checkout against the *same*
config the deployed one uses:

    MEMKIT_CONFIG=$CONSUMER/<path to your config>.json \
      python3 <memkit checkout>/src/memkit/memory_prompt_recall.py < probe.json

Same config, same stores, same wordlist (it resolves beside the source file).
Or point one rebuild at a local tree with `--override-input memkit <path>`.

This is deliberately **not** the default and deliberately not wired into the
module: it is the ambient-configuration path the design rejects, kept available
for a human debugging deliberately and never for a host.

**The plugin channel does not weaken that rejection.** A plugin install
delivers the config path through Claude Code — a typed `userConfig` option set
at install time, else a file in the plugin's own data directory, **and nothing
else**: a config inside the payload would let the code you installed choose
which of your directories an every-prompt hook reads. Its wrapper exports
`MEMKIT_CONFIG` from whichever of those answered. What matters for this section is the other half: when none of
them answers, the wrapper **unsets** the variable rather than leaving it. So a
machine that has both channels — a nix-managed hook and a plugin install, which
is the author's own case — cannot have the plugin quietly serve whatever config
the launching shell exported. Delivery is per install and controlled by Claude Code,
which is the same property `configFile` gets by baking the path into the
wrapper, arrived at by a different route.

The escape hatch above still works while a plugin is installed, and it is still
a human-only path: `MEMKIT_CONFIG=…` in front of a checkout's hook affects that
one invocation and nothing Claude Code runs.

**Promotion threshold.** Keep it an informal workaround while it stays rare. It
becomes a supported first-class dev mode — a documented flag, a module option,
a tested path — when either of these is true:

- an *urgent* fix cannot reach hosts within one working day through the normal
  loop (the dashboard-gated input is the likeliest cause; measure it, do not
  assume it), or
- the hatch is reached for more than twice in a quarter, which is the point at
  which "informal workaround" has become "undocumented dependency".

Record each use here with the date and the reason. Two lines is enough; the
count is the signal.
