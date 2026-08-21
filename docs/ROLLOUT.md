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

## The one unsafe window

The harness invokes the hook by path — `~/.claude/hooks/memory-prompt-recall.py`
in a `settings.json` the harness reads, not something memkit controls. Before
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
a host whose harness is broken — which is why check 1 below tests every entry
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
count), so the same command run from elsewhere searches nothing — but it now
says so rather than looking like a store with no answer: exit **3** and
`inert — … gated to another tree` on stderr. Read the code, not the silence.
Exit **1** is the one that means the stores really were opened and nothing
matched; **2** is a failure to search at all. Pick a term you know matches a
`search/`-tier memory: `hot/` memories are excluded from retrieval by design,
because they are in context already.

`memory-recall --debug-config` answers the same question with the reasoning
shown — which config resolved, and per store whether it is `searched`, `NOT on
disk`, or `NOT searched here` — and exits 3 when none of them is searchable.

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

This probe bypasses the harness's own wiring, so finish with a real session on
a prompt you know is good: only that exercises the `settings.json` entry, and
a settings-side breakage looks identical to a healthy host from the shell.

**4. The consumer checkout is still clean.**

    git -C $CONSUMER status --short

Must be as clean as it was at pre-flight. Typechanges (`T`) on tracked hook
files, or untracked `*.backup` entries beside them, are the conversion defect
written into your working tree: stop rolling out and repair this host first.
Do not `git clean` that directory before you have looked — the `.backup` files
are the only remaining copies of what those tracked files used to hold.

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
removals were one commit, so undoing it cannot leave the harness pointing at a
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
  (not just the lock) and say in the PR body why, or the rollback is temporary
  by construction.

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
week emits nothing that distinguishes it from a quiet week. Nothing in this
milestone changes that — the per-host checks above are the only detector, and
they run when a human runs them. An agent-invocable doctor that answers "is
retrieval actually working here" is deferred; until it exists, treat the verify
recipe as mandatory per host rather than as a spot check.

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

Every stage is load-bearing. The eval gate is the only pre-merge check that
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
