# Rollout and rollback

memkit is not an ordinary dependency of the repository that consumes it. Its
hook runs on **every prompt, on every host**, and it is fail-open by contract:
when it cannot run it prints nothing and exits 0. A rollout that breaks it
therefore produces no error, no failed unit and no red check — only pointers
that stop arriving, which nobody notices until they go looking for a memory
that used to surface. Everything below exists because that failure is silent.

Written from one real rollout: a Nix flake consumer with a mixed darwin and
NixOS fleet. Host names and aliases below are placeholders (`$CONSUMER` is the
consumer checkout); the shapes are what transfers.

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

NixOS hosts never see the window: their deploy aliases fetch, hard-reset to
`origin/main` and `nixos-rebuild switch` inside a single remote invocation, and
unattended auto-upgrade does the same. Pull and switch are already one step
there, so nothing about the order below is special for them.

## Rollout order

1. **Pre-flight.** `git -C $CONSUMER status --short` is clean on every host you
   are about to touch. A dirty consumer checkout is not cosmetic here — see the
   stale-directory-symlink hazard under [Verify](#per-host-verify), which
   deposits build output into the working tree.
2. **Merge** the cutover PR (first adoption) or the input-pin bump PR
   (every rollout after that).
3. **First host: one darwin machine, and only one.** `git pull && <rebuild>` as
   a single command. Verify it (below). **Then drill the cutover rollback on
   this host, before any second host rebuilds** — a rollback path that has only
   been read is not a rollback path. Roll forward again, verify again.
4. **Remaining darwin hosts**, same single command, same verification.
5. **NixOS hosts** via their deploy aliases, or by letting unattended
   auto-upgrade pick the change up on its own schedule.

Steps 3 and 4 are separated for the first adoption because the cutover is the
only change that converts the hooks directory's *layout*. Routine input-pin
bumps do not; for those, step 3's drill is optional and step 4 can be a sweep.

## Per-host verify

Four checks. The first three prove the tool is installed and answers; the
fourth is the one that caught a real fault.

**1. The hooks directory has the per-file layout.**

    ls -l ~/.claude/hooks/

Expect `memory-prompt-recall.py` and `common-words.txt` as symlinks into
`/nix/store`, beside whatever other per-file hook entries your consumer keeps.
What you must *not* see is `~/.claude/hooks` still being a **directory symlink**
into the consumer checkout — that is the pre-cutover layout:

    ls -ld ~/.claude/hooks        # want a real directory, not a symlink
    readlink ~/.claude/hooks      # want no output

If the directory symlink survives the rebuild, home-manager writes the new
per-file entries *through* it and they land in the consumer's git working tree:
tracked files turn into symlinks (`git status` reports `T`, a typechange),
`.backup` copies of the originals appear untracked beside them, and the
deployed hook is now a file inside a git checkout that any `git clean` will
delete. Check for it explicitly; the hook keeps working, so nothing else tells
you.

**2. On-demand search returns hits.**

    cd $CONSUMER && memory-recall --search "<a term you know is in the store>"

Run it **from inside the consumer checkout**, not from `$HOME`. A store with a
`cwd_gate` is searched only from inside the named root (its git worktrees
count), so the same command run from elsewhere returns nothing at all and exits
0 — an empty verify that reads exactly like a broken one. Pick a term you know
matches a `search/`-tier memory: `hot/` memories are excluded from retrieval by
design, because they are in context already.

**3. The hook injects pointers.** Without starting a session:

    echo '{"session_id":"verify-'$(date +%s)'","transcript_path":"/dev/null",
           "cwd":"'$PWD'","hook_event_name":"UserPromptSubmit",
           "prompt":"<a prompt you expect pointers for>"}' \
      | ~/.claude/hooks/memory-prompt-recall.py

Expect the pointer block on stdout and exit 0. **Use a fresh `session_id` every
probe.** Pointers already served to a session are suppressed for it, so a
second probe reusing the same id returns *different, lower-ranked* pointers —
or none — and that reads as a regression when it is the deduplication working.

This probe bypasses the harness's own wiring, so finish with a real session on
a prompt you know is good: only that exercises the `settings.json` entry, and
a settings-side breakage looks identical to a healthy host from the shell.

**4. The consumer checkout is still clean.**

    git -C $CONSUMER status --short

Must be as clean as it was at pre-flight. Anything new under the hooks
directory means check 1 found the fault and you should stop rolling out.

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
canonical path, then edits made in a *worktree* of that repo are not live, and
a checker run from that worktree reads and regenerates the canonical tree
rather than your own. Both are fixed the same way, by the env override the
root declares:

    <ENV>=$PWD memory-integrity --write

where `<ENV>` is `roots.<name>.env` from your config. `memory-integrity`'s own
remediation lines name the variable for you when your config declares one.

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
