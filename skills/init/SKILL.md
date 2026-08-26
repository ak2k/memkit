---
name: init
description: Set up memkit on this machine — create the memory store, write the config, and seed a memory that proves retrieval works. Use ONLY when the user asks to set up, initialise or configure memkit. This command writes files and requires the user's explicit consent between two turns.
disable-model-invocation: true
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/bin/memkit init --dry-run:*), Bash(${CLAUDE_PLUGIN_ROOT}/bin/memkit init --confirm:*)
---

# memkit init

**Two turns, and they are not the same command twice.** The first prints what
would happen and writes nothing. The second applies exactly that, and only if
nothing underneath it has moved.

## Turn one — show, do not do

```
${CLAUDE_PLUGIN_ROOT}/bin/memkit init --dry-run
```

Optional flags, each of which changes the plan and therefore the digest, so
whichever you pass here you must pass again in turn two. Both grants are prefix
matches, so passing any of them keeps you inside the pre-approval — an exact
grant on turn one would have dropped you into a permission prompt in the middle
of the handshake for using a flag this page told you to use:

- `--store PATH` — where the memory store goes.
- `--config PATH` — where the config goes.
- `--wire-claude-md` — append an `@-import` of the store's `MEMORY.md` to the
  user's `CLAUDE.md`. Read the manifest's own note about what that buys before
  recommending it: it puts each hot memory's *description* in every session,
  not its body.
- `--auto-dream-off` — turn the harness's own auto-memory off, so it stops
  writing and consolidating memories beside memkit's.

**Relay the manifest verbatim, including the digest.** It names every path,
every write, where each symlink lands and whether a target is tracked by git.
Then stop and ask.

## Turn two — only after the user says yes, in a new message

```
${CLAUDE_PLUGIN_ROOT}/bin/memkit init --confirm <the digest from turn one>
```

The digest binds the state of the tree, not the text that was read. If anything
under the manifest changed in between, this refuses and nothing is written —
re-run the dry-run, relay the new manifest, and ask again.

## Exit codes

| code | meaning | what to do |
|---|---|---|
| 0 | done, or the manifest printed | relay it |
| 2 | usage error | fix the arguments |
| 5 | **refused, and nothing was written** | stderr names which refusal and why. Relay it. Do not retry the same command |
| 6 | started and did not finish | the journal says how far it got; re-running converges on the remainder |

A refusal is a decision, not a failure to try harder. `foreign-config` means
somebody else wrote the config and init will not overwrite it;
`flat-store-adoption` means the store already holds memories that creating
`search/` would silently un-retrieve, and the message spells out the one-step
migration. Relay the reason and let the person choose.

## Afterwards

`/memkit:doctor` should report zero `FAIL`, and the next **new** session's first
prompt should surface the canary memory. Hooks are registered when a session
starts, so the session that ran init is not the one that will show it.
