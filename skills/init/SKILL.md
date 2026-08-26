---
name: init
description: Set up memkit on this machine — create the memory store, write the config, and seed a memory that proves retrieval works. Use ONLY when the user asks to set up, initialise or configure memkit. This command writes files and requires the user's explicit consent between two turns.
disable-model-invocation: true
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/bin/memkit init --dry-run:*)
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
whichever you pass here you must pass again in turn two. The turn-one grant is
a prefix match, so passing any of them keeps this read-only call inside the
pre-approval — an exact grant would have dropped you into a permission prompt
for using a flag this page told you to use, on the turn that writes nothing:

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

**This call is deliberately not pre-approved.** Turn one is, because it writes
nothing; this one will raise a permission prompt showing the exact argv, and
that prompt IS the consent — the only part of this handshake the harness
enforces rather than the model observing. A pre-approved `--confirm:*` would
let the whole thing happen inside one turn: run the dry-run, read the digest
out of your own tool result, and apply it with `--wire-claude-md` and
`--auto-dream-off` attached, writing to the user's `CLAUDE.md` and
`settings.json` without a message they ever saw. Do not work around the prompt.

The digest binds the state of the tree, not the text that was read. If anything
under the manifest changed in between, this refuses and nothing is written —
re-run the dry-run, relay the new manifest, and ask again.

## Exit codes

| code | meaning | what to do |
|---|---|---|
| 0 | done, or the manifest printed | relay it |
| 1 | memkit could not start at all — no interpreter, or an incomplete payload | stderr names what is missing; nothing about the arguments will change it |
| 2 | usage error | fix the arguments |
| 5 | **refused, and nothing was written** | stderr names which refusal and why. Relay it. Do not retry the same command |
| 6 | started and did not finish | the journal says how far it got. **Recover with both turns again, not by repeating the confirm**: what landed has changed the digest, so the old `--confirm` now refuses as stale. Run `--dry-run`, relay the new manifest — it lists only what is left — and confirm that |

A refusal is a decision, not a failure to try harder. `foreign-config` means
somebody else wrote the config and init will not overwrite it;
`flat-store-adoption` means the store already holds memories that creating
`search/` would silently un-retrieve, and the message spells out the one-step
migration. Relay the reason and let the person choose.

## Afterwards

`/memkit:doctor` should report zero `FAIL`, and the next **new** session's first
prompt should surface the canary memory. Hooks are registered when a session
starts, so the session that ran init is not the one that will show it.
