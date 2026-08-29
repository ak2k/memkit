---
name: doctor
description: Report whether memkit's memory retrieval is actually working on this machine. Use when memkit was installed and no pointers appear, when a prompt should have surfaced a memory and did not, when asked to check or diagnose a memkit install, or before concluding that a memory does not exist.
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/bin/memkit doctor:*)
---

# memkit doctor

Run it, relay it, act only on what it says is yours to act on.

```
${CLAUDE_PLUGIN_ROOT}/bin/memkit doctor --json
```

Two follow-ups are covered by the same grant, and no other subcommand is:

- `--config <path>` re-runs against a specific config. One check's remedy asks
  for exactly this, and without it that remedy named a command this skill
  could not issue.
- `--check <id>` re-runs one check by name after a fix, which is cheaper than
  a whole report and is how you show something is now green.

## Relay the report verbatim

The envelope's `report` field is the human text. **Print it as it is.** Do not
summarise it, do not re-derive its conclusions, and do not answer the user's
question from your own reading of the checks.

That is not a style preference. The failure this command exists to prevent is a
confident wrong answer about an install, and the most reliable way to produce
one is to attach a plausible summary to a correct report — the reader then has
two accounts and no way to tell which one was measured.

## Then branch, on exactly these fields

Every check carries `status`, `actor` and `terminal`.

- `schema` is the envelope's own version, and the rule for it is the config
  reader's: **a number higher than the one you know is not a report to read
  half of.** Say so and stop. The check ids and the status set may both grow
  without it moving; a bump means the shape changed.
- `status` is one of `PASS`, `INFO`, `ASSUMPTIONS-UNVERIFIED`, `UNKNOWN`,
  `FAIL`. Nothing else is ever emitted; a value outside that set is a bug worth
  reporting, not a case to guess at.
- **All-green is zero `FAIL`.** `INFO`, `ASSUMPTIONS-UNVERIFIED` and `UNKNOWN`
  never block. The harness version stamp mismatches for almost every adopter,
  and a criterion that counted it would be unreachable.
- `verdict` is `OK`, or `PROBLEMS: <n> FAIL, <m> unverified`. The exit code says
  the same thing: 0 for OK, 1 for problems.

  **Exit 1 has a second meaning on this binary** and the two are worth telling
  apart: `memkit`'s own table also uses 1 for "could not start at all". A
  report on stdout means this command ran and found problems; an empty stdout
  with a `memkit:` line on stderr means it never started, and no argument will
  change that.

**You may act only on a check whose `actor` is `agent` and whose `terminal` is
`false`.** For anything else, relay the `remedy` to the person and stop:

- `actor: user` means the remedy changes the harness's own configuration, or
  decides which directories an every-prompt hook reads. That is the person's to
  decide, and their machine to change.
- `terminal: true` means retrying with different arguments cannot help —
  Windows, no interpreter, no checker route. Say so and stop; a retry loop
  against a machine that cannot run memkit is the worst outcome available.

## What not to conclude

A `canary-retrieval` FAIL means the store did not answer a query that can only
be answered by memkit's own file. It does not mean the user's memory is absent.
Nothing in this report licenses "there is no such memory" — that is what the
search command's exit 1 says, and only after this report is green.
