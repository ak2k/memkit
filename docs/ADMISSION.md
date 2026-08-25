# What a memkit install puts on your machine

Written because memkit's payload is a hook that runs before every prompt you
type, and "it's on the marketplace" is not an answer to what that means. Every
number here is read out of the tree at the pinned sha rather than remembered.

## What arrives

`/plugin install memkit@memkit` clones **the whole tracked tree** at the sha
pinned in `.claude-plugin/marketplace.json` — not a built artifact, and not a
subset chosen for the hook. At the sha this release pins that is **63 files,
about 1.3 MiB**:

| what | files | why it is there |
|---|---|---|
| `bin/`, `src/memkit/`, `hooks/`, `.claude-plugin/` | 13 | the payload proper — the wrappers, the hook module, the manifests |
| `tests/` | 27 | not needed at run time; see below |
| `.github/`, `nix/`, `tools/`, `flake.*`, `pyproject.toml`, config files | 16 | likewise |
| `README.md`, `LICENSE`, `NOTICE`, and all of `docs/` | 7 | including this file |
| `.git/` | ~44 | the clone's own history, about 0.7 MiB on top of the tracked files. Varies with your git version |

Measured on a real install: the tracked files above, plus a `.git` of **roughly
44 more files and about 0.7 MiB** — roughly, because the sample hooks and pack
indexes git writes vary by git version, and a second machine measured 48. The
tracked-tree numbers are exact; the clone's are not, and the argument rests on
the exact ones. It is
there because the plugin system installs by cloning, and it is what makes the
verification below possible.

This file and `docs/STORE.md` are in the copy you install, from 0.2.0 onward —
they were written after the commit 0.1.0 pinned, so on that release the
installed tree carried `docs/ROLLOUT.md` alone and the README's links to them
dangled.

**The payload minimum is pinned; the maximum is not.** `tests/test_plugin_surface.py`
holds a list of the files the wrappers need and fails if one is missing, so the
plugin cannot ship without them. Nothing prunes the other direction — the plugin
system installs a git tree, so what an adopter receives is whatever the
repository tracks. That is a deliberate accepted cost rather than an oversight:
a build step that produced a smaller payload would be a second thing that can
disagree with the repository about what memkit is, and the failure mode of
*that* is a plugin whose shipped code is not the code its tests ran against.

Nothing outside `bin/` reaches your `PATH`. The plugin system puts `bin/` on the
agent's `PATH` and nothing else, so the three executable files in the tree that
are not wrappers — the rig's pty driver, the hook dumper, the wordlist checker —
are inert unless you run them yourself by path.

## What runs, and when

One hook, registered by `hooks/hooks.json`:

```
UserPromptSubmit → ${CLAUDE_PLUGIN_ROOT}/bin/memkit-hook   (timeout 15s)
```

It runs on every prompt you submit. It exits 0 on every path it has — including
every failure — because on `UserPromptSubmit` a non-zero exit takes your prompt
away from you.

Until you give it a config it is **inert**: it reads no directory of yours,
writes nothing but a record of its own refusal, and prints nothing. With a
config it reads the store directories that config names, and nothing else.

## Where the trust boundary sits

**Nothing in the payload decides what memkit reads or what runs it.** Both of
those come from the config, and the config comes from you:

- The config path is admitted from exactly two places — the `memkitConfig`
  install option you typed, and `$CLAUDE_PLUGIN_DATA/memkit.json` — and from
  nowhere else. A `memkit.json` sitting inside the plugin tree is **not** a
  route: `test_a_config_inside_the_payload_is_not_a_rung` writes one there and
  asserts the hook ignores it.
- `$MEMKIT_CONFIG` from the ambient environment is not a route either; the
  wrappers override it in both directions, so the repository your session
  happens to be standing in cannot point the hook at its own directories.
- The `interpreter` field — the binary exec'd on every prompt — is read from
  that config and refused unless it is an absolute, canonical path to an
  executable file. Relative paths and the process-relative namespaces
  (`/proc/self/...`, `/dev/fd/...`) are refused by name.

**Two residuals, because the claim above has a ceiling and the source names
it.**

- `$CLAUDE_PLUGIN_DATA` is Claude Code's directory and it is **writable by the
  payload** — memkit's own hook writes `trust.json` there, beside the
  `memkit.json` that rung 2 reads. So a release could write that file on one
  prompt and be honoured by every later, clean release. What makes it
  tolerable rather than theoretical: nothing in this build writes it, and
  detecting a config the user did not author is deferred to `memkit init`.
- Both routes are environment variables, so both **trust Claude Code's
  environment contract**. Anything that can put `CLAUDE_PLUGIN_DATA` into the
  launching environment — a wrapper script, a nested invocation, another
  plugin's tooling — reaches the same rung. The honest form of the claim is
  independence from *your shell's* `$MEMKIT_CONFIG`, which is what the wrappers
  enforce by unsetting it; it is not independence from anything that can write
  Claude Code's own environment.

So the payload is code you can read that acts on decisions you made elsewhere,
within those two limits.
The thing worth auditing before installing is not this tree's size; it is
`bin/memkit-hook` and `bin/lib/common.sh`, which are about 550 lines of POSIX shell
between them — mostly comment — and run no command that is not a shell builtin.

## Reproducing these numbers

In the repository, against the sha the marketplace entry names:

```
sha=$(python3 -c 'import json;print(json.load(open(".claude-plugin/marketplace.json"))["plugins"][0]["source"]["sha"])')
git ls-tree -r --name-only "$sha" | wc -l
git ls-tree -r -l "$sha" | awk '{s+=$4} END {printf "%.0f KiB\n", s/1024}'
git ls-tree -r "$sha" | awk '$1=="100755"{print $4}'
```

On the machine, against what was actually installed. This is the stronger
check — it asks the clone which commit it is, rather than counting files:

```
CFG="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
cd "$(ls -d "$CFG"/plugins/cache/memkit/memkit/*/ | tail -1)"
git rev-parse HEAD          # must equal the pin below
git status --porcelain      # must be empty: nothing edited the payload after the clone

python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["plugins"][0]["source"]["sha"])' \
  "$CFG"/plugins/marketplaces/memkit/.claude-plugin/marketplace.json
```

Compare against the **marketplace clone's** manifest, not the one inside the
payload. A commit cannot name its own sha, so the installed copy carries the
pin of the release before it — comparing against that file reports tampering on
a perfectly clean install.

The repository commands above do not work inside the installed copy *with the
pinned sha*: it is a shallow clone, so `git ls-tree <that sha>` answers
`fatal: not a tree object` there. `git ls-tree -r HEAD` works, and returns the
same count as the table above — the release procedure derives that number from
the tree this file ships in, so the two agree by construction rather than by
somebody remembering.
