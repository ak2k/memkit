# What a memkit install puts on your machine

Written because memkit's payload is a hook that runs before every prompt you
type, and "it's on the marketplace" is not an answer to what that means. Every
number here is read out of the tree at the pinned sha rather than remembered.

## What arrives

`/plugin install memkit@memkit` clones **the whole tracked tree** at the sha
pinned in `.claude-plugin/marketplace.json` — not a built artifact, and not a
subset chosen for the hook. At the sha this release pins that is **57 files,
about 1.2 MiB**:

| what | files | why it is there |
|---|---|---|
| `bin/`, `src/memkit/`, `hooks/`, `.claude-plugin/` | 13 | the payload proper — the wrappers, the hook module, the manifests |
| `tests/` | 26 | not needed at run time; see below |
| `.github/`, `nix/`, `tools/`, `flake.*`, `pyproject.toml`, config files | 14 | likewise |
| `docs/`, `README.md`, `LICENSE`, `NOTICE` | 4 | this file among them |

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

So the payload is code you can read that acts on decisions you made elsewhere.
The thing worth auditing before installing is not this tree's size; it is
`bin/memkit-hook` and `bin/lib/common.sh`, which are 550 lines of POSIX shell
between them — mostly comment — and run no command that is not a shell builtin.

## Reproducing these numbers

```
sha=$(python3 -c 'import json;print(json.load(open(".claude-plugin/marketplace.json"))["plugins"][0]["source"]["sha"])')
git ls-tree -r --name-only "$sha" | wc -l
git ls-tree -r -l "$sha" | awk '{s+=$4} END {printf "%.0f KiB\n", s/1024}'
git ls-tree -r "$sha" | awk '$1=="100755"{print $4}'
```
