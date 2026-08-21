# Sourced by the three wrappers in the directory above. Not executable, and in
# a subdirectory, because `bin/` is on the agent's PATH while `bin/lib/` is not
# — a shared file beside the wrappers would be a command an agent could invoke.
#
# What lives here is the answer to two questions every wrapper asks and none of
# them may answer differently: which config this install serves, and which
# interpreter runs it. Three copies of that would be three chances to drift, and
# the config question in particular is the memory-poisoning surface of the whole
# design — the set of directories an every-prompt hook reads.
#
# POSIX sh only. The harness runs these with whatever `/bin/sh` is, and on a
# stock macOS that is bash 3.2 in POSIX mode; nothing here may need bash 4.

# Each wrapper derives the plugin tree from its own `$0` before sourcing this
# file — it has to, since that is how it finds this file — so the derivation
# lives inline in all three rather than here. What it does and why, once:
#
#   - From `$0`, not from `$CLAUDE_PLUGIN_ROOT`. A script can always find
#     itself, which is what makes the third config rung below work when the
#     harness exports nothing at all. Doctor, which runs the wrapper directly,
#     is exactly that case.
#   - `command -v` when `$0` carries no slash. `bin/` is on the agent's PATH,
#     so `memkit-recall …` typed as a bare command arrives with argv[0] of
#     `memkit-recall` and no directory to walk up from.
#   - `pwd -P` to normalize. The harness expands `${CLAUDE_PLUGIN_ROOT}` with a
#     TRAILING SLASH (measured on 2.1.238), so argv[0] arrives as
#     `<root>//bin/x` and naive string arithmetic carries the doubled separator
#     into every path this then builds.

# `~/x` as a person types it. The option value is a string the adopter typed
# into an install command, not a shell word the shell ever expanded, so a
# config named `~/.cache/...` arrives with a literal tilde and every rung below
# would silently miss it.
memkit_expand_home() {
    case $1 in
        "~/"*) printf '%s\n' "$HOME/${1#\~/}" ;;
        "~") printf '%s\n' "$HOME" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

# The config this install serves, or nothing. Three rungs, in order, first
# existing file wins:
#
#   1. CLAUDE_PLUGIN_OPTION_MEMKITCONFIG — the harness's own typed userConfig
#      mechanism, settable non-interactively at install with
#      `--config memkitConfig=<path>`. The variable name is the manifest key
#      uppercased; `tests/test_plugin_surface.py` pins the two together, since
#      nothing else connects a key in a manifest to a name in this file.
#   2. $CLAUDE_PLUGIN_DATA/memkit.json — SKIPPED ENTIRELY when the variable is
#      unset rather than built from an empty expansion. `${unset}/memkit.json`
#      is `/memkit.json`, and a hook that reads every prompt must never stat a
#      root-level path it did not mean to name.
#   3. <plugin root>/memkit.json — derived from this file's own location, and
#      the only rung that depends on no environment at all. The first two are
#      both plugin env exports into the hook process, i.e. one failure mode
#      wearing two hats.
#
# Nothing found is not an error: the wrapper goes on to run the hook with no
# config, which is inert by construction — no stores, no pointers, exit 0.
memkit_resolve_config() {
    _root=$1
    if [ -n "${CLAUDE_PLUGIN_OPTION_MEMKITCONFIG:-}" ]; then
        _candidate=$(memkit_expand_home "$CLAUDE_PLUGIN_OPTION_MEMKITCONFIG")
        [ -f "$_candidate" ] && {
            printf '%s\n' "$_candidate"
            return 0
        }
    fi
    if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
        _candidate="$CLAUDE_PLUGIN_DATA/memkit.json"
        [ -f "$_candidate" ] && {
            printf '%s\n' "$_candidate"
            return 0
        }
    fi
    if [ -n "$_root" ]; then
        _candidate="$_root/memkit.json"
        [ -f "$_candidate" ] && {
            printf '%s\n' "$_candidate"
            return 0
        }
    fi
    return 1
}

# The interpreter, preferring the one the config recorded.
#
# PATH probing alone hands the process that reads every prompt to whatever
# direnv/mise/venv shim the launching shell happened to carry, and the nix
# channel pins its interpreter absolutely — the plugin channel must not
# silently drop that guarantee. So init records what it resolved, and this
# prefers it.
#
# "Unusable" is `not an executable file`, deliberately not `fails to run`:
# establishing the second costs a fork on every prompt, and the failure it
# would catch (an executable that is not a working python) surfaces one layer
# down as a traceback the harness reports without blocking the turn.
#
# The extraction is a grep for one top-level string key rather than a JSON
# parse, because the only thing available to parse with at this point is the
# interpreter being looked for. It is bounded by what writes the field: init,
# with an absolute path it resolved itself.
memkit_config_interpreter() {
    [ -n "$1" ] && [ -f "$1" ] || return 1
    _found=$(sed -n 's/.*"interpreter"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$1" \
        2>/dev/null | head -n 1)
    [ -n "$_found" ] || return 1
    printf '%s\n' "$_found"
}

memkit_resolve_interpreter() {
    _config=$1
    _recorded=$(memkit_config_interpreter "$_config") || _recorded=""
    if [ -n "$_recorded" ] && [ -x "$_recorded" ]; then
        printf '%s\n' "$_recorded"
        return 0
    fi
    for _candidate in python3 python; do
        _path=$(command -v "$_candidate" 2>/dev/null) || continue
        [ -x "$_path" ] && {
            printf '%s\n' "$_path"
            return 0
        }
    done
    return 1
}

# Where the fallback checker comes from when this machine has no python new
# enough to run the one in this tree. `uvx` provisions its own interpreter,
# which is what makes a stock-python mac (3.9.6) able to run a 3.12 checker at
# all. Unpinned, and knowingly: it resolves whatever `main` holds, so a plugin
# pinned to an older release can route checker work through a newer checker.
# The store format is what both speak and it has not moved, but the skew is
# real and belongs in a release note rather than in a comment nobody diffs.
MEMKIT_UVX_SPEC="git+https://github.com/ak2k/memkit"

# The checker's floor, which is NOT the hook's. Kept as two numbers rather than
# a string so the test that scrapes `sys.version_info < (3, 12)` out of
# memory_integrity.py has something to compare against: the number lives in two
# files by necessity — one of them cannot import the other — and a floor that
# drifted would route a 3.11 python straight into the guard it exists to avoid.
MEMKIT_CHECKER_FLOOR_MAJOR=3
MEMKIT_CHECKER_FLOOR_MINOR=12

memkit_python_meets_checker_floor() {
    "$1" -c "import sys; sys.exit(0 if sys.version_info[:2] >= ($MEMKIT_CHECKER_FLOOR_MAJOR, $MEMKIT_CHECKER_FLOOR_MINOR) else 1)" \
        >/dev/null 2>&1
}

# Resolve the route for checker-backed work and export it. Three states, and
# every caller has to be able to tell them apart:
#
#   python  a local interpreter meets the floor, so the checker that runs is
#           THIS tree's — same release as the hook, by construction.
#   uvx     no local interpreter does, but uvx can provision one.
#   none    neither. The operation that needs it refuses by name and writes
#           nothing; a seeded memory with no ledger row is a broken store, so
#           half-completing is worse than not starting.
memkit_resolve_checker() {
    _root=$1
    _base=$2
    MEMKIT_CHECKER_ROUTE=none
    MEMKIT_CHECKER_CMD=""
    # The already-resolved interpreter first: on a machine where it is new
    # enough, that is the whole probe and it costs one fork.
    for _cand in "$_base" python3.14 python3.13 python3.12 python3; do
        [ -n "$_cand" ] || continue
        _path=$(command -v -- "$_cand" 2>/dev/null) || continue
        if memkit_python_meets_checker_floor "$_path"; then
            MEMKIT_CHECKER_ROUTE=python
            MEMKIT_CHECKER_CMD="$_path -m memkit.memory_integrity"
            break
        fi
    done
    if [ "$MEMKIT_CHECKER_ROUTE" = none ] && command -v uvx >/dev/null 2>&1; then
        MEMKIT_CHECKER_ROUTE=uvx
        MEMKIT_CHECKER_CMD="uvx --from $MEMKIT_UVX_SPEC memory-integrity"
    fi
    export MEMKIT_CHECKER_ROUTE MEMKIT_CHECKER_CMD
}

# What the adopter is told when no interpreter resolves. Named rather than
# silent: silence here is indistinguishable from a corpus with nothing to say,
# and this is the one failure that no amount of fixing the store will cure.
#
# The reader is doctor, which runs this wrapper directly and captures stderr.
# In a live session it goes to the harness's debug log, because the wrapper
# exits 0 whatever happens — see the exit contract in `memkit-hook`.
memkit_no_interpreter_message() {
    printf '%s\n' \
        "memkit: no python3 on PATH and none recorded in the config, so the" \
        "recall hook cannot run. Install python 3.9 or newer, or record an" \
        "absolute interpreter path as \"interpreter\" in the memkit config." \
        "Config in use: ${1:-<none resolved>}"
}
