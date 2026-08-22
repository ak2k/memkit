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
#     itself, and doctor runs the wrapper directly with none of the harness's
#     variables set — so a derivation that needed one would leave the tree
#     unlocatable in the state an adopter reaches for diagnosis.
#   - `command -v` when `$0` carries no slash. `bin/` is on the agent's PATH,
#     so `memkit-recall …` typed as a bare command arrives with argv[0] of
#     `memkit-recall` and no directory to walk up from.
#   - `pwd -P` to normalize. The harness expands `${CLAUDE_PLUGIN_ROOT}` with a
#     TRAILING SLASH (measured on 2.1.238), so argv[0] arrives as
#     `<root>//bin/x` and naive string arithmetic carries the doubled separator
#     into every path this then builds.

# The name the running wrapper answers to, set by each wrapper as its first
# act and read by every message below. One deliberate cross-function value, and
# it is not the pseudo-local pattern this file otherwise avoids: it is set once
# before anything is called, never written here, and its absence is survivable.
#
# It exists because the messages are shared and the exit codes are not. The
# no-interpreter refusal said `memkit:` from all three wrappers while
# `memkit-recall` exits 4 for it — and 4 in the `memkit` table an agent would
# then look it up in means "the subcommand is not in this build", which is the
# wrong diagnosis reached by trusting the name in the message.
# The fallback is for a caller that sources this library directly — doctor's
# probes, and the tests — and NOT a licence for a wrapper to omit it: a wrapper
# that did would name the wrong binary in every message with nothing failing.
# Nothing here can enforce that (a hard failure would be on the every-prompt
# path, for a diagnostic), so the enforcement is a test that reads each
# wrapper.
MEMKIT_SELF=${MEMKIT_SELF:-memkit}

# `~/x` as a person types it. The option value is a string the adopter typed
# into an install command, not a shell word the shell ever expanded, so a
# config named `~/.cache/...` arrives with a literal tilde and every rung below
# would silently miss it.
memkit_expand_home() {
    # shellcheck disable=SC2088  # the LITERAL tilde is what this matches: the
    # value never passed through a shell, so an unexpanded `~` is the input,
    # not a mistake in this pattern.
    case $1 in
        "~/"*) printf '%s\n' "$HOME/${1#\~/}" ;;
        "~") printf '%s\n' "$HOME" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

# The config this install serves, or nothing. Two rungs, in order, first
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
#
# BOTH RUNGS TRUST THE HARNESS'S ENV CONTRACT. They are two environment
# variables, and nothing here vets where they came from — so whatever can put
# `CLAUDE_PLUGIN_DATA` into the launching environment (a wrapper script, a
# nested harness invocation, another plugin's tooling) reproduces the failure
# the `unset MEMKIT_CONFIG` below exists to prevent. There is no
# harness-signed signal to check against, so this is recorded rather than
# guarded; what it means in practice is that the README may not claim more
# independence from the environment than "not from YOUR shell's MEMKIT_CONFIG".
#
# THE RULE THIS ENFORCES, exactly: no rung reads a path inside the payload
# TREE. That is why there is no rung reading a `memkit.json` beside the
# wrappers — a plugin install is a clone of a pinned commit, so a file in the
# tree is a file the repo can ship, and a config decides which directories an
# every-prompt hook reads and which binary it exec's.
#
# It is deliberately NOT the stronger "nothing the payload carries can answer
# this". Rung 2's directory is harness-owned but payload-WRITABLE — memkit's
# own hook writes `trust.json` there — so a release could write
# `$CLAUDE_PLUGIN_DATA/memkit.json` on one prompt and be honoured by every
# later, clean release. The escalation over "a malicious payload already runs
# code" is persistence and laundering, and it is real; what makes it tolerable
# here is that nothing in this build writes that file. The check that would
# make it detectable — refusing, or recording a distinct outcome, when that
# file exists with no init-journal entry claiming authorship — belongs with
# `init` in U3, which is the only thing that will ever legitimately write it.
#
# ABSOLUTE, on every rung, for the same reason the interpreter must be: a
# relative candidate is resolved against the wrapper's CWD, which under the
# harness is whatever directory the session stands in. An adopter who typed
# `--config memkitConfig=memkit.json` at install would otherwise have every
# repository they later open hand the every-prompt hook its own `memkit.json`,
# naming both the store roots whose contents are injected and the binary that
# runs. The manifest asks for an absolute path; this is what enforces it.
#
# Nothing found is not an error: the wrapper goes on to run the hook with no
# config, which is inert by construction — no stores, no pointers, exit 0.
memkit_resolve_config() {
    if [ -n "${CLAUDE_PLUGIN_OPTION_MEMKITCONFIG:-}" ]; then
        _candidate=$(memkit_expand_home "$CLAUDE_PLUGIN_OPTION_MEMKITCONFIG")
        case $_candidate in
            /*) ;;
            *) _candidate="" ;;
        esac
        [ -n "$_candidate" ] && [ -f "$_candidate" ] && {
            printf '%s\n' "$_candidate"
            return 0
        }
    fi
    if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
        _candidate=$(memkit_expand_home "$CLAUDE_PLUGIN_DATA")/memkit.json
        case $_candidate in
            /*) ;;
            *) _candidate="" ;;
        esac
        [ -n "$_candidate" ] && [ -f "$_candidate" ] && {
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
# The extraction is a grep for a string key rather than a JSON parse, because
# the only thing available to parse with at this point is the interpreter being
# looked for. Two consequences, and the guard below is written for them rather
# than around them:
#
#   - The pattern cannot express "top level". A `"interpreter"` nested in a
#     store or a root entry matches, and which of several occurrences wins is
#     an artifact of the file's line breaks — first on a multi-line file, last
#     on a minified one. So the value is what gets checked, not the key.
#   - It runs before anything has established the file is JSON at all, so a
#     corrupt config still gets to name a binary. That is a lower bar than the
#     stores are held to, which a real parse admits, and it is stated rather
#     than fixed: validating shape here would mean writing a JSON parser in
#     POSIX sh to decide which python to run.
#
# Said out loud when a recorded value is present and not honoured. Silence
# here is the wrong answer: the install goes on working, under a python the
# adopter did not choose — on a stock mac, 3.9.6 rather than the 3.12 they
# recorded — and no surface in this build reports the resolved interpreter, so
# there is nowhere else the difference could show up.
#
# stderr, because stdout on the hook path is the injected block. Doctor runs
# these wrappers directly and captures it; in a live session it reaches the
# harness's debug log.
memkit_interpreter_refused() {
    printf '%s\n' \
        "$MEMKIT_SELF: the config records \"interpreter\": \"$1\", which $2." \
        "Falling back to the python3 on PATH; retrieval is unaffected." >&2
}

# ABSOLUTE, or it does not count. A slashless or relative value is two
# different files depending on who resolves it: `[ -x ]` below tests it against
# the wrapper's CWD — a directory chosen by whoever launched the harness — while
# `exec` searches PATH for a slashless word and the CWD for a relative one.
# Requiring a leading `/` collapses those into one resolution, which is also
# what closes the measured exit-127 path: a value that passed `[ -x ]` against
# the CWD and was then absent from PATH left `exec` failing on the every-prompt
# hook, i.e. a blocked turn from a config field.
#
# `~/…` is expanded first. The rung above expands it for the config path, so
# the file teaches that a tilde works, and a rule that silently rejected it one
# field over would be a trap of this file's own making.
#
# `/proc/*` and `/dev/fd/*` are absolute and still session-relative: the kernel
# resolves them through the RUNNING PROCESS, so `/proc/self/cwd/python3` names
# an executable in whatever directory the session stands in — the outcome the
# absoluteness rule exists to prevent, restored on the platform the nix channel
# and this repo's CI target. A prefix test rather than a realpath: resolving
# costs a fork on every prompt, and these two trees are the whole of the
# process-relative namespace. A symlink at an ordinary absolute path is NOT in
# this class and is not rejected — that is the adopter's own filesystem, not
# the session's choice.
memkit_config_interpreter() {
    [ -n "$1" ] && [ -f "$1" ] || return 1
    _found=$(sed -n 's/.*"interpreter"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$1" \
        2>/dev/null | head -n 1)
    [ -n "$_found" ] || return 1
    _found=$(memkit_expand_home "$_found")
    case $_found in
        *//* | */./* | */../* | */. | */..)
            # CANONICAL, and this arm has to come first. The two prefixes below
            # are a literal test, and the kernel normalises before it resolves —
            # so `//proc/self/cwd/python3`, `/./proc/…`, `/tmp/../proc/…`,
            # `/dev//fd/3/…` and `/./dev/fd/3/…` all name exactly what the next
            # arm refuses while walking past it. Measured: six spellings
            # admitted, one refused.
            #
            # Still no realpath. Resolving costs a fork on every prompt, and
            # requiring the value to be canonical is the same guarantee for
            # free: a canonical absolute path cannot reach a process-relative
            # tree by another name, because there is only one spelling of it.
            # The cost is that a config recording a legitimate but
            # non-canonical path is refused — which the message says out loud,
            # and which the PATH probe covers.
            memkit_interpreter_refused "$_found" \
                "is not a canonical path, so what it names depends on who resolves it"
            return 1
            ;;
        /proc/* | /dev/fd/*)
            memkit_interpreter_refused "$_found" \
                "the kernel resolves through this process, so it names whatever directory the session stands in"
            return 1
            ;;
        /*) ;;
        *)
            memkit_interpreter_refused "$_found" "is not an absolute path"
            return 1
            ;;
    esac
    printf '%s\n' "$_found"
}

# A recorded value this build will not honour — absent, relative, or naming
# something that is not an executable FILE — falls through to the PATH probe
# rather than ending the resolution. One bad character in a config field must
# not be able to turn a working install inert.
#
# `-f` as well as `-x`, and the pair is not belt-and-braces: `[ -x ]` alone is
# true of a DIRECTORY, whose execute bit means "searchable". So
# `"interpreter": "/opt/homebrew/opt/python@3.12/libexec/bin"` — a PATH entry
# with its last segment dropped, which is how the value gets written by hand —
# passed the guard, skipped the PATH probe, and left `exec` dying 126 on every
# prompt of every session. Measured from all three wrappers.
memkit_resolve_interpreter() {
    _config=$1
    _recorded=$(memkit_config_interpreter "$_config") || _recorded=""
    if [ -n "$_recorded" ]; then
        if [ -f "$_recorded" ] && [ -x "$_recorded" ]; then
            printf '%s\n' "$_recorded"
            return 0
        fi
        memkit_interpreter_refused "$_recorded" "is not an executable file"
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
# all.
#
# UNPINNED, and it is the one hole in the marketplace entry's sha promise: that
# entry pins a released commit so `marketplace add` does not mean "whatever is
# on main", and this spec resolves whatever `main` holds. An adopter on an
# older release can therefore route checker work through a newer checker
# without anything saying so. The store format is what both speak and it has
# not moved, so the skew is tolerable rather than invisible — it is stated here
# and in the README's marketplace paragraph, and the rev pin lands with the
# first tagged release, which is the first commit that can name one.
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
    _base=$1
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
# It names the RUNNING wrapper, because the exit code beside it is that
# wrapper's. Saying `memkit:` from `memkit-recall` sends an agent to look up
# that binary's 4 in the `memkit` table, where 4 means "the subcommand is not
# in this build" — a wrong diagnosis produced by the message's own name.
#
# The reader is doctor, which runs this wrapper directly and captures stderr.
# In a live session it goes to the harness's debug log, because the wrapper
# exits 0 whatever happens — see the exit contract in `memkit-hook`.
memkit_no_interpreter_message() {
    printf '%s\n' \
        "$MEMKIT_SELF: no python3 on PATH and none recorded in the config, so the" \
        "recall hook cannot run. Install python 3.9 or newer, or record an" \
        "absolute interpreter path as \"interpreter\" in the memkit config." \
        "Config in use: ${1:-<none resolved>}"
}
