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
# POSIX sh only. The harness runs these with whatever `/bin/sh` is. On the
# Linux workstations these are written for that is dash or bash 5, either of
# which would take more than this file uses — the constraint comes from the
# FLOOR case, a stock macOS where `/bin/sh` is bash 3.2 in POSIX mode. Nothing
# here may need bash 4. Held by shellcheck --shell=sh in CI, on Linux.
#
# THE DEPENDENCY CONTRACT, and it is the empty set: no wrapper and nothing in
# this file runs a command that is not a shell builtin. Not `sed`, not `head`,
# not `grep` — nothing that has to be found on a PATH.
#
# The PATH these run on is composed by the harness, not by the adopter, and it
# is not required to have coreutils on it. Reading the `interpreter` field
# through `sed | head` was enough to break the whole resolution inside a Linux
# nix sandbox, where neither exists: `head: not found`, the recorded
# interpreter silently unread, and the wrapper still exiting 0 — an install
# that answers nothing while reporting healthy. Every external command here is
# one more way for that to happen on a machine nobody tested.
#
# `command -v`, `printf`, `read`, `cd`, `pwd` and `[` are builtins in every
# shell that satisfies the floor above. `tests/test_plugin_surface.py` pins the
# contract twice: it scrapes this file and the wrappers for anything that looks
# like an external command, and it runs each wrapper with a PATH holding
# nothing but a python shim.

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

# --- where the messages below actually go ------------------------------------
#
# THE SINGLE MOST REPEATED DEAD END ACROSS EVERY REVIEW OF THIS PROJECT: the
# refusals in this file are among the clearest text it contains, and in the
# product they are unreachable. Claude Code swallows hook stderr, and
# `claude --debug -p` showed zero hook lines in three separate attempts across
# two walkthroughs. So the same message goes to two places — stderr, which a
# terminal caller and doctor's own probe both see, and a bounded file doctor
# can tail.
#
# WRITTEN ONLY IF THE STATE DIRECTORY ALREADY EXISTS, and that rule is forced
# twice over. `mkdir` is not a shell builtin, so this file could not create it
# without breaking the dependency contract at the top; and an install nobody
# has configured deliberately creates no state directory, so writing one here
# would be a mutation on behalf of somebody who has not consented to anything.
# What it costs is the never-configured case, which is the one state doctor's
# `config-route` can already separate by reading the settings value directly.
# What it buys is the was-working-and-broke case, where the directory is there
# because the install used to serve.
#
# NO TIMESTAMP PER LINE, because there is no builtin clock: `date` is a
# command, and the floor here is a bash 3.2 with no `EPOCHSECONDS` and no
# `printf %(...)T`. The file's own mtime is when it was last written, and
# doctor reports it.
MEMKIT_ERRLOG_NAME=hook-errors.log
# Lines. Bounded the way the trust marker is, so the thing that reports on a
# cache never becomes the thing it reports on. Eviction reads and rewrites the
# whole file, which is affordable only because nothing reaches here unless
# something is already wrong.
MEMKIT_ERRLOG_MAX=200

# The shared derived-state directory, resolved the same way the hook resolves
# it: `$XDG_CACHE_HOME` when it is set to an ABSOLUTE path, else `~/.cache`. A
# relative value is ignored rather than honoured, because the directory an
# every-prompt hook writes into is not the session's to choose.
memkit_state_dir() {
    case ${XDG_CACHE_HOME:-} in
        /*) printf '%s\n' "${XDG_CACHE_HOME}/memory-recall" ;;
        *) printf '%s\n' "$HOME/.cache/memory-recall" ;;
    esac
}

memkit_errlog() {
    _dir=$(memkit_state_dir)
    [ -d "$_dir" ] || return 0
    _errlog=$_dir/$MEMKIT_ERRLOG_NAME
    # A file that exists and cannot be READ skips rotation, so repeated
    # refusals grow it without bound — the file that reports on a cache
    # becoming the thing it reports on. Nothing here can read it to keep the
    # newest half, so the bound is kept the only way left: start again. What is
    # lost is a log this process could not have shown anybody anyway.
    if [ -e "$_errlog" ] && [ ! -r "$_errlog" ]; then
        : > "$_errlog" 2>/dev/null || return 0
    fi
    if [ -r "$_errlog" ]; then
        _lines=0
        # `read` reports failure on a final line with no newline, which is a
        # line like any other — hence the second test, the same one the
        # interpreter scrape uses.
        while IFS= read -r _line || [ -n "$_line" ]; do
            _lines=$((_lines + 1))
        done < "$_errlog"
        if [ "$_lines" -ge "$MEMKIT_ERRLOG_MAX" ]; then
            _skip=$((_lines - MEMKIT_ERRLOG_MAX / 2))
            _seen=0
            _kept=""
            while IFS= read -r _line || [ -n "$_line" ]; do
                _seen=$((_seen + 1))
                if [ "$_seen" -gt "$_skip" ]; then
                    _kept=$_kept$_line'
'
                fi
            done < "$_errlog"
            printf '%s' "$_kept" > "$_errlog" 2>/dev/null || return 0
        fi
    fi
    for _message do
        printf '%s: %s\n' "$MEMKIT_SELF" "$_message" >> "$_errlog" 2>/dev/null \
            || return 0
    done
}

# Say something the adopter needs and cannot otherwise reach. Both channels,
# every time: a message that went only to the file would be invisible to the
# terminal caller, and one that went only to stderr is the state this exists
# to end.
memkit_stderr() {
    # The wrapper's name on the FIRST line and not on the continuations, which
    # is the shape every message here already had: the name is there so an
    # agent looks the exit code up in the right table, and repeating it down a
    # four-line refusal reads as four refusals. The file prefixes every line
    # instead, because there the lines are interleaved across invocations and a
    # continuation with no owner belongs to nothing.
    _first=1
    for _message do
        if [ "$_first" = 1 ]; then
            printf '%s: %s\n' "$MEMKIT_SELF" "$_message" >&2
            _first=0
        else
            printf '%s\n' "$_message" >&2
        fi
    done
    memkit_errlog "$@"
}

# `~/x` as a person types it. The option value is a string the adopter typed
# into an install command, not a shell word the shell ever expanded, so a
# config named `~/.cache/...` arrives with a literal tilde and every rung below
# would silently miss it.
# The one admission rule for every path this library will act on, as an exit
# status: 0 admits, non-zero refuses and names the reason on stdout.
#
# ONE helper because the rule is one rule. The interpreter field had all three
# arms and the config rungs had only the first, which is the wider blast
# radius guarded more weakly: a config decides which directories the
# every-prompt hook reads AND which binary it execs, so on Linux —
# this repo's CI and the nix channel — `/proc/self/cwd/memkit.json` is
# absolute, passes a leading-slash test, and resolves through the running
# process, handing the hook whatever `memkit.json` the session's directory
# holds.
#
# No realpath, deliberately: resolving costs a fork on every prompt, and
# requiring the value to be CANONICAL is the same guarantee for free, because a
# canonical absolute path has exactly one spelling and cannot reach a
# process-relative tree under another.
memkit_path_refusal() {
    case $1 in
        "") printf '%s\n' "is empty" ;;
        *//* | */./* | */../* | */. | */..)
            printf '%s\n' \
                "is not a canonical path, so what it names depends on who resolves it"
            ;;
        /proc/* | /dev/fd/*)
            printf '%s\n' \
                "the kernel resolves through this process, so it names whatever directory the session stands in"
            ;;
        /*) return 1 ;;
        *) printf '%s\n' "is not an absolute path" ;;
    esac
    return 0
}

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
        if _why=$(memkit_path_refusal "$_candidate"); then
            memkit_stderr \
                "the memkitConfig option names \"$_candidate\", which $_why." \
                "Ignoring it; this install will behave as if no config was given."
            _candidate=""
        fi
        # A path that is merely WRONG passes every shape rule above, so
        # without this the option that was set and the option that was never
        # set produce the same silence and the same `config: none` — and the
        # adopter who typed the path is the one person who can be certain a
        # config exists. Said only for this rung: rung 2's file is absent on
        # every plugin install until something writes it, which is a state, not
        # a mistake, and a line about it on every prompt would be noise.
        if [ -n "$_candidate" ] && [ ! -r "$_candidate" ]; then
            if [ -e "$_candidate" ]; then
                _why="exists but cannot be read by this process"
            else
                _why="does not exist"
            fi
            memkit_stderr \
                "the memkitConfig option names \"$_candidate\", which $_why." \
                "Ignoring it; this install will behave as if no config was given."
            _candidate=""
        fi
        [ -n "$_candidate" ] && [ -r "$_candidate" ] && {
            printf '%s\n' "$_candidate"
            return 0
        }
    fi
    if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
        # The DIRECTORY is what the rule is about, checked before the
        # basename is appended: `/proc/self/cwd` and `/proc/self/cwd/memkit.json`
        # are the same admission question.
        _candidate=""
        memkit_path_refusal "$(memkit_expand_home "$CLAUDE_PLUGIN_DATA")" >/dev/null \
            || _candidate=$(memkit_expand_home "$CLAUDE_PLUGIN_DATA")/memkit.json
        [ -n "$_candidate" ] && [ -r "$_candidate" ] && {
            printf '%s\n' "$_candidate"
            return 0
        }
    fi
    return 1
}

# The interpreter, preferring the one the config recorded.
#
# A lookup over the session's PATH hands the process that reads every prompt to
# whatever direnv/mise/venv shim the launching shell happened to carry, and the
# nix channel pins its interpreter absolutely — the plugin channel must not
# silently drop that guarantee. So init records what it resolved, this prefers
# it, and what it falls back to is a fixed list of absolute paths rather than a
# search.
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
    memkit_stderr \
        "the config records \"interpreter\": \"$1\", which $2." \
        "Falling back to a pinned system python; retrieval is unaffected."
}

# The pinned fallback, for an install whose config records no interpreter — a
# fresh one, before `memkit init` has written the field it writes.
#
# ABSOLUTE PATHS, NOT A LOOKUP. `command -v python3` reads the session's own
# PATH, which a checkout steers through direnv, a checked-in venv or a
# `node_modules/.bin`; what came back was exec'd on every prompt, before any
# rule in this package existed to have an opinion about it. There is no way to
# filter that lookup correctly here: POSIX sh under this project's own
# zero-external-command rule has no `realpath`, so a filter is a string prefix
# test against a logical `$PWD`, and its success path — every entry rejected —
# prints the empty string, which POSIX reads as the current directory.
#
# So this is an ALLOW-LIST, and its incompleteness is a support ticket rather
# than a vulnerability: an install it does not cover REFUSES, visibly, and the
# refusal names the repair — record an absolute `interpreter` in the config,
# which is what `memkit init` does unprompted. The alternative failure is
# silence plus somebody else's python.
MEMKIT_SYSTEM_PYTHONS="/usr/bin/python3
/opt/homebrew/bin/python3
/usr/local/bin/python3
/run/current-system/sw/bin/python3
/nix/var/nix/profiles/default/bin/python3"

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
    _found=""
    # `read` reports failure on a final line with no newline, which is a line
    # like any other — hence the second test.
    while IFS= read -r _line || [ -n "$_line" ]; do
        _rest=$_line
        # Every occurrence on the line, left to right, until one parses as a
        # field. The key appearing inside some other field's VALUE is then a
        # line that still yields the real one rather than a line that is
        # skipped.
        while :; do
            case $_rest in
                *'"interpreter"'*) _rest=${_rest#*'"interpreter"'} ;;
                *) break ;;
            esac
            _value=$_rest
            while :; do
                case $_value in
                    [[:blank:]]*) _value=${_value#?} ;;
                    *) break ;;
                esac
            done
            case $_value in
                :*) _value=${_value#:} ;;
                *) continue ;;
            esac
            while :; do
                case $_value in
                    [[:blank:]]*) _value=${_value#?} ;;
                    *) break ;;
                esac
            done
            case $_value in
                '"'*) _value=${_value#\"} ;;
                *) continue ;;
            esac
            case $_value in
                *'"'*) _found=${_value%%\"*} ;;
                *) continue ;;
            esac
            break
        done
        if [ -n "$_found" ]; then
            break
        fi
    done < "$1"
    [ -n "$_found" ] || return 1
    _found=$(memkit_expand_home "$_found")
    if _why=$(memkit_path_refusal "$_found"); then
        memkit_interpreter_refused "$_found" "$_why"
        return 1
    fi
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
    for _candidate in $MEMKIT_SYSTEM_PYTHONS; do
        if [ -f "$_candidate" ] && [ -x "$_candidate" ]; then
            printf '%s\n' "$_candidate"
            return 0
        fi
    done
    return 1
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
#
# The tried paths go in as one message EACH, and the splitting is the point.
# The list is newline-separated, so interpolating it whole put five physical
# lines into a single message — and the error log's rule is that every line
# carries the wrapper's name, so four of them arrived owned by nothing. That
# happened only where this message fires at all, which is a machine none of
# the five paths exists on: a NixOS install, and this repo's own Linux build.
# One path per line is also the shape the reader wants, since the question
# they are answering is whether their python is on the list.
memkit_no_interpreter_message() {
    # shellcheck disable=SC2086  # unquoted so each path becomes its own word
    memkit_stderr \
        "no interpreter is recorded in the config and none of the pinned" \
        "system paths exists, so the recall hook cannot run. Record an" \
        "absolute python 3.9-or-newer path as \"interpreter\" in the memkit" \
        "config — \`memkit init\` writes that field for you. A python found" \
        "only through this session's PATH is deliberately not used: that" \
        "lookup is one a checkout steers, and what it returns would run on" \
        "every prompt. Pinned paths tried:" \
        $MEMKIT_SYSTEM_PYTHONS \
        "Config in use: ${1:-<none resolved>}"
}
