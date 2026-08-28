# Releasing memkit

A release is **two pull requests**, in order, run as one operation. The reason
is mechanical and worth understanding before you change anything here.

## Why two

A marketplace entry pins a commit sha, and **a commit cannot name its own sha**.
So the pin always names some commit other than the one that writes it.

With squash merges the consequence is sharper than it looks. If one PR both
edits the release state and moves the pin, the pin can only name that PR's
*parent* — and every edit the PR made is therefore in a commit adopters never
install. That is not hypothetical: v0.2.0 shipped `"version": "0.1.0"` in the
manifest an adopter reads, a quoted install rev pointing at the previous tag,
and an `ADMISSION.md` whose file count described a different tree.

Splitting it fixes the ordering:

| | what it does | what it must not do |
|---|---|---|
| **PR A — release state** | versions, ADMISSION's numbers, the marker sweep, docs | touch `marketplace.json`'s `sha` |
| **PR B — the pin** | move `marketplace.json`'s `sha` to A's squash commit | anything else |

A's squash commit — call it **S1** — is then the tree adopters install, and it
already contains the release state. B is one line and ships in the *next*
release's payload, which is exactly where a stale pin does no harm.

**Tag `v<x.y.z>` on S1**, not on B's merge and not on the branch tip. S1 is what
adopters install, what the docs' quoted rev forward-references, and what
hatch-vcs reads to stamp the version. Tagging anywhere else makes the version an adopter
gets differ from the version the tag names.

Run A and B back to back. Between them the pin still names the previous release,
which is the ordinary state this project's docs already describe.

## The checklist

Everything below has been missed at least once.

### PR A

1. **Versions.** `.claude-plugin/plugin.json` and the `marketplace.json` entry.
   A test holds them equal; nothing holds them to reality, so read them.
2. **The tag the docs quote.** `README.md` and `docs/STORE.md` print
   `uvx --from git+…@vX.Y.Z` for the no-install channel. Nothing in `bin/` or
   `src/` names a rev any more — the code fetches nothing — so these are the
   only two quotes left, and a test holds both equal to `plugin.json`'s
   version. Step 1 therefore carries this one: update the version and the test
   names the docs if they lag.
4. **Sweep `(from the next release)`.** `grep -rn "from the next release"`.
   Every behaviour marker becomes a plain statement — the pin is about to carry
   it. Two occurrences are *not* markers and stay: the `## Status` sentence that
   defines the convention, and the paragraph that refers to it. A third is a
   test string in `tests/test_plugin_surface.py` and stays as well.

   **The sweep belongs in A even though A's own pin still lags**, which is the
   one place this reads backwards. A's tree is what B pins, so it is the tree
   adopters install: a marker left in it tells every reader of this release
   that this release does not have the thing they are running. Waiting for the
   pin to agree ships that sentence for the whole release window rather than
   for the minutes between the two merges.

   The marker tests ask the pin, which is the right question everywhere except
   here. `_is_release_state()` in `tests/test_plugin_surface.py` is the
   discriminator — the pinned tree's `plugin.json` version against this tree's,
   equal in every state but a release PR — and the two pin-derived cases
   (`test_a_claim_about_a_command_the_pin_cannot_serve_carries_the_marker`,
   `test_the_docs_mark_what_the_pinned_release_does_not_carry_yet`) switch on
   it: markers required while `main` runs ahead of a current pin, and the
   positive claim — both pages naming the live hook count — required in their
   place here. Both directions have a check.

   Write the sentences keyed to a release rather than to the pin — "on 0.3.0
   and later `Hooks (1)` is a failure; on 0.2.x it is the whole registration
   and healthy" — and they stay true in both windows and at the next release
   without being flipped again. `README.md`'s Quick start, its "Why nothing
   appeared" table, its Install (details) check and `docs/ROLLOUT.md`'s step 2
   are the four sites that used to need that flip.

   **`docs/ADMISSION.md`'s marker is not in this class and stays.** It marks
   the gap between this tree's file count and the pinned tree's, and that gap
   is real for an adopter of this release: their copy carries the previous
   pin, by construction, and the note says which number goes with which tree.
   `test_the_admission_notes_recipe_returns_the_number_it_states` retires it
   when the counts agree, which is after B.
5. **The vocabulary table.** When a release changes an `outcome` value, the old
   one becomes a dated row rather than disappearing — a machine on the previous
   release is writing it now, and that table is what decodes its log. The pin
   requires the row; drop it only when nobody is reading a log from that release.
   Right now that means **`gate:shape`**, which 0.1.x writes for what 0.2.0+
   records as `gate:empty` / `gate:slash` / `gate:short` / `gate:long`.
6. **Re-derive ADMISSION's numbers** — from **this branch's tree**, which is what
   S1 will be and therefore what an adopter installs. `test_the_admission_note…`
   asserts the file count against the working tree for exactly this reason. The
   `.git` figures are hedged on purpose; they vary by git version.
7. **Verify the "What the installed copy is" paragraph** rather than assuming it
   survived. It has needed rewriting at two of three releases, because it is easy
   to write as a description of one particular payload. It must read correctly
   both immediately after a release and mid-window.
8. **Full gates**, including the Linux checks, and the mutation sweep. Expect to
   re-anchor the release-state mutations: the pinned sha, the quoted rev,
   ADMISSION's count, and the vocabulary's dated row.

### PR B

9. **One line**: `marketplace.json`'s `sha` → S1. Run the suite anyway — the
   bidirectional payload/README test reads the object store at that sha, and it
   is the check that the pin carries a complete payload.

### After the merges

10. **Tag S1** and cut the GitHub release.
11. **The consumer bump carries any vocabulary change.** nix-config's tripwire
    reads memkit's source and fails on an `outcome` it has not classified, which
    is the design working. Classify the new names in the same PR that bumps the
    input.
12. **Verify from the outside.** Install from the marketplace into a scratch
    profile — `HOME` and `CLAUDE_CONFIG_DIR` redirected — and confirm the
    behaviour this release was cut for. Every release so far has been reviewed
    that way, and it is the only check that sees what an adopter sees.
