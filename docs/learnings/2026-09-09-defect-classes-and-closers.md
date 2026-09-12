# Defect classes found while adopting harness auto-memory, and what closes each

Written 2026-09-09. Nine classes of defect that came up more than once while this repository
adopted the harness's auto-memory store. Each is a mechanism plus a tripwire: the mechanism makes
the next instance impossible by construction, the tripwire makes it a CI failure instead of a
review finding. Instances are named by the surface they were found on.

## 1. A harness path rendered into output

Instances: an auto-memory remedy that named the user's home directory; four sites that capped a
rendered path for display; a decomposed `$HOME` that the cap measured in one normal form and the
comparison in another; a settings path the parser's own message echoed a second time, uncapped —
the same class opened by the fix meant to close it, four times.
Mechanism: the doctor auto-memory row reports counts, presence and which scope decided; remedies
name files by role, never by rendered path. Where a path must be rendered, exactly one seam turns a
path into a string: it normalizes home and candidate the same way (realpath, NFC, case-fold where
the filesystem folds), redacts, then caps.
Tripwire: an AST assertion over `cli_doctor.py` that no display cap and no path interpolation
exists outside the seam; the band test runs with production caps on two home lengths and a
decomposed home.

## 2. Judge here, act there

Instances: in the shape tool, the directory judged is not the one written into; in `cli_init.py`,
the escape guard's resolution discarded; in the recall hook, the spelling asked about is not the
spelling scanned.
Mechanism: a guard returns the handle the caller then acts on; directory descriptors carry from the
judgment to the write (`dir_fd`, `O_NOFOLLOW`, `O_EXCL`); nothing is re-derived from a path string
after it was judged.
Tripwire: a lint that no guard call stands alone as a statement, over every module that judges
paths.

## 3. An exception set narrower than the syscall's

Instances: `ValueError` on a NUL byte; an `os.dup` above the `try` that guards it; an unguarded
`..` open; an unguarded write to stdout; a raw errno reaching the caller; `RecursionError` on a
deep tree.
Mechanism: one contract boundary per entry point maps the whole family (`OSError`, `ValueError`,
`RecursionError`, `UnicodeError`) to a one-line exit 2; interior code never catches for the
contract.
Tripwire: one property test per tool that feeds hostile trees (NUL names, FIFOs, symlink loops,
unreadable ancestors, an exhausted descriptor table, a vanishing parent) to the entry point and
asserts exit is 0 or 2 and stderr holds no traceback.

## 4. A sentence names something the process is without

Instances: a remedy naming an unset variable, twice; a remedy that sets the value the process
already holds; a remedy naming the wrong file; a remedy naming the directory instead of the file
inside it.
Mechanism: remedies are rendered from the same state object that decided the verdict, never from a
template with free variables.
Tripwire: enumerate the reachable states (the row already walks five branches) and assert for each
that the remedy's named file exists in that state and its named value differs from the held one.

## 5. A silent swallow that reads as "nothing here"

Instances: a per-entry read that failed without saying so; the same one level down, per file; a
walk that drops a whole project because one name will not stat; a vanished `projects/` reported as
a healthy empty machine.
Mechanism: any read that can fail to look returns a distinct "could not look" that propagates to the
row; an empty collection is never the answer to a failed read.
Tripwire: a lint that no `except OSError` body is a bare `pass`, `continue`, `return []` or
`return False`, allowlisted only by a comment naming the row that discloses it.

## 6. Guards wider than their probes

Instances: two `if`-then-exit guards of 109 and 98 lines with no probe spanning either; eleven
probes narrower than the guard they were written for; zero probes over `settings_scopes` out of the
420 the corpus held then; every test that lifted the display caps; the admission-notes probe
anchor-broken for ten commits, because the anchors test is scoped to the closure modules and the
full sweep ran nowhere in CI.
Mechanism: tests run with production constants unless marked as being about the constant; the
mutation sweep runs in CI over the full corpus, 662 probes over 14 modules measured on this tree
with `tools/mutation_sweep.py --list`.
Tripwire: a lint that a guard over a line budget needs a spanning probe or a split; the ratchet keyed
by content hash, not ordinal (the ordinal identity drifted under an insertion at 0bc9a07).

## 7. Prose no extractor reads

Instances: four claims on the adoption pages pinned by nothing; two vacuity-table anchors that no
longer matched the text they named; hand-copied file counts and sizes in `docs/ADMISSION.md`; the
README frame sizes; the probe total in `CHANGELOG.md`.
Mechanism: numbers are generated from their recipe and the page fails when it differs; the
page-execution machinery in `tests/test_plugin_surface.py` stays the way a shell block is proven.
Tripwire: a lint listing sentences that make a checkable claim without an anchor in the vacuity
table.

## 8. A module frozen across parallel branches

Instances: `harness_memory.py` frozen on two branches at once, so the silent-swallow fix could only
disclose and the shape tool's per-file fix had to wait for the merge; the hook module's display
helpers blocking a doctor fix.
Mechanism: merge early so the freeze ends; the shared helpers become a small module with one owner
and a stable interface.
Tripwire: none needed once merged — the class is a coordination artifact.

## 9. A recognizer that enumerates spellings instead of asserting the outcome

Instances: an extractor reading a command's flags where the question is where the command reaches;
a matcher anchored to the `run:` line, so a block scalar reds a gate that is intact; an allowlist of
argv tokens that cannot be spelled two ways without a red.
Mechanism: assert the outcome the sentence claims, with no allowlist of accepted spellings.
Tripwire: a probe class the corpus lacks — a benign edit to the surface under test stays green, so
a recognizer that accepts only one spelling fails its own probe.

## Where each closer lands

1 and 4 — the path-free doctor row, pinned by
`tests/test_doctor.py::test_the_auto_memory_row_renders_no_path`, and at the shipped caps by
`test_the_auto_memory_row_names_no_path_at_the_shipped_caps`.

2 — `tests/test_packaging.py::test_every_guard_call_uses_the_value_it_returns`, one parametrized
lint over every path-judging module.

3 — the contract boundary in `tools/harness_shape.py`, held by
`tests/test_harness_shape.py::test_hostile_trees_exit_zero_or_two_with_no_traceback` and
`test_the_two_walks_answer_the_same_way_about_a_hostile_tree`.

5 — `tests/test_init.py::test_no_oserror_handler_swallows_silently`, parametrized over the three
modules that hold the class, printing the allowlist it accepted.

6 — `tests/test_packaging.py::test_every_guard_in_the_auto_memory_closure_has_a_probe`, keyed by the
guard's own text, with `test_a_guard_too_wide_for_one_probe_has_one_that_empties_it` beside it, and
the `mutation-sweep` job in `.github/workflows/check.yml` pinned by
`tests/test_plugin_surface.py::test_the_mutation_sweep_gate_runs_the_whole_corpus_and_asserts_its_outcome`.

7 — `tests/test_plugin_surface.py::test_every_checkable_claim_in_the_adoption_section_has_an_anchor`
over the adoption section.

8 — closed by the merge itself; nothing stays frozen once the branches are one tree.

9 — the outcome assertion in
`tests/test_plugin_surface.py::test_the_mutation_sweep_gate_runs_the_whole_corpus_and_asserts_its_outcome`,
and the benign-edit probe
`tests/test_plugin_surface.py::test_the_whole_suite_gate_survives_a_benign_rewrite_of_the_step`.
