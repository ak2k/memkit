# Brief: rebuild the localisation pipeline for the product copy

We support nine languages and we ship copy changes weekly. The current pipeline
is a spreadsheet, an email thread with an agency, and a developer who pastes the
results back in. It works, badly, and it is the reason the non-English versions
of the product are between two weeks and two months behind the English one.

Your job is to design and build the replacement.

## Why the current thing fails

It fails at the seams rather than in the middle. The agency does good work and
the developer is careful. What goes wrong is everything around them:

- There is no way to tell which strings changed since the last send. So either
  everything goes, which is expensive, or somebody eyeballs a diff, which is
  unreliable, and both happen depending on who is doing it.
- Context does not travel with the string. The agency gets a list of fragments
  with no indication of where they appear, so a word that is a noun in one
  screen and a verb in another comes back translated as whichever the
  translator guessed. We have a standing list of eleven such cases and it grows.
- There is no review step in the target language. Nobody on our side reads the
  German before it ships. When it is wrong we find out from a customer.
- Nothing is versioned. A string that is corrected after a complaint is
  corrected in the product and not in the spreadsheet, so the next send
  reintroduces the original.

## Requirements

- Source of truth in the repository, beside the code, so that a string change
  and the code change that motivated it move together and are reviewed together.
- Automatic detection of what has changed since the last export, with the export
  itself a scripted, repeatable operation rather than a manual one.
- Context attached to every string: where it appears, a screenshot where that is
  feasible, the character budget, and a note for the ones that are genuinely
  ambiguous.
- A review step per language, with a named reviewer, blocking the release of
  that language rather than of the whole product. A late German release should
  not hold up Spanish.
- Full history. Any string, any language, who changed it and when, back to the
  beginning of the new system. We will not backfill the old spreadsheet.
- Fallback behaviour that is explicit: an untranslated string shows the English
  rather than a key, and there is a report of what is falling back, per language,
  visible to the people who care.

## Explicitly out of scope

Machine translation. It comes up every time and it is not what is broken here.
The agency's output is good; the pipeline around it is not. Introducing machine
translation into a broken pipeline produces bad copy faster.

Right-to-left support. We do not ship a right-to-left language and adding the
capability speculatively will cost more than it saves. Design so as not to
preclude it and do not build it.

## Approach

Start with the string extraction and the change detection, because everything
else is downstream of knowing what changed. Get that working and running weekly
before you build anything else, even manually driven — a reliable weekly diff
delivered by hand is more valuable than a half-built system.

Then the context capture, then the review workflow, then the reporting.

## What I want to see first

A one-page design and a working change-detection script, within two weeks. Not a
prototype of the whole thing. The design will be argued about and I would rather
argue about it against something concrete than against a document.

Talk to the agency before you finalise the export format. They have opinions
about what makes a file workable and they have never been asked.

## The people part

The developer who currently does the pasting has done it for three years and
will be relieved to stop. He also knows every exception and workaround in the
current arrangement, and none of it is written down. Get it out of him early,
while he still has a reason to care, rather than at handover.

The head of marketing believes the agency is expensive and will use this project
as an opportunity to reopen that. Do not get drawn in. Build the pipeline so
that changing agencies is a configuration change rather than a rewrite, and let
that be your entire contribution to the argument.