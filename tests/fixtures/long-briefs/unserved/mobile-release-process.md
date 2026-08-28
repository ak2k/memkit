# Brief: get the mobile release cadence from six weeks to two

The web product ships continuously. The mobile applications ship every six weeks
and the last three releases have each slipped by a week or more. The gap is
becoming a product problem: features land on the web and wait a month and a half
for their mobile counterpart, and the two experiences have drifted far enough
apart that support has to ask which one a customer is on before they can answer.

I want a two-week cadence, held reliably, within two quarters.

## Why it takes six weeks now

From the outside it looks like the release itself is slow. It is not; the
mechanics of building, signing and submitting take about a day and a half. The
six weeks is everything around it:

- Regression testing is manual and takes eight working days across two people.
  It has grown by roughly a day per quarter for two years as features have been
  added and nothing has ever been removed from the script.
- The release branch is cut, and then work continues on it for two weeks,
  because whatever is not quite finished at cut is finished on the branch. Every
  such change reopens the regression window.
- Store review is unpredictable — usually a day, occasionally five — and the
  process treats the worst case as the plan.
- The release requires a coordination meeting with four teams, which is
  scheduled when all four are available, which is not soon.

## What I think has to change

The regression script is the load-bearing problem and automation is the obvious
answer, but the obvious answer here is a year of work if it is done as a
straight translation of the existing script. Look instead at what the script is
actually protecting against. My guess is that most of it protects against
classes of defect that have not occurred in two years, and a much smaller
automated suite over the paths that actually break would carry most of the value.

The branch discipline is a decision, not a technical problem. If nothing lands
on the release branch after the cut, the regression window closes. That requires
somebody senior to say no, repeatedly, for about three releases, after which it
becomes normal.

The coordination meeting should not exist. Work out what it is for and replace
it with something asynchronous.

## What I want from you

- An analysis of the regression script against the defect history. Which parts
  have ever caught anything, which have not, and what a minimal automated suite
  would need to cover.
- A proposal for the automation, phased so that the first phase pays for itself
  within a quarter.
- The branch policy, written down, with the exception process — there will be
  genuine emergencies and a policy with no exception route gets ignored entirely
  rather than occasionally.
- A plan for the transition, because we cannot stop shipping while we change how
  we ship. I would expect to run one release at the new cadence alongside the
  old process before switching.

## What I do not want

A recommendation to rewrite either application. Both are fine. This is a process
problem wearing a technical costume, and the technical work involved is
substantial but narrow.

## Constraints

The two applications are built by two teams who do not enjoy each other's
company and who have different opinions about almost everything. Whatever you
propose has to work for both, and the fastest way to fail is to design it with
one team and present it to the other.

The store accounts are held by one person who is on leave until the end of next
month. Anything that touches submission credentials waits for her.

## Measurement

Cadence is the headline number and it is not enough on its own — a two-week
cadence achieved by shipping less is not a win. Track the time from a change
merging to that change being available to a customer, which is the number the
product problem is actually about, and report it alongside.

## Timing

Analysis in three weeks. Proposal a week after that. Then we agree the phasing
and you start.