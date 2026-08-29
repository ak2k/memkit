# Brief: rework how inbound support conversations are triaged

Support is drowning, and the shape of the drowning has changed. Volume is up
about a fifth year on year, which is roughly what we would expect from growth,
but time to first meaningful reply has doubled and satisfaction has fallen
further than the reply time alone explains. Something about how work reaches the
right person is broken, and adding people to the current system is going to be
expensive and not fix it.

## What triage looks like today

Everything lands in one queue. Two people rotate through a triage shift, read
each conversation, apply tags, and either answer it or route it to one of six
specialist queues. The tags drive the reporting and, in theory, the routing.

In practice the tags are applied inconsistently, the specialist queues have
wildly different response times, and about a fifth of conversations get routed
more than once. A conversation that is routed twice takes on average four times
as long to resolve as one routed correctly first time — that is from the data,
not an impression — and the customers in that fifth account for most of the
satisfaction decline.

The triage shift is unpopular. It is cognitively heavy, it is interrupt-driven,
and the people who are best at it are the specialists we can least afford to
have doing it.

## What I want you to work out

Why conversations get routed twice. I have three theories and no evidence:

- The taxonomy does not match how customers describe their problems, so triage
  is guessing at which specialist queue a description belongs to.
- The specialist queues have overlapping remits at the edges, so the same
  conversation legitimately belongs to two of them and lands wherever the triage
  person happened to look first.
- Triage is being done too fast, because the queue depth is visible on the
  dashboard and the person doing it feels it.

They are not mutually exclusive and I would guess all three contribute. What I
need is a sense of the proportions, because the remedies are completely
different and we can only afford one this quarter.

## Method

Sample rather than survey. Take two hundred conversations that were routed more
than once, read them properly, and classify why. Two hundred read carefully will
tell you more than two thousand skimmed, and it is a number one person can get
through in a fortnight.

Interview the triage rotation. All of them, individually, and ask what they do
rather than what the process says. The gap between the two is where the answer
usually is.

Look at the specialist queues' own intake. If a queue is quietly bouncing things
back rather than resolving them, that shows up as a re-route and is not a triage
failure at all.

## Deliverables

- The classification of the two hundred, with the proportions.
- A recommendation, singular. I do not want three options with trade-offs; I
  want your best judgement and the reasoning, and I will argue with it if I
  disagree.
- Whatever the recommendation is, an estimate of what it costs to implement and
  what it saves, in hours per week rather than in money, because hours per week
  is what the head of support can act on.
- A note on what we should measure to know whether it worked, chosen before the
  change rather than after. Re-route rate is the obvious one and it is gameable;
  say what else.

## Constraint

Whatever you propose has to work with the tooling we have. We are not changing
the platform this year — that decision is made and it is not yours or mine to
reopen — so a recommendation that requires a different system is a recommendation
we cannot act on.

## Timing and reporting

Three weeks for the sampling and interviews, one week to write. Come to me at
the end of week two with what you have, not because I want a status update but
because that is the point at which the theories usually collapse into one, and
if it collapses into one I would rather spend the third week deepening the
evidence for it than completing a classification we no longer need.

Do not present to the wider team before we have talked. Some of what you find
will be about individuals and it needs a filter before it circulates.