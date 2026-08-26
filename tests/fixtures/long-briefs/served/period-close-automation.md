# Brief: automate the period close, without automating the mistake in it

Finance wants the monthly close automated. The manual version takes four days
and two people, most of it spent chasing the same three categories of
discrepancy every month. I have said yes in principle. I want you to scope it,
and I want you to be careful about one thing in particular, which is in the
section headed "the ordering problem" below.

## The current process, briefly

The books are closed on the third working day. Statements arrive from the four
institutions between the first and the fifth, not reliably in that order. The
two-person team works through each account, matches the entries, investigates
what does not match, and posts corrections. When everything ties out, the period
is marked closed and the reporting pack is generated from it.

That is the described process. The actual process, which is what we would be
automating, differs from it in ways the team has stopped noticing, and the
scoping work starts with sitting with them for a close and writing down what
they actually do.

## The ordering problem

Here is the thing I most want you to get right, because it is the thing an
automation project will get wrong by default.

The matching against the statement has to happen before the period is closed,
not after. Once the period is closed, a discrepancy discovered afterwards cannot
be resolved by matching it — the entries are frozen — so it gets resolved by
posting an adjustment into the following period instead. The books tie out, the
adjustment is legitimate, and the audit line that would have shown what actually
happened is gone. All you can see afterwards is that an adjustment was made.

This matters for the automation because the automation's convenient design is
the wrong one. The convenient design closes on schedule and reconciles
afterwards, since that makes the close deterministic and the reconciliation
asynchronous, which is what an engineer wants. It would work, in the sense that
the numbers would be right, and it would quietly convert every discrepancy from
something with a resolution into something with an adjustment.

So the design constraint is: the period does not close until the matching has
run and its exceptions have been dealt with. If a statement is late, the close
waits. If waiting is not acceptable to finance in some month, that is a decision
a person makes explicitly, with the consequence written down, not a default the
system falls into because an institution was slow.

## Scope

In:

- Ingesting the four statement feeds, including the one that is a PDF that
  somebody currently retypes. Especially that one.
- Matching entries automatically, with a clear exception list for what did not
  match rather than a best-effort guess that ties out.
- The exception workflow: who sees it, how it is resolved, what is recorded.
- The close itself, gated as described above.

Out, for now:

- The reporting pack. It works and nobody complains about it.
- Anything touching the general ledger structure. That is a separate
  conversation and it will eat this project if it is allowed in.
- The quarterly and annual processes. Get the monthly one right first; they
  share most of the mechanics and none of the deadlines.

## What I want from the scoping

- A written description of what the team actually does, from watching a close,
  including the parts that are not in the documented process.
- The three recurring discrepancy categories, with numbers: how often, how big,
  how long they take to resolve. If two of the three are trivial and one is not,
  the project is smaller than it looks.
- A design that respects the ordering constraint, with the failure mode called
  out explicitly so that a future maintainer under deadline pressure does not
  quietly reverse it.
- A phased plan. I would rather have the PDF feed and the exception list this
  quarter than the whole thing next year.
- An honest estimate, including the part where the team has to work the manual
  and automated processes side by side for two closes before anybody trusts it.

## One caution

The team is not resistant to this and they are also not going to tell you when
your design breaks something they rely on, because they will assume you know
something they do not. Ask directly, more than once, and show them the design in
their own vocabulary rather than yours.
