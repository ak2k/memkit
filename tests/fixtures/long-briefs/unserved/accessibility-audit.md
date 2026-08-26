# Brief: accessibility audit of the public booking flow

We have a legal deadline in the spring and a booking flow that has never been
audited. Your job is the audit, not the remediation, though I expect the two to
overlap where a fix is obvious and cheap.

## Scope

The public booking flow only: landing, search, results, seat selection, details,
payment, confirmation. Seven screens, plus the two error states that sit between
payment and confirmation. The account area, the help centre and the mobile
applications are out of scope and will be their own piece of work.

Both breakpoints. The desktop layout and the narrow layout are different enough
in the results screen that auditing one tells you very little about the other.

## Standard

WCAG 2.2 at AA. Where a criterion is ambiguous for our pattern, say which
reading you took and why, rather than picking one silently. The auditor who
reviews this in the spring will have their own reading, and a document that
shows its reasoning survives that conversation; one that shows only verdicts
does not.

## Method

Automated scanning first, because it is cheap and it clears the noise, but do
not stop there. The automated tools find roughly a third of what matters on a
flow like this and they find it in a way that overstates the trivial and misses
the structural. Budget most of your time for manual work.

Manual passes I want covered:

- Keyboard only, start to finish, including both error states. Every
  interactive element reachable, every focus state visible, no traps, and a tab
  order that corresponds to the visual order rather than to the DOM order that
  happens to have accumulated.
- Screen reader, on at least two combinations. The seat selection grid is the
  screen I am most worried about; it was built as a canvas with a click handler
  and I do not believe it announces anything useful.
- Zoom to four hundred percent and reflow. The results screen has a filter rail
  that I suspect disappears entirely.
- Colour contrast throughout, including the states nobody checks: disabled
  buttons, placeholder text, the error text on the payment screen, and the focus
  ring against every background it can land on.

## Deliverables

- A findings list, each finding with the criterion, the screen, a reproduction,
  a severity, and a suggested remediation. Severity by user impact rather than
  by how hard it is to fix; the effort estimate is a separate column.
- A summary that a product manager can read in ten minutes and act on, listing
  the five things that matter most.
- A short section on what would have to change in how we build screens for this
  not to recur. The seat grid is not an accident of one sprint; it is what
  happens when a team ships a component with no accessibility review and nobody
  notices for two years.
- The raw automated output as an appendix, so the next audit can diff against it.

## Practicalities

Use the staging environment. Production has real bookings behind it and a
screen-reader pass involves a lot of tabbing through a payment form.

The staging seat map is stale and shows a hall configuration we no longer use.
That does not affect the audit but it will confuse you when the seat numbers do
not match the ones in the ticket examples.

Three weeks. If the seat grid turns out to need a rebuild rather than a fix,
stop and tell me before you write it up in detail — that is a scoping
conversation and not an audit finding.

## Who to talk to

The designer who owns the flow has been here six months and inherited it. She is
keen and she is also the person most likely to hear a findings list as a
criticism of her work, which it is not. Bring her along from the start rather
than presenting at the end.

The front-end lead has opinions about the component library and will want the
audit to become an argument for replacing it. It may well be an argument for
replacing it. Keep that out of the findings document and raise it separately,
because a findings document that carries a platform recommendation gets read as
advocacy and its findings get discounted along with it.

Legal wants a date and a number, not a document. Give them the date and the
number when you have them and do not send them the full audit.
