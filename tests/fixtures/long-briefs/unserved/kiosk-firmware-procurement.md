# Brief: choose a supplier for the visitor kiosks

We are replacing the four visitor kiosks in reception. This is a procurement
exercise, not a design exercise — the flow is settled and signed off, and what
we need now is somebody to build and support the hardware.

## Where we are

Three suppliers have responded to the enquiry. All three can meet the physical
specification and all three have references we can call. On price they are
within about fifteen percent of each other, which is close enough that price
should not decide this.

What separates them, as far as I can tell from the responses, is how they handle
the software over the life of the units. Supplier A pushes firmware updates
centrally and charges an annual fee for it. Supplier B ships the units and
expects us to manage updates ourselves through a console they provide. Supplier
C subcontracts the whole software side to a fourth party we have never heard of
and would not have a contract with.

## What I want established

- What each supplier's update process actually looks like in practice, from a
  reference customer rather than from the response document. Ask specifically
  what happens when an update goes wrong on a Friday afternoon.
- What we are committing to over five years, in money and in our own staff time.
  The annual fee is visible; the staff time is not, and the option that looks
  cheapest is usually the one that quietly costs a day a month.
- What happens at end of life. Two of the three responses are silent on whether
  the units keep working when support ends, and a kiosk that bricks itself when
  a certificate expires is a reception desk with four dead screens in it.
- Whether any of them will accept our accessibility requirements as a
  contractual term rather than a best-efforts statement. This is the one place I
  am prepared to lose a supplier over.

## Constraints

- Four units, installed over a weekend, before the end of the financial year.
- No cloud dependency for the core check-in flow. If the link is down, reception
  must still be able to sign a visitor in and print a badge. Two of the three
  responses are ambiguous about this and it needs a straight answer.
- The badge printers are staying. They are two years old, they work, and nobody
  is buying four more.

## Deliverable

A recommendation with a one-page comparison behind it, and the reasoning written
so that somebody who was not in the meetings can follow it in a year when the
first thing goes wrong. Include the option we are rejecting and why, because
that is the one that gets revisited.

## Timing

Three weeks to the recommendation, then a fortnight for legal to look at the
terms before anybody signs. Do not let the supplier's quarter-end drive our
timetable; two of them have already mentioned it and it is not our problem.

## Who to involve

Reception staff, early and properly. They are the people who will meet every
failure mode, and the last two things we bought for that desk were chosen
without asking them and are both disliked. Sit with them for an hour before you
shortlist, and put their objections in the comparison rather than in a footnote.

IT will want to own the update process because it is the part that resembles
their existing work. That may be right, and it may also be the reason the
staff-time column comes out wrong — ask them for a number and hold them to it in
the comparison rather than accepting an assurance.

Security have a standing requirement about anything on the guest network. Get
their sign-off in writing before the recommendation goes out, not after; the
last project of this shape lost three weeks to a conversation that could have
happened in week one.