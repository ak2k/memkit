# Brief: re-slot the north warehouse before the peak season

The north warehouse is slotted the way it was when it opened, which was for a
catalogue half the current size and a very different order profile. Pickers walk
too far, the fast movers are spread across three aisles, and the two busiest
aisles are the two narrowest. We have eleven weeks before peak and we are going
to fix as much of that as eleven weeks allows.

## What we know

The order data is good. Two years of line-level history, clean, with the
seasonal shape visible in it. Whatever we do should be argued from that rather
than from anybody's sense of what moves.

The physical constraints are less well documented. Aisle widths are on the
original drawing and the drawing does not reflect two of the racking changes
made since. Somebody needs to walk it with a tape before any plan is drawn on
top of the old numbers.

Roughly fifteen percent of the catalogue accounts for something like seventy
percent of the picked lines. That ratio is from a rough cut somebody did last
year and it should be redone properly, but if it is even approximately right it
says most of the benefit is available from moving a small number of items.

## Constraints

- No shutdown. The warehouse ships every weekday through the whole window, so
  the move happens in evenings and at weekends, aisle by aisle, and every aisle
  has to be pickable on the next working morning.
- The heavy and bulky items cannot move to the mezzanine and cannot move to the
  top level anywhere. That constraint has removed two of the obvious plans
  already.
- Two product groups have to stay together for regulatory reasons. They are
  tagged in the item master, mostly correctly; assume the tagging is about
  ninety percent right and check the exceptions by hand.
- The pick-face labels are printed in-house and there is one printer. Ten
  thousand labels through one printer is a schedule item, not a detail.

## Approach I expect

Start from the order history, not from the current layout. Work out where the
fast movers should be, then work backwards through the constraints to something
achievable in the window. It is very easy to start from the current layout and
end up with a series of local improvements that add up to less than they cost,
because every one of them is negotiated against what is already there.

Sequence the moves so that the biggest wins land first. If we run out of window
we should run out having done the valuable part, not having done a third of a
plan that only pays off complete.

## Deliverables

- The analysis, with the movement classes and the proposed slotting, in a form
  the warehouse manager can read.
- A move plan by aisle and by evening, with the label printing on the same
  timeline.
- A measurement: what we expect walking distance per order to be before and
  after, and how we will actually check it rather than assume it. If we cannot
  measure it we cannot tell next year whether this was worth doing.
- A list of what we chose not to do and why, because the same question will be
  asked again next spring and the reasoning should not have to be reconstructed.

## Risks I already see

The evening moves depend on the same team that picks during the day. There is a
limit to how many evenings that team can do before the daytime error rate goes
up, and the error rate going up during peak is a far worse outcome than a
suboptimal layout. Build the plan around a sustainable number of evenings and
say what that number is.

## The thing I do not want

I do not want a proposal that depends on new equipment. There is a version of
this project that starts with conveyors and ends with a capital request, and it
is a perfectly reasonable project for some other year. This year the answer has
to be achievable with the racking, the trucks and the people we have.

If the analysis genuinely says the layout cannot be fixed without equipment, say
so, but say it with the numbers attached and say it early, because that changes
what we do with the eleven weeks rather than being a footnote at the end of them.