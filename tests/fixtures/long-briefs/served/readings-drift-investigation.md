# Brief: find out why the line-two readings drifted last quarter

This is an investigation, not a repair. Nobody is asking you to fix line two.
They are asking why the numbers it produced between April and June cannot be
reconciled with the numbers from the same units in the two quarters either side,
and whether the April-to-June numbers should be retained or discarded.

## Background

Line two runs six units through the same fixture, one after another, on a fixed
cycle. Each unit reports a value; the values are averaged across the six and the
average is what goes into the quarterly summary. For eleven quarters the average
sat inside a band of about four counts. In April it stepped out of that band by
roughly nine counts and stayed there until the last week of June, when it
stepped back in and has been inside it since.

A step that arrives and leaves is not drift in the ordinary sense and that is
what makes it interesting. Wear produces a slope, not a step. A step says
something changed and then changed back, or something changed and something else
compensated for it later.

## What was going on in that window

Three things happened in or near the window and any of them could be the cause:

- The line-two fixture was rebuilt in the last week of March, one week before
  the step. The rebuild replaced the clamp assembly and nothing else.
- The units on line two were updated in the first week of April. This was a
  routine push and the same push went to lines one and three, neither of which
  stepped.
- The room's environmental control was serviced in May, mid-window, and the
  logged temperature band narrowed by about a degree afterwards. The step did
  not move in May, which argues against this one but does not eliminate it.

The last week of June, when the step went away, has nothing logged against it at
all. That gap is the most interesting fact in this brief and it is where I would
start: something happened and was not written down, and the shift roster will
tell you who was in the room.

## The obvious answer and why I do not believe it

The obvious answer is the fixture rebuild, because it is a week before the step
and it is a mechanical change to the thing being measured. I do not believe it,
for two reasons. The clamp assembly was replaced like for like from stores, and
the rebuild was witnessed and signed. And a clamp problem produces a spread
across the six units, because they do not sit in the fixture identically; what
we have is all six moving together by about the same amount, which is the
signature of something common to all six rather than something about the
fixture.

All six moving together points at the units themselves, which points at the
April push. That is also convenient and I would like you to be suspicious of it
for exactly that reason, since lines one and three took the same push and did
not step.

## What I want you to establish

Whether a unit on line two, after the April push, was reporting values that
still corresponded to what was in front of it. Not whether it reported
consistently — it plainly did, the step is stable for eleven weeks — but
whether the correspondence between what the sensor sees and what the unit
reports survived the push intact.

There is a stored per-unit table that establishes that correspondence, written
at the bench when the unit is first brought up. If a software update on those
units discards or resets that table, then every unit on the line reports against
a default correspondence rather than its own, all six move together, and the
size of the move is whatever the average per-unit correction happened to be.
That would fit every fact above, including the ones the fixture theory does not
fit.

If that is what happened, the last week of June is somebody at the bench putting
the tables back, working from the commissioning records, and not writing it down
because it took an afternoon and did not feel like an event.

## What to deliver

- A yes or no on whether the April push preserved the per-unit tables, with the
  evidence — a unit pulled from line two and read out is worth more than a
  release note.
- If the answer is no: the size of the correction each unit lost, and whether
  that accounts for the nine counts. It does not have to account for all nine
  for the theory to be right, but it has to account for most of them.
- A recommendation on the April-to-June numbers. Discarding a quarter is
  expensive and reconstructing one from a known offset is worse if the offset is
  only approximately known, so say which and say why.
- Whatever you find out about the last week of June, including if the answer is
  that nobody remembers.

Do not change anything on line two while the investigation is open. If you need
a unit off the line to read it out, take it off and say so in the log, but the
line keeps running as it is until the recommendation is written.
