# Brief: a repeatable rig for measuring backlash at the output sprocket

This is a build brief with an investigation folded into it. We need a rig that
measures backlash at the output sprocket repeatably, and we need it because the
measurements we currently take are not repeatable enough to argue with a vendor
about. Read all of it — the constraints in the second half are the reason the
obvious design does not work.

## Why we need it

Twice in the last year a drive has come back from rebuild, passed on the stand,
and been reported from the field with excessive backlash within a month. Both
times the argument with the vendor stalled at the same place: our number and
their number did not agree, neither of us could show that our number was
repeatable, and the conversation became a negotiation instead of a measurement.

The one unit we stripped ourselves had a shim stack that had been reassembled to
the pre-rebuild dimension rather than to the dimension the new bearings needed.
That is a real finding and it is the finding I expect this rig to be able to
detect from the outside, on the stand, before a unit is accepted.

## What the rig has to do

Apply a controlled reversing torque at the output and measure the angular lost
motion, with enough resolution to see a difference that currently only shows up
after weeks in service, and with enough repeatability that the same unit
measured twice on the same afternoon produces the same number to within a small
fraction of the tolerance we would judge against.

Repeatability is the requirement. Resolution is easy to buy and repeatability is
not, and a rig with beautiful resolution and poor repeatability is exactly the
instrument we already have — it produces a number, the number is precise, and
the number moves when nothing has changed.

## The obvious design and why it does not work

The obvious design clamps the input, hangs a dial on the output, and rocks the
output by hand between two stops. It is what we do now with a magnetic base and a
lever, and it fails for three reasons that the rig has to address:

- The clamp on the input is not rigid enough. Some of what we measure is the
  clamp winding up, and how much depends on how the clamp was set that morning.
- Rocking by hand does not apply the same torque twice. The measured lost motion
  depends on how hard the operator pushes, and different operators get different
  numbers on the same unit, reliably enough that we can tell who took a reading
  by looking at it.
- The dial reads at one point on the circumference, so it sees runout as lost
  motion. On a unit with any eccentricity at all, the number depends on where
  the output happened to be parked when the reading started.

Fix all three or the rig is the current instrument with a nicer frame.

## Constraints

- It has to accept the three drive sizes we actually handle, without a change
  part per size if that can be avoided. A rig with three sets of tooling has two
  sets missing when you need them.
- It has to live in the corner by the stand and be usable by one person. If it
  needs two people it will be used when two people are free, which is never in
  the last week of a month.
- Setup has to be under ten minutes. The acceptance procedure this feeds into
  has an hour per unit and the measurement cannot eat most of it.
- It must record to the shared log automatically. A number written on a sheet is
  a number that gets transcribed, and a transcribed number is one we cannot put
  in front of a vendor.

## The thermal question

This is the constraint that most changes the design and it is the one that gets
forgotten, so it is in its own section.

A drive that has just run and a drive that has sat overnight do not measure the
same. The difference is larger than the tolerance we would be judging against,
which means a rig that does not control for it produces numbers that cannot be
compared to each other, let alone to a vendor's.

Measure cold. A warm unit reads short, and short is the direction that makes a
bad unit look acceptable, which is the worst direction for an acceptance test to
be wrong in. Define cold as a stated number of hours since the last run, put the
elapsed time on the record, and make the rig refuse to record — or at least flag
the record — when the stated interval has not passed. A field on a form that can
be left blank will be left blank.

## What I want back, and in what order

1. A one-page concept with the three failure modes above addressed explicitly.
   Not a drawing yet. I want to argue about the approach before anybody spends a
   week on a model.
2. Once the concept is agreed: a build, using stock sections and bought-in
   components wherever possible. This does not need to be pretty and it does
   need to be stiff.
3. A repeatability study before it is used in anger. Same unit, ten readings,
   two operators, both thermal states. Publish the spread. If the spread is not
   comfortably inside the tolerance we would judge against, the rig is not
   finished and we should not start quoting numbers from it.
4. The two units in the receiving bay measured on it, with the results set
   beside what the current lever-and-dial method says about the same units.

## What would make this a failure

Producing a rig that gives a confident number nobody has shown to be repeatable.
We already have one of those. The whole value of this project is being able to
say, to somebody who does not want to hear it, that the number is the unit and
not the instrument or the operator or the time of day — and that claim rests
entirely on step 3, which is the step most likely to be skipped because by then
the rig will exist and everyone will want to use it.

Budget is not the constraint here. Time is not especially the constraint either.
Do it properly and do step 3 before anybody sees a number they like.

## A note on the vendor conversation

Do not talk to the vendor about any of this until step 3 is done and published.
The moment we quote a number we cannot defend, the conversation goes back to
where it was last year, and we do not get a second first attempt at it. When we
do go back, go with the repeatability study attached and the two receiving-bay
units measured, and let the numbers carry it.