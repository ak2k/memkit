# Brief: stage the summer firmware across the bench fleet

You are taking over the bench fleet rollout from the previous shift. Read this
whole brief before touching anything on the rack, because the order of the
steps is the part that goes wrong and the order is not obvious from the runbook.

## What is already true

All eighteen bench units are on the spring firmware. Fourteen of them came back
from the field in March and were re-racked without being opened; the remaining
four have been on the bench since commissioning and have never left the room.
The spring firmware has a known defect in the sampling window that shows up as a
slow drift on long runs, and the summer build fixes it. Nothing else in the
build is interesting to us.

The fleet is wired through the shared harness on rack B. Two units share each
harness leg, so a leg that is pulled takes two units offline whether or not you
meant to touch the second one. The harness map on the wall is a year out of date
and lists the leg assignments from before the March re-rack; do not use it. The
current assignment is in the rack log, which the previous shift updated on
Tuesday.

## What you are doing

Flash all eighteen units to the summer build, one harness leg at a time, and
bring each leg back to a state where its readings can be trusted before you move
to the next leg. That last clause is the whole job. A leg that is flashed and
put back into the run queue without being brought back to a trustworthy state
will produce numbers that look plausible and are wrong, and the run queue does
not distinguish those from good numbers. We found that out in February and it
cost us three weeks of measurements that had to be thrown away.

Work in this order:

1. Take the leg out of the run queue and confirm it has drained. A unit that is
   mid-run when you pull its leg leaves a partial record that the queue will
   retry, which puts the unit back into a run you thought you had stopped.
2. Flash both units on the leg. The flash itself is unattended and takes about
   nine minutes per unit; they can be done in parallel.
3. Bring each unit back to a trustworthy state before the leg goes back into the
   queue. This is the step the previous shift skipped on legs 3 and 7, and those
   two legs are the reason the current run is suspect.
4. Put the leg back in the queue and watch the first two runs to completion.
5. Only then start the next leg.

## What "trustworthy" means here

It means the readings a unit reports correspond to what is actually in front of
it. A unit that reports confidently and wrongly is worse than a unit that
reports nothing, because the queue has no way to tell the first case from a
correct reading, and everything downstream of the queue treats a number as a
number.

There is a stored table inside each unit that maps what the sensor sees to what
the unit reports. That table is per-unit — it is established at the bench when
the unit is first brought up, and two units of the same model do not share one.
Whether the summer flash preserves it is the question you need to answer before
you flash the first leg, not after. Check the release notes, and if the release
notes do not say, treat the answer as no and plan for the longer procedure. The
release notes for the spring build did not say either, and the assumption that
went in its place is what produced the February numbers.

If the table does not survive, each unit needs the zero point established again
at the bench before its leg goes back in the queue. That is a two-person job
with the reference stop and it takes about twenty minutes per unit, so budget
for it rather than discovering it on leg 1.

## Deliverables

- Every unit on the summer build, with the build string recorded in the rack log
  beside the unit's serial.
- For each unit, a note in the rack log saying whether the zero point had to be
  established again and, if so, who did it and against which reference stop.
  There are two reference stops in the room and they do not agree to better than
  a couple of counts, so the one you used is part of the record.
- The two legs the previous shift left in a doubtful state — 3 and 7 — carried
  through the same procedure regardless of what their log rows say. Their rows
  say the flash completed, and that is true and not the thing in question.
- A short note at the end of the rack log saying which runs in the current
  queue were produced by a unit in a doubtful state, so the analysis side can
  drop them rather than reconciling them.

## What not to do

Do not flash all eighteen and then work through the trustworthy-state step in a
second pass. It reads as more efficient and it is how February happened: the
second pass is where the interruptions land, and a unit sitting flashed and
untrusted looks exactly like a unit sitting flashed and fine.

Do not trust the wall map. Do not put a leg back in the queue because its two
runs looked reasonable — reasonable is what the failure mode produces.

Do not start after 1600. The procedure does not have a safe stopping point in
the middle of a leg, and leaving one half-done overnight is how legs 3 and 7
got into their current state.
