# Brief: build the acceptance procedure for rebuilt gearboxes

We send gearboxes out for rebuild and take them back on the vendor's word plus a
ten-minute run on the test stand. Two of the last nine came back with a fault
that the ten-minute run did not catch and that showed up within a month in
service. We are not going to change vendors over two out of nine, but we are
going to stop accepting on a ten-minute run.

Your job is to write the acceptance procedure we will use instead, and to run it
against the two units currently in the receiving bay so we know it catches what
we already know is there.

## The two failures we know about

Both were the same complaint from the field: excessive backlash measured at the
output, showing up between two and five weeks after installation, on a drive
that was quiet on the stand.

The first one came back and was stripped here. The finding was in the shim
stack — the stack had been reassembled to the pre-rebuild dimension rather than
to the dimension the new bearings wanted, and the difference is small enough that
it does not make a noise on a stand and large enough that it opens up under load
in a few weeks.

The second one was returned to the vendor before anybody here opened it, which I
regret, because we now have one data point and one anecdote where we could have
had two data points.

## The tempting wrong answer

Every time this complaint comes in, the first response from the floor is to look
at the chain. It is the visible thing, it is the thing that can be adjusted
without opening anything, and adjusting it does make the measured backlash
change, which is what makes it so convincing. It is also, on both of the units
we have any real evidence about, not the cause. Tensioning the chain masks the
symptom for a few days and puts a load into a drive that was not designed to
carry it.

The procedure you write needs to be arranged so that the chain is eliminated
before anybody has the chance to adjust it, rather than after. Order the steps
so the measurement that distinguishes the two cases comes first. If the
procedure begins with something that can be tweaked, it will get tweaked.

## Constraints on the procedure

- It has to run on the existing stand. We are not buying a loaded stand this
  year and a procedure that requires one is a procedure that does not get used.
- It has to fit in an hour per unit including setup, or receiving will not run
  it and units will start going straight to the shelf again.
- It has to produce a number, not a judgement. "Sounds right" is what we have
  now. A pass or fail against a recorded figure is what lets receiving reject a
  unit without an argument, and an argument at receiving is how the second
  failure went back to the vendor unopened.
- It has to say what to do when a unit fails: who is told, where the unit sits,
  and what goes back to the vendor with it. A failing unit with nowhere to go
  ends up back in the queue.

## Thermal state

Whatever measurement you land on, say explicitly what thermal state the unit is
to be in when it is taken. A gearbox that has just come off the stand and one
that has sat overnight do not read the same, and the difference is comfortably
larger than the tolerance we would be judging against. Pick one state, say how
long the unit has to have been in it, and make the sheet ask for the time since
the last run so a reading taken in the wrong state is visible rather than
invisible.

## Deliverables

- The procedure, one page, written for somebody in receiving rather than for an
  engineer.
- The record sheet it fills in, with the fields for thermal state and elapsed
  time.
- Results from running it against both units in the receiving bay, with your
  assessment of whether it would have caught the two known failures. If it would
  not have, say so plainly — a procedure adopted on the assumption that it
  catches the known cases and does not is worse than the ten-minute run, because
  it comes with confidence attached.
- A short note on what it would take to do this under load, and what that would
  cost, so the loaded-stand conversation can happen next year on numbers.

## Timing

Two weeks. The receiving bay units are not urgent and there is nothing else
waiting on this, so take the time to run the procedure more than once on the
same unit and see whether it repeats. A procedure that produces a different
number on Tuesday than it did on Monday is not a procedure, and that is the
failure mode most likely to bite here.
