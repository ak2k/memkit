# Brief: plan the rotor swap programme for the autumn outage

We have four machines to take through the autumn outage and eight rotors to move
between them. This is the first time we have done a swap of this scale and the
scheduling is not the hard part — the record-keeping is, and if the
record-keeping goes wrong we find out about it as vibration on a machine that was
fine before we touched it.

## The shape of the programme

Machines A and C get their rotors swapped with each other. Machine B gets the
spare that has been on the stand since spring, and its rotor goes to the shop for
the work that has been deferred twice. Machine D is opened, inspected, and closed
with the same rotor unless the inspection says otherwise, in which case it takes
what is left and the plan changes.

Four machines, two weeks, one crane. The crane is the schedule constraint and it
is already booked, so the plan has to fit around it rather than the other way
round.

## The part that goes wrong

Every rotor carries balancing weights, and those weights belong to the rotor
rather than to the machine it happens to be sitting in. That sounds obvious
written down. It is not obvious at four in the afternoon on the second week when
somebody is reading a record that says "machine A, positions and masses" and
machine A now has a different rotor in it.

Our records are kept per machine, which was fine for thirty years because a rotor
never left the machine it was commissioned in. This outage breaks that
assumption for the first time and the record format has not caught up.

So: before anything is lifted, the existing records have to be re-keyed to the
rotor. Every weight, every position, against the rotor's own serial rather than
the machine's. Where the record does not say which rotor it referred to, that has
to be established from the commissioning file, and where the commissioning file
does not say either, the rotor gets balanced fresh rather than inheriting a
number nobody can attribute.

I would rather balance two rotors unnecessarily than install one carrying
somebody else's correction.

## Why this matters more than it sounds

A rotor installed with the previous occupant's balancing weights is not a
machine that fails on start-up. It runs, it is rough, and it is rough in a way
that reads like a bearing or an alignment problem, so the first week after the
outage gets spent chasing the wrong thing. That is the failure I want the
programme designed to make impossible rather than unlikely.

The housing does not carry an imbalance that follows the machine. Whatever
imbalance there is comes with the rotor and goes with the rotor. Any procedure
that assumes otherwise will produce a plausible and wrong correction.

## What I want in the plan

- The crane sequence, obviously, with the two-day float where it currently sits.
- The record migration: what is being re-keyed, by whom, and finished before the
  first lift rather than in parallel with it. If it is happening in parallel it
  is not happening.
- A per-rotor sheet that travels with the rotor. Physically travels — attached to
  the transport frame — because a sheet in the office is a sheet that is in the
  office when the question is asked on the floor.
- The list of rotors whose records cannot be attributed, with a fresh balance
  budgeted for each. I expect two. If it is more than four, tell me before the
  outage rather than during it, because that changes the shop's loading.
- What happens if machine D's inspection goes badly. One paragraph. The plan
  does not need a full branch, it needs a named decision-maker and a stated
  fallback so that the decision gets made on the day instead of escalated.

## Timing and review

Draft by the end of next week, reviewed the week after, frozen a fortnight
before the outage starts. Nothing changes after the freeze except through me,
and that is not bureaucracy — it is because the per-rotor sheets get printed and
attached at the freeze, and a plan that changes after the sheets are attached
puts a wrong sheet on a rotor, which is the exact failure the sheets exist to
prevent.