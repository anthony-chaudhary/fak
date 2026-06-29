---
title: "fak rollup — the executive activity roll-up"
description: "One signal-dense page that folds the agentic-fleet planes — closure honesty, dark loops, ship-stamp rate, box liveness — into a GREEN/WATCH/RED verdict and a ranked what-needs-you list, so one person can keep up with a fleet of agents."
---

# fak rollup — the executive activity roll-up

A fleet of agents produces more than a person can read. In a day it closes a
hundred-odd issues, lands dozens of commits, runs a dozen loops, and re-scores
half a dozen scorecards. Almost all of that is noise to someone deciding where
to look. The few things that are signal are narrow: how much of the volume is
real rather than merely claimed done, what is trending the wrong way, and the
short list of things that actually need a human right now.

`fak rollup` is that page. It folds the per-plane folds the fleet already emits
into one envelope and prints, in a glance, the answer to "what do I owe attention
today".

```
fak rollup            # human page on stdout
fak rollup --fast     # skip the slow planes (closure audit + scorecard pane)
fak rollup --json     # the control-pane envelope
fak rollup --md docs/today.md   # write the page with front-matter
fak rollup --check    # exit non-zero when the fleet verdict is RED (for cron)
```

## How to read it

The page opens with one word — **GREEN**, **WATCH**, or **RED** — and a
one-line headline. Then three blocks:

- **Signal-to-noise.** The marquee is *closure honesty*: of everything marked
  closed, how much is witnessed-resolved versus merely claimed. That ratio is
  the literal signal-to-noise of the fleet's "done", and it is the number a
  velocity story lives or dies on. Below it sits the *ship-stamp rate* — how
  many of the window's commits carry a real per-leaf stamp, i.e. how much of the
  committed work is attributed rather than anonymous.
- **What needs you.** The ranked list, critical before merely worth-a-glance.
  This is the part to act on. A clean fleet shows nothing here — silence is the
  absence of signal, so it is the absence of lines.
- **Plane coverage.** A small table of which planes were measured and what each
  reported, so you can see what was and was not looked at.

The verdict is conservative on purpose. Any critical item makes it RED. A
warning, or a plane that could not be measured, makes it WATCH. Only a fleet
where every plane reported and nothing deviated reads GREEN.

## What it folds

| plane | source | what it contributes |
|---|---|---|
| dispatch | `tools/dispatch_status.py` | closure honesty (the marquee) + throughput vs target |
| loops | `loopfleet` cross-ledger fold | dark loops — automation a human thinks is running but isn't |
| cadence | git work-done + the scorecard pane | ship-stamp rate + quality-debt trend |
| fleet | the lab roster fold (`fak lab status`) | box liveness / GPU waste |

The slow planes (the closure audit and the ~4-minute scorecard pane) are skipped
by `--fast`, and each plane accepts a pre-captured payload (`--dispatch-from`,
`--scores-from`, `--loops-from`, `--fleet-from`) so a scheduled job can run the
slow folds once and hand the report a deterministic input.

## The honesty rules

The roll-up is built to delete noise, not manufacture comfort, so three rules
hold:

- A quiet plane contributes no line. Only deviations surface.
- An unmeasured plane is never GREEN. If a collector failed or was skipped, the
  fleet reads WATCH and the coverage table says so — a missing witness is honest,
  a fake green is a defect.
- Every surfaced number carries a provenance label: **WITNESSED** (proven from
  git/tests), **OBSERVED** (a live reading relayed from a box or a loop tick), or
  **CLAIMED** (self-reported, no witness yet). The discipline is the point — the
  same one the rest of fak's control panes keep.

This is the operational companion to the strategic
[executive roll-up](EXECUTIVE-ROLLUP.md): that page answers "what should
leadership know about fak"; this one answers "what is the fleet doing right now,
and is it real".

The fold itself lives in `internal/execrollup` (pure and table-tested); the live
collectors live in `cmd/fak/rollup.go`.
