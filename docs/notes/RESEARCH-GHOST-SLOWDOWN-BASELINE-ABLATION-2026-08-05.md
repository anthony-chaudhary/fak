# Ghost slowdowns: a measured baseline and an ablation of where the churn comes from

**Date:** 2026-08-05. **Box:** the shared Windows fleet desktop (256 GB RAM, NVMe).
**Status of the box during capture:** not idle — ordinary fleet load (python
dispatchers, a peer `go test` run, 22 resident `fak` workers, 8 guarded Claude
sessions). Everything below is therefore a *working-box* baseline, not a quiescent one.

Companion to `internal/stallscan` (the classifier) and
`internal/dispatchtick/preflight_churn.go` (the admission gate). Landed with
commit `5491894c2a95`.

## The question

A "ghost slowdown" is a whole-machine stall where every usage meter reads fine.
Two things were unknown: what this box actually looks like when it is *not*
stalling, and where the churn that produces the stalls comes from. This note
answers both by measurement, and records one result that changed the design.

## 1. Baseline (10 min, 5 s samples)

| metric | median | p95 | max |
|---|---|---|---|
| CPU | 18.3 % | 83.5 % | — |
| context switches/s | 82,797 | 212,553 | — |
| syscalls/s | 843,439 | 1,792,515 | — |
| page faults/s | 307,510 | 1,170,222 | 2,074,026 |
| available RAM | never below 218 GB | | |
| disk idle | 94.9 % | | |
| disk sec/read | ~0 | | |
| process count | 362 | | 331–502 range |

**Disk and memory are conclusively ruled out.** 218 GB free and a 94.9 %-idle disk
cannot produce a stall. The load is entirely in the fault and scheduler paths —
which is the definition of the ghost class.

**The net-delta blindness, quantified.** Across the run the process count moved a
*net* −1, against **1,425** summed absolute movement. A monitor that samples a
census and subtracts sees ~0 while ~1,425 process transitions happen underneath it.

## 2. Ablation: where the churn comes from (240 s, 1 s ticks, 101 samples)

### 2a. The fault load is not charged to any live process

| | median | p95 | max |
|---|---|---|---|
| `\Memory\Page Faults/sec` | 567,196 | 1,253,451 | 2,211,373 |
| `\Process(_Total)\Page Faults/sec` | 65,428 | 279,104 | 738,997 |

Both counters are sampled in the *same* `Get-Counter` call, and `_Total` is
computed by the OS rather than by summing an enumeration, so the gap is not an
artifact of the probe racing itself.

**Unattributed share: median 87.2 %; on high-fault ticks (>500k/s, n=63), 90.7 %.**
The processes generating the fault load are not alive when the per-process counters
are read. That is the whole shape of the problem in one number.

### 2b. What is being born

**2,610 births in 101 ticks — 25.8 processes/sec sustained.**

| image | births | /min | share |
|---|---|---|---|
| conhost | 718 | 426 | 27.5 % |
| python | 525 | 312 | 20.1 % |
| git | 388 | 231 | 14.9 % |
| bash | 226 | 134 | 8.7 % |
| grep | 170 | 101 | 6.5 % |
| link | 82 | 49 | 3.1 % |
| vet | 62 | 37 | 2.4 % |
| docker | 48 | 29 | 1.8 % |
| wc | 44 | 26 | 1.7 % |
| fak | 32 | 19 | 1.2 % |
| sh, cmd, node, compile, head, wsl, pwsh, go, … | rest | | |

`conhost` is not an independent source — Windows creates one per console process,
so its 27.5 % is a *shadow* of the `git`/`bash`/`grep`/`sh`/`cmd`/`wc`/`head`
spawns. Netting that out, the picture is: **python dispatchers shelling out to git
and to small POSIX utilities in a loop**, plus the Go toolchain (`link`, `vet`,
`compile`, `go`) from a peer's test run.

Per-tick: median 22 births, p95 63, max 83.
**Gross births 2,586 vs net live-count movement −23** — the net axis is ~112× blind.

### 2c. Births drive the fault and scheduler load

Ticks with ≥3 births show median 567,980 faults/s against 274,318 on the (n=1)
zero-birth tick — **2.07×**; context switches 168,942 vs 76,133 — **2.22×**.
Suggestive, but the zero-birth class is a single sample, so §2d does it properly.

### 2d. Controlled injection: what one process creation costs

Four alternating control/treatment rounds (10 s each), injecting a known 20
`cmd.exe /c exit` per second so the box's own uncontrollable background load
cancels out of the difference:

| | control mean | treatment mean | per process creation |
|---|---|---|---|
| page faults/s | 639,752 | 803,077 | **8,166** |
| context switches/s | 222,765 | 266,817 | **2,203** |
| syscalls/s | 1,451,711 | 1,711,324 | **12,981** |

The fault delta was positive in 4/4 rounds. These are **floors**: `cmd /c exit` is
about the cheapest process Windows can make. A `python` or `git` or Go-linker start
costs considerably more.

**Closing the arithmetic:** 25.8 births/s × 8,166 faults ≈ **211,000 faults/s**, or
**≥37 %** of the 567k median — using an undercounted birth rate (§3) and a
cheapest-process cost. The true share is higher. Process creation is the dominant
term in this box's fault load, and the reason it is invisible to per-process
counters is simply that the processes are dead by the time anything asks.

### 2e. One source is fak's own Stop hook

The ablation put `git` third at 231 births/min, which pointed back at our own harness.
`runGuardStopHook` (`cmd/fak/guard_stophook.go:352`) fires `runWipAutoCheckpoint` at
**every turn end**. That capture creates a fresh temp index, seeds it with
`git read-tree HEAD` — so it carries **no stat cache** — and then runs `git add -A`,
re-hashing the whole worktree: 12,227 tracked files, 44,529,715 bytes.

Reproduced directly (n=5, same counters as §2d):

| | idle | during checkpoint | delta |
|---|---|---|---|
| page faults/s | 107,166 | 156,016 | +48,850 |
| context switches/s | 73,301 | 90,911 | +17,610 |
| syscalls/s | 436,810 | 518,255 | +81,445 |

**Mean 1.33 s wall-clock (max 1.36 s) and ~64,757 page faults per checkpoint** — by §2d's
conversion, about **eight process-creations' worth of fault cost in a single step**, on top
of the ~6–10 short-lived `git` spawns the checkpoint itself issues. Once per turn, per
session, with 8 guarded sessions resident.

It ran on `context.Background()` — **unbounded** — while every sibling side-effect in the
same hook is deadline-capped, and the hook discards its output. So on a stalling host the
one operation that scales with tree size had no ceiling and no voice. It is now bounded at
30 s (~22× the measured mean, so an ordinary turn is untouched) and a deadline is spelled
`capture-timeout`, distinct from `capture-error`: host pressure and a broken capture are
different facts and must not share a name.

This is the sharpest form of the whole finding — **the reaper's own harness was a
first-order contributor to the churn the reaper exists to detect**, and it was invisible
for the same reason everything else here was: nothing measured it.

## 3. The result that changed the design: poll samplers cannot see this

Every poll-based spawn axis — including fak's — counts births by diffing PID sets
between two enumerations. It can therefore only see a birth that *survives* to the
next enumeration.

Measured against a known ground truth: **200 injected `cmd /c exit` (~40 ms
lifetime), 1 s sampling → the sampler caught 10. 5 %. A 20× undercount.**

The bias is not a constant to divide out. It scales with process lifetime, so it is
~1× for a storm of long-lived workers and ~20× for a storm of `git rev-parse` and
`grep` — **worst exactly where the churn is worst**.

Two consequences, both now encoded in the code:

1. **The spawn axis is corroborating, not primary.** Page faults, context switches
   and syscalls are counted by the kernel *per event*; sampling cannot miss them.
   They see at full strength the storm the spawn axis reads a twentieth of. The
   per-spawn costs in §2d are the conversion between the two.
2. **A spawn count is meaningless without its window.** The axis previously compared
   a bare count to a bare threshold of 8. Against real gross births — median 22/s,
   p95 63, max 83 — that fires on **95 % of ticks of a healthy box**. The fix carries
   `SpawnWindowSeconds`, *measures* the window rather than assuming the nominal sleep
   (observed 1.31 s against a 1.00 s sleep — a 31 % error that would otherwise be
   baked into every rate), and compares births/sec against 150/sec, ~1.8× above the
   measured working-box maximum.

## 4. Observability fak should build

The general lesson is that the ghost class is **not a metric that reads bad — it is a
metric that reads fine because nothing populated it**. Three design rules follow,
each earned by a defect found here:

**Never spell "unmeasured" the same way as "healthy."** The churn gate failed open
silently: a tick with no ledger and a tick on a genuinely calm host produced
byte-identical payloads, so it sat inert for weeks while the box kept freezing.
`internal/stallscan/arming.go` makes this a first-class state
(missing/garbled/stale/disabled/armed) and dispatch ticks now carry a `host_churn`
block naming which one it is. Failing open is right; failing open *silently* is the
bug. **Generalize this**: every gate that reads an external signal needs an arming
readout, not just a value.

**Measure the consequence, not the event.** Events (process births) must be caught in
the act and are structurally missable. Consequences (faults, context switches,
syscalls) are counted by the kernel and cannot be. Prefer the axis that cannot go
blind, and calibrate the missable one against it.

**Carry the window with every count, and publish the derived rate next to the raw
numbers.** A count whose interval is implicit silently changes meaning when a caller
samples differently. Both gates now emit count, window, and rate together so a
reader can audit the division instead of trusting it.

**Record which half of a threshold is measured.** The defaults here bound *false
positives* against a measured negative class (a loaded but working host). No freeze
has yet been captured with a gross-birth axis armed, so **sensitivity is unproven** —
a firing is a real signal, a non-firing is not proof of calm. That limitation is
written at the threshold rather than left to be rediscovered.

## 5. Open follow-ons

- **Capture the positive class.** The thresholds need a freeze recorded with
  `SpawnWindowSeconds` set before sensitivity can be claimed. The self-monitor is not
  installed as a scheduled task on this box; the ledger at
  `%LOCALAPPDATA%\Fleet\stallscan.jsonl` was 22 days stale (last write 2026-07-14).
- **Consider an event-driven birth source.** ETW (Kernel Process provider) or a WMI
  `__InstanceCreationEvent` subscription is push, not poll, and would remove the 20×
  blindness entirely. That is the only way to get a true spawn rate.
- **Attribute the python→git spawn loop.** 312 python and 231 git births/min is the
  single largest reducible source; batching those shell-outs would cut the box's
  fault load materially.
- **Make the Stop-hook checkpoint incremental.** §2e is now *bounded*, not *cheap*: the
  30 s deadline stops it wedging a stalled host but does not remove the ~64,757 faults per
  turn. The cost is structural — a fresh `read-tree HEAD` index has no stat cache, so
  `add -A` re-hashes 44.5 MB every time. Reusing a persistent per-session index would let
  git's stat cache skip unchanged files and should cut this by orders of magnitude. That is
  a real design change to a hot path and is deliberately **not** folded into this one.
- **Count the skips.** `capture-timeout` is now distinguishable, but the Stop hook still
  calls the checkpoint with `io.Discard` and ignores its exit, so a host slow enough to blow
  a 30 s budget silently loses that turn's WIP. The skip needs to reach the stop ledger
  (`guard_stops.go`) before the deadline can be called fully wired.
