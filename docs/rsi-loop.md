---
title: "fak RSI loop: self-measured recursive self-improvement"
description: "How fak's rsiloop closes its recursive self-improvement loop: it derives every keep-or-revert witness from a real measurement run fork-isolated off main."
---

# The RSI closed loop (`rsiloop`)

> fak's recursive-self-improvement loop, **closed**. `internal/shipgate` is the
> non-forgeable keep-bit and [`cmd/rsicycle`](https://github.com/anthony-chaudhary/fak/tree/main/cmd/rsicycle) is a *one-shot* that
> takes the witnesses as flags. `internal/rsiloop` + [`cmd/rsiloop`](https://github.com/anthony-chaudhary/fak/tree/main/cmd/rsiloop)
> are the **true loop**: they *derive* every witness from a real measurement the
> loop runs itself, fork-isolated off `main`, so the loop author cannot forge the
> numbers that drive a KEEP. This is the runnable assembly of the four-part process
> the repo already names in [`EXTENDING.md`](https://github.com/anthony-chaudhary/fak/blob/main/EXTENDING.md).

This loop is the top rung of fak's *loops-all-the-way-down* picture. See
[Engineering is building loops](explainers/engineering-is-building-loops.md) for how
the RSI loop nests with the inner tool-call, turn, session, and fleet loops, and the
orthogonal threads (trust, cost, memory, observability, governance) that cut across them.

## The gap this closes

`rsicycle` is honest about being hand-fed:

```bash
# the one-shot: YOU supply before/after/suite-green/truth-clean as flags
go run ./cmd/rsicycle -metric hit_rate -before 0.07 -after 0.16 -suite-green -truth-clean
```

The keep-bit (`shipgate.Evaluate`) is non-forgeable *in code* — only `Evaluate`
sets the `improvedBit`. But its **inputs** are author-supplied flags. A *true* loop
has to measure them. That is the whole job of `rsiloop`.

## The four modular parts → four seams

| Part of the cycle | Seam (`rsiloop.Harness`) | What the real impl does (`worktree.go`) |
|---|---|---|
| 1. **Propose** | `Candidates()` | yields candidate `DefaultCacheSize` values |
| 2. **Verify-correct** | `Measure().SuiteGreen` | runs a real `go build`+`go vet` in the worktree |
| 3. **Measure-faster** | `Measure().Metric` + `.TruthClean` | runs `cmd/kpiprobe` in the worktree; checks the worktree's `git status` |
| 4. **Keep-or-revert** | `shipgate.Evaluate` + `shipgate.Gate` | the keep-bit + the escalation breaker |

Each candidate is applied to a **fresh detached git worktree off `main`**, so `main`
is never touched while a candidate is adjudicated (the same isolation
`shipgate.ApplyInWorktree` gives the one-shot). A KEEP advances the **running
baseline** in memory — the next candidate competes against the improved metric (the
*recursion*). The loop never auto-lands to `main`; surfacing the kept patch for a
human/gated step is the separate "Land it" stage in `EXTENDING.md`.

## The metric is a legal witness (deterministic)

The demo KPI is an **LRU cache hit-rate over a fixed reference trace**
(`internal/rsiloop/kpi.go`). It is wall-clock-free and RNG-free, so it reproduces
**bit-for-bit on any platform** — the rule for an RSI witness in
[`docs/proofs/00-METHOD.md`](proofs/00-METHOD.md). The hit-rate is monotonically
non-decreasing in the cache size and strictly rises over the candidate range, so the
loop has a *real* gain to find. The measured curve (`go run ./cmd/kpiprobe -dump`):

```
size  4  KPI=0.068182   <- DefaultCacheSize on main (the baseline)
size  6  KPI=0.157197
size  8  KPI=0.284091
size 10  KPI=0.467803
size 12  KPI=0.706439
```

With the default candidates `6,8,8,10` the loop produces **KEEP, KEEP, REVERT, KEEP**:
each strict gain is kept (advancing the baseline), and the no-op `8` (no gain over the
already-kept `8`) is reverted — driven by the *measurement*, not a flag.

## S0 as the objective: session outcomes

The same engine now has a session-observability harness for the dev-ex learning loop:

```bash
go run ./cmd/rsiloop -mode improve -harness sessionobs
```

That harness uses the full `loop_index` score as S0, with the Learn stage derived
from `internal/sessionobs.Score` and `LowerBetter=false`. The first candidate is a
no-op sessionobs toolchain proposal and REVERTs because S0 does not move. The second
links value and waste outcomes, marks the scrubbed corpus consumed by the loop, and
KEEPs only after the S0 loop-index rises to 100 with a clean sessionobs report. This
closes the session->outcome->toolchain loop for issue #1161 without adding a
separate keep/revert path; it reuses `shipgate.Evaluate`.

## Run it

```bash

# the closed improvement loop: propose, measure, keep-or-revert, recurse
go run ./cmd/rsiloop -mode improve -repo . -baseline-ref main \
  -candidates 6,8,8,10 -journal /tmp/rsi.jsonl

# the ongoing benchmark against latest main (append one point; alert on regression)
go run ./cmd/rsiloop -mode track -repo . -baseline-ref main -journal /tmp/rsi.jsonl
```

Exit codes: `0` = normal (run completed without escalation), `1` = error, `3` = ESCALATE (the breaker
tripped after K consecutive non-keeps — hand to a human) or, in `track` mode, a
detected regression on `main` (alert).

## Witnessed run (the loop, run for real against `main`)

`go run ./cmd/rsiloop -mode improve -candidates 6,8,8,10` — every `suite=` /
`truth=` / `cand=` field below was DERIVED from a real worktree run, not supplied:

```
baseline lru_hit_rate@5459aa1c4e65 = 0.068182
  cycle 1  DefaultCacheSize=6   base=0.068182 cand=0.157197 improved=true suite=true truth=true -> KEEP   (kept=true,  breaker=0)
  cycle 2  DefaultCacheSize=8   base=0.157197 cand=0.284091 improved=true suite=true truth=true -> KEEP   (kept=true,  breaker=0)
  cycle 3  DefaultCacheSize=8   base=0.284091 cand=0.284091 improved=false suite=true truth=true -> REVERT (kept=false, breaker=1)
  cycle 4  DefaultCacheSize=10  base=0.284091 cand=0.467803 improved=true suite=true truth=true -> KEEP   (kept=true,  breaker=0)
SUMMARY cycles=4 kept=3 final=KEEP final_baseline=0.467803 escalated=false
```

The baseline was measured at `main@5459aa1c` — a SHA that landed *after* the loop's
own commit, because `main` advanced under the run. The loop re-derived its baseline
from **latest** `main` with no prompting: that is the "benchmark against latest main"
property, observed live. Cycle 3 is the load-bearing case — a candidate with a green
suite AND a clean tree is still REVERTED because the metric did not strictly improve;
no amount of "looks fine" buys a KEEP without a measured gain.

## Ongoing benchmark-against-main

`-mode track` measures the KPI on `main` and appends one row to the JSONL journal,
tagged with the `main` SHA it was measured at. Run on a cadence (a cron / `/loop`),
the journal becomes a **time series of `main`'s KPI** — and each run compares to the
last recorded point, exiting `3` on a regression. Because the *improve* baseline is
also re-derived from `main` every run, a kept gain is always a gain over **latest
main**, never a number that drifted from ground truth. (A regression caused by `main`
getting *faster at the arm a number depends on* — the F1 tombstone case in
[`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md) — is exactly what the series
surfaces.)

**Enforced in CI (not just a cadence).** [`.github/workflows/ci.yml`](https://github.com/anthony-chaudhary/fak/blob/main/.github/workflows/ci.yml)
runs `rsiloop -mode track` on every push, re-measuring the deterministic main KPI in
a worktree off `HEAD` and comparing it against a committed baseline floor
([`internal/rsiloop/testdata/main-kpi-baseline.jsonl`](https://github.com/anthony-chaudhary/fak/blob/main/internal/rsiloop/testdata/main-kpi-baseline.jsonl)),
failing the build (exit `3`) on a **strict** drop. The track verdict mirrors
`dos improve`'s REVERT (a non-improving candidate); wired this way it stops being
inert telemetry and becomes a hard gate — a regression on the loop's own KPI now
blocks the trunk. Because the KPI is wall-clock-free and RNG-free (a single integer
`hits/total` division), the floor is bit-identical on the runner, so the gate fires
only on a real drop (e.g. `DefaultCacheSize` lowered), never on platform noise.

## Extending it to a real subsystem

The demo wires one tunable. A real optimization (a cache-eviction policy, a quant
kernel, an admission rung) plugs in by supplying its own `Harness`: a `Candidates()`
that proposes real changes, and a `Measure()` that applies each in a worktree and
returns the measured KPI + suite-green + truth-clean. The keep-bit, the breaker, the
journal, and the vs-main discipline are reused unchanged — the loop is the harness,
your subsystem is the payload.
