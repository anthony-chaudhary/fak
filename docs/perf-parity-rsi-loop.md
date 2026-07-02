---
title: "The performance-parity RSI loop: guard-hop overhead → 0, driven by live dogfood telemetry"
description: "Defines the fitness signal the gateway/forward RSI loop optimizes — guard-hop TTFB delta and cache_read preservation rate over a window of real /v1/messages turns — and wires it as a payload of the existing rsiloop Harness, with the MEASURED arm honestly fenced as hardware-gated."
---

# The performance-parity RSI loop

> **Issue:** [#733](https://github.com/anthony-chaudhary/fak/issues/733) ·
> **Epic:** [#637](https://github.com/anthony-chaudhary/fak/issues/637) (throughput parity over the shared spine) ·
> Track B ([#306](https://github.com/anthony-chaudhary/fak/issues/306)).
>
> The dispatch fleet now runs **through** the kernel by default (`0302eed`): every
> unattended dev turn flows over `fak serve`'s `/v1/messages` passthrough. That makes the
> fleet a **continuous, non-self-authored performance signal** — the right *fitness
> function* for an RSI propose→witness→keep/revert loop over gateway/forward changes. This
> doc defines that signal and wires it onto the loop the repo already ships, so a
> gateway/perf change is kept only when it measurably helps the **real** workload, not a
> synthetic micro-bench.

This is the **rsi** rung of fak's loop ladder
([engineering-is-building-loops](explainers/engineering-is-building-loops.md)) pointed at
one concrete payload. It does not introduce a new loop engine — it reuses
[`docs/rsi-loop.md`](rsi-loop.md)'s `rsiloop.Harness` seam and the guard-hop fitness
harness from [#734](https://github.com/anthony-chaudhary/fak/issues/734). The pieces exist
independently; this is the contract that binds them.

## What's already shipped (and what is not)

| Piece | Where | State |
|---|---|---|
| RSI loop engine — propose/measure/keep-or-revert, worktree-isolated off `main`, non-forgeable keep-bit, breaker, journal, vs-`main` `track` gate | `internal/rsiloop` + [`cmd/rsiloop`](https://github.com/anthony-chaudhary/fak/tree/main/cmd/rsiloop), [`docs/rsi-loop.md`](rsi-loop.md) | **shipped** (demo payload: `DefaultCacheSize` / LRU hit-rate) |
| Guard-hop fitness harness — overhead + prompt-cache-preservation row with a `--check` honesty gate | [`tools/guard_hop_bench.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/guard_hop_bench.py), [`docs/benchmarks/GUARD-HOP-OVERHEAD-PENDING.md`](benchmarks/GUARD-HOP-OVERHEAD-PENDING.md) | **harness shipped**; overhead arm PROJECTED; **TTFB + cache arms PENDING (hardware-gated)** |
| Loop-health metric — `closure_rate` / `regression_rate` | [`tools/issue_closure_audit.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_closure_audit.py), [#382](https://github.com/anthony-chaudhary/fak/issues/382) | `closure_rate` **shipped**; `regression_rate` is an honest placeholder pending a live verdict-journal |
| Loop-verdict telemetry — each keep/revert mirrored to the DOS audit journal | `cmd/rsiloop -dos-observe` (`dosobserve.go`, #588) | **shipped** (observe-only; never re-gates the keep-bit) |
| The live signal source — per-request latency, in-flight, status mix, `cache_read_input_tokens` | `fak serve` `/v1/messages`; the **FAK Dogfood Slow Requests** dashboard; `fak guard`'s exit summary (`provider cache — N prompt tokens served from cache`) | **emitting** on the live fleet ([`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md), [`docs/fak/always-on-dogfood-server.md`](fak/always-on-dogfood-server.md)) |

The one thing **not** present is a *measured win kept by a live run*. That arm is
hardware-gated — see [The honest gate](#the-honest-gate-what-is-not-done-here) at the
bottom. Everything above is the machinery; this doc is the wiring; the live keep is the
deferred step.

## 1. The fitness signal (the witness reads this, not its own work)

The loop optimizes **two** quantities, both read from the live fleet's `/v1/messages`
telemetry over a window of `W` real turns (not a synthetic bench). Both are *non-self-
authored*: the loop proposes the change, but the numbers come from the workload, not from
the author of the change.

**A. Guard-hop TTFB delta** — `ttfb_delta_ms = p50(guarded) − p50(direct)`.

- `guarded` = time-to-first-byte for a `/v1/messages` turn served **through** `fak serve`
  (the dogfood default; the kernel hop is in the path).
- `direct` = the same turn served against the upstream with the hop bypassed (the
  control arm; `tools/guard_hop_bench.py measure --direct-url …` stands up the matched
  mock).
- **Goal: drive `ttfb_delta_ms` toward 0.** The PROJECTED floor/ceil
  ([GUARD-HOP-OVERHEAD-PENDING](benchmarks/GUARD-HOP-OVERHEAD-PENDING.md)) is sub-
  millisecond per 50-turn session, so a real regression here is the gateway doing
  *work* it shouldn't (an extra copy, a re-parse, a lock) — exactly what the loop hunts.

**B. `cache_read` preservation rate** — `cache_preservation = cache_read(guarded) /
cache_read(direct)`, summed over the window.

- Source field: the provider's `cache_read_input_tokens`, reported straight back to the
  client on every `/v1/messages` response ([`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md)).
- **Goal: hold `cache_preservation` at 1.0.** A gateway change that perturbs the prompt
  prefix by even one byte collapses the provider prompt-cache; the rate falls below 1.0
  the instant a candidate breaks byte-for-byte `cache_control` forwarding. This is the
  loop's **correctness floor** on any TTFB win — a faster hop that breaks the cache is a
  REVERT, not a KEEP.

**The combined keep predicate** (what `Measure()` returns to the keep-bit):

```
KEEP  iff  suite_green
      AND  truth_clean                         (worktree git status clean)
      AND  cache_preservation >= 1.0           (the cache floor — non-negotiable)
      AND  ttfb_delta_ms < baseline_ttfb_delta (a STRICT improvement, like rsiloop)
```

The `>= 1.0` floor and the strict-improvement rule mean a candidate that "looks fine"
(green suite, clean tree) is still REVERTED unless it *measurably* lowered the hop cost
**without** spending the cache — the same load-bearing discipline as `rsiloop`'s cycle-3
case ([rsi-loop.md](rsi-loop.md)), here keyed on a real fleet metric.

### Window and stability

- `W` = a fixed count of recent `/v1/messages` turns (the dashboard window; default the
  last 200 turns or the trailing hour, whichever is smaller). Both arms read the **same**
  window so the delta is paired.
- The signal is a `p50`/ratio over `W`, never a single turn — provider-side variance
  (queueing, model warmth) dominates one turn and would make a single sample a coin flip.
  A candidate must beat the baseline on the windowed statistic to KEEP.

## 2. Wiring it as an `rsiloop.Harness` payload

[`docs/rsi-loop.md`](rsi-loop.md) already states the extension contract: *"A real
optimization plugs in by supplying its own `Harness`: a `Candidates()` that proposes real
changes, and a `Measure()` that applies each in a worktree and returns the measured KPI +
suite-green + truth-clean. The keep-bit, the breaker, the journal, and the vs-`main`
discipline are reused unchanged."* The perf-parity payload is exactly that:

| Cycle part | `rsiloop.Harness` seam | Perf-parity payload |
|---|---|---|
| 1. **Propose** | `Candidates()` | one gateway/forward optimization per candidate (e.g. reuse the forward buffer; avoid a header re-parse; widen a flush window) |
| 2. **Verify-correct** | `Measure().SuiteGreen` | `go build` + `go vet` + the `internal/gateway` suite in the worktree (the byte-for-byte `cache_control` forwarding tests already live here) |
| 3. **Measure-faster** | `Measure().Metric` + `.TruthClean` | the **fitness signal** of §1: run `guard_hop_bench.py measure` against the worktree's `fak serve` over the window `W`; `.Metric` = `−ttfb_delta_ms` gated by `cache_preservation >= 1.0`; `.TruthClean` = worktree `git status` clean |
| 4. **Keep-or-revert** | `shipgate.Evaluate` + `shipgate.Gate` | unchanged: the non-forgeable keep-bit + the K-consecutive-non-keep breaker |

Because the metric is `−ttfb_delta_ms` (so "larger is better", matching `rsiloop`'s
monotone-up convention) and the cache floor is folded **into** the returned metric (a
sub-1.0 preservation forces a non-improving value), the existing keep-bit needs no change
— the payload supplies the witness, the loop supplies the discipline.

```bash
# the closed perf-parity loop (once a live gateway + cache-reporting provider exist):
go run ./cmd/rsiloop -mode improve -repo . -baseline-ref main \
  -journal /tmp/perf-rsi.jsonl -dos-observe        # propose→witness→keep/revert, recurse

# ongoing benchmark against latest main (append one point; exit 3 on regression):
go run ./cmd/rsiloop -mode track -repo . -baseline-ref main -journal /tmp/perf-rsi.jsonl
```

> The `Harness` impl that yields gateway candidates + drives `guard_hop_bench.py measure`
> is the **payload to land next** (a Go seam in `internal/rsiloop` / a `cmd/` driver; the
> `tools` lane, not this `docs` lane). This doc is the contract it implements; it is
> deliberately not the implementation, so the spec lands before — and outlives — any one
> candidate.

## 3. Recording loop-health (the loop is itself checkable)

The perf loop must be auditable the same way it audits candidates. Two already-shipped
rungs cover it:

- **Per-verdict telemetry.** `cmd/rsiloop -dos-observe` mirrors every keep/revert to the
  DOS audit journal as a `dos improve --observe` receipt (`dosobserve.go`, #588) —
  *record-only*: it journals what the loop decided without letting the external command
  re-gate the keep-bit. Run with `-dos-observe` and the loop's decisions become a
  non-forgeable trail.
- **Loop-health metric** ([#382](https://github.com/anthony-chaudhary/fak/issues/382)).
  `tools/issue_closure_audit.py` computes `closure_rate` from witnesses the loop did not
  author (git ancestry + `dos commit-audit`). `regression_rate` is today an honest
  placeholder — it needs the verdict-journal trail the `-dos-observe` rung above writes; a
  perf-loop run with `-dos-observe` is exactly the source that turns the placeholder into a
  real number.

```bash
python tools/issue_closure_audit.py --json     # closure_rate over the audited slice
python tools/guard_hop_bench.py describe --json # the PROJECTED+PENDING fitness row
```

The `track`-mode gate is the regression backstop: `.github/workflows/ci.yml` runs
`rsiloop -mode track` on every push and fails the build on a strict KPI drop. When the
perf payload lands, its committed baseline floor extends this gate — a gateway change that
silently regresses the hop now blocks the trunk, not just the dashboard.

## The honest gate (what is NOT done here)

This doc lands the **definition and the wiring** — issue task 1 ("define a fitness signal
from the live fleet") as a concrete, checkable contract, plus the runbook for tasks 2–3.
It does **not** land a measured win, because that arm is hardware-gated and cannot be
honestly produced without:

1. a live `fak serve` gateway **and** a matched direct mock on one box
   (`guard_hop_bench.py measure --gateway-url … --direct-url …`),
2. a cache-reporting provider so `cache_read_input_tokens` is non-zero on both arms (the
   prompt-cache-preservation arm is PENDING per
   [GUARD-HOP-OVERHEAD-PENDING](benchmarks/GUARD-HOP-OVERHEAD-PENDING.md)), and
3. the live dogfood fleet emitting the window of real `/v1/messages` turns the witness
   reads.

Per the BENCHMARK-AUTHORITY honesty rules, no measured TTFB or cache number is asserted
here — the harness's `--check` gate refuses any row that smuggles a number into a PENDING
arm. The smallest next step to close [#733](https://github.com/anthony-chaudhary/fak/issues/733)
fully: implement the gateway-candidate `rsiloop.Harness` (the `tools`/`internal` payload),
stand up the gateway+mock+provider, run `-mode improve -dos-observe`, and fold the first
KEPT measured TTFB reduction into `BENCHMARK-AUTHORITY.md` — tombstoning the PENDING row.

## See also

- [`docs/rsi-loop.md`](rsi-loop.md) — the loop engine this payload plugs into.
- [`docs/benchmarks/GUARD-HOP-OVERHEAD-PENDING.md`](benchmarks/GUARD-HOP-OVERHEAD-PENDING.md) — the fitness harness + its honesty gate.
- [`docs/explainers/engineering-is-building-loops.md`](explainers/engineering-is-building-loops.md) — where the rsi rung sits in the loop ladder.
- [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) · [`docs/fak/always-on-dogfood-server.md`](fak/always-on-dogfood-server.md) — the live telemetry source.
