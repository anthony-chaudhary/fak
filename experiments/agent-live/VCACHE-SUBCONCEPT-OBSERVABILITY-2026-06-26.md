---
title: "vCache sub-concept observability over a real Claude account (2026-06-26)"
description: "10x observability into every vCache sub-concept (M1-M5 + cachemeta + score), grounded in 845 real Claude Code assistant turns on the live account, via the new fak vcache observe verb."
date: 2026-06-26
---

# vCache sub-concept observability over a real Claude account (2026-06-26)

## Question

The base provider prefix cache obviously works — any Claude session with
`cache_read_input_tokens > 0` proves that. The real question is everything the
vCache **sub-concepts** add on top of it: M1 calibration, M2 star anchors, M3
dedicated warming, M4 chains & recall, the M5 governor, the score composite, and the
cachemeta canonicalization floor. For each, what is true on **this account's real
traffic** — not on the scorecard's synthetic defaults?

This run answers that with one command over 845 real assistant turns.

## What was measured

`fak vcache observe` (new verb, this change) ingested the live Claude account's own
transcripts, grouped 845 cache-bearing assistant turns into 7 prefix families (one
Claude session = one shared system prefix), and ran the **shipped** M1-M5 decision
leaves over that real data. Frozen replay artifact:
`experiments/agent-live/vcache-claude-session-telemetry-2026-06-26.jsonl` (token
counters + session ids only; no prompt payload).

Reproduce (no network, no key — replays the committed snapshot):

```bash
go run ./cmd/fak vcache observe \
  --telemetry experiments/agent-live/vcache-claude-session-telemetry-2026-06-26.jsonl
```

Or point it straight at your own live transcripts:

```bash
go run ./cmd/fak vcache observe --transcript ~/.claude/projects/<ns>/<uuid>.jsonl
```

## TL;DR — one verdict per sub-concept

| sub-concept | package | provenance | verdict | real-account value |
|---|---|---|---|---|
| base provider cache | cachemeta (tier-1) | OBSERVED | **WORKS** | hit 93.0%, 103,091,957 cached tokens served |
| M2 star anchors | vcachestar (#717) | OBSERVED | **WORKS** | saved 77.1% (85,520,048 token-equiv), first-positive turn 1 |
| M1 concentration | vcachecal (#716 §5.2) | DECISION | **DEFEATED** | measured Zipf s=0.54 over 7 families (flat) |
| M1 warmth belief | vcachecal (#716 §7) | OBSERVED | **SAFE** | false-warm 0.00% (the lethal direction) |
| M3 dedicated warming | vcachewarm (#718) | DECISION | **NATURAL-FIRST** | 7/7 families net-positive by turn 1 |
| M4 chains & recall | vcachechain (#719 §11.0) | DECISION | **REFUSED** | 1312x loss at the 131,212-tok mean prefix, break-even 1313 siblings |
| M5 governor | vcachegov (#720 §5.4) | DECISION | **RIDE-NATURAL** | λT≥1 on all 7 families |
| score composite | vcachescore | DECISION | **CONTEXT-DEPENDENT** | MEASURED C (60/100) vs SYNTHETIC A (100/100) |
| cachemeta canonicalization | cachemeta (tier-1 §6) | OBSERVED | **HOLDING** | 0.00% false-warm → prefix stayed byte-stable |

Aggregate: **845 turns / 7 families · hit 93.0% · saved 77.1% · multiplier 4.37x ·
mean prefix 131,212 tokens.**

## The headline: within a family vs across families

The two findings that matter, and they pull in opposite directions:

1. **Within a prefix family, the cache concept is excellent.** Every family is
   net-positive from its very first turn (first-positive = 1), hit rates run
   88.9%-97.0%, and the shared ~131k-token system prefix is warmed once and read back
   for the rest of the session. That is the 77.1% realized saving and the 4.37x
   multiplier.

2. **Across families, the workload is flat, so the cross-family sub-concepts abstain.**
   The 7 families' reuse is near-uniform (measured Zipf `s=0.54`, well below the
   `s=1` flat-workload cliff), so M1 concentration flags **DEFEATED**: a hot-anchor
   index over the *tail* would not help here. The honest move is to manufacture skew
   (canonicalize/aggregate) or abstain — not to warm the tail.

This is exactly why the score is **CONTEXT-DEPENDENT**: the same realized economics
grade **A (100/100)** under the scorecard's synthetic steep workload (`s=1.74`) and
**C (60/100)** under the account's measured-flat distribution. The 40-point gap is
*almost entirely* the concentration assumption, not the savings: 35 of the 40 points
are concentration plus its hot-anchor index, and the other 5 are the benign false-cold
under-claim penalty in the risk term. An observability surface
that only reported the synthetic A would be hiding the most important fact about this
account's traffic.

## Per-family economics (each family = one shared system prefix)

| family | turns | hit% | saved_teq | saved% | first-pos | governor |
|---|---:|---:|---:|---:|---:|---|
| a9ab6d61 | 132 | 92.7% | 16,434,100 | 76.5% | 1 | ride_natural |
| 01bd5348 | 146 | 97.0% | 15,401,528 | 84.8% | 1 | ride_natural |
| bd203f71 | 115 | 88.9% | 12,877,289 | 69.4% | 1 | ride_natural |
| 4969fcb0 | 138 | 93.1% | 11,383,675 | 77.5% | 1 | ride_natural |
| 9fe98e8c | 90 | 93.1% | 10,663,217 | 77.4% | 1 | ride_natural |
| 751bfe8e | 110 | 93.8% | 9,589,254 | 78.7% | 1 | ride_natural |
| 8ce56aac | 114 | 92.5% | 9,170,986 | 76.4% | 1 | ride_natural |

## The safety signal (Law A1)

The one number that must never be non-zero is the **false-warm rate**: the lethal
"manifest says HIT, provider says MISS" case where you book a save you did not get. On
845 real turns it is **0.00%**. The shipped warmth-belief estimator (`vcachecal`),
run across each family's turns, never once predicted a warm that the provider then
missed. Verify-then-trust holds on real traffic.

The companion **false-cold rate reads 100%**, which sounds alarming and is not: it is
a per-predicted-cold-class rate, and the only turns the estimator predicts cold are
the first turn of each family — which on this account were already cross-session warm.
So every predicted-cold was actually warm: an *under*-claim (a missed warming chance),
never an over-claim. The dangerous direction is clean; the benign direction is what
moved.

This 0% false-warm is also the proxy behind the **cachemeta canonicalization
HOLDING** verdict: a volatile byte in the prefix (a timestamp, a UUID) would surface
as a believed-warm turn that reads `cache_read=0`. None did.

## Honest caveats

- **Concentration is a 7-family fit.** `s=0.54` is measured from only 7 active
  prefix families, so the exact exponent is noisy; the qualitative finding (near-
  uniform shares, not steeply concentrated) is robust from the raw per-family read
  volumes regardless of the fit.
- **The snapshot is a point in time.** These are live sessions that were still being
  written; the committed `…-telemetry-2026-06-26.jsonl` is a dated freeze. Re-running
  `observe` on fresh transcripts will move the totals, not the shape.
- **Every economics number is OBSERVED**: the cache counters (hit, cache_read) are
  relayed straight from Anthropic, and the token-equivalent savings are derived from
  them via Anthropic's published 0.1/1.25/2.0 multipliers — never a fak-caused effect.
  Every verdict labeled DECISION is fak's deterministic call over those counters. Budgeting still happens at the uncached
  price; a hit is a realized rebate, never a trust claim, and correctness never
  depends on one (Law A2).

## What shipped

- `internal/vcacheobserve` (tier-2 leaf): groups real telemetry by prefix family and
  composes the shipped `vcachecal` / `vcachechain` / `vcachegov` / `vcachescore` leaves
  into one per-sub-concept report. Pure, deterministic, clock-free, not registered.
- `fak vcache observe [--transcript FILE]… [--telemetry FILE] [--json]`: the 10x
  observability verb. Ingests real Claude transcripts natively (no glue script).
- Frozen witnesses: `vcache-claude-session-telemetry-2026-06-26.jsonl` (input) and
  `vcache-subconcept-observe-2026-06-26.json` (the full report).

## Sources

- `docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md` — the design note (§5.2
  concentration, §7 warmth belief, §11.0 recall cost gate, §5.4 governor).
- `experiments/agent-live/VCACHE-CLAUDE-PREFIX-PROBE-2026-06-25.md` — the prior
  controlled star-anchor probe this generalizes from a probe to the whole account.
- `internal/vcacheobserve`, `cmd/fak/vcache_observe.go` — the instrument.
