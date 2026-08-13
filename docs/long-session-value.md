---
title: "Long-session value: the 100/200/300-turn savings story"
description: "What a 100/200/300-turn agent session costs with and without fak-managed context, with every number traced to a committed artifact and honestly labeled simulated or observed."
date: 2026-07-07
---

# Long-session value — the 100/200/300-turn savings story

*Every number on this page cites the committed artifact it comes from, and every
row is labeled by provenance (`SIMULATED` = analytic model, `OBSERVED` = captured
from a real wire). Where an observed number does not exist yet, this page says
"not yet" and names the open issue — never a fabricated pass. Quote
[`CLAIMS.md`](../CLAIMS.md) for shipped scope; [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md)
is the number authority.*

## The problem

A coding agent's transcript is append-only, so every turn re-sends the whole
history. The provider prompt-cache discounts most of it — but **only while the
cached prefix stays byte-identical**. The industry-standard fix, summarize-and-
recompact (Claude Code's built-in auto-compact, and most frameworks'), rewrites
the prompt body near the context wall: new bytes → cache miss → the provider
re-bills a near-full prefix at full price on every compaction. Long sessions get
simultaneously **expensive** (billed tokens grow superlinearly) and
**cache-busted** (each compaction forces a large re-prefill).

## The mechanism, in one paragraph

fak's lever is one flag: `fak guard --compact-history-budget <tokens>` (default
**on** at a 48,000-token resident budget, `gateway.DefaultCompactHistoryBudget`,
`internal/gateway/gateway.go`; `0` disables). Past the budget it **drops whole
old middle turns and splices the original bytes back together** — a memcpy of
the protected `cache_control` prefix, never a re-marshal — then proves the
spliced prefix is byte-for-byte identical before shipping, and forwards the body
unchanged on any doubt. The cached head survives, so the provider's discount
keeps paying while the un-cacheable middle (which the provider re-bills every
turn anyway) is shed. The deep-dive, including exactly what is proven vs
hypothesized and how each industry harness compacts, is
[Compaction vs. the industry](notes/COMPACTION-VS-INDUSTRY-2026-06-25.md).

## Results at 100 / 200 / 300 turns

Source artifact for every number in this table:
[`internal/bench/testdata/relayvscompaction_report.json`](../internal/bench/testdata/relayvscompaction_report.json)
(schema `relayvscompaction.v1`, provenance `SIMULATED`, committed by #2707's
resolving commit `c54e57ce2`). The model is a hermetic analytic sweep — a
200k-token window, 6k tokens of growth per turn, cache read/write multipliers
0.1×/1.25×, auto-compact triggering at 95% of the window (`model` block of the
artifact). The fak-managed arm in this model is the **relay** strategy (rotate
at 60% of the window with a 2k-token externalized baton); the baseline arm
models built-in summarize-recompact. These numbers witness the **sign and
shape** of the comparison, not a provider-billed total — the artifact's own
provenance note says exactly this.

| Turns | Strategy | Billed tokens | % saved vs baseline | Cache-bust tokens | Provenance | Source (JSON pointer) |
|---|---|---|---|---|---|---|
| 100 | built-in auto-compact (baseline) | 2,062,048 | — | 241,192 | SIMULATED | `sweep[goal_turns=100].compaction` |
| 100 | fak-managed (relay model) | 1,269,000 | **38.46%** | 30,000 | SIMULATED | `savings_gates[goal_turns=100]`, `sweep[goal_turns=100].relay` |
| 200 | built-in auto-compact (baseline) | 4,224,314 | — | 483,467 | SIMULATED | `sweep[goal_turns=200].compaction` |
| 200 | fak-managed (relay model) | 2,553,000 | **39.56%** | 60,000 | SIMULATED | `savings_gates[goal_turns=200]`, `sweep[goal_turns=200].relay` |
| 300 | built-in auto-compact (baseline) | 6,469,807 | — | 786,322 | SIMULATED | `sweep[goal_turns=300].compaction` |
| 300 | fak-managed (relay model) | 3,852,000 | **40.46%** | 90,000 | SIMULATED | `savings_gates[goal_turns=300]`, `sweep[goal_turns=300].relay` |
| 100/200/300 | fak compaction ON vs plain-Claude auto-compact | *not yet* | *not yet* | *not yet* | OBSERVED | no committed artifact yet — open issue [#2708](https://github.com/anthony-chaudhary/fak/issues/2708) |

**The honest gate verdict.** The operator's target is a ≥50% billed-token
reduction at all three horizons. The committed modeled sweep does **not** meet
it: 38.46% / 39.56% / 40.46%, shortfalls of 11.54 / 10.44 / 9.54 points
(`savings_meets_50pct_target: false` per horizon,
`all_horizons_meet_50pct: false` in the artifact). What the model does show is a
consistent ~40% billed-token saving that **grows with horizon**, an 8×–8.7×
reduction in cache-busted tokens (e.g. 786,322 → 90,000 at 300 turns), and a
flat 120k peak context vs a baseline pinned at 96% of the window. Per this
repo's net-true-value doctrine the number is reported as measured, not rounded
up to the claim.

**Provenance ladder.** The OBSERVED row is empty by design, not omission: the
loader seam that accepts real per-leg records
(`BuildRelayVsCompactionReportFromLegLedger`, `internal/bench/relayvscompaction.go`)
is shipped and witnessed by a replayed 6/12-turn sample
([`internal/bench/testdata/relayvscompaction_observed_legs.json`](../internal/bench/testdata/relayvscompaction_observed_legs.json),
[`relayvscompaction_observed_report.json`](../internal/bench/testdata/relayvscompaction_observed_report.json)
— per its own note, a loader/schema witness, **not** a live provider-billed
run). The controlled 100/200/300-turn A/B capture (fak compaction ON via
`fak guard --compact-history-budget` vs plain `claude` auto-compact, same task,
matched length) is open work: [#2708](https://github.com/anthony-chaudhary/fak/issues/2708).
When it lands, its leg-ledger JSON feeds the same report shape and either
confirms or demotes the modeled numbers above.

## What is already observed on real wires today

Two committed real-wire artifacts witness the mechanism (not the controlled
A/B) at long horizons:

- **A 900-turn session through the real gateway path**:
  [`experiments/agent-live/compact-100k-session-dogfood-2026-06-25.json`](../experiments/agent-live/compact-100k-session-dogfood-2026-06-25.json)
  — 142,516 estimated inbound tokens trimmed to 6,597 forwarded (**95.4% shed**,
  budget 4,000), with the OFF-vs-ON protected cache prefix **sha256-identical**
  (`verdict: PASS`). Replayed against a mock upstream through the real splice
  code — it witnesses the byte-splice and prefix preservation, not provider
  billing.
- **Live `fak guard -- claude` sessions**:
  [`docs/nightrun/cache-savings.jsonl`](nightrun/cache-savings.jsonl) — the
  longest 2026-07-06 session fired compaction 7 times at the default 48,000
  budget, shedding **≈106,708 tokens per fire** (746,956 cumulative ÷ 7).
  Fence, per [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md): the ledger's
  `compaction_shed_tokens` field is **cumulative across fires** and re-counts
  the same aged middle turns each fire, so cite the per-fire figure, never the
  session sum as a distinct-token count or as a "share" of cache reads.

## Reproduce it

```sh
# 1. The modeled 100/200/300 sweep + the >=50% gate (verifies the committed artifact):
go test ./internal/bench -run RelayVsCompaction

# 2. Live, on your own Claude Code session (compaction is on by default; 0 disables):
fak manage --compact-history-budget 48000 -- claude

# 3. Price real Claude Code transcripts (~/.claude/projects/**/*.jsonl) with the same economics:
fak cachevalue report --dev-sessions
```

What each shows: (1) regenerates and byte-compares the
`relayvscompaction_report.json` artifact behind the table above, including the
honest per-horizon gate verdicts; (2) runs the byte-splice lever on a real
session — the guard exit summary prints what compaction shed; (3) prices your
own transcripts so the numbers are yours, not ours.

## Related

- [Compaction vs. the industry](notes/COMPACTION-VS-INDUSTRY-2026-06-25.md) — the mechanism deep-dive and honest fences.
- [Long sessions keep the cache hit](explainers/long-sessions-keep-the-cache-hit.md) — the reader-facing explainer.
- [Cache-value roll-up](cache-value-rollup.md) — the ongoing P&L view (`fak cachevalue report`).
- Epic [#745](https://github.com/anthony-chaudhary/fak/issues/745) · bench sweep [#2707](https://github.com/anthony-chaudhary/fak/issues/2707) · observed capture [#2708](https://github.com/anthony-chaudhary/fak/issues/2708).
