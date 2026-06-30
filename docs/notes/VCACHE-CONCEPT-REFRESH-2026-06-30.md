---
title: "Virtual-cache concept refresh: O(1) context, vBlocks, vCache M1-M5, and honest attribution"
description: "A 2026-06-30 re-grounding of every fak caching concept against the live tree: what fires on a normal run, what is now owner-attributed, what only has a live decision witness, and what is still built-but-gated-off. The companion plan is VCACHE-DEFAULT-ON-TOP50-2026-06-30.md."
---

# Virtual-cache concept refresh (2026-06-30)

This page re-states fak's caching vocabulary against the tree as it stands on
`main` today, and draws the one line that the goal behind it is really about:

> The default cache number now has to be read as an **owner split**, not one
> blended win: provider prompt-cache rebate is OBSERVED/provider-authored, while
> compaction, KV-prefix reuse, and vDSO avoidance are WITNESSED/fak-authored.
> The remaining gap is activation: fak's authored slice is still often near zero
> on normal proxy traffic because only the narrow Anthropic M2 system-anchor
> preflight now acts, M3/M4 actions are not live, and M5 now records live
> decisions but does not yet warm, pin, evict, or route.

The companion is the actionable list:
[`VCACHE-DEFAULT-ON-TOP50-2026-06-30.md`](VCACHE-DEFAULT-ON-TOP50-2026-06-30.md).
The standing design is
[`VCACHE-VIRTUAL-API-CACHE-2026-06-24.md`](VCACHE-VIRTUAL-API-CACHE-2026-06-24.md);
the product spine is
[`../CACHE-FRONTIER-OPERATING-PLAN.md`](../CACHE-FRONTIER-OPERATING-PLAN.md); the
reader-facing vocabulary is [`../glossary.md`](../glossary.md#the-vcache-streaming-economy-what-fak-guard-prints-per-turn).

## The five caching mechanisms, and who owns each saving

fak has **five** distinct token-saving mechanisms. They are usually discussed as
"the cache," which is the root of the conflation. Each has a different owner —
who *authored* the saving — and that owner is what attribution must report.

| # | Mechanism | What it saves | Owner of the saving | Live by default? | Where |
|---|-----------|---------------|---------------------|------------------|-------|
| 1 | **Provider prompt-cache** (Anthropic `cache_read`/`cache_creation`, OpenAI `cached_tokens`, Gemini `cachedContentTokenCount`, SGLang `cached_tokens`, vLLM prefix-cache) | re-prefill of a warm byte-prefix | **PROVIDER** (fak only forwards the breakpoint and relays the counter) | yes — observed every turn | `internal/gateway/cache_pricing.go`, `internal/agent/adapters.go` |
| 2 | **cache_control preservation + injection** | keeps the provider rebate from being *lost* by a mid-prefix edit; injects a breakpoint a caller forgot; hoists volatile Anthropic system blocks behind the stable anchor | **fak** (the act of not bursting/stabilizing the prefix) | yes (Anthropic path) | `internal/agent/anthropic_cachebp.go`, `anthropic_elide.go` |
| 3 | **Compaction shed** (`--compact-history-budget`) | input tokens dropped from the outbound body *before they are sent* | **fak** (WITNESSED) | yes — default budget on | `internal/gateway/debug.go`, `metrics.go` (`compactShed`) |
| 4 | **fak KV-prefix reuse** (in-kernel serve: `radixkv.Tree` over `model.KVCache`) | KV recompute avoided when fak serves the model itself | **fak** (WITNESSED, byte-identical) | yes when `fak serve --engine inkernel` | `internal/radixkv/`, `internal/cacheobs/`, `internal/agent/inkernel_planner.go:665` |
| 5 | **vDSO call-avoidance** (`fak_read`, repeated-call short-circuit) | a whole engine round-trip skipped | **fak** (WITNESSED) | yes | `internal/vdso/`, `internal/gateway/mcp.go` |

The honest law fak already follows (glossary): **cost is booked at the full
uncached price first; a *confirmed* hit refunds part of it.** The current tree now
decomposes that value in the default guard summary, per-turn debug line, `/metrics`,
`fak vcache observe`, and the Track-2 cache-value ledger. The next problem is no
longer "which owner got the saving?" but "which fak-authored mechanisms actually
fired on this run?"

## The vCache milestones — built vs. firing

vCache (epic #715–#720) is the active control loop that would make mechanism #1
*serve fak's intent* instead of being a passive observation. Its status today:

| Milestone | What it is | Code | Status |
|-----------|------------|------|--------|
| **M1 observe & calibrate** | probe each provider for TTL `T`, min-prefix `Mₘᵢₙ`, read discount `r`; build the warmth-belief estimator | `internal/vcachecal/`, `internal/gateway/metrics.go`, `internal/gateway/vcache_warmth_journal.go` | **partial live metrics + demotion alarm** — `/metrics` emits rolling-window warmth prediction-error counters and cumulative false-warm demotions; `/debug/vars` records the hash-chained demotion journal; **no live calibration loop, byte-diff, or request steering** (#716/#1497 open) |
| **M2 star anchors** | canonicalize a byte-stable anchor, let the first natural request warm it, fan siblings onto it | `internal/vcachestar/`, `cachemeta.RecommendLayout`, `internal/agent/anthropic_cachebp.go` | **partial live preflight** — the Anthropic raw path now applies the `RecommendLayout` volatile-to-tail rule to top-level system blocks before placing `cache_control`; full star manifests, sibling fan-out, and cross-surface anchoring remain open (#717/#1493) |
| **M3 dedicated warming** | `max_tokens:0` (explicit) / decode-1 (implicit) under the break-even gate | — | **not built into the live path** (#718 open) |
| **M4 chains & recall** | prefix DAG, topological replay, cost-gated rebuild | `internal/vcachechain/` | **UP but gated OFF** — `ProveRecall` runs only from the `fak vcache` CLI; `Replay` is never called live |
| **M5 governor** | pin / lazy / evict, warm budget, affinity routing, secret gate | `internal/vcachegov/`, `internal/gateway/vcache_governor_journal.go` | **decision witness live; actions gated OFF** — the gateway folds live provider-cache families into governor verdicts on `/metrics` and a hash-chained `/debug/vars` journal, but it still does not warm, pin, evict, or route on those verdicts |

So on a normal `fak guard -- claude` run: mechanisms #1–#5 are *observed* and
owner-attributed, M1 warmth-prediction error is *emitted* on `/metrics` with
false-warm demotions recorded to `/debug/vars`, the narrow M2 Anthropic
system-anchor preflight can *rewrite* volatile-before-stable system heads, M5's
governor verdict is *recorded*, and **full M2 fan-out plus M3/M4/M5 actions
still do not act**. The `fak vcache` subcommands
(`status/prove/prove-telemetry/prove-recall/observe/score`) remain mostly an
**offline lens** over real transcripts — they prove the economics and grade a
recorded session; they do not warm, pin, or recall anything live.

## The other concepts, re-grounded

- **O(1) context** is the one big agentic win that *does* fire by default:
  `--ctx-view-budget 8000` keeps the resident view bounded while the full history
  stays queryable (`internal/ctxplan`, `gateway.go` planner). It is "O(1)" in the
  sense that the resident prefill tail stays constant as history grows — priced in
  [`../explainers/o1-context-window-economics.md`](../explainers/o1-context-window-economics.md).
  It is a *context* economy, not a *cache hit*, and must not be folded into the
  provider-cache number.
- **vBlock** is the unit of cacheable work — a `cachemeta.Entry` keyed by content
  digest × model × tokenizer × serializer × vary-axes × position-in-DAG. It is a
  live identity type; the **prefix DAG** that would order vBlocks for recall is
  M4, gated off.
- **Warmth belief** is modeled (`vcachecal` prediction error, false-warm /
  false-cold), emitted live as `fak_vcache_warmth_*` metrics, and a false-warm
  now records a bounded `vcache_warmth_demotions` alarm that marks the
  reconstructed belief cold. It is still **not used to gate correctness** — no
  live run aborts or suppresses context on a broken-warmth prediction; byte-diff
  localization and traffic steering remain open.
- **cachemeta** is the tier-1 passive substrate. Live from it: `FromProviderCache`
  (every response), `ShapeGLMTurnSegmentWitnessed` (every GLM turn),
  `RecommendLayout` (star preflight). Offline-only: `Lifecycle`, `manifest`,
  `placement`, `kvresidency`, `kvtransfer`, `pool`, `hardware`.

## The activation gap, in live code

The old default `fak guard` exit printed the provider line and the fak lines
separately, and the per-turn `fak-turn` line's `saved=` was the provider net only.
That conflation is now closed in the live surfaces:

```go
// cmd/fak/guard.go — one default owner/mechanism headline
"fak guard: avoided-spend attribution — provider ~P (...) + fak ~F (...) = ~T token-equiv [...]"
```

```go
// internal/gateway/debug_stats.go — per-turn owner split
" prov=... tok (...% of prompt) fak=... tok"
```

The remaining gap is that the fak-authored side is still usually small on the
proxy path. The diagnostic now names likely causes (`anchor-starved`, no local
KV-prefix reuse, no multi-turn reuse, compaction not firing), and the M5 governor
decision witness appears as `fak_vcache_governor_decision_families{decision=...}`
plus the `/debug/vars` `vcache_governor_journal` rows. Those rows are evidence of
classification, not evidence that a warm/pin/evict action ran.

## Legacy assumptions worth removing

1. **"saved=" means fak's cache.** Resolved in the live surfaces: use
   `prov=... fak=...` and the `avoided-spend attribution` headline.
2. **The governor/recall engines are "up."** Tightened: M5 has a live decision
   witness, but action remains unregistered; M4 remains implemented/off-path.
3. **One Zipf forecast as the default score.** `fak vcache score` defaults to a
   synthetic `s=1.74` workload when no live snapshot is present — a forecast
   dressed as a measurement unless the snapshot path is populated. The default
   should be the realized number from the last session, falling open to the
   forecast only with a loud label.
4. **Cache value is reported by week and session-type, but never by mechanism or
   provider.** Both axes are missing from `cachevalueledger.Row`.
5. **Integration caches are read-only.** fak observes SGLang radix / vLLM KV
   events but never *commands* them, and has **no llama.cpp integration at all**.
   The vBlock/anchor abstraction stops at the provider API; it does not flow into
   the engines fak can actually drive.

The top-50 plan turns each of these into ranked, default-on, witnessed work.
