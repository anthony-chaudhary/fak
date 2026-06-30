---
title: "Virtual-cache concept refresh: O(1) context, vBlocks, vCache M1-M5, and honest attribution"
description: "A 2026-06-30 re-grounding of every fak caching concept against the live tree: what fires on a normal run, what is built-but-gated-off, and the one accounting gap that makes the provider's prompt-cache look like fak's win. The companion plan is VCACHE-DEFAULT-ON-TOP50-2026-06-30.md."
---

# Virtual-cache concept refresh (2026-06-30)

This page re-states fak's caching vocabulary against the tree as it stands on
`main` today, and draws the one line that the goal behind it is really about:

> The single cache number `fak guard` shows an operator by default is **100% the
> provider's prompt-cache**. fak's own caching mechanisms are measured but never
> folded into that headline, and the agentic loop that would make fak's number
> large (vCache M2–M5) is **built but gated OFF**. So "cache saved 85%" reads as
> a fak win when it is, today, almost entirely the provider's — and "our agentic
> things aren't firing" is literally true.

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
| 2 | **cache_control preservation + injection** | keeps the provider rebate from being *lost* by a mid-prefix edit; injects a breakpoint a caller forgot | **fak** (the act of not bursting the prefix) | yes (Anthropic path) | `internal/agent/anthropic_cachebp.go`, `anthropic_elide.go` |
| 3 | **Compaction shed** (`--compact-history-budget`) | input tokens dropped from the outbound body *before they are sent* | **fak** (WITNESSED) | yes — default budget on | `internal/gateway/debug.go`, `metrics.go` (`compactShed`) |
| 4 | **fak KV-prefix reuse** (in-kernel serve: `radixkv.Tree` over `model.KVCache`) | KV recompute avoided when fak serves the model itself | **fak** (WITNESSED, byte-identical) | yes when `fak serve --engine inkernel` | `internal/radixkv/`, `internal/cacheobs/`, `internal/agent/inkernel_planner.go:665` |
| 5 | **vDSO call-avoidance** (`fak_read`, repeated-call short-circuit) | a whole engine round-trip skipped | **fak** (WITNESSED) | yes | `internal/vdso/`, `internal/gateway/mcp.go` |

The honest law fak already follows (glossary): **cost is booked at the full
uncached price first; a *confirmed* hit refunds part of it.** What is missing is
not honesty — it is **decomposition**. The five savings are reported on separate,
unrelated surfaces, so no one number says "of the spend you avoided, this slice
was the provider and these four slices were fak."

## The vCache milestones — built vs. firing

vCache (epic #715–#720) is the active control loop that would make mechanism #1
*serve fak's intent* instead of being a passive observation. Its status today:

| Milestone | What it is | Code | Status |
|-----------|------------|------|--------|
| **M1 observe & calibrate** | probe each provider for TTL `T`, min-prefix `Mₘᵢₙ`, read discount `r`; build the warmth-belief estimator | `internal/vcachecal/` | partial — estimator + probe types exist; **no live calibration loop** (#716 open) |
| **M2 star anchors** | canonicalize a byte-stable anchor, let the first natural request warm it, fan siblings onto it | `internal/vcachestar/`, `cachemeta.RecommendLayout` | `RecommendLayout` is **advisory-only** and enforced only inside the star preflight; **not a default pre-flight gate** (#717 open) |
| **M3 dedicated warming** | `max_tokens:0` (explicit) / decode-1 (implicit) under the break-even gate | — | **not built into the live path** (#718 open) |
| **M4 chains & recall** | prefix DAG, topological replay, cost-gated rebuild | `internal/vcachechain/` | **UP but gated OFF** — `ProveRecall` runs only from the `fak vcache` CLI; `Replay` is never called live |
| **M5 governor** | pin / lazy / evict, warm budget, affinity routing, secret gate | `internal/vcachegov/` | **UP but gated OFF** — "deliberately NOT registered into the kernel … adds zero rungs to the request path" (`vcachegov/doc.go:25`) |

So on a normal `fak guard -- claude` run: mechanisms #1–#5 are *observed*, M2's
layout advice is *computed*, and **M3–M5 never fire**. The `fak vcache`
subcommands (`status/prove/prove-telemetry/prove-recall/observe/score`) are an
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
  false-cold) but **not used to gate anything** — no live run aborts or re-plans
  on a broken-warmth prediction. It scores; it does not steer.
- **cachemeta** is the tier-1 passive substrate. Live from it: `FromProviderCache`
  (every response), `ShapeGLMTurnSegmentWitnessed` (every GLM turn),
  `RecommendLayout` (star preflight). Offline-only: `Lifecycle`, `manifest`,
  `placement`, `kvresidency`, `kvtransfer`, `pool`, `hardware`.

## The attribution gap, in live code

The default `fak guard` exit prints the provider line and the fak lines
**separately**, and the per-turn `fak-turn` line's `saved=` is the **provider's**
net only:

```go
// cmd/fak/guard.go:1279 — the ONLY cache-savings headline a default run shows
"fak guard: prompt-cache saving … the provider cache saved ~%s (%.0f%% off) …
 fak forwarded the cache_control prefix intact; it relays this provider-reported
 value and did not author this saving."
```

```go
// internal/gateway/debug_stats.go:255 — saved= is vcacheProofFromCounters over
// the PROVIDER's cache_read/cache_creation only
" saved=%s tok (%s%% of prompt)"
```

Compaction shed (a fak-authored saving) is a *different* line
(`guard.go:1306`), KV-reuse is in a *different* ledger
(`docs/nightrun/cache-value.jsonl`), and vDSO avoidance is on `/metrics`. There
is no single frame that says **"total avoided spend = provider X% + fak Y%,"** and
nothing tells the operator that Y is small *because M2–M5 are off*, not because
the workload had no reuse.

## Legacy assumptions worth removing

1. **"saved=" means fak's cache.** It means the provider's. Rename/relabel the
   headline so the provider slice and the fak slice are both visible, always.
2. **The governor/recall engines are "up."** They are *built and tested* but
   unregistered. "Up" should mean "fires on a real request"; until then say
   *implemented, off-path*.
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
