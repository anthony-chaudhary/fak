---
title: "Borrow scout: pxpipe → fak (2026-07-08)"
description: "Study of pxpipe's measurement-honesty and gate techniques for porting into fak: 7 candidates scored, only 1 (a free count_tokens probe) worth borrowing."
---

# Borrow scout: pxpipe → fak (2026-07-08)

Study of the **pxpipe** context-image proxy (MIT; pinned SHA `b1f5a01b21607b32d347eebe5a81f4dccc8e2e49`,
cloned read-only to scratch) for techniques worth porting into fak. Every borrow is **INSPIRE**,
not INTEGRATE: pxpipe is TypeScript, fak is Go, so any port is a clean-room reimplementation —
no source is copied, and the MIT attribution note below is courtesy, not obligation.

Scope note: pxpipe's core product (render context to PNG, glyph-confusability legibility audit,
NOT-OCR framing, atlas/render pipeline) is **out of scope** for fak — fak is not an image proxy;
its token-reduction is textual (drop-and-splice compaction, progressive tool disclosure, KV reuse).
So the candidates below are pxpipe's *measurement-honesty* and *gate/control* techniques, which map
onto fak's existing `cachevalue` / `gateway` / `anthropic_compact` machinery.

## Scorecard (7 candidate techniques, one technique each)

| # | Technique (pxpipe anchor @ SHA b1f5a01) | fak witness | Verdict |
|---|---|---|---|
| A | **Free `count_tokens` counterfactual probe** — fire the free (unbilled) `count_tokens` endpoint on the *untouched* body to get a provider-MEASURED baseline token count; savings = measured(baseline) − measured(actual). `src/core/measurement.ts:32-60`, `src/core/baseline.ts` | fak *serves* `/v1/messages/count_tokens` (`internal/gateway/messages.go:1947`) but **never calls it as a client probe** (grep: zero client-side `count_tokens`). fak's counterfactual is **MODELED** from usage counters + price table (`UncachedCostUSD`, `internal/gateway/cache_pricing.go:165-174`). | **PARTIAL — the one real gap. Ticket A.** |
| B | OBSERVED-vs-authored attribution fence; report negative savings honestly, never floored (`baseline.ts:28-123`) | **PRESENT, fak ahead**: `MechanismSavings` owner/mechanism split; provider rows OBSERVED vs fak rows WITNESSED; "a provider-only win cannot satisfy a fak-authored cache claim"; `SavingsUSD` cold-write turn reads NEGATIVE by design (`cache_pricing.go:176-185`). | PRESENT — no borrow |
| C | Honest end-to-end denominator (report over the whole bill, not the touched slice) | **PRESENT**: `MechanismSavings` folds each session across all mechanisms; `ProviderCacheNetSavings` accounts both read rebate and write premium (`cache_pricing.go:201-228`). | PRESENT — no borrow |
| D | Conservative profitability gate biased to no-op ("mispredictions leave money on the table, never net-loss") — `transform.ts:154-381` | **PRESENT**: `CompactReasonBurstUnprofitable` bail (`internal/agent/anthropic_compact.go:118-122,482`). | PRESENT — no borrow |
| E | Amortized/horizon-aware gate: value a one-time cost over its cache-warm lifetime — `isCompressionProfitableAmortized`, `transform.ts:383-426` | **PRESENT** and near-identical: `CacheBurstBreakEvenTurns` prices the one-time suffix-rewrite penalty, `CacheBurstPaysBack` gates on `remainingTurns >= breakEven` (`anthropic_compact.go:807-842`). Also `cachemeta.prefix_score.BreakEvenTokens`. | PRESENT — no borrow |
| F | Symmetric-burn anti-flapping hysteresis: a mode switch busts the warm cache; add a burn penalty to the side that would flip, pinning current mode until per-turn savings exceed switch cost — `transform.ts:370-380` | **ABSENT, but low applicability**: fak prices cache-burst cost *one-directionally* (compaction is a one-way ratchet); there is no bidirectional per-turn mode on the cache-warm path that could flap. The failure mode this addresses does not currently exist in fak. | ABSENT / latent — no ticket (see below) |
| G | Verbatim keep-sharp guard: never lossily transform byte-exact content (IDs/hashes/secrets); keep originals recoverable — README + `library.ts` keepSharp/emitRecoverable | **PRESENT/N-A, fak ahead**: drop-and-splice keeps the cached prefix byte-identical, and dropped bytes are CAS-recoverable (`fak_context_restore` / `fak_context_spans`, ctxmmu quarantine + PageIn-refused-until-Clear). fak never lossily rewrites, so the verbatim risk never arises. | PRESENT — no borrow |

**Outcome: 1 borrow of 7.** Witnessing prevented 5 duplicate/N-A tickets against machinery fak
already ships (often more rigorous — the OBSERVED/WITNESSED fence, drop-and-splice + CAS recovery).

## Ticket A — FILED as [#3349](https://github.com/anthony-chaudhary/fak/issues/3349) — measure the compaction counterfactual with a free upstream `count_tokens` probe

**Kind:** enhancement, `internal/gateway` / `internal/cachevaluereport`. INSPIRE from pxpipe (MIT, SHA b1f5a01).

**Gap:** fak's compaction / cache-savings token baseline (the "what the untouched body would have
cost" counterfactual) is *modeled/locally-counted*, not measured against the provider's own tokenizer.
`UncachedCostUSD` (`cache_pricing.go:165`) prices a *no-caching* counterfactual over the token counts
the provider already billed; it does **not** measure what the *pre-compaction* (pre-drop) body would
have cost on the wire. fak already speaks `count_tokens` as a server but never fires it as a client.

**Borrow:** pxpipe fires the **free, unbilled** `count_tokens` endpoint on the exact untouched wire
bytes (`buildBaselineCountTokensBody` filters to `count_tokens`-accepted fields and synthesizes
minimal tool_results for orphan `tool_use` ids so the endpoint won't 400). The returned count is a
**provider-ground-truth** baseline, so savings become measured−measured, not measured−modeled.

**Proposed scope (small, sampled):**
- Add a client-side upstream `count_tokens` call in the gateway compaction path, gated to a sample
  rate (it's free but adds one round-trip of latency), firing on the *pre-compaction* body.
- Emit the measured baseline alongside the modeled one in the Track-2 ledger row, labeled
  `baseline=witnessed` vs the existing `baseline=modeled`, so the two can be differenced and the
  model's accuracy tracked (fits the WITNESSED-beats-MODELED provenance discipline already in
  `deepseek_budget.go` / `MechanismSavings`).
- Port pxpipe's orphan-`tool_use` synthesis + field-filter (clean-room Go) so the probe body is
  always `count_tokens`-valid.

**Honesty fence:** the probe measures token COUNT, not dollars; the $ figure stays modeled via
`CachePricing`. Provenance label must stay `witnessed`(count) + `modeled`($), never conflated.

**Witness on merge:** a gateway test that the probe body round-trips `count_tokens`-valid on a body
with orphan tool_use ids, and that ledger rows carry both `baseline_witnessed_tokens` and the
existing modeled figure.

## Not filed: candidate F (symmetric-burn anti-flap) — latent, revisit-if trigger

Do **not** file today. fak's compaction is a one-way ratchet and its cache-burst pricing is
one-directional, so the bidirectional per-turn flap this technique guards against does not exist.
**Trigger to revisit:** if fak ever adds a *reversible* per-turn mode on the cache-warm path (e.g.
per-turn compress on/off, or the wirescreen model-arm toggling per request), port pxpipe's symmetric
burn term — add the cache-bust cost to whichever side would flip, pinning the current mode until
per-turn savings clear the switch cost. Anchor to re-read then: `transform.ts:370-380`.

## Trail

- Filed: 1 issue — [#3349](https://github.com/anthony-chaudhary/fak/issues/3349) (Ticket A). F not filed (latent, no trigger).
- pxpipe pinned: `b1f5a01b21607b32d347eebe5a81f4dccc8e2e49` (MIT), read-only clone in session scratch.
- fak witness sites cited inline at `path:line` (HEAD of `main` at study time).
- Method: pxpipe README + load-bearing modules (`transform.ts`, `measurement.ts`, `baseline.ts`,
  `cache_pricing`-equivalents) → 7 candidates → witnessed each against fak via `fak_feature_query`
  (dogfood catalog) + raw grep (guard against the lexical ranker's false-ABSENT) + reading the two
  decisive fak sites (`anthropic_compact.go` break-even, `cache_pricing.go` counterfactual).
