---
title: "Production model readiness inventory (2026-07-14)"
description: "The three exact model IDs carrying the Claude fleet, chosen by observed 7-day turns rather than repository mentions, with the session-audit witness command."
---

# Production model readiness inventory — 2026-07-14

## Scope and selection

This inventory covers the **three exact model IDs carrying the current Claude fleet**, selected by observed turns rather than by repository mention count or a future model wish list:

| Exact provider model | 7-day turns | Sessions containing model | Output tokens | Cache-read tokens | Fleet role |
|---|---:|---:|---:|---:|---|
| `claude-opus-4-8` | 2,900 | 61 | 1,429,640 | 243,772,304 | T0 / ceiling and complex work |
| `claude-sonnet-4-6` | 2,690 | 465 | 695,336 | 94,749,376 | T1 / normal implementation |
| `claude-haiku-4-5-20251001` | 1,195 | 177 | 194,208 | 27,383,352 | T2 / bounded routine work |

Witness command:

```powershell
fak dev session-audit audit --since-days 7 --all --include-subagents --max 5000 --json $env:TEMP\all-session-audit.json
```

The companion summary audited 919 transcripts and reported 6,919 non-synthetic turns. The exact-ID totals above are a deterministic fold of `sessions[].per_model` from that audit. This is **OBSERVED adoption evidence**, not a capability or production-readiness claim. Opus 4.7 and Sonnet 5 appear in the window but rank fourth/fifth (99 and 35 turns), so they are outside this top-three inventory.

This scope is different from the local in-kernel Mac candidate study in [`notes/MAC-MANYAGENT-MODEL-SELECTION-2026-07-13.md`](notes/MAC-MANYAGENT-MODEL-SELECTION-2026-07-13.md), whose provisional checkpoint is Qwen2.5-7B Q8 and whose device witnesses remain under #3809.

## What “real production ready” means

A model is production-ready only when all six independently checkable gates pass. A code path or traffic volume does not substitute for another gate.

1. **Configured** — exact provider ID is an intentional, tested runtime choice with a documented workload tier.
2. **Used** — recent real traffic exists and is attributable to that exact ID.
3. **Protocol + cache** — streaming, tool calls, compaction/elision, refusal handling, and provider-cache accounting are model-scoped and green.
4. **Capability** — a fixed, repeated agentic corpus meets a predeclared correctness floor for the permitted tier.
5. **Cost truth** — raw/canonical identity and input/output/cache pricing provenance are explicit; unknown never means free.
6. **Reliability + rollback** — exact-model SLOs, canary thresholds, alerts, safe fallback, and a rollback drill are witnessed.

Readiness percentage is `passed gates / 6`; it is a coverage measure, not a probability estimate. `PARTIAL` remains a hold and earns no point.

## Current verdict

| Model | Configured | Used | Protocol + cache | Capability | Cost truth | Reliability + rollback | Coverage | Production verdict |
|---|---|---|---|---|---|---|---:|---|
| Opus 4.8 | PASS | PASS | PARTIAL | PASS | PARTIAL | PASS | **4/6 (67%)** | **HOLD** |
| Sonnet 4.6 | PASS | PASS | PARTIAL | PASS | PARTIAL | PASS | **4/6 (67%)** | **HOLD** |
| Haiku 4.5 | PASS | PASS | PARTIAL | PASS | HOLD | PASS | **4/6 (67%)** | **HOLD** |

**Portfolio verdict: heavily used, not yet proven production-ready.** The bounded pilot capability and reliability/rollback gates now pass for all three exact IDs, while the portfolio remains HOLD. The remaining evidence is model-scoped: full protocol/cache conformance and exact identity/cost truth.

## Exact-ID capability provenance

The machine-readable join is produced by:

```powershell
fak model readiness-inventory --input examples/model-acceptance-prospective-v3-report.json --artifact-revision examples/model-acceptance-prospective-v3-report.json@r1+gf722bd39cd --expected-corpus top3-prospective-sentinel-v3 --as-of 2026-08-05T23:00:00Z
```

The source artifact is [`examples/model-acceptance-prospective-v3-report.json`](../examples/model-acceptance-prospective-v3-report.json) at `examples/model-acceptance-prospective-v3-report.json@r1+gf722bd39cd`, evaluated by `internal/modelaccept@r6+gf722bd39cd`. Corpus `top3-prospective-sentinel-v3` was declared and published before observation; the 18 fixed attempts ran from `2026-08-05T15:26:54-07:00` through `2026-08-05T15:29:37-07:00`. Each exact ID has six eligible samples and passes only its declared tier: Opus T0, Sonnet T1, and Haiku T2. The inventory fails closed for missing/mismatched/stale provenance and cannot borrow a PASS across model IDs.

These PASS rows cover only the two declared production-shaped task classes; they do not change the independent HOLD/PARTIAL gates below. Runtime dispatch consumes this exact-ID artifact and refuses above-tier, absent, stale, malformed, alias, or HOLD evidence before provider invocation. The scrubbed comparison and raw-stream digest are in [`docs/model-acceptance-prospective-v3-readout.md`](model-acceptance-prospective-v3-readout.md).

## Evidence and remaining work by model

### `claude-opus-4-8`

- **Configured — PASS:** `internal/dispatchtick/launchprofile.go` declares Opus worker profiles; `cmd/fak/accounts_launch.go` and the Fable integration use this exact launch ID.
- **Used — PASS:** 2,900 observed turns across 61 sessions in the seven-day audit; 61.4% of audited output and 95.0% of estimated spend at the tier level.
- **Protocol + cache — PARTIAL:** Anthropic gateway/agent suites cover messages, tools, compaction/elision and cache semantics; `docs/benchmarks/ABLATE-RESULTS.md` also records a measured Opus 4.8 run. There is no single exact-model production conformance artifact that closes the full gate.
- **Capability — PASS (bounded pilot):** exact ID `claude-opus-4-8` passed 6/6 fixed prospective T0 samples in the predeclared acceptance corpus and is witnessed at tier 0. This does not close protocol/cache, cost, or reliability gates. #4633.
- **Cost truth — PARTIAL:** the audit emits an estimate and cache-price tests exist, but production needs raw/canonical-ID provenance and fail-closed unknown handling. #4635.
- **Reliability + rollback — PASS:** the checked-in 60-minute SLO policy reports this exact ID, the clean-checkout dogfood gate rolled unhealthy Opus back to exact Sonnet, and the live provider-seam drill witnessed exact Opus failure attribution plus recovered Sonnet traffic. #4634.

### `claude-sonnet-4-6`

- **Configured — PASS:** account launch/default settings and model fixtures name the exact ID; the tier is recognized throughout session audit and routing surfaces.
- **Used — PASS:** 2,690 observed turns across 465 sessions, the broadest session footprint of the three.
- **Protocol + cache — PARTIAL:** Sonnet traverses the shared Anthropic path, but shared-path tests alone do not prove exact-model streaming/tool/cache behavior in production.
- **Capability — PASS (bounded pilot):** exact ID `claude-sonnet-4-6` passed 6/6 fixed prospective T1 samples in the predeclared acceptance corpus and is witnessed at tier 1. This does not close protocol/cache, cost, or reliability gates. #4633.
- **Cost truth — PARTIAL:** tier cost is estimated, but exact identity and cache-price provenance need the canonical report. #4635.
- **Reliability + rollback — PASS:** the checked-in 60-minute SLO policy and alert contract name exact Sonnet; the live drill recovered through Sonnet and the no-safe-fallback control held rather than selecting tier-2 Haiku for tier-1 work. #4634.
- **Configuration debt:** #3929 already tracks the contradictory ultra-bucket launch table versus preset/docs and must close before the matrix can be called coherent.

### `claude-haiku-4-5-20251001`

- **Configured — PASS:** the exact dated ID is present in account defaults and model-aware tests, and the session auditor recognizes the Haiku tier.
- **Used — PASS:** 1,195 observed turns across 177 sessions.
- **Protocol + cache — PARTIAL:** real traffic proves reachability, not model-scoped tool/cache conformance.
- **Capability — PASS (bounded pilot):** exact ID `claude-haiku-4-5-20251001` passed 6/6 fixed prospective T2 samples in the predeclared acceptance corpus and is witnessed at tier 2. This does not prove capability above T2 or close protocol/cache, cost, or reliability gates. #4633.
- **Cost truth — HOLD:** the exact dated emitted name is the known alias hazard documented in `notes/BORROW-ROUTING-SIGNALS-GATEWAY-PLANO-STUDY-2026-07-13.md`; unknown or mismatched catalog names must not become `$0`. #4635.
- **Reliability + rollback — PASS:** the checked-in 60-minute SLO policy and alert contract name exact Haiku; executable evaluator tests witness Haiku failure holding/escalating with no lower fallback, while the live tier-1 control proves Haiku is not used as an unsafe downgrade. #4634.

## Reliability + rollback

The exact-model operations gate is reproducible from committed inputs:

```powershell
fak model canary-gate --input examples/modelops-top3-canary.json
```

[`examples/modelops-top3-canary.json`](../examples/modelops-top3-canary.json) is the machine-readable report source: observations, outcome counters, thresholds, and alert contracts are keyed separately by `claude-opus-4-8`, `claude-sonnet-4-6`, and `claude-haiku-4-5-20251001`. Every policy declares a 60-minute window, 20-sample minimum, 95% success floor, 2% provider-error ceiling, 1% invalid-tool ceiling, 5,000 ms p95 latency ceiling, 3% throttle ceiling, and 5% fallback ceiling. Every breach routes to owner `modelops-oncall` via `model-provider-incidents`, with a 10-minute acknowledgement SLA and this runbook. Missing/malformed windows, thresholds, ownership, exact-ID observations, or capability-safe fallback data fail closed to `HOLD`.

The executable command returns typed statuses: `0` PROMOTE, `3` ROLLBACK, and `4` HOLD. The clean-checkout [dogfood readout](notes/MODELOPS-TOP3-DOGFOOD-2026-07-15.md) captures deterministic Opus→Sonnet rollback with exact-ID counters. The bounded [live provider drill](notes/MODELOPS-LIVE-PROVIDER-DRILL-2026-07-15.md) captures exact Opus provider failure, recovered exact Sonnet traffic through the same provider seam, and a no-safe-fallback `HOLD` control that refuses to downgrade tier-1 work to Haiku. Haiku's terminal failure behavior is independently executable in `internal/modelops` tests and holds/escalates because no lower production model exists. Together these artifacts satisfy #4634's report separation, checked-in SLO window/threshold/owner, executable canary/rollback, safe traffic drain, recovery read-back, and bottom-tier hold requirements.

## Delivery map

| Work item | Closes |
|---|---|
| #4633 — three-model capability acceptance matrix | Capability gate for all three; routing tier enforcement |
| #4634 — exact-model canary, SLO, alert and rollback | Reliability/rollback gate for all three |
| #4635 — canonical exact IDs and fail-closed pricing | Cost-truth gate for all three |
| #3929 — reconcile ultra-bucket preset and docs | Sonnet/Opus configuration coherence |
| #4632 — parent production-readiness epic | Final model-by-model read-back and portfolio closure |

## Promotion rule

Do not promote a model from HOLD because it has many turns, because another model passed the same shared code path, or because an issue closed. Promote only when the linked artifact names the exact provider ID, declares its sample/window and threshold, and can be reproduced from a committed command. The final #4632 read-back must re-run the inventory and mark all six gates PASS for each model.
