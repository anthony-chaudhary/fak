## Production model readiness inventory — 2026-07-14

### Scope and selection

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

### What “real production ready” means

A model is production-ready only when all six independently checkable gates pass. A code path or traffic volume does not substitute for another gate.

1. **Configured** — exact provider ID is an intentional, tested runtime choice with a documented workload tier.
2. **Used** — recent real traffic exists and is attributable to that exact ID.
3. **Protocol + cache** — streaming, tool calls, compaction/elision, refusal handling, and provider-cache accounting are model-scoped and green.
4. **Capability** — a fixed, repeated agentic corpus meets a predeclared correctness floor for the permitted tier.
5. **Cost truth** — raw/canonical identity and input/output/cache pricing provenance are explicit; unknown never means free.
6. **Reliability + rollback** — exact-model SLOs, canary thresholds, alerts, safe fallback, and a rollback drill are witnessed.

Readiness percentage is `passed gates / 6`; it is a coverage measure, not a probability estimate. `PARTIAL` remains a hold and earns no point.

### Current verdict

| Model | Configured | Used | Protocol + cache | Capability | Cost truth | Reliability + rollback | Coverage | Production verdict |
|---|---|---|---|---|---|---|---:|---|
| Opus 4.8 | PASS | PASS | PARTIAL | HOLD | PARTIAL | HOLD | **2/6 (33%)** | **HOLD** |
| Sonnet 4.6 | PASS | PASS | PARTIAL | HOLD | PARTIAL | HOLD | **2/6 (33%)** | **HOLD** |
| Haiku 4.5 | PASS | PASS | PARTIAL | HOLD | HOLD | HOLD | **2/6 (33%)** | **HOLD** |

**Portfolio verdict: heavily used, not yet proven production-ready.** All three are roughly four gate closures away. The common-path implementation is substantial, but the missing evidence is model-scoped: capability thresholds, exact identity/cost truth, and operable SLO/canary/rollback.

### Evidence and remaining work by model

#### `claude-opus-4-8`

- **Configured — PASS:** `internal/dispatchtick/launchprofile.go` declares Opus worker profiles; `cmd/fak/accounts_launch.go` and the Fable integration use this exact launch ID.
- **Used — PASS:** 2,900 observed turns across 61 sessions in the seven-day audit; 61.4% of audited output and 95.0% of estimated spend at the tier level.
- **Protocol + cache — PARTIAL:** Anthropic gateway/agent suites cover messages, tools, compaction/elision and cache semantics; `docs/benchmarks/ABLATE-RESULTS.md` also records a measured Opus 4.8 run. There is no single exact-model production conformance artifact that closes the full gate.
- **Capability — HOLD:** Opus is treated as the ceiling, but no current repeated three-model acceptance matrix binds that role to a declared correctness threshold. #4633.
- **Cost truth — PARTIAL:** the audit emits an estimate and cache-price tests exist, but production needs raw/canonical-ID provenance and fail-closed unknown handling. #4635.
- **Reliability + rollback — HOLD:** no exact Opus 4.8 SLO/canary/rollback drill was found. #4634.

#### `claude-sonnet-4-6`

- **Configured — PASS:** account launch/default settings and model fixtures name the exact ID; the tier is recognized throughout session audit and routing surfaces.
- **Used — PASS:** 2,690 observed turns across 465 sessions, the broadest session footprint of the three.
- **Protocol + cache — PARTIAL:** Sonnet traverses the shared Anthropic path, but shared-path tests alone do not prove exact-model streaming/tool/cache behavior in production.
- **Capability — HOLD:** there is no thresholded witness that Sonnet 4.6 reliably owns T1 implementation work. #4633.
- **Cost truth — PARTIAL:** tier cost is estimated, but exact identity and cache-price provenance need the canonical report. #4635.
- **Reliability + rollback — HOLD:** no exact Sonnet 4.6 SLO or exercised Opus→Sonnet / Sonnet→hold fallback contract was found. #4634.
- **Configuration debt:** #3929 already tracks the contradictory ultra-bucket launch table versus preset/docs and must close before the matrix can be called coherent.

#### `claude-haiku-4-5-20251001`

- **Configured — PASS:** the exact dated ID is present in account defaults and model-aware tests, and the session auditor recognizes the Haiku tier.
- **Used — PASS:** 1,195 observed turns across 177 sessions.
- **Protocol + cache — PARTIAL:** real traffic proves reachability, not model-scoped tool/cache conformance.
- **Capability — HOLD:** no repeated corpus proves the T2 boundary or proves escalation when a task exceeds it. #4633.
- **Cost truth — HOLD:** the exact dated emitted name is the known alias hazard documented in `notes/BORROW-ROUTING-SIGNALS-GATEWAY-PLANO-STUDY-2026-07-13.md`; unknown or mismatched catalog names must not become `$0`. #4635.
- **Reliability + rollback — HOLD:** Haiku is the bottom of this production set, so failure must hold/escalate rather than silently downgrade; that drill is missing. #4634.

### Delivery map

| Work item | Closes |
|---|---|
| #4633 — three-model capability acceptance matrix | Capability gate for all three; routing tier enforcement |
| #4634 — exact-model canary, SLO, alert and rollback | Reliability/rollback gate for all three |
| #4635 — canonical exact IDs and fail-closed pricing | Cost-truth gate for all three |
| #3929 — reconcile ultra-bucket preset and docs | Sonnet/Opus configuration coherence |
| #4632 — parent production-readiness epic | Final model-by-model read-back and portfolio closure |

### Promotion rule

Do not promote a model from HOLD because it has many turns, because another model passed the same shared code path, or because an issue closed. Promote only when the linked artifact names the exact provider ID, declares its sample/window and threshold, and can be reproduced from a committed command. The final #4632 read-back must re-run the inventory and mark all six gates PASS for each model.
