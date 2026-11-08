---
title: "fak serve config: hot-capability inventory (all 149 flags)"
description: "Per-knob classification of every fak serve flag by hot-swap tier and live-consumption status, with the stored-but-inert defect list and the operator posture each knob serves."
---

# serve-config-hot-inventory.md — every `fak serve` knob, classified

Companion to [serve-config.md](serve-config.md) (operator reference) and
[notes/CONCEPT-HOT-SWAPPABLE-SERVING-ARCHITECTURE-2026-09-03.md](notes/CONCEPT-HOT-SWAPPABLE-SERVING-ARCHITECTURE-2026-09-03.md)
(the 4-tier taxonomy and 7-spine roadmap this inventory operationalizes).
Every row below was verified against the code at revision `ae5527a3d24c`
(origin/main ancestor, 2026-11-06); consumption sites are `file:line`
citations, not guesses. The burndown that turns this inventory into ordered
work is [serve-config-hot-burndown.md](serve-config-hot-burndown.md).

## Why this exists

The concept doc's pathology is real here: most `fak serve` knobs are startup
flags read once at boot. Changing one means a full restart — cold weights
re-read, prefix caches annihilated, in-flight sessions dropped. This doc
answers, per knob: **can it change without a restart today?** and if not,
**what exactly is missing?** The burndown doc turns the answer into ordered
spines.

## Classification vocabulary

| Class | Meaning | Hot today? |
|---|---|---|
| **wired-live** | Read per-request (or per-stream/session) from a hot-swappable slot — atomic pointer, RCU snapshot, or live-swappable governor | **yes** |
| **T1 candidate** | Read per-request but off an immutable struct or armed closure; the read site exists, so conversion to a ScalarConfig/atomic slot is mechanical | no (mechanical) |
| **hot-swappable slot** | A live setter exists (mutex slot, admission setter, preempt-limit setter) but the flag is applied once at boot; needs a control endpoint to drive it | no (endpoint gap) |
| **boot-armed** | Installs a self-driving closure/probe/loop at boot; the flag value is consumed once, then the thing runs itself | no (by design) |
| **boot-only** | Consumed during boot (backend resolution, model load, planner construction, bind); changing it requires a real restart | no |
| **client-scope** | Writes a client config file and exits; never serves | n/a |
| **ops/telemetry** | Observability sinks, feature catalogs, diagnostics; no serving behavior | n/a |
| **stored-inert** | Parsed/advertised but never consumed on any live path — a defect | broken |

Tier mapping: wired-live ≈ Tier 0/1 of the concept taxonomy (already
realized); T1 candidate ≈ Tier 0/1 (mechanical conversion); hot-swappable
slot ≈ Tier 2 (setter exists, drain semantics already handled by the setter);
boot-only splits into Tier 0/1/2/3 by what the knob actually controls — the
per-flag notes say which.

## The one seam that is already hot: ScalarConfig (Tier 0, landed)

`internal/gateway/control.go` publishes a versioned scalar table behind a
monotonic epoch and an atomic pointer swap (`ScalarConfig()` at control.go:107,
`SetScalarConfig` at :170), driven by `PATCH /v1/control/config` +
`POST /v1/control/apply` (+`?dry_run=true`), with `/v1/control/events` and a
UDS control socket. Six knobs ride it:

| Knob | Verdict | Evidence |
|---|---|---|
| `completion_deadline_ms` | **wired-live** | read per-request via `s.ScalarConfig()` |
| `max_waiting_seqs` | **wired-live** | admission consults the scalar table per-request |
| `compact_history_budget` | **wired-live** | `internal/gateway/messages.go:548` reads `scalarCfg.CompactHistoryBudget` per-request |
| `compact_anchor_head` | **wired-live** | `messages.go:576` reads `scalarCfg.CompactAnchorHead` on the passthrough path |
| `stream_progress_timeout_ms` | **stored-inert** | PATCHable but never read on any live path; the live twin is the `--stream-progress-timeout` flag (planner field, per-stream read at `internal/agent/stream_stall.go:506-517`) |
| `log_level` | **stored-inert** | PATCHable but never read on any live path |

4 of 6 wired, 2 of 6 stored-inert. The two inert knobs are the control-plane
defects below.

## The stored-inert defect list (knobs that lie)

These are parsed, help-advertised, or PATCHable but never consumed on any
live path. Each is a defect the burndown burns down first — a knob that
accepts a value and ignores it is worse than a knob that demands a restart,
because it *lies*:

| Knob | Surface | Evidence |
|---|---|---|
| `--claude-config` | flag | parsed at `serve.go:302`, suppresses the model prompt (`serve_model_prompt.go:114`), but has **no dispatch arm and no writer** — `serve.go:898-933` dispatches codex/opencode/pi config arms only. Silently boots a full server. |
| `--write-claude-config` | flag | same: parsed `serve.go:303`, prompt suppression `serve_model_prompt.go:115`, no dispatch arm, no writer anywhere. |
| `--gpudirect-overflow` | flag | registered, help-advertised (`serve_features.go:192`), no serve implementation consumes it. |
| `stream_progress_timeout_ms` | control API | PATCHable via `/v1/control/config`, never read on any live path (the live twin is the boot-only flag). |
| `log_level` | control API | PATCHable, never read. |
| 14 control.Config Tier1/2 fields | control API | all 14 fields of the Tier1/2 sections are accepted by the validator and dropped at the observer boundary (`internal/gateway/http_control.go:58-76`); `internal/control` is imported only by `http_control.go:10`, and the watchdog consumes externally-POSTed telemetry only. |

## Fleet tallies (149 flags)

| Class | Count | Flags |
|---|---|---|
| wired-live | 4 | compact-history-budget, compact-anchor-head, memory-governor, route-manifest |
| T1 candidate | 31 | see the T1 list below |
| hot-swappable slot | 4 | max-total-tokens, max-batch-prefill-tokens, native-admission-token-budget, native-kv-host-swap-bytes |
| boot-armed | 2 | readiness-coherence, topology-shadow-input |
| boot-only | 94 | grouped below |
| client-scope | 10 | claude, codex, codex-config, codex-config-path, opencode, opencode-config, pi, pi-config, pi-config-path, write-codex-config |
| ops/telemetry | 11 | appliance-observability, debug-stats, dojo, metrics-snapshot, notify-native, notify-slack, notify-webhook, otlp-traces-endpoint, perf-ledger, print-effective-config, print-features |
| stored-inert | 3 | claude-config, write-claude-config, gpudirect-overflow |

4+31+4+2+94+10+11+3 = 149 ✓ (aliases `--ctx`/`--context-budget-tokens`,
`--gpu-layers`/`--native-gpu-layers`, and the `--cpu-offload-experts`/
`--expert-parallel`/`--n-cpu-moe` trio counted once each).

### The 31 T1 candidates

Batch A/B (21): `allow-lan`, `assume-session-turns`, `ctx-view-budget`,
`defer-cold-tools`, `defer-tools`, `elide-result-bytes`,
`elide-stale-reads`, `expose`, `key-principal`, `model`, `native`,
`native-max-turns`, `native-speculate`, `llama-slot-affinity`,
`llama-soft-slot`, `llama-soft-slot-policy`,
`positive-residual-substitution`, `speculative`, `stream-progress-timeout`,
`stream-soft-progress-timeout`, `tool-ceiling`.

Batch C (10): `policy-canary-turns`, `require-key-env`, `reset-on-budget`,
`route-accounts`, `session-id`, `spend-cap`, `spend-scope-trace`,
`vcache-anchor`, `vdso-proxy-fill`, `workspace-admission-permissive`.

Each has a per-request read site off an immutable struct or armed closure;
converting the read to `s.ScalarConfig()` (or a dedicated atomic slot) makes
the knob hot with no behavior change. `session-id` already has a setter
(`tool_routing.go:984 SetDefaultTraceID`); `spend-cap` already has one
(`SetSpendGovernor`).

## Boot-only flags, grouped by what a restart actually buys

94 boot-only flags are not homogeneous. Grouped by the subsystem that owns
them, with the tier the concept taxonomy assigns:

### Backend/model load (Tier 3 — real restart work)
`--backend`, `--gguf`, `--tokenizer`, `--metal`, `--mock`, `--engine`,
`--gpu-layers`/`--native-gpu-layers`, `--cuda-graph`, `--llama-server`,
`--llama-startup-timeout`, `--n-cpu-moe`/`--cpu-offload-experts`/
`--expert-parallel`, `--expert-ring-bytes`, `--tensor-parallel`,
`--qwen38-runtime`, `--native-chat-template-file`/`-model`/`-sha256`,
`--native-context-tokens`, `--native-qwen-q4k-prefill-chunk-tokens`,
`--native-qwen35-metal-gdn-sequence`, `--native-q4k-gateup-slab`,
`--native-v41-engram-lookahead`, `--native-v41-prefill-sum-readback`
(atomic global, hot-writable today but experimental), `--kv-precision`
(per-session realizable today via `SetDeviceKVPrecision`/`ConvertToPrecision`
— see hot-adjust precedents; fleet-wide hot tier is Tier 3 work),
`--native-prefix-profile`, `--native-compact-history-budget`,
`--vulkan-q4k-profile`, `--vulkan-stage-q4k`

### Planner construction (Tier 1/2 boundary)
`--base-url`, `--api-key-env`, `--provider`, `--replica-base-url`,
`--engine-cache-base-url`, `--engine-cache-engine`,
`--engine-cache-admin-key-env`, `--engine-cache-idle-timeout`,
`--engine-cache-require-exact-span`, `--embedding-pooling`,
`--execution-policy`, `--invalidation`, `--vdso` (SetVDSO seam exists),
`--native-allow-bash-command`, `--native-bash-command-timeout`,
`--native-code-tools`, `--native-code-workspace`

### Admission/memory (Tier 2)
`--context-budget-tokens`/`--ctx`, `--budget-warn-fraction`,
`--budget-webhook` (both boot-armed observers), `--fit`, `--slo-tpot-ms`,
`--slo-ttft-ms`, `--workload-prompt-tokens`, `--workload-output-tokens`,
`--workload-concurrency` (fit-auto sizing inputs, consumed once at boot
planning). (`--memory-governor` is wired-live; the four hot-swappable-slot
flags are listed in the tallies, not here.)

### Transport/bind (Tier 3)
`--addr`, `--unsafe-allow-unauthenticated-bind`, `--stdio`, `--role`,
`--session-registry`, `--session-state`, `--keep-awake`, `--fleet-bus`,
`--fleet-bus-dir`, `--fleet-bus-id`, `--fleet-bus-interval`,
`--arm-lease-store`, `--launch-evidence`, `--config`, `--plan-json`,
`--dry-admit`, `--policy-check`, `--dojo`

### Boot-armed closures (self-driving after boot)
`--readiness-coherence`, `--topology-shadow-input`

### Policy/routing boot seeds (hot content, boot-only flag value)
`--policy`, `--profile` — floor applied once at boot; the policy *content*
is hot-swappable via `POST /v1/fak/policy/reload` + canary, but the flag
value itself is never re-read. (`--route-manifest` is wired-live and
`--route-accounts` is T1; both are in the tallies, not here.)

## Hot-adjust precedents (proof the engine accepts live writes)

These exist in the engine today and are the mechanisms the burndown reuses:

- **macobs memory-pressure governor** (`internal/macobs/memory_pressure.go:385-430`):
  hot-calls `mtpGovernor.SetDepth` + admission-state transitions + page
  reclaim on pressure transitions — a live Tier-2-style adjust with no restart.
- **Admission setters** (`internal/gateway/admission.go`):
  `SetTokenBudgetWithProvenance` (:455, live, "measured" vs "explicit"
  provenance), `SetMaxWaiting` (:485), `SetTable` (:767), `SetSequencer`
  (:787), `SetFleet` (:807), `SetOrder` (:846); `admission_server.go:70-120`
  wires warmup capacity through them live.
- **`Server.SetMaxTotalTokens`** (`admission_server.go:232-257`): mutex slot +
  per-request `CheckMaxTotalTokens` — the hot-swappable-slot pattern.
- **`Engine.SetKVHostSwapByteLimit`** (`internal/modelengine/nativesched_preempt.go:155-185`):
  live setter, refuses shrink below ownership.
- **KV precision per-session**: `model.SetDeviceKVPrecision`
  (`internal/model/device_kv_precision.go:30-60`, still-empty HAL store only)
  and `KVCache.ConvertToPrecision` (`internal/model/kvcache_q8.go:625-650`,
  in-place f32↔q8_0), driven per-decode from `internal/agent/inkernel_decode.go:56-71`.
- **Policy floor reload + canary** (`internal/gateway/policy_canary.go`,
  `http_fak_endpoints.go:343-420`): `POST /v1/fak/policy/reload` →
  `armPolicyCanary` → per-served-turn adjudication outcomes → deny-all streak
  ≥ `--policy-canary-turns` → automatic rollback + metric.
- **Route manifest reload** (`internal/modelroute/reload.go:40-60`):
  `POST /v1/fak/route/reload` drives the atomic-pointer `modelroute.Live`;
  last-good gate, malformed manifest → 400, byte-identical → 200 no-op.

## Posture evidence (what the adaptive layer must serve)

- **P50/P99**: `internal/perfledger` Summary already records
  `TTFTP50MS`/`TTFTP99MS`, `PrefillTPSP50`/`PrefillTPSWeighted`,
  `DecodeTPSP50`, `E2EP50MS`/`E2EP99MS`, `CacheHitShare`, `ByRegime`
  (warm/cold TTFT split, #5630) — perfledger.go:453-480,549-560,730-735,774.
  Posture decisions can be evidenced without new plumbing.
- **Prefill/decode ratio**: lessons PHL-171 (prefill/decode on 5090L),
  PHL-132 (prefill kernel lift), PHL-129 (prefill = P parallel decodes),
  PHL-015 (concurrent prefill starves decode), PHL-110/109 (cold prefill) —
  the ratio is a real, measured trade-off, not a hypothetical.
- **Context-length trade-offs**: fit-auto (T04,
  `cmd/fak/serve_fit_auto.go`) already searches context × KV-dtype × slots
  against `--workload-prompt-tokens` (2048 default) /
  `--workload-output-tokens` (512) / `--workload-concurrency` (1) with
  `--slo-tpot-ms`/`--slo-ttft-ms` gates; pinned axes are never moved
  (PIN_INFEASIBLE).
- **Interactivity vs throughput**: the ScalarConfig knobs
  (`max_waiting_seqs`, `completion_deadline_ms`) and the admission setters
  are exactly the levers that trade queue latency against batch efficiency;
  the memory-pressure governor already demonstrates the closed-loop version.
- **Anti-masking canonical case**: `defaultServeKVPrecisionForBackend`
  (`cmd/fak/serve_native_config.go:159-164`) returns explicit f32 on CUDA
  because the fp16 default regressed long-generation decode ~3.8× in
  interaction with per-request `speculative.n_max:0` shapes [HW-WITNESSED
  raider-fak 2026-11-06 receipts cited in the comment]; fp16/bf16 stay
  opt-in; the underlying fix is tracked in ticket
  `cuda-kv-f16-typed-single-token-decode-load-20261106`. An adaptive tier
  must never use a tier choice to paper over a worse default.

## Evidence appendix

The full per-flag tables (all 149 rows with consumption sites) live in the
private repo scratch: `fak-private/_scratch/serve-hot-inventory-evidence.md`
(§4 batch A 48 flags, §5 batch B 51 flags, §6 batch C 50 flags, §7 tallies,
§8 posture evidence). The public doc carries the classification and the
defect list; the scratch carries the raw citation tables.
