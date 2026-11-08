---
title: "fak serve config: hot-capability burndown plan"
description: "Ordered spines to make every fak serve config knob hot-capable, the operator postures the adaptive layer must serve, and the anti-masking guardrail that keeps adaptation from excusing worse defaults."
---

# serve-config-hot-burndown.md — making every knob hot, in order

Companion to [serve-config-hot-inventory.md](serve-config-hot-inventory.md)
(the per-knob classification this plan burns down) and
[notes/CONCEPT-HOT-SWAPPABLE-SERVING-ARCHITECTURE-2026-09-03.md](notes/CONCEPT-HOT-SWAPPABLE-SERVING-ARCHITECTURE-2026-09-03.md)
(the taxonomy and control-plane design). Every spine names its mechanism
(reusing an existing engine seam), its witness, and its anti-masking
obligation. Revision basis: `ae5527a3d24c` (2026-11-06).

## Ordering principle

1. **Defects before capabilities.** A knob that lies (stored-inert) is worse
   than a knob that demands a restart; fix lies first.
2. **Cheapest mechanism first.** A knob whose per-request read site already
   exists (T1 candidate) converts mechanically; a knob needing a new setter
   or drain state machine costs more; a knob needing model reload is Tier 3.
3. **Posture levers before posture automation.** Wire the levers an operator
   or agent would actually move for interactivity/throughput/P50/P99 posture
   before building closed-loop automation on top of them.

## Spine 0 — Burn down the lies (stored-inert defects)

**Goal:** no knob accepts a value it ignores.

| # | Defect | Fix | Witness |
|---|---|---|---|
| 0.1 | `--claude-config` / `--write-claude-config` parsed + prompt-suppress but no dispatch arm, no writer (`serve.go:898-933` has codex/opencode/pi arms only) | add the claude arm + writer (mirror `serve_pi.go`), or remove both flags from the surface | `go test ./cmd/fak/ -run TestServeConfigWriters -count=1` (new test: each advertised config flag either writes+exits or is removed) |
| 0.2 | `--gpudirect-overflow` registered, never consumed (`serve_features.go:192`) | implement or remove; if implemented, gate on backend capability | `go test ./cmd/fak/ -run TestServeFeatures -count=1` |
| 0.3 | `stream_progress_timeout_ms` PATCHable, never read | wire the control-plane field to the live planner field the flag already feeds (`gateway_planners.go:234` → `stream_stall.go:506-517`), or drop it from the PATCH schema | `go test ./internal/gateway/ -run TestControl -count=1` |
| 0.4 | `log_level` PATCHable, never read | wire to the logging seam or drop from the PATCH schema | same as 0.3 |
| 0.5 | 14 control.Config Tier1/2 fields accepted then dropped at the observer boundary (`http_control.go:58-76`) | either wire each field to its live seam or reject unknown/unwired fields at validation (fail-closed schema) | `go test ./internal/control/... -count=1` |

**Anti-masking note:** 0.3-0.5 must not be "fixed" by silently accepting and
ignoring — the fail-closed schema (reject unwired fields) is the honest
default until each field is wired.

## Spine 1 — Convert the 31 T1 candidates (mechanical, per-request reads exist)

**Goal:** every knob with a per-request read site off an immutable struct
becomes hot via the landed ScalarConfig seam (or a dedicated atomic slot
where the value is not a scalar).

**Mechanism:** extend `internal/gateway/control.go` ScalarConfig (or add
sibling atomic slots on `Server`) and repoint the per-request read sites.
The read sites already exist — that is what makes these T1.

**Order within the spine** (posture value first):

1. **Interactivity/throughput levers:** `tool-ceiling`, `defer-tools`,
   `defer-cold-tools`, `expose`, `native-max-turns`, `native-speculate`,
   `speculative` (mode selection per-request already; make the *mode* hot),
   `assume-session-turns`, `positive-residual-substitution`,
   `elide-result-bytes`, `elide-stale-reads`, `ctx-view-budget`.
   Witness: per-knob `go test ./internal/gateway/ -run TestControl -count=1`
   + a per-request read assertion.
2. **Auth/routing levers:** `require-key-env` (bearer keyset), `allow-lan`,
   `key-principal`, `route-accounts`, `session-id` (setter exists:
   `SetDefaultTraceID`), `vcache-anchor`, `vdso-proxy-fill`,
   `workspace-admission-permissive`.
   Witness: same, plus an auth-path test.
3. **Admission levers:** `spend-cap` (setter exists: `SetSpendGovernor`),
   `spend-scope-trace`, `reset-on-budget`, `policy-canary-turns`.
   Witness: `go test ./internal/gateway/ -run TestSpend -count=1`.
4. **Stream levers:** `stream-progress-timeout`,
   `stream-soft-progress-timeout` (planner-field reads at
   `stream_stall.go:506-517`/`:535-538` — convert to ScalarConfig reads so
   the control-plane twin stops being inert; this closes defect 0.3 too).
   Witness: `go test ./internal/agent/ -run TestStreamStall -count=1`.
5. **Attribution-only:** `model`, `llama-slot-affinity`, `llama-soft-slot`,
   `llama-soft-slot-policy` (per-request attribution reads; lowest posture
   value, do last or leave boot-only).

**Anti-masking note:** converting a knob to hot does not change its default.
The guardrail below (§Guardrail) applies to any future default change.

## Spine 2 — Drive the 4 hot-swappable slots (endpoint gap)

**Goal:** the setters that exist get a control-plane driver.

| Flag | Setter (exists) | Driver to add |
|---|---|---|
| `max-total-tokens` | `Server.SetMaxTotalTokens` (`admission_server.go:232-257`) | control endpoint → setter, with shrink-drain semantics the setter already enforces |
| `max-batch-prefill-tokens` | same slot family | same |
| `native-admission-token-budget` | `AdmissionController.SetTokenBudgetWithProvenance` (`admission.go:455`) | control endpoint → setter with provenance stamping ("explicit" for operator, "measured" for governor) |
| `native-kv-host-swap-bytes` | `Engine.SetKVHostSwapByteLimit` (`nativesched_preempt.go:155-185`) | control endpoint → setter; setter already refuses shrink-below-ownership |

**Witness:** `go test ./internal/gateway/ ./internal/modelengine/ -run
"TestControl|TestMaxTotalTokens|TestKVHostSwap" -count=1` + a live
PATCH→setter→per-request-read integration test.

**Anti-masking note:** provenance stamping is the guardrail's enforcement
point — every live write records who wrote it and why, so an adaptive layer
cannot silently move a knob an operator pinned.

## Spine 3 — Posture levers (the operator-facing adaptive layer)

**Goal:** the four postures the objective names become first-class, evidenced
adjustments rather than folklore.

### 3.1 Interactivity vs throughput
- Levers: `max_waiting_seqs` (ScalarConfig, hot), `completion_deadline_ms`
  (hot), `max-total-tokens`/`max-batch-prefill-tokens` (Spine 2), admission
  token budget (Spine 2).
- Posture: interactive = small waiting queue + tight deadlines (favor TTFT);
  throughput = deep queue + generous deadlines (favor tok/s).
- Evidence: perfledger `TTFTP50MS`/`TTFTP99MS`/`DecodeTPSP50` before/after.
- Mechanism: document posture presets in this doc; presets are just named
  ScalarConfig patches, no new machinery.

### 3.2 Prefill/decode ratio
- Levers: `max-batch-prefill-tokens` (Spine 2), admission budget, the
  memory-pressure governor (already closed-loop).
- Posture: prefill-heavy = larger prefill batch ceiling (favor TTFT for
  long prompts); decode-heavy = smaller ceiling (favor decode tok/s; PHL-015:
  concurrent prefill starves decode).
- Evidence: perfledger `PrefillTPSP50` vs `DecodeTPSP50` before/after.

### 3.3 Context-length trade-offs
- Levers: `compact_history_budget`/`compact_anchor_head` (ScalarConfig, hot),
  `native-kv-host-swap-bytes` (Spine 2), `kv-precision` (per-session
  realizable; fleet-wide hot is Tier 3).
- Posture: long-context = q8_0/fp16 KV tier + host-swap headroom + compact
  budget; short-context interactive = f32 exact + no swap.
- Evidence: perfledger `CacheHitShare` + `ByRegime` (warm/cold TTFT split).

### 3.4 P50 vs P99
- Levers: all of the above; the *policy* is which percentile the operator
  optimizes.
- Posture: P50-lean = aggressive batching/deep queues (best average tok/s);
  P99-lean = admission throttling + deadlines that cap queue wait (bounded
  tail).
- Evidence: perfledger P50/P99 fields directly; no new plumbing.

**Anti-masking note:** posture presets are named ScalarConfig patches — they
must never change a default; they change a *running* server. A preset that
turns out generally better becomes a default only via its own witnessed
ticket (see Guardrail).

## Spine 4 — Tier 3 work (real restarts, named but not scheduled)

Named for completeness; each is its own ticket when scheduled:
- `--kv-precision` fleet-wide hot tier (per-session realizability exists;
  fleet-wide needs the Tier 3 Prepare→Load→Warm→Quiesce→Drain lifecycle).
- Draft-model swap (speculative Tier 3).
- Backend/engine swap (Tier 3).
- `--native-context-tokens` resize (Tier 2/3 boundary: KV pool resize).

## Guardrail: adaptation must not mask a worse default

**The rule:** an adaptive/hot layer may only *move* a knob away from its
default when the default itself is the best known general value. If a
default is wrong, fix the default (with its own witnessed ticket); do not
ship a worse default and let adaptation paper over it.

**Canonical case (why this rule exists):** the fp16-KV default on CUDA
regressed long-generation decode ~3.8× in interaction with
`speculative.n_max:0` request shapes [HW-WITNESSED raider-fak 2026-11-06;
receipts cited in `cmd/fak/serve_native_config.go:144-158`]. The fix was NOT
"make kv-precision adaptive so it can dodge the bad case" — it was to revert
the default to explicit f32 (opt-in fp16/bf16) and track the underlying
decode-load fix in ticket `cuda-kv-f16-typed-single-token-decode-load-20261106`.
Adaptation would have hidden the regression behind a moving knob.

**Enforcement mechanisms (all already exist):**
1. **Provenance stamping** — `SetTokenBudgetWithProvenance` records
   "measured" vs "explicit" on every live write; an adaptive write is always
   distinguishable from an operator write.
2. **Canary + rollback** — policy reloads arm a canary
   (`policy_canary.go`); a deny-all streak ≥ `--policy-canary-turns` rolls
   back automatically. ScalarConfig swaps get the same watchdog treatment
   (concept doc §4.2 LKG double-buffering).
3. **Pinned axes** — fit-auto never moves an explicitly-set axis
   (PIN_INFEASIBLE); the same rule must hold for any adaptive layer: an
   operator-pinned knob is not adaptive's to move.
4. **Defaults-change-by-ticket** — a default changes only with its own
   witnessed ticket carrying before/after perfledger evidence; the adaptive
   layer never changes defaults, only running state.

**Test the guardrail, not just the mechanism:** every spine's witness
includes a default-preservation assertion — after a hot swap and a rollback,
the effective config equals the boot default unless an explicit write
changed it.

## Landing sequence

Spine 0 → 1 → 2 → 3 are ordered; Spine 4 is independent. Each spine lands
as its own ticket(s) with the named witness; the inventory doc's per-knob
rows are the checklist (a knob is done when its row's class upgrades to
wired-live in the inventory doc).
