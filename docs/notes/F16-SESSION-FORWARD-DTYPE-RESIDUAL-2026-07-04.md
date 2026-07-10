---
title: "F16 upload-dtype path through the modelbench Session forward —"
description: "Triage note classifying the F16 upload-dtype Session-forward residual (issue 1481) as gen/next and handing off the ready-to-implement seam, not resolving it."
---

<!-- fak:generation-triage curated note. Records the horizon classification + exact
triage for the F16 Session-forward dtype residual (epic 1476 child C4, issue 1481).
This note does NOT resolve the residual; it hands the next worker a classified,
ready-to-implement seam. Deliberately carries no closure-binding issue stamp. -->

# F16 upload-dtype path through the modelbench Session forward — horizon classification

**Topic:** the "Lever 4 residual" from the H100 kernel roadmap — thread an **F16
upload-dtype** device path through the modelbench `Session` forward, alongside the
existing Q8 and Q4_K device paths. Tracked as epic 1476 child C4 (issue 1481).

**Status of this note:** triage/classification only. The forward path is **not**
implemented here. The generation intent arrived unclassified, and the dispatch frame
allowed triage only (classify the horizon, repair labels/milestone, or leave a
clarification note) — so this note classifies the horizon and hands off the seam.

## The seam (verified against the tree, 2026-07-04)

- `internal/model/hal.go:172` `matWeightHAL` routes a weight upload through **Q8**
  (`useHALQ8Weights` → `s.M.q8w`) then **Q4_K** (`useHALQ4KWeights` → `s.M.q4kw`),
  then falls through to `weightHAL` (an **F32** upload). There is **no F16 branch**.
  `lmHeadMatHAL` (`hal.go:193`) mirrors the same two-branch select.
- `internal/model/kv.go` `Session` carries `Quant bool` (Q8, line 59), `Q4K bool`
  (line 75), `MetalQ4K bool` (line 123) — but **no `F16 bool`**.
- The device F16 GEMM already exists at the compute layer (#484, BUILT + GATED):
  `internal/compute/cuda.go:846` `uploadF16` + `fcuda_matmul_f16`
  (`cublasGemmEx` tensor-core HGEMM, F32 accumulate), recorded cosine floor
  `cudaFP16CosineMin = 0.997` (`cuda.go:100`), witness `cuda_fp16_test.go`.

So the only missing piece is a **Session-level device-dtype select** that reaches
the F16 upload — the compute primitive is done.

## Smallest honest next step (the implementation, when a gen-classified frame allows it)

Mirror the Q8/Q4_K branches, do not touch the compute layer:

1. Add `F16 bool` to `Session` in `internal/model/kv.go` (beside `Quant`/`Q4K`).
2. Add `useHALF16Weights()` in `hal.go` — `s.F16 && s.M has F16-tagged weights &&
   s.Backend.Caps().UploadDtype` — and a `weightHALF16(name)` staged upload
   (`weightHALStaged("f16:"+name, mk, compute.F16)`), mirroring `weightHALQ8`.
3. Route it as the first (or Q8/Q4_K-adjacent) branch in `matWeightHAL` and
   `lmHeadMatHAL`.

**Off-GPU witness (this host):** a no-build-tag unit test that the Session select
routes an F16-tagged weight to the F16 upload (not Q8/Q4_K), shaped exactly like
`internal/compute/tf32_enable_test.go` (runs in the default non-cuda `go test`).
Note: native `go test` is blocked on the Windows dev box (OS Application-Control on
freshly-compiled test binaries) — run it under WSL (`./test.ps1`) or CI.

**GPU-gated witness (hardware, not this host):** device-vs-cpuref cosine ≥
`cudaFP16CosineMin` (0.997) on the F16 forward — the same acceptance shape as
`tools/run_484_acceptance_on_gpu.sh`.

## Generation horizon: gen/next

The generation labels/milestones exist in the repo (`gen/now`..`gen/future`,
`generation`; milestone "Generation G1 - Next Gen"). Classification from evidence:

- **gen/next** — a *near-term foundation* that becomes agent-runnable after a gate.
  The F16 Session path lands **inert** on the default non-cuda build (no consumer
  wires an `fak-cuda-f16` bench engine yet) and its *full* witness (device cosine)
  is **GPU-gated**. That is the textbook gen/next signal: "runnable soon, still
  needs a gate / default-exposure proof." It is *not* gen/second-next or gen/future:
  F16 is a shipped compute primitive (#484), not a future-architecture bet.

### Promotion evidence (what would move it toward gen/now)
- The off-GPU routing test landing green (proves the seam) plus an `fak-cuda-f16`
  bench engine consuming it — i.e. a current-product consumer exists.
- The GPU cosine acceptance (≥0.997) passing on an H100 node, retiring the
  hardware-gated unknown.

### Demotion / retirement evidence (what would push it later or close it)
- If the H100 roadmap drops the F16 bench arm (TF32/Q8/Q4_K judged sufficient for
  decode parity), this residual retires — the seam has no consumer.
- If `Caps().UploadDtype` backends never expose F16 in practice, the path is dead
  code and should not land until a consumer is real.

### Invalidating assumption
- This is classified **gen/next** on the premise that the off-GPU routing seam is
  *foundation* (inert until a GPU consumer). **If** the routing seam alone is deemed
  a current-product improvement — e.g. a CPU-ref-checkable bench arm exercises the
  F16 select today — the correct horizon is **gen/now**, not gen/next. Re-check the
  bench-engine consumer before promoting.

## Triage to apply on issue 1481 (could NOT be applied live this session)

Intended by the issue body + this classification; **left unapplied** because the
outward GitHub write is blocked on this host (see below):

- **Add labels:** `model`, `gpu`, `track/A-model-support`, `priority/P2` (issue-body
  ask; all exist), plus `generation` + `gen/next` (this classification). Existing
  `enhancement`/`cuda`/`model-support`/`quantization` are harmless — keep them.
- **Milestone:** `#4 Decode parity` (the issue body's explicit ask — the roadmap
  deliverable milestone). Note the tension: the generation-next *view* wants the
  "Generation G1 - Next Gen" milestone, but GitHub allows one milestone slot and the
  reporter named #4; carry the generation stream via the `gen/next` label and let a
  curator reconcile the milestone slot.

### Why the live write was blocked (host/session capability)
- `gh issue edit` (GraphQL) — the GraphQL quota was exhausted this session (REST
  core still had budget).
- REST `gh api --method POST .../issues/1481/labels` — refused by the fak
  preview-confirm/`ESCALATE` gate (outward-facing write); the `_fak_confirm`
  re-proposal channel is documented-broken in this harness.
- There is **no `fak issue` edit/label/milestone verb** (only `contract`/`cohort`/
  `fanout`), unlike `fak commit`/`fak sync push` which bypass the gate for git.

Net: applying live labels/milestone needs an operator (or a session with a working
confirm channel + GraphQL budget). This note is the durable handoff so the
classification is not lost as a self-report.
