# Plan: #9050 — top ten fak-native Qwen laptop improvements

- Owner: Codex coordinator, 2026-08-25
- Umbrella: #9050; parent context #8011
- First spine: #8692; baseline prerequisite #8486
- Centrality: Core
- Work shape: finite phased deliverable; one measured performance leaf per phase
- Target: Dell XPS 17 9730, RTX 4070 Laptop 8,188 MiB (SM89), i9-13900H 14C/20T, 64 GiB DDR5-4800

## Hero metric

Ship `10 / 10` laptop-specific Qwen leaves with positive net-true end-to-end movement, preserved quality, and a receipt naming `engine=fak-native`. A HOLD, REVERT, or EXCLUDE result remains useful evidence but does not advance the numerator; replace it with the next profile-selected lever.

Current result: `0 / 10 KEEP`.

## For / Problem / Today / Better because / Witness

- **For:** operators who need a Qwen checkpoint larger than laptop VRAM to run locally through fak's native engine.
- **Problem:** the exact local Qwen3.6-27B Q4_K_M artifact is 15.40 GiB while the GPU has 8 GiB, and current Qwen native execution has no bounded dense CPU/GPU layer placement.
- **Today:** current HEAD recognizes the artifact as `qwen35` with 851 tensors but reports `FIT_UNKNOWN`; Qwen hybrid decode is excluded from graph replay and the generic batch path, dense Q5/Q6 residency is disabled at serve load, and several CPU Q4 optimizations are present only as guarded or dormant hooks.
- **Better because:** the exact model becomes runnable first, then each following change is chosen from the measured mixed-path bottleneck rather than from an unprofiled kernel wish list.
- **Witness:** one independently rerunnable receipt per kept phase with fixed model/prompt/context, engine and fallback identity, quality gate, cold/warm timing, TTFT, prefill/decode rates, peak host RAM/VRAM, transfer/synchronization time, and power/temperature envelope.

## Problem checks

- P1, quality: preserve model identity, tokens, context, recurrent state, KV state, and the phase-specific numerical oracle.
- P2, accounting: count load, host/device transfer, synchronization, verification, memory, and thermal effects in the end-to-end result.
- P3, operating envelope: pin artifact, quant, hardware, engine, placement, fallback policy, and trial count; fail closed on a different engine.
- P4, proof: require a replayable command, machine-readable receipt, regression tests, pushed commit, and rollback for every KEEP.

## Current state

- Trunk was synchronized at `7f5a883e62ec2ff060f1a4f9a4a541484836fde0` when the program began.
- Exact artifact: `C:\Users\antho\models\Qwen3.6-27B-Q4_K_M.gguf`, 16,547,398,784 bytes.
- Current-HEAD header preflight: `READY`; architecture `qwen35`; 851 tensors; estimated load 16,536,406,016 bytes (15.4007 GiB); `FIT_UNKNOWN` because the Windows control build had no CUDA capacity backend.
- Historical same-machine fak-native control: Qwen2.5-3B Q8 at 25.14 decode tok/s and about 25 tok/s prefill, from v0.21/CUDA 12.6; it is stale context, not current proof.
- A live llama.cpp server currently holds the exact artifact with 20 GPU layers. It is an explicit reference-only confounder and cannot count as fak-native evidence.
- WSL exposes the GPU, but the default Ubuntu image currently lacks the repository's build/runtime tools; `.wslconfig` limits it to four CPUs and 16 GiB.
- #8692 is assigned and active in a managed detached worker worktree. #8711 is closed as its duplicate.

## Ordered phases

### 0. Register the baseline (#8486)

Record the current laptop and current-HEAD header witness. Add the representative three-trial baseline and catalog/ledger registration before crediting the phase. A fully resident smaller Qwen control may establish the current native toolchain while the exact 27B mixed path is unavailable.

### 1. Bounded contiguous dense placement (#8692)

Make the exact 27B model runnable without engine fallback by placing a bounded contiguous layer band on the GPU and the remainder on CPU. Account for both memory domains, perform explicit activation handoffs, identify placement in the receipt, and fail closed when the requested plan cannot be honored.

### 2. Attribute the mixed path (#8393)

Profile the landed path and name the dominant kernel, transfer, or CPU-resident cost. The output is a replayable profile and a justified next leaf, not a generic optimization list.

### 3. Conditional Q4 MMVQ (#8635)

If P2 selects Q4 GEMV, advance the Q8_1/DP4A MMVQ arm and compare exact quality plus end-to-end movement. If another cost dominates, record EXCLUDE for this envelope and replace the phase candidate.

### 4. Graph-safe Qwen hybrid decode

Create the smallest graph-replay leaf around the explicit Qwen disable in `internal/model/hal.go`. Patch mutable positions, pointers, KV and recurrent state; prove eager/graph parity and measure launch-overhead movement.

### 5. Remaining GDN recurrence (#8820 re-scope)

Target only the sequential recurrence still inside the GDN kernels. The outer Qwen prompt loop is already panel-shaped and is outside this rewrite.

### 6. Direct dense Q5_K/Q6_K residency

Use the existing loader/HAL/kernel seams to keep supported dense K-quants resident in the Qwen CUDA path. Remove the serve-time disable only for proven types and shapes, with memory and parity witnesses.

### 7. Real-weight CPU Q4 integer gate

Exercise the existing `FAK_KQ_INT8` AVX2/VNNI reducer against real Qwen weights. Promote selection only when the quality oracle and mixed-path end-to-end result are positive.

### 8. Extract-once Q4 prefill

Replace the amd64 extract-once prefill hook's always-false stub with the smallest measured production path. Attribute extraction cost and reuse break-even.

### 9. Affine-split AVX2 decode

Wire the tested affine-split Q4_K decode kernel into production selection. Retain it only when exact A/B and full decode receipts are positive.

### 10. Bounded long-context KV arm

Exercise one quality-gated KV precision/device-paging arm under #8321/#8395 only if the long-context profile selects memory pressure as binding. Otherwise substitute the next attributable Qwen lever and keep the hero metric at ten actual KEEP results.

Only P0 and P1 are armed initially. Before arming P2-P10, run self-query, raw repository grep, all-state issue deduplication, live attribution, and exact source/revision/license pinning. Each ship-alone phase receives its own issue or is reconciled to the named existing issue.

## Prior-art route

- DEFAULT: fak-native execution with no silent llama.cpp or other-engine recovery.
- ADAPT: llama.cpp MIT layer placement and `ggml-quants`; Marlin Apache-2.0 fused INT4 MMA when selected by profile; ExLlamaV3 MIT graph patch/replay; KIVI MIT KV quantization; FlexLLMGen Apache-2.0 explicit placement planning.
- WATCH: Qwen MTP self-speculation until accepted-token yield repays draft and verification work inside the 8 GiB envelope.
- EXCLUDE: generic GTX/Pascal/Turing defaults for this Ada SM89 machine and any result that changes model, context, quality target, or engine.

Pinned study ledger (observed 2026-08-26 UTC):

| Source | Revision | Candidate mechanism | License |
|---|---|---|---|
| llama.cpp | `d222767c7a6516559a3f49e7721b6c6b1acc87b4` | Layer offload, pinned host allocation, `ggml-quants` oracle | MIT |
| ExLlamaV3 | `82c0f73690fa2f6586204ad4221f9fc0930ce9c9` | CUDA graph parameter patch/replay and shape/device tuning | MIT |
| Marlin | `1f25790bdd49fba53106164a24666dade68d7c90` | W4A16 `cp.async` plus tensor-core MMA | Apache-2.0 |
| KIVI | `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` | Per-channel K and per-token V quantization | MIT |
| FlexLLMGen | `004ffef82b46e8dc8685c55d0cdda650bdaf1269` | Explicit GPU/CPU/storage placement planning | Apache-2.0 |
| Qwen3.8 | `2ea10dc725823bf7c3e21ce8557cbe15245132ae` | Official hybrid GDN/full-attention architecture | Apache-2.0 |

Kernel commits must cite the exact upstream revision, path, and license and include the applicable `Prior-art:` trailer.

## Gold-plating boundary

No new inference engine, silent fallback, whole-runtime port, multi-GPU scheduler, MoE scheduler, or unprofiled kernel rewrite. Keep one representative Qwen path runnable. Serving breadth and productization fan out only after the smallest measured spine ships.

## Execution log

- 2026-08-25: inventoried the laptop, exact artifact, active reference server, WSL limits, current trunk, and stale historical control.
- 2026-08-25: built the current `modelbench` outside the repository and captured the exact artifact's header-only preflight.
- 2026-08-25: ran the native SOTA matrix and pinned external primary-source candidates; no upstream code copied.
- 2026-08-25: searched all issue states, closed #8711 as a duplicate of #8692, assigned #8692, and opened umbrella #9050.
- 2026-08-25: audited current Qwen source and revised P2-P10 to match actual gaps rather than speculative features.
- 2026-08-25: acquired the `internal/model/**`, `cmd/fak/**`, `cmd/modelbench/**`, and issue-witness lane and launched #8692 in the sanctioned detached worker worktree.

## Completion audit

- Hero metric reads `10 / 10 KEEP`; every credited phase is closed by a pushed issue-bound commit.
- Every receipt says `engine=fak-native`, zero fallback, fixed model/prompt/context, and passes its quality threshold.
- Final exact-artifact table includes cold/warm load, TTFT, prefill, decode, peak host RAM, peak VRAM, transfer/synchronization time, power, and temperature.
- Scope-correct WSL tests and `fak validate --mine` are green per leaf; committed trunk passes `fak-dev ci-preflight`.
- Changed module versions are stamped and cited as `module@rev`.
- Every residual is deduplicated and filed, or explicitly excluded with a reopening witness.

## Release and rollback

Do not release for P0 or documentation-only state. Release only when an implementation phase reaches its committed test and laptop witness boundary. Each phase is one issue-bound commit and is independently revertible; a reverted phase loses KEEP credit.
