---
title: "GLM-5.2 on fak: the decode path to 10 tok/s (root cause + levers)"
description: "Why the real 753B GLM-5.2 cpu-offload serve decodes at <0.1 tok/s on GPU server, the host expert-kernel ceiling measured on a 32-core dev box, and the prioritized lever decomposition (pure-CPU vs hybrid, batched expert dispatch, a vectorized int8 reducer) that gets the pure fak kernel to 10 tok/s — with the GPU server experiment matrix to confirm it."
---

# GLM-5.2 decode: the path to 10 tok/s on the pure fak kernel

_2026-06-27._ Companion to
[GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md)
and [GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md).
It targets the **real 753B `glm_moe_dsa` serve** (UD-Q4_K_M, ~436 GB), not the synthetic
reduced-layer `glmdsatput` micro-number. The goal: drive sustained **decode ≥ 10 tok/s**
on the **pure fak kernel** on GPU server (8× datacenter GPU, sm_80, ~2 TB host RAM).

## TL;DR

The cpu-offload hybrid (`--backend cuda --cpu-offload-experts`) measured **< 0.1
tok/s** (prior session — 16/32-token smokes timed out at 0 tokens). The number is
**not** a kernel-arithmetic wall and **not** a load-speed problem. The leading
cause is **structural device glue**: the GLM-DSA forward issues ~12 *synchronous*
device↔host round-trips **per layer** (one `cudaMemcpy`+`be.Read` stream-sync per
dense projection), and at ~92 layers that is **~1100 synchronous round-trips per
decoded token**. The expert kernel quant lever (int8 vs f32 dequant) is real but
**secondary** (~1.3–1.8×).

So the climb from < 0.1 to 10 tok/s decomposes into three independent levers, in
priority order:

| # | Lever | Est. gain | Where | Code? |
|---|---|---|---|---|
| 1 | **Drop the device glue** — serve pure-CPU (no `--backend`), or make the dense forward device-resident | ~30× | serve config / future device-resident forward | none (config) |
| 2 | **Batch the MoE expert dispatch** — one parFor across all active experts per proj, not ~24 tiny parFors/layer | ~1.8× | `internal/model` host expert path | yes |
| 3 | **Vectorize the int8 reducer** — AVX2/AVX512-VNNI Q4_K / K-quant GEMV (scalar caps ~0.55 GiB/s/core) | ~2–4× | `internal/model` amd64 kernel | yes (asm) |

Levers 1×2 alone land in the **~5–8 tok/s** band on a 256-core host (estimate
below); clearing 10 reliably wants lever 3 (which is also why llama.cpp's CPU
GLM-class decode beats fak's today — its AVX512-VNNI kernels).

## UPDATE (same day) — re-measured against current trunk: lever 3 is PARTLY SHIPPED

The scalar ceiling numbers below were measured against the session-start archive
(HEAD `ad7b8a13`). Current trunk (`38806053`) already carries **AVX2/AVX512 `VPMADDWD`
int8 reducers for BOTH Q4_K and Q5_K** (`quant_amd64_q4k.s`, `quant_amd64_kquant.s`,
dispatched by `q4kReduceRow`/`q5kReduceRow` when `qtier >= tierAVX2`) — a peer landed
**most of lever 3** after the first measurement. Re-measuring on the same box (now
`qtier=AVX512`):

- batched int8 = **11.26 GiB/s** on 32 cores (was 7.4 scalar), **2.09× over f32** (was 1.3×);
- pure-CPU decode ceiling ≈ **~8.5 tok/s @ 256 cores** (8× extrapolation; batched int8 AVX512) —
  **near 10**, not the ~5–6 the scalar estimate implied. The *current, un-batched* serve
  (no lever 2) is ~`/1.8` of that ≈ **~4.7 tok/s @ 256 cores**.

**Corrected lever picture:** lever 3's SIMD reducer (AVX2/AVX512 `VPMADDWD`) is **shipped**;
the remaining vectorization headroom is **AVX512-VNNI** (`VPDPBUSD`, int8×int8→int32 in one
op vs the two-step `VPMADDWD`) — another ~2–4× on a VNNI host. So the top *implementable*
lever is now **lever 2 (batch the expert dispatch, ~1.8×)**: it is the difference between
the ~4.7 tok/s current serve and the ~8.5 tok/s ceiling — and it is bit-identical to the
proven path. Caveat: these are THIS box's numbers; GPU server's host-CPU tier (AVX2 vs AVX512 vs
VNNI) and the real active-params/token set the actual figure — the matrix-row-C measurement
still decides it.

## UPDATE-2 (same day, later) — lever 3 fully landed: AVX512-VNNI reducers shipped

The VNNI headroom UPDATE-1 named is now **shipped** for both dominant GLM-5.2 expert quants,
each bit-identical to the scalar reference (every reduction carries an asm-matches-scalar test):

- **Q4_K** (gate/up): `VPDPBUSD` (unsigned nibble × signed qx → int32 in one op) added as the top
  tier in `quant_amd64_q4k.s`, dispatched by `q4kVNNI` (CPUID `(7,0):ECX` bit 11). Witnessed
  `c9ae28e9`.
- **Q5_K** (down): the same `VPDPBUSD` inner dot keyed on `q5kUseVNNI`, sharing the AVX2 path's
  qh-bit unpack and outer loop (no duplication). Witnessed `fa585c70`.

**Witnessed tiers** on a GLM-5.2-shaped expert (out=2048, in=6144), single worker, Zen 5
(AVX512-VNNI), via `BenchmarkQ4KInt8GEMV` / `BenchmarkQ5KInt8GEMV` with `FAK_QKERNEL` pinning:

| Quant | scalar int8 | AVX2 | VNNI | VNNI vs scalar |
|---|---|---|---|---|
| Q4_K | ~10.4 ms/op | ~1.17 ms/op | **~0.875 ms/op** | **~11.9×** |
| Q5_K | ~10.6 ms/op | ~2.94 ms/op | **~2.42 ms/op** | **~4.4×** |

So lever 3 is **done** on amd64: AVX2 is the floor (every CPU-server-class host has it), VNNI auto-lights
on Cascade Lake / Ice Lake / Zen 4+ via CPUID. The Q4_K win (11.9×) is far above the original
~2–4× estimate — Go's scalar int8 path emits per-byte sign-extend + scalar `imul`, so SIMD buys
much more than the f32 comparison implied. The aggregate forward gain is bounded by Q5_K's smaller
win (its cost is the qh unpack, not the dot) and by the non-expert work, but the expert GEMVs —
the #971 wall — are no longer scalar.

**Remaining to the goal (kept honest — `not yet`):**

- The top *unshipped* lever is now **lever 2 (batch the expert dispatch, ~1.8×)** and the
  pure-CPU **serve config (lever 1)** — both are the path from the per-kernel wins to sustained
  decode tok/s.
- The **end-to-end CPU server decode tok/s is NOT yet measured** — the witness is host-gated and the
  Slack bridge to CPU server was unresponsive this session (120 s ping timeout; GPU server round-trips also
  empty). Matrix row C (pure-CPU + `FAK_KQ_INT8=1`, read sustained decode after prefill, A/B
  `FAK_QKERNEL=scalar` vs default) still decides the real figure.

  **CPU server witnessed (this session, via the control bridge):** AMD **EPYC 7742 (Rome/Zen2), 256
  cores, 1007 GB RAM / 462 GB avail, AVX2-only (no AVX512/VNNI)**. So on CPU server the reducers run the
  **AVX2 tier** (Q4_K ~8.9×, Q5_K ~3.6× over scalar — the VNNI tier needs Cascade Lake / Ice Lake /
  Zen 4+; CPU server is older). Scaling UPDATE-1's batched-int8 ceiling to AVX2 on 256 cores keeps the
  pure-CPU + batched estimate **near 10**, but it is an estimate. The e2e witness is **provisioning-
  gated, not bridge-gated — and every blocker is removable ON CPU server (no scp, no Slack binary
  transfer):**
    - **`go` is missing on CPU server, but it can be ADDED on-box:** `git` + internet both work
      (github.com and huggingface.co return HTTP/2 200), so `wget` the go.dev linux-amd64 tarball,
      extract it, `git clone` fak, and `go build ./cmd/fak` natively on the EPYC — the 26 MB binary
      never has to cross Slack.
    - **the ~436 GB GLM-5.2 q4 GGUF fits on `/mnt/nvme-glm` (3.1 TB free)** — root `/` is 95% full
      (~22 GB), but the **nvme drive** (plus `/projects` 3.6 T, `/home/mplservice` 8.9 T) is the
      staging target; pull the GGUF straight from HF to `/mnt/nvme-glm/glm52-q4/`.
    - then a pure-CPU serve (`FAK_KQ_INT8=1`, no `--backend`) + decode is the remaining step.

  **ATTEMPTED on CPU server (2026-06-28 ~06:00Z) — hit a hard, quantified RAM-capacity wall.** Built fak
  on-box (Go installed to `/mnt/nvme-glm/go`, cloned + built `/mnt/nvme-glm/fak-bin`; the GGUF is at
  `/mnt/nvme-glm/glm52-q4/UD-Q4_K_M/`), then launched `fak-bin serve --gguf <shard1> --addr :8077
  --context-budget-tokens 2048` (`FAK_KQ_INT8=1`). It refuses with a typed `FitTooBig` every time:
  `weights 433.82 GiB + kv 1.04 GiB needs 434.91, host has ~394 GiB`. Cause: `cmd/fak/serve.go:713
  serveGGUFHostHeadroom = 0.15` requires `weights ≤ MemAvailable × 0.85`; CPU server's MemAvailable is
  ~464 GiB, ×0.85 ≈ 394 → refuse. And the headroom guards a REAL cost — GLM-5.2 UD-Q4_K_M is **~458
  GiB RESIDENT** in fak's path (#974 struct overshoot over the 433 on-disk), so even at zero headroom
  the ~458 GiB barely fits ~464 GiB free and would risk an OOM mid-load. CPU server's remaining RAM is held
  by a protected 66-day SWE-bench eval that must not be killed. **So the e2e fak-kernel decode tok/s
  on CPU server is RAM-capacity-gated:** GLM-5.2 q4's ~458 GiB resident footprint does not safely fit
  CPU server's ~464 GiB free alongside the protected eval, and fak (correctly) loads weights resident with
  no mmap/lazy path. Unblocking needs ONE of: a host with ≥ ~520 GiB free (CPU server idle, or a GPU server host's
  RAM), a fak mmap weight-load path (doesn't exist), or a smaller GLM quant (q3/q2 ~250–330 GiB —
  none staged; a multi-hundred-GB download). The AVX2 reducers (CPU server's tier) are what would run once
  it fits; the microbench numbers above are the proven kernel witness.

## What is MEASURED vs INFERRED (kept honest)

**Measured:**

- The hybrid serve decodes **< 0.1 tok/s** on GPU server (prior session; smoke `max_tokens 8`
  timed out at 120 s ⇒ > 15 s to first token). Witnessed in the GPU server collection note.
- The **host expert-GEMV kernel ceiling** (this session, 32-core amd64 dev box, the
  same `!arm64` scalar path GPU server's host CPUs run; reproducible via
  `BenchmarkGLMExpertDispatch{Looped,Batched}` in `internal/model`):
  - single-core Q4_K GEMV ≈ **0.55 GiB/s** (int8 ≈ f32 single-core — the int8 win is
    parallel-only, from quantizing the activation once and reusing it across rows);
  - parallel scaling is **poor**: a per-expert loop of GEMVs reaches only **~6–9×** of
    32 cores; **batching all experts into one parFor recovers ~1.8×** (4.3 → 7.4 GiB/s
    int8, still only ~14× of 32 cores; the looped path also does ~7× the allocations);
  - the int8-vs-f32 *aggregate* speedup at MoE-layer scale is **~1.3×** scalar.
- The forward issues **~12 device round-trips per layer** in the hybrid (code-counted:
  `glmDsaIndexStep` 4 projections + `indexSelect`, `glmDsaAppendAttentionKV` 4
  projections, `glmDsaAttendCached` `sparseAttend` + `o_proj`, `glmMoeFFN` router — the
  experts themselves are host-resident in the split). `q_a_proj` is even computed
  **twice** per layer (index step *and* kv step).

**Inferred (the leading hypothesis, not yet directly timed):** that the ~1100
synchronous round-trips/token dominate the hybrid's < 0.1 tok/s. Each `backendKernel.mul`
does `uploadHostF32Class` (H2D) → `MatMul` → `be.Read` (D2H **stream-synchronize**); a
contended GPU makes each sync ms-scale, and ~1100 × ~10 ms ≈ 11 s/token ≈ 0.09 tok/s —
which matches the measured number. The code comment names the design explicitly: *"The
device↔host copy per GEMM keeps the glue simple (correctness-first); a fully
device-resident GLM-DSA forward is the next slice."* **Next checkable step:** a per-op
timing pass on GPU server (decode-only, after prefill) to confirm the round-trip share before
investing in lever 1's device-resident variant.

## Why not "resident across all 8 GPUs"?

The 436 GB model would fit in 640 GB of aggregate VRAM, but fak's CUDA backend is
**single-GPU** today: `fcuda_init` pins `cudaSetDevice(0)`, `DeviceMemory()` reports one
GPU's `totalGlobalMem`, and there is **no** device-side collective (NCCL/RCCL) — the
tensor-parallel seam is shipped but CPU-only (hardware-gated, #295). So multi-GPU
resident sharding is a separate, larger effort; it is **not** the near-term path to 10
tok/s. The near-term path keeps the experts on the host (where they already are under
`--cpu-offload-experts`) and removes the per-token tax around them.

## The pure-CPU ceiling estimate (GPU server host)

Treating the experts as a host weight stream (they dominate the parameter count), and
using the measured batched-int8 throughput:

- active expert stream/token ≈ `(K + shared) × moe_layers × per-expert-bytes`. At
  estimated GLM-5.2 dims (H≈5120, expert-intermediate≈1536, K≈8, ~89 MoE layers) this is
  **~10 GiB/token** (dims are estimates — the real values come from the GGUF header; the
  tok/s scales **inversely** with this number).
- batched-int8 host throughput measured **~7.4 GiB/s on 32 cores** ⇒ ~0.7 tok/s here.
  The kernel is **compute-bound** (far below memory bandwidth), so it scales with cores
  until bandwidth saturates: a 256-core GPU server host at ~8× ⇒ **~5–6 tok/s** (lever 1+2,
  scalar int8). Lever 3 (vectorized reducer, ~2–4×) carries it past 10.

This is an **estimate**, not a serve measurement — the real number needs a free host
(see matrix). But it already says the right config is **pure-CPU, not the hybrid**: the
hybrid pays ~1100 device syncs/token to run a *small* fraction of the FLOPs on the GPU,
while the experts (the bulk) sit on the host either way. Serving pure-CPU on GPU server's host
cores also **frees the GPUs** for other users — a good-neighbor bonus on a shared box.

## GPU server experiment matrix (run when a host is free)

One GLM-class load at a time; gate on `free -g`; NVMe-first weights
(`/mnt/sglang_dv3/glm52-q4/`); the GPUs/ports 8000-8001 may belong to another user.

| # | Config | Command sketch | Measures |
|---|---|---|---|
| A | hybrid, default | `fak serve --gguf S1 --backend cuda --cpu-offload-experts` | reproduce < 0.1 tok/s baseline |
| B | hybrid + int8 experts | `FAK_KQ_INT8=1 … --backend cuda --cpu-offload-experts` | int8 lever on the hybrid (expect small — glue-bound) |
| C | **pure-CPU + int8** | `FAK_KQ_INT8=1 fak serve --gguf S1` (no `--backend`) | the lever-1 win: decode tok/s with NO device glue |
| D | per-op decode timing | C + a decode-only timing pass | confirm the round-trip share / kernel share |

Decode tok/s should be read **after** prefill completes (the goal is sustained decode,
not first-token latency). Capture `fak_*` /metrics + `FAK_WORKERS`/`FAK_BUDGET`.

## Status

**Not yet at 10 tok/s.** Root cause localized (device glue, with the host-kernel ceiling
quantified); levers prioritized; the confirming measurement is host-gated (GPU server and the
CPU-only CPU server node were both in active peer use this session). Next: run matrix row C/D
on a free host to validate lever 1, then implement levers 2 and 3.
