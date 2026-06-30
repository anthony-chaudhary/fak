---
title: "GLM-5.2 native on fak — first throughput, next steps, and the honest benchmarking plan"
description: "The first native GLM-5.2 (glm_moe_dsa) decode/prefill tok/s measured on a GPU server, the VRAM-leak fix that unblocked it, the prioritized next steps, and a category-honest framework for comparing fak to llama.cpp / vLLM / SGLang."
---

# GLM-5.2 native on fak: first throughput, next steps & benchmarking plan

_2026-06-25._ Living companion to
[native-753b-track-staged-plan](native-753b-track-staged-plan.md). It records the
**first native GLM-5.2 (`glm_moe_dsa`) decode/prefill tok/s** measured on a GPU server, the
bug fixed to get there, what is next, and — most important — **how to compare fak's
number to the field without a category error**.

## 1. What shipped this cycle

- **Forward correctness** — GLM-5.2's DSA forward is bit-exact on fak's own CUDA
  kernels: cosine `1.000000`, argmax-exact, re-witnessed at HEAD `f39796e`
  (sm_80, datacenter GPU). See the `glm-gpu-witness/1` records under
  `experiments/glm-gpu-witness/`.
- **First native throughput** — `cmd/glmdsatput` times the native `glm_moe_dsa`
  decode/prefill on a real compute backend (the CUDA datacenter GPU path). Numbers in §2.
- **Leak fix `b68a182`** (shipped; `dos verify fak model` = SHIPPED) — the GLM-DSA
  device seam leaked one resident VRAM buffer per prefill position: the per-call
  activation/operand uploads (`backendKernel.mul`/`sparseAttend`/`indexSelect`) were
  never `Free`d, and the bespoke `Prefill`/`Step` loop never `Recycle`d transient
  op-outputs (decode's HAL path does both every token; this path bypassed it). So a
  512-token prefill grew `g_live` until `cudaMalloc` failed — and `fcuda_malloc`'s
  `CK` macro swallowed the real error. Fix: `Free` per-call uploads, `Recycle` per
  token, and surface the true `cudaMalloc` reason. No-op on cpu-ref (host forward
  stays byte-for-byte). Validated: 4/6 sweep configs that previously OOM'd now run.

## 2. First native GLM-5.2 tok/s (GPU server, datacenter GPU, sm_80)

> **HONEST SCOPE.** `glmdsatput` builds a **synthetic, reduced-layer, dense-FFN**
> `glm_moe_dsa` (real architecture + real per-layer dims, **random** weights, **no
> MoE experts**). The tok/s is fak's per-token **device-kernel cost at a fits-one-GPU
> scale** — *not* the 753B serving rate. The `scope` field travels in every record so
> the number cannot be quoted out of its caveat.

P=512 prefill, **Q8_0**, decode-reps 5 (the committed sweep, head `b68a182`):

| layers | hidden | topk | decode tok/s | prefill tok/s |
|---:|---:|---:|---:|---:|
| 8  | 2048 | 256 | **26.53** | 27.02 |
| 8  | 2048 | 512 | 15.80 | 22.27 |
| 16 | 2048 | 256 | 13.44 | 12.95 |
| 16 | 4096 | 256 | 8.49  | 6.85  |

Small-context (P=8 / 64, the bisection that pinned the leak):

| precision | P | decode tok/s | prefill tok/s |
|---|---:|---:|---:|
| f32  | 8  | 49.78 | 23.27 |
| Q8_0 | 8  | 56.61 | 41.82 |
| Q8_0 | 64 | 40.38 | 40.65 |

Decode tok/s falls with depth, width, and **context length** (DSA attends the
selected keys per token, so a 512-context decode does more work than a P=8 one). The
4-config `glm-throughput/1` record persists on the box at
`<private-scratch>/glmw116b7ed250b7.result` — landing it is a P1 below.

## 3. Open issues / next steps (prioritized)

- **P0 — DSA-kernel illegal-memory-access at the largest configs.** With the leak
  gone, the two biggest sweep configs (32-layer; hidden-5120/40-head) now run far
  enough to hit a **separate, pre-existing** kernel bug:
  `fak-cuda: cudaMalloc(...) failed: an illegal memory access was encountered`
  (surfaced by the new error path — it is *not* OOM, and it is *not* in the changed
  code; the `k_dsa_*` kernels were untouched). **Next:** single-variable on-box
  bisection (vary layers / hidden / heads / topk one at a time) to pin the kernel
  out-of-bounds, fix it, and re-run the full 6-config sweep clean.
- **P1 — Runner GPU auto-pick.** `private throughput runner` hardcodes
  `--gpu 0`; a transiently-busy GPU 0 produced a false "allocation failed" earlier.
  Select the freest GPU (`nvidia-smi` min-used) before the sweep.
- **P1 — Land the throughput record.** Scrubbed a100 rollup public + raw private
  (the witness pipeline), committed under an `add(experiments)` trailer so
  `dos verify` binds it.
- **P2 — Couple the harness to real weights.** `glmdsatput` uses random synthetic
  weights. `cmd/modelbench` is already arch-blind; point it at a real `glm_moe_dsa`
  GGUF (the native-753B P1 loader is done) for a **real-weight** tok/s — the next
  rung toward a number that is not synthetic.
- **The wall — 753B native serving (multi-month).** Device NCCL/TP collective +
  MLA-aware sharding + paged experts. See the staged plan.

## 4. Benchmarking comparisons — the honest framework

### 4.1 The category boundary (read this first)

Three numbers exist; only some are comparable:

- **fak native kernel cost (this note):** synthetic `glm_moe_dsa`, per-token device
  cost, fits-one-GPU, dense-FFN (no MoE). **Not** 753B serving.
- **llama.cpp 753B baseline:** the **real** GLM-5.2 753B Q4_K_M (8 shards, ~425 GB),
  CPU-offloaded (`--n-cpu-moe`), **2.62 tok/s single / 4.84 tok/s agg @ concurrency 2**
  on GPU server (8-GPU datacenter server + ~1 TB host RAM). Real model, real serving.
- **vLLM / SGLang GLM-5.2:** stock-engine serving; the real DSA path needs sm_90
  (Hopper). A stock-SGLang comparison server already runs on GPU server (Qwen today).

> **Do NOT put "fak 26 tok/s" next to "llama.cpp 2.62 tok/s".** That is a category
> error: fak's number is a *synthetic, dense-FFN, no-MoE, fits-one-GPU kernel cost*;
> llama.cpp's is a *real 425 GB MoE checkpoint streamed off host RAM*. Different
> model, different work, different scale. Honesty rule: never claim "fak serves 753B",
> and never compare the synthetic kernel number to a real-serving number.

### 4.2 What a fair comparison requires (apples-to-apples)

Hold {model weights, hardware, precision, context, batch} equal and report the same
metrics for every engine:

- single-stream **decode** tok/s and **prefill** tok/s, **TTFT**
- **throughput @ concurrency** (the real serving metric)
- **VRAM / host-RAM** footprint
- **correctness** vs the reference (cosine / argmax) — fak's actual differentiator

### 4.3 Comparison ladder (each rung a real apples-to-apples run)

- **B1 — clean synthetic kernel curve** (after the P0 fix): full 6-config `glmdsatput`
  sweep — fak's device-kernel cost curve. *(partial: 4/6 today.)*
- **B2 — real tiny checkpoint:** fak vs llama.cpp vs vLLM on the **same** small real
  `glm_moe_dsa` checkpoint (`yujiepan/glm-5-tiny-random`, already an oracle here). The
  first honest cross-engine number, at small scale.
- **B3 — real mid checkpoint, one GPU:** a quantized GLM that fits one datacenter GPU — fak vs
  llama.cpp vs vLLM/SGLang on decode/prefill/throughput@conc **+ correctness**.
  - **B3 (CPU host):** the same apples-to-apples cross-engine row on a **CPU-only** server
    (no GPU) for the full ~433 GB GLM-5.2 UD-Q4_K_M — fak-native CPU vs llama.cpp (mmap,
    `-ngl 0`) on decode/prefill tok/s **+ correctness + safety/reuse**. The bench node is
    now registered (`cpu-server-a`: 256-thread x86_64, ~1 TB RAM) and the first attempt is
    committed as an honest *pending/negative* record —
    `experiments/benchmark/runs/by-machine/cpu-server-a/20260627T000000Z-glm52-cpu-wedge/`
    (#976). Both engines' tok/s + the correctness column remain **open and host-gated**: the
    memory-safe llama.cpp baseline is pending host recovery, and the fak-native all-resident
    CPU serve wedged on host RAM (blocked by **#974**; the load-curve datum and the
    missing CPU memory-fit pre-flight are recorded in
    [GLM52-FAK-NATIVE-CPU-SERVE-MEMORY-WEDGE-2026-06-27](GLM52-FAK-NATIVE-CPU-SERVE-MEMORY-WEDGE-2026-06-27.md)).
- **B4 — 753B (the wall):** fak (once multi-GPU TP + paged experts land) vs the
  llama.cpp 2.62 tok/s baseline vs vLLM/SGLang on sm_90. The only comparison where a
  753B number means anything.

### 4.4 What to benchmark *for*

fak's pitch is not raw tok/s (the correctness-first quant kernels are honestly slower
than tensor-core SGEMM — the win is VRAM/bandwidth, not FLOPs). It is:

- **bit-exact forward** (cosine 1.0) — a correctness guarantee the stock engines do
  not make;
- the **adversarial capability gate + tamper-evident journal** layered on serving;
- **cross-worker / cross-session KV reuse** (see [SOTA-COMPARISON.md](../../SOTA-COMPARISON.md)).

So every comparison table should carry a **correctness column** and a
**safety/reuse column**, not tok/s alone.

## 5. Reproduce

```sh
# Forward-correctness witness (cosine 1.0):
python private witness fetcher GPU server --runner private witness runner
# Native throughput sweep (this note's numbers):
python private witness fetcher GPU server --runner private throughput runner
# Local single run on a CUDA node:
go run -tags cuda ./cmd/glmdsatput -layers 8 -hidden 2048 -backend cuda -decode-steps 64 -json
```
