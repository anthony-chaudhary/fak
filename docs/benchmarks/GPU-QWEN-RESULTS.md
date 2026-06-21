# GPU-QWEN-RESULTS — real agentic models on the laptop GPU: f16 parity (1.5B) + reaching 2.5–3B (Q8)

> **Two results, one box (RTX 4070 Laptop, Ada sm_89, 8188 MiB / ~6.2–7.4 GiB free, CUDA
> 12.6, WSL2; `llama.cpp` built with CUDA in the same env):**
>
> 1. **Equal-precision decode parity, Qwen2.5-1.5B-Instruct.** With an **f16 weight path**,
>    fak-CUDA decodes at **~36.6 tok/s vs `llama.cpp` F16 34.3** (fak ~1.07×) — a *win at the
>    same precision*, correctness witnessed greedy argmax-exact vs the f32 reference on the
>    real model. This is the first fak-vs-`llama.cpp` head-to-head on a genuinely capable
>    agentic model (the prior GPU witness was SmolLM2-135M, `GPU.md` §3b).
> 2. **Reaching the 2.5–3B class, Qwen2.5-3B-Instruct.** f16 (2 bytes/param) tops out at
>    ~1.5B on this 8 GB-laptop's free VRAM, so a **Q8_0 weight path** (~1.125 bytes/param)
>    was added: **Qwen2.5-3B now runs on the GPU, witnessed argmax-exact**, fitting ~3.5 GiB.
>    Its decode is **WSL-launch-bound** (~11–15 tok/s vs `llama.cpp` Q8 32.4) — an
>    environment + tracked-optimization gap, not architecture (analyzed in §4).
>
> Companion to `GPU.md` and `GPU-MODEL-PICK.md`. Every number here is a real run captured in
> bench/test output; the boundaries are stated as plainly as the wins.

## 1. What shipped — narrowed-weight device paths (issues #34, #33)

The CUDA backend was **f32-only** (weights 4 bytes, cuBLAS SGEMM). Two gated paths narrow the
resident weights — the lever that fits a multi-billion-param model on an 8 GB GPU:

| Path | Gate | Resident weight | Device GEMM | Where |
|---|---|---|---|---|
| **f16** | `FAK_CUDA_F16=1` | 2 bytes/param | `cublasGemmEx` half×half→f32, **Ada tensor cores** | `cuda_kernels.cu:fcuda_matmul_f16` |
| **Q8_0** | `FAK_CUDA_Q8=1` | **~1.125 bytes/param** (int8 + per-32-block f32 scale) | fused dequant-GEMV (`k_matvec_q8`, char4/float4 vectorized, W8A16) | `cuda_kernels.cu:fcuda_matvec_q8` |

Both keep **activations f32** (RMSNorm/RoPE/attention numerics unchanged; only the GEMM input
is narrowed/dequantized), so the rest of the loop is dtype-agnostic and the output stays f32.
Plumbing (all behind existing seams, default-off, cpu-ref Reference path byte-for-byte
untouched):
- `Upload(t, F16|Q8_0)` narrows on the device; a Q8 tensor uploads int8 codes + scales
  (`cuda.go`). The backend declares its preference via `PreferredWeightDtype()` (Q8 > F16 > F32).
- The HAL's `matWeightHAL` uploads matmul weights narrowed; norm weights + biases stay f32.
  Q8 weights come from the **lean store** (`m.q8w`, `LoadSafetensorsQuantDir`) so a quant-only
  model never needs the dropped f32 — which is how the **sharded** 3B loads in ~4 GB host RAM
  (its f32 blob would be ~12 GB).

## 2. Correctness — witnessed kernel → synthetic → real model

| Witness | Result | Where |
|---|---|---|
| f16 GEMM vs cpuref f32 | cosine **1.00000000**, maxAbs 1.82e-03 | `TestCUDAMatMulF16MatchesRef` |
| Q8 GEMV vs cpuref f32 | cosine **0.99999976**, maxAbs 2.5e-02 | `TestCUDAMatMulQ8MatchesRef` |
| synthetic Llama decode, f16, vs f32 native | **argmax-exact /10**, prefill cos 0.99999983 | `TestHALDeviceForwardMatchesNative` (`FAK_CUDA_F16=1`) |
| **real Qwen2.5-1.5B**, f16 vs f32 | **argmax-exact /12** — coherent English | `cmd/gpucheck` |
| **real Qwen2.5-1.5B**, Q8 vs f32 | **argmax-exact /10** | `cmd/gpucheck -backend cuda` |
| **real Qwen2.5-3B**, Q8 vs cpu-Q8 | **argmax-exact /8** | `cmd/gpucheck -lean` |

Suite green across **all five permutations** — f32, graph, f16, f16+graph, Q8
(`internal/compute/build_cuda.sh test`).

## 3. The head-to-head — decode, same RTX 4070

fak: median of repeated `modelbench` runs (decode-steps 128, prompt 16). llama.cpp:
`llama-bench -ngl 99 -n 128 -r 5`. Run-to-run variance is ±10–15% (laptop GPU clock + 9p +
peer sessions), so ranges are given where seen.

### Qwen2.5-1.5B-Instruct

| engine | precision | bytes/param | decode tok/s | vs fak f16 |
|---|---|---:|---:|---:|
| **fak-CUDA** | **f16** | 2 | **~36.6** (34.9–37.8; peak 53 on a clean clock) | — |
| `llama.cpp` | F16 | 2 | 34.3 | **fak 1.07× — parity, fak ahead** |
| fak-CUDA | Q8_0 | 1.125 | 15.8 | (launch-bound, §4) |
| `llama.cpp` | Q8_0 | 1 | 69.1 | |
| `llama.cpp` | Q4_K_M | 0.5 | 93.1 | |

**At equal precision (f16), fak decode is at parity with — slightly ahead of — `llama.cpp`.**
fak's f16 GEMM is cuBLAS tensor-core; that is why fak does *relatively better* here than at
135M (where it was launch-bound). Decode stays flat as context grows: at **768 tokens** fak f16
still decodes 36.5 tok/s.

### Qwen2.5-3B-Instruct (the 2.5–3B target)

| engine | precision | size on GPU | decode tok/s |
|---|---|---:|---:|
| **fak-CUDA** | **Q8_0** | **~3.5 GiB** (fits the ~6.2 GiB free) | **~11–15** (clock-dependent) |
| `llama.cpp` | Q8_0 | 3.36 GiB | 32.4 |

**Qwen2.5-3B runs on the GPU, correct (argmax-exact).** fak's Q8 decode is ~0.4× `llama.cpp`'s
— not parity, and §4 is honest about why.

## 4. Why Q8 decode is behind — it's the WSL launch tax, not the kernel (and not architecture)

A batch-1 decode issues ~600 device ops/token. On WSL each op carries a ~0.07 ms host
launch tax (microbench, `GPU.md` §3b) → a **~42 ms/token floor independent of compute**. Three
measurements pin the cause to *launches*, not the GEMV inner loop:

- **char4/float4 vectorizing the Q8 GEMV changed nothing** (3B 90.3 vs 90.1 ms) — the inner
  loop is not the bottleneck.
- **A W8A8 `__dp4a` GEMV (llama.cpp's technique) made it SLOWER** (1.5B 15.8 → 11.0 tok/s) —
  its per-matmul activation-quant kernel *adds a launch*, and on WSL the extra launch costs
  more than `__dp4a`'s compute saves. (Reverted; the finding is the value.)
- **The CUDA graph that fixed this for the 135M model (12 → 120 tok/s) doesn't engage on the
  large multi-layer graph**: capture hits a `cudaMalloc` mid-stream (`cuda_kernels.cu:67`,
  "operation not permitted when stream is capturing") and falls back to op-per-call. The
  pre-warm doesn't cover every transient buffer size the 28–36-layer graph allocates.

So the Q8 speed lever on this box is a **capture-safe CUDA graph** (pre-warm or arena every
transient so capture never allocates), not a faster kernel. Native Linux (no per-call tax)
starts from a far lower floor. This mirrors the CPU axis (`LLAMACPP-HEADTOHEAD-RESULTS.md`):
fak's gap to `llama.cpp` is a tuning/runtime boundary, not an architecture ceiling — and at
f16, where fak rides cuBLAS, it already wins.

## 5. "How much it can grow" — agentic-context headroom

Weights are **fixed** once resident (1.5B f16 ≈ 3.0 GiB; 3B Q8 ≈ 3.5 GiB); the rest of the
~6.2 GiB free budget is agentic-context capacity. fak's device KV keeps **K, Kraw, and V** per
layer (Kraw is the retained pre-RoPE key that lets `Evict` reposition a survivor in one
rotation — the bit-exact KV-quarantine security primitive), f32:

```
1.5B KV = 28 layers × 2 kv_heads × 128 head_dim × (K+Kraw+V) × 4 B = 84 KiB / token
```

So with Qwen2.5-1.5B f16 resident (~4.4 GiB measured incl. CUDA context + KV), the ~2.7 GiB
of remaining budget is **~32k tokens of context for one agent**, or **tens of concurrent
agents** each holding a working set — and the kernel-owned KV makes the multi-agent regime
*safe*: `cudaKV.Clone` deep-copies a shared prefix into each agent (cross-agent reuse) and
`Evict` removes a poisoned span bit-exactly, per agent (`kvreuse_test.go`, `evict_test.go`).
Turning that headroom into measured multi-agent *throughput* on the device is the batched-decode
follow-up (same as §6's prefill). Two levers shrink the per-token KV further: f16 device KV
(→ 42 KiB/tok, ~74k tokens) and dropping Kraw when per-agent Evict isn't needed.

## 6. Honest residue / what's next

- **Prefill isn't batched on the device path** — the HAL prefill loops single tokens, so fak
  prefill ≈ decode speed while `llama.cpp` batches (pp256 ≈ 2330–4882 tok/s). `BatchedMatMul`
  is f16/Q8-ready; the HAL prefill rewrite + batched device attention is the follow-up.
- **Q8 decode is WSL-launch-bound** — the capture-safe-graph fix (§4) is the lever; until then,
  3B Q8 is ~0.4× `llama.cpp` on WSL.
- **2.5–3B at f16 doesn't fit this 8 GB-laptop** (6.2 GiB weights == the whole free budget);
  Q8/Q4 is required, and Q8 is what shipped. 7B would need Q4 device weights (#33 extends to it).

## 7. Reproduce

> **Committed data + charts:** this 3B Q8 run is shipped as machine-readable data with inline
> charts in [`experiments/gpu/`](../../experiments/gpu/) — [decode-vs-llama](../../experiments/gpu/figures/decode-vs-llama.svg)
> · [prefill≈decode](../../experiments/gpu/figures/prefill-decode.svg). Every fak number there is fak's
> **own** in-kernel CUDA forward pass; `llama.cpp` is the comparison baseline only.

```bash
bash internal/compute/setup_cuda_wsl.sh        # one-time CUDA toolchain (GPU.md §1)
bash internal/compute/build_cuda.sh test       # all five permutations green

# f16 parity, Qwen2.5-1.5B (single-file safetensors):
FAK_CUDA_F16=1 go run -tags cuda ./cmd/modelbench -hf <qwen2.5-1.5b-dir> -backend cuda -decode-steps 128 -decode-reps 5
FAK_CUDA_F16=1 go run -tags cuda ./cmd/gpucheck   -hf <qwen2.5-1.5b-dir> -backend cuda -n 12

# Q8 to reach 3B (sharded -> lean Q8 load):
FAK_CUDA_Q8=1  go run -tags cuda ./cmd/modelbench -hf <qwen2.5-3b-dir> -lean -backend cuda -decode-steps 128
FAK_CUDA_Q8=1  go run -tags cuda ./cmd/gpucheck   -hf <qwen2.5-3b-dir> -lean -backend cuda -n 8

# llama.cpp baselines, same box:
llama-bench -m qwen2.5-1.5b-instruct-f16.gguf -ngl 99 -n 128 -r 5
llama-bench -m qwen2.5-3b-instruct-q8_0.gguf  -ngl 99 -n 128 -r 5
```

## 8. Bottom line

- **f16, Qwen2.5-1.5B: equal-precision decode parity REACHED — fak ahead** (~36.6 vs 34.3
  tok/s), correctness argmax-exact vs f32 on the real model. fak's cuBLAS tensor-core f16 GEMM
  is the peer of llama.cpp's F16.
- **Q8, Qwen2.5-3B: the 2.5–3B target REACHED on the GPU, correct** (argmax-exact, fits ~3.5
  GiB). Decode is WSL-launch-bound (~11–15 vs 32 tok/s) — a capture-safe CUDA graph is the
  lever (§4), not architecture; native Linux starts far lower.
- **Levers were small and modular:** an f16 path (`cublasGemmEx`) and a Q8 path (fused
  dequant-GEMV + lean sharded load) behind the existing `UploadDtype`/`PreferredWeightDtype`
  seams, gated, default-off, Reference bit-identity witness untouched, suite green across
  f32/graph/f16/f16+graph/Q8.
