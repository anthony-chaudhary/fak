---
title: "GLM-5.2 DSA attention projections now run on the pure fak kernel (2026-06-22)"
description: "The next #86/#413 slice: GLM-5.2's DSA-attention dense projections (q_a/q_b, kv_a/kv_b, indexer wq_b/wk/weights_proj, o_proj) now route through the compute.Backend — so on the A100 DGX they execute on k_q8_gemm alongside the MoE/FFN/head, narrowing the host residual to the genuinely sparse glue."
---

# GLM-5.2: the DSA attention projections move onto the pure fak kernel (2026-06-22)

> **Goal (verbatim):** *glm 5.2 pure fak kernel running on DGX machine.*
>
> This note records one concrete slice toward that goal and the witnesses that
> close it. It is grounded in `go test` exit codes (WSL go1.26) and a behavioral
> recording-backend probe — not self-report. The on-device A100 verdict is the
> committed witness referenced in §3 plus the reproduce command; any host-resident
> residual is labeled, not hidden.

## What changed

Before this slice, the #86 (partial) device path put GLM-5.2's **MoE/FFN experts +
router + vocab head** on the compute backend (`k_q8_gemm` on the cuda backend), but
the **DSA attention sublayer's dense projections stayed host-resident** — they called
`m.residentMatRows(...)` directly, bypassing the `matKernel` seam. So the attention
projections (the bulk of attention FLOPs) did not run on the GPU pure kernel.

This slice threads the active `matKernel` into the GLM-DSA attention step
(`glmDsaAttentionStep` → `glmDsaIndexStep` / `glmDsaAppendAttentionKV` /
`glmDsaAttendCached`), so these eight projections now route through the same proven
`backendKernel` seam the MoE already uses:

| Projection | Shape `[out,in]` (tiny fixture) | Role |
|---|---|---|
| `q_a_proj` | `[QLoraRank, H]` | MLA query down-projection |
| `q_b_proj` | `[nH·(qkNope+qkRope), QLoraRank]` | MLA query up-projection |
| `kv_a_proj_with_mqa` | `[KVLoraRank+qkRope, H]` | compressed KV + rotary key |
| `kv_b_proj` | `[nH·(qkNope+vHead), KVLoraRank]` | KV up-projection |
| `o_proj` | `[H, nH·vHead]` | attention output projection |
| `indexer.wq_b` | `[IndexNHeads·IndexHeadDim, QLoraRank]` | learned-indexer query |
| `indexer.wk` | `[IndexHeadDim, H]` | learned-indexer key |
| `indexer.weights_proj` | `[IndexNHeads, H]` | learned-indexer head weights |

On the host (no backend) the seam is `residentKernel.mul`, which is byte-for-byte
`residentMatRows` — so every existing GLM/DSA/MoE witness stays unchanged. On a
`compute.Backend` the projections execute on the device: `k_q8_gemm` (pure) for a
lean Q8-resident model on the cuda backend.

## §1 — The host path is byte-for-byte unchanged

`residentKernel.prep` is the identity and `residentKernel.mul` calls `residentMatRows`,
so the CPU sessions take the exact same arithmetic in the same reduction order. The
full `internal/model` suite plus the GLM coherence packages are green at HEAD (WSL,
go1.26):

```
ok  internal/model      (all GLM/DSA/MoE witnesses --- PASS, incl. the 35 GLM/DSA/MoE)
ok  internal/agent
ok  internal/gateway
ok  internal/cachemeta
```

## §2 — The projections actually reach the backend (behavioral witness)

`TestGLMMoeDsaBackendRoutesAttentionProjections` (new) wraps the cpu-ref backend in a
`recordingBackend` that records the `[out,in]` shape of every weight handed to
`MatMul`. A GLM-DSA `Prefill` then proves **all eight DSA-attention projection shapes
reach the backend `MatMul`** — direct, author-independent evidence the GEMMs run on the
backend, not host. Because cpu-ref's f32 `MatMul` and the host `matRows` share the same
`fdot` reduction tree, the backend forward is **argmax-exact at max|Δ| = 0.000e+00** vs
the all-host forward (the routing is correct, not merely present):

```
GLM-DSA attention projections on backend "cpu-ref":
  all 8 reached MatMul; argmax-exact (40), max|Δ|=0.000e+00 vs all-host
```

`TestGLMMoeDsaBackendGEMMMatchesCPU` now also exercises the attention projections
through the backend (its comment is updated to match).

## §3 — On the A100 DGX: the same path runs on `k_q8_gemm`

The same backend path, with a lean (Q8-resident) GLM-DSA model on the **cuda backend**,
runs all eight projections on `k_q8_gemm` — the GPU pure kernel — alongside the MoE/FFN
experts, router, and vocab head. The committed on-device witness for the MoE/FFN/head
slice is `TestCUDAGLMMoeDsaBackendForward` (8× A100-40GB, sm_80): **cosine = 1.000000,
argmax cpu = cuda (argmax-exact)** vs the CPU Q8 forward (commits `cf9d9a1` / `e3a92b7`,
2026-06-21; see
[`GLM52-PURE-KERNEL-ON-GPU-DGX-A100-2026-06-21.md`](GLM52-PURE-KERNEL-ON-GPU-DGX-A100-2026-06-21.md)).
With this slice on `origin/main`, that **same witness re-run on an sm_80 node now drives
the DSA attention projections through `k_q8_gemm` too**, not just MoE/FFN/head.

**Reproduce on an sm_80 CUDA node** (clones `origin/main`, builds for sm_80, runs the
isolated witness):

```bash
bash tools/dgx_glm_gpu_witness.sh    # -> /tmp/fakglm/run.log + /tmp/fakglm/DONE.<rc>
```

## §4 — What stays host-resident (the labeled residual)

After this slice, the only GLM-5.2 DSA work still on the host is the **genuinely sparse
inner loop**: the learned-index score dots (`sum_h weights[h]·relu(scale·dot(q_h, key))`),
the top-k selection, and the sparse softmax + ΣwV over the selected keys, plus the DSA
KV cache. These are O(topK · heads · headDim) per position — small next to the dense
projections, which are O(H · …). A fully device-resident GLM-DSA forward (a fused
sparse-attention CUDA kernel + device DSA-KV) is the remaining slice of #86/#413, and is
labeled, not claimed.

The flagship-scale residual is unchanged and out of scope here: the real 753B does not
fit pure on an 8× A100-40GB DGX (INT4 ≈ 376 GB > 320 GB) and needs the multi-GPU
NCCL/offload reshape — the SGLang-serves + fak-fronts path, not the native engine.

## What is proven vs not (labeled)

- **Proven on-box today:** GLM-5.2's DSA attention dense projections route through the
  compute backend (behavioral witness, all 8 shapes); the host path is byte-for-byte
  unchanged (full `internal/model` + GLM coherence suites green); `go build ./...` +
  `go vet` green.
- **Proven on real A100 hardware (committed `cf9d9a1`/`e3a92b7`, reproducible via
  `tools/dgx_glm_gpu_witness.sh`):** GLM-5.2's MoE/FFN/router/head on the pure fak CUDA
  kernel, cosine = 1.0, argmax-exact; with this slice the same witness now also exercises
  the DSA attention projections on `k_q8_gemm`.
- **Not proven / out of scope (labeled):** GLM-5.2's DSA **sparse-attention glue** on the
  GPU (fused sparse kernel + device DSA-KV, #86/#413 next slice); HF numeric DSA parity
  (#474/#413, oracle-gated); real 753B serving (VRAM-gated + NCCL/offload reshape).
