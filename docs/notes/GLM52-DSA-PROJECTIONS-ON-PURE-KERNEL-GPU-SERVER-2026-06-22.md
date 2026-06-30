---
title: "GLM-5.2 DSA attention projections now run on the pure fak kernel (2026-06-22)"
description: "The next #86/#413 slice: GLM-5.2's DSA-attention dense projections (q_a/q_b, kv_a/kv_b, indexer wq_b/wk/weights_proj, o_proj) now route through the compute.Backend — so on the GPU server they execute on k_q8_gemm alongside the MoE/FFN/head, narrowing the host residual to the genuinely sparse glue."
---

# GLM-5.2: the DSA attention projections move onto the pure fak kernel (2026-06-22)

> **Goal (verbatim):** *glm 5.2 pure fak kernel running on GPU server machine.*
>
> This note records one concrete slice toward that goal and the witnesses that
> close it. It is grounded in `go test` exit codes (WSL go1.26) and a behavioral
> recording-backend probe — not self-report. The on-device datacenter GPU verdict is the
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

## §3 — On the GPU server: witnessed fresh today, the projections run on `k_q8_gemm`

The same backend path, with a lean (Q8-resident) GLM-DSA model on the **cuda backend**,
runs all eight projections on `k_q8_gemm` — the GPU pure kernel — alongside the MoE/FFN
experts, router, and vocab head. This was **re-run on the lab 8-GPU datacenter server on
2026-06-22 at the slice's HEAD (`498a4ab`)**, via the live private control bridge
(`private GPU witness runner`: clone `origin/main` → `nvcc -arch=sm_80` → isolated
`-tags cuda` test). The node's own `go test` output (not self-report):

```
=== HEAD 498a4ab ===
=== build libfakcuda.a (sm_80) === [cuda] OK build
=== go test -tags cuda -run TestCUDAGLMMoeDsaBackendForward ./internal/model/ -v ===
    glm_dsa_cuda_test.go:55: GLM-MoE-DSA forward with MoE/FFN+head + DSA attention
      projections on cuda backend (k_q8_gemm): cosine=1.000000 argmax cpu=40 cuda=40
      tier=sm_80 class=approx
--- PASS: TestCUDAGLMMoeDsaBackendForward (0.26s)
=== GLM GPU WITNESS DONE rc=0 ===
```

So GLM-5.2's forward — the MoE/FFN experts + router, the vocab head, **and now the DSA
attention's dense projections** — executes on the pure fak CUDA kernel on real datacenter GPU
hardware, **cosine = 1.000000, argmax-exact** vs the CPU Q8 forward. (The prior MoE/FFN/head
slice was committed `cf9d9a1` / `e3a92b7`, 2026-06-21; see
[`GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md`](GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md).)

**Reproduce on an sm_80 CUDA node** (clones `origin/main`, builds for sm_80, runs the
isolated witness):

```bash
bash private GPU witness runner    # -> <private-scratch>/run.log + <private-scratch>/DONE.<rc>
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
fit pure on an 8-GPU datacenter server (INT4 ≈ 376 GB > 320 GB) and needs the multi-GPU
NCCL/offload reshape — the SGLang-serves + fak-fronts path, not the native engine.

> **Superseded by progress (#917; see the [staged plan](native-753b-track-staged-plan.md)).**
> The "SGLang-serves, not the native engine" posture above was the 2026-06-22 snapshot; once
> `--cpu-offload-experts` shipped, fak's own engine loaded the full 466 GB model natively
> ([2026-06-25 native-serve note](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md)). The
> DSA-projection slice this note records stands; the flagship-serving direction does not.

## What is proven vs not (labeled)

- **Proven on-box today:** GLM-5.2's DSA attention dense projections route through the
  compute backend (behavioral witness, all 8 shapes); the host path is byte-for-byte
  unchanged (full `internal/model` + GLM coherence suites green); `go build ./...` +
  `go vet` green.
- **Proven on real datacenter GPU hardware, fresh on 2026-06-22 at HEAD `498a4ab`** (and
  reproducible via `private GPU witness runner`): GLM-5.2's MoE/FFN/router/head **plus
  the DSA attention projections** on the pure fak CUDA kernel (`k_q8_gemm`), cosine =
  1.000000, argmax-exact (`TestCUDAGLMMoeDsaBackendForward`, sm_80). The prior MoE/FFN/head
  slice was committed `cf9d9a1`/`e3a92b7`.
- **Not proven / out of scope (labeled):** GLM-5.2's DSA **sparse-attention glue** on the
  GPU (fused sparse kernel + device DSA-KV, #86/#413 next slice); HF numeric DSA parity
  (#474/#413, oracle-gated); real 753B serving (VRAM-gated + NCCL/offload reshape).
