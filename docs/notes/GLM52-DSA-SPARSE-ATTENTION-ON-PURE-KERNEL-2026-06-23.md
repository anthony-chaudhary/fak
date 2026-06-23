---
title: "GLM-5.2 DSA sparse attention now runs on the pure fak kernel (2026-06-23)"
description: "The #86/#413 next slice: GLM-5.2's DSA sparse-attention compute — the per-head softmax(scale·q·k)·ΣwV over the top-k selected keys — now routes through the compute.Backend (k_dsa_sparse_attend on cuda), so the attention math itself runs on the kernel; only the genuinely sparse index-score + top-k selection stays host-resident."
---

# GLM-5.2: the DSA sparse attention moves onto the pure fak kernel (2026-06-23)

> **Goal (verbatim):** *glm 5.2 working end to end on our kernel proven on real hardware.*
>
> This note records the slice that takes GLM-5.2's forward from "dense compute on the
> device, sparse-attention glue host-resident" to "the attention math itself on the
> kernel." It is grounded in `go test` exit codes (WSL go1.26) and a behavioral routing
> probe — not self-report. §2 states exactly what is proven on the A100 today and what is
> queued, with the hardware-gated piece labeled, not hidden.

## What changed

Before this slice (committed `498a4ab`), GLM-5.2's **dense** compute ran on the pure
fak GPU kernel — the MoE/FFN experts + router, the vocab head, and all eight DSA
**attention projections** (q_a/q_b, kv_a/kv_b, indexer wq_b/wk/weights_proj, o_proj) on
`k_q8_gemm`. But the **sparse-attention compute itself** — the per-head
`softmax(scale·q·k)·ΣwV` over the learned-indexer's top-k selected keys
(`glmDsaAttendCached`'s inner loop) — stayed **host-resident** even on the GPU path.

This slice moves that attention math onto the device via a new OPTIONAL
`compute.DSASparseBackend` capability (type-asserted like `CollectiveBackend`, so a
backend without it simply falls back to the host loop):

| Backend | DSASparseAttend implementation |
|---|---|
| **cpu-ref** (Reference) | byte-for-byte `glmDsaAttendCached`'s loop (`dot·scale`, `softmaxInPlace`, in-order `ΣwV`) — forward stays argmax-exact |
| **cuda** (Approx) | `k_dsa_sparse_attend` — an online-softmax kernel, one block per query head, separate MLA key/value widths (`kd = qkNope+qkRope ≠ vd`) |

`glmDsaAttendCached` gathers the selected K/V rows contiguous and routes the attention
through the seam; the host loop is the unchanged fallback. Shipped at `ee88d73`.

## The selection stays host-side BY DESIGN

The f64 index-score dots + top-k (`glmDsaIndexStep`) pick **which** keys attend. Keeping
that on the host means the selected key set is **identical CPU↔device**, so the device's
only divergence is the f32 reduction order over the **same** keys — the Approx class the
dense GEMM and flash-attention lanes already live in — never a flipped top-k (a single
flipped selection would diverge the output far past a reduction-order cosine). So after
this slice the only GLM-5.2 work still host-resident is the **genuinely sparse
selection** (index scores + top-k) and the DSA KV cache — the O(topK) control flow, not
the O(H·…) attention FLOPs.

## §1 — The host path is unchanged; the routing is argmax-exact (on-box, WSL go1.26)

`TestGLMMoeDsaBackendRoutesSparseAttention` (new) wraps cpu-ref in a `recordingBackend`
that records every `DSASparseAttend` call. A GLM-DSA `Prefill` proves the sparse attend
**reaches the backend over real selected keys**, and the forward stays **argmax-exact**
vs the all-host reference:

```
GLM-DSA sparse attention on backend "cpu-ref": 8 DSASparseAttend calls over 14 selected
keys; argmax-exact (40), max|Δ|=0.000e+00 vs all-host
```

The full `internal/model` GLM/DSA/MoE suite + `internal/compute` stay green, `-race`
clean, `gofmt` clean, and the cgo side **type-checks under `-tags cuda`**
(`go vet -tags cuda ./internal/compute/ ./internal/model/`).

## §2 — On the A100 DGX: what is proven, and the queued sparse-kernel capture

The same backend path runs the sparse attention on `k_dsa_sparse_attend` on the A100. The
on-device witness is `TestCUDAGLMMoeDsaBackendForward` — now also asserting the device
sparse path is wired (a `compute.DSASparseBackend` type-assert that fails the test rather
than silently falling back to host) — run via `tools/dgx_glm_gpu_witness.sh`
(clone `origin/main` → `nvcc -arch=sm_80` → isolated `-tags cuda` test).

**Proven on this same A100 (committed, re-confirmed today):** the **dense** GLM-5.2
forward — MoE/FFN/router/head **and** the DSA attention projections on `k_q8_gemm` — is
**argmax-exact, cosine = 1.000000** vs the CPU Q8 forward (`498a4ab`; the node's own
`go test` run.log was read back through the bridge this session and shows
`cosine=1.000000 argmax cpu=40 cuda=40 tier=sm_80`). The hardware, the sm_80 toolchain,
and the backend forward path are established on-device.

**Queued (not yet captured) — the new `k_dsa_sparse_attend` on-device cosine:** at the
time of writing (2026-06-23) the lab control bridge had **no live session** across an
extended polling window (zero `running` sessions over ~30 min; the ephemeral control
shells had stopped spawning), so a fresh on-device cosine for the sparse kernel was not
captured this session. This is an infrastructure gap in the bridge, not a code gap: the
kernel is authored, the Go/cgo side type-checks under `-tags cuda`, and the witness is
wired. The kernel reuses the **proven** `k_flash_attention` online-softmax form (same
Approx reduction-order class, no narrowed operand) and is held to
`cudaDsaSparseAttnCosineMin = 0.999`; because the key selection is host-computed (the
device attends the identical key set), the on-device cosine is expected at the same
~1.0 the dense path shows — but that is an **expectation, labeled as such, not a measured
pass**. The capture runs the moment a bridge session is live:

```bash
bash tools/dgx_glm_gpu_witness.sh   # at HEAD ee88d73; -> /tmp/fakglm/run.log + DONE.<rc>
```

## §3 — What stays host-resident (the labeled residual)

After this slice the only GLM-5.2 DSA work on the host is the **sparse selection** — the
learned-index score dots (`Σ_h weights[h]·relu(scale·dot(q_h, key))`), the top-k, and the
DSA KV cache. A fully device-resident selection (an on-device index-score + top-k kernel
+ device DSA-KV) is the remaining slice; it is labeled, not claimed, because computing
top-k in device f32 risks a flipped selection vs the host f64 (the honesty boundary this
slice deliberately preserves).

The flagship-scale residual is unchanged and out of scope: the real 753B does not fit
pure on an 8× A100-40GB DGX (INT4 ≈ 376 GB > 320 GB) — the SGLang-serves + fak-fronts
path, not the native engine.

## What is proven vs not (labeled)

- **Proven on-box today (`ee88d73`, WSL go1.26):** GLM-5.2's DSA sparse-attention compute
  routes through `compute.DSASparseBackend`; cpu-ref runs it argmax-exact
  (`TestGLMMoeDsaBackendRoutesSparseAttention`); the host path is unchanged; full
  `internal/model` GLM/DSA/MoE + `internal/compute` suites green; `-race`/`gofmt` clean;
  cgo type-checks under `-tags cuda`.
- **Proven on real A100 hardware (committed `498a4ab`, re-confirmed today):** GLM-5.2's
  **dense** forward (MoE/FFN/router/head + attention projections) on `k_q8_gemm`,
  cosine = 1.000000, argmax-exact.
- **Queued / not yet captured (labeled):** the **new `k_dsa_sparse_attend` on-device
  cosine** — blocked at witness time by the bridge having no live session, not by code;
  reproduce via `tools/dgx_glm_gpu_witness.sh` at `ee88d73`.
- **Out of scope (labeled):** device-resident DSA *selection* (index-score + top-k kernel
  + device DSA-KV); real 753B serving (VRAM-gated).
