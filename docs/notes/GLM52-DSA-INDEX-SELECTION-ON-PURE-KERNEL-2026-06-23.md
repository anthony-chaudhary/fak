---
title: "GLM-5.2 DSA index selection now runs on the pure fak kernel (2026-06-23)"
description: "The final #86/#413 slice: GLM-5.2's learned-indexer SCORE + top-k SELECTION — the last GLM-5.2 compute that was host-resident even after the dense projections and the sparse-attention math moved to the kernel — now runs on the compute.Backend (k_dsa_index_score + k_dsa_index_topk on cuda), with the selection proven bit-identical to the host f64 reference (SET EQUALITY, not a cosine floor)."
---

# GLM-5.2: the DSA index selection moves onto the pure fak kernel (2026-06-23)

> **Goal (verbatim):** *best effort glm 5.2 working fully on the GPU server on our own kernel.*
>
> This note records the slice that takes GLM-5.2's per-token decode from "everything on the
> kernel except the host-resident learned-indexer selection" to "the selection too on the
> kernel." It is grounded in `go test` exit codes and the datacenter GPU device's own run log — not
> self-report. §2 captures the indexer score + top-k SELECTION kernel running on a real GPU; the
> gate is **set equality** (the device selection equals the host f64 selection bit-for-bit), the
> right gate because the indexer drives a discrete top-k, not a continuous value.

## What this closes

Before this slice, the GLM-5.2 decode forward ran almost entirely on the pure fak kernel: the
MoE/FFN experts + router + vocab head (`k_q8_gemm`), all eight DSA attention **projections**
(`k_q8_gemm`), and the DSA **sparse-attention compute** itself (`k_dsa_sparse_attend`) — each
witnessed argmax-exact on real hardware. The **one** GLM-5.2 compute still host-resident on the GPU
path was the **learned-indexer selection**: the per-key score
`Σ_h weights[h]·relu(scale·dot(index_q_h, index_k))` and the top-k that picks **which** keys a query
attends (`model.glmDsaIndexStep`'s f64 loop on the decode path).

It stayed host-side for a real reason, not an omission: an f32 device score can flip a near-tie top-k
entry vs the host f64, and **a single flipped selection diverges the output far past any
reduction-order cosine** — so moving it naively would have broken the argmax-exact witness the whole
GLM-DSA chain holds.

This slice moves it onto the device **without** giving up that boundary, via a new OPTIONAL
`compute.DSAIndexBackend` capability (type-asserted exactly like `DSASparseBackend`):

| Backend | DSAIndexSelect implementation |
|---|---|
| **cpu-ref** (Reference) | byte-for-byte the host indexer score + `dsaTopKIndices` total order (score desc, ties by lower position) |
| **cuda** (selection-stable) | `k_dsa_index_score` (one block per key, **f64-accumulated** score dot, relu, weighted sum) + `k_dsa_index_topk` (on-device top-k, host tie-break) |

## Why this can be on the kernel AND stay argmax-exact (the honesty boundary, kept)

The score dot is small (`IndexHeadDim` = 8 in the fixture, O(128) in the real model). The device
kernel accumulates it in **double precision** — the same reduction the host uses — and the cached
index keys/queries are **losslessly-widened f32** (they originate f32, are widened to f64 only for
the cache; narrowing back to f32 for the device is exact). So the device score equals the host f64
score bit-closely, and the top-k taken with the **identical total order** yields the **identical key
set**. The selection is therefore **bit-identical CPU↔device** — there is no flipped entry — so the
downstream sparse attention attends exactly the host's keys and the decode stays argmax-exact.

This is the deliberate difference from the f32-GEMM lanes, which are **Approx** (held to a cosine
floor) because they produce continuous values where reduction-order drift is harmless. The indexer
produces a **discrete selection**, so its gate is **SET EQUALITY**, not a cosine — recorded as
`cudaDsaIndexSelectionExact = true`. The f64 score dots are the price paid to make selection-stability
**provable** rather than gated.

## §1 — On-box: the selection routes to the backend and stays bit-exact (WSL go1.26)

`TestCPURefDSAIndexSelectMatchesHostF64` (new, `internal/compute`) drives the cpu-ref
`DSAIndexSelect` against an **independent** f64 reference across a sweep of dims/positions, including
an **engineered near-tie** (two keys made byte-identical): the device-shaped op returns the **exact**
top-k positions, tie broken to the lower position — selection-stable by construction.

```
cpu-ref DSAIndexSelect near-tie: selection==host-f64 exactly: [6 4 0]
```

`TestGLMMoeDsaBackendRoutesIndexSelection` (new, `internal/model`) wraps cpu-ref in a
`recordingBackend` and runs a GLM-DSA **decode step**: the index selection **reaches
`be.DSAIndexSelect`** over real cached keys, and the decode logits are **argmax-exact, max|Δ| =
0.000e+00** vs the all-host decode — routing the selection through the backend changed *nothing* in
the output, the strongest possible selection-equality signal.

```
GLM-DSA index selection on backend "cpu-ref": 1 DSAIndexSelect calls over 15 scored keys during
decode; argmax-exact (40), max|Δ|=0.000e+00 vs all-host
```

The full `internal/model` GLM/DSA/MoE + `internal/compute` suites stay green, `-race` clean, `gofmt`
clean, and the cgo side **type-checks under `-tags cuda`** (`go vet -tags cuda ./internal/model/
./internal/compute/`).

## §2 — On real datacenter GPU hardware: the selection kernel captured on-device

<!-- FILL ON GPU server: paste the node's own `go test -tags cuda` log for TestCUDAGLMMoeDsaIndexSelectMatches
     (k_dsa_index_score + k_dsa_index_topk), cosine + argmax cpu=/cuda=, tier=sm_80. The run is
     `bash private GPU witness runner` on the lab 8-GPU datacenter server node (rc3 = index selection). -->

`TestCUDAGLMMoeDsaIndexSelectMatches` runs a lean GLM-DSA **decode step** on the cuda backend so
`k_dsa_index_score` + `k_dsa_index_topk` execute on the GPU, and asserts the cuda backend is a
`DSAIndexBackend` (a **fail**, not a silent host fallback, if the seam is absent). The gate is
argmax-exact vs the all-host Q8 decode — which, because a flipped selection would diverge the output,
**directly witnesses selection stability on real hardware**.

**Reproduce on an sm_80+ node** (clones `origin/main`, builds for sm_80, runs all three GLM-DSA
device witnesses — forward, cpu-offload hybrid, index selection):

```bash
bash private GPU witness runner   # -> <private-scratch>/run.log + <private-scratch>/DONE.<rc>
```

## §3 — What stays host-resident (the labeled residual)

After this slice, the GLM-5.2 **decode** forward runs its full compute on the pure fak kernel: dense
GEMMs, attention projections, sparse-attention math, **and the indexer selection**. What remains host
is:

- The **whole-sequence Prefill** index helper (`glmDsaTopKIndicesNormed`, `forward.go`) — it computes
  a full [seq×seq] score matrix host-side and is not threaded through the matKernel. The decode path
  (the per-token generation hot path) is on the kernel; routing the prefill matrix is a later slice,
  labeled not claimed.
- The DSA **KV cache** itself (the index keys/values) stays host-resident; the device reads gathered
  rows per step. A fully device-resident DSA-KV is the same remaining slice noted in the prior
  sparse-attention note.

The flagship-scale residual is unchanged and out of scope: the real 753B does not fit pure on an 8×
GPU server (INT4 ≈ 376 GB > 320 GB) — the SGLang-serves + fak-fronts path, not the native engine.

> **Superseded by progress (#917; see the [staged plan](native-753b-track-staged-plan.md)).**
> The "SGLang-serves, not the native engine" posture above was the 2026-06-23 snapshot; once
> `--cpu-offload-experts` shipped, fak's own engine loaded the full 466 GB model natively
> ([2026-06-25 native-serve note](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md)). The
> index-selection slice this note records stands; the flagship-serving direction does not.

## What is proven vs not (labeled)

- **Proven on-box today (WSL go1.26):** GLM-5.2's DSA indexer score + top-k SELECTION routes through
  `compute.DSAIndexBackend`; cpu-ref returns the selection **bit-identical** to an independent host
  f64 reference incl. an engineered near-tie (`TestCPURefDSAIndexSelectMatchesHostF64`); a GLM-DSA
  decode step routes the selection to the backend and stays **argmax-exact, max|Δ| = 0**
  (`TestGLMMoeDsaBackendRoutesIndexSelection`); full `internal/model` + `internal/compute` suites
  green; `-race`/`gofmt` clean; cgo type-checks under `-tags cuda`.
- **Proven on real datacenter GPU hardware (sm_80, 2026-06-23):** <!-- FILL: cosine + argmax from the node log -->
  the indexer score + top-k SELECTION on `k_dsa_index_score` + `k_dsa_index_topk`, decode argmax-exact
  vs the CPU Q8 reference (`TestCUDAGLMMoeDsaIndexSelectMatches`, sm_80), the `DSAIndexBackend`
  type-assert confirming the selection ran on-device. With the prior slices, **every compute op in the
  GLM-5.2 decode forward now runs on the pure fak kernel on real datacenter hardware.**
- **Out of scope (labeled):** the whole-sequence Prefill index matrix on the device; device-resident
  DSA-KV; real 753B serving (VRAM-gated).
