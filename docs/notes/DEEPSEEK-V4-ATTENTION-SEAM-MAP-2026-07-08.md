---
title: "DeepSeek V4 CSA/HCA sparse-attention — seam map to fak, and the"
description: "A design-only seam map matching DeepSeek V4's CSA/HCA sparse attention onto fak's existing DSA/MLA code seams, marking direct fits and gaps."
---

# DeepSeek V4 CSA/HCA sparse-attention — seam map to fak, and the prior-art baseline

**2026-07-08.** Issue **#3016**, parent epic **#3006** (native DeepSeek-V4 kernel track).
This is a **design note only** — a seam map plus prior-art comparison. No kernel code lands
here, and no performance is claimed. Current-state claims are witnessed against the exact
`path:line` cited (read 2026-07-08 on `main`). It is the artifact every downstream V4
kernel ticket (#3017 KV, #3018 MoE, #3019 quant) cites before touching code.

## Thesis — why this must be a map before an edit

DeepSeek V4 is **not** a dense-attention Llama port. Its load-bearing attention is a
**hybrid of Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)** with
a lightning indexer driving sparse top-k selection, a sliding-window side branch, and a
grouped output projection. fak already carries a *closely related* sparse-attention path —
the GLM-5.2 **MLA + DSA lightning-indexer** kernels — but it is a *single-tier* DSA/MLA
path, not V4's *two-tier* CSA(rate 4)/HCA(rate 128) split. So the honest move is to map each
V4 attention piece onto the real fak seam that is nearest, and to *fence* the pieces where
fak has no seam yet — rather than start editing `dsa_index.go` blind.

## The V4 architecture facts that drive the map (from the issue grounding)

Source: DeepSeek V4 technical report, https://arxiv.org/html/2606.19348v1 (numbers as cited
in #3016's Grounding section).

| V4 Pro fact | Value | Consequence for the seam map |
|---|---|---|
| Layers | 61 | per-layer indexer/KV state × 61 |
| Hidden size | 7168 | query/index projection dims |
| CSA compression rate | 4 | "compressed" KV plane (light compression) |
| HCA compression rate | 128 | "heavily compressed" KV plane (aggressive) |
| Attention top-k | 1024 | sparse selection width per query |
| Query heads | 128 | grouped output projection |
| Head dim | 512 | MLA-scale head (matches fak MLA head dims) |
| Query compression dim | 1536 | q-LoRA down-projection |
| SWA window | 128 | sliding-window side branch |
| Selector optimization | 2× selector speedup, 99.7% KV-entry recall | the top-k selector is the hot inner loop |

CSA applies **DSA sparse top-k selection** and uses a **lightning indexer**; HCA is dense
attention over a heavily compressed KV. That "DSA + lightning indexer" phrasing is exactly
the mechanism fak already models for GLM-5.2 (below), which is why the CSA half of V4 has a
real fak seam today and the HCA half is a proposed new one.

## Seam map — V4 attention requirement → fak seam (`path:line`) or proposed

| V4 requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| **Lightning indexer score** (per-query/key relu-scaled dot, weighted over index heads) | `internal/model/dsa_index.go:20` `dsaIndexScores` | **Direct fit.** Same formula shape (`sum_h w[q,h]·relu(scale·dot(idx_q,idx_k))`). Learned projections happen upstream in `glm_dsa.go:135` `glmDsaTopKIndices`. |
| **CSA sparse top-k selection** (top-1024 keys/query under causal mask) | `internal/model/dsa_index.go:66` `dsaTopKIndices` | **Direct fit.** Already selects top-k key *positions* (not dense offsets) after the DSA causal mask; `topK` is a parameter, so 1024 is a config value, not a rewrite. |
| **CSA sparse attention output** over selected keys | `internal/model/dsa_index.go:115` `dsaSparseAttention`; layer path `internal/model/glm_dsa.go:18` `glmDsaAttentionOutputFromTopK` | **Direct fit** for the compute; V4's *two* compression tiers need the layout split (#3017). |
| **Compressed latent KV production** (MLA-style down/up projection) | `internal/model/kvlayout.go:98` `mlaKVLayout`; `:151` `mlaProject`; `:192` `attendOne` | **Fit for CSA rate-4.** The `kvLayout` interface (`kvlayout.go:28`) is the extension point; HCA rate-128 is a **second** `kvLayout` impl (proposed). |
| **HCA dense attention over heavily-compressed KV** | *No fak seam yet* — proposed new `kvLayout` (rate-128) + a dense-over-compressed attend path sibling to `attendOne` | **Gap.** fak has one MLA compression tier; V4 has two. Fence, do not fake. |
| **SWA / state-cache branch** (window 128) | `internal/model/swa.go:25` `MaxWindow`, `:48` `TrimToWindow` | **Partial fit.** SWA trim-to-window exists and is bit-exact vs full-window; wiring it as a *concurrent side branch* alongside CSA/HCA (not a whole-cache trim) is new — see #3017. |
| **Grouped output projection** (128 q-heads) | `internal/model/glm_dsa.go` MLA output path (`glmDsaAttentionOutputFromTopKNormed:26`) | **Fit** via existing MLA grouped projection; head/group counts are config. |
| **Indexer-sharing across layers** (reuse one selection for several layers) | `internal/model/dsa_index.go:208` `dsaIndexShare`; digest `:232` `dsaIndexDigest` | **Direct fit** — fak already shares a full-layer selection across layers keyed by digest. |
| **Device-side sparse attend / index-select hooks** (offload split) | `internal/model/moe_offload.go:55` `sparseAttend`, `:72` `indexSelect` | **Fit** — the split kernel keeps indexer *projections* on-device while the expert bulk offloads, i.e. the selection hook is already device-routable. |
| **Attention backend contract** (K/V gather + attend) | `internal/compute/compute.go:341` `Attention(q, kv, layer, causal, grp, scale)` | **Fit** — the `compute.Backend` attention seam is where any native V4 kernel would bind; it is dtype/layout-agnostic at the interface. |
| **Prefill vs decode divergence** for sparse selection | `internal/model/glm_dsa.go:135` (indexer over `seq` tokens) + `swa.go` trim | **Partial.** Batched prefill indexer exists; decode-time incremental top-k at 1M context is not yet modeled at V4 scale. |

**Honest note on CSA vs HCA:** fak's `dsa_index.go` + `glm_dsa.go` implement **one** sparse
indexer/MLA tier (GLM-5.2's `glm_moe_dsa`). V4 needs **two** compression tiers wired to the
**same** indexer selection. The CSA half maps onto today's seams almost directly; the HCA
half is a *proposed new `kvLayout` implementation*, not an existing file.

## Prior-art baseline (what native work would be measured against)

Per the issue grounding (SGLang V4 roadmap, https://github.com/sgl-project/sglang/issues/23602)
and the standard sparse-attention stack:

| Prior art | What it provides | Relation to fak |
|---|---|---|
| **SGLang DeepSeek V4** (FP4 indexer, FlashMLA sparse prefill, ragged indexer, breakable CUDA graph, HiSparse pool) | The tuned reference serving path | The **parity oracle** — fak fixtures must match its shape/layout before any native claim. |
| **FlashMLA** | Fused MLA prefill/decode kernel | The kernel fak's `mlaKVLayout` + `attendOne` would either wrap or reimplement. |
| **FlashInfer / CUTLASS FP4** | FP4 GEMM + indexer primitives | Feeds #3019 (the FP4 indexer QK path). |
| **TileLang / TileKernels** | Kernel-authoring DSL | Candidate if fak chooses *native* over *wrap*. |
| **vLLM recipes** | Reference config + numerics | Secondary numeric oracle. |

`fak sota` / `internal/sotamatrix` is the in-repo home for keeping this matrix live; this
note records the snapshot, it does not re-run the tool.

## The implement / wrap / observe decision (recommended)

Three options, in order of increasing commitment:

1. **Observer-only hooks first (recommended start).** The indexer selection is already
   digestible (`dsaIndexDigest:232`); expose the V4 selection trace as an observed artifact
   and diff it against SGLang's, with **no** device kernel. This de-risks layout parity
   before any GEMM.
2. **Wrap a tuned engine** (FlashMLA / SGLang) behind `compute.Backend.Attention:341` —
   fak owns routing + the KV plane, the engine owns the kernel.
3. **Native kernel** — only after a small fixture proves CSA/HCA shape + layout parity
   against an external reference. This note **refuses** native implementation until that
   fixture exists (matches #3016's acceptance gate).

## Honest fences (what is NOT decided or built)

- **No HCA seam exists** — the rate-128 tier is a proposed new `kvLayout`, not a file.
- **No two-tier CSA/HCA cache object** — that is #3017's job; this note only maps compute.
- **No V4 weights, no numerics** — every fit above is a *shape/layout* fit, not a verified
  bit-exact match to V4.
- **No 1M-context decode model** — incremental top-k at V4 scale is unproven here.
- **No FP4 indexer** — the lightning indexer's FP4 QK path is #3019.

## Next rungs

1. Land a **weight-free fixture** that drives `dsaIndexScores`/`dsaTopKIndices` at V4 shapes
   (128 heads, top-k 1024, head-dim 512) and digests the selection — the parity oracle.
2. File the **HCA `kvLayout`** proposal as its own leaf (rate-128 tier).
3. Bind the SWA branch as a *concurrent* side state (hand-off to #3017).
4. Decide observe-vs-wrap-vs-native from the fixture's parity result — not before.
