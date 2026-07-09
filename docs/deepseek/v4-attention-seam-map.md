---
title: "DeepSeek V4 CSA/HCA sparse attention → fak seam map"
description: >
  Design/plan note (issue #3016, epic #3006). Maps DeepSeek V4's hybrid CSA
  (Compressed Sparse Attention) + HCA (Heavily Compressed Attention) onto the
  concrete fak GLM-DSA / MLA / MSA seams it can reuse, and names each missing
  seam distinctly. No throughput or latency is claimed; every estimate is
  labeled MODELED or host-gated.
---

# DeepSeek V4 CSA/HCA sparse attention → fak seam map

**Status: design/plan only.** Issue **#3016**, parent epic **#3006** (native
DeepSeek-V4 kernel track). This is the canonical home under `docs/deepseek/`;
it is a **sibling of the other `docs/deepseek/*.md` plan notes** (the KV-layout,
FP4-quant, and MoE-dispatch plans) and supersedes the earlier snapshot filed at
`docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md`.

No kernel code lands here and **no throughput/latency/recall number is claimed as
measured**. Current-state facts are witnessed against the exact `path:line` cited
(read 2026-07-09 on `main`). External architecture facts are grounded only in the
sources listed under **Sources (researched July 2026)**; the V4 selector's "2×
speedup / 99.7% KV-entry recall" is quoted as **DeepSeek's own claim**, never a
fak result.

## Purpose and scope

DeepSeek V4 is **not** a dense-attention Llama port. Its load-bearing attention
is a **two-tier hybrid**: **CSA** (Compressed Sparse Attention, compression rate
4) applies DSA-style sparse top-k selection driven by a **lightning indexer**, and
**HCA** (Heavily Compressed Attention, compression rate 128) runs **dense**
attention over an aggressively compressed KV plane, with a **sliding-window
(SWA, window 128)** side branch and a **grouped output projection** over 128 query
heads.

fak already carries a *closely related but single-tier* path — the GLM-5.2
`glm_moe_dsa` **MLA + DSA lightning-indexer** kernels, plus a **block-granular MSA
selector** (MiniMax-M3). This note maps each of the four V4 attention pieces named
in #3016's deliverable — (a) CSA compressed-KV production, (b) lightning indexer /
top-k sparse selection, (c) HCA dense attention over compressed blocks, (d) grouped
output projection — onto the nearest real fak seam, and **fences** the pieces where
fak has no seam yet.

## V4 attention facts that drive the map

Source: DeepSeek V4 technical report, `https://arxiv.org/html/2606.19348v1`
(numbers as cited in #3016's grounding), and the SGLang V4 roadmap
`https://github.com/sgl-project/sglang/issues/23602` for prior art.

| V4 Pro fact | Value | Consequence for the seam map |
|---|---|---|
| Layers | 61 | per-layer indexer + two KV planes × 61 |
| Hidden size | 7168 | query / index projection input width |
| CSA compression rate | 4 | "compressed" KV plane (light compression) |
| HCA compression rate | 128 | "heavily compressed" KV plane (aggressive block compression) |
| Attention top-k | 1024 | sparse selection width per query (a config value, not a rewrite) |
| Query heads | 128 | grouped output projection width |
| Head dim | 512 | MLA-scale head (matches fak's MLA head geometry) |
| Query compression dim | 1536 | q-LoRA down-projection rank |
| SWA window | 128 | sliding-window side branch |
| Selector optimization | "2× selector speedup, 99.7% KV-entry recall" | **DeepSeek's claim** — the top-k selector is the hot inner loop; never a fak measurement |

The phrase "CSA applies **DSA sparse top-k selection** via a **lightning
indexer**" is exactly the mechanism fak already models for GLM-5.2, which is why
the **CSA half of V4 has a real fak seam today** and the **HCA half is a proposed
new one**.

## fak's existing DeepSeek-lineage machinery (what we build on)

- **MLA latent attention + DSA indexer forward** — `internal/model/glm_dsa.go`:
  q_a/q_b + kv_a/kv_b MLA projections, GLM interleaved RoPE, sparse causal masking
  by top-k, and o_proj (`glmDsaAttentionOutputFromTopK:18` /
  `glmDsaAttentionOutputFromTopKNormed:26`; learned indexer projections in
  `glmDsaTopKIndices:135`).
- **DSA index scoring / selection primitives** — `internal/model/dsa_index.go`:
  `dsaIndexScores:20`, `dsaTopKIndices:66`, `dsaSparseAttention:115`,
  `dsaIndexShare:208`, `dsaIndexDigest:232`.
- **Block-granular sparse selector (MSA)** — `internal/model/msa_index.go`:
  `msaBlockScores:44`, `msaSelectBlocks:78`, `msaSelectedKeyPositions:129`,
  `msaAttention:171`. This is fak's *block-level* selection witness and reuses
  `dsaSparseAttention` for the softmax — the nearest structural analogue to HCA's
  block-compressed plane.
- **KV-layout seam** — `internal/model/kvlayout.go`: the `kvLayout` interface
  (`:28`), `mlaKVLayout:98`, write side `mlaProject:151`, decompress-then-attend
  `attendOne:192`, selector `modelLayout:176`.
- **SWA windowed decode** — `internal/model/swa.go`: `MaxWindow:25`,
  `TrimToWindow:48` (bit-exact vs full-window, `TestBoundedWindowMatchesFullWindow`).
- **Device seams (optional, type-asserted)** — `internal/compute/dsa.go`:
  `DSASparseBackend:35` (`DSASparseAttend`), `DSAIndexBackend:102`
  (`DSAIndexSelect`); the generic backend attention contract is
  `internal/compute/compute.go:341` `Attention(q, kv, layer, causal, grp, scale)`.
- **Offload split hooks** — `internal/model/moe_offload.go`: `sparseAttend:55`,
  `indexSelect:72` (keeps indexer projections device-routable while experts offload).
- **GGUF/config plumbing** — `internal/ggufload/gguf_glm_tensors.go` (the
  `glm_moe_dsa` indexer/MLA/router tensor-name map + KV-b 2→1 merge) and the MLA +
  indexer config fields in `internal/model/config.go:185-193` (`QLoraRank`,
  `KVLoraRank`, `QKNopeHeadDim`, `QKRopeHeadDim`, `VHeadDim`, `IndexNHeads`,
  `IndexHeadDim`, `IndexTopK`, `IndexerTypes`).
- **Paged exact-span KV cache** — `internal/model/paged_glmdsa.go`
  (`newPagedGLMDsaKVCache`), the block-table cache the compressed planes would page.

## Seam map (a) — CSA compressed-KV production

CSA compresses the KV to a latent plane (rate 4) and selects over it. **Nearest
seam:** the MLA latent write/read path.

- **Reuse:** `internal/model/kvlayout.go:151` `mlaProject` (the write side that
  produces the cached compressed latent `c_KV`) and `:114` `mlaKVLayout.reconstructKV`
  / `:192` `attendOne` (decompress-then-attend). The layout selector
  `modelLayout:176` picks MLA when MLA geometry is present.
- **Fit:** the *shape* fits — MLA is a learned KV compression with a RoPE side
  channel, exactly CSA's family. The GLM-DSA forward already reads `kv_a_proj_with_mqa`
  → latent → `kv_b_proj` up-projection (`glm_dsa.go:59-69`).
- **Gap:** `mlaKVLayout` is **not parameterized by a "compression rate 4" block
  factor**, and `modelLayout` returns **one** layout — it cannot co-resident two
  compression tiers. See Missing seams #1 and #3.

## Seam map (b) — lightning indexer / top-k sparse selection

- **Reuse (direct fit):** `internal/model/dsa_index.go:20` `dsaIndexScores`
  computes exactly `Σ_h weights[q,h]·relu(scale·dot(index_q[q,h], index_k[k]))` —
  the lightning-indexer score shape. `dsaTopKIndices:66` selects the top-k key
  **positions** under the causal mask; **`topK` is a parameter**, so V4's top-k
  1024 is a config value, not a code rewrite. The learned indexer projections
  (wq_b / wk / weights_proj / k_norm + RoPE) already exist in
  `glm_dsa.go:135` `glmDsaTopKIndices`.
- **Reuse (selection sharing):** `dsaIndexShare:208` already reuses one full-layer
  top-k across several layers keyed by `dsaIndexDigest:232` — the "shared indexer"
  contract.
- **Device routability:** `internal/compute/dsa.go:102` `DSAIndexBackend.DSAIndexSelect`
  is the optional kernel seam; the selection stays f64-faithful so CPU↔device pick
  the identical key set.
- **Gap:** V4 feeds **one** selection into **two** attention tiers (CSA + HCA)
  within a layer; today's sharing is layer→layer, not tier→tier. See Missing seam #4.
  The **FP4 indexer QK path** (prior art) is out of scope here (host-gated; #3019 /
  the FP4-quant sibling note).

## Seam map (c) — HCA dense attention over compressed blocks

HCA runs **dense** attention over a **heavily** compressed (rate-128) KV plane —
i.e. each cached entry summarizes a *block* of tokens.

- **Nearest structural witness:** `internal/model/msa_index.go` already models
  **block-granular** attention: `msaBlockScores:44` max-pools per-key scores into
  blocks, `msaSelectBlocks:78` selects blocks (top-k ∪ always-on local window),
  `msaSelectedKeyPositions:129` broadcasts the block choice back to keys, and
  `msaAttention:171` runs the softmax via `dsaSparseAttention`. HCA's rate-128 plane
  is a **block-compressed** KV, so the block bookkeeping in `msa_index.go` is the
  closest existing shape.
- **Fence — no HCA seam exists yet.** MSA selects blocks but still attends the
  *uncompressed* K/V; HCA attends a *compressed-block* K/V that must first be
  **produced**. fak has **no rate-128 compressed-KV write path** and **no
  dense-over-compressed-block attend** sibling to `attendOne`. This is Missing
  seams #2 and #3 — a proposed new `kvLayout` implementation, not an existing file.

## Seam map (d) — grouped output projection

- **Reuse (fit):** the MLA grouped output path in
  `internal/model/glm_dsa.go:127` (batched `o_proj` over `nH*vHead` per-head concat,
  then per-token bias) already projects 128-head-scale attention concat to hidden.
  Head / group counts are config (`config.go:185-193`), so 128 q-heads is data.
- **Gap:** V4 must **combine** the outputs of the CSA branch, the HCA branch, and
  the SWA branch **before** `o_proj`. Today `glmDsaAttentionOutputFromTopKNormed`
  produces one attend output and projects it. A **multi-branch attention combine**
  ahead of the grouped projection is missing. See Missing seam #5.

## Reused fak seams (summary table)

| V4 requirement | fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| Lightning indexer score | `internal/model/dsa_index.go:20` `dsaIndexScores` | Direct fit (same formula). |
| CSA sparse top-k (1024/query) | `internal/model/dsa_index.go:66` `dsaTopKIndices` | Direct fit; `topK` is a parameter. |
| CSA sparse attend output | `internal/model/dsa_index.go:115` `dsaSparseAttention`; layer path `glm_dsa.go:18` | Direct fit for compute; two-tier layout is new. |
| Learned indexer projections | `internal/model/glm_dsa.go:135` `glmDsaTopKIndices` | Direct fit. |
| Compressed latent KV (CSA rate 4) | `internal/model/kvlayout.go:151` `mlaProject`, `:192` `attendOne`, `:28` `kvLayout` | Fit for shape; rate-4 parameter + two-tier compose missing. |
| HCA dense over compressed blocks | `internal/model/msa_index.go:44/78/129/171` (block machinery) | Nearest witness only; rate-128 KV production missing. |
| SWA branch (window 128) | `internal/model/swa.go:25` `MaxWindow`, `:48` `TrimToWindow` | Partial fit; concurrent side branch is new. |
| Grouped output projection (128 heads) | `internal/model/glm_dsa.go:127` `o_proj` path | Fit; multi-branch combine before it is missing. |
| Indexer sharing across layers | `internal/model/dsa_index.go:208` `dsaIndexShare`, `:232` `dsaIndexDigest` | Direct fit (layer→layer). |
| Device sparse attend / index select | `internal/compute/dsa.go:35` `DSASparseBackend`, `:102` `DSAIndexBackend` | Fit (host-gated device kernels). |
| Offload split (index vs expert) | `internal/model/moe_offload.go:55` `sparseAttend`, `:72` `indexSelect` | Fit. |
| Backend attention contract | `internal/compute/compute.go:341` `Attention(...)` | Fit; where a native V4 kernel binds. |
| MLA/indexer config + GGUF map | `internal/model/config.go:185-193`; `internal/ggufload/gguf_glm_tensors.go` | Fit for `glm_moe_dsa`; V4 two-rate fields + family spec missing. |
| Paged block KV cache | `internal/model/paged_glmdsa.go` `newPagedGLMDsaKVCache` | Fit as the paging substrate for the compressed planes. |

## Missing seams (named distinctly)

1. **Dual-rate / multi-plane `kvLayout`.** `modelLayout:176` returns a single
   `kvLayout`; V4 needs the CSA rate-4 plane and the HCA rate-128 plane **co-resident**
   for the same layer, both driven by the same selection. No composition seam exists.
2. **HCA rate-128 compressed-KV *production* + dense-over-compressed *attend*.** A new
   `kvLayout` impl (block-compression write side + a dense attend sibling to
   `attendOne:192`). MSA gives block *selection*, not block *compression*.
3. **CSA compression-rate parameterization.** `mlaKVLayout` is rate-agnostic latent;
   a "compression rate 4" block factor is not a config field or a layout parameter today.
4. **One-selection → two-tier fan-out.** `dsaIndexShare:208` shares a selection
   layer→layer; V4 shares one lightning-indexer top-k across the CSA and HCA attends
   **within a layer**. That fan-out does not exist.
5. **Multi-branch attention combine before the grouped projection.** Combining CSA +
   HCA + SWA branch outputs ahead of `o_proj` (`glm_dsa.go:127`) has no seam.
6. **SWA as a concurrent side branch.** `swa.go` trims the *whole* cache; running a
   window-128 branch **in parallel** with the sparse tiers (not a whole-cache trim) is new.
7. **DeepSeek-V4 config parse (two compression rates).** `config.go` has the MLA +
   indexer fields but **no** `csa_compression_rate` / `hca_compression_rate` fields and
   no HF/GGUF parse for them.
8. **`deepseek_v4` resolver family spec.** `internal/model/tensor_resolver.go` has
   per-family specs (llama/mixtral/gptoss/…) and `ggufload` has the `glm_moe_dsa` map;
   a distinct V4 tensor-name family (two KV planes, dual indexer) is unmapped.
9. **FP4 lightning-indexer QK path (host-gated).** Prior art ships an FP4 indexer; fak's
   indexer scores in f32/f64. This is a quant seam (sibling FP4-quant note / #3019), fenced here.
10. **Decode-time incremental top-k at 1M context (host-gated).** Batched prefill
    indexing exists (`glm_dsa.go:135` over `seq`); incremental decode-time selection at
    V4 scale is unmodeled.

## Gateway / agent seams already landed (context, not attention)

These are already in fak under epic #3006 and bound V4 into the serving path without
touching the kernel — cited so this note is not re-proposing them:

- **Long-context budget calculator (deterministic, provenance-labeled)** —
  `internal/gateway/deepseek_budget.go`. Every figure carries a closed label
  (`SOURCE_DOCUMENTED` / `PAPER_CLAIMED` / `MODELED` / `WITNESSED`) and
  `claimNativeSupport` **refuses** a "1M native support" claim while every load-bearing
  number is MODELED/PAPER_CLAIMED. This note adopts the same discipline.
- **DeepSeek pricing / usage counters** — `internal/gateway/deepseek_pricing.go`,
  `internal/agent/deepseek_usage_test.go` (cache-hit vs cache-miss input rates; a cost
  *projection* over OBSERVED provider counters, never a fak benchmark).
- **DeepSeek Anthropic-compatible route + fences** —
  `internal/gateway/deepseek_anthropic.go` (`/v1/messages` parity, unsupported-block
  refusals, model-id resolution) and the self-host smoke `deepseek_selfhost_smoke_test.go`.
- **Reasoning-content preservation / conformance** — `internal/agent/reasoning.go`,
  `internal/agent/deepseek_reasoning_test.go`, `deepseek_conformance_test.go`.

## Prior-art baseline (what native work would be measured against)

Per #3016's grounding (SGLang V4 roadmap `.../issues/23602`) and the standard sparse
stack. This is the *parity oracle* fak fixtures must match before any native claim —
**not** a set of numbers reproduced here.

| Prior art | What it provides | Relation to fak |
|---|---|---|
| SGLang DeepSeek V4 (FP4 indexer, FlashMLA sparse prefill, ragged indexer, breakable CUDA graph, HiSparse pool) | Tuned reference serving path | Parity oracle for shape/layout before a native claim (host-gated). |
| FlashMLA | Fused MLA prefill/decode kernel | What a native `mlaKVLayout` + `attendOne` would wrap or reimplement (host-gated). |
| FlashInfer / CUTLASS FP4 | FP4 GEMM + indexer primitives | Feeds Missing seam #9 (FP4 indexer). |
| vLLM recipes | Reference config + numerics | Secondary numeric oracle. |

`fak sota` / `internal/sotamatrix` is the in-repo home for keeping this matrix live;
this note records the snapshot, it does not re-run the tool.

## The implement / wrap / observe decision (recommended)

1. **Observer-only first (recommended start).** The selection is already digestible
   (`dsaIndexDigest:232`); drive `dsaIndexScores`/`dsaTopKIndices` at V4 shapes (128
   heads, top-k 1024, head-dim 512) in a **weight-free fixture** and diff the selection
   digest against the reference — **no** device kernel. De-risks layout parity before any GEMM.
2. **Wrap a tuned engine** (FlashMLA / SGLang) behind
   `internal/compute/compute.go:341` `Attention` — fak owns routing + the KV planes,
   the engine owns the kernel (host-gated).
3. **Native kernel** — only after the fixture proves CSA/HCA shape + layout parity
   against an external reference. This note **refuses** native implementation until that
   fixture exists (matches #3016's acceptance gate).

## Acceptance mapping

The issue's acceptance criteria, bullet by bullet:

- **"Names each existing fak seam it reuses"** — the **Reused fak seams** table and the
  four per-piece seam-map sections cite concrete `path:line` seams:
  `dsa_index.go:20/66/115/208/232`, `glm_dsa.go:18/26/127/135`,
  `kvlayout.go:28/151/176/192`, `msa_index.go:44/78/129/171`, `swa.go:25/48`,
  `compute/dsa.go:35/102`, `compute/compute.go:341`, `moe_offload.go:55/72`,
  `config.go:185-193`, `ggufload/gguf_glm_tensors.go`, `paged_glmdsa.go`.
- **"Names each missing seam"** — the **Missing seams** section lists ten distinct,
  individually named gaps (dual-rate layout, HCA production+attend, CSA rate param,
  one-selection→two-tier fan-out, multi-branch combine, concurrent SWA branch, V4 config
  parse, `deepseek_v4` resolver spec, FP4 indexer, decode-time incremental top-k).
- **"Cites sources with research date"** — see **Sources (researched July 2026)** below;
  the V4 facts table and prior-art table each name their source URL.
- **"Makes NO throughput claim"** — no throughput/latency/recall number appears as
  measured. The selector "2×/99.7%" figure is quoted explicitly as **DeepSeek's claim**;
  estimates are labeled MODELED and GPU/engine items are labeled **host-gated**.
- **Cross-links** — parent epic **#3006**; sibling `docs/deepseek/*.md` plan notes and
  the earlier `docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md` snapshot are
  linked in the header.

## Honest fences (what is NOT decided or built)

- **No HCA seam exists** — the rate-128 tier is a proposed new `kvLayout`, not a file.
- **No two-tier CSA/HCA cache object** — that composition is Missing seam #1/#2; this
  note maps compute only.
- **No V4 weights, no numerics** — every "fit" above is a *shape/layout* fit, not a
  verified bit-exact match to V4.
- **No 1M-context decode model** — incremental top-k at V4 scale is unproven (Missing #10).
- **No FP4 indexer** — the lightning indexer's FP4 QK path is fenced (Missing #9, host-gated).

## Next rungs

1. Land the **weight-free V4-shape fixture** driving `dsaIndexScores`/`dsaTopKIndices`
   and digesting the selection — the parity oracle.
2. File the **HCA rate-128 `kvLayout`** proposal and the **dual-rate compose** as their
   own leaves (Missing seams #1–#3).
3. Add the **`deepseek_v4` config fields + resolver family spec** (Missing seams #7/#8).
4. Bind the SWA branch as a *concurrent* side state (Missing seam #6; hand-off to the
   KV-layout sibling note).
5. Decide observe-vs-wrap-vs-native from the fixture's parity result — not before.

## Sources (researched July 2026)

- DeepSeek V4 technical report — `https://arxiv.org/html/2606.19348v1` (61 layers,
  hidden 7168, CSA rate 4 / HCA rate 128, top-k 1024, 128 query heads, head dim 512,
  q-compression dim 1536, SWA window 128; the "2× selector speedup / 99.7% KV-entry
  recall" selector claim — DeepSeek's own, not a fak measurement).
- DeepSeek-V4-Pro model card — `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro`
  (parameter counts, precision widths; as recorded in `deepseek_budget.go`).
- SGLang DeepSeek V4 roadmap — `https://github.com/sgl-project/sglang/issues/23602`
  (FP4 indexer, FlashMLA sparse prefill, ragged indexer, breakable CUDA graph, HiSparse
  pool) — the prior-art parity baseline.
- fak in-repo seams (read 2026-07-09 on `main`): `internal/model/dsa_index.go`,
  `glm_dsa.go`, `msa_index.go`, `kvlayout.go`, `swa.go`, `config.go`,
  `tensor_resolver.go`, `paged_glmdsa.go`; `internal/compute/dsa.go`, `compute.go`;
  `internal/model/moe_offload.go`; `internal/ggufload/gguf_glm_tensors.go`;
  `internal/gateway/deepseek_budget.go`, `deepseek_pricing.go`, `deepseek_anthropic.go`;
  `internal/agent/reasoning.go`.
