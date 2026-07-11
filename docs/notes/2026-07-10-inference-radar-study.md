---
title: "Inference Radar borrow study: 2 filed leaves, mostly-parity finding across an 8-axis MoE-serving roundup"
description: >
  A /study-repo pass over RunAnywhere's "Inference Radar" article (Hy3 / sparse-MoE
  long-context serving). The article is a pointer-collection, so each axis is pinned to
  the upstream PR/issue it cites (vLLM TLI #38174 @ 3af8789; SGLang DSA fused-topk #30274
  @ be70bfb), and witnessed on-axis against a real fak path:line. Decisive finding: fak
  is already at/near parity with most of this roundup — 2 PRESENT (cache_salt tenant
  isolation vllm_identity.go:70; KV-aware routing residency_router.go / kv_fleet_routing.go),
  4 already-tracked (MTP #3197/#3020, KV-FP8 #2240, T>0 spec #4202, disagg #3259), 1
  premature (SGLang DSA fused-topk maps to a production paged kernel fak has only as a
  scalar reference), 1 host-gated (NVFP4/Blackwell FP4 GEMM). 2 genuinely-new borrows
  filed under epic #4207: #4208 heterogeneous-vocab (TLI) drafting, #4209 FP8 checkpoint
  load + reference dequant. Both inspire (clean-room Go); no bytes vendored.
metadata:
  type: project
---

# Inference Radar borrow study

**Input:** RunAnywhere "Inference Radar" — a survey of what the open inference stacks
(vLLM, SGLang, ai-dynamo, DeepGEMM/CUTLASS) are adding to serve **Hy3-class sparse-MoE
long-context** models. Umbrella epic: **#4207**. Parent effort: #3357 (scout-loop).
Sibling study passes: #3946 (kernelwiki), #3921 (apex), #3900 (ktransformers), #3366
(lmcache), #3922 (apexagents).

## Acquisition + method (honesty)

The repo guard blocked a local `git clone` of the upstream repos (`POLICY_BLOCK` /
`TERMINAL`) and some `WebFetch` (`TRUST_VIOLATION`). Upstream anchors are therefore
pinned via `git ls-remote` HEAD + the GitHub API PR merge-SHA / changed-file list — real
and checkable, but **not** a local `path:line@sha` grep. The **fak-side** witnesses are
grounded at real `path:line` from a two-explorer deep read over `internal/compute`,
`internal/model`, `internal/modelengine`, `internal/gateway`, `internal/polymodel`,
`internal/engine` plus targeted follow-up reads. Next checkable step for full upstream
grounding: an operator clone at the pinned SHAs, then `grep` the cited symbols.

Pinned sources:

- vLLM @ `26ff616` (HEAD at study time). TLI: issue **#38173** → PR **#38174** merged
  at `3af878955935dd356182f6dd2ea9660acb1757be`; files `vllm/v1/spec_decode/vocab_mapping.py`,
  `draft_model.py`, `llm_base_proposer.py`, `vllm/config/speculative.py`,
  `tests/v1/spec_decode/test_vocab_mapping.py`. Algorithm: ICML 2025 Token-Level
  Intersection, arXiv:2502.05202.
- SGLang @ `4fcc994` (HEAD at study time). DSA fused-topk: PR **#30274** merged at
  `be70bfbdbbf69bdda78b8d932fe362f6f339e18a`; files
  `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`, `dsa_backend.py`,
  `triton_ops/dsa_metadata.py`.
- DeepGEMM (deepseek-ai/DeepGEMM) FP8 GEMM + CUTLASS 4.5.3 FP8/NVFP4; Hy3 FP8-only
  checkpoint (~300 GB E4M3).

## Per-axis witness table

| # | Article axis | Upstream anchor | fak seam (path:line) | Verdict |
|---|---|---|---|---|
| 1 | Prompt-cache **tenant isolation** (cache_salt) | vLLM cache_salt RFC | `internal/engine/vllm_identity.go:70` `deriveCacheSalt`; `internal/vcachegov/affinity.go:49` | **PRESENT** |
| 2 | **KV-aware / prefix-affinity routing** | ai-dynamo KV router | `internal/gateway/residency_router.go` (`PrefixResidencyIndex`, `CacheAwarePolicy.pickWorker`); `internal/gateway/kv_fleet_routing.go` (`FleetCacheRouter`) | **PRESENT** |
| 3 | **Heterogeneous-vocab (TLI) drafting** | vLLM #38174 @ 3af8789 | `internal/polymodel/polymodel.go:47-50,63` Family/vocab gate (`""` ⇒ no drafter) | **ABSENT → filed #4208** |
| 4 | **FP8 weight** load + compute | DeepGEMM / Hy3 FP8-only | `internal/compute/compute.go:62` FP8 dtype (`Quantized()==true`) but `internal/ggufload` has **zero** FP8 handling | **ABSENT (capability) → filed #4209** |
| 5 | **MTP / self-speculative decode** | Hy3 MTP; LightLLM Qwen-MTP | MTP tensors loaded (`internal/ggufload/gguf_qwen35_nextn_test.go`); verify/accept core (`internal/model/verify.go`, `polymodel.AcceptGreedy/AcceptTree`) | **tracked #3197, #3020** |
| 6 | **Low-bit KV cache** (int8/FP8 KV) | vLLM/SGLang KV quant | `internal/cachemeta/quantized_demote.go`, `internal/compute/kvcost.go` | **tracked #2240** |
| 7 | **Disaggregated prefill/decode** | article disagg | `internal/modelengine/native_pd.go` (intra-engine P/D) | **PRESENT (intra) / tracked #3259 (cross-worker)** |
| 8 | **SGLang DSA fused-topk v2** | SGLang #30274 @ be70bfb | `internal/model/dsa_index.go:66` (scalar reference — "without claiming a native DSA forward kernel"); `internal/model/moe.go:323 routeTopKSoftmax` full-sort | **premature (see below)** |
| — | **NVFP4 / Blackwell FP4 expert GEMM** | CUTLASS 4.5.3 / DeepGEMM | `internal/dsparity/dsparity.go:262` FP4 "proposed", `WitnessHostGated` | **host-gated note** |
| — | **Stochastic (T>0) lossless spec decode** | — | — | **tracked #4202** |
| — | **Tool-call (agent-layer) speculation** | — | `internal/abi/speculate.go` | **tracked #4102/#4201** |

## Filed this pass

- **#4207** — `epic(inference-radar-study)` umbrella.
- **#4208** — `feat(polymodel)`: heterogeneous-vocab (TLI) drafting — lift the shared-`Family`
  drafter restriction so any small model (different tokenizer) can draft for a hosted
  target via a token-level vocab intersection, lossless. Milestone: Model support &
  routing parity. CPU-provable, behind `FAK_POLYMODEL`.
- **#4209** — `feat(ggufload+compute)`: FP8 (E4M3/E5M2) checkpoint load + reference dequant —
  fak's FP8 dtype is a declared-but-unreachable seam (loader has zero FP8), so fak cannot
  load Hy3's FP8-only checkpoint. Reference dequant unblocks serving; native FP8 GEMM is
  the host-gated follow-on. Milestone: Decode parity across every backend.

## Not filed — reasoning

- **DSA fused-topk (#30274)** folds the page-table into a *production paged* fused top-k
  decode kernel ("drop page_size=1 expansion"). fak's DSA/MoE top-k are **scalar reference
  helpers** — `dsa_index.go:66` is explicit that it does not claim a native DSA forward
  kernel, and `moe.go:323` full-sorts. There is no production paged fused-topk kernel to
  apply the optimization to. Premature until a native DSA/MoE decode kernel exists (relates
  #3197, #1026). Incidental, article-unsourced fak observation: both top-k paths full-sort
  `O(E log E)`; a partial-selection `O(E log k)` would help Hy3-class 192-expert routing —
  a candidate for a separate perf witness, not a borrow.
- **NVFP4 / Blackwell FP4 expert GEMM** — already `"proposed"` + `WitnessHostGated` at
  `dsparity.go:262`; fak's CUDA is f32-only + off by default and there is no Blackwell
  on-fleet. Filing a Blackwell kernel ticket without a fak-side measurement would be
  speculative (same posture as #3946's 8 deferred decode-kernel borrows).

## Decisive finding

The headline is **parity, not gaps**: fak already ships the article's two hardest cache
disciplines (per-tenant `cache_salt` isolation, KV-aware prefix-affinity routing bench'd
vs Dynamo) and has the remaining serving frontier already tracked in a mature backlog
(#2236 superset epic + children). The two genuinely-new borrows both live in the
*decode-acceleration / low-bit-load* corner: cross-vocab drafting (widen who can draft)
and FP8 load (serve FP8-only checkpoints). Both are single-axis, CPU-provable, `inspire`.
