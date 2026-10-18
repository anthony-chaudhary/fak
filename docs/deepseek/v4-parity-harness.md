---
title: "DeepSeek V4 deterministic batch-invariant parity-harness spec + offline witness"
description: "The parity-harness SPECIFICATION every future DeepSeek-V4 native kernel must pass before a perf claim can close: batch-size / cache-hit / request-order / seed invariance rows with bitwise-preferred, FP4/FP8-bounded tolerance rules, mapped onto fak's real GLM-DSA seams, plus a weight-free witness on the real internal/model selector seam that needs no GPU. A PLAN, not a benchmark; no throughput/latency/score is claimed."
---

# DeepSeek V4 deterministic batch-invariant parity-harness

**2026-07-09.** Issue **[#3021](https://github.com/anthony-chaudhary/fak/issues/3021)**,
parent epic **[#3006](https://github.com/anthony-chaudhary/fak/issues/3006)** (native
DeepSeek-V4 kernel track). This is a **specification + offline witness only** — no native
V4 kernel lands here, and **no performance number is claimed**. Current-state claims are
witnessed against the exact `path:line` cited (read 2026-07-09 on `main`). It is a sibling
of the other `docs/deepseek/*.md` / `docs/notes/DEEPSEEK-V4-*.md` plan notes:

- [`docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md`](../notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md) — CSA/HCA sparse-attention seam map (#3016)
- [`docs/notes/DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md`](../notes/DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md) — two-tier KV layout (#3017)
- [`docs/notes/DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md`](../notes/DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md) — MoE dispatch baseline (#3018)
- [`docs/notes/DEEPSEEK-V4-FP4-QUANT-PLAN-2026-07-08.md`](../notes/DEEPSEEK-V4-FP4-QUANT-PLAN-2026-07-08.md) — FP4 expert/indexer quant (#3019)
- [`docs/benchmarks/DEEPSEEK-V4-PERF-SCORECARD.md`](../benchmarks/DEEPSEEK-V4-PERF-SCORECARD.md) — TTFT/TPOT scorecard (#3014)

## Thesis — determinism is a *correctness* property that a perf kernel must not silently trade away

DeepSeek ships **deterministic, batch-invariant kernel libraries** for bitwise
reproducibility (V4 technical report, https://arxiv.org/html/2606.19348v1). That is not a
nicety: a decode's output must not depend on *how it happened to be batched*, whether a
prefix was served from cache, or the order concurrent requests arrived in. V4's three
load-bearing fused paths are exactly where a naive kernel loses that property:

- **Fused mHC / heavily-compressed attention (HCA)** — a fused compressed-KV accumulation
  whose reduction order can drift with batch size (the classic batch-variant matmul).
- **MoE-overlap dispatch** — experts batched across concurrent requests; a routing or
  grouped-GEMM race makes one request's output depend on its neighbours' arrival order.
- **Sparse-attention lightning-indexer top-k** — a top-k selector whose tie-breaks wobble
  unless the ordering is total and seed-stable.

So the honest deliverable *before* any V4 kernel is written is a **parity harness**: a
fixed table of invariance rows + expected fields, with tolerance rules that prefer bitwise
and bound the genuinely mixed-precision paths — and a **claim-discipline** rule that no
perf ticket closes on "faster" alone. This document is that spec, and the row table below is
its single source; `internal/model/dsa_v4parity_test.go` is its offline, weight-free witness on
the real selector seam.

## The V4 facts that drive the harness (from the issue grounding)

Source: DeepSeek V4 technical report, https://arxiv.org/html/2606.19348v1, and the HF model
card https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro (numbers as cited in #3021 / the
sibling notes; recorded here as **SOURCE_DOCUMENTED** / **PAPER_CLAIMED**, never as a fak
measurement).

| V4 fact | Value | Consequence for the harness |
|---|---|---|
| Deterministic batch-invariant kernels | shipped by DeepSeek | bitwise reproducibility is the *target* tolerance, not an aspiration |
| Attention top-k | 1024 keys/query | the top-k selector's tie-break must be total + seed-stable |
| CSA / HCA compression rates | 4 / 128 | two fused attention tiers, both batch-invariance surfaces |
| MoE experts | FP4 | grouped expert GEMM gets an FP4-**bounded** tolerance, not bitwise |
| Most other params | FP8 | fused-attention accumulation gets an FP8-**bounded** tolerance where bitwise is unreachable |
| Prompt caching | on by default (hit/miss counters) | a cache hit must be a compute short-cut, never a semantics change |

## Seam map — parity requirement → fak seam (`path:line`) or proposed

fak already carries the GLM-5.2 **MLA + DSA lightning-indexer** machinery, which is the same
mechanism family as V4's CSA/indexer half. The harness is grounded on those *real* seams;
the pieces fak has no kernel for yet are named as distinct **proposed** seams.

| Parity requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| **Sparse top-k selector, stable tie-break** | `internal/model/dsa_index.go:66` `dsaTopKIndices` (sort: score desc, then key position asc) | **Direct fit.** The tie-break is already total and deterministic — the exact property the seed/request-order rows assert. `topK` is a parameter, so V4's 1024 is a config value. |
| **Lightning-indexer score** | `internal/model/dsa_index.go:20` `dsaIndexScores` (`sum_h w[q,h]·relu(scale·dot)`) | **Direct fit.** Same formula shape as V4's indexer. |
| **Selected-index digest (drift detector)** | `internal/model/dsa_index.go:232` `dsaIndexDigest` (sha256 over indices) | **Direct fit.** Turns any index drift into a bit-diff — the comparison primitive the batch/seed digest rows use. |
| **Every-4-layer index-share contract** | `internal/model/dsa_index.go:208` `dsaIndexShare` (full computes, shared reuses prior top-k) | **Direct fit.** Seed-stability must hold end-to-end across shared layers. |
| **Compressed-KV production (CSA rate-4)** | `internal/model/kvlayout.go:28` `kvLayout` interface; `:98` `mlaKVLayout` | **Fit for CSA.** The `kvLayout` interface is the extension point. |
| **Heavily-compressed attention (HCA rate-128)** | *No seam* — a **second** `kvLayout` impl | **Gap (proposed).** The fused HCA accumulation is the batch-invariance surface for the `batch/hca-logits` + `cache/prefix-logits` rows. |
| **MoE router (top-k routing)** | `internal/ggufload/gguf_glm_tensors.go:57` `ffn_gate_inp` + `:58` `exp_probs_b` (router gate + score-correction bias) | **Fit for the routing decision**; the fused MoE-overlap *dispatch* kernel is proposed (see #3018 note). |
| **FP4 expert GEMM** | *No native FP4 dtype* — see the FP4 quant note (#3019) | **Gap (proposed).** Experts are FP4, so the request-order expert-output row is FP4-**bounded**, not bitwise. |
| **Cache-hit prefix accounting** | `internal/gateway/deepseek_pricing.go` (hit/miss counters); `internal/gateway/deepseek_budget.go` (KV budget); `internal/agent` `Usage.CachedPromptTokens`/`UncachedPromptTokens` | **Fit for the accounting**; the cached-vs-recompute numeric probe is proposed. |
| **Reasoning-content preservation across paths** | `internal/agent` `deepseek_reasoning_test.go` (`reasoning_content` never lifted into tool calls, never dropped) | **Adjacent fit** — the existing determinism-of-parsing discipline the harness extends to kernels. |

## The parity rows (the harness table)

The rows live in this table. (They were first encoded as pure data in a standalone
`internal/dsparity` fixture; it had no caller and checked only a synthetic re-implementation of
the tie-break, so it was removed.)

| id | axis | kernel | compare field | variants | tolerance | witness |
|---|---|---|---|---|---|---|
| `batch/indexer-scores-1-N-M` | batch-size | lightning-indexer | index-digest | batch 1/8/64 | **bitwise** | offline-synthetic |
| `batch/hca-logits-1-N-M` | batch-size | fused-mhc | logits | batch 1/8/64 | fp8-bounded (1e-3) | **host-gated** |
| `cache/prefix-next-token` | cache-hit | fused-mhc | next-token-id | cold vs warm | **bitwise** | offline-synthetic |
| `cache/prefix-logits-bounded` | cache-hit | fused-mhc | logits | cold vs warm | fp8-bounded (5e-4) | **host-gated** |
| `order/expert-routing-permutation` | request-order | moe-overlap | expert-routing | identity/reversed/shuffled | **bitwise** | offline-synthetic |
| `order/expert-output-bounded` | request-order | moe-overlap | logits | identity vs shuffled | fp4-bounded (2e-2) | **host-gated** |
| `seed/topk-selection-fixed` | seed | sparse-attention | topk-indices | seed=1 runA/runB | **bitwise** | offline-synthetic |
| `seed/index-share-layers` | seed | sparse-attention | index-digest | seed=7 runA/runB | **bitwise** | offline-synthetic |

Each row carries the expected FIELDS: `id`, `axis`, `kernel`, `compare_field`, `variants`,
`tolerance`, `max_abs_tol`, `max_rel_tol`, `witness`, `fak_seam`, `rationale`.

## Tolerance rules (bitwise preferred; FP4/FP8 bounded)

A closed tolerance-class vocabulary:

- **`bitwise`** — the default and the goal. `max_abs_tol == max_rel_tol == 0`. Batch-invariant
  kernels make this *achievable* for index selection, routing decisions, and greedy
  next-token identity, so those rows demand it.
- **`fp8-bounded`** — for a fused-attention (HCA/mHC) accumulation where FP8 reduction
  associativity cannot be pinned bit-for-bit. A **documented**, non-zero abs/rel bound is
  required, and the bounds shown here (1e-3 / 5e-4) are
  **MODELED** placeholders that must be replaced by a **tuned baseline** measured on the
  native kernel — they are not witnessed tolerances.
- **`fp4-bounded`** — for the FP4 expert GEMM's grouped accumulation (experts are FP4 in the
  V4 checkpoint). Same rule, a wider **MODELED** placeholder bound (2e-2) pending a tuned
  baseline.

Bitwise rows must stay a strict majority, so the FP4/FP8 escape hatch cannot quietly become
the default.

> The numeric tolerances (1e-3, 5e-4, 2e-2) are **MODELED** starting points, **not**
> witnessed error bounds. Each **host-gated** row's bound must be re-derived from a
> **tuned-baseline** run of the real native kernel before it is trusted.

## Offline witness on the real selector seam (no weights, no GPU)

The weight-free witness in `internal/model/dsa_v4parity_test.go` drives the ACTUAL `dsaIndexScores` / `dsaTopKIndices` /
`dsaIndexShare` / `dsaIndexDigest` at V4 scale (top-k 1024 over a synthetic 4096-key slice),
so the rows it covers are checked on the real code path, not a re-implementation:

- `TestDSAV4ScaleSelectionIsRequestOrderInvariant`: `order/expert-routing-permutation` and
  `seed/topk-selection-fixed` on the selector: the selection digest is bitwise identical no
  matter what order the keys are presented in.
- `TestDSAV4ScaleTieBreakIsTotalAndSeedStable`: a full tie resolves by the total,
  position-ascending tie-break, identically across runs.
- `TestDSAV4ScaleIndexShareIsBitwiseAcrossLayers`: `seed/index-share-layers`: shared layers
  reuse the computed top-k bitwise.
- `TestDSAV4ScaleDigestDetectsDrift`: the index digest the batch/seed rows compare against
  changes on a single perturbed index, so drift cannot go unnoticed.
- `TestDSAV4ScaleTopKIsCausalAndBounded` and `TestDSAV4ScaleScoringPipelineIsDeterministic`:
  causal well-formedness and a deterministic scoring pipeline.

Run it with `go test ./internal/model -run 'TestDSAV4Scale' -count=1`.

The **host-gated** rows (`batch/hca-logits`, `cache/prefix-logits`, `order/expert-output`)
need a real native V4 kernel on a GPU to produce the compared logits, and are intentionally
left un-witnessed here rather than faked. The batch-size half of `batch/indexer-scores-1-N-M`
and `cache/prefix-next-token` have no offline witness yet beyond that digest primitive.

## Claim discipline — no perf ticket closes on "faster" alone

This harness is the referee for the epic's perf tickets, mirroring the honesty fence in
[`internal/deepseekbench`](../../internal/deepseekbench/deepseekbench.go) (`CompareSpeedup`
refuses a delta without shared prompt shape + verified quality parity + two live rows) and
the provenance discipline in [`internal/gateway/deepseek_budget.go`](../../internal/gateway/deepseek_budget.go)
(`claimNativeSupport` refuses "1M native support" while every load-bearing figure is
MODELED/PAPER_CLAIMED). The rule for #3006's kernel tickets:

> A V4 kernel perf claim is **inadmissible** until the parity rows for the axes it touches
> are **witnessed** (bitwise, or inside a tuned-baseline bound). "Faster" without parity is
> an undetected regression, not a result — the same reason a "tests pass" commit that
> deleted the assertions is not a pass.

The wiring: each perf ticket names the parity row ids it must keep green; the row's
`witness` field says whether that is an offline `go test` or a host-gated kernel run; and
the ship commit for a kernel cites the witnessed rows, exactly as the budget/scorecard
notes require a provenance label on every number.

## Acceptance mapping

Bullet-by-bullet against #3021's acceptance / witness criteria:

- **"a test fixture defines the parity ROWS and expected FIELDS"** → the row table and the
  expected-fields list above (`id`, `axis`, `kernel`, `compare_field`, `variants`, `tolerance`,
  `max_abs_tol`, `max_rel_tol`, `witness`, `fak_seam`, `rationale`). The original data fixture
  (`internal/dsparity`) was removed; this table is now the source.
- **"offline-runnable"** → `go test ./internal/model -run 'TestDSAV4Scale'` needs no weights,
  network, or GPU, and every axis has at least one weight-free row.
- **Parity rows: same prompt at batch 1/N/M** → `batch/indexer-scores-1-N-M` (bitwise,
  offline) + `batch/hca-logits-1-N-M` (fp8-bounded, host-gated).
- **same prefix with and without cache hit** → `cache/prefix-next-token` (bitwise, offline)
  + `cache/prefix-logits-bounded` (fp8-bounded, host-gated).
- **same expert routing under different request ordering** → `order/expert-routing-permutation`
  (bitwise, offline) + `order/expert-output-bounded` (fp4-bounded, host-gated); the offline
  witness is executed by `TestDSAV4ScaleSelectionIsRequestOrderInvariant`.
- **same sparse-attention top-k selector under deterministic seeds** → `seed/topk-selection-fixed`
  + `seed/index-share-layers` (both bitwise, offline), mapped to `dsaTopKIndices` /
  `dsaIndexShare`.
- **tolerance: bitwise where possible; bounded for FP4/FP8** → the closed tolerance-class
  vocabulary (`bitwise` ⇒ zero tol; `fp4/fp8-bounded` ⇒ documented positive tol), with bitwise
  rows a strict majority.
- **tie to commit/claim discipline: no perf ticket closes with only "faster"** → the "Claim
  discipline" section, wired to the same refusal posture as `deepseekbench.CompareSpeedup`
  and `deepseek_budget.claimNativeSupport`.
- **first witness needs NO real weights (synthetic/offline)** → `dsa_v4parity_test.go` is
  synthetic + offline; the numeric rows that genuinely need a kernel are labeled host-gated,
  not faked.

## Host-gated / MODELED / follow-up

- **Host-gated** (need a real native V4 kernel on a GPU to witness): `batch/hca-logits-1-N-M`,
  `cache/prefix-logits-bounded`, `order/expert-output-bounded`.
- **MODELED, needs a tuned baseline** (not witnessed error bounds): the FP8 tolerances
  (1e-3, 5e-4) and the FP4 tolerance (2e-2). Re-derive each from a tuned-baseline run.
- **Proposed seams** (no fak kernel yet): the HCA (rate-128) `kvLayout` impl, the fused
  MoE-overlap dispatch kernel, and the FP4 expert GEMM — all named distinctly above and in
  the sibling notes (#3016 / #3017 / #3018 / #3019).
- **Follow-up**: once the native kernels exist, add a witness runner that flips the
  host-gated rows from spec to measured, and cite the witnessed row ids in each kernel's ship
  commit.

## Sources (researched July 2026)

- DeepSeek V4 technical report — https://arxiv.org/html/2606.19348v1 (batch-invariant
  deterministic kernels; CSA/HCA; attention top-k 1024; FP4 experts + FP8 rest).
- DeepSeek-V4-Pro HF model card — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
  (mixed FP4/FP8 precision, 1M context, param counts).
- DeepSeek pricing / prompt-cache counters — https://api-docs.deepseek.com/quick_start/pricing
  (hit/miss token axes the cache-hit rows account against).
