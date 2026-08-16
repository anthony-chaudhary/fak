# Phi-3.5-mini canonical bound-routing transfer — 2026-08-16

**Verdict: reject admission at the frozen 23/24 bar.** Phi-3.5-mini produced
**84/96 exact calls and 21/24 unique requests**, perfectly invariant across four
source-catalog orders. It ties the best measured smaller route, Qwen2.5-1.5B, but does
not establish an admitted local tier.

## Question

Does Phi-3.5-mini's earlier binary-option robustness transfer to the final frozen
24-tool canonical named-tool plus deterministic-binding route?

This is the captured spine for [#6981](https://github.com/anthony-chaudhary/fak/issues/6981)
under cross-model sensitivity umbrella
[#6692](https://github.com/anthony-chaudhary/fak/issues/6692).

## Frozen protocol

- **Model:** resident Phi-3.5-mini-instruct checkpoint, provenance digest captured below.
- **Hardware:** one sanctioned NVIDIA L4-class accelerator.
- **Workload:** the same 24 held-out requests and 24 confusable tools used throughout
  the final transfer ladder.
- **Order control:** canonical input, reversed input, and two seeded input shuffles;
  every arm canonicalizes to one byte-identical lexicographically sorted catalog.
- **Decode:** unchanged canonical named-tool prompt, JSON Schema prefix constraint,
  greedy generation, 128-token cap.
- **Postprocessor:** unchanged fail-closed `send_email` literal binder; selected-tool
  changes are forbidden.
- **Calls:** 96 total (24 requests × 4 source orders).
- **Admission rule, declared before the run:** at least 23/24 unique exact requests.

## Results

| Valid | Correct tool | Exact before | Exact after | Unique exact | Output tokens | Generation sum | Median/call |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96/96 | **96/96** | **84/96** | **84/96** | **21/24** | 3,772 | 153.160 s | 1482.622 ms |

Every source-order arm scored 21/24. Prompt digests and raw generations were identical
per request across all four arms. The binder reported 16 bound email calls, but those
calls were already exact; it changed no correctness outcomes.

Final frozen-route ladder:

| Model | Exact calls | Unique exact | Admission |
|---|---:|---:|---|
| Qwen2.5-0.5B | 32/96 | 8/24 | reject |
| Qwen2.5-1.5B | **84/96** | **21/24** | reject |
| SmolLM2-1.7B | 80/96 | 20/24 | reject |
| Qwen2.5-3B | 80/96 | 20/24 | reject |
| SmolLM3-3B | 80/96 | 20/24 | reject |
| Phi-3.5-mini | **84/96** | **21/24** | **reject** |

Phi selected the expected tool in all 96 calls. Its only failures were the three
knowledge-base requests, where it selected `search_kb` but paraphrased or copied the
request instead of producing the exact expected query. This isolates the measured
remaining floor as search-argument normalization rather than structural validity,
tool discrimination, or source-order sensitivity.

## Interpretation and boundary

Phi's earlier option robustness does transfer into perfect selected-tool accuracy, but
not into the exact argument fidelity required by the declared admission rule. It ties
rather than surpasses Qwen2.5-1.5B and uses more output tokens in this run. Therefore no
small local checkpoint in the measured 0.5B-through-mini ladder is admitted on this
exact-call benchmark.

This result does not claim Phi is globally equivalent to Qwen2.5-1.5B, nor that every
small checkpoint fails. It covers one frozen task set, prompt, tool catalog, and decode.
Any search-query normalizer or semantic argument metric would be a separately declared
intervention, not a post-hoc change to this witness.

## Reproduction and validation

- Model provenance digest: `18ca3acc3337f7a5958a4e62df4c4ec91257bde08c4ef07d10fc1d605ffa8cb1`
- Runner SHA-256: `7f289a844f87a61d8887acd6c363a6da7825f7becb2fdbc970c567202900e233`
- Raw artifact SHA-256: `dcadaba03d1f772a8a7c70a39017b604cfb4a4b8b118b906b1889b734ec0535c`
- Captured log SHA-256: `c78696d2441298928c5ebc8fbd065dded2bb9f2923096884532cfa1e8c0e6420`
- Runtime: 210.831 seconds
- Environment: Python 3.10.12, PyTorch 2.13.0+cu130,
  Transformers 5.14.1, `lm-format-enforcer`
  0.11.3, CUDA device `NVIDIA L4`.

An independent validator recomputed the canonical catalog digest and every aggregate,
checked all 96 rows and 24 rows per source order, matched prompt digests, proved raw
output invariance, and verified binding never changed the selected tool. Remote and
local raw hashes matched. The accelerator returned to 1,022 MiB and 0% utilization.

Captured artifact:
[`2026-08-16-phi35-canonical-binding-transfer.json`](2026-08-16-phi35-canonical-binding-transfer.json).
