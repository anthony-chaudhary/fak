# Qwen2.5-1.5B prompt-paraphrase robustness — 2026-08-15

**Verdict:** At 1.5B, the best concise contracts and the exact solve/check scaffold tied at 19/24, while alternate scaffold phrasings fell to 9/24 and 13/24. Compared with 3B, scale improved the exact solve/check prompt by 3 answers but did not make scaffolding robust as a treatment class.

## Question

Below the observed Qwen size crossover, does a six-arm wording matrix explain why concise contract was the safer default, and which prompt-level behavior changes at 3B?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Model: Qwen2.5-1.5B-Instruct, weights SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Protocol: the same 24 fixed four-choice tasks; 64-token cap; first explicit answer-letter grading.
- Arms: the same three concise-contract and three scaffold paraphrases used at Qwen2.5-3B and SmolLM3-3B.
- Raw result: [`2026-08-15-qwen25-15b-prompt-paraphrases.json`](2026-08-15-qwen25-15b-prompt-paraphrases.json), SHA-256 `f7f7ef5a873f63b1e14cbc7fe98e4a767f699462f87f4d24eee30c3aad672bb4`.

## Results

| Treatment | Wording | 1.5B correct | 3B correct | 1.5B strict | 1.5B output tokens |
|---|---|---:|---:|---:|---:|
| concise contract | minimal contract | 19/24 | 19/24 | 9/24 | 374 |
| concise contract | only-letter/no-explanation | 19/24 | 20/24 | 0/24 | 115 |
| concise contract | precise-solver role | 17/24 | 19/24 | 22/24 | 55 |
| reasoning scaffold | solve/check/options | 19/24 | 22/24 | 20/24 | 60 |
| reasoning scaffold | eliminate/verify | 9/24 | 14/24 | 0/24 | 968 |
| reasoning scaffold | private two-pass check | 13/24 | 16/24 | 7/24 | 302 |

## Interpretation

- Contract range was **17–19/24** at 1.5B and **19–20/24** at 3B.
- Scaffold range was **9–19/24** at 1.5B and **14–22/24** at 3B.
- The exact solve/check scaffold improved from 19/24 to 22/24 with scale; the two alternate scaffolds also improved, but remained below every 3B contract variant.
- At 1.5B, the strongest robust choice is not uniquely scaffold or contract: two contract variants and the exact scaffold tie at 19/24. The contract family has the better worst case, 17/24 versus 9/24.
- This resolves the apparent conflict with the earlier Q8 1.5B cell, where the canonical contract scored 20/24 and scaffold 16/24. Runtime/quantization and prompt realization can move individual cells; both studies agree that generic scaffold routing is unsafe at 1.5B.

## Honest boundary

- The 1.5B and 3B Transformers checkpoints are architecture-matched but not the same quantization/runtime as the pure-fak Q8 ladder.
- Three phrasings per treatment do not exhaust prompt space.
- Single deterministic runs over 24 tasks do not support population-level significance.
- Strict-format compliance and semantic correctness remain distinct outcomes.

## Decision

Keep concise contract as the robust Qwen2.5-1.5B fallback. The exact solve/check scaffold may be pinned as an equal-scoring low-output alternative in this runtime, but generic scaffold paraphrases are not admissible substitutes. At 3B, the same exact prompt becomes a witnessed optimization; prompt identity remains part of the routing key.
