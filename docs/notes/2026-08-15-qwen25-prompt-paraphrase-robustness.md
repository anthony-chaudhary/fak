# Qwen2.5-3B prompt-paraphrase robustness — 2026-08-15

**Verdict:** Qwen2.5-3B reproduced the cross-family pattern: concise contracts were stable at 19–20/24, while reasoning-scaffold phrasings ranged from 14/24 to 22/24. The exact solve/check/options scaffold is a strong pinned optimization, but “reasoning scaffold” is not a robust treatment-level default without prompt identity.

## Question

Is the Qwen2.5-3B scaffold advantage robust to reasonable prompt paraphrases, or does it depend on the exact wording that produced the size-ladder win?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Model: Qwen2.5-3B-Instruct, concatenated-shard SHA-256 `04a84318cdec5543e99a10e0db40df3f7843226f88a8c0e840da95bbfa52487c`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Protocol: the same 24 fixed four-choice tasks; 64-token output cap; first explicit answer-letter grading.
- Six arms: three concise-contract paraphrases and the same three scaffold paraphrases used for SmolLM3.
- Raw result: [`2026-08-15-qwen25-3b-prompt-paraphrases.json`](../benchmarks/model-sensitivity/2026-08-15-qwen25-3b-prompt-paraphrases.json), SHA-256 `633277f7d9a223838f0d31461df37e47e3b2f6b51ff0218aa7cbdebbd9845f63`.

## Results

| Treatment | Wording | Correct | Strict one-letter | Output tokens | Median latency |
|---|---|---:|---:|---:|---:|
| concise contract | minimal contract | 19/24 | 24/24 | 48 | 84.903 ms |
| concise contract | only-letter/no-explanation | 20/24 | 3/24 | 104 | 165.077 ms |
| concise contract | precise-solver role | 19/24 | 24/24 | 48 | 83.977 ms |
| reasoning scaffold | solve/check/options | 22/24 | 24/24 | 48 | 84.711 ms |
| reasoning scaffold | eliminate/verify | 14/24 | 5/24 | 747 | 185.257 ms |
| reasoning scaffold | private two-pass check | 16/24 | 23/24 | 110 | 85.931 ms |

## Cross-family interpretation

| Family/checkpoint | Contract range | Scaffold range | Best pinned arm |
|---|---:|---:|---|
| SmolLM3-3B | 19–19/24 | 5–20/24 | solve/check scaffold, 20/24 |
| Qwen2.5-3B | 19–20/24 | 14–22/24 | solve/check scaffold, 22/24 |

- Contract spread was 0 answers for SmolLM3 and 1 answer for Qwen.
- Scaffold spread was 15 answers for SmolLM3 and 8 answers for Qwen.
- In both families, the same solve/check/options wording was the best arm, perfectly concise on Qwen and nearly so on SmolLM3.
- In both families, adding seemingly stronger internal procedures—elimination or private two-pass checking—hurt correctness and often increased output volume.
- This is replicated evidence that treatment labels are too coarse. Prompt identity, native thinking mode, model identity, and decode budget belong in the routing key.

## Honest boundary

- Three paraphrases per treatment do not characterize all prompt variation.
- The Transformers checkpoint is not numerically identical to the earlier Q8 GGUF cell, although model family and task protocol match.
- Results are single deterministic runs over 24 items; no population-level significance is claimed.
- The best exact scaffold may still fail under other task distributions or formatting contracts.

## Decision

Use the concise contract as the robust fallback for Qwen2.5-3B. Use the exact solve/check/options scaffold when its full prompt identity is pinned and witnessed for this model. Do not route on a generic `scaffold` label or parameter count alone.
