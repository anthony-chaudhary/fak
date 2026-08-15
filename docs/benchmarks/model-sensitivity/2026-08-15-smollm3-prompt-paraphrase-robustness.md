# SmolLM3 prompt-paraphrase robustness — 2026-08-15

**Verdict:** The concise-contract treatment was stable across three phrasings at 19/24 each. Reasoning-scaffold phrasings ranged from 5/24 to 20/24 and from 91 to 1,536 output tokens. The previously observed 1/24 scaffold lead is therefore a wording-specific result, not evidence that the scaffold treatment class robustly wins at 3B.

## Question

Does SmolLM3-3B's narrow scaffold advantage survive reasonable paraphrases of both the concise contract and the reasoning scaffold?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Model: SmolLM3-3B, concatenated-shard SHA-256 `ebf0ae1f748a2b1c484d2e18f73ba41c6133086bc3e6f22303307d56b1377c21`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Native control: `/no_think` system instruction plus `enable_thinking=False`.
- Protocol: the same 24 fixed four-choice tasks; 64-token output cap; first explicit answer-letter grading.
- Six arms: three concise-contract paraphrases and three scaffold paraphrases.
- Raw result: [`2026-08-15-smollm3-3b-prompt-paraphrases.json`](2026-08-15-smollm3-3b-prompt-paraphrases.json), SHA-256 `a21205d0559872ff62486aa389ca70a4db07b67c50b280dcbfffdebd0cc5e8b6`.

## Results

| Treatment | Wording | Correct | Strict one-letter | Output tokens | Median latency |
|---|---|---:|---:|---:|---:|
| concise contract | minimal contract | 19/24 | 0/24 | 1,132 | 2,112.771 ms |
| concise contract | only-letter/no-explanation | 19/24 | 0/24 | 965 | 1,359.312 ms |
| concise contract | precise-solver role | 19/24 | 0/24 | 1,045 | 2,042.346 ms |
| reasoning scaffold | solve/check/options | 20/24 | 7/24 | 91 | 161.910 ms |
| reasoning scaffold | eliminate/verify | 5/24 | 0/24 | 1,425 | 2,481.274 ms |
| reasoning scaffold | private two-pass check | 10/24 | 0/24 | 1,536 | 2,481.822 ms |

## Interpretation

- Contract correctness had **zero observed spread**: all three variants scored 19/24.
- Scaffold correctness spanned **15 answers** and its output volume spanned **1,445 tokens**.
- Only the exact solve/check/options wording produced the earlier 20/24 result and short outputs. The two seemingly stronger internal-reasoning phrasings were substantially worse and nearly saturated the 64-token cap on every task.
- The robust decision is therefore not “3B models prefer scaffolding.” It is that this checkpoint has a high-performing prompt-shaped fast path whose behavior is brittle to wording.
- For an unattended default, the contract has lower peak score by 1/24 but far lower wording sensitivity in this tested set. The exact scaffold remains useful as a pinned model-specific optimization when its prompt hash is controlled.

## Honest boundary

- Three phrasings per treatment do not exhaust prompt space.
- Results are one deterministic run per cell over 24 tasks; no population-level significance is claimed.
- Contract outputs remained verbose despite explicit one-letter instructions, so its 19/24 semantic score does not establish formatting compliance.
- The comparison does not separate every lexical change; each arm is an operational prompt bundle.

## Decision

Prefer the concise contract as the robust SmolLM3-3B default candidate. Permit the exact solve/check/options scaffold as a pinned optimization only with a model-and-prompt-specific witness; do not generalize it to paraphrased scaffolds or to a parameter-size tier.
