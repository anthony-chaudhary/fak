# SmolLM prompt-scaffold size and token-budget ablation — 2026-08-15

**Verdict:** The concise answer contract remained best at 360M and 1.7B, while the reasoning scaffold led by only 1/24 at 3B. Unlike the matched Qwen ladder, this second family did not show a robust size-only crossover: the 3B checkpoint required its documented no-thinking control, and its contract/direct arms often exhausted the decode cap whereas the scaffold terminated quickly. Prompt routing therefore depends on model family and generation behavior, not parameter count alone.

## Question

Does Qwen2.5's observed contract-to-scaffold reversal between 1.5B and 3B replicate in another small local model family, and is any apparent reversal stable to output-token budget?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Protocol: the same 24 fixed four-choice tasks used by the Qwen ladder; 6 arithmetic, 6 logic, 6 programming, and 6 state-transition items.
- Arms: direct, concise one-letter contract, and reasoning scaffold with hidden-reasoning instruction.
- Primary decode cap: 24 new tokens. SmolLM3 also ran at 64 tokens because two arms saturated the primary cap.
- Scoring: first explicit answer-letter phrase. Strict compliance: the entire response is one answer letter, optionally punctuated.
- SmolLM2-360M-Instruct weights SHA-256: `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
- SmolLM2-1.7B-Instruct weights SHA-256: `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`.
- SmolLM3-3B concatenated-shard SHA-256: `ebf0ae1f748a2b1c484d2e18f73ba41c6133086bc3e6f22303307d56b1377c21`; component hashes are preserved in the raw model repository provenance.
- SmolLM3 used its documented `/no_think` system control and `enable_thinking=False`. Without that control, all arms spent the entire 24-token budget inside `<think>` and yielded 0/24, which is a protocol mismatch rather than a meaningful quality cell.

## Results

| Model | Cap | Arm | Correct | Accuracy | Strict one-letter | Output tokens | Median latency |
|---|---:|---|---:|---:|---:|---:|---:|
| SmolLM2-360M-Instruct | 24 | direct | 5/24 | 20.8% | 0/24 | 113 | 175.232 ms |
| SmolLM2-360M-Instruct | 24 | concise contract | 13/24 | 54.2% | 1/24 | 111 | 159.042 ms |
| SmolLM2-360M-Instruct | 24 | reasoning scaffold | 6/24 | 25.0% | 1/24 | 115 | 175.863 ms |
| SmolLM2-1.7B-Instruct | 24 | direct | 19/24 | 79.2% | 0/24 | 190 | 133.391 ms |
| SmolLM2-1.7B-Instruct | 24 | concise contract | 21/24 | 87.5% | 0/24 | 114 | 133.055 ms |
| SmolLM2-1.7B-Instruct | 24 | reasoning scaffold | 19/24 | 79.2% | 0/24 | 114 | 133.072 ms |
| SmolLM3-3B, no-think | 24 | direct | 7/24 | 29.2% | 0/24 | 573 | 917.957 ms |
| SmolLM3-3B, no-think | 24 | concise contract | 19/24 | 79.2% | 0/24 | 556 | 925.835 ms |
| SmolLM3-3B, no-think | 24 | reasoning scaffold | 20/24 | 83.3% | 7/24 | 91 | 159.705 ms |
| SmolLM3-3B, no-think | 64 | direct | 13/24 | 54.2% | 0/24 | 1,303 | 2,437.401 ms |
| SmolLM3-3B, no-think | 64 | concise contract | 19/24 | 79.2% | 0/24 | 1,132 | 2,105.696 ms |
| SmolLM3-3B, no-think | 64 | reasoning scaffold | 20/24 | 83.3% | 7/24 | 91 | 160.529 ms |

Raw artifacts:

- [`2026-08-15-smollm2-instruct-size-scaffold.json`](../benchmarks/model-sensitivity/2026-08-15-smollm2-instruct-size-scaffold.json), SHA-256 `2f489e1f158953c2dfccc4bd5a61a685bd89dbd06c8a445825a1438eee6221ab`.
- [`2026-08-15-smollm3-3b-scaffold-24.json`](../benchmarks/model-sensitivity/2026-08-15-smollm3-3b-scaffold-24.json), SHA-256 `1bc19e63dd8cbd24bf42589fdad147c27d4817cf183949806694881dc33dfc40`.
- [`2026-08-15-smollm3-3b-scaffold-64.json`](../benchmarks/model-sensitivity/2026-08-15-smollm3-3b-scaffold-64.json), SHA-256 `124483363b98caefbef0b08a1a04d83e44ff135404bb765292e52d693b8d0f12`.

## Observed interactions

- At 360M, the contract beat the scaffold by **7/24**; at 1.7B it beat the scaffold by **2/24**.
- At 3B, the scaffold beat the contract by **1/24**, and this aggregate difference was unchanged when the cap increased from 24 to 64 tokens.
- The 3B scaffold used only 91 output tokens and gained strict one-letter compliance on 7 tasks. Contract used 556 tokens at cap 24 and 1,132 at cap 64 without changing its 19/24 score.
- Raising the cap improved 3B direct from 7/24 to 13/24, but did not change contract or scaffold correctness. This isolates a direct-arm truncation effect while leaving the narrow instructed-arm ordering unchanged.
- The Qwen2.5 3B scaffold led its contract by 3/24 under the matched pure-fak Q8 study; SmolLM3's lead was only 1/24 and came with sharply different termination behavior. Parameter count alone is therefore not an adequate prompt router.

## Honest boundary

- SmolLM2 and SmolLM3 are adjacent generations, not an architecture-held-constant ladder. The 3B cell also requires a model-specific thinking-mode control.
- These are single deterministic runs over 24 items per cell. A one-answer difference is descriptive, not a population-level win.
- Strict one-letter compliance is low for most SmolLM cells; semantic grading follows the predeclared first-answer-phrase rule.
- Latency is observed single-host wall time and is not a portable throughput or cost claim.
- No claim is made that the 64-token cap is globally optimal; it is a bounded truncation sensitivity check.

## Decision

Use the concise contract as the measured default candidate for the tested SmolLM2 checkpoints through 1.7B. At SmolLM3-3B, retain the scaffold as a model-specific candidate because it is one answer better and terminates much more efficiently, but do not generalize a 3B crossover threshold across families. Route from model-specific witnesses that include native thinking-mode controls and token-budget sensitivity.
