# SmolLM3 exhaustive option-permutation robustness — 2026-08-16

**Verdict:** Across all 24 answer-option permutations, SmolLM3's concise contract and exact solve/check scaffold are correctness-equivalent: 19.33/24 versus 19.00/24, with paired permutation wins of 12 versus 11 and one tie. The scaffold remains operationally distinct—it uses 91.5% fewer output tokens and is much more format-compliant—but its canonical-order correctness lead does not survive the complete ordering envelope.

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Model: SmolLM3-3B, concatenated-shard SHA-256 `ebf0ae1f748a2b1c484d2e18f73ba41c6133086bc3e6f22303307d56b1377c21`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Native control: `/no_think` plus `enable_thinking=False`.
- Protocol: 24 fixed four-choice tasks; 64-token cap; first explicit answer-letter grading.
- Treatments: canonical concise contract and exact solve/check/options scaffold.
- Nuisance control: all 24 permutations of four choices, with exact expected-letter remapping.
- Scope: 576 task-permutation observations per treatment; 1,152 generations total.
- Raw result: [`2026-08-16-smollm3-option-permutations.json`](2026-08-16-smollm3-option-permutations.json), SHA-256 `b8d875272da9adfad3722670e12351cc6ee3858fa77a3da9e9d898aeaa3066bf`.

## Permutation-level results

| Prompt | Mean correct | Median | Min–max | Total correct | Wins–ties–losses vs other prompt |
|---|---:|---:|---:|---:|---:|
| concise contract | 19.33/24 | 19.5 | 15–23 | 464/576 | 12–1–11 |
| solve/check scaffold | 19.00/24 | 19.0 | 14–24 | 456/576 | 11–1–12 |

The paired scaffold-minus-contract difference averages **-0.333 answers**, has median **-0.5**, and ranges from **-4 to +3**. This is a practical correctness tie over the complete finite ordering set.

## Task-permutation paired outcomes

| Both correct | Both wrong | Contract only | Scaffold only |
|---:|---:|---:|---:|
| 431 | 87 | 33 | 25 |

Discordant observations slightly favor contract, 33 to 25, but most observations—518/576—have the same correctness outcome under both prompts.

## Answer-letter sensitivity

Each expected letter occurs exactly 144 times per treatment.

| Prompt | A | B | C | D |
|---|---:|---:|---:|---:|
| concise contract | 98.6% | 76.4% | 79.2% | 68.1% |
| solve/check scaffold | 89.6% | 72.2% | 89.6% | 65.3% |

Balancing letter exposure removes distributional advantage but reveals substantial residual label sensitivity in both prompts, especially on D.

## Output behavior

- Contract used **25,049 output tokens** and achieved strict one-letter format on **0/576** observations.
- Scaffold used **2,138 output tokens** and achieved strict format on **258/576**.
- Scaffold therefore reduced output tokens by **22,911**, or **91.5%**, while giving up 8 correct observations across the full 576-observation envelope.

The output-efficiency difference is descriptive and large, but no monetary or occupancy-adjusted value is claimed.

## Cross-family complete-envelope result

| Checkpoint | Contract mean | Scaffold mean | Scaffold minus contract | Paired scaffold wins–ties–losses |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 15.42 | 17.79 | +2.375 | 20–4–0 |
| Qwen2.5-3B | 19.79 | 19.92 | +0.125 | 11–7–6 |
| SmolLM3-3B | 19.33 | 19.00 | -0.333 | 11–1–12 |

Only Qwen2.5-1.5B shows a robust correctness separation for this exact prompt pair. Both 3B checkpoints are effectively tied after exhaustive ordering control, despite canonical-order scaffold leads.

## Honest boundary

- The 24 permutations are the complete ordering set for each question, not independent semantic tasks; no population-level significance is claimed.
- Results cover one exact contract and one exact scaffold. Earlier paraphrase evidence still shows large scaffold wording sensitivity.
- SmolLM3's native no-thinking control is model-specific and required for this comparison.
- Multiple-choice order robustness does not establish open-ended or tool-use robustness.

## Decision

Treat contract and exact scaffold as correctness-equivalent for SmolLM3-3B. Select the exact scaffold when output volume and format compliance matter, but pin its full prompt identity and do not describe it as a generic reasoning-scaffold win. Across model families, require complete option-order balancing before promoting a multiple-choice prompt treatment.
