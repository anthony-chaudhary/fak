# Qwen2.5 exhaustive option-permutation robustness — 2026-08-16

**Verdict:** Exhausting all 24 answer-option permutations overturns the single-order interpretation at 1.5B and confirms a tie at 3B. The exact solve/check scaffold beats or ties the concise contract on every 1.5B permutation, averaging 17.79/24 versus 15.42/24. At 3B, the prompts are effectively identical across the complete permutation set: 19.92/24 versus 19.79/24, with a median paired difference of zero.

## Question

Do the canonical concise contract and exact solve/check scaffold retain their relative performance across every possible ordering of four answer choices, rather than only the four cyclic rotations?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Models: Qwen2.5-1.5B-Instruct, weights SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`; Qwen2.5-3B-Instruct, concatenated-shard SHA-256 `04a84318cdec5543e99a10e0db40df3f7843226f88a8c0e840da95bbfa52487c`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Protocol: 24 fixed four-choice tasks; 64-token cap; first explicit answer-letter grading.
- Treatments: canonical concise contract and exact solve/check/options scaffold.
- Nuisance control: all `4! = 24` option permutations, with the expected answer letter remapped exactly for every task.
- Scope: 576 task-permutation observations per treatment per checkpoint; 1,152 generations per checkpoint.
- Raw 1.5B result: [`2026-08-16-qwen25-15b-option-permutations.json`](2026-08-16-qwen25-15b-option-permutations.json), SHA-256 `5fc35513e13c99fae8ce76cb656a34cb765426c156f7a23c9f4166eb827d56e4`.
- Raw 3B result: [`2026-08-16-qwen25-3b-option-permutations.json`](2026-08-16-qwen25-3b-option-permutations.json), SHA-256 `67af60b42f9a13c99d065b59607787f07e0576fd469feb488d84db8d3ce67746`.

## Permutation-level results

| Model | Prompt | Mean correct | Median | Min–max | Total correct | Permutation wins–ties–losses vs other prompt |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | concise contract | 15.42/24 | 15.5 | 12–19 | 370/576 | 0–4–20 |
| Qwen2.5-1.5B | solve/check scaffold | 17.79/24 | 18.0 | 13–22 | 427/576 | 20–4–0 |
| Qwen2.5-3B | concise contract | 19.79/24 | 20.0 | 16–23 | 475/576 | 6–7–11 |
| Qwen2.5-3B | solve/check scaffold | 19.92/24 | 20.0 | 16–23 | 478/576 | 11–7–6 |

At 1.5B, the paired scaffold-minus-contract difference averages **+2.375 answers**, has median **+2**, and ranges from **0 to +6**. At 3B, it averages **+0.125**, has median **0**, and ranges from **-4 to +3**.

## Task-permutation paired outcomes

| Model | Both correct | Both wrong | Contract only | Scaffold only |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 356 | 135 | 14 | 71 |
| Qwen2.5-3B | 446 | 69 | 29 | 32 |

The 1.5B scaffold gains are not merely permutation-level score reshuffling: there are 71 scaffold-only correct observations versus 14 contract-only observations. At 3B, discordant outcomes are nearly balanced, 32 versus 29.

## Answer-letter sensitivity

Each expected letter occurs exactly 144 times per treatment and checkpoint.

| Model | Prompt | A | B | C | D |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | concise contract | 50.7% | 68.1% | 72.9% | 65.3% |
| Qwen2.5-1.5B | solve/check scaffold | 67.4% | 76.4% | 80.6% | 72.2% |
| Qwen2.5-3B | concise contract | 80.6% | 84.0% | 72.9% | 92.4% |
| Qwen2.5-3B | solve/check scaffold | 86.1% | 84.0% | 77.8% | 84.0% |

Exhaustive balancing prevents the original answer-letter distribution from favoring one prompt, but substantial residual label sensitivity remains inside each model. The 3B contract is especially uneven between C and D despite balanced exposure.

## Output behavior

- At 1.5B, contract used 9,877 output tokens and was strictly one-letter on 152/576 observations. Scaffold used 1,444 tokens and was strict on 479/576.
- At 3B, both treatments used exactly 1,152 output tokens and were strict on all 576 observations.
- Thus the 1.5B scaffold is better on correctness, formatting, and output volume for this exact prompt pair. At 3B, the treatments are tied on those operational outputs as well as aggregate correctness.

## Relationship to the four-rotation witness

The earlier cyclic-only study estimated 1.5B means of 15.50 contract and 18.25 scaffold, and 3B means of 20.25 and 20.75. Exhausting all permutations preserves the qualitative conclusions but tightens them:

- **1.5B:** exact scaffold advantage is robust and appears on 20 of 24 permutations, with no permutation loss.
- **3B:** the small cyclic mean difference contracts further, from +0.50 to +0.125, with balanced paired wins and equal medians/ranges.

## Honest boundary

- The 24 permutations are the complete finite ordering set for these tasks, not independent samples from a broader population. No population-level significance claim is made.
- The 576 observations reuse 24 semantic questions, so task-level and permutation-level counts are descriptive, not independent-trial confidence evidence.
- This witness covers one exact contract and one exact scaffold. The paraphrase study still shows that generic scaffold wording is unsafe.
- The Transformers checkpoints and runtime differ from the earlier pure-fak Q8 cells.
- Multiple-choice order robustness does not imply robustness on open-ended or tool-use tasks.

## Decision

For Qwen2.5-1.5B in this runtime, promote the **exact solve/check scaffold** over the canonical contract: it wins the complete option-permutation envelope and is substantially more concise. Do not promote generic scaffold paraphrases. For Qwen2.5-3B, treat the exact contract and scaffold as correctness-equivalent under exhaustive option ordering; use contract as the wording-robust fallback or scaffold as a pinned equivalent. Future multiple-choice routing witnesses should balance all option permutations when the task count permits it, rather than relying on a single order.
