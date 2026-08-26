# Qwen2.5 option-order robustness — 2026-08-15

**Verdict:** Deterministic option rotation materially changes absolute scores and narrows the apparent 3B scaffold advantage. Across four rotations, Qwen2.5-1.5B averaged 15.5/24 for contract versus 18.25/24 for the exact scaffold; Qwen2.5-3B averaged 20.25/24 versus 20.75/24. The canonical-order 3-answer scaffold lead at 3B becomes a 0.5-answer mean difference, so option order belongs in the evaluation witness.

## Question

Do the canonical concise contract and exact solve/check scaffold retain their ordering when answer choices—and therefore correct answer letters—are rotated without changing question semantics?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Models: Qwen2.5-1.5B-Instruct, SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`; Qwen2.5-3B-Instruct, concatenated-shard SHA-256 `04a84318cdec5543e99a10e0db40df3f7843226f88a8c0e840da95bbfa52487c`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Protocol: 24 fixed four-choice tasks; 64-token cap; canonical concise contract and exact solve/check scaffold.
- Rotations: left-rotate each task's four options by 0, 1, 2, and 3 positions and remap the expected answer letter exactly.
- Raw result: [`2026-08-15-qwen25-option-rotation.json`](../benchmarks/model-sensitivity/2026-08-15-qwen25-option-rotation.json), SHA-256 `ff5aacfc2dc7382dad6de2b000ac7f43fb5c2de3a54b7a325bb7d4a85dd975c7`.

## Results

| Model | Prompt | r0 | r1 | r2 | r3 | Mean | Range |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | concise contract | 19 | 12 | 16 | 15 | 15.50 | 7 |
| Qwen2.5-1.5B | solve/check scaffold | 19 | 16 | 20 | 18 | 18.25 | 4 |
| Qwen2.5-3B | concise contract | 19 | 21 | 21 | 20 | 20.25 | 2 |
| Qwen2.5-3B | solve/check scaffold | 22 | 20 | 21 | 20 | 20.75 | 2 |

Each cell contains 24 tasks. `r0` is the original option order.

## Interpretation

- At 1.5B, the prompts tie on canonical order, but the scaffold has a 2.75-answer higher four-rotation mean and a smaller range.
- At 3B, the canonical scaffold lead is 3 answers, but the four-rotation mean lead is only 0.5 answer. Contract wins one rotation, scaffold wins one, and two tie.
- Scaling from 1.5B to 3B improves the contract's rotation mean by 4.75 answers and reduces its range from 7 to 2. It improves the scaffold mean by 2.5 answers and reduces its range from 4 to 2.
- The robust scaling result is therefore broader at 3B: both exact prompts become much less sensitive to answer-label placement. The evidence does not support a strong treatment-level preference between them at 3B.
- A single canonical option order can exaggerate both absolute quality and treatment separation. Prompt-routing witnesses should rotate or randomize options with semantic remapping when the task format permits it.

## Honest boundary

- These rotations preserve option content but can change token positions and local prompt structure; that is the intended nuisance variable under test.
- Four cyclic rotations are exhaustive for cyclic placement but do not cover all 24 option permutations.
- Results are deterministic single runs on one 24-task set; means summarize these cells and are not population estimates.
- This test covers the exact canonical contract and solve/check scaffold, not all paraphrases.

## Decision

For Qwen2.5-3B, treat contract and exact scaffold as effectively tied under the four-rotation witness; prefer contract as the robust wording-level fallback and use the scaffold only for its pinned concise behavior. For Qwen2.5-1.5B in this runtime, the exact scaffold is more option-order robust than the canonical contract, but generic scaffold paraphrases remain unsafe. Require option-order controls in future multiple-choice prompt-routing claims.
