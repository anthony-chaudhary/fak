# SmolLM3 option-order robustness — 2026-08-15

**Verdict:** SmolLM3-3B's canonical 1-answer scaffold lead disappears exactly across four cyclic option rotations: both concise contract and exact solve/check scaffold average 19/24. This independently replicates the Qwen finding that a fixed answer-label placement can manufacture apparent prompt-treatment separation.

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Model: SmolLM3-3B, concatenated-shard SHA-256 `ebf0ae1f748a2b1c484d2e18f73ba41c6133086bc3e6f22303307d56b1377c21`.
- Runtime: Transformers 5.15.0, PyTorch 2.13.0+cu130, greedy CUDA generation.
- Native control: `/no_think` plus `enable_thinking=False`.
- Protocol: 24 fixed four-choice tasks; 64-token cap; canonical contract and exact solve/check scaffold.
- Rotations: all four cyclic option placements with exact expected-letter remapping.
- Raw result: [`2026-08-15-smollm3-option-rotation.json`](2026-08-15-smollm3-option-rotation.json), SHA-256 `1d1292c16ee9f6f66b3d9c2746e806c2cf5beba4c31688c9fff49472a3a8c72f`.

## Results

| Prompt | r0 | r1 | r2 | r3 | Mean | Range |
|---|---:|---:|---:|---:|---:|---:|
| concise contract | 19 | 21 | 21 | 15 | 19.00 | 6 |
| solve/check scaffold | 20 | 18 | 22 | 16 | 19.00 | 6 |

Each cell contains 24 tasks; `r0` is canonical ordering.

## Cross-family result

- SmolLM3 canonical ordering suggested scaffold +1; rotation mean is an exact tie.
- Qwen2.5-3B canonical ordering suggested scaffold +3; rotation mean was scaffold +0.5.
- In both model families, rotating semantically unchanged options substantially reduced the apparent treatment separation.
- SmolLM3's scaffold remains operationally concise across rotations—72 to 105 output tokens versus contract's 950 to 1,153—but that efficiency does not translate into a mean correctness advantage.

## Honest boundary

- Four cyclic rotations do not cover all 24 permutations.
- These are deterministic single runs over one task set, not population estimates.
- Rotation changes token positions by design; it is a nuisance-variable robustness test, not a claim of bitwise invariance.
- Only the canonical contract and exact scaffold are covered.

## Decision

Treat contract and exact scaffold as correctness-tied for SmolLM3-3B under option-order control. Choose the scaffold only when its much shorter output is itself valuable and its exact prompt is pinned. Require answer-option rotation in future multiple-choice prompt-routing witnesses across model families.
