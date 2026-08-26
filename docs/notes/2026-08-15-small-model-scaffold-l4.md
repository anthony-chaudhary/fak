# Small-model prompt-scaffold ablation — NVIDIA L4 — 2026-08-15

**Verdict:** A concise answer contract improved accuracy on both pinned local models; the additional "solve carefully" scaffold gave back part of that gain. The treatment is model-sensitive, so this is an extension witness—not native-comparator evidence and not a general quality claim.

## Question

For small local models, does a minimal output contract or a heavier reasoning scaffold improve a fixed multiple-choice task slice relative to a direct prompt?

This is a bounded first cell for #6692. It holds tasks and decoding fixed across arms and varies only prompt treatment. It does **not** satisfy the full cross-model issue: strong-frontier and cost-oriented provider cells, provider token/cost fields, and native comparator reports remain outstanding.

## Pinned envelope

- Node: sanctioned GCP `fak-cuda-build-l4`; NVIDIA L4 23,034 MiB; driver 580.159.03.
- Runtime: Python 3.10; PyTorch 2.13.0+cu130; Transformers 5.15.0; greedy decoding; `max_new_tokens=24`.
- Tasks: 24 fixed four-choice questions: 6 arithmetic, 6 logic, 6 programming, and 6 state-transition items. Exact task definitions and per-item outputs are in the JSON artifact.
- SmolLM2-135M local checkpoint: `LlamaForCausalLM`, hidden size 576, weights SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
- Qwen2-0.5B local checkpoint: `Qwen2ForCausalLM`, hidden size 896, weights SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Artifact: [`2026-08-15-small-model-scaffold-l4.json`](../benchmarks/model-sensitivity/2026-08-15-small-model-scaffold-l4.json), SHA-256 `b9d4114ab93c2063b480461ecfb90a9aa28df4248fbadb575aa40f6d45a51269`.

## Arms and observed results

| Model | Arm | Correct | Accuracy | Strict one-letter outputs | Output tokens | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | direct | 9/24 | 37.5% | 0/24 | 288 | 159.069 ms |
| SmolLM2-135M | concise contract | 12/24 | 50.0% | 3/24 | 105 | 128.181 ms |
| SmolLM2-135M | reasoning scaffold | 10/24 | 41.7% | 2/24 | 108 | 130.726 ms |
| Qwen2-0.5B | direct | 6/24 | 25.0% | 0/24 | 549 | 623.390 ms |
| Qwen2-0.5B | concise contract | 14/24 | 58.3% | 0/24 | 492 | 605.649 ms |
| Qwen2-0.5B | reasoning scaffold | 12/24 | 50.0% | 0/24 | 404 | 497.596 ms |

The concise contract's observed accuracy delta versus direct was **+12.5 percentage points** on SmolLM2-135M and **+33.3 points** on Qwen2-0.5B. Adding the reasoning scaffold reduced accuracy versus the concise contract by **8.3 points** on both models. This is descriptive on 24 items per cell; no confidence or population-level claim is made.

The strict-format metric is intentionally separate from semantic answer extraction. Qwen often emitted text such as `The correct answer is C. 15.`: semantically gradeable, but not compliant with the requested one-letter contract. Therefore this witness does not claim reliable structured output.

## Reproduction

The artifact is self-contained for review: it records the exact task definitions, arm prompt templates, checkpoint config, hashes, environment, aggregate metrics, and every decoded response. To rerun, load each pinned local directory with Transformers, apply its tokenizer chat template when available, greedily generate 24 tokens for every task×arm cell, and grade the first explicit answer-letter phrase; separately require a full-response one-letter regex for strict compliance.

The execution used a throwaway remote Python runner because this was a diagnostic experiment. The durable deliverable is the protocol and captured result, not a new repository Python tool. A repeated benchmark surface should be promoted as a Go `fak` verb rather than preserving that scratch runner.

## Interpretation and next cells

- **Observed:** prompt scaffolding is not monotonic; more instruction was worse than the concise contract on both small models.
- **Observed:** the treatment interaction magnitude differs by model; Qwen2-0.5B benefited much more from the concise contract on this slice.
- **Not yet:** native Caveman/Ponytail comparator parity, frontier/cost-provider models, calibrated semantic judges, confidence intervals, tokens-in, monetary cost, and repeated trials.
- **Decision:** use the concise contract—not the heavier scaffold—as the small-local-model default candidate in the next larger ablation. Keep direct and reasoning-scaffold arms as controls.

