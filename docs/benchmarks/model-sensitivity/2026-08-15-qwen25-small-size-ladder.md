# Qwen2.5 small-model size ladder — pure-fak CPU Q8 — 2026-08-15

**Verdict:** Across matched Qwen2.5-Instruct Q8 checkpoints at 0.5B and 1.5B, the concise answer contract was the best arm. Scaling to 1.5B improved all three arms, but the heavier reasoning scaffold still scored below the concise contract. This strengthens the small-model anti-over-scaffolding result under a matched architecture, quantization, runtime, and task protocol.

## Question

Does the prompt-arm ordering observed on earlier small checkpoints persist when model size changes while family, quantization, pure-fak runtime, tasks, and decoding remain fixed?

## Pinned envelope

- Node class: sanctioned CPU lab host.
- Runtime: pure-fak in-kernel GGUF serve through the OpenAI-compatible chat endpoint.
- Protocol: 24 fixed four-choice tasks; 6 arithmetic, 6 logic, 6 programming, and 6 state-transition items.
- Decoding: `temperature=0`, `max_tokens=24`.
- Semantic grading: first explicit answer-letter phrase. Strict compliance: entire response is one answer letter, with optional punctuation.
- 0.5B artifact: Qwen2.5-0.5B-Instruct Q8_0 GGUF, SHA-256 `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- 1.5B artifact: Qwen2.5-1.5B-Instruct Q8_0 GGUF, SHA-256 `d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8`.
- Raw 0.5B result: [`2026-08-15-qwen25-05b-q8-scaffold.json`](2026-08-15-qwen25-05b-q8-scaffold.json), SHA-256 `8e193fe6c6de1436fb1c8f9015665d51a31da69d44a69fc54c16423da2ec776d`.
- Raw 1.5B result: [`2026-08-15-qwen25-15b-q8-scaffold.json`](2026-08-15-qwen25-15b-q8-scaffold.json), SHA-256 `0b3362a737970ddae067821fc146c4f777085ae73d48a183916e2714f6294432`.

## Results

| Model | Arm | Correct | Accuracy | Strict one-letter | Prompt tokens | Completion tokens | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B Q8 | direct | 3/24 | 12.5% | 0/24 | 990 | 450 | 2,101.800 ms |
| Qwen2.5-0.5B Q8 | concise contract | 16/24 | 66.7% | 22/24 | 1,518 | 29 | 1,813.639 ms |
| Qwen2.5-0.5B Q8 | reasoning scaffold | 8/24 | 33.3% | 24/24 | 1,854 | 24 | 1,836.902 ms |
| Qwen2.5-1.5B Q8 | direct | 13/24 | 54.2% | 3/24 | 990 | 241 | 6,560.908 ms |
| Qwen2.5-1.5B Q8 | concise contract | 20/24 | 83.3% | 24/24 | 1,518 | 24 | 5,965.448 ms |
| Qwen2.5-1.5B Q8 | reasoning scaffold | 16/24 | 66.7% | 23/24 | 1,854 | 24 | 6,190.898 ms |

## Observed interactions

- Scaling 0.5B → 1.5B improved direct by **41.7 percentage points**, concise contract by **16.7 points**, and reasoning scaffold by **33.3 points**.
- Concise contract versus direct improved accuracy by **54.2 points** at 0.5B and **29.2 points** at 1.5B.
- Reasoning scaffold versus concise contract was **-33.3 points** at 0.5B and **-16.7 points** at 1.5B.
- The larger checkpoint narrowed—but did not reverse—the small-model over-scaffolding penalty.
- At 0.5B, the reasoning scaffold achieved perfect strict formatting while only half of those answers were correct. Format compliance is not a correctness proxy.
- The concise contract reduced completion output by 421 tokens at 0.5B and 217 tokens at 1.5B versus direct, while improving correctness.

These are descriptive differences on 24 items per cell. There are no repeated trials or confidence intervals, so no population-level claim is made. CPU latency is reported for operational context but is not a model-quality comparison: the 1.5B checkpoint performs more work per request.

## Relationship to earlier cells

The earlier 135M and base 0.5B Transformer cells also favored the concise contract over the reasoning scaffold. The local 14B instruction model reversed that ordering and reached 24/24 under the scaffold. Together, the observed boundary is:

- **135M through 1.5B tested local models:** concise contract wins.
- **14B tested local instruction model:** reasoning scaffold wins by 2/24.
- **27B pure-fak CUDA:** incomplete because the GDN path panicked; tracked by #6906.

The evidence supports routing prompt scaffolds by measured model behavior rather than applying one global prompt treatment.

## Honest boundary

- No monetary cost is reported because these are locally served checkpoints.
- Missing machine-occupancy cost is not imputed.
- No native Caveman/Ponytail comparator claim is made.
- No provider-frontier claim is made.
- The result does not establish where between 1.5B and 14B the arm ordering changes.

## Decision

Use the concise answer contract as the default candidate for the tested ≤1.5B local-model tier. Retain the reasoning scaffold as an explicit ablation arm, not a default, until a model-specific witness shows it wins.
