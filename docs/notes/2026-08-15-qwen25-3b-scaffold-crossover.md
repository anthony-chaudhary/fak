# Qwen2.5 prompt-order crossover at 3B — pure-fak CPU Q8 — 2026-08-15

**Verdict:** The reasoning scaffold first overtook the concise answer contract at the matched 3B checkpoint, scoring 18/24 versus 15/24. This places the observed Qwen2.5-Instruct Q8 prompt-order crossover between 1.5B and 3B on this fixed 24-item protocol; it does not establish a universal size threshold.

## Question

Where does the prompt-arm ordering change between the 1.5B checkpoint, where the concise contract won, and the 14B checkpoint, where the reasoning scaffold won?

## Pinned envelope

- Node class: sanctioned CPU lab host.
- Runtime: pure-fak in-kernel GGUF serve through the OpenAI-compatible chat endpoint.
- Protocol: the same 24 fixed four-choice tasks used by the 0.5B, 1.5B, and 14B cells; 6 arithmetic, 6 logic, 6 programming, and 6 state-transition items.
- Decoding: `temperature=0`, `max_tokens=24`.
- Semantic grading: first explicit answer-letter phrase. Strict compliance: entire response is one answer letter, with optional punctuation.
- Artifact: Qwen2.5-3B-Instruct Q8_0 GGUF, SHA-256 `6dcc22694c8654b045ec40bbe350212b88893fd9010e8474bae5b19a43578ba1`.
- Raw result: [`2026-08-15-qwen25-3b-q8-scaffold.json`](../benchmarks/model-sensitivity/2026-08-15-qwen25-3b-q8-scaffold.json), SHA-256 `11e9fa922fc2d565bc0af160b99248a68646816c08d634ca94fdd7259bd1f77b`.
- Isolation: a temporary serve container bound only to loopback port 8083; removed after capture.

## Results

| Model | Arm | Correct | Accuracy | Strict one-letter | Prompt tokens | Completion tokens | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B Q8 | direct | 5/24 | 20.8% | 0/24 | 990 | 571 | 14,923.555 ms |
| Qwen2.5-3B Q8 | concise contract | 15/24 | 62.5% | 24/24 | 1,518 | 24 | 12,362.947 ms |
| Qwen2.5-3B Q8 | reasoning scaffold | 18/24 | 75.0% | 24/24 | 1,854 | 24 | 12,780.899 ms |

## Observed interaction

- Reasoning scaffold versus concise contract was **+12.5 percentage points** at 3B, reversing the **-16.7-point** difference at 1.5B.
- On the matched Qwen2.5-Instruct Q8 ladder, the observed ordering therefore changes between **1.5B and 3B**.
- The scaffold improved over direct by **54.2 points** and used 547 fewer completion tokens; the contract improved over direct by **41.7 points** and used the same 24 completion tokens as the scaffold.
- Both instructed arms were perfectly format-compliant, yet differed by 3 correct answers. Format compliance again did not determine correctness.
- Direct performance was non-monotonic across these single checkpoint runs: 13/24 at 1.5B but 5/24 at 3B. The evidence is about arm ordering within each pinned checkpoint, not a claim that parameter count alone monotonically improves this task set.

These are descriptive differences on 24 items per cell. There are no repeated trials or confidence intervals, so no population-level claim is made. CPU latency is operational context, not a cross-size performance comparison.

## Updated boundary

- **0.5B and 1.5B matched Qwen2.5-Instruct Q8 cells:** concise contract wins.
- **3B and 14B tested Qwen2.5 instruction cells:** reasoning scaffold wins.
- **27B pure-fak CUDA:** incomplete because the GDN path panicked; tracked by #6906.

This narrows the empirical crossover interval but does not prove that every model inside or outside it follows the same ordering. Repeated seeds, alternate task sets, and other model families remain outside this witness.

## Decision

Keep the concise answer contract as the measured default candidate through the tested 1.5B tier. Starting at the tested 3B checkpoint, require a model-specific ablation instead of inheriting the small-model default; for this 3B cell, choose the reasoning scaffold.
