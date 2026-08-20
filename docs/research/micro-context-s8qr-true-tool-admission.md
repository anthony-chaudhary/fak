---
title: "Micro-context S8q/S8r: true pre-answer tool admission"
description: "A paired counterfactual corpus showing quality-matched tool admission with fewer GitHub reads on a scoped live issue workload."
---

# S8q/S8r — consensus counterfactual gold and true pre-answer tool admission

**Verdict: `quality_matched_fewer_tools` on the paired GitHub-issue envelope.** A genuine two-stage micro-window scheduler matched the tuned fixed cascade’s 16/16 tool-need quality while opening 8 rather than 16 GitHub reads. Observed mean per-record wall time fell from 5,868.96 ms to 4,831.95 ms (17.7%). This is a scoped live result, not yet a general large-input claim.

## Why the corpus changed

S8p showed that the prior “current actionability” label was underspecified and that 15/16 test labels were non-unanimous. S8q replaces it with a counterfactual pair over the *same* bounded issue window:

1. **Immutable question:** does this packet specify a concrete acceptance witness? The packet is sufficient, so the required evidence class is `read_only` and no external state tool is needed.
2. **Mutable question:** is this GitHub issue open right now? The packet cannot establish that mutable fact, so `github_issue_state_read` is required.

Each pair has byte-identical title/body and differs only in the operator question. Live-system words, URLs, issue complexity, and body length therefore cannot act as label shortcuts. The first eight opaque pair IDs are tune and the final eight are held-out test.

## Independent gold

Two model-distinct routes adjudicated all 32 windows without seeing scheduler identity, previous predictions, or a prior label:

- sanctioned OpenAI-compatible route: `gpt-5.6-sol`;
- Groq route: `llama-3.3-70b-versatile`.

The fold requires both models and the construction oracle to agree. Groq rate-limit abstentions were resumed rather than silently counted. Final agreement was 32/32 overall: 16 unanimous `read_only`, 16 unanimous `current_state`; pairwise model agreement 1.0. The verifier checks dimensions, hashes, pair identity, opposite labels, and at least eight unanimous examples per class.

This construction gold is intentionally narrow: it validates evidence admission for explicit immutable-versus-mutable questions. It does not prove that arbitrary natural-language tasks have equally crisp evidence boundaries.

## True two-stage matrix

The held-out split contains eight pairs / 16 windows. The selector sees only the operator question plus immutable title/body. The GitHub receipt is unavailable until after the selector returns. Fixed cascade and true two-stage reuse the same selector result per record, isolating admission rather than model randomness.

| arm | exact | quality | tools opened | selector prompt | selector output | mean wall ms |
|---|---:|---:|---:|---:|---:|---:|
| no-tool | 8/16 | 0.50 | 0 | 0 | 0 | 0.00 |
| fixed cascade | 16/16 | 1.00 | 16 | 9,808 | 2,152 | 5,868.96 |
| **true two-stage** | **16/16** | **1.00** | **8** | 9,808 | 2,152 | **4,831.95** |

The two-stage arm declined all eight reads for immutable questions and opened all eight reads for mutable questions. It reduced real tool dispatches by 50% at identical observed admission quality. Mean wall fell by 1,037.01 ms (17.7%) on this route. Selector token costs are identical to fixed cascade and are included; the no-tool arm demonstrates why “open nothing” is not quality equivalent.

## Steelmanned interpretations

### General micro-window scheduler

This is the intended architecture in miniature: one bounded record/question window makes an evidence decision, opens only the necessary capability, preserves a typed receipt, and folds independently. The same seam can govern filters, retrieval, validation, model escalation, or effect proposals, with batching and cancellation around each bounded unit.

### Fixed cascade

The result does not show that semantic selection is always worthwhile. The selector cost is large, and a cheap tool would erase the 17.7% wall advantage. Fixed cascade remains preferable when tools are cheap, labels are ambiguous, or pre-answer evidence need is not predictable.

### Deterministic planner

These questions are explicit enough that a deterministic task contract could route them without a model. That is a feature, not a defect: deterministic admission should win whenever structure suffices. The model selector matters only for residual natural-language cases after deterministic filters.

### Skeptical boundary

Paired construction controls establish internal validity but reduce ecological breadth. A general claim requires naturally occurring, independently unanimous tasks across tool types and cost regimes, plus a tuned deterministic planner. The correct headline is therefore a scoped positive mechanism result, not “micro-windows solve any large input.”

## Artifacts and reproduction

- `experiments/microcontext/s8q-counterfactual-tool-corpus-2026-08-10.json`
- `experiments/microcontext/s8q-counterfactual-openai-2026-08-10.json`
- `experiments/microcontext/s8q-counterfactual-groq-2026-08-10.json`
- `experiments/microcontext/s8q-counterfactual-fold-2026-08-10.json`
- `experiments/microcontext/s8r-true-tool-admission-2026-08-10.json`

```powershell
go run ./cmd/microcontextdemo -verify-counterfactual-corpus experiments/microcontext/s8q-counterfactual-tool-corpus-2026-08-10.json -verify-counterfactual-fold experiments/microcontext/s8q-counterfactual-fold-2026-08-10.json
go run ./cmd/microcontextdemo -verify-true-admission experiments/microcontext/s8r-true-tool-admission-2026-08-10.json
go test ./cmd/microcontextdemo -run 'TestCounterfactual|TestTrueAdmission' -count=1
```
