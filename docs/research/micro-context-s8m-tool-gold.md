---
title: "Micro-context S8m: tool-routing gold stabilization"
description: "A three-adjudicator fold that sharpens read-only versus live-state tool need while preserving disagreement and confidence limits."
---

# S8m: three-adjudicator tool-routing gold stabilization

Date: 2026-08-10  
Issue: #6140  
Parent: #6033

## Verdict

The sharpened rubric materially stabilizes the **read-only versus current-live-state** distinction,
but not enough to treat confidence as calibrated production truth. Two model-distinct v2 adjudicators
agree on `tool_need` for 22/32 records (68.75%), versus 7/32 (21.875%) for the original independent
OpenAI/Qwen pair. A predeclared strict 2-of-3 fold resolves all 32 records into 21 read-only and 11
current-state labels. Only 8/32 votes are unanimous, despite every mean confidence landing in the
0.75-1.0 bin. The labels are sufficient for the next experimental tuning/grade slice, with explicit
provenance; they are not production routing truth or calibrated confidence.

## Rubric change

`semantic-tool-need-v2` asks two questions independently:

1. `answer_evidence`: freshest evidence required to answer what the issue says or requests;
2. `action_evidence`: freshest evidence required to decide whether the issue is actionable now.

Each is `packet | repository | live`; `tool_need` is their maximum under
`packet < repository < live`, mapped to `none | read_only | current_state`.

Counterexamples fixed in the prompt:

- A URL, command, symbol, or historical outage does **not** by itself require current live state.
- An implementation request usually needs repository evidence to decide whether work remains.
- Mutable issue/deployment/service/API state is `live` only when that state can change the requested
  answer or present actionability decision.

This addresses the old ambiguity between “a tool could provide useful evidence” and “the answer or
current actionability actually requires that evidence.”

## Independent adjudicators and frozen policy

The same packet (`sha256:2889531f4cf83df09f9335a239af76a2539ffc116b70c36ac8af6438c4f274e6`),
including its original tune/test split, was sent blind to:

- `independent-openai-v2` / `gpt-5.6-sol` on the sanctioned OpenAI-compatible route;
- `independent-groq-llama` / `llama-3.3-70b-versatile` on Groq's separately hosted route.

The fold policy is encoded before folding: one original legacy vote plus the two model-distinct v2
votes; strict 2-of-3 majority, otherwise explicit abstention. The second original adjudicator is used
only to reproduce the old 21.875% agreement baseline. Input artifact hashes are embedded in the fold.

## Observed agreement

| Pair | Exact `tool_need` agreement |
|---|---:|
| Original OpenAI versus original Qwen | 7/32 = 21.875% |
| Legacy Qwen versus OpenAI v2 | 12/32 = 37.5% |
| Legacy Qwen versus Groq Llama v2 | 14/32 = 43.75% |
| OpenAI v2 versus Groq Llama v2 | 22/32 = 68.75% |

Majority classes:

| Class | Majority | Unanimous |
|---|---:|---:|
| `read_only` | 21 | 7 |
| `current_state` | 11 | 1 |
| `none` | 0 | 0 |
| `abstain` | 0 | — |

Confidence is visibly overconfident: all 32 mean-confidence values fall in 0.75-1.0, while only 25%
of records are unanimous. Confidence must therefore remain descriptive telemetry, never fold authority.

## Artifacts

- `experiments/microcontext/s8m-adjudicator-openai-2026-08-10.json`
- `experiments/microcontext/s8m-adjudicator-groq-2026-08-10.json`
- `experiments/microcontext/s8m-semantic-tool-fold-2026-08-10.json`

The fold records packet hash, source hashes, per-record votes, majority label, confidence, unanimity,
per-class counts, pairwise agreement, old/new change, and a deterministic gold digest.

## Exact rerun

```powershell
go run ./cmd/microcontextdemo `
  -semantic-triple-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -semantic-triple-old-a experiments/microcontext/s8i-adjudicator-a-2026-08-10.json `
  -semantic-triple-old-b experiments/microcontext/s8i-adjudicator-openai-2026-08-10.json `
  -semantic-triple-v2-a experiments/microcontext/s8m-adjudicator-openai-2026-08-10.json `
  -semantic-triple-v2-b experiments/microcontext/s8m-adjudicator-groq-2026-08-10.json `
  -semantic-triple-output experiments/microcontext/s8m-semantic-tool-fold-2026-08-10.json
```

The endpoint adjudication commands use `-semantic-prompt-version semantic-tool-need-v2`; API keys are
supplied through environment variables and are not recorded.

## Steelman and boundary

- A human expert panel could create more semantically defensible labels, but would be slower, harder to
  blind, and still require an explicit actionability rubric. It remains the stronger production-gold
  route.
- Majority vote can turn correlated model bias into false certainty. Distinct hosting and model families
  reduce but do not eliminate that risk; the low unanimity rate exposes it.
- A deterministic planner should supersede model labels whenever repository/live need follows directly
  from query structure.
- The 32-record set is enough to unblock the next experiment, not to estimate broad deployment rates.
- Zero `none` labels is plausible for implementation-oriented open issues but makes this fold unsuitable
  as a balanced three-class production benchmark. Synthetic or stratified `none` cases remain necessary
  for that claim.

The result advances #6167 by supplying agreed read-only/current-state cases. It does not close #6033 or
establish a net-true micro-context winner.
