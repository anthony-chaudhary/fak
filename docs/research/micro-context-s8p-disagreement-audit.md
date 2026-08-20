---
title: "Micro-context S8p: live scheduler disagreement audit"
description: "An audit of S8o errors, gold-label agreement, and tool timing that tests whether any pre-answer signal supports adaptive admission."
---

# S8p — live scheduler disagreement audit

**Verdict: `no_pre_answer_signal`.** The S8o errors do not currently justify an adaptive tool-admission policy. The audit instead exposed two limits in the preceding live matrix: 15/16 held-out labels were majority rather than unanimous gold, and every S8o “adaptive” arm fetched the bounded GitHub receipt *before* classification. S8o therefore measured receipt-conditioned classification, not a decision about whether to open the tool.

## Frozen inputs

- S8i packet SHA-256: `2889531f4cf83df09f9335a239af76a2539ffc116b70c36ac8af6438c4f274e6`
- S8m fold SHA-256: `bd7133bafb7e7558623d07ff2f08be832bc136a6bb8c8b4cc42ffeed17a9a611`
- S8o live artifact SHA-256: `38fc5a1e3ed230faea931931871bc518557dc0396c3d260c902a4be8cd6c42d9`
- S8p artifact: `experiments/microcontext/s8p-live-disagreement-audit-2026-08-10.json`
- S8p SHA-256: `10E155DB9A04A319F0981A47CD1E5CEE2D4069F941DBC63A3FE685891398FA52`
- Model/route: `gpt-5.6-sol`, sanctioned OpenAI-compatible endpoint

## Error atlas

The verifier reconstructs all 16 records and all 192 S8o predictions. A record receives a causal label only as strong as its evidence permits.

| finding | records |
|---|---:|
| non-unanimous/questionable majority gold | 15 |
| unanimous record with cold/warm variance | 1 |
| any cross-arm disagreement | 9 |
| any identical-policy cold/warm flip | 6 |

This means an exact-match miss cannot be uniquely called a selector or model error for 15 records. The only unanimous record also changed across identical passes, so it is typed as stochastic variance rather than a stable routing failure. No fold implementation error or mutable-state transition was observed in the frozen receipts; those classes remain possible but unsupported.

## Blind re-grade

A policy-blind prompt omitted scheduler identity, prior predictions, and the folded gold label. It restated the distinction that implementing a live feature, mentioning “current,” or linking a service does not itself require mutable external state.

The fresh pass returned 15 `read_only` and one `current_state`. This is not replacement gold—the model family is not independent—but it is strong evidence that the S8m majority labels are prompt/rubric sensitive. Retuning a scheduler to those labels would risk learning adjudicator wording rather than evidence need.

## Paired diagnostic panel

Before execution, source fixed six records: the three largest persistent `current_state` misses and three contrasting disagreement records. Each received three `no_receipt` and three `bounded_receipt` calls. This panel is diagnostic only and never rescored as held-out quality.

- 36 calls completed.
- Six of 18 pairs changed label.
- Zero records showed a stable, repeatable receipt effect across all three pairs.
- Every no-receipt call returned `read_only`; receipt-conditioned calls occasionally flipped in both directions across repeats.

The bounded receipt is necessarily available only *after* tool admission. Even a stable receipt effect would not be a pre-answer admission signal. Here the effect was also unstable.

## Steelmanned interpretations

### Adaptive scheduler

The architecture remains compelling when a cheap observable feature predicts evidence need before opening a costly model/tool stage. Micro-windows provide the right cancellation, batching, cache, and typed-fold boundary. S8n proves that mechanism under controlled calibrated durations.

### Fixed cascade

On this live envelope, a fixed cascade avoids a second noisy semantic decision and was uniquely best-quality in S8o. The audit strengthens this position: current gold is too disputed to train a selector, and the tested receipt is post-admission and unstable.

### Stronger skeptical view

The task contract itself may be underspecified. “Current actionability” blends whether an issue *describes* a live system with whether answering it requires mutable state. Until independent humans or a model-distinct panel produce substantially more unanimous labels, scheduler optimization is premature.

## Decision boundary

Do not implement or claim a live adaptive tool-admission winner from S8o/S8p. The next admissible experiment must:

1. establish a higher-consensus evidence-need corpus;
2. expose only signals available before tool dispatch;
3. include a true two-stage arm that can decline the tool call;
4. compare against the tuned fixed cascade at equal independently graded quality; and
5. account for selector, tool, cancellation, cache, and retry costs.

## Reproduction

```powershell
go run ./cmd/microcontextdemo -verify-disagreement-audit experiments/microcontext/s8p-live-disagreement-audit-2026-08-10.json
go test ./cmd/microcontextdemo -run 'TestBuildAuditRecords|TestVerifyDisagreementAudit' -count=1
```
