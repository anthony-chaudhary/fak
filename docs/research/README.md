---
title: "Research route: hypotheses, evidence, and promotion"
description: "Index of fak research notes and captured stage witnesses, with the lifecycle rule that keeps a hypothesis out of the maintained authority set."
---

# Research: hypotheses, evidence, and promotion

**Primary audience:** research readers evaluating an idea's provenance, evidence, and maturity before using it to guide implementation.
**Lifecycle:** research; conclusions here remain hypotheses or scoped observations until a maintained authority adopts them.
**Generation:** usually `gen/second-next` or `gen/future`, with some research performed to validate `gen/next` work; the route itself is `gen/now` documentation.
**Authority:** current behavior comes from maintained guides, contracts, tests, and shipped implementation linked by the [documentation index](../index.md).
**Support:** use the [contributor route](../../CONTRIBUTING.md) for implementation help and the [operator guides](../operator/) for supported operation.

**Next action:** before applying a research claim, find its provenance, maturity, and promotion witness; if any is missing, keep the claim in research and verify it independently.

## Active focus

- [Micro-context fabrics for 100–10,000 parallel agents](micro-context-fabrics.md) — split one cached agent base into many bounded logical contexts; includes the runnable 10k synthetic spine and controlled-kernel/API-only research ladder.
- [Micro-context operators for general large-input LLM work](micro-context-large-input-operators.md) — partition record/field/group inputs, adaptively select filters or tools, emit typed facts, fold with provenance, and stop only under an answer-safe contract.
- [Micro-context S8 large-input operator witness](micro-context-s8-large-input-operator.md) - fixture-backed 1,000-record partition, deterministic prefilter, semantic map, exact reuse/invalidation, bounded fold, and oracle proof.
- [Micro-context S8a adaptive filter-selector witness](micro-context-s8a-filter-selector.md) — allowlisted per-record routing across exact exclusion, semantic filtering, group widening, and escalation with confusion/cost/replay telemetry.
- [Micro-context S8b read-only tool-enrichment witness](micro-context-s8b-tool-enrichment.md) — typed allowlisted reads with cross-record dedupe, quotas, timeout/retry, cancellation, restart cache, recursive output bounds, and independently read-back citations.
- [Micro-context S8c provenance-fold witness](micro-context-s8c-provenance-fold.md) — bounded typed reducers, content-addressed trees, minority/uncertainty retention, source-resolved claims, and path-local invalidation.
- [Micro-context S8d tuned-baseline falsification spine](micro-context-s8d-falsification-spine.md) — fixture-modeled five-pipeline decision boundaries that expose both micro-context wins and losses while leaving live net-true evidence open.
- [Micro-context S8e witnessed effect receipts](micro-context-s8e-effect-receipts.md) — capability/idempotency/resource-bound effects with denial, conflict, partial failure, cancellation ambiguity, restart replay, breaker, and independent read-back states.
- [Micro-context S1 real-endpoint witness](micro-context-s1-real-endpoint.md) — 100/100 contexts through four bounded workers, with TTFT/usage telemetry and a retained 16-worker overload finding.
- [Micro-context S2 shared-prefix A/B](micro-context-s2-prefix-ab.md) — no cache benefit observed on the first real endpoint; scoped concurrency improved aggregate work while worsening TTFT.
## Read research by maturity

Research material makes options and uncertainty inspectable. It can identify a useful mechanism, report a measurement, or define a future architecture without claiming that fak currently supports it.

| Maturity | What the material establishes | What you may do next |
|---|---|---|
| **Hypothesis** | A falsifiable idea, assumption set, or proposed mechanism. | Reproduce or design the named experiment; do not present it as product behavior. |
| **Observed** | A result with a named source, environment, date, and measurement method. | Re-run it in the target environment before generalizing beyond its stated scope. |
| **Validated option** | Evidence has retired specified technical uncertainty, but product integration or support gates remain. | Follow the promotion trigger and linked implementation issue. |
| **Promoted** | A maintained contract, guide, test, or shipped implementation has adopted the conclusion. | Leave this route and use that maintained authority for current behavior. |

A paper, benchmark, model output, or committed study can be valuable evidence without being support evidence. Support begins only where a maintained product route states its scope and names a reproducible witness.

## Provenance contract

A research claim should make these fields discoverable:

1. **Question or hypothesis** — the proposition being tested.
2. **Source and date** — paper, repository revision, dataset, model, interview, or local observation.
3. **Method and environment** — baseline, hardware, software version, workload, mode, and constraints.
4. **Result and uncertainty** — what was observed, its scope, and what remains unknown.
5. **Maturity and generation** — hypothesis, observed, validated option, or promoted; plus the applicable generation stream.
6. **Promotion witness** — the evidence and maintained destination required before the claim becomes implementation guidance.

For performance or efficiency claims, apply the [net-true-value standard](../standards/net-true-value.md): compare against the real tuned alternative, include added costs, state scope and provenance, and provide a reproducible witness.

## Promotion, demotion, and retirement

The [Generation Contract](../generation.md#promotion-verbs) governs generation changes. Promote research closer to `gen/now` only when its named blocker is retired by evidence. Demote it when an assumption fails or a witness regresses. Retire it when the option is superseded, rejected with evidence, completed, or no longer has an owner and witness path.

Promotion does not rewrite history. Put durable instructions in the maintained contract, guide, test, or implementation route, then link the research record for rationale. Superseded research remains available through the [dated-notes route](../notes/); retired material belongs in the [archive](../archive/) with its replacement or retirement decision.

## Mode and discovery

Research often depends on a specific backend, model, release, hardware tier, dataset, or offline/live mode. Apply a result only to the mode and generation it names. A future-generation label expresses horizon rather than support, and priority expresses importance rather than maturity.

The repository's studies and dated investigations currently live under [`docs/notes/`](../notes/); the curated [Notes & research index](../../INDEX.md#notes--research-docsnotes) is the human route. [`docs/sota/`](../sota/) tracks state-of-the-art comparisons. Use [`llms.txt`](../../llms.txt) for machine-oriented discovery.

- [Micro-context cache-value Track-1 fold](micro-context-cachevalue-track1.md) — controlled S2b reuse enters the witnessed P&L while synthetic and provider-dollar evidence remain fenced out.
- [Micro-context S2b controlled in-kernel prefix-cache A/B](micro-context-s2b-kernel-cache-ab.md) — fresh-process arms reconcile response usage with RadixAttention counters and observe a fixture-scoped 2.16x shared-base service gain.
- [Micro-context S3 hibernation/restart](micro-context-s3-hibernation-restart.md)
- [Micro-context S4a lightweight descriptor](micro-context-s4-lightweight-descriptor.md)
- [Micro-context S4b compatibility scheduler](micro-context-s4-compatibility-scheduler.md)
- [Micro-context S4c effect safety](micro-context-s4-effect-safety.md)
- [Micro-context S4d: bounded multi-turn continuation](micro-context-s4d-multi-turn-descriptor.md) — exact 1,000×3 turn accounting with continuation-token and byte-verified mid-task restore.
- [Micro-context S4e real compatibility-batch execution](micro-context-s4e-compat-batch-execution.md) — planner batches execute through the in-kernel batch seam; the first mixed-length CPU fixture is an honest 0.539x negative result.
- [Micro-context S5a controlled-kernel 1,000-context ramp](micro-context-s5a-controlled-kernel-1k.md)
- [Micro-context S6 API-only adapter](micro-context-s6-api-only.md)
- [Micro-context health scorecard](micro-context-health-scorecard.md) — deterministic witness fold grades the controlled 1k CUDA ledger A/100 and names the missing second-run drift baseline.
- [Micro-context outcome counters](micro-context-outcome-counters.md) — the existing quality ledger exposes reconciled success/error/refusal totals; controlled 1k CUDA readout is 1000/0/0.
- [Micro-context quality and observability ledger](micro-context-quality-ledger.md)
- [Micro-context S7 mixed-tenant fairness](micro-context-s7-fairness.md)
