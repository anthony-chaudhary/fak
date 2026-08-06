# Research: hypotheses, evidence, and promotion

**Primary audience:** research readers evaluating an idea's provenance, evidence, and maturity before using it to guide implementation.
**Lifecycle:** research; conclusions here remain hypotheses or scoped observations until a maintained authority adopts them.
**Generation:** usually `gen/second-next` or `gen/future`, with some research performed to validate `gen/next` work; the route itself is `gen/now` documentation.
**Authority:** current behavior comes from maintained guides, contracts, tests, and shipped implementation linked by the [documentation index](../index.md).
**Support:** use the [contributor route](../../CONTRIBUTING.md) for implementation help and the [operator guides](../operator/) for supported operation.

**Next action:** before applying a research claim, find its provenance, maturity, and promotion witness; if any is missing, keep the claim in research and verify it independently.

## Active focus

- [Micro-context fabrics for 100–10,000 parallel agents](micro-context-fabrics.md) — split one cached agent base into many bounded logical contexts; includes the runnable 10k synthetic spine and controlled-kernel/API-only research ladder.
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
