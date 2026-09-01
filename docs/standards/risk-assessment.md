# Proportionate risk assessment

Risk assessment is one field carried through the existing **issue intake → preflight → witness** lifecycle. It is not a parallel review, register, or approval system. Record enough detail for the change's credible failure modes; do not add ceremony where no meaningful hazard exists.

## Required vocabulary

Use one **Risk assessment** block with these fields:

- **Risk event:** the concrete unwanted event, not a category label.
- **Exposed subject/system:** people, data, processes, hosts, services, or workflows that could be harmed.
- **Severity:** `Low`, `Medium`, or `High` impact if the event occurs.
- **Likelihood:** `Unlikely`, `Plausible`, or `Likely` within the stated operating envelope.
- **Blast radius:** the maximum credible affected scope (one operation, child, user, repository, host, fleet, or external system).
- **Mitigations / containment:** controls that prevent the event or bound its spread.
- **Rollback / recovery:** how to return to a safe state and preserve diagnosis or progress.
- **Negative-path witness:** the executable or captured evidence that exercises the failure or limit path and proves containment, recovery, and operator-visible diagnosis.

Prefer observable statements. "Tests added" is not a risk event or witness; name what can fail and what evidence will show that it fails safely.

## When `None` is valid

`None` is valid only when the change introduces no credible new failure mode or exposure, and it must include a concrete rationale, for example: `None — corrects spelling in prose; no executable behavior, contract, external state, or operator procedure changes.` Placeholder-only values such as `None`, `N/A`, `low risk`, or `covered by tests` are incomplete.

Use a full assessment whenever the work adds or changes any of these triggers:

- process or resource guards, quotas, limits, cancellation, or termination;
- parent/child lifecycle, cleanup, orphaning, restart, or retry behavior;
- destructive operations or writes outside the repository;
- authentication, authorization, policy, permissions, secrets, or trust boundaries;
- data, schema, configuration, protocol, or CI/CD migrations;
- concurrency, locking, leases, ordering, shared state, or race-sensitive behavior;
- performance or resource ceilings, including CPU, memory, handles, disk, network, latency, and cost.

A trigger does not prescribe `High`; it requires an explicit, proportionate assessment.

## Lifecycle contract

### Intake

State the assessment before implementation so the issue's intended outcome and witness cover both benefit and credible harm. Follow [Benefit-harm defaults](benefit-harm-defaults.md): compare the change with the real alternative and account for exposed parties, not only the happy-path user.

### Preflight

Before admitting worker-ready work, reject a missing or placeholder-only assessment. Confirm that mitigations, rollback, and the planned negative-path witness match the stated severity, likelihood, blast radius, and operating envelope. High-severity or broad-blast-radius work needs containment and recovery evidence before rollout, not merely a follow-up.

Reassess whenever scope, implementation, dependency, operating envelope, or deployment plan introduces a new hazard or increases severity, likelihood, or blast radius. Update the issue and worker packet before continuing; do not rely on the original assessment.

### Witness and completion

Completion evidence must prove the promised behavior at the appropriate boundary:

1. Exercise the named failure, breach, cancellation, rollback, or partial-application path.
2. Prove the exposed subject/system remains safe or the damage stays within the stated blast radius.
3. Prove cleanup, rollback, or recovery completes without stranded state or unbounded retries.
4. Capture an operator-visible diagnosis when intervention may be required.
5. Confirm the positive path still works within the stated performance and resource ceilings.

Use the artifact type required by [Symptom witness](symptom-witness.md): reproduce and capture the actual observable symptom, then show it is removed or safely contained. A happy-path test alone does not close a non-`None` risk assessment. If the planned negative path cannot be exercised safely, provide the strongest bounded simulation or invariant proof available and state the remaining evidence gap; do not claim the risk retired.
