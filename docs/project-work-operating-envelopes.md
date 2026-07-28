---
title: "Operating-envelope declarations"
description: "Project maturity and operating scope are separate declarations: how fak issue contract reads a ticket's completion standard and its target operating envelope."
---

# Operating-envelope declarations

Project maturity and operating scope are separate declarations. `demo`, `development`,
`integrated`, and `production` say how mature a work item is; an operating envelope says
where its evidence applies. A model that answers one toy request has useful demo evidence,
but it has not proved a production target of 1,000 concurrent requests.

`fak issue contract` reads these issue-body sections:

```markdown
## Completion standard
production

## Target operating envelope
- concurrency: >= 1000 requests
- duration: >= 60 minutes
- error rate: <= 1 percent
- regions: not-applicable (single-region product contract)

## Witnessed operating envelope
- concurrency: 1 requests
- duration: 1 minutes
- error rate: 0 percent
```

Each entry is `dimension: operator value unit`. Target operators are `>=`, `<=`, or
`=`; omitted target operators default to `>=`. Dimension and unit names are normalized
to lowercase but otherwise remain domain-defined, so producers must use the same unit on
both sides. Unit conversion is deliberately not guessed. A `not-applicable` target needs
a reason.

For an explicit production completion standard, review fails closed when:

- no target envelope is declared (`ISSUE_TARGET_ENVELOPE_MISSING`);
- an entry is malformed or a `not-applicable` entry has no reason
  (`ISSUE_OPERATING_ENVELOPE_INVALID`); or
- a target dimension is missing, uses a different unit, or is below/above its declared
  bound (`ISSUE_OPERATING_ENVELOPE_UNDER_TARGET`).

Human output prints the changed dimension and comparison. JSON emits the same result in
`operating_envelope`, including normalized target values, witnessed values, and typed
`gaps`. Explicit non-production work may record a narrow witnessed envelope without
claiming production coverage. Legacy issues with no completion standard remain
`undeclared` for the migration tracked by #4637/#4642; first-party authoring defaults are
tracked by #4638.

A small-scope item uses exactly the same contract. If a local command's real supported
scope is one user for one minute, target and witness can both declare those values and the
review reports `met`; it does not manufacture a thousand-unit load requirement.

## Staged scale evidence

A single witnessed envelope can now be replaced by provenance-labeled stages:

```markdown
## Required scale stages
toy, target-load, soak, recovery

## Scale evidence
- toy; witnessed; concurrency: 1 requests, duration: 1 minutes; environment=dev
- target-load; modeled; concurrency: 1000 requests, duration: 60 minutes; workload=synthetic
- soak; witnessed; concurrency: 1000 requests, duration: 60 minutes; environment=staging
- recovery; observed; concurrency: 1000 requests, duration: 60 minutes
```

The record form is `stage; provenance; dimensions; optional key=value attributes`.
Supported stages are `toy`, `development`, `representative`, `soak`, `target-load`,
`overload`, `degradation`, and `recovery`. Provenance is `witnessed`, `observed`,
`modeled`, or `extrapolated`. Optional attributes are `duration`, `workload`, and
`environment`.

Only directly `witnessed` or `observed` values qualify toward the target envelope.
Modeled and extrapolated records remain visible in JSON but cannot silently satisfy a
production target. The ticket selects relevant required stages; omitted stages are not
manufactured, but every selected stage must appear or review returns
`ISSUE_SCALE_EVIDENCE_STAGE_MISSING`. Malformed stages/provenance/attributes return
`ISSUE_SCALE_EVIDENCE_INVALID`. This lets a one-user local command require only its
`target-load` witness while a serving model can explicitly require target load, soak,
overload, and recovery.

## Continuous reconciliation

`fak issue reconcile --file SNAPSHOT.json [--now RFC3339] [--json]` compares the
previous contractual target, current target, direct witness, observed operating scope,
and witness freshness. The snapshot uses the same typed envelope values emitted by
`fak issue contract` and adds `witnessed_at`, `max_age`, and an optional
`contraction_reason`.

Its closed status vocabulary is:

- `aligned`: current witness covers the unchanged target before its freshness deadline;
  production credit remains current.
- `gap`: witness does not cover a target dimension.
- `stale`: witness is retained, but its declared maximum age elapsed.
- `expanded`: a target increased, a new target dimension appeared, or observed demand
  exceeded the declared capacity target; re-test is required.
- `contracted`: target scope decreased or a dimension disappeared; an operator reason
  and parent-denominator audit are required before credit can return.
- `unknown`: required data is absent or units/operators are incompatible; conversion is
  never guessed.

Every non-aligned result exits 3, sets `production_credit_current=false`, and emits an
exact action. This makes the command suitable for both a periodic loop and an on-demand
status/dispatch preflight. Supplying `--now` makes freshness witnesses reproducible.

## Canonical project-work contract

`fak issue contract --strict-project-work` turns project maturity and weighted scope into
a dispatch gate. Every dispatchable ticket must provide:

```markdown
## Work estimate
Estimate: 5 points (medium). Uncertainty: consumer inventory.

## Overall completion contribution
Parent scope baseline: #4636 rollout, 47 points. Contribution: 5/47 points (10.6%).

## Completion standard
production
```

The estimate and contribution must be positive, contribution cannot exceed the parent
baseline, and estimate points must equal contribution points so the work estimate and
portfolio numerator cannot silently diverge. `Parent context` must bind a `#N` parent.
The normalized completion vocabulary is `research`, `experiment`, `prototype`, `demo`,
`development`/`dev`, `integrated`, `staging`, or `production`.

JSON emits `project_work` with normalized points, denominator, contribution share,
completion standard, production-credit eligibility, invalid fields, and repair actions.
An explicit non-production ticket is valid but has `production_credit=false`. Strict
review returns `ISSUE_PROJECT_WORK_MISSING` for an undeclared legacy ticket and
`ISSUE_PROJECT_WORK_INVALID` for malformed or contradictory metadata. Without the strict
flag, the readout remains advisory for the measured migration in #4642; first-party
producers turn strict-compatible production defaults on under #4638 rather than guessing
legacy values.
