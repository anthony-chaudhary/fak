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
