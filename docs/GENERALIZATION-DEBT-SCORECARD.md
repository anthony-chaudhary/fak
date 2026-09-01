# Generalization debt scorecard

`fak score generalization` makes singular production implementations visible when they are coupled to one model, backend, or provider instead of an interface, adapter, factory, or registry.

```bash
fak score generalization
fak score generalization --json
fak score generalization --workspace <root> --json
```

The JSON schema is `fak-generalization-debt-scorecard/1`. Output is deterministic: files, findings, evidence terms, and interest drivers have stable ordering.

## What scores

The first schema version scans production Go declarations for model-, backend-, and provider-specific identifiers or string literals. A singular implementation starts at 8 debt points (`high` / `accelerating`). Multiple unsupported variants and exported blast radius add points, so wider or mixed coupling becomes `critical` / `compounding`.

Generic abstraction declarations score zero. This includes interface types and declarations whose subject is a registry, registration function, factory, or adapter. Generic words in a comment or local variable do not erase a concrete finding.

The scan excludes `_test.go`, generated Go, `testdata`, examples, docs, vendor and scratch trees, and paths dedicated to historical Qwen3.6 artifacts. A historical mention inside an otherwise current production file does not hide current debt.

## Debt disposition

Every finding remains in `totals.debt_points`; accepting debt changes its disposition, not its visibility or score.

| Disposition | Meaning | Required metadata |
|---|---|---|
| `accidental_unaccepted` | No complete, explicit acceptance record is attached. | None |
| `accepted_temporary` | A singular implementation is intentionally carried to reach a stage or migration gate. | `rationale`, `owner`, and `exit_gate` |

Attach accepted temporary debt to the declaration's doc comment:

```go
// fak:generalization-debt accepted_temporary rationale="ship the first native spine" owner="runtime" exit_gate="retire after the backend registry serves two implementations"
func Qwen38NativeBackend() {}
```

All three values must be present and non-empty, in that order. A malformed or incomplete annotation fails closed as `accidental_unaccepted`.

The exit gate is the retirement contract. Once the gate is met, remove or generalize the implementation and remove the annotation; retired debt is therefore absent from the live findings rather than represented as a third kind of current debt.

## Operational interest

`interest` is an operational carrying-cost signal, not money or a forecast:

| Points | Band | Rate label |
|---:|---|---|
| 0–4 | `low` | `baseline` |
| 5–7 | `moderate` | `elevated` |
| 8–9 | `high` | `accelerating` |
| 10+ | `critical` | `compounding` |

Drivers explain the band deterministically: `implementation_specificity`, `unsupported_variant_count`, `blast_radius`, and (for accepted debt) `explicit_retirement_gate`. The retirement-gate driver records governance; it does not discount the underlying debt.

## Adoption contract for other scorecards

New or migrated debt scorecards can adopt this shape additively:

1. Keep a stable schema identifier and deterministic finding order.
2. Separate debt magnitude from disposition; accepted debt remains in totals.
3. Use the closed disposition values `accidental_unaccepted` and `accepted_temporary` for live debt.
4. Require rationale, owner, and a checkable exit gate before classifying debt as accepted.
5. Emit a qualitative interest band, rate label, and sorted drivers; do not manufacture monetary precision.
6. Represent retirement by removal from the live debt set after the exit gate is proved, while preserving history in the issue, plan, or ledger that accepted it.

This is an adoption contract, not a forced migration of existing scorecards. Each scorecard should migrate only when its own findings have a meaningful debt magnitude and retirement witness.
