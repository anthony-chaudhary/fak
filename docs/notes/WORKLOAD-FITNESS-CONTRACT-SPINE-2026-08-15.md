# Workload fitness contract spine

**Date:** 2026-08-15  
**Issue:** [#6893](https://github.com/anthony-chaudhary/fak/issues/6893)  
**Status:** working contract spine; legal-domain review is still missing, so this is not legal certification.

`internal/workloadfit` separates technical stack compatibility from purpose-specific fitness. A versioned domain contract classifies requirements as hard capability, authority control, evidence floor, preference, or cost. Only hard/authority/evidence requirements lower to `internal/stackresolve`; preference and cost remain assessment/ranking inputs and cannot silently become composition gates.

The same two technically catalogued candidates produce different outcomes:

```text
CODING fit: choose ponytail@r8
LEGAL REVIEW fit: choose legal-review-harness@r4
PONYTAIL for legal: refuse
  citations: missing — capability is not evaluated for this candidate
  confidential: missing — capability is not evaluated for this candidate
  human-review: unsupported — candidate evidence says capability is unsupported
BOUNDARY: fitness fixture is not legal certification; domain review remains required
SELFCHECK PASS
```

This does not say ponytail is globally bad. Its repository patch/test claims and observed loop preference make it the coding choice. For the draft legal-review contract, absence of evaluated citation traceability and confidentiality evidence remains `missing`, while an explicit lack of human-approval control is `unsupported`. Unknown, unsupported, and expired are distinct states. A vendor declaration cannot meet an `evaluated` evidence floor.

The legal contract vocabulary is deliberately marked `legal-workload-adapter-draft`. The `external-review-fixture` authority in the synthetic candidate is fixture data used to exercise proof-tier semantics, not evidence that an external legal expert reviewed fak. Domain-informed review is a remaining acceptance criterion of #6893 and validation is tracked in #6894.

## Existing authority bindings

- composition requirements lower to `stackresolve.Relation` rather than duplicating graph semantics;
- policy/control claims retain their policy source references;
- benchmark preferences retain benchmark authority, scope, tier, and expiry;
- model or tool evaluators can supply claims through the generic catalog without workloadfit owning their scores.

## Reproduce

```bash
go test -count=1 ./internal/workloadfit ./cmd/workloadfitdemo
go vet ./internal/workloadfit ./cmd/workloadfitdemo
go run ./cmd/workloadfitdemo -selfcheck
go run ./cmd/workloadfitdemo -selfcheck -json
```

Machine output is captured at `internal/workloadfit/testdata/selfcheck-witness.json`.
