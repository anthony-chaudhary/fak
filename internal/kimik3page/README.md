# internal/kimik3page

Paged-attention memory layout, block-table metadata, and physical block pool for
Kimi K3 hybrid (KDA + MLA) architectures, plus the Kimi K3 marketing-page
validator (required structural tokens present, unwitnessed superlative claims
rejected).

## Production wiring status (audit 2026-11-02, ticket debtwave2-internal-kimik3page)

The paged-attention structures (`PagePool`, `BlockTable`) and the page validator
are **not wired into any production command or registration graph yet**.
Evidence:

- Zero import sites repo-wide:
  `rg -n "github.com/anthony-chaudhary/fak/internal/kimik3page" --glob '*.go'` -> no matches.
- Only out-of-package references: `internal/architest/architest_test.go:756`
  (test-expectation map) and `internal/maturity/runtime-proofs.json:1780-1783`
  (a `fak version modules` presence witness — it lists every module on disk, so
  it proves presence, not invokability).
- Debt-lanes reader agrees: `integrated=false`, `dependents_count=0`, lane absent
  from `critical_paths`; maturity is capped at the 5.0 "tested" ceiling while
  unintegrated (`internal/debtlane/curve.go`).

Why this ticket did not wire it: the minimal connection needs a first import
edge from outside this package (a `cmd/fak` verb handler plus its dispatch case,
or a registration seam in a sibling package), which this ticket's boundary
forbids (edits confined to `internal/kimik3page/`). Recorded ABSENT-with-reason
rather than forcing new wiring.

Next step: a follow-up ticket without this boundary adds the first import edge
(e.g. a read-only `fak kimik3page` inspect verb over `DefaultConfig`/`PagePool`,
or a ctxmmu registration slot) so the lane can advance past the tested ceiling.
