# Wiring audit — debtwave2-internal-providerjobaccounting

Read-only production-wiring audit required by flow ticket
`debtwave2-internal-providerjobaccounting` (work item 1), run 2026-11-03 from the
ticket worktree at base commit `3231fc0688`.

## Verdict: ABSENT (disconnected — never invoked by a production call path)

This package is the docs/standards #9575 provider-job-accounting schema-conformance
suite: it validates committed JSONL ledger fixtures against the closed issue contract.
It declares no exported production API and no `init()`; its only invocation surface is
`go test ./internal/providerjobaccounting/...`. Presence of the package (and its tests)
is not invokability; the import scan proves no production caller exists.

## Evidence (re-run these to reproduce)

```sh
# 1. Pinned opt-in repository-wide scan (non-test .go files under cmd/, internal/,
#    pkg/, tools/, examples/): zero importers, positive control on internal/modver holds.
FAK_PROVIDER_JOB_ACCOUNTING_WIRING_AUDIT=1 \
  go test ./internal/providerjobaccounting/ -run TestWiringAuditRecordsAbsentProductionCallPath -count=1 -v

# 2. Manual import scan (same finding, one command):
rg -n 'github.com/anthony-chaudhary/fak/internal/providerjobaccounting' --glob '*.go' -g '!*_test.go'

# 3. Debt-lanes reader agrees: integrated=false (unreachable from the cmd/ import graph
#    via internal/debtlane/scan.go scanReachableFromCmd).
go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane providerjobaccounting --json
```

The ABSENT finding is pinned in code, not just prose:
`wiring_audit_test.go` (`TestWiringAuditRecordsAbsentProductionCallPath`) fails on
purpose if a non-test importer ever appears, so a future real connection cannot land
silently past this record.

## Why no wiring was forced

Two independent fences make the minimal connection out of scope for this ticket:

1. **Ticket boundary.** Edits are restricted to `internal/providerjobaccounting/`.
   Every viable seam edits a sibling file: a blank import in
   `internal/registrations/registrations.go` or a new probe verb under `cmd/fak/`.
   Per work item 1, when the minimal connection does not fit the allowed surface the
   finding is recorded ABSENT-with-reason instead of forcing new wiring.
2. **Facade ratchet.** `internal/architest/facade_guard_test.go`
   (`TestRegisteredLeavesFacadeImports`, issues #12873 / fak-private#2609) refuses a
   blank import of any package that declares no `init()`: such an import links nothing
   and only makes reachability readers count the package as wired — "an island
   masquerading as a connection." This suite has no `init()` and none is warranted, so
   the one-line defconfig link is not a sanctioned route here; allow-listing it would
   mean editing the shared architest gate, also outside this boundary.

## Pinned lane numbers (2026-11-03 census)

Maturity stays **5.0 / target 8** and total_debt stays **6.48** under the
`integrated=false` hard cap (`internal/debtlane/curve.go` L79-82); every other curve
term is already maxed (raw 7.5: tested base 4.0 + multi-test-file 0.5 + dogfooded 1.0
+ benchmarked 1.0 + clean comments 0.5 + low dependents 0.5). Interest is at the
enabling floor 0.08 with no active surcharges (`internal/debtlane/interest.go`
L76, `CalculateInterest`). Census skips non-Go files (`internal/debtlane/scan.go`
L1061), so landing this record moves no scored field.

## Next checkable step

A follow-on, separately-scoped ticket may promote the test-local `validateLedger`
contract into an exported production API in this package and wire a real caller
(smallest seam: a `fak` verb that validates provider-job-accounting ledger JSONL
against the #9575 contract, `cmd/fak/` + dispatch registration). That flips
`integrated: true`, unclamps maturity past the tested ceiling, and retires the
remaining 3.0-point gap on this lane.
