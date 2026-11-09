<!-- fak-benchwave-key: benchwave-benchsweep-l1 -->
# Bench-wave benchsweep L1: restart-minimizing combo ordering in new internal/benchsweep (oss-port-bench-sweep-20261109 leaf 1)

```routing
lane: internal/benchsweep
tracker: pub
```

Process cause: none

## Provenance (parent oss-port-bench-sweep-20261109)
- Source: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, license
  MIT-declared with no root LICENSE → **clean-room conceptual adaptation, ADAPT verdict; no
  source copied**. Behavior contracts below come from the pinned private study note (§D).
- Parent plan: fak-private `docs/tickets/borrow-port-parts/oss-port-bench-sweep-20261109.md`.
- This leaf is leaf 1 of 2; leaf 2 (eligibility chain + always-written summary) is
  dependency-chained and dispatches after this leaf lands.

## Work items (this leaf only)
1. Create `internal/benchsweep/order.go`: restart-minimizing sweep ordering — stable-sort
   combos by a hashable server-config key so combos sharing a config run consecutively, and a
   restart plan that counts distinct config groups, restarts only on key change, and excludes
   all combos whose config already failed (no silent drops).
2. Closed-contract acceptance:
   - sort is stable (equal keys keep input order — asserted);
   - restart plan: contiguous groups per distinct key, restart count = distinct healthy keys,
     failed-key combos excluded and returned in a typed excluded record, never dropped
     silently;
   - empty input → valid empty plan, not an error.
3. Recon BEFORE writing (read-only): `internal/armbench/manifest.go` (existing bench manifest
   shapes to stay naming-compatible with).

## Boundaries
- Create and edit ONLY `internal/benchsweep/order.go` and its exact test sibling
  `internal/benchsweep/order_test.go` in this public `fak` repo. Never touch go.mod, go.sum,
  sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Engine-agnostic mechanics only: no server launching, no HTTP, no process management.
- Anti-brittle tests: assert closed contracts (typed records / sentinels), never error prose.

## Done condition
- `go test ./internal/benchsweep -run 'TestSortCombos|TestRestartPlan' -count=1` green in a
  fresh run (forward slashes); `go vet ./internal/benchsweep/...` clean; `gofmt` clean.
- Committed by explicit path with DCO sign-off, subject = so-what, line 2 = why, line 3 =
  witness, and the parent trailer `(fak oss-port-bench-sweep-20261109)` in the subject so
  borrow-queue plan detects the parent from commit history.
- The public pre-commit hook runs the secret leak audit and boundary gates automatically; do
  not bypass them.
