<!-- fak-benchwave-key: benchwave-cacheobs-l1 -->
# Bench-wave cacheobs L1: capability probe + metric alias table in internal/cacheobs (oss-port-bench-cacheobs-contract-20261109 leaf 1)

```routing
lane: internal/cacheobs
tracker: pub
```

Process cause: none

## Provenance (parent oss-port-bench-cacheobs-contract-20261109)
- Source: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, license
  MIT-declared with no root LICENSE → **clean-room conceptual adaptation, ADAPT verdict; no
  source copied**. Behavior contracts below come from the pinned private study note (§B-§C).
- Parent plan: fak-private `docs/tickets/borrow-port-parts/oss-port-bench-cacheobs-contract-20261109.md`.
- This leaf is leaf 1 of 2; leaf 2 (counter-delta tier decomposition + lineage vocab) is
  dependency-chained and dispatches after this leaf lands.

## Work items (this leaf only)
1. Create `internal/cacheobs/capability.go`: a capability probe that derives explicit
   availability booleans from metric PRESENCE in a scraped name set (consumers gate on
   capability, never infer from silent zeros), plus an alias table resolving canonical metric
   names from upstream variants in both directions.
2. Closed-contract acceptance:
   - probe returns a typed availability struct with explicit bools per capability; absence and
     zero are distinguishable (presence-based only);
   - alias resolution: canonical name from any registered alias (both directions tested);
     unknown name → typed `ErrUnknownMetric` (`errors.Is`);
   - alias table is a data literal (no logic in the table); renaming an upstream metric is a
     table edit, proven by a test that adds an alias and resolves through it.
3. Recon BEFORE writing (read-only): `internal/cacheobs/metricspec.go:77-169` (existing typed
   metric-spec surface — complement it, do not modify it).

## Boundaries
- Create and edit ONLY `internal/cacheobs/capability.go` and its exact test sibling
  `internal/cacheobs/capability_test.go` in this public `fak` repo. Never touch go.mod,
  go.sum, metricspec.go, tiers.go, or any other existing file; never touch fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- No scrape/HTTP wiring in this leaf; the probe consumes a caller-supplied name set.
- Anti-brittle tests: assert closed contracts (sentinels / `errors.Is`), never error prose.

## Done condition
- `go test ./internal/cacheobs -run 'TestCapability|TestMetricAlias' -count=1` green in a
  fresh run (forward slashes); `go vet ./internal/cacheobs/...` clean; `gofmt` clean.
- Committed by explicit path with DCO sign-off, subject = so-what, line 2 = why, line 3 =
  witness, and the parent trailer `(fak oss-port-bench-cacheobs-contract-20261109)` in the
  subject so borrow-queue plan detects the parent from commit history.
- The public pre-commit hook runs the secret leak audit and boundary gates automatically; do
  not bypass them.
