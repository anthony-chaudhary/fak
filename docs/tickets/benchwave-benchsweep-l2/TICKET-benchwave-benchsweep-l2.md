<!-- fak-benchwave-key: benchwave-benchsweep-l2 -->
# Bench-wave benchsweep L2: eligibility chain + always-written summary in internal/benchsweep (oss-port-bench-sweep-20261109 leaf 2)

```routing
lane: internal/benchsweep
tracker: pub
```

Process cause: none

## Provenance (parent oss-port-bench-sweep-20261109)
- Source: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, MIT-declared
  with no root LICENSE → **clean-room conceptual adaptation, ADAPT verdict; no source copied**.
- Parent plan: fak-private `docs/tickets/borrow-port-parts/oss-port-bench-sweep-20261109.md`.
- Leaf 2 of 2. Depends on leaf 1 (order.go, already landed) — same package.

## Work items (this leaf only)
1. Create `internal/benchsweep/eligibility.go` and `internal/benchsweep/summary.go`: the
   pluggable eligibility chain — ordered verdict checks (artifacts-exist → hard-exclude
   statuses → status whitelist → wall-clock duration rescue) with first-decisive-check-wins
   semantics and logged rescues — plus an interrupt-safe always-written sweep summary with
   per-status counts.
2. Closed-contract acceptance:
   - chain verdicts are typed (accept / reject with reason token / rescue) and the first
     decisive check wins; a rescued combo is flagged and logged, never silently included;
   - status whitelist and minimum-duration rescue are data parameters, not literals;
   - summary writer: single-call idempotent write including per-status counts; designed for
     defer/finally use so an interrupted sweep still yields a complete ledger; zero-eligible
     input → valid empty ledger, not an error.
3. Recon BEFORE writing (read-only): the landed `internal/benchsweep/order.go` (match its
   typed-record style) and `internal/armbench/manifest.go`.

## Boundaries
- Create and edit ONLY `internal/benchsweep/eligibility.go`, `internal/benchsweep/summary.go`,
  and their exact test siblings in this public `fak` repo. Never touch order.go beyond imports
  if needed, go.mod, go.sum, sibling packages, or fak-private.
- Engine-agnostic mechanics only: no server launching, no HTTP, no process management.
- No new scripts (.ps1/.sh/.bat/.cmd). Anti-brittle tests: closed contracts (typed verdicts /
  reason tokens), never error prose.

## Done condition
- `go test ./internal/benchsweep -run 'TestEligibility|TestSummaryWriter' -count=1` green in a
  fresh run (forward slashes); `go test ./internal/benchsweep/... -count=1` green;
  `go vet ./internal/benchsweep/...` clean; `gofmt` clean.
- Committed by explicit path with DCO sign-off, subject = so-what, line 2 = why, line 3 =
  witness, subject ends with `(fak oss-port-bench-sweep-20261109)`.
