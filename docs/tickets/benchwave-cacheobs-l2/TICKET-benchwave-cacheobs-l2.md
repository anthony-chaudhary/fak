<!-- fak-benchwave-key: benchwave-cacheobs-l2 -->
# Bench-wave cacheobs L2: counter-delta tier decomposition + lineage vocab in internal/cacheobs (oss-port-bench-cacheobs-contract-20261109 leaf 2)

```routing
lane: internal/cacheobs
tracker: pub
```

Process cause: none

## Provenance (parent oss-port-bench-cacheobs-contract-20261109)
- Source: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, MIT-declared
  with no root LICENSE → **clean-room conceptual adaptation, ADAPT verdict; no source copied**.
- Parent plan: fak-private `docs/tickets/borrow-port-parts/oss-port-bench-cacheobs-contract-20261109.md`.
- Leaf 2 of 2. Depends on leaf 1 (capability.go, already landed) — same package.

## Work items (this leaf only)
1. Create `internal/cacheobs/tierdecompose.go`: counter-delta tier decomposition over two
   cumulative snapshots — L3 = Δprefetch, L2 = max(0, Δloadback − Δprefetch) (through-path
   subtraction), L1 = cached − L2 − L3 residual, miss = Δprompt × (1 − hit_rate) — with
   proportional rescale so L1+L2+L3 equals cached exactly, plus a closed layer-classify status
   vocabulary and NaN-not-zero sentinels for not-measured fields.
2. Closed-contract acceptance:
   - invariant holds exactly after rescale (property asserted over randomized delta inputs,
     including the overshoot case where transfers exceed cached);
   - status vocabulary is a closed enum (`server`/`derived`/`derived_no_signal`/`no_movement`
     semantics) with `String()`; no movement → `no_movement`, not zeros;
   - not-measured fields carry a NaN/absent sentinel, never 0 (0 is a measurement);
   - counter-reset detection: a decreasing cumulative counter flips the row to the
     reset/crash-affected status instead of clamping silently.
3. Recon BEFORE writing (read-only): `internal/cacheobs/tiers.go:200-260` (explicit-tier axis
   — the decomposer consumes counter deltas, never fabricates TierAccess bookings) and the
   landed `internal/cacheobs/capability.go` (alias-resolved metric names may feed the
   decomposition inputs).

## Boundaries
- Create and edit ONLY `internal/cacheobs/tierdecompose.go` and its exact test sibling in this
  public `fak` repo. Never touch capability.go beyond imports if needed, metricspec.go,
  tiers.go, or any other existing file; never touch fak-private.
- No scrape/HTTP code; the decomposer consumes caller-supplied snapshot deltas.
- No new scripts (.ps1/.sh/.bat/.cmd). Anti-brittle tests: closed contracts, property-based
  invariant assertions, never error prose.

## Done condition
- `go test ./internal/cacheobs -run 'TestTierDecompose' -count=1` green in a fresh run
  (forward slashes); `go test ./internal/cacheobs/... -count=1` green;
  `go vet ./internal/cacheobs/...` clean; `gofmt` clean.
- Committed by explicit path with DCO sign-off, subject = so-what, line 2 = why, line 3 =
  witness, subject ends with `(fak oss-port-bench-cacheobs-contract-20261109)`.
