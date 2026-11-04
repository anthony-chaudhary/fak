# Wiring audit — production reachability of `internal/nativeperfcorrelation`

Audit date: 2026-11-02 · Ticket: flow ticket `debtwave2-internal-nativeperfcorrelation`
(work item 2: read-only wiring audit; presence != invokability).

## Method

- Repo-wide import search: `rg -n "anthony-chaudhary/fak/internal/nativeperfcorrelation" --glob '*.go'`
  (test vs non-test importers separated), then the same search for each importer, transitively,
  up to `cmd/`.
- Cross-check against the debt-lanes reader's own definition of integration:
  `internal/debtlane/scan.go` `scanReachableFromCmd` seeds a BFS from imports found in `*.go`
  files under `cmd/` and walks the internal import graph. A lane is `Integrated` only if its
  package name lands in that reachable set.

## Findings

- Sole non-test importer: `internal/nativeperfartifact/index.go` (uses the `NativeEngine`
  constant inside its scrub path). Zero hits under `cmd/`, `pkg/`, or `tools/`.
- Non-test importers of `internal/nativeperfartifact`: **none** (self-references plus the
  test-only depth budget in `internal/architest/architest_test.go`).
- Import chain therefore terminates short of every `cmd/` binary:

  ```text
  cmd/* ──(no edge)──> internal/nativeperfartifact ──> internal/nativeperfcorrelation
  ```

- Corroborating references (not call paths): `docs/observability/native-performance-correlation.md`
  (producer example using `NewIndex`), `docs/observability/native-performance-artifacts.md`,
  `docs/releases/v0.48.0.md`, `docs/learning/mastery.md`, architest import-depth budgets
  (`nativeperfcorrelation: 2`). No matches under `docs/benchmarks/` or `examples/`.

## Verdict: ABSENT-with-reason

No production call path invokes this package; only its own tests exercise it, so the debt-lanes
reader reports `disconnected_wiring` / `integrated=False` and caps maturity at the tested ceiling
(5.0) regardless of comment hygiene.

Reason the fix is not landed here: a minimal connection would require editing sibling paths —
at minimum a new `cmd/fak` verb leaf importing this package (or `internal/nativeperfartifact`)
plus the `cmd/fak/main.go` dispatch table (≈2–3 files, e.g. a `fak nativeperfcorr` verb or an
extension of the existing bench verbs). This ticket's boundary allows edits ONLY under
`internal/nativeperfcorrelation/`, so no wiring was forced. The true top-level orphan one level
up is `internal/nativeperfartifact`; wiring it wires both.

## Next checkable step

Open a follow-on lane/ticket for the production verb (suggested: `fak nativeperfcorr` leaf in
`cmd/fak/` backed by `internal/nativeperfartifact` + this package, with one CLI witness test).
Witness that it landed: `fak debt-lanes --lane nativeperfcorrelation` shows `integrated=True`
and the lane climbs past the 5.0 tested ceiling.
