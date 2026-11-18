# Wiring audit — internal/kvint2eval (debtwave2-internal-kvint2eval)

Read-only wiring audit for the `kvint2eval` debt lane (ticket
`docs/tickets/debtwave2-internal-kvint2eval/TICKET-debtwave2-internal-kvint2eval.md`,
baseline census 2026-11-02). Audited 2026-11-03 at branch HEAD `d3157c9e11`
(branch `fak/ticket-debtwave2-internal-kvint2eval`) and independently re-verified at
trunk tip `b882683481` (`origin/main`) via `git grep` + delta log; three concurrent
subagents (import/wiring sweep, scoring-model re-derivation, deterministic witness run)
all HOLD. Out-of-boundary follow-on tracked as open #14197.

## Finding: ABSENT-with-reason (disconnected, deliberately not wired here)

The lane's only health issue is `disconnected_wiring`: no production call path invokes
this package. Presence != invokability, so the audit establishes the full importer chain:

| Question | Answer | Re-check command (repo root) |
|---|---|---|
| Who imports `internal/kvint2eval`? | Nobody — zero hits tree-wide at branch HEAD and trunk tip | `rg -n '"github.com/anthony-chaudhary/fak/internal/kvint2eval"' -g '*.go'`; `git grep -n '"github.com/anthony-chaudhary/fak/internal/kvint2eval"' origin/main -- '*.go'` |
| Any non-test reference outside the package? | One metadata string: `internal/boundarylint/changedetector_baseline.go:183` lists `internal/kvint2eval/contract_test.go` in a changed-file baseline — data, not an import edge | `rg -n 'kvint2eval' -g '*.go' -g '!*_test.go'` |
| A cmd/ entry point or registry row? | No. Absent from the `cmd/fak/main.go` dispatch table, `cmd/fak/verbs_gen.go`, and the `internal/devindex` catalog | `rg -n -i 'kvint2eval' cmd/fak/verbs_gen.go cmd/fak/main.go internal/devindex/` |
| Does BFS from cmd/ reach it? | No — with zero inbound edges, `scanReachableFromCmd` (`internal/debtlane/scan.go`) cannot reach the lane; census reports `evidence.integrated=false` | `go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane kvint2eval --json` |

The package ships exactly two API surfaces — `EvaluateJSON` and `Evaluate`
(`contract.go`) — both ready to be called; nothing calls them.

## Why no wiring was landed in this ticket

1. **Boundary.** The ticket permits edits ONLY under `internal/kvint2eval/`. A minimal real
   connection (a `fak kvint2eval` CLI verb driving `EvaluateJSON` over a JSON evidence record)
   requires six files, ALL outside the boundary: new `cmd/fak/kvint2eval.go`, the dispatch case
   in `cmd/fak/main.go`, the generated row in `cmd/fak/verbs_gen.go` (regenerated via
   `fak-dev index verbs --write-usage`), and the `internal/devindex` catalog rows
   (`verbs.go` + `tiers.go` + `devreuse.go` — see commit `f55aed9919` for the computetune
   curation precedent). Per ticket work item 2, when the minimal connection does not fit the
   boundary the verdict is recorded ABSENT-with-reason and no new wiring is forced.
2. **No hollow wiring.** A blank import or presence-only registration would flip
   `integrated=true` without any call path — exactly the gaming this audit forbids.

## Reader impact (why the numbers are frozen until wiring lands)

While `integrated=false`, `internal/debtlane/curve.go` hard-caps the lane at the tested
ceiling: `if !e.Integrated && score > 5.0 { score = 5.0 }` (curve.go L80-81). Pre-clamp raw is
already at the curve's max for a disconnected unit: 4.0 tested base + 0.5 testfiles>1 +
1.0 dogfooded + 1.0 benchmarked + 0.5 clean comments + 0.5 dependents<=25 = 7.5. Debt: gap
8−5 = 3 × weight 2 = principal 6.0; enabling base rate 0.08 (no surcharge applies while
unintegrated) → carrying 0.48; total_debt 6.48 (< ticket baseline 6.78 after comment-pruning
leaf `8669c844c6` retired the `excess_comment_bloat` driver: ratio 3.5% → 1.8%, rate 0.13 →
0.08). No edit inside `internal/kvint2eval/` can move maturity or total_debt: `Integrated` is
set exclusively from cmd/-reachability (`scan.go` L1032-1034, `scanReachableFromCmd` L1479),
and this document is census-inert (`inspectUnitEvidence` skips non-Go files).

Once out-of-boundary wiring flips `integrated=true`, the same evidence scores 7.5 + 1.5
integration + 1.0 production-grade bonus = 10.0 (rung `production_grade`) ≥ target 8.0,
retiring the lane automatically (gap 0 → debt 0) — nothing else in-lane blocks the target.

## Next checkable step

Land #14197 (boundary spanning `cmd/fak` + `internal/devindex` + `internal/kvint2eval`): add
the `fak kvint2eval` verb per the six-file choke point above, then re-run the Q1 sweep to
confirm the first production import edge appears and the census flips `integrated=true`
(expected: maturity 10.0/8.0, total_debt 0).
