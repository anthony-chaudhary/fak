# Wiring audit: internal/workspaceslot (debtwave2)

Records the debtwave2 read-only wiring-audit verdict for this lane; presence of the package in the tree does not mean invokability from any shipped command.

Lane ticket: `docs/tickets/debtwave2-internal-workspaceslot/TICKET-debtwave2-internal-workspaceslot.md` (baseline 2026-11-02).

## Verdict

ABSENT-with-reason: no production call path invokes this package. The sole non-test importer is `internal/harnessbench`, which itself has zero inbound imports and is unreachable from `cmd/`.

## Evidence

Commands run 2026-11-02 in this checkout:

- `rg -n "github.com/anthony-chaudhary/fak/internal/workspaceslot" --glob '*.go'` -> exactly one hit: `internal/harnessbench/harnessbench.go:16` (non-test). No test files import it.
- `rg -n "github.com/anthony-chaudhary/fak/internal/harnessbench" --glob '*.go'` -> zero import hits repo-wide; `harnessbench` is referenced only by its own package files (an unrelated map key `"harnessbench": 5` exists at `internal/architest/architest_test.go:52`, not an import).
- No `.go` file under `cmd/` references either import path (`cmd/fak/harness_compare.go` defines an unrelated local type `HarnessBenchOutcome`).
- Reachability rule (`internal/debtlane/scan.go:1479` `scanReachableFromCmd` + `:1413` `BuildInternalImportGraph`): `Evidence.Integrated` for a Go-package lane is true only when the package is reachable via BFS from imports found in `cmd/` `.go` files. Sole edge into workspaceslot is harnessbench -> workspaceslot; zero edges into harnessbench; therefore workspaceslot is unreachable from `cmd/` and `Integrated=false`.
- Maturity cap (`internal/debtlane/curve.go:79-82`): `!Integrated` caps score at 5.0, so the lane reads 5.0/8 with total debt 6.48 (principal 6.0, carrying 0.48 at rate 0.08) until a reachable consumer imports it.
- Package contents are confined to the lane: `workspaceslot.go`, `dispatch_helper_unix.go`, `dispatch_helper_windows.go`, `workspaceslot_test.go`.

## Why not wired now

The lane ticket boundary restricts edits to files under `internal/workspaceslot/`; connecting into a production command or registration graph requires a consumer-side edit outside the lane (a `cmd/` seam or a reachable internal package), which exceeds that boundary, so per ticket work item 1 the audit evidence is landed instead of forcing new wiring.

Note: `harnessbench` is itself a witness suite (architest catalog: "high-churn soak and thundering-herd saturation witness suite", #11183), not a production seam, so its import of this package cannot count as a call path.

## Reader impact

While `Integrated=false`, the debt-lanes reader caps maturity at 5.0 (`internal/debtlane/curve.go:79-82`); the lane stays at 5.0/8, total debt 6.48.

## Next checkable step

Land a consumer-side import of `internal/workspaceslot` from a package reachable from `cmd/` (production seam that allocates agent-execution slots), then re-run:

```
go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane workspaceslot --json
```

and confirm `Integrated=true` / maturity > 5.0.

## Re-validation 2026-11-03 (HEAD 21b6512ec8)

Re-run for the debtwave2 ticket close-out; the verdict above still holds.

- Independent subagent re-audit: all seven load-bearing claims re-verified **HOLDS** at HEAD 21b6512ec8 (importer scans, reachability rule `scanReachableFromCmd` BFS seeded only from `cmd/*.go` imports, cap rule, scoring-rule stability since baseline, ticket-boundary argument). No new consumer anywhere in the repo (`rg -ln "workspaceslot|harnessbench" --glob '*.go'` = 8 hits, all under `internal/`).
- Trunk drift check: none of the commits between this worktree's base and the live trunk tip touches `internal/workspaceslot` or `internal/harnessbench`; re-confirmed at landing base 49c3ff7773 (origin/main), where the importer scan still returns exactly the one `harnessbench -> workspaceslot` edge, so the verdict is stable across the pending trunk integration.
- Fresh witness receipts: `go test ./internal/workspaceslot/... -count=1` -> ok (78s); `go vet ./internal/workspaceslot/...` -> clean.
- Witness command output at 21b6512ec8: maturity **5.0/8** (rung `tested`), `total_debt` **6.48** (principal 6.0, carrying 0.48 @ rate 0.08), `integrated=false`, `has_tests=true`, `dogfooded=true`, `benchmarked=true`, `comment_ratio=0.127`, `excess_comments=false`, health `degraded (0.85)` with issue `disconnected_wiring`, next_action `integrate workspaceslot: connect into production command or registration graph`.

### Why the numbers cannot move inside this lane

The live census already credits every achievable axis: pre-cap score is 4.0 (tested base) + 0.5 (clean comments) + 0.5 (dependents <= 25) + 1.0 (dogfooded) + 1.0 (benchmarked) = **7.0**, held at **5.0** solely by the `!Integrated` cap (`internal/debtlane/curve.go:79-82`). No in-lane edit (tests, benchmarks, docs, comment hygiene) can raise the score or lower `total_debt` while the cap binds; the only unlock is a consumer-side import edge from a `cmd/`-reachable package, which the ticket boundary forbids this lane from making. Per ticket work item 1 this is therefore recorded as ABSENT-with-reason with the audit evidence landed, not forced wiring.

### Status (growth-mindset)

Integration is not wired yet. Evidence: this section + the receipts above. Missing: one consumer-side import in a `cmd/`-reachable package (outside `internal/workspaceslot/`). Next step: land that seam in the owning lane (a production entry point that allocates agent-execution slots), then re-run the witness command and confirm `Integrated=true` / maturity 7.0.
