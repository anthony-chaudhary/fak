<!-- fak-debtwave-key: debtwave2-internal-preflight -->
# Debt lane preflight: advance internal/preflight from benchmarked (8.5) toward target 10 (#debtwave2-internal-preflight)

```routing
lane: internal/preflight
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/preflight` · Criticality: core · Weight: 3
- Maturity: **8.5 / 10** · Debt principal: 4.5 pts · Carrying cost: 1.22 pts · Total debt: 5.72 pts
- Evidence: 4 file(s), 1173 code lines, has tests=True, integrated=True, comment_ratio=0.176
- Health: critical (score 0.65); issues: critical_interest, modularity_deficit
- Interest drivers: core_runtime_criticality; modularity_surcharge (+0.08: high_coupling (11 deps, 4 dependents)); urgent_pacing_accelerator
- Reader next_action: `decouple preflight: reduce high dependency coupling (11 transitive dependencies, 4 dependents)`

## Work items (in order)
1. Split the named god-constructs in `internal/preflight` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Remove model-specific hardcoding in `internal/preflight`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane preflight --json
   ```

## Boundaries
- Edit ONLY files under `internal/preflight/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/preflight/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/preflight/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `preflight` (see witness command).
- Committed by explicit path with trailer `(fak internal/preflight)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
