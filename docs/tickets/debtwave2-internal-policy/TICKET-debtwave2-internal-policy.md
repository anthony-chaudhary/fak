<!-- fak-debtwave-key: debtwave2-internal-policy -->
# Debt lane policy: advance internal/policy from benchmarked (8.5) toward target 10 (#debtwave2-internal-policy)

```routing
lane: internal/policy
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/policy` · Criticality: core · Weight: 3
- Maturity: **8.5 / 10** · Debt principal: 4.5 pts · Carrying cost: 1.4 pts · Total debt: 5.9 pts
- Evidence: 33 file(s), 8956 code lines, has tests=True, integrated=True, comment_ratio=0.264
- Health: critical (score 0.65); issues: critical_interest, modularity_deficit
- Interest drivers: core_runtime_criticality; moderate_blast_radius (9 dependents); modularity_surcharge (+0.08: high_coupling (10 deps, 9 dependents)); urgent_pacing_accelerator
- Reader next_action: `decouple policy: reduce high dependency coupling (10 transitive dependencies, 9 dependents)`

## Work items (in order)
1. Split the named god-constructs in `internal/policy` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Remove model-specific hardcoding in `internal/policy`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane policy --json
   ```

## Boundaries
- Edit ONLY files under `internal/policy/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/policy/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/policy/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `policy` (see witness command).
- Committed by explicit path with trailer `(fak internal/policy)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
