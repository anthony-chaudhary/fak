<!-- fak-debtwave-key: debtwave2-internal-adjudicator -->
# Debt lane adjudicator: advance internal/adjudicator from benchmarked (8) toward target 10 (#debtwave2-internal-adjudicator)

```routing
lane: internal/adjudicator
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/adjudicator` · Criticality: core · Weight: 3
- Maturity: **8 / 10** · Debt principal: 6 pts · Carrying cost: 2.1 pts · Total debt: 8.1 pts
- Evidence: 54 file(s), 15260 code lines, has tests=True, integrated=True, comment_ratio=0.212
- Health: critical (score 0.65); issues: critical_interest, modularity_deficit
- Interest drivers: core_runtime_criticality; high_blast_radius (29 dependents); modularity_surcharge (+0.08: god_file (reversibility.go: 1520 lines), high_coupling (11 deps, 29 dependents)); urgent_pacing_accelerator
- Reader next_action: `modularize adjudicator: split god-constructs (1 god-files >1500 lines, 0 god-funcs >200 lines) into focused single-responsibility units`

## Work items (in order)
1. Split the named god-constructs in `internal/adjudicator` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane adjudicator --json
   ```

## Boundaries
- Edit ONLY files under `internal/adjudicator/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/adjudicator/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/adjudicator/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `adjudicator` (see witness command).
- Committed by explicit path with trailer `(fak internal/adjudicator)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
