<!-- fak-debtwave-key: debtwave2-internal-perfledger -->
# Debt lane perfledger: advance internal/perfledger from integrated (6.5) toward target 8 (#debtwave2-internal-perfledger)

```routing
lane: internal/perfledger
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/perfledger` · Criticality: enabling · Weight: 2
- Maturity: **6.5 / 8** · Debt principal: 3 pts · Carrying cost: 0.69 pts · Total debt: 3.69 pts
- Evidence: 6 file(s), 1719 code lines, has tests=True, integrated=True, comment_ratio=0.13
- Health: critical (score 0.65); issues: unproven_runtime, unbenchmarked, modularity_deficit
- Interest drivers: enabling_infrastructure; modularity_surcharge (+0.08: god_func (Summarize: 201 lines)); unbenchmarked_enabling_perf_hazard (+0.04); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `dogfood perfledger: execute real runtime path and capture passing proof`

## Work items (in order)
1. Split the named god-constructs in `internal/perfledger` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane perfledger --json
   ```

## Boundaries
- Edit ONLY files under `internal/perfledger/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/perfledger/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/perfledger/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `perfledger` (see witness command).
- Committed by explicit path with trailer `(fak internal/perfledger)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
