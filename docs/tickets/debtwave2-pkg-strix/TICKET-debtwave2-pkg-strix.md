<!-- fak-debtwave-key: debtwave2-pkg-strix -->
# Debt lane strix: advance pkg/strix from integrated (5.5) toward target 8 (#debtwave2-pkg-strix)

```routing
lane: pkg/strix
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `pkg/strix` · Criticality: enabling · Weight: 2
- Maturity: **5.5 / 8** · Debt principal: 5 pts · Carrying cost: 1 pts · Total debt: 6 pts
- Evidence: 1 file(s), 32 code lines, has tests=True, integrated=True, comment_ratio=0.375
- Health: degraded (score 0.7); issues: excess_comments, unproven_runtime, unbenchmarked
- Interest drivers: enabling_infrastructure; excess_comment_bloat (37.5% comments); unbenchmarked_enabling_perf_hazard (+0.04); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `dogfood strix: execute real runtime path and capture passing proof`

## Work items (in order)
1. Prune excess comment bloat (37.5% comment ratio) in `pkg/strix`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane strix --json
   ```

## Boundaries
- Edit ONLY files under `pkg/strix/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./pkg/strix/... -count=1` green in a fresh run (forward slashes).
- `go vet ./pkg/strix/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `strix` (see witness command).
- Committed by explicit path with trailer `(fak pkg/strix)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
