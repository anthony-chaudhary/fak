<!-- fak-debtwave-key: debtwave2-internal-servewarmup -->
# Debt lane servewarmup: advance internal/servewarmup from integrated (5.5) toward target 8 (#debtwave2-internal-servewarmup)

```routing
lane: internal/servewarmup
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/servewarmup` · Criticality: enabling · Weight: 2
- Maturity: **5.5 / 8** · Debt principal: 5 pts · Carrying cost: 1 pts · Total debt: 6 pts
- Evidence: 1 file(s), 33 code lines, has tests=True, integrated=True, comment_ratio=0.485
- Health: degraded (score 0.7); issues: excess_comments, unproven_runtime, unbenchmarked
- Interest drivers: enabling_infrastructure; excess_comment_bloat (48.5% comments); unbenchmarked_enabling_perf_hazard (+0.04); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `dogfood servewarmup: execute real runtime path and capture passing proof`

## Work items (in order)
1. Prune excess comment bloat (48.5% comment ratio) in `internal/servewarmup`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane servewarmup --json
   ```

## Boundaries
- Edit ONLY files under `internal/servewarmup/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/servewarmup/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/servewarmup/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `servewarmup` (see witness command).
- Committed by explicit path with trailer `(fak internal/servewarmup)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
