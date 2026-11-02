<!-- fak-debtwave-key: debtwave2-pkg-sysproc -->
# Debt lane sysproc: advance pkg/sysproc from integrated (6) toward target 8 (#debtwave2-pkg-sysproc)

```routing
lane: pkg/sysproc
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `pkg/sysproc` · Criticality: enabling · Weight: 2
- Maturity: **6 / 8** · Debt principal: 4 pts · Carrying cost: 0.96 pts · Total debt: 4.96 pts
- Evidence: 4 file(s), 152 code lines, has tests=True, integrated=True, comment_ratio=0.388
- Health: degraded (score 0.7); issues: excess_comments, unproven_runtime, unbenchmarked
- Interest drivers: enabling_infrastructure; moderate_blast_radius (8 dependents); excess_comment_bloat (38.8% comments); unbenchmarked_enabling_perf_hazard (+0.04)
- Reader next_action: `dogfood sysproc: execute real runtime path and capture passing proof`

## Work items (in order)
1. Prune excess comment bloat (38.8% comment ratio) in `pkg/sysproc`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane sysproc --json
   ```

## Boundaries
- Edit ONLY files under `pkg/sysproc/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./pkg/sysproc/... -count=1` green in a fresh run (forward slashes).
- `go vet ./pkg/sysproc/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `sysproc` (see witness command).
- Committed by explicit path with trailer `(fak pkg/sysproc)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
