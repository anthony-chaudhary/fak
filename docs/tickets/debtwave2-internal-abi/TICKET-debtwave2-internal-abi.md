<!-- fak-debtwave-key: debtwave2-internal-abi -->
# Debt lane abi: advance internal/abi from benchmarked (7.5) toward target 10 (#debtwave2-internal-abi)

```routing
lane: internal/abi
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/abi` · Criticality: core · Weight: 3
- Maturity: **7.5 / 10** · Debt principal: 7.5 pts · Carrying cost: 2.4 pts · Total debt: 9.9 pts
- Evidence: 15 file(s), 3020 code lines, has tests=True, integrated=True, comment_ratio=0.453
- Health: critical (score 0.7); issues: excess_comments, critical_interest
- Interest drivers: core_runtime_criticality; high_blast_radius (101 dependents); excess_comment_bloat (45.3% comments); urgent_pacing_accelerator
- Reader next_action: `clean abi: prune excess comment bloat and formulaic noise (45.3% comment ratio); make code self-documenting and verify invariants in tests`

## Work items (in order)
1. Prune excess comment bloat (45.3% comment ratio) in `internal/abi`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane abi --json
   ```

## Boundaries
- Edit ONLY files under `internal/abi/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/abi/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/abi/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `abi` (see witness command).
- Committed by explicit path with trailer `(fak internal/abi)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
