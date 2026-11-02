<!-- fak-debtwave-key: debtwave2-internal-architest -->
# Debt lane architest: advance internal/architest from benchmarked (8) toward target 10 (#debtwave2-internal-architest)

```routing
lane: internal/architest
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/architest` · Criticality: core · Weight: 3
- Maturity: **8 / 10** · Debt principal: 6 pts · Carrying cost: 1.44 pts · Total debt: 7.44 pts
- Evidence: 1 file(s), 52 code lines, has tests=True, integrated=True, comment_ratio=0.981
- Health: degraded (score 0.8); issues: excess_comments
- Interest drivers: core_runtime_criticality; excess_comment_bloat (98.1% comments); urgent_pacing_accelerator
- Reader next_action: `clean architest: prune excess comment bloat and formulaic noise (98.1% comment ratio); make code self-documenting and verify invariants in tests`

## Work items (in order)
1. Prune excess comment bloat (98.1% comment ratio) in `internal/architest`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane architest --json
   ```

## Boundaries
- Edit ONLY files under `internal/architest/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/architest/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/architest/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `architest` (see witness command).
- Committed by explicit path with trailer `(fak internal/architest)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
