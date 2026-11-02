<!-- fak-debtwave-key: debtwave2-pkg-gpubinding -->
# Debt lane gpubinding: advance pkg/gpubinding from tested (5) toward target 8 (#debtwave2-pkg-gpubinding)

```routing
lane: pkg/gpubinding
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `pkg/gpubinding` · Criticality: enabling · Weight: 2
- Maturity: **5 / 8** · Debt principal: 6 pts · Carrying cost: 0.78 pts · Total debt: 6.78 pts
- Evidence: 1 file(s), 75 code lines, has tests=True, integrated=False, comment_ratio=0.4
- Health: degraded (score 0.7); issues: excess_comments, disconnected_wiring, unproven_runtime
- Interest drivers: enabling_infrastructure; excess_comment_bloat (40.0% comments)
- Reader next_action: `integrate gpubinding: connect into production command or registration graph`

## Work items (in order)
1. Prune excess comment bloat (40.0% comment ratio) in `pkg/gpubinding`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. READ-ONLY wiring audit first: from the repo root run `rg -n "github.com/anthony-chaudhary/<module>" --glob '*.go' -g '!*_test.go'` (and grep the verb/registry tables for this package) to establish whether a production call path actually invokes it (presence != invokability). Record the finding. If disconnected and a minimal connection fits <= 3 files, wire it; otherwise record ABSENT-with-reason and land the audit evidence only — do NOT force new wiring.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane gpubinding --json
   ```

## Boundaries
- Edit ONLY files under `pkg/gpubinding/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./pkg/gpubinding/... -count=1` green in a fresh run (forward slashes).
- `go vet ./pkg/gpubinding/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `gpubinding` (see witness command).
- Committed by explicit path with trailer `(fak pkg/gpubinding)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
