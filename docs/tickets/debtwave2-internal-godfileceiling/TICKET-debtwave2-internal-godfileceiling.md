<!-- fak-debtwave-key: debtwave2-internal-godfileceiling -->
# Debt lane godfileceiling: advance internal/godfileceiling from tested (5) toward target 8 (#debtwave2-internal-godfileceiling)

```routing
lane: internal/godfileceiling
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/godfileceiling` · Criticality: enabling · Weight: 2
- Maturity: **5 / 8** · Debt principal: 6 pts · Carrying cost: 0.96 pts · Total debt: 6.96 pts
- Evidence: 3 file(s), 273 code lines, has tests=True, integrated=False, comment_ratio=0.278
- Health: critical (score 0.6); issues: disconnected_wiring, modularity_deficit, model_hardcoding
- Interest drivers: enabling_infrastructure; modularity_surcharge (+0.08: model_hardcoding (baseline.go: llama))
- Reader next_action: `integrate godfileceiling: connect into production command or registration graph`

## Work items (in order)
1. READ-ONLY wiring audit first: from the repo root run `rg -n "github.com/anthony-chaudhary/<module>" --glob '*.go' -g '!*_test.go'` (and grep the verb/registry tables for this package) to establish whether a production call path actually invokes it (presence != invokability). Record the finding. If disconnected and a minimal connection fits <= 3 files, wire it; otherwise record ABSENT-with-reason and land the audit evidence only — do NOT force new wiring.
2. Split the named god-constructs in `internal/godfileceiling` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
3. Remove model-specific hardcoding in `internal/godfileceiling`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
4. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane godfileceiling --json
   ```

## Boundaries
- Edit ONLY files under `internal/godfileceiling/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/godfileceiling/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/godfileceiling/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `godfileceiling` (see witness command).
- Committed by explicit path with trailer `(fak internal/godfileceiling)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
