<!-- fak-debtwave-key: debtwave2-internal-nativeperfbackend -->
# Debt lane nativeperfbackend: advance internal/nativeperfbackend from tested (5) toward target 8 (#debtwave2-internal-nativeperfbackend)

```routing
lane: internal/nativeperfbackend
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/nativeperfbackend` · Criticality: enabling · Weight: 2
- Maturity: **5 / 8** · Debt principal: 6 pts · Carrying cost: 1.26 pts · Total debt: 7.26 pts
- Evidence: 1 file(s), 245 code lines, has tests=True, integrated=False, comment_ratio=0.065
- Health: critical (score 0.5); issues: excess_comments, disconnected_wiring, modularity_deficit, model_hardcoding
- Interest drivers: enabling_infrastructure; excess_comment_bloat (6.5% comments); modularity_surcharge (+0.08: model_hardcoding (contract.go: qwen), model_hardcoding (contract.go: qwen))
- Reader next_action: `integrate nativeperfbackend: connect into production command or registration graph`

## Work items (in order)
1. Prune excess comment bloat (6.5% comment ratio) in `internal/nativeperfbackend`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. READ-ONLY wiring audit first: from the repo root run `rg -n "github.com/anthony-chaudhary/<module>" --glob '*.go' -g '!*_test.go'` (and grep the verb/registry tables for this package) to establish whether a production call path actually invokes it (presence != invokability). Record the finding. If disconnected and a minimal connection fits <= 3 files, wire it; otherwise record ABSENT-with-reason and land the audit evidence only — do NOT force new wiring.
3. Split the named god-constructs in `internal/nativeperfbackend` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
4. Remove model-specific hardcoding in `internal/nativeperfbackend`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
5. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane nativeperfbackend --json
   ```

## Boundaries
- Edit ONLY files under `internal/nativeperfbackend/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/nativeperfbackend/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/nativeperfbackend/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `nativeperfbackend` (see witness command).
- Committed by explicit path with trailer `(fak internal/nativeperfbackend)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
