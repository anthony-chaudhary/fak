<!-- fak-debtwave-key: debtwave3-internal-grammarintegration -->
# Debt lane grammarintegration: advance internal/grammarintegration from proposed (0) toward target 8 (#debtwave3-internal-grammarintegration)

```routing
lane: internal/grammarintegration
tracker: pub
```

## Lane facts (baseline 2026-11-03, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/grammarintegration` · Criticality: enabling · Weight: 2
- Maturity: **0 / 8** · Debt principal: 16 pts · Carrying cost: 1.28 pts · Total debt: 17.28 pts
- Evidence: 0 file(s), 0 code lines, has tests=true, integrated=false, comment_ratio=0
- Health: degraded (score 0.75); issues: disconnected_wiring, high_maturity_gap, unproven_runtime, unbenchmarked
- Interest drivers: enabling_infrastructure
- Reader next_action: `implement grammarintegration: add initial implementation in internal\grammarintegration`

## Work items (in order)
1. READ-ONLY wiring audit first: from the repo root run `rg -n "github.com/anthony-chaudhary/fak/internal/grammarintegration" --glob '*.go' -g '!*_test.go'` (and grep the verb/registry tables for this package) to establish whether a production call path actually invokes it (presence != invokability). Record the finding. If disconnected and a minimal connection fits <= 3 files, wire it; otherwise record ABSENT-with-reason and land the audit evidence only — do NOT force new wiring.
2. Advance `internal/grammarintegration` toward its target maturity following the reader next_action in the Lane facts section; pick the smallest witnessed gap-closing step first.
3. Prove the runtime: execute the applicable workload through the ordinary supported entrypoint for `internal/grammarintegration` and record the real result as a runtime-proof entry (private: platform/maturity/runtime-proofs.json; public: the lane's registered proof route). CLI help establishes command wiring only, not integration or production qualification.
4. Add substantive benchmarks (Benchmark*) for the hot paths of `internal/grammarintegration` exercising production operations on a meaningful workload; empty/stub benchmarks and comment-ratio edits do not establish maturity.
5. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane grammarintegration --json
   ```

## Boundaries
- Edit ONLY files under `internal/grammarintegration/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/grammarintegration/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/grammarintegration/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `grammarintegration` (see witness command).
- Committed by explicit path with trailer `(fak internal/grammarintegration)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
