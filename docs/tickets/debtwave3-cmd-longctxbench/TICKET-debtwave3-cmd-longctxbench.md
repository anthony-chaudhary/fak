<!-- fak-debtwave-key: debtwave3-cmd-longctxbench -->
# Debt lane longctxbench: advance cmd/longctxbench from prototyped (2.5) toward target 8 (#debtwave3-cmd-longctxbench)

```routing
lane: cmd/longctxbench
tracker: pub
```

## Lane facts (baseline 2026-11-03, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `cmd/longctxbench` · Criticality: enabling · Weight: 2
- Maturity: **2.5 / 8** · Debt principal: 11 pts · Carrying cost: 0 pts · Total debt: 11 pts
- Evidence: 1 file(s), 112 code lines, has tests=false, integrated=true, comment_ratio=0
- Health: degraded (score 0.6); issues: missing_tests, high_maturity_gap, unproven_runtime, unbenchmarked
- Interest drivers: none
- Reader next_action: ``

## Work items (in order)
1. Add meaningful unit tests (`*_test.go`) covering the core behavior of `cmd/longctxbench`: table-driven where natural, assert closed contracts/sentinels (`errors.Is`, typed sentinels), never error prose; cover hot paths and rejection/fallback branches. Tests must exercise real logic, not mock echoes.
2. Advance `cmd/longctxbench` toward its target maturity following the reader next_action in the Lane facts section; pick the smallest witnessed gap-closing step first.
3. Prove the runtime: execute the applicable workload through the ordinary supported entrypoint for `cmd/longctxbench` and record the real result as a runtime-proof entry (private: platform/maturity/runtime-proofs.json; public: the lane's registered proof route). CLI help establishes command wiring only, not integration or production qualification.
4. Add substantive benchmarks (Benchmark*) for the hot paths of `cmd/longctxbench` exercising production operations on a meaningful workload; empty/stub benchmarks and comment-ratio edits do not establish maturity.
5. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane longctxbench --json
   ```

## Boundaries
- Edit ONLY files under `cmd/longctxbench/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./cmd/longctxbench/... -count=1` green in a fresh run (forward slashes).
- `go vet ./cmd/longctxbench/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `longctxbench` (see witness command).
- Committed by explicit path with trailer `(fak cmd/longctxbench)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
