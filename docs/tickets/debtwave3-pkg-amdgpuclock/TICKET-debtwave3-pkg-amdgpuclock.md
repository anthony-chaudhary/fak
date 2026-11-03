<!-- fak-debtwave-key: debtwave3-pkg-amdgpuclock -->
# Debt lane amdgpuclock: advance pkg/amdgpuclock from integrated (6.5) toward target 8 (#debtwave3-pkg-amdgpuclock)

```routing
lane: pkg/amdgpuclock
tracker: pub
```

## Lane facts (baseline 2026-11-03, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `pkg/amdgpuclock` · Criticality: enabling · Weight: 2
- Maturity: **6.5 / 8** · Debt principal: 3 pts · Carrying cost: 0.45 pts · Total debt: 3.45 pts
- Evidence: 1 file(s), 232 code lines, has tests=true, integrated=true, comment_ratio=0.034
- Health: degraded (score 0.9); issues: unproven_runtime, unbenchmarked
- Interest drivers: enabling_infrastructure; unbenchmarked_enabling_perf_hazard (+0.04); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `dogfood amdgpuclock: execute real runtime path and capture passing proof`

## Work items (in order)
1. Prove the runtime: execute the applicable workload through the ordinary supported entrypoint for `pkg/amdgpuclock` and record the real result as a runtime-proof entry (private: platform/maturity/runtime-proofs.json; public: the lane's registered proof route). CLI help establishes command wiring only, not integration or production qualification.
2. Add substantive benchmarks (Benchmark*) for the hot paths of `pkg/amdgpuclock` exercising production operations on a meaningful workload; empty/stub benchmarks and comment-ratio edits do not establish maturity.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane amdgpuclock --json
   ```

## Boundaries
- Edit ONLY files under `pkg/amdgpuclock/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./pkg/amdgpuclock/... -count=1` green in a fresh run (forward slashes).
- `go vet ./pkg/amdgpuclock/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `amdgpuclock` (see witness command).
- Committed by explicit path with trailer `(fak pkg/amdgpuclock)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
