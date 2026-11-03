<!-- fak-debtwave-key: debtwave3-pkg-moecache -->
# Debt lane moecache: advance pkg/moecache from integrated (6.5) toward target 8 (#debtwave3-pkg-moecache)

```routing
lane: pkg/moecache
tracker: pub
```

## Lane facts (baseline 2026-11-03, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `pkg/moecache` · Criticality: enabling · Weight: 2
- Maturity: **6.5 / 8** · Debt principal: 3 pts · Carrying cost: 0.48 pts · Total debt: 3.48 pts
- Evidence: 1 file(s), 268 code lines, has tests=true, integrated=true, comment_ratio=0.451
- Health: degraded (score 0.75); issues: excess_comments, unproven_runtime
- Interest drivers: enabling_infrastructure; excess_comment_bloat (45.1% comments); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `dogfood moecache: execute real runtime path and capture passing proof`

## Work items (in order)
1. Prune excess comment bloat in `pkg/moecache`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms. (observed comment ratio 45.1%).
2. Prove the runtime: execute the applicable workload through the ordinary supported entrypoint for `pkg/moecache` and record the real result as a runtime-proof entry (private: platform/maturity/runtime-proofs.json; public: the lane's registered proof route). CLI help establishes command wiring only, not integration or production qualification.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane moecache --json
   ```

## Boundaries
- Edit ONLY files under `pkg/moecache/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./pkg/moecache/... -count=1` green in a fresh run (forward slashes).
- `go vet ./pkg/moecache/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `moecache` (see witness command).
- Committed by explicit path with trailer `(fak pkg/moecache)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
