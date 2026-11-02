<!-- fak-debtwave-key: debtwave2-internal-engine -->
# Debt lane engine: advance internal/engine from benchmarked (8.5) toward target 10 (#debtwave2-internal-engine)

```routing
lane: internal/engine
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/engine` · Criticality: core · Weight: 3
- Maturity: **8.5 / 10** · Debt principal: 4.5 pts · Carrying cost: 1.49 pts · Total debt: 5.99 pts
- Evidence: 36 file(s), 9437 code lines, has tests=True, integrated=True, comment_ratio=0.255
- Health: critical (score 0.65); issues: critical_interest, modularity_deficit, model_hardcoding
- Interest drivers: core_runtime_criticality; moderate_blast_radius (9 dependents); modularity_surcharge (+0.10: model_hardcoding (cacheevents.go: qwen), god_file (continuous_batcher.go: 1561 lines), model_hardcoding (continuous_batcher.go: qwen), model_hardcoding (continuous_batcher.go: qwen), model_hardcoding (continuous_batcher.go: qwen) (+58 more)); urgent_pacing_accelerator
- Reader next_action: `modularize engine: decouple model-specific hardcoding (60 matches) and split god-constructs (1 god-files >1500 lines, 1 god-funcs >200 lines)`

## Work items (in order)
1. Split the named god-constructs in `internal/engine` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Remove model-specific hardcoding in `internal/engine`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane engine --json
   ```

## Boundaries
- Edit ONLY files under `internal/engine/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/engine/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/engine/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `engine` (see witness command).
- Committed by explicit path with trailer `(fak internal/engine)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
