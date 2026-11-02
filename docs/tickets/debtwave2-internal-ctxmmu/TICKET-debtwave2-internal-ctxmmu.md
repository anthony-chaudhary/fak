<!-- fak-debtwave-key: debtwave2-internal-ctxmmu -->
# Debt lane ctxmmu: advance internal/ctxmmu from benchmarked (8) toward target 10 (#debtwave2-internal-ctxmmu)

```routing
lane: internal/ctxmmu
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/ctxmmu` · Criticality: core · Weight: 3
- Maturity: **8 / 10** · Debt principal: 6 pts · Carrying cost: 2.1 pts · Total debt: 8.1 pts
- Evidence: 47 file(s), 16169 code lines, has tests=True, integrated=True, comment_ratio=0.161
- Health: critical (score 0.65); issues: critical_interest, modularity_deficit, model_hardcoding
- Interest drivers: core_runtime_criticality; high_blast_radius (28 dependents); modularity_surcharge (+0.08: model_hardcoding (checkpoint.go: cohere), model_hardcoding (checkpoint.go: cohere), model_hardcoding (checkpoint.go: cohere), model_hardcoding (checkpoint.go: cohere), model_hardcoding (fork.go: cohere) (+6 more)); urgent_pacing_accelerator
- Reader next_action: `decouple ctxmmu: remove model-specific hardcoding (10 occurrences); route via model catalog or provider abstraction`

## Work items (in order)
1. Split the named god-constructs in `internal/ctxmmu` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Remove model-specific hardcoding in `internal/ctxmmu`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane ctxmmu --json
   ```

## Boundaries
- Edit ONLY files under `internal/ctxmmu/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/ctxmmu/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/ctxmmu/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `ctxmmu` (see witness command).
- Committed by explicit path with trailer `(fak internal/ctxmmu)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
