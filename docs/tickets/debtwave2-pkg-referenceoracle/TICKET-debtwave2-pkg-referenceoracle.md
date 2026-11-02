<!-- fak-debtwave-key: debtwave2-pkg-referenceoracle -->
# Debt lane referenceoracle: advance pkg/referenceoracle from integrated (6.5) toward target 8 (#debtwave2-pkg-referenceoracle)

```routing
lane: pkg/referenceoracle
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `pkg/referenceoracle` · Criticality: enabling · Weight: 2
- Maturity: **6.5 / 8** · Debt principal: 3 pts · Carrying cost: 0.69 pts · Total debt: 3.69 pts
- Evidence: 3 file(s), 652 code lines, has tests=True, integrated=True, comment_ratio=0.192
- Health: critical (score 0.65); issues: unproven_runtime, unbenchmarked, modularity_deficit, model_hardcoding
- Interest drivers: enabling_infrastructure; modularity_surcharge (+0.08: model_hardcoding (legacy.go: llama), model_hardcoding (legacy.go: llama)); unbenchmarked_enabling_perf_hazard (+0.04); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `dogfood referenceoracle: execute real runtime path and capture passing proof`

## Work items (in order)
1. Split the named god-constructs in `pkg/referenceoracle` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
2. Remove model-specific hardcoding in `pkg/referenceoracle`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane referenceoracle --json
   ```

## Boundaries
- Edit ONLY files under `pkg/referenceoracle/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./pkg/referenceoracle/... -count=1` green in a fresh run (forward slashes).
- `go vet ./pkg/referenceoracle/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `referenceoracle` (see witness command).
- Committed by explicit path with trailer `(fak pkg/referenceoracle)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
