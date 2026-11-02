<!-- fak-debtwave-key: debtwave2-internal-sessionref -->
# Debt lane sessionref: advance internal/sessionref from prototyped (2.5) toward target 8 (#debtwave2-internal-sessionref)

```routing
lane: internal/sessionref
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/sessionref` · Criticality: enabling · Weight: 2
- Maturity: **2.5 / 8** · Debt principal: 11 pts · Carrying cost: 2.53 pts · Total debt: 13.53 pts
- Evidence: 3 file(s), 272 code lines, has tests=False, integrated=True, comment_ratio=0.301
- Health: degraded (score 0.5); issues: missing_tests, high_maturity_gap, unproven_runtime, unbenchmarked
- Interest drivers: enabling_infrastructure; integrated_untested_hazard; unbenchmarked_enabling_perf_hazard (+0.04); unproven_runtime_enabling_perf_hazard (+0.03)
- Reader next_action: `test sessionref: add unit tests (*_test.go) covering internal\sessionref to climb past 2.0`

## Work items (in order)
1. Add meaningful unit tests (`*_test.go`) covering the core behavior of `internal/sessionref`: table-driven where natural, assert closed contracts/sentinels (`errors.Is`, typed sentinels), never error prose; cover hot paths and rejection/fallback branches. Tests must exercise real logic, not mock echoes.
2. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane sessionref --json
   ```

## Boundaries
- Edit ONLY files under `internal/sessionref/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/sessionref/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/sessionref/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `sessionref` (see witness command).
- Committed by explicit path with trailer `(fak internal/sessionref)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
