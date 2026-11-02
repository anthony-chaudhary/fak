<!-- fak-debtwave-key: debtwave2-internal-model -->
# Debt lane model: advance internal/model from dogfooded (7) toward target 10 (#debtwave2-internal-model)

```routing
lane: internal/model
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/model` · Criticality: core · Weight: 3
- Maturity: **7 / 10** · Debt principal: 9 pts · Carrying cost: 3.15 pts · Total debt: 12.15 pts
- Evidence: 710 file(s), 182020 code lines, has tests=True, integrated=True, comment_ratio=0.194
- Health: critical (score 0.55); issues: excess_comments, critical_interest, modularity_deficit
- Interest drivers: core_runtime_criticality; high_blast_radius (28 dependents); excess_comment_bloat (19.4% comments); modularity_surcharge (+0.08: god_func (tokenHALOutput: 369 lines), god_file (kv.go: 1568 lines), god_func (stepRoundImpl: 382 lines), god_func (stepRoundTreeLocked: 268 lines), god_func (Qwen35MetalForwardSequence: 247 lines) (+36 more))
- Reader next_action: `clean model: prune excess comment bloat and formulaic noise (19.4% comment ratio); make code self-documenting and verify invariants in tests`

## Work items (in order)
1. Prune excess comment bloat (19.4% comment ratio) in `internal/model`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Split the named god-constructs in `internal/model` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
3. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane model --json
   ```

## Boundaries
- Edit ONLY files under `internal/model/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/model/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/model/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `model` (see witness command).
- Committed by explicit path with trailer `(fak internal/model)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
