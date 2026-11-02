<!-- fak-debtwave-key: debtwave2-internal-gateway -->
# Debt lane gateway: advance internal/gateway from benchmarked (7.5) toward target 10 (#debtwave2-internal-gateway)

```routing
lane: internal/gateway
tracker: pub
```

## Lane facts (baseline 2026-11-02, `fak debt-lanes` census)
- Repo: **fak** · Unit of work: `internal/gateway` · Criticality: core · Weight: 3
- Maturity: **7.5 / 10** · Debt principal: 7.5 pts · Carrying cost: 2.63 pts · Total debt: 10.13 pts
- Evidence: 331 file(s), 88631 code lines, has tests=True, integrated=True, comment_ratio=0.251
- Health: critical (score 0.55); issues: excess_comments, critical_interest, modularity_deficit, model_hardcoding
- Interest drivers: core_runtime_criticality; moderate_blast_radius (14 dependents); excess_comment_bloat (25.1% comments); modularity_surcharge (+0.10: god_file (admission.go: 1729 lines), model_hardcoding (anthropic_messages.go: qwen), model_hardcoding (anthropic_messages.go: qwen), model_hardcoding (anthropic_messages.go: qwen), model_hardcoding (anthropic_messages.go: qwen) (+485 more))
- Reader next_action: `clean gateway: prune excess comment bloat and formulaic noise (25.1% comment ratio); make code self-documenting and verify invariants in tests`

## Work items (in order)
1. Prune excess comment bloat (25.1% comment ratio) in `internal/gateway`: remove formulaic "Contract:"/"Invariant:" header noise and duplicated doc chatter; KEEP every comment that explains a non-obvious why. Target comment ratio <= 35% without losing required invariant forms.
2. Split the named god-constructs in `internal/gateway` (god-file > 1500 lines / god-function > 200 lines) into cohesive units with narrow interfaces. Behavior stays byte-identical; scoped tests green before and after. One split per commit when the construct is large.
3. Remove model-specific hardcoding in `internal/gateway`: replace hardcoded model names with config/registry lookup at the seam. Keep observable behavior identical; scoped tests green.
4. Run the witness command and confirm maturity rose or total debt fell; if it did not register, check Test*/Benchmark* naming before reporting done.
   ```
   go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane gateway --json
   ```

## Boundaries
- Edit ONLY files under `internal/gateway/` in the public `fak` repo. Never touch go.mod, go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- Keep comment hygiene clean: remove formulaic "Contract:"/"Invariant:" header noise only when listed as an issue; never delete a comment that explains a non-obvious why.

## Done condition
- `go test ./internal/gateway/... -count=1` green in a fresh run (forward slashes).
- `go vet ./internal/gateway/...` clean.
- Debt-lanes reader shows maturity up or total_debt down for lane `gateway` (see witness command).
- Committed by explicit path with trailer `(fak internal/gateway)` and DCO sign-off, subject = so-what, line 2 = why, line 3 = witness. The public pre-commit hook runs the secret leak audit and boundary gates automatically; do not bypass them.
