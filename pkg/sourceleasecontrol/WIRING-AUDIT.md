# Wiring audit — pkg/sourceleasecontrol (debtwave2-pkg-sourceleasecontrol)

Verdict: **ABSENT** — no production call path invokes this package. Recorded per ticket
work item 1 ("record ABSENT-with-reason and land the audit evidence only — do NOT force
new wiring"); the ticket boundary forbids edits outside `pkg/sourceleasecontrol/`, so no
connection was forced.

Date: 2026-11-02 · Audit: read-only, dual-repo (fak worktree + fak-private companion)

Re-verified: 2026-11-03 (twice) — first against trunk tip f308e6e0e1 (95 commits ahead
of the audit base), then again at trunk tip e72e5bf5f5 after the lane's own commit
landed; the only intervening trunk commit (`e72e5bf5f5`, internal/sessionref tests)
touches no verdict-relevant path, and the out-of-package history check
(`git log --all -p -S 'github.com/anthony-chaudhary/fak/pkg/sourceleasecontrol' --
':!pkg/sourceleasecontrol'`) stayed empty both times, so ABSENT stands.

## Findings

| Check | Result | Evidence |
|---|---|---|
| Non-test importers in fak | none | `rg -n "anthony-chaudhary/fak/pkg/sourceleasecontrol" --glob '*.go'` → hits only inside `pkg/sourceleasecontrol/` (its own external test) |
| Tool name `fak_source_lease_control` referenced by any verb/registry table | none | single occurrence: `control.go` (`Tool` constant); no cmd/, internal/, toml, json, or md reference |
| Runtime proof registered (dogfooded source) | absent | `internal/maturity/runtime-proofs.json` exists, no `sourceleasecontrol` lane entry; `platform/maturity/runtime-proofs.json` does not exist |
| Benchmark authority entry | absent | `BENCHMARK-AUTHORITY.md` has no `sourceleasecontrol` mention (lane now carries a substantive `Benchmark*` in-package instead) |
| Companion repo callers | none | `rg sourceleasecontrol fak-private` → zero tracked-file references; generated `.fak/feature-index.json` records only test-only self-imports |
| Git history ever wired it | never | `git log --all -S 'github.com/anthony-chaudhary/fak/pkg/sourceleasecontrol' -- '*.go'` → only in-package adds/modifies (`53dd08e95aa`, `8c5f6984b4c`) plus later orphaned-lane deletions; no out-of-package import line in any commit |
| Lane registration | present (tracking only) | `dos.toml` lane list + `sourceleasecontrol = ["pkg/sourceleasecontrol/**"]` tree |

## Reason (why ABSENT is expected, not an accident)

The package is deliberately host-facing: `Authority` implementations and lease definitions
live in protected host code (see package doc in `control.go`). Public runtime pins are
explicitly *not* authority. The production caller is therefore a host integration point
that does not exist yet in either repository — the leaf was created by `fak new-leaf`
(`dos.toml` "new-leaf:lane" marker) ahead of its first consumer.

## Effect on debt-lane reader

While `integrated=false`, `internal/debtlane/curve.go` caps this lane at maturity 5.0
("tested ceiling", curve.go:80), and interest surcharges for unbenchmarked/unproven enabling
lanes apply only when integrated (interest.go:128, :138). Concrete arithmetic: the lane's
pre-cap score is 5.5 (base 4.0 + multi-test-file 0.5 + clean-comments 0.5 + bounded-blast-radius
0.5; +1.0 benchmarked once this commit lands, i.e. 6.5) — capped to 5.0 either way. Interest
stays at the 0.08 enabling base (no dependents, no comment bloat, no modularity deficit), so
total_debt = (8−5)×2 × 1.08 = 6.48 both before and after. Clearing `unbenchmarked` here
removes one degraded-health issue (score 0.75 → 0.80) but cannot move maturity or total_debt
until a host wires the package.

## Next checkable step (out of this ticket's boundary)

Add one importing call site under `cmd/**` or `internal/**` (or the host facade in
fak-private) that resolves an `Authority` and drives `Prepare`/`Execute`; then re-run
`fak debt-lanes --workspace . --target-repo fak --lane sourceleasecontrol --json` and confirm
`integrated=true` lifts the lane past the tested ceiling.

## Closure receipt (2026-11-03, debtwave2-pkg-sourceleasecontrol)

Final witness (`go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane
sourceleasecontrol --json`, tree af3e926efe / trunk e72e5bf5f5):

| Field | Value |
|---|---|
| maturity | 5.0 / 8.0, rung `tested` (gap 3.0) |
| total_debt | 6.48 (principal 6.0 + carrying 0.48 @ rate 0.08, driver `enabling_infrastructure`) |
| health | degraded, score 0.80 (was 0.75 pre-benchmark), issues `[disconnected_wiring, unproven_runtime]` |
| evidence | integrated=false, dogfooded=false, benchmarked=true, has_tests=true, test_files_count=2, dependents_count=0, excess_comments=false |

Done-condition status: `go test ./pkg/sourceleasecontrol/... -count=1` PASS; `go vet`
clean; gofmt clean; all three `Benchmark*` execute green. Maturity did not rise and
total_debt did not fall, and neither can from this boundary: pre-cap maturity is 6.5
(base 4.0 + multi-test 0.5 + benchmarked 1.0 + clean-comments 0.5 + bounded-blast-radius
0.5), hard-capped at 5.0 while `integrated=false` (the debtlane curve's unintegrated
ceiling), and the interest rate sits at the 0.08 enabling floor because every surcharge
is conditional on integration or on an already-clean evidence flag. Total debt is
therefore invariant at (8−5)×2×1.08 = 6.48 under any in-boundary edit. Work item 1's
mandated branch applies: ABSENT-with-reason recorded, audit evidence landed, no forced
wiring.

Projected effect of the next step (out of this ticket's boundary): landing one
importing call site sets `integrated=true`, which lifts maturity 5.0 → 8.0 (target
reached, rung `integrated`), drops total_debt 6.48 → 0 (gap 0 ⇒ quiescent), and raises
health 0.80 → 0.95 with only `unproven_runtime` open — assuming comment hygiene stays
clean and dependents stay ≤ 25.
