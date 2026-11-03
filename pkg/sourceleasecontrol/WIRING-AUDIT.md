# Wiring audit — pkg/sourceleasecontrol (debtwave2-pkg-sourceleasecontrol)

Verdict: **ABSENT** — no production call path invokes this package. Recorded per ticket
work item 1 ("record ABSENT-with-reason and land the audit evidence only — do NOT force
new wiring"); the ticket boundary forbids edits outside `pkg/sourceleasecontrol/`, so no
connection was forced.

Date: 2026-11-02 · Audit: read-only, dual-repo (fak worktree + fak-private companion)

Re-verified: 2026-11-03 — all seven checks independently re-run against a trunk 95 commits
ahead of the audit base (trunk tip f308e6e0e1); the out-of-package history check
(`git log --all -p -S 'github.com/anthony-chaudhary/fak/pkg/sourceleasecontrol' --
':!pkg/sourceleasecontrol'`) stayed empty, so ABSENT stands.

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
