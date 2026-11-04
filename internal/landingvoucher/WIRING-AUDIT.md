# Wiring audit - internal/landingvoucher

- ticket: debtwave2-internal-landingvoucher
- audit date: 2026-11-03
- lane: internal/landingvoucher

## Method

The audit re-ran the debt-lanes census probe for external call paths into this
lane, from the repository root:

```bash
rg -n "github.com/anthony-chaudhary/fak/internal/landingvoucher" --glob '*.go' -g '!*_test.go'  # zero hits outside the package
rg -n "landingvoucher" cmd/ --glob '*.go' -g '!*_test.go'                                       # exit 1, zero hits
rg -n "landingvoucher" dos.toml                                                                 # two metadata lines only
rg -n '"landingvoucher"' internal/ -g '*_test.go'                                               # test-file mentions only
```

Test-file and doc mentions were inventoried separately so the record names every
external reference, not just the import-path scan.

The first probe excludes `*_test.go` files, so the executable contract's own
string literal in `wiring_audit_test.go` does not count as a reference. The
fourth probe likewise matches the contract's own `docPath` literal in
`wiring_audit_test.go`; test files are outside the reference set by design,
and neither self-match is a call path.

Every `.go` file was searched for the full import path; `cmd/` was searched for
any mention of the lane name; `dos.toml` was searched to classify the registry
references. The executable counterpart of this record is
`wiring_audit_test.go` (TestWiringAuditContract), which recomputes the reference
set on every run and cross-checks the verdict line below.

## Findings

- Zero import declarations of `github.com/anthony-chaudhary/fak/internal/landingvoucher`
  in any non-test `.go` file outside the package itself.
- Zero mentions of `landingvoucher` anywhere under `cmd/`, including
  `cmd/fak/verbs_gen.go` (no verb or generated registry-table entry).
- Every external mention that does exist is classified, and none is a call path:
  - `dos.toml:345` (leaf-name registry list) and `dos.toml:1222`
    (`landingvoucher = ["internal/landingvoucher/**"]` path mapping) — DOS metadata.
  - `internal/architest/architest_test.go:779` (`"landingvoucher": 1` tier table) —
    a test-file mention, excluded from the import scan by design.
  - `docs/releases/v0.54.0.md` — release-note prose.
  - `docs/tickets/debtwave2-internal-landingvoucher/` — this ticket's spec.
- Census baseline agrees: `integrated=false dependents=0 dogfooded=false benchmarked=false`,
  with health degraded on issues `disconnected_wiring`, `unproven_runtime`,
  `unbenchmarked`.

verdict: ABSENT

## Reason

The ticket fence allows edits only under `internal/landingvoucher/`. The minimal
out-of-lane seam - a read-only advisory `fak landing-voucher` verb that mints the
voucher or emits the closed refusal token LANDING_INVARIANT_REFUSED - needs five
files: `cmd/fak/landingvoucher.go` (new verb), the `cmd/fak/main.go` dispatch case,
the `cmd/fak/verbs_gen.go` generated index row, and the `internal/devindex`
`tiers.go` + `verbs.go` catalog entries. That exceeds work item 1's <= 3-file
minimal-connection budget and the fence forbids it outright, so per work item 1
the finding is recorded instead of forcing new wiring.

## Structural note

While the lane stays disconnected, `internal/debtlane/curve.go` caps maturity at
the 5.0 tested ceiling and `internal/debtlane/interest.go` pins the enabling base
rate at 0.08 (its unbenchmarked/unproven surcharges only apply once
`Integrated=true`), so the census figures (maturity 5.0, total_debt 6.48) are
structurally invariant under any in-fence edit. No in-lane change can move them;
only a real call path out of the lane can. What the in-fence pass does move:
substantive `Benchmark*` functions flip `benchmarked=true`, retiring the
`unbenchmarked` health issue (health score 0.75 -> 0.80).

## Next step

An operator-approved follow-on ticket lifting the lane fence, landing the 5-file
`fak landing-voucher` verb seam described in the Reason section. After that
lands, this record flips to `verdict: WIRED` and lists the seam files.
