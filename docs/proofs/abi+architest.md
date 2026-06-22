---
title: "fak proof: abi spine and architest layering"
description: "Proof that fak's frozen ABI wire enums stay additive and every verdict fold orders by FoldRank over a layered, subprocess-free package DAG."
---

# A8 · abi+architest

`internal/abi` is the frozen wave-0 spine of fak — the one tree every fleet worker imports and no worker may edit except additively. It *computes* nothing numeric; what it provides is a **structural contract**: a closed, byte-stable set of wire enums (verdict kinds, status, outcome, taint, scope, ref-kind, fallback, the reason vocabulary) plus the `FoldRank` restrictiveness lattice that orders verdict folds. `internal/architest` is the machine-checker that proves the *composition* the rest of the proofs assume actually holds in the build: a layered package DAG with no upward imports, a subprocess-free hot path, and every most-restrictive-wins fold routed through `abi.FoldRank`. "Correct" for A8 is **regime A (algebraic / structural)**: the defining invariants — a total fold order, a layered DAG, a frozen additive-only wire shape — are preserved, witnessed by round-trip / invariant / structural-contract tests, not numerical parity. All witnesses below ran green natively on the macOS fleet node (go1.26 darwin/arm64) on 2026-06-20.

---

## Theorem 1 — every verdict-fold site orders by `abi.FoldRank`, a total order

**THEOREM.** Every internal package that folds a verdict chain most-restrictive-wins (`kernel`, `kvmmu`, `recall`, `agent`) orders that fold by `abi.FoldRank`, and `FoldRank` is a total function `VerdictKind → int` into a totally-ordered codomain — so no fold can silently order by raw `VerdictKind` value (which is a registration-block id, not a restrictiveness rank).

**REGIME.** A — structural contract / total order.

**PROOF.**
- *Ordering authority is total.* `abi.FoldRank` (`fak/internal/abi/registry.go:744`) returns an `int` for **every** `VerdictKind`: the constant switch maps the 6 core kinds (lines 745–758, `Allow`=0 … `Deny`=100), the snapshot lookup (line 759) covers registered kinds, and line 762 returns 100 (max, fail-closed) for any unknown kind. Every kind therefore has exactly one rank, and `int` is totally ordered, so the codomain is totally ordered by construction.
- *Every fold consults it.* The four real fold loops compare by `abi.FoldRank`, never by `.Kind`: `fak/internal/kernel/kernel.go:142` and `:204`–`:207`, `fak/internal/kvmmu/kvmmu.go:79`–`:82`, `fak/internal/recall/recall.go:353`–`:356`, `fak/internal/agent/transcript.go:77`.
- *The regression gate.* `TestFoldSitesOrderByFoldRank` (`fak/internal/architest/architest_test.go:770`) re-derives, via an AST scan (`pkgCallsSelector`, `architest_test.go:662` — not a text grep), that each of the four declared `foldSites` (`architest_test.go:745`) still calls `abi.FoldRank` in non-test code. A revert to `v.Kind > best.Kind` (valid Go, since `VerdictKind` is a `uint16`) drops the call and turns the gate RED.
- *Honest scope.* `TestFoldRankOrdering` (`abi_test.go:68`) witnesses three lattice relations (`Deny`>`Quarantine`, `Allow`==0, unknown→`FallbackDeny`); it does **not** exhaustively check the pairwise totality of all 1024+ ranks. Totality holds because the codomain is `int` (proven by construction above), not by an exhaustive test.

**WITNESS.**
```
go test ./internal/architest/ ./internal/abi/ -count=1 -timeout 120s \
  -run 'TestFoldSitesOrderByFoldRank|TestFoldRankOrdering' -v
```
`PASS: TestFoldSitesOrderByFoldRank (0.00s)` · `PASS: TestFoldRankOrdering (0.00s)`.

**VERDICT.** PROVEN (2026-06-20).

**DOS.** bound at ship — gate added in `c59bb28 test(architest): gate that every verdict-fold site orders by abi.FoldRank (fak architest)`; `dos commit-audit` / `dos verify` binding recorded at release.

---

## Theorem 2 — the internal package graph is a layered DAG with no upward imports; the hot path has no `os/exec`

**THEOREM.** The internal package graph is a layered DAG — Go forbids import cycles (acyclicity), and the architest tier rule forbids any cross-package edge from a lower tier to a higher one (no upward imports). Additionally, no package on the live tool-call hot path (`adjudicator`, `kernel`, `vdso`, `grammar`, `preflight`, `ctxmmu`, `ratelimit`) imports `os/exec`.

**REGIME.** A — structural contract.

**PROOF.**
- *Acyclicity.* The Go compiler rejects any import cycle, so the package graph is a DAG before any test runs; architest relies on the toolchain for this (the suite would not compile otherwise) rather than re-proving it.
- *No upward edge.* `TestNoUpwardImports` (`fak/internal/architest/architest_test.go:174`) walks every internal cross-package edge — `imports()` (`architest_test.go:115`) reads import blocks via `parser.ImportsOnly`, **build-tag-blind by design** so a GOOS/GOARCH-hidden upward import is still caught — maps both ends through the tier table (`architest_test.go:45`; tiers 0 root … 4 integrator) and fails if `tier(imported) > tier(importer)`. A passing run means every edge is non-upward; with compiler acyclicity that is exactly a layered DAG.
- *No gaps.* `TestEveryPackageDeclaresTier` (`architest_test.go:144`) fails if any on-disk package lacks a tier or the table names a vanished one, so no edge can hide in an untiered package.
- *Hot path subprocess-free.* `hotPath` (`architest_test.go:70`) names the 7 decision-path packages; `TestHotPathHasNoExec` (`architest_test.go:209`) fails if any imports `os/exec`, keeping the per-decide path interpreter/subprocess-free (DIRECTION.md).
- *Honest scope.* The gate proves no *upward* edge, not that same-tier edges are acyclic — that acyclicity is the compiler's guarantee. The tier table was seeded at the 2026-06-17 status quo (zero false positives) and is tightened over time, never loosened to admit a new violation.

**WITNESS.**
```
go test ./internal/architest/ -count=1 -timeout 120s \
  -run 'TestNoUpwardImports|TestHotPathHasNoExec|TestEveryPackageDeclaresTier' -v
```
`PASS: TestNoUpwardImports (0.01s)` · `PASS: TestHotPathHasNoExec (0.00s)` · `PASS: TestEveryPackageDeclaresTier (0.00s)`.

**VERDICT.** PROVEN (2026-06-20).

**DOS.** bound at ship — the layering + hot-path gates live in `internal/architest/architest_test.go`; `dos commit-audit` / `dos verify` binding recorded at release.

---

## Theorem 3 — the frozen wave-0 ABI spine is stable (abi_test round-trips)

**THEOREM.** The closed-enum wire contract of the frozen wave-0 ABI (`VerdictKind`, `Status`, `Outcome`, `TaintLabel`, `ShareScope`, `RefKind`, `FallbackClass`, `ABIMajor`/`ABIMinor`, and the closed `ReasonCode` vocabulary) is stable: every closed value round-trips byte-identically against the committed golden, so any renumber/removal/repurpose fails the build; only appending a new value is allowed.

**REGIME.** A — round-trip / additive-only freeze.

**PROOF.**
- *The closed enums.* Defined as `iota` blocks in `fak/internal/abi/types.go` (`VerdictKind` 205–214, `Status` 188–192, `Outcome` 132–136, `TaintLabel` 85–89, `ShareScope` 94–98, `RefKind` 76–80, `FallbackClass` 252–256, `ABIMajor`/`ABIMinor` 38–41) and the `ReasonCode` vocabulary in `reasons.go` (`CoreReasonCount = 12`, `reasons.go:119`).
- *Round-trip witness.* `TestABIGoldenFreeze` (`fak/internal/abi/abi_test.go:14`) builds a `name → int(value)` map over every closed value (`abi_test.go:15`–`29`), `json.MarshalIndent`s it, and requires string equality with `testdata/abi_v0.1.golden` (`abi_test.go:43`, confirmed present on disk: `{"abi":{"Major":0,"Minor":1},…}`). A renumber, removal, or repurpose changes the marshalled bytes and `t.Fatal`s; only appending a value at the end and regenerating with `UPDATE_GOLDEN=1` is admitted — exactly the additive-only freeze the package doc promises (`types.go:1`–`27`).
- *Reason vocab pinned.* `TestClosedReasonVocabulary` (`abi_test.go:51`) independently asserts `len(coreReasonNames)-1 == CoreReasonCount` and that every core reason has a stable name while unknowns render `REASON_<n>`.
- *Honest scope.* The freeze covers the CLOSED enums + reason vocab; the OPEN registered ranges (`OpCode`, `ExtKey`, registered `VerdictKind`s) are intentionally not frozen and not in this golden.

**WITNESS.**
```
go test ./internal/abi/ -count=1 -timeout 120s \
  -run 'TestABIGoldenFreeze|TestClosedReasonVocabulary' -v
```
`PASS: TestABIGoldenFreeze (0.00s)` · `PASS: TestClosedReasonVocabulary (0.00s)`.

**VERDICT.** PROVEN (2026-06-20).

**DOS.** bound at ship — the golden + freeze test live in `internal/abi/abi_test.go` and `internal/abi/testdata/abi_v0.1.golden`; `dos commit-audit` / `dos verify` binding recorded at release.
