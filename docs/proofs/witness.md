---
title: "fak proof: require-witness fail-closed gate"
description: "Soundness proof for fak's witness rung: a claimed effect is denied unless independent git or filesystem evidence confirms it, failing closed on uncertainty."
---

# D8 · witness

The `internal/witness` package is the in-process realization of the **require-witness rung** — the DOS `dos_verify` effect-verify brought inside the kernel. When an adjudicator returns a `VerdictRequireWitness` carrying a `WitnessPayload.Claim` (e.g. `ancestor:<ref>`, `committed:<path>`, `clean:<pathspec>`), the kernel does **not** take the agent's claim on faith. It consults the registered `WitnessResolver`, which corroborates the claimed EFFECT against evidence the agent did not author — git ancestry, object existence, a tracked path, the filesystem — and returns `Confirmed`, `Refuted`, or `Abstain`. The kernel folds that outcome **fail-closed**: only `Confirmed` opens the gate to dispatch; everything else (refuted, every-resolver-abstains, or no resolver at all) is a `Deny`. "Correct" for this module is **decision-procedure soundness** (regime D): the gate never admits an unwitnessed effect, and it fails closed on its own uncertainty.

---

## Theorem T1 — the require-witness rung is sound (denied unless the witness is present)

**THEOREM.** ∀ tool calls whose adjudicated verdict is `VerdictRequireWitness` with claim `c`: the call is **denied and never dispatched** unless some registered resolver returns `WitnessConfirmed` for `c`. The resolver returns `WitnessConfirmed` **only** on positive independent evidence (a real git/filesystem read-back), returns `WitnessRefuted` on contradicting evidence, and **ABSTAINs (never falsely Confirms)** on an unparseable claim or a missing git. The kernel then folds `Confirmed → Allow` (dispatch exactly once), `Refuted → Deny/TRUST_VIOLATION` (no dispatch), and `Abstain ∨ no-resolver → Deny/UNWITNESSED` (no dispatch).

**REGIME.** D — decision-procedure soundness (fail-closed gate, monotone fold).

**PROOF.** The guarantee composes two layers.

*Resolver layer (in-scope, `internal/witness`).* `Resolver.Resolve` (`fak/internal/witness/witness.go:85`) parses the `kind:arg` claim via `splitClaim` (`witness.go:164`); a colon-less, empty, or trailing-colon claim returns `ok=false` and resolves to `WitnessAbstain` (`witness.go:88`) — it can never `Confirm`. The dispatch switch (`witness.go:90`) returns `WitnessConfirmed` **only** on positive evidence: `git merge-base --is-ancestor` exit 0 (`:98`), `cat-file -e <ref>^{commit}` exit 0 (`:110`), `ls-files --error-unmatch` exit 0 (`:122`), `os.Stat` success (`:132`), a non-empty `git log --grep` `%H` (`:143`), or an empty `git status --porcelain` (`:155`). Every branch where git **could not run** (`err != nil`) returns `WitnessAbstain` (`:94`, `:107`, `:119`, `:140`, `:152`), so a missing git never yields a false `Confirm`. A known-negative git exit (1) yields `WitnessRefuted`; an indeterminate ref (exit 128) yields `WitnessAbstain` — "a bad ref is not evidence of absence" (`:103`). The zero value of `WitnessOutcome` is `WitnessAbstain` (`fak/internal/abi/registry.go:611`), so the type defaults to fail-to-abstain.

*Kernel fold (soundness completion, `internal/kernel`).* `Kernel.resolveWitness` (`fak/internal/kernel/kernel.go:159`) folds over `abi.Witnesses()`: the **first** `WitnessConfirmed` returns `VerdictAllow` (`:167`–`:169`), and only that path leads `Submit` to store the call for dispatch (`kernel.go:271`,`:277`). Any `WitnessRefuted` sets `refuted=true` → `Deny`/`ReasonTrustViolation` (`:176`–`:181`); if nothing confirms and nothing refutes — every resolver abstained, **or** `abi.Witnesses()` is empty — the default is `Deny`/`ReasonUnwitnessed` (`:174`,`:180`). Hence the **only** route to dispatch is a positive independent corroboration. The fold is monotone fail-closed: `FoldRank(RequireWitness)=4` while `FoldRank(Deny)=100` (`fak/internal/abi/registry.go:754`), so an unresolved require-witness can only become more restrictive, never less.

The ten witnesses pin each arm. Resolver: `TestAncestorClaim` (confirmed/refuted/abstain on git exit 0/1/128), `TestCommittedAndGrep` (tracked vs untracked, grep match vs no-match), `TestCleanClaim` (clean vs dirty vs git-missing), `TestGitMissingAbstains` (missing git → abstain, fail-to-abstain), `TestUnparseableClaimAbstains` (5 malformed claims all abstain), `TestRealGitAncestor` (DEFAULT real-git runner end-to-end against this repo: `HEAD` is its own ancestor → Confirmed, `fak/go.mod` tracked → Confirmed, bogus path → Refuted, null sha ≠ ancestor). Kernel: `TestRequireWitnessConfirmedOpensGate` (Confirmed → Allow, dispatch n=1), `TestRequireWitnessRefutedStaysClosed` (Refuted → Deny/TRUST_VIOLATION, n=0), `TestRequireWitnessAbstainFailsClosed` (Abstain → Deny/UNWITNESSED, n=0), `TestRequireWitnessNoResolverPreservesV01` (no resolver → Deny, n=0, deny-counter=1).

**WITNESS.**
```
go -C fak test ./internal/witness/ -count=1 -timeout 120s -run \
  'TestAncestorClaim|TestGitMissingAbstains|TestUnparseableClaimAbstains|TestCommittedAndGrep|TestRealGitAncestor|TestCleanClaim' -v
go -C fak test ./internal/kernel/ -count=1 -timeout 120s -run \
  'TestRequireWitnessConfirmedOpensGate|TestRequireWitnessRefutedStaysClosed|TestRequireWitnessAbstainFailsClosed|TestRequireWitnessNoResolverPreservesV01' -v
```

**VERDICT.** **PROVEN** (2026-06-20, macOS arm64 native go1.26). All six in-scope resolver tests PASS (`ok internal/witness 0.263s`) and all four kernel-fold tests PASS (`ok internal/kernel 0.272s`). Together they witness both halves: the resolver Confirms only on real evidence and abstains-never-confirms on uncertainty, and the kernel folds that outcome fail-closed so a require-witness call is denied unless the witness is present.

> **Honesty note.** The prompt's hint `TestVDSOSoundness` is a **misnomer for this module**: no such test exists in `internal/witness`. The one `TestVDSOSoundness` in the tree (`fak/internal/steward/steward_test.go:40`, unit 88) is the **vDSO dedup-cache divergence** steward probe — it has nothing to do with the require-witness rung. The witnesses that actually discharge T1 are the ten named above.

**DOS.** Shipped in `aedce3b fak/witness: in-process dos_verify effect-verify backing the require-witness gate` (resolver) and the kernel fold; the witness-package leaf last moved at `04e4b23`. _bound at ship._
