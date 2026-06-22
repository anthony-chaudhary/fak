---
title: "fak proof: vDSO tiered cache soundness"
description: "Correctness proof for fak's vDSO fast-path cache: tier-1 equals recompute, a tier-2 hit never serves a stale answer, and the integrity epoch is monotone."
---

# vDSO — proof obligations (witnessed)

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/vdso/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

## D13 (vdso) — adversarially re-verified per-theorem verdicts

- [PROVEN] tier1-fastpath-equals-engine: For a tier-1 pure tool, the vDSO fast-path result is identical to the direct pure recomputation (the ground-truth engine answer): ∀ (a,b), Lookup(calculate{a,b}).sum == a+b == calcSum(args).sum.
    witness_cmd: go test -run 'Unit38_Soundness' ./internal/vdso/ -count=1 -timeout 120s -v
    witness_tests: TestUnit38_SoundnessTier1EqualsRecompute
    mechanism_refs: fak/internal/vdso/vdso.go:241, fak/internal/vdso/vdso.go:246, fak/internal/vdso/vdso.go:507

- [PROVEN] tier2-hit-equals-fresh-call: A tier-2 cache hit returns the same answer a fresh engine call would: the cache re-serves the exact engine-produced payload Ref it was filled with (identity by construction), and any condition that could make that stale — a write-shaped completion, an explicit world bump, a finer-scope write on the read's chain, or a refutation of the admitting witness — turns the would-be hit into a MISS (→ engine), so the cache NEVER serves a stale answer.
    witness_cmd: go test -run 'Unit26_27|Unit28|Scope_Soundness|Revoke_Sound|Revoke_RefusesReAdmit|Revoke_EvictsAll' ./internal/vdso/ -count=1 -timeout 120s -v
    witness_tests: TestUnit26_27_Tier2CacheAndCanonicalization, TestUnit28_BumpWorldInvalidates, TestUnit28_WriteCompletionBumpsWorld, TestScope_Soundness_SameRouteAlwaysInvalidated, TestScope_Soundness_CoarseWriteCatchesFineRead, TestScope_Soundness_UnknownWriteFlushesAll, TestRevoke_Sound_RevokedNeverServed, TestRevoke_RefusesReAdmitUnderRefutedWitness, TestRevoke_EvictsAllConsumersOfWitness
    mechanism_refs: fak/internal/vdso/vdso.go:290, fak/internal/vdso/vdso.go:317, fak/internal/vdso/vdso.go:347, fak/internal/vdso/vdso.go:276, fak/internal/vdso/scope.go:156, fak/internal/vdso/scope.go:303, fak/internal/vdso/scope.go:319, fak/internal/vdso/revoke.go:85
    (CITATION FIX: the prior scope.go:86 ref pointed at `return "namespace"` inside Granularity.String(); the finer-scope write/read-chain mechanism actually lives at scope.go:156 readChain, scope.go:178 writeTags, scope.go:319 bumpAndPublish, and the resource-mode cacheability gate scope.go:303 resourceMisnamed.)

- [PROVEN] three-tier-lookup-deterministic: The 3-tier lookup is deterministic: Lookup consults a FIXED order (tier-1 pure → tier-3 static → tier-2 cache), the tier-2 key is a pure function of (tool, canonicalized-args, epoch-chain) so semantically-equal calls map to the same key (and thus the same hit/miss verdict), and an unknown/unhinted call deterministically misses.
    witness_cmd: go test -run 'Unit25|Unit29|Unit26_27|Unit34_Miss' ./internal/vdso/ -count=1 -timeout 120s -v
    witness_tests: TestUnit25_Tier1Pure, TestUnit29_Tier3Static, TestUnit26_27_Tier2CacheAndCanonicalization, TestUnit34_Miss
    mechanism_refs: fak/internal/vdso/vdso.go:237, fak/internal/vdso/vdso.go:317, fak/internal/vdso/vdso.go:151, fak/internal/vdso/vdso.go:161

- [OPEN] integrity-epoch-advances-monotonically: The integrity (trust) epoch advances monotonically: a refutation (Revoke of a non-empty witness) strictly increases TrustEpoch by 1, an empty-witness Revoke is a no-op (epoch unchanged), and across a sequence of N refutations the epoch is strictly increasing and never decreases.
    witness_cmd: go test -run 'Revoke_Orthogonal|Revoke_EmptyWitnessNoOp|Revoke_PublishesOnCoherenceBus' ./internal/vdso/ -count=1 -timeout 120s -v
    witness_tests: TestRevoke_OrthogonalToWorldVersion, TestRevoke_EmptyWitnessNoOp, TestRevoke_PublishesOnCoherenceBus
    mechanism_refs: fak/internal/vdso/revoke.go:91, fak/internal/vdso/revoke.go:136, fak/internal/vdso/vdso.go:85
    (Still OPEN: the witnessed tests cover only a single +1 increment, the empty no-op, and one published epoch=1. The N-refutation strict-increase clause is not directly exercised by any multi-revoke test, though revoke.go:91's atomic.AddUint64 makes it true by construction.)

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **integrity-epoch-advances-monotonically** → ✅ PROVEN by `TestProof_IntegrityEpochMonotonicSequence`. Closed. Two deterministic asserting tests. TestProof_IntegrityEpochMonotonicSequence drives a fixed-seed (rand.NewSource(0x5150)) mix of 2000 Revoke steps: ~1/5 empty-witness no-ops, the rest non-empty refutations over a small witness alphabet so re-revocation of an already-refuted witness occurs frequently. After EVERY step it asserts (a) TrustEpoch never decreased; (b) empty-witness Revoke is a total no-op (evicted==0, epoch unchanged, Revocations() unchanged); (c) non-empty Revoke strictly increases TrustEpoch by exactly +1 and Revocations() by +1, and re-revoking an already-refuted witness STILL ticks (revoke.go has no already-revoked short-circuit on the epoch path at revoke.go:91); (d) TrustEpoch equals the running count of non-empty Revokes at every step; final TrustEpoch==Revocations()==total refutations, with a non-vacuous guard that refutations>0. TestProof_IntegrityEpochStrictlyIncreasingNonDecreasing isolates the strict-increase/non-decrease halves on a clean sequence of K=256 distinct refutations yielding epochs 1..256, each strictly greater than the last. Mechanism confirmed at revoke.go:86 (empty early return -> no-op), revoke.go:91 (atomic.AddUint64(&v.trustEpoch,1)), revoke.go:136 (TrustEpoch load). Package clause and helper New reused from existing internal tests (package vdso). Both tests PASS; whole vdso package PASSES with the file present.
