# Borrow scout: mlx-dspark (speculative decoding) → fak

**Date:** 2026-07-10
**Study repo:** `mlx-dspark` — Apple-Silicon speculative decoding (DSpark/DFlash/DeepSpec drafters) in MLX.
**Pin:** `37319c2` — *"v0.3.1 — speculative speedup that holds at long context"*.
**License gate:** MIT (© 2026 erahim3). Idea-borrow, reimplemented in Go, is clean with attribution. No code copied.
**Method:** 5 parallel readers (one per subsystem), each returning mechanism + `path:line@37319c2` citations + a transferable design axis + the worldview reason; then each axis ablated against fak's *actual code* (PRESENT / PARTIAL / ABSENT), no ego-dismissal, no over-claim.

---

## The one finding worth keeping

**mlx-dspark's core worldview is fak/DOS's `verify-don't-trust` discipline, independently re-derived in a token-generation codebase.** Its losslessness invariant — *the fast path is provably identical to the slow path because the authoritative producer re-verifies every unit; a cheap proposal from any source (small model, n-gram, heuristic) is welcome but never trusted until ratified; a clean rollback makes speculation free of downside* (`generate.py:8-12,741@37319c2`) — is the same distrust floor as `dos_verify`/`dos_commit_audit`/`relay.VerifiedProgress` (evidence over self-report). This convergence is the study's headline: it validates fak's spine from an unrelated domain, and it is why the one borrow below is safe to chase hard.

---

## Axis ablations

### A — Online yield-per-cost cap controller → **PARTIAL → FILED**
- **Study mechanism:** `--max-draft auto` measures the target's *verify* cost curve once on the actual host (`calibrate._bench`, cached at `~/.cache/mlx_dspark/calibration.json` keyed by device/MLX-version/model), then a live `CapController` picks the draft cap by **argmax over C of `rate(C) = expected_committed(C) / (draft_cost(C) + verify_ms(C+1))`**, where `expected_committed(C) = 1 + Σ pⁱ` uses a live per-position **acceptance EWMA** `p`, re-picked every 4 rounds with a ×1.03 **hysteresis** guard against flap; a per-batch-width `cap_for(B)` variant shrinks the cap as B widens. Lossless: the cap bounds only how many tokens get *verified*, never what the target emits — so the controller may chase throughput aggressively at zero correctness risk. `calibrate.py:241-285,300-310@37319c2`, `generate.py:537-538,571-572@37319c2`.
- **fak today:** `loopmgr.Governor` picks concurrency via `Policy.MaxConcurrent` — an **operator-set static budget** (`governor.go:71`), and the code comment names the gap: *"a dispatch loop can be given a derived ceiling instead of a fixed cap (#1333)"*. fak already **folds the acceptance-EWMA analog** (`loop.Witnessed/loop.Ended`, `governor.go:160-169`) but consumes it only as a *hold* gate. `trajctl.Calibrate` (`calibration.go:85`) measures whether a *scorer* tracks reality (Pearson vs W3 outcome) but never drives a control parameter. The live dispatch picker ranks by base priority absolutely (`dispatch_aging.go:17-19`).
- **Verdict:** fak has every ingredient (governor, folded yield rate, propose-only fold pattern) but **not** the yield-per-cost *argmax* controller. Filed as a scoped, propose-only borrow (below), complementary to #3976.
- **Honest non-transfer:** the convex `qmm`-knee cost model is Apple-Silicon-matmul-specific and does **not** apply; fak's "cost" is coarse (run wall-time / token spend / host CPU) and "acceptance" is per-run not per-position. What transfers is the *control discipline* — argmax(expected-useful-yield / measured-cost) + hysteresis + emboldened-by-losslessness — **not** the cost model. Prerequisite named in the issue: `LoopSnapshot` does not yet carry a per-run cost signal.

### B — Continuous mid-flight admission + immediate retirement → **PARTIAL, already-tracked → not filed**
- **Study mechanism:** `SpecSlots` retires a finished row the instant it completes, compacts active rows to a contiguous prefix, admits new work mid-flight into freed slots, and runs every forward at the shrunken *active width* (tail falls to `B_act=1`, the bit-exact serial path). Barrier baseline (finished row keeps stepping until the slowest finishes) is the named foil. `batch_engine.py:431-542@37319c2`.
- **fak today:** live dispatch is wave/barrier; the continuous-lifecycle analog (`microagent.Host`: Spawn→Step→retire→Reap) is **[SIMULATED]/off the live path** (#2002); the anti-starvation aging weight is computed but the live picker ignores it (`dispatch_aging.go`; census #3590); "one spawn-admission authority" is an open loop-map proposal.
- **Verdict:** genuinely PARTIAL, but the gap is **already in the backlog**. Filing would duplicate. Recorded, not filed.

### C — Fail-closed reuse cache + LRU workload slotting → **PRESENT, arguably ahead → not filed**
- **Study mechanism:** `PrefixCache` **checks a slot out** on acquire (removes it from the pool) and only a re-validating `store()` re-admits it, so an aborted generation leaves nothing to desync; on **any** `BaseException` the engine `reset()`s the whole cache; un-trimmable/wrapped rotating layers are refused at store; MRU slots (default 2) keep a chat and an agent from thrashing each other; SSD spill is the L2 tier. Fail *closed* to re-prefill (a correctness-preserving cost) rather than serve desynced KV. `prefix_cache.py:78-206@37319c2`, `server.py:318-323@37319c2`.
- **fak today (from devindex witness, not re-read this session):** `#109` bit-exact-or-fault reuse (`RECOMPUTE`/`REFUSE`, never a false HIT), `radixkv` LRU budget, `cachemeta` coherence, quarantine-fail-closed. fak's splice faults to recompute on any non-exact state — same fail-closed guarantee by a different route, arguably stricter (proactive bit-exactness vs invalidate-on-error).
- **Verdict:** convergent; fak is at-or-ahead. The one nuance I did **not** re-confirm in fak source this session is the checkout-on-read/re-admit-only-via-store *lifecycle*; flagged as an unverified fence, not a gap.

### D — Loud refusal from a closed taxonomy + fidelity self-probe before trust → **split**
- **Refusal half — PRESENT.** `from_json` refuses mis-packaged checkpoints from a closed taxonomy before any partial construction (`config.py:95-131@37319c2`); strict-by-default weight load raises on tensor-name mismatch rather than half-load (`load.py:193@37319c2`). fak mirror: `governor.go:99-107` closed refusal vocabulary "in the spirit of the DOS refusal set", the 12-reason adjudicator, `dos_refuse_reasons`. Convergent.
- **Fidelity-self-probe half — LATENT/PARTIAL, not filed.** `Target.verify_tap` proves the *replicated* mlx-lm forward reproduces the model's own forward on a 4-token probe (tol 1e-3) **before** any drafter is trusted, and refuses structurally-unverifiable families (sliding-window/alternating attention) outright. `target.py:105-123@37319c2`. fak has *test-time* bit-exact witnesses + `vcachecal` false-warm/false-cold + the calibration-corpus-drift CI gate, but a **runtime, load-time, refuse-if-the-derived-path-diverges gate on the live path** is latent. Deep model-internals, hard to ship small, arguably subsumed by fak's runtime-witness family — recorded as a worldview-aligned extension, not filed.

### E — Speculate → verify → rollback (lossless) + verbatim n-gram reuse → **PRESENT (inference) / deliberate DIVERGENCE (orchestration) → not filed**
- **Study mechanism:** greedy accept-length via `cumprod(match).sum()`, commit accepted prefix + one bonus token, rollback the rejected suffix by `trim = len(draft) - n` offset arithmetic; temp>0 uses Leviathan/Chen `min(1,p/q)` sampling. Hybrid n-gram lookup reuses a verbatim earlier occurrence for free through the *same* verify path. `generate.py:659-933@37319c2`, `lookup.py:72-89@37319c2`.
- **fak today:** the inference layer has exactly this — `internal/spec` SpeculativeGreedy/Tree with bit-exact rollback via `KVCache.Evict` (`claim:Single-pass batched + tree-attention verify exec`). At the **orchestration** layer fak is deliberately fail-*closed* / block-until-verified (worktree-land is STOP-and-replan on a rejected apply, not optimistic-pipeline-then-rollback). That is a considered worldview choice, not a gap; speculative dispatch on unverified claims would cut against fak's floor. Recorded, not filed.

---

## Filed

- **#4199 feat(loopmgr): derive `MaxConcurrent` as a propose-only yield-per-cost argmax** — Axis A. Complementary to #3976 (backoff-nudge fold) and #1333 (derived-ceiling aspiration). Provenance credited to mlx-dspark `calibrate.CapController@37319c2`; honest non-transfer (convex-knee cost model does not apply) and the `LoopSnapshot`-lacks-a-cost-signal prerequisite both stated in the body.

## Deliberately not filed (with reason)
- **B** continuous admission — already tracked (#2002, #3590, loop-map).
- **C** fail-closed reuse — PRESENT / fak ahead (#109, radixkv, quarantine).
- **D-probe** runtime fidelity self-probe — latent, deep, likely subsumed; named not filed.
- **E** speculative rollback at orchestration — deliberate divergence from fak's fail-closed floor.

## Provenance of fak-side claims
Read in this session: `internal/loopmgr/governor.go`, `internal/trajctl/calibration.go`, `internal/devindex/tiers.go`, `cmd/fak/dispatch_aging.go`. From the devindex witness (sha256-pinned index entries), **not** re-opened this session: `#109`/`KVCache`, `microagent.Host`, `radixkv`, `vcachecal`, `internal/spec`, `internal/gateway/a2a.go`. The C/E fak-side facts should be treated as index-witnessed, not source-re-read.
