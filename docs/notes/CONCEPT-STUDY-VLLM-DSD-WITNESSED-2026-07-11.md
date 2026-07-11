---
title: "Study-repo (4th pass): vLLM Dynamic Speculative Decoding → fak — witnessed, 0 new leaves, 2 enrichment comments (2026-07-11)"
description: "A witness+adversarial-verify pass over four DSD borrow candidates (P1 range-lookup invariants, P1 step-advice tiers, P2 adaptive concurrency, P4 config-incompat reconciliation) against a fak tree that has evolved since the three prior DSD passes. Terminal outcome matches all three priors: 0 new leaves filed. P1/P1-tiers re-confirm PRESENT/ALREADY_EXISTS; P2's gap is #3368's own open scope (unfed WorkerFloor); P4 surfaces a NEW concrete surface — the guard managed-cache ON-branch structurally over-claims ACTIVE — but it lands inside the existing cache-verify epic #3569 / leaf #3624, so it is recorded as a prevent-at-source enrichment comment, not a competing leaf."
---

# Study-repo (4th pass): vLLM Dynamic Speculative Decoding → fak — witnessed (2026-07-11)

A fourth witness pass over vLLM's **Dynamic Speculative Decoding** (PRs **#32374** `4ef4492e`,
**#45953** `07516fda`; Apache-2.0, read read-only in scratch, **INSPIRE** only — no bytes vendored),
run against a fak tree that has moved since the three prior DSD passes landed on 2026-07-10. Four
candidate borrows were mapped onto fak by a parallel witness fan-out, then each was handed to an
**independent adversarial verifier** told to prove fak already has it.

## Why a 4th pass, and the honesty boundary it holds

This source has already been studied three times, every one filing **0 leaves**:

- [`study-vllm-dynamic-sd-borrow-2026-07-10.md`](study-vllm-dynamic-sd-borrow-2026-07-10.md) — distilled T1 (sparse-range→dense `batch_size→K` lookup), T2 (grade depth DOWN as load rises), T3 (depth as per-call param). Terminal **DIVERGENT / PRESENT-by-other-means, 0 filed**; recorded one deferred lead (couple an optimistic-depth knob to measured fleet load).
- [`CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md`](CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md) — delta pass; homed the agent-layer speculator seam to **#809**, carried the K=0 guardrail as a design note on **#3078**. Still **0 filed**.
- [`CONCEPT-STUDY-SPECDECODE-2026-07-10.md`](CONCEPT-STUDY-SPECDECODE-2026-07-10.md) — dismissed adaptive-K to the Dynamic-SD scout (DIVERGENT).

The value of a 4th pass is **not** to re-derive that verdict. It is (a) to re-witness the "present-by-
other-means" claims against a tree that has since gained new tickets (**#3574** adaptive throttle,
**#3569**/#3624 cache-verify, #3650, #3892, #4155…), confirming they still hold rather than assuming
it; and (b) to sweep the P4 axis (config-time incompat reconciliation) the three spec-decode-focused
passes never touched, since it reaches a different fak subsystem (the guard managed-cache posture).
The bar remains the kvcached study's: *filing a test for a non-existent seam — or a leaf that dups a
tracked one — is noise.*

## Candidates and verdicts

| # | Candidate (DSD pattern) | Witness | Adversarial verify | Disposition |
|---|---|---|---|---|
| **P1** | Validated sparse-range → dense-total `scalar→value` lookup (`spec_decode/dynamic/utils.py` non-overlap + coverage-from-1 + carry-forward) | NOT_A_GAP | CONFIRM NOT_A_GAP | **Drop — PRESENT-by-other-means** |
| **P1-tiers** | Declared/validated tier table vs magic-number soup (`step_advice`) | ALREADY_EXISTS | CONFIRM NOT_A_GAP | **Drop — ALREADY_EXISTS** |
| **P2** | Adapt concurrency to a live load scalar in BOTH directions, toward a goodput objective | PARTIAL | CONFIRM REAL_GAP | **Comment on #3368 — its own open scope** |
| **P4** | Config-time reconciliation: a forced optimization yields to a hard-incompatible mode with a NAMED reason | EXISTING_BUG (XS) | CONFIRM REAL_BUG (XS) | **Comment on #3624 (epic #3569) — prevent-at-source; no competing leaf** |

### P1 — range-lookup invariants: total-by-construction already

vLLM's `validate_and_normalize_dynamic_sd_schedule` exists because its schedule is a **human-authored
data list** (`num_speculative_tokens_per_batch_size`) that can have gaps/overlaps — hence non-overlap
+ "first range must start at 1" + clamp-to-max. fak has **no operator-declared `(min,max]→value`
bracket table that expands against a runtime scalar**. Every analog is total-by-construction or
fail-closed: `dispatchorder.go:976-981` collides *conservatively* on unparseable ranges (the inverse
of the bug), `abi/speculate.go:98-99` is an exact-signature map with default-deny, `tierpolicy.go`
floors at T0/T2, `attemptbudget.go:334-337` falls back to a nonzero `FailureClassOther`. No
remediation target.

### P1-tiers — step-advice: named constants, single-sourced, exhaustively tested

`adviseCtxStep` (`internal/gateway/ctxvalue.go:311-355`) is a `switch` over a closed `StepClass`
vocab with a terminal `default` (total-by-construction — no coverage invariant to protect). Every
threshold is a **named constant** (`ctxvalue.go:74-86`), and the 80% checkpoint line is
single-sourced with the human debug nudge via `compactionNudgeNearPercent` (`ctxvalue.go:75` =
`debug_stats.go:433`, consumed `:453`) — the two surfaces agree by construction. Exhaustive rung/class
coverage in `ctxvalue_test.go`. The DSD borrow is correctly grounded but non-applicable. (Only latent
item: a slightly-wrong reason string on the degenerate `Budget>0 && Resident==0` turn; the class is
still fail-closed `unknown`. Cosmetic.)

### P2 — adaptive concurrency: the DOWNWARD half is rich; the UPWARD half is an unfed kernel

fak's dispatch cap is computed **downward-only**: `EvaluatePreflight` starts at the static ceiling
(`FallbackMaxWorkers=20`, `dispatchtick.go:32`) and only `min()`s down; the three live-load folds
(`ApplyGateBackpressure`/`ApplyRateLimitBackpressure`/`ApplyChurnBackpressure`, composed in
`cmd/fak/dispatch_tick_preflight.go:52-69`) each "only LOWER the effective cap." The one **upward**
term — `PreflightInput.WorkerFloor` (**#3368**'s two-timescale forecast floor) — is built and tested
but **has no live producer** (`.WorkerFloor` assigned only in tests; the `PreflightInput` literal in
`dispatchPreflight` omits it), so `cap_terms.limiting == "floor"` is unreachable in production.
`ReconcileSetpoint`/`ContractionTarget` (#4036/#4038) are likewise callerless outside tests. No
goodput objective steers concurrency anywhere. **This is #3368's own open scope**, and the "feed it a
goodput signal" framing is the prior scout's already-recorded deferred lead — so it is an enrichment
comment on #3368, not a new leaf.

### P4 — config-incompat reconciliation: the one new surface, homed to #3569/#3624

The DSD discipline (`vllm/config/vllm.py` `_maybe_disable_dynamic_sd_for_data_parallel` /
`_maybe_override_dynamic_sd_cudagraph_mode`): a **force-enabled** optimization still yields to a
hard-incompatible mode and downgrades **with a named reason**. fak's `resolveGuardManagedCache`
implements exactly this on the **AUTO** path (`cmd/fak/guard_managed_cache.go:72-84` — `localModel` /
`provider!=anthropic` → passive with a named reason). But the **ON branch returns `active:true`
unconditionally** (`:68-69`): it overrides not only the *billing* gate (intended, the 2026-07-10
on-by-default policy) but also the *structural* gate. Since `guard_cache_posture.go:46-49` flipped
unset `FAK_MANAGED_CACHE`→`on`, every unconfigured non-anthropic/local seat (`fak codex`, `--gguf`)
now resolves ON and prints the **ACTIVE** banner (`:96`, "…upgraded on the outbound wire") while
`internal/gateway/messages.go:809` gates the upgrade on `anthropicPassthroughFor` (`Provider==anthropic`
only, `:1349-1352`) — a **provable wire no-op**. Banner claims ACTIVE; wire is PASSIVE.

Severity is **XS**: the wire is *safe* (no corruption — distinct from **#3892**, which is the anthropic
`-p` malformed-400 case). This is a pure **observability over-claim**. Test gap:
`guard_managed_cache_test.go:36-38` covers ON only for `provider:"anthropic"`.

The mismatch symptom — "a session can claim ACTIVE and behave PASSIVE" — is the verbatim premise of
**#3624** (`post-run managed-cache posture reconciliation (banner claim vs wire reality)`) under epic
**#3569**. #3624 *detects* it after the run; P4 is the *prevent-at-source* complement (mirror the AUTO
structural checks in the ON branch → forced-but-inert passive with a named reason, keeping the billing
override). Because it lands inside that epic's scope, it is recorded as an enrichment comment on
#3624, **not a competing leaf**.

## Terminal outcome

**0 new leaves filed** (matching all three prior DSD passes). Two enrichment comments (anti-re-file,
same discipline as the [LMCache study](CONCEPT-STUDY-LMCACHE-2026-07-08.md)'s "borrows became
enrichment comments"):

- **#3624** (epic **#3569**) — P4 prevent-at-source: the ON branch must reconcile structural
  incompatibility, so the emitted ACTIVE claim is true by construction and the post-run auditor only
  fires on genuine residue.
- **#3368** — P2 live-signal: the forecast-floor upward term is inert for want of a goodput-style
  signal, not for want of a mechanism.

Adversarial verify sustained every witness verdict. All candidates INSPIRE (Py→Go clean-room, zero
bytes vendored). **This is the 4th pass on this source — a 5th should not re-run it** unless a new DSD
PR lands or fak's dispatch/managed-cache posture surface changes materially.
