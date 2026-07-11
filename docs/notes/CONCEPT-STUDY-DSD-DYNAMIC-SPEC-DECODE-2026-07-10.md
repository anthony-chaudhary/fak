---
title: "Supplement to the vLLM Dynamic-SD scout: the agent-layer speculator seam + corrected kernel anchors (2026-07-10)"
description: "Delta pass over Cohere/vLLM Dynamic Speculative Decoding (PR #32374 @ 4ef4492e). The primary scout is study-vllm-dynamic-sd-borrow-2026-07-10.md (same source, same DIVERGENT/0-filed verdict on T1 lookup / T2 population / T3 per-call depth). This note records ONLY what that pass did not: (1) the sharpest on-axis analog it never read — the agent-layer tool-call speculator internal/abi/speculate.go, which gates on SuccessProb (benefit) but NOT on live slot/budget LOAD, while internal/microagent/slotsched.go:49-54 already names the unaccounted-speculative-call hole → DSD's never-regress-under-load lesson, home epic #809; (2) a corrected kernel anchor — the prior scout cites internal/spec SpeculativeGreedy, which is ABSENT from the live tree (parked at .dos/_dos_park/_iso_build/internal/spec/); the shipped substrate is internal/model/verify.go:36 + internal/polymodel/polymodel.go:410, so literal adaptive-K is a follow-on to the OPEN drafter gap #3197/#3078, not the parked driver; (3) internal/compute/prewarm_admit.go as a DIVERGENT hard-fence route to the same never-regress property. Still 0 new issues."
---

# Dynamic-SD → fak — supplement (agent-layer speculator seam + corrected anchors)

**Read the primary scout first: [study-vllm-dynamic-sd-borrow-2026-07-10.md](study-vllm-dynamic-sd-borrow-2026-07-10.md).**
It studies the same merge — vLLM **PR #32374** "[V1][Spec Decode] Add Dynamic SD", squash
`4ef4492e9b7a5a7ba295da783d456d45db5eb9d6` (2026-06-14) — distils it into T1 (sparse-range→dense O(1)
`batch_size→K` lookup), T2 (grade draft depth down as load rises, K→0 fallback), T3 (depth as a per-call
param), and reaches **DIVERGENT / PRESENT-by-other-means, 0 leaves filed**: fak's dispatch control plane
already occupies the load-grading space (setpoints #4036/#4038/#3368, effort ladder #4069), and the literal
K-by-batch-size analog is dormant because fak's spec decode is not a batched serving path. I do **not**
re-litigate any of that — it is correct.

This supplement exists because the prior pass read the **dispatch-population** plane
(`loopmgr/governor.go`, `dispatchtick/*`, `modelroute/effortcost.go`, `dispatch_tick.go`) but not the two
seams below, and cited one anchor that has since moved. Also a sibling of today's
[DFlash study](CONCEPT-STUDY-DFLASH-2026-07-10.md) and the
[DeepSeek-V4 MTP eval](DEEPSEEK-V4-MTP-SPECULATIVE-EVAL-2026-07-08.md), which map the same kernel substrate.

Added `07516fda` provenance the prior note predates: **PR #45953** "[MRV2][SD] Make Dynamic SD compatible
with Full Cuda Graphs" (merged 2026-07-04) — captures all decode shapes when DSD is on so any runtime
`optimal_K < K_max` still hits a graph. Plumbing, not new policy.

## Delta 1 — the sharpest analog the prior scout never read: `internal/abi/speculate.go` (home: epic #809)

The prior scout mapped T2 ("grade optimistic depth by load") onto the dispatch **wave/population** plane and
found it PRESENT-by-other-means. But fak has an actual **speculator** one level up that it never opened:
`internal/abi/speculate.go` — SEAM 4, PASTE-style tool-call prediction (arXiv 2603.18897): predict the next
tool call, run it on slack, commit on match / squash on miss. Default-OFF, default-deny-on-effects.

Its admission gate is `SpecPattern.SuccessProb` — an **empirical hit-rate (benefit) gate** plus "run on
slack." It has **no live-load gate**, and that is DSD's exact axis one level up:

- The per-host slot scheduler `internal/microagent/slotsched.go:55` bounds concurrent model calls to K; the
  gateway budget layer (`internal/gateway/admission.go`, Σ tokens ≤ budget) bounds tokens. `slotsched.go`
  names the hole in its own invalidating-assumption block (`:49-54`): *"speculative/parallel decode that
  issues several concurrent provider requests per logical turn would break the one-call-one-slot
  accounting."* A speculative provider call is an **extra, unaccounted** unit of scarce provider concurrency.
- So on a saturated host the speculator competes with demand work — the precise DSD regression (speculation
  that accelerates an idle host steals throughput from a loaded one). fak gates speculation on *benefit*
  (SuccessProb) but never on *cost-under-saturation*: there is no "back off toward zero as slots/budget fill."

**On-axis verdict: PARTIAL → ABSENT** (benefit gate present, load gate absent). This is a sharper, more
literal home for T2's "grade the *optimism* down as load rises" than the population plane — because
`speculate.go` is an actual speculator, not a wave-sizer — and the prior scout's population reading did not
reach it. **Not a new issue:** it belongs as a scoped consideration under **epic #809** (speculative
agent-loop execution), which already owns the promotion-on-match lifecycle and the riskier #809(b)/(c)
siblings. Low urgency — both `speculate.go` and the slot scheduler are opt-in/experimental (nothing in the
default serve/guard/dispatch path constructs them) — but it is a real, code-anchored gap on DSD's own axis
that the prior tally does not capture.

## Delta 2 — corrected kernel anchor: the literal adaptive-K borrow, and where it actually lives

The prior scout's axis-4 seam cites `internal/spec` `SpeculativeGreedy`/`SpeculativeTree` as fak's
fixed-`k` spec path. **That package is not in the live buildable tree** — `ls internal/spec` is absent;
`SpeculativeGreedy` survives only as a doc-reference in `internal/model/verify_test.go`; the real
`internal/spec/` is a **parked isolated-build sandbox** at `.dos/_dos_park/_iso_build/internal/spec/`. This
matches today's [DFlash correction](CONCEPT-STUDY-DFLASH-2026-07-10.md): the **shipped** substrate is

- verify — `internal/model/verify.go:36` `VerifyForward` (single-forward K-candidate verify, bit-exact),
- accept — `internal/polymodel/polymodel.go:410` `AcceptGreedy` / `:473` `AcceptTree`,
- rollback — `internal/model/kvcache.go:94` `KVCache.Evict`,

with the **draft SOURCE as the one open gap** — `Session.Generate` (`internal/model/kv.go:742-753`) is plain
greedy and never calls verify (#3197 EAGLE-3 head; #3078 GLM-5.2 variant; epic #2236). So the axis-4
disposition is unchanged in *conclusion* (literal K-by-batch-size is dormant) but corrected in *anchor and
reason*: it is not "fixed-k in `internal/spec`," it is **"no drafter yet, so no K exists to make dynamic"** —
DSD is strictly **downstream of #3197/#3078**, and fak's kernel is single-stream anyway
(`verifyForwardBatchedOK` refuses every real backend, `internal/model/verify.go:64-73`), so the batch-size
regime DSD keys on does not exist here. The one thing worth carrying to that lane: DSD's **AL (acceptance
length)** is exactly the "measure acceptance + tok/s" #3078 already asks for, and its **K=0 fallback** is the
guardrail a fak drafter must honour (never regress single-stream tok/s vs plain `Generate`).

## Delta 3 — `internal/compute/prewarm_admit.go`: a DIVERGENT route to DSD's never-regress property

Not read by the prior scout. `DecidePrewarmAdmission` is fak's closest shipped structure to "an acceleration
that must not hurt under load," and it reaches DSD's never-regress guarantee by a **binary, fail-safe
pollution fence** — `WarmPoolFree` false → `WarmSkip` (a warm may never evict demand residency) — over a
*byte-known pure prefetch* ("it cannot be wrong"), rather than DSD's **graded, offline-profiled table**.
DSD's table extracts more benefit in the *intermediate* regime (K=1 vs hard on/off) at the cost of
profiling + mis-tuning risk; fak's hard fence is provably safe and drift-free (the house form, matching
`discard_admit.go`/`batchsched`). **DIVERGENT**, and for fak's auditable posture arguably the better
mechanism — recorded as a design contrast, not a borrow. (MoE's non-monotonic optimal-K from the blog is
expert-physics-specific with no fak analog — also DIVERGENT.)

## Verdict

Supplements, does not supersede, [study-vllm-dynamic-sd-borrow-2026-07-10.md](study-vllm-dynamic-sd-borrow-2026-07-10.md).
Net new signal over that scout's tally:

- **New seam (sharper T2 home):** `internal/abi/speculate.go` gates speculation on benefit (`SuccessProb`)
  but not on live slot/budget load; `internal/microagent/slotsched.go:49-54` already names the unaccounted
  speculative-call hole. On-axis **PARTIAL→ABSENT**; home **epic #809**; low urgency (opt-in components).
- **Corrected anchor:** literal adaptive-K attaches to the shipped `verify.go:36`/`polymodel.go:410`
  substrate with an OPEN drafter gap (#3197/#3078), **not** the parked `internal/spec`; premature until the
  drafter lands; carry DSD's AL + K=0 guardrail as a design note on #3078.
- **New contrast:** `internal/compute/prewarm_admit.go` reaches never-regress by a hard fail-safe fence —
  **DIVERGENT** from DSD's graded table.

**Status: `not yet` — 0 new issues** (agrees with the primary scout; every axis maps onto an already-open
issue: #809 for the agent speculator, #3197/#3078 for the kernel, #2236 the superset epic). Durable output
is this note.
