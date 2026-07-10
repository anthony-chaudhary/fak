---
title: "Borrow study: SDPO / SDPO++ off-policy continual learning — witnessed against fak's RSI loops"
description: "Deep-read of lasgroup/SDPO and siyan-zhao/OPSD (self-distillation policy optimization) plus the SDPO++ off-policy writeup, witnessing each idea against fak's RSI/off-policy substrate; filed the one net-new borrow (#3917) and recorded the rest as PRESENT or ML-only."
---

# Borrow study: SDPO / SDPO++ off-policy continual learning (2026-07-10)

Studied the **SDPO** (self-distillation policy optimization) method family and its **SDPO++**
off-policy scaling extension for what a Go agent-orchestrator's **self-improvement (RSI) loops** could
borrow. The article's thesis is pure RL training, but its *architecture* — continual learning from
single, stale, off-policy production samples — maps onto fak's RSI loops, which learn from an
accumulated corpus produced against since-changed levers. Every idea was witnessed against fak's
existing substrate at `path:line`; only the net-new one was filed.

## Sources (deep-read, cited at `path:line@sha`)

- **lasgroup/SDPO** — `github.com/lasgroup/SDPO` @ `7c457fc1b1f636ae794eb0362ba37d4743b06fbc` — **Apache-2.0** (integrate-eligible).
  - `verl/trainer/ppo/core_algos.py:1168-1176` — `compute_self_distillation_loss`: IS ratio `exp(student−old_log_probs)` **clamped to `is_clip`**.
  - `verl/trainer/config/sdpo.yaml:18-32` — knobs `self_distillation.is_clip: 2.0`, `rollout_correction.rollout_is_threshold: 2.0`.
- **siyan-zhao/OPSD** — `github.com/siyan-zhao/OPSD` @ `7448751f307a9cdbcc1246dd1565a1a605b443df` — **NO LICENSE (all-rights-reserved) → inspire-only, code reuse forbidden**.
  - `opsd_trainer.py:415-464` — `generalized_jsd_loss(..., token_clip=…)`: per-token divergence **capped before reduction** ("prevents style tokens from dominating … over math tokens").
  - `opsd_trainer.py:229-233,1459-1507` — explicit **on-policy vs off-policy** loss / step-equivalent accounting.
- **Trajectory.ai "Scaling SDPO" / SDPO++** — the off-policy/continual-learning writeup. **No public code** (concept only).

## Witness verdicts (so this is not re-done)

| Candidate | Verdict | fak evidence |
|---|---|---|
| "Constrain the step, not the destination" — bounded per-update move | **PRESENT** | `internal/rsiloop/metarsi.go:89-95,144,154` — `GainStep` bounded increment clamped by a ceiling, "no unbounded swing"; propose-only + witness-gated. |
| Fresh-re-measurement keep-bit (avoids off-policy staleness by re-running) | **PRESENT** | dojo RSI keep-bit derives every witness from a worktree re-measurement (#1021); `internal/rsiloop/loopvariant.go:152,167,201` evaluates against a fresh `TaskSetWitness`. |
| Off-policy **evaluation** with divergence frontier | **PRESENT (different axis)** | `internal/turnbench` OPE (`ope.go`): `bounded@i` doubly-robust, IS explicitly refused — but the **call-index** axis, not temporal/generational staleness. |
| Age/staleness signal on samples | **PRESENT (refresh polarity)** | `internal/dojocal/select.go:137` `Staleness=clamp01(age/recheck)` — weights a cell **UP for re-checking**, i.e. schedules re-measurement; does not down-weight a stale sample's *contribution*. |
| **Off-policy-age trust-decay on archive-mining folds** | **PARTIAL → FILED #3917** | The arms that mine the accumulated archive instead of re-running it weight every historical sample equally regardless of off-policy age: `internal/rsiloop/loopvariant.go:56` `ProposeLoopVariants(baseline, archive)`, `internal/dojocal` selection over accumulated `CalibErr`, `metarsi` crude recent-cycle window. Add a lever-generation `staleness∈[0,1]` (SDPO-K) + per-sample influence cap (`is_clip`), reducing to today's equal-weight fold at generation-delta 0. |
| Reverse-KL self-distillation loss; top-k+tail distillation; PPO/IS gradient correction | **ML-only, DROPPED** | `core_algos.py:1085-1188`, `:1106-1131`, `sdpo.yaml:30-32` — no home in a Go orchestrator that trains no weights. |

## Filed

- **#3917** feat(rsiloop): off-policy-age trust-decay on the archive-mining RSI arms (SDPO++ staleness-K + is_clip). Parent #1021 (dojo autonomous RSI loop); architectural relative #2834 (witnessed learning-loop kernel primitives). Label `sdpo-inspired` (new, matching the `ds4-inspired`/`hermes-inspired` convention).

## Takeaway

fak's off-policy/RSI substrate is already strong — bounded step, fresh-witness keep-bit, OPE
counterfactual axis, and an age signal all exist. The single genuine gap is **generational off-policy
trust-decay on the folds that cannot re-measure** (proposal/selection/trend), which is exactly the
SDPO++ staleness-K + `is_clip` pairing. Everything else is either already present or ML-training
internals with no orchestration home.
