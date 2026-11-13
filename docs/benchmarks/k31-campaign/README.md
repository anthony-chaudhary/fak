---
title: "K3.1 Bench-Campaign Arm-Matrix Plan [PLANNED] (2026-11-12)"
description: "The #14374 witnessed bench-campaign scaffold: the fixed arm matrix (speculative-decode arms x concurrency ladder, plus hot-expert offload placeholders), the per-arm fak.serving-sweep.v1 receipt contract, and the fold into one campaign index. No arm is witnessed yet."
---

# K3.1 bench-campaign arm-matrix plan [PLANNED] (2026-11-12)

> ### PROVENANCE STATUS: PLANNED — NO MEASURED NUMBER IN THIS DOCUMENT
>
> **Every arm in this matrix is PLANNED. This directory contains no measured
> performance number yet.** The campaign plan (`fak.k31-campaign-plan.v1`) and
> the fold (`fak.k31-campaign-index.v1`) are contracts, not results: they define
> what a witnessed arm must look like and how per-arm receipts combine. Any
> tok/s, TTFT, ITL, or acceptance-length value that appears here before its
> per-arm `fak.serving-sweep.v1` artifact lands under this directory is an
> estimate, never a measurement.
>
> **Why the banner:** the sweep claim gate
> (`internal/webbench/serving_sweep_claim.go`) refuses peak/knee claims without
> a per-arm sweep artifact; this campaign inherits that gate unchanged. A plan
> row is not a result, and a missing arm folds as `missing`, never as zero.
>
> **Tracking:** campaign [#14374](https://github.com/anthony-chaudhary/fak/issues/14374)
> · spec arms [#14369](https://github.com/anthony-chaudhary/fak/issues/14369)
> · offload arm [#14372](https://github.com/anthony-chaudhary/fak/issues/14372)
> · spine map [docs/plans/k31-top-items-spine-2026-11-12.md](../../plans/k31-top-items-spine-2026-11-12.md)

Canonical plan schema: `fak.k31-campaign-plan.v1` (`internal/webbench/campaign_plan.go`, `K31CampaignPlan`)
Canonical index schema: `fak.k31-campaign-index.v1` (`FoldCampaignIndex`)
Per-arm receipt schema: `fak.serving-sweep.v1` (`internal/webbench/serving_sweep.go`)
CLI wiring: not wired yet — the `fak webbench campaign` surface is a filed follow-on leaf; the plan and fold are library seams today.

## 1. What this campaign is

One fixed arm matrix that measures the fak-native serving engine against
itself: how speculative decoding (#14369) and hot-expert CPU-RAM offload
(#14372) move throughput, p99 latency, and acceptance length across a
concurrency ladder. Every witnessed arm is exactly one `fak.serving-sweep.v1`
artifact (produced by `RunServingSweep` / `WriteServingSweepReport`) for that
arm; `FoldCampaignIndex` projects the per-arm receipts onto the plan matrix and
folds the comparable metrics into one index. The fold is pure and
deterministic: row order follows the plan, never the input order, and missing
arms fold as `missing` with a reason instead of a zero.

## 2. Arm matrix (12 planned sweep points + 3 gated placeholders)

| Arm | Speculative config | Concurrency | Status |
|---|---|---:|---|
| `spec-none` | `none` (no-spec baseline) | 1, 8, 32, 64 | PLANNED |
| `spec-dspark4` | `dspark:4` (DSpark drafter, block 4, #14369) | 1, 8, 32, 64 | PLANNED |
| `spec-dspark5` | `dspark:5` (DSpark drafter, block 5, #14369) | 1, 8, 32, 64 | PLANNED |
| `offload-h050` | hot-expert offload, h=0.50 (#14372) | TBD by #14372 | GATED on #14372 |
| `offload-h075` | hot-expert offload, h=0.75 (#14372) | TBD by #14372 | GATED on #14372 |
| `offload-h085` | hot-expert offload, h=0.85 (#14372) | TBD by #14372 | GATED on #14372 |

The three offload placeholders are not planned sweep points: their receipt
fields (the hot-expert hit-rate join) do not exist until #14372 lands them, so
`FoldCampaignIndex` excludes them from the index until then.

## 3. Metrics folded per (arm, concurrency) row

| Metric | Source in the per-arm receipt | Notes |
|---|---|---|
| `single_stream_tok_s` | the arm's c=1 ours-track point | repeated on every row of that arm (the single-stream anchor) |
| `aggregate_tok_s` | the row's own ours-track point | unit `usage.completion_tokens/s`, non-mixed basis `usage.completion_tokens` |
| `ttft_p99_ms` / `itl_p99_ms` | the row's own ours-track point | p99 quantiles from `ServingStats` |
| `acceptance_length` | additive spec join on the ours-track point | mean accepted tokens per verify round; source `fak_engine_spec_accepted_per_round` histogram sum/count (`internal/enginestep`) and the snapshot `SpecStat` |

Fold row statuses: `measured` (evaluator-valid point, spec arms also carry the
join), `present-without-spec-metrics` (valid point, spec arm without the join —
the number is never invented), `missing` (no report, or the sweep evaluator
refused the point — the typed invalidity reason is carried in the row).

## 4. Comparability contract

Every witnessed arm must share one model identity and one immutable workload
digest; `FoldCampaignIndex` refuses the fold otherwise. Engine identity may
differ across arms by design (the spec arms run a different engine
configuration and receipt). Each per-arm receipt must pass
`EvaluateServingSweep` unchanged — the fold only projects evaluator-certified
valid ours-track points, so a corrupt or above-capacity receipt folds as
`missing` with its typed reason, never as numbers.

## 5. Anchors to beat or explain (external, cited — not fak measurements)

These are prior published/reference numbers the witnessed arms will be read
against; they are context, not claims fak has made or verified:

- Kimi K3 on SGLang, 16xH200: **16.8 tok/s** (per-stream decode reference).
- NVIDIA GB300: **375.9 tok/s/user** (vendor reference point).
- Roofline: **~98% overhead share** (non-weight-read work dominates decode at
  this scale — the structural reason spec decode and offload are the levers).

## 6. Promotion gates

- Campaign closure and any peak/knee claim: **Promotion-requires #14368,
  #14371, #14370** (per the issue contract) — witnessed arms run after the
  serving spine, the FlashKDA prefill arm, and the FlashMLA decode arm exist.
- Offload arms: **gated on #14372** (advisor serve hookup + receipt fields).
- Per the bench-workstream doctrine, this campaign is a separate workstream
  from fixes: its receipts live here under `docs/benchmarks/k31-campaign/`,
  every number net-true end-to-end, no estimate promoted to a measurement
  without its per-arm artifact.

## 7. Witnessed-arm receipts (to be appended)

None yet. When an arm is witnessed, its `fak.serving-sweep.v1` artifact and
the folded campaign index land in this directory with the run's machine/model/
workload identity, and the arm's row in section 2 moves from PLANNED to
witnessed with a link. Nothing below this line is evidence of a measurement.
