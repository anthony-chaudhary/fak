---
title: "GLM-5.2 expert-parallel fit planning is rank-local (#2997): the per-rank residency pre-check charges one expert band, not the full model, per GPU"
description: "Host-side witness that `--expert-parallel N` capacity planning is sharded per rank (#2997). The primitive `compute.ExpertParallelPerRankPlan` (45fff3426) and the serve pre-check `refuseEPPlanIfUnfit` (8acb5ba1b, landed under the #971 EP-residency umbrella) partition resident weights into the replicated remainder plus the busiest rank's expert band, so an EP-N serve is graded against per-GPU capacity instead of the whole 433 GiB model. The public regression test `TestServeGGUFExpertParallelMemoryPlanChargesPerRankExpertBand` (cmd/fak/serve_memory_test.go) pins the invariant on synthetic weights: EP-4 charges one 1024 B expert band per rank, not the full 4096 B routed blob; a device sized for one band fits; an undersized device is refused with a *compute.FitError whose Want is the per-rank side. Numbers here are host-computed deterministic plan math, not a live A100 readback — the scrubbed remote-readback witness (#2997 Witness bullet 2) still needs the private route and is the sole open item."
---

# GLM-5.2 expert-parallel fit planning is rank-local (#2997)

> **What this is:** the host-side witness that `--expert-parallel N` no longer refuses
> each GPU as if it had to hold the whole model. #2997 reported an A100 EP route failing
> its capacity pre-check with `weights=433.82 GiB` **per rank** — the full model — even
> though each rank knows its expert slice (e.g. `[0,37)` … `[220,256)` of 256). The fix
> makes the per-rank fit plan charge only the **replicated remainder + this rank's expert
> band**. The numbers below are grounded in the `go test` exit code and the committed
> assertions, not a self-report. They are **host-computed deterministic plan math** on
> synthetic weights — **not** a live A100 capture; see §3 for the one witness that still is.

The implementation landed under the #971 EP-residency umbrella and was never referenced
back to #2997, so this note closes #2997 by ancestry:

- `45fff3426` `feat(compute)` — `ExpertParallelPerRankPlan` / `ExpertParallelPerRankWeightBytes` /
  `ExpertParallelLargestBandExperts`: the busiest rank holds the replicated weights plus
  `totalExpertWeightBytes × band / numExperts`, where `band = ceil(numExperts / ranks)`.
- `8acb5ba1b` `feat(serve)` — `refuseEPPlanIfUnfit` in `cmd/fak/serve.go`: after load it
  partitions the resident weights (`model.MoEResidentWeightBytes`), builds the busiest
  rank's per-card plan, and checks it against the device backend's **per-GPU** capacity at
  the same `serveGGUFDeviceHeadroom` (0.15) the load-time fit uses. Fail-open by
  construction: non-MoE, unaccountable, `ranks<=1`, nil backend, or unknown capacity all
  skip the check, so no host/single-GPU serve changes behavior.

## 1. The per-rank plan invariant — PASS

`TestServeGGUFExpertParallelMemoryPlanChargesPerRankExpertBand`
(`cmd/fak/serve_memory_test.go`) drives `serveGGUFExpertParallelMemoryPlan` at EP-4 on a
synthetic weight source whose routed `ffn_*_exps` blob is 4096 B across four experts, and
pins the resulting per-rank byte plan:

| Plan class / detail | bytes | meaning |
|---|---|---|
| `gguf-ep-replicated-load` | 3840 (=1024+512+256+2048) | dense/router/attention/shared-expert, **replicated every rank** |
| `gguf-ep-routed-expert-shard` | **1024** | **one** expert band — **not** the full 4096 B blob |
| per-rank weights (`MemoryWeights`) | 4864 | replicated + one band |
| kv_cache / activation / scratchpad | 3072 / 128 / 3584 | per-rank replicated (EP does not shard these) |
| `DeviceTotal()` | **11648** | the busiest rank's per-GPU footprint |

The load-bearing line is `gguf-ep-routed-expert-shard = 1024`, a **4× reduction** vs. the
4096 B full routed blob at EP-4 — the synthetic mirror of the real `433 GiB → replicated +
one band` reduction #2997 needed. Routed experts are classed resident, not host offload
(`MemoryOffload == 0`).

## 2. The refusal stays fail-closed and rank-local-accurate — PASS

Same test, against a probing device backend:

- device sized for one band (`14 << 10` = 14336 B) → **fits** (`fitServeGGUFExpertParallelOnDevice` returns nil).
- device too small for even one rank (`11 << 10` = 11264 B) → **refused** with a
  `*compute.FitError` whose `Want == 11648` — the **per-rank** device side, not the full
  model. The residual blocker an operator sees is the corrected rank-local number.

This is #2997 Done-condition **arm (2)** — *"refuses with a rank-local plan whose weights
are correctly reduced to the rank shard and whose residual blocker is accurate."*

## 3. What this closes, and the one witness still open

**Closed (host side):** #2997 In-scope items 1–3 (per-rank EP capacity planning,
preserved fail-closed behavior, public regression test) are landed and green. Acceptance
gate `go test ./internal/model ./internal/compute ./internal/ggufload ./cmd/fak -run
"Expert|Parallel|Fit|GLM|Serve" -count=1` passes; the named test is `--- PASS`.

**Still open (do not over-claim):** #2997 **Witness bullet 2** — a *scrubbed remote
readback from the EP route candidate showing a corrected rank-local refusal (or endpoint
readiness)* — requires the private A100 route running the fixed binary and **cannot be
produced from a build host**. It is tracked as a follow-up and is the sole remaining item;
Done-condition **arm (1)** (bind + serve a scrubbed `/v1/models` + tiny chat witness) is
likewise a GPU-route deliverable, out of scope here.
