---
title: "Ablate: Self-Ablation Feature Sweep (Deterministic Core)"
description: "The deterministic, $0, no-model half of the self-ablation benchmark harness (epic #607): one frozen tool-call trace replayed under N feature configs, deltas read straight off the kernel counters."
---

# ABLATE-RESULTS — the deterministic self-ablation sweep (`fak ablate`)

> This is **Regime A** of the self-ablation benchmark harness (epic
> [#607](https://github.com/anthony-chaudhary/fak/issues/607)): a *kernel-feature*
> ablation. It asks "what does feature X cost/save?" by varying **one `FAK_*`
> knob** while holding the model, the task, **and the tool calls** constant.
> That makes it the **exact-workload** twin of the cross-agent ablation (Regime B
> — pure fak vs Claude Code / ultracode), which varies the whole agent + model
> and can only be scored distributionally. The `Trace.WorkloadHash` equality guard
> that makes this regime ironclad is *exactly wrong* for Regime B; the two are
> **two harnesses sharing one record schema**, and Regime B is not claimed here.

## What shipped (rung 1)

`fak ablate` generalizes the 2-arm `fak bench` (vDSO on/off) into an N-ARM
matrix: replay ONE frozen tool-call trace through each feature config and emit
one `AblationRun` per arm, every arm bound to the trace's single workload hash
by an N-arm identical-workload guard (`ablate.Report.Validate`, generalizing
`metrics.Report.Validate` from a fixed pair to N). The deltas are
apples-to-apples by construction — every arm ran the same work.

## The committed artifact — `tau2-airline-smoke`, vDSO on/off

The minimal first experiment the epic names: a deterministic feature-sweep over
one frozen `tau2` trace, committed as a reproducible artifact at zero cost.

**Artifact:** [`experiments/ablate/tau2-smoke-vdso-ablation.json`](../../experiments/ablate/tau2-smoke-vdso-ablation.json)
**Reproduce:** `go run ./cmd/fak ablate --trace testdata/tau2/tau2-smoke.json --sweep vdso`

| arm | features | calls | vdso_hits | engine_calls | denies | quar | tokens | Δ tokens |
|---|---|---|---|---|---|---|---|---|
| all-off (baseline) | vdso=off | 12 | 0 | 12 | 0 | 0 | 937 | — |
| vdso | vdso=on | 12 | 7 | 5 | 0 | 0 | 417 | **−520** |

The vDSO fast path serves **7 of 12** repeated calls from cache, so only **5**
reach the engine (12 → 5 `engine_calls`); the 7 cached decisions carry a bounded
verdict instead of the full result, cutting **520 tokens** (937 → 417) from the
model context on this 12-call trace. `denies`/`quarantines` are 0 on both arms —
this read-heavy airline trace triggers neither the deny floor nor a quarantine,
so those counters are an honest zero (the vDSO is the only feature this trace
exercises).

## Reproducibility — what is and is not machine-bound

The **counter fields reproduce byte-identical** on any host:
`workload_hash`, `calls`, `vdso_hits`, `engine_calls`, `denies`,
`quarantines`, `input_tokens`, `output_tokens` — they are read from the kernel's
own deterministic event counters on a frozen trace, with no model and no decode.

The **timing fields are single-box and NOT cited as a headline**:
`p50_ns`, `p99_ns`, `mean_ns`, the latency `buckets`, and `wall_seconds` are
clock measurements on the build host (Windows, go1.26.3 here). They are
committed for completeness and fenced as illustrative, exactly the way the
model-ladder wall-clocks are single-box while the deterministic token/hit-rate
metrics reproduce exactly (see
[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) §"The model-ladder thesis").

## Honesty fences (what this rung does NOT measure)

- **One runtime knob.** Rung 1 sweeps only the runtime-settable vDSO toggle
  (`kernel.SetVDSO`, reused via `bench.RunArm`). The ~40 env-gated features
  (`FAK_NORMGATE` / `FAK_INKERNEL_RADIX` / `FAK_COMPRESSOR` / …) are read at
  process start; sweeping them needs a subprocess-re-exec rung (the next child
  issue on the epic). `BuildSweep` fails loud on a non-runtime feature rather
  than silently measuring nothing.
- **No cross-agent arm.** Regime B (pure fak vs Claude Code / ultracode) needs a
  live model + API tokens and a distributional validity contract (success-gate,
  N-run variance, model-named baseline, decomposed cache). It is a separate
  harness on the epic and is **not** claimed here.
- **Two regimes, two numbers.** This document reports only kernel-efficiency
  (Regime A). It never blends into an agent-capability claim.

## Witness

`go test ./internal/ablate ./cmd/fak` — `TestSweep_VDSO_NArmGuardAndIsolatedDelta`,
`TestValidate_RefusesMismatchedWorkloadHash`, `TestBuildSweep_UnknownAndDuplicate`,
`TestAblateJSONReport`, `TestAblateUnknownFeatureUsageError`. On a native-Windows
host run the suite under WSL (`./test.ps1`); `go build` / `go vet` are native.
