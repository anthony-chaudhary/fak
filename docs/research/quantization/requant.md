---
title: "ReQuant fixed-grid refinement evaluation"
description: "Status: modeled contract witness, not an observed model- or hardware-performance result."
---
# ReQuant fixed-grid refinement evaluation

Status: **modeled contract witness**, not an observed model- or hardware-performance result.  
Issue: [#6253](https://github.com/anthony-chaudhary/fak/issues/6253), child of #6221.

## Pinned research and implementation envelope

| Component | Pin | Provenance |
|---|---|---|
| Research artifact | `arXiv:2608.07019v1` | PDF SHA-256 `5505cab5060f170e5e7a03b07fce83682c7af3bb0c3cfbb9cdb5920c220d4beb`; Yongge Ma et al., submitted 2026-08-07 |
| Neutral contract | `requanteval/v1` | `internal/requanteval/contract.go` |
| Recipe | `requant-arxiv-2608.07019v1-fixed-grid-v1` | deterministic fixed-grid coordinate refinement modeled from Algorithm 1 |
| Runtime | `fak-go-reference-cpu/v1` | standard-library Go objective evaluator; not a production inference runtime |
| Initializer | fixture-pinned, currently `round-to-nearest@fixture-v1` | carried in every result alongside artifact ID/version/digest/source |

ReQuant is a post-processing method: it starts from an already quantized assignment, keeps the original quantization grid fixed, and accepts discrete coordinate changes that reduce a reconstruction objective. This leaf evaluates that defining seam against the **same initial codes and grid**. It does not create a fak artifact format, rank PTQ methods, implement a model kernel, or silently substitute another runtime.

## Typed adjudication

`requanteval.Evaluate` always returns one of:

- `supported / REQUANT_EVALUATED`: the pinned reference recipe ran, returning the original and refined codes, initial/final reconstruction MSE, same-example synthetic prediction MSE, gains, sweeps, coordinates visited, candidate evaluations, and accepted updates.
- `unsupported`: unknown contract/recipe, malformed fixed grid or shape, or incomplete artifact/initializer provenance is refused with a public reason code.
- `delegate / REQUANT_RUNTIME_DELEGATION`: a known request that names a runtime other than the pinned CPU evaluator is handed to that runtime rather than emulated or silently downgraded.

The v1 evaluator is deliberately bounded (2–256 finite ordered grid levels, 1–4096 coordinates, 1–1000 sweeps). The seed is captured for reproduction, although this deterministic recipe consumes no random numbers.

## Named witness

The independently readable golden files under `internal/requanteval/testdata/` are the reproduction artifact. Together they cover three cases on Windows and WSL/Linux:

1. `improves.json` — a coupled two-coordinate quadratic where refinement starts from the same RTN assignment, remains on `[-1,0,1]`, and lowers modeled reconstruction MSE from `0.164` to `0.144` (12.20%); the same pinned three-example linear probe exposes a counterexample—prediction MSE worsens from `0.08` to `0.7467`, so lower reconstruction loss is not reported as a universal quality gain; the artifact records seed, grid, initial/refined codes, sweeps, conversion work, metrics, provenance, and claim-check result.
2. `stable.json` — an initialization for which no strictly improving fixed-grid update is accepted.
3. `delegate.json` — an unknown provider runtime produces a typed delegation and no fabricated metrics.

Run:

```bash
go test ./internal/requanteval -run 'TestEvaluateWitnessFixtures|TestTypedUnsupportedOutcomes|TestSameInitializationAndFixedGrid' -count=1
```

The golden read-back fails if recipe behavior, cost accounting, pins, evidence class, or adjudication drifts.

## Evidence boundary and claim check

The fixture's reconstruction MSE, synthetic prediction-quality MSE, and conversion-operation counts are **observed outputs of the deterministic evaluator**, but they describe a synthetic quadratic. Therefore the overall evidence classification is `modeled`, and its embedded claim-check verdict is `not-yet` for downstream model quality or hardware performance. `hardware_envelope` explicitly says that no wall time, throughput, GPU, or model-quality run was measured.

The paper reports broad model/task improvements and runtime overheads in its own experimental envelope (including RTN, GPTQ/GPTAQ and AWQ initializers), but fak has not independently reproduced those values. They are research provenance, not fak-observed results, and are intentionally not copied into a performance claim. A future observed claim must pin the weights/dataset/calibration recipe/runtime/device/software stack, compare the same quantized initialization and format, capture downstream quality plus conversion wall time, and pass `fak claim-check` against the tuned alternative.
