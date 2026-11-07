---
title: "fak optimize — the inference optimization driver"
description: "Plan, measure, verify, and loop over the committed native-performance lever graph with bounded, fail-closed verbs."
---

# fak optimize — the inference optimization driver

`fak optimize` drives the optimization loop itself. The planning and evidence
halves already exist as separate verbs: `fak native-performance` holds the
committed lever graph and judges receipts, `fak ablate` runs two-arm sweeps over
frozen traces, and `fak quality run` is the output oracle. `fak optimize`
composes them into one bounded loop.

```bash
fak optimize plan --json
fak optimize measure --lever ID --trace FILE --out receipt.json --json
fak optimize verify --baseline base.json --candidate cand.json [--quality q.json] --json
fak optimize loop --max-iters N [--dry-run] --json
```

## Subcommands

| Subcommand | Effect | What it does |
|---|---|---|
| `plan [--profile FILE]` | read | Selects the first dependency-ready, unwitnessed lever from `nativeperf.ActiveGraph()` (or the lever the profile classifier names) and emits `fak-optimize-plan/v1` with the lever, its envelope, its expected floor..roof band, and the exact `measure` and `verify` commands to run next. |
| `measure --lever ID --trace FILE` | writes only `--out` | Replays the frozen trace with the lever's mapped `ablate` feature off and on, and emits `fak-optimize-receipt/v1` (baseline and candidate tok/s, delta, trace SHA-256, repetitions). |
| `verify --baseline FILE --candidate FILE` | read | Judges the pair under the `nativeperf` comparison criterion when the receipts are native experiment receipts, otherwise against the lever's planning band. A failing `--quality` result forces reject. Emits `fak-optimize-verdict/v1`. |
| `loop --max-iters N` | bounded | Runs plan, measure, verify per iteration and prints one JSONL ledger line each. `--dry-run` stops after plan. The loop stops at the first reject so unverified gains never stack. |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success; for `verify`, the candidate was accepted. |
| 1 | `plan` found no ready lever; `verify` rejected the candidate. |
| 2 | Typed refusal: unreadable input, unknown lever, `--max-iters <= 0`, or a lever with no measurable arm. |

## Refusals are the contract, not a failure mode

`measure` never reports a number it did not measure. A refusal carries one
closed class and a detail line:

| Class | When |
|---|---|
| `no_ready_lever` | `plan` found no dependency-ready, unwitnessed lever in the graph. |
| `unknown_lever` | The lever ID is not in the committed graph. |
| `trace_unreadable` | The `--trace` file is missing or cannot be loaded. |
| `no_arm_mapping` | The lever has no trace-backed arm. Every lever in the committed graph today is a Metal or CUDA hardware lever outside `ablate`'s runtime features, so `measure` refuses all of them until a hardware arm exists. |
| `invalid_input` | A receipt or quality file cannot be decoded. |
| `non_positive_iterations` | `loop --max-iters` is zero or negative. |
| `no_planning_band` | `verify` has no registered band to judge the lever against. |
| `criterion_unresolved` | `verify` cannot resolve the `nativeperf` comparison criterion for a native receipt pair. |

## Current limit

The loop can plan but cannot yet climb: no committed lever has a measurable arm,
so `loop` without `--dry-run` records each iteration as refused. Measuring the
hardware levers needs receipts from a physical device for an envelope that
matches that device. A Metal, A100, or RTX 5090 envelope is a separate curve and
receipts from one never judge another.

## Related

- [Native raw-model performance hill climb](NATIVE-PERFORMANCE-HILLCLIMB.md) — the committed lever graph `plan` reads.
- [Native performance regression gate](NATIVE-PERFORMANCE-REGRESSION-GATE.md) — the receipt policy `verify` reuses.
