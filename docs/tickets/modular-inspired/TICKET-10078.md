---
lane: webbench
paths:
  - internal/webbench/serving_sweep.go
  - internal/webbench/serving_sweep_test.go
  - experiments/benchmark/runs/by-machine/modular-10078-live/
expected_steps: 5
issue: https://github.com/anthony-chaudhary/fak/issues/10078
---
<!-- fak-webbench-key: modular-qwen38-capacity-10078 -->

# TICKET-10078 - Native Qwen3.8 capacity evidence

## Current state

The 2026-09-10 physical run produced an invalid qualification: an earlier
per-arm capacity of 16 was omitted from the original downstream scheduler
manifest, the 32-request point received 16 HTTP 429 refusals, and total observed
setup elapsed time exceeded measurement time. Successful outputs were exact;
native identity and model/binary hashes were stable before and after the run.

Evidence: `experiments/benchmark/runs/by-machine/modular-10078-live/README.md`.
This ticket tracks the existing GitHub issue; it does not create a duplicate.

## Core through-line

Sanctioned native endpoint -> immutable model/engine snapshot -> fixed workload
and bounded hardware capture -> effective-capacity readback -> typed invalid
qualification -> reject partial-failure points -> independently verified landing.

## Gold-plating boundary

One model and one GPU endpoint. No kernel, scheduler, arm-limit, configuration,
deployment, comparator, dashboard, or credential changes. Keep all refused
requests visible. Do not certify throughput or an SLA knee.

## Done condition

- [x] Capture real `fak.serving-sweep.v1` point data and exact-output evidence.
- [x] Record native identity, model/binary hashes, system state, and setup costs.
- [x] Preserve the original receipt and explain the two admission layers.
- [x] Classify the capture through the issue's disconfirming-witness branch.
- [x] Independently verify the partial-failure validation guard for landing.
- Remote commit/artifact ancestry is checked after landing and recorded on the linked GitHub issue.
- [ ] A positive capacity-valid peak/knee is established (not claimed by this run).

## Witness

Physical source: `fak@d3dd8447de3707afbc05bc450812cbce3858b66b`.
Capture library: `fak@6e1c5eb7c4c998211ba6764b7314e3eff0682594`.

The original evaluator marks a point with Requests=32, OK=16, Failed=16 and
positive throughput valid. The focused deterministic regression is
`go test ./internal/webbench -run 'ServingSweep.*Partial' -count=1`.
Package validation is `go test ./internal/webbench -count=1`.

The software guard is confined to `EvaluateServingSweep`; invalidating a partial
measurement preserves diagnostic statistics while excluding it from peak/knee
claims. Hardware refusals remain enforced, and other serving lanes are untouched.

## Likely files

The routing paths above and this local ticket.

## Lane

`webbench`; public benchmark evidence and validation.
