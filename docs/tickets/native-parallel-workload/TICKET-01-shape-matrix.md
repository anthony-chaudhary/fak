<!-- fak-model-key: f32-decode-projection-shape-matrix-v1 -->
<!-- fak-public-issue: anthony-chaudhary/fak#12702 -->

# bench(model): baseline parallel f32 decode across SmolLM2 projection shapes

## Current state

At public `fak` commit `828ff563c25c1213d1a432c29552dd43eaacc966`,
`internal/model` has a parallel f32 row-projection path and
`TestParallelMatchesSerial`, which independently checks the relevant projection
shapes bit-for-bit against the serial implementation. The package does not have
a single stable benchmark that measures the five distinct projection workloads
from the documented SmolLM2-135M architecture under a frozen CPU worker setup.

The missing baseline covers five unique shape classes: `q_o_proj` (`576x576`),
`k_v_proj` (`192x576`), `gate_up_proj` (`1536x576`), `down_proj`
(`576x1536`), and `lm_head` (`49152x576`). These are CPU f32
decode projection shapes; this ticket does not establish end-to-end model or
agent performance.

## Classification

- Portfolio tier: 2 — native model serving.
- Centrality: Core.
- Work unit: S1 leaf.
- Priority: P2.

## Problem frame

- **Centrality:** Projection work is on the public native model execution path,
  and these five shapes cover the materially different SmolLM2 decode matrix
  geometries already exercised by the correctness witness.
- P1: preserved — benchmark-only coverage does not change model output or APIs.
- P2: advanced — produce a reproducible CPU f32 baseline for later, separately
  scoped optimization decisions.
- P3: preserved — the benchmark uses a fixed four-worker envelope and creates no
  new runtime configuration.
- P4: preserved — results are moment-in-time component measurements, with no GPU,
  task-quality, end-to-end latency, or default-promotion claim.

## Core through-line

Five documented SmolLM2 projection shapes -> existing parallel f32 row engine ->
Go benchmark measurements under a frozen worker envelope -> reproducible stdout
receipt, guarded by the existing serial-parity test.

## Parent context

This leaf supports the public milestone in `docs/local-agent-milestone.md` by
measuring one native CPU component before any optimization is proposed. The
model dimensions come from `docs/benchmarks/IN-KERNEL-MODEL-RESULTS.md`:
hidden size 576, three KV heads at head dimension 64, intermediate size 1536,
and vocabulary size 49152.

## Why this is next

The five shapes have different output widths and work sizes, so a single matrix
shape cannot show whether parallel row scheduling behaves consistently
across attention, MLP, and vocabulary projections. A frozen shape matrix is the
smallest evidence needed before considering any candidate tuning.

## Working spine

`BenchmarkF32DecodeMatRowsSmolLM2Shapes` -> named sub-benchmark for each of
q/o, k/v, gate/up, down, and lm_head -> existing parallel projection operation ->
`go test -bench` timing and allocation output.

## Gold-plating boundary

Do not add or change runtime constants, worker-selection policy, tuning
candidates, production defaults, projection implementations, GPU benchmarks, or
end-to-end model/task benchmarks. Do not interpret this component baseline as a
task-quality, user-latency, hardware-qualification, or promotion claim.

## Done condition

- [x] Add `BenchmarkF32DecodeMatRowsSmolLM2Shapes` in
      `internal/model/parallel_workload_bench_test.go`.
- [x] Include exactly these five named shape classes: `q_o_proj` `576x576`,
      `k_v_proj` `192x576`, `gate_up_proj` `1536x576`, `down_proj`
      `576x1536`, and `lm_head` `49152x576`.
- [x] Exercise the existing parallel f32 row-projection operation without adding
      runtime constants or changing production behavior.
- [x] Preserve the existing bit-for-bit serial-parity witness for all five shapes.
- [x] Run the frozen seven-sample benchmark and retain its complete stdout with
      the source revision and command when using it as future evidence.
- [ ] Pass the full affected-package test gate.

## Definition of Done

- [x] The benchmark names make each projection class and dimensions identifiable.
- [x] The benchmark reports timing and allocations through Go's standard
      benchmark output for every shape.
- [ ] Correctness and the full `internal/model` package suite pass with the frozen
      environment.

## Acceptance gate

The independent serial-parity test passes, benchmark stdout contains all five
named results with seven samples each at 500 ms per sample and allocation
reporting, and the full affected package passes, all with `GOMAXPROCS=4`, `FAK_WORKERS=4`,
`CGO_ENABLED=0`, `GOENV=off`, and `GOWORK=off`.

## Closure binding

Close the public issue only from the DCO-signed resolving commit after all three
frozen witness commands pass. The benchmark receipt is baseline evidence only;
any tuning or default-promotion work requires a separate issue and evidence.

## Witness

The existing `TestParallelMatchesSerial` is the independent correctness oracle
for all five shapes. Run the following from the public `fak` repository:

### Verifiable Witness

```bash
GOMAXPROCS=4 FAK_WORKERS=4 CGO_ENABLED=0 GOENV=off GOWORK=off go test ./internal/model -run '^TestParallelMatchesSerial$' -count=1 -v
GOMAXPROCS=4 FAK_WORKERS=4 CGO_ENABLED=0 GOENV=off GOWORK=off go test ./internal/model -run '^$' -bench '^BenchmarkF32DecodeMatRowsSmolLM2Shapes$' -benchmem -benchtime=500ms -count=7
GOMAXPROCS=4 FAK_WORKERS=4 CGO_ENABLED=0 GOENV=off GOWORK=off go test ./internal/model -count=1
```

The benchmark command exiting zero is insufficient by itself: its retained
stdout must contain seven result lines for each of `q_o_proj`, `k_v_proj`,
`gate_up_proj`, `down_proj`, and `lm_head`. This component matrix does not make
the current single-benchmark RSI gate representative of the whole matrix;
matrix-level integration and independent model/task acceptance remain later,
separately scoped work.

## Validation status (2026-09-09)

The targeted `TestParallelMatchesSerial` oracle and `go vet ./internal/model`
pass under the frozen CPU environment. The benchmark receipt contains seven
samples for each of the five required named shapes (`5x7` total).

The full `internal/model` package gate remains red. The same failures reproduce
at base `828ff563c25c1213d1a432c29552dd43eaacc966` with the new benchmark file
excluded, so they are not caused by this benchmark:

- `TestQwen35PrefillAndStepCPUContinuation`: non-finite Q2_K prefill logit,
  tracked by [#12433](https://github.com/anthony-chaudhary/fak/issues/12433).
- `TestQwen35LinearAttnBatchedResumesState`: f32 bit-parity mismatch.
- `TestQwen35LinearAttnBatchedMatchesScalar`: f32 bit-parity mismatch.
- `TestQwen35ChunkedBatchedMatchesScalar`: f32 bit-parity mismatch.

The three GDN parity failures are recorded under
[#12038](https://github.com/anthony-chaudhary/fak/issues/12038), with resumed-state
scope also tracked by [#11987](https://github.com/anthony-chaudhary/fak/issues/11987).
The full-suite acceptance criterion stays unchecked, and this issue stays open.

## Likely files

- `internal/model/parallel_workload_bench_test.go`
- `docs/tickets/native-parallel-workload/TICKET-01-shape-matrix.md`

## Blast radius and affected lanes

- Affected: `internal/model` benchmark-only test code.
- Unaffected: runtime configuration and defaults, model numerics, public APIs,
  GPU backends, serving routes, agent workflows, and task-quality evaluation.
- Fallback: production behavior remains unchanged if the benchmark is absent or
  skipped; no quarantine or runtime shim is required for benchmark-only code.

## Lane

model

## Expected steps

4

```routing
lane: model
paths: internal/model/parallel_workload_bench_test.go, docs/tickets/native-parallel-workload/TICKET-01-shape-matrix.md
expected_steps: 4
```

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points.

## Completion standard

development

## Tracking

Tracked by [#12702](https://github.com/anthony-chaudhary/fak/issues/12702).
