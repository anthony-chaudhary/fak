---
title: "Lane ledger: bottom-up model of one decode launch"
description: "What the fak bench lane-ledger verb family models, which inputs are measured and which are model-only, and the limits of each."
---

# Lane ledger: bottom-up model of one decode launch

This note explains the `fak bench lane-ledger` verb family as implemented in
`internal/compute/lane_ledger.go` and `cmd/fak/bench_lane_ledger*.go`. The verb reference is in [cli-reference](../cli-reference.md#bench-lane-ledger-model-only-lane-slot-partition-of-one-decode-launch).

## What a lane slot is

A lane slot is one warp lane on one streaming multiprocessor (SM) during one wave.
The unit is dimensionless: it carries no clock and no time. A launch of `G` blocks on
a device with `D` SMs runs in `waves = ceil(G / (D * B))` waves, where `B` is the number
of blocks one SM can hold, computed by the occupancy arithmetic in
`internal/compute/decode_occupancy.go`. Capacity is

    capacity = waves * D * lanes_per_sm

with `lanes_per_sm = MaxWarpsPerSM * WarpSize` taken from the arch row.

## The cells

Every lane slot is classified into exactly one of five cells:

- `active`: a lane holding a thread of a block resident on its SM.
- `partial_warp`: a lane of a warp that `threads_per_block` does not fill.
- `resident_unfilled`: a lane in a block slot the SM could hold but this wave does not fill.
- `resource_capped`: a lane that the binding per-SM limit (registers, shared memory,
  warps or blocks) keeps out of every resident block.
- `unit_idle`: a lane on an SM that receives no block in this wave.

## Conservation

The cells partition the capacity exactly. `LaneLedger.Conserved()` reports true when
every cell is non-negative and their sum equals `capacity`. The findings are the non-zero
losses, and each lost lane-slot-wave belongs to exactly one finding, so the findings sum
to `capacity - active`. The findings are `underfill` (single-wave launches), `wave_tail`
(the final wave's idle and unfilled lanes), `resident_unfilled`, `resource_capped` and
`partial_warp`.

The register file is partitioned separately, in registers per SM rather than lane slots,
so its findings are a different type and are never added to the lane-slot total.

## Measured inputs and model-only parts

Measured or caller-supplied inputs:

- Device properties: `--device-props` reads a CUDA probe JSON and takes the SM count
  and compute capability from it, then checks the arch row field by field. A disagreement
  refuses the run. Without that file, `--sms` is a caller input with no default.
- Registers per thread: `--regs` must come from a compiler or profiler measurement. The
  sweep and observed verbs read it from an `nvcc -Xptxas -v` log.
- Shared memory: `--smem` is dynamic bytes per block, a caller input; the default verb has
  no static shared-memory input. The sweep and observed verbs read static bytes from the
  ptxas log. Sweep does not price dynamic shared memory: a launch whose dynamic expression
  is not the literal 0 is flagged `smem_dynamic_unknown`.
- Measured window (default verb only, with `--cycles-per-sm`): the window length on one SM,
  optional `--wave-weights` (cycles per modelled wave), optional `--active-lane-cycles`
  (an integral of occupied lanes), and `--tolerance`.
- Observed profile (`observed`): per-launch-shape kernel time as a JSON array.

Model-only parts:

- The placement assumes one block per SM before any SM takes a second.
- Waves are static. A slot freed inside a wave is not reused until the wave finishes, so
  `wave_tail` is an upper bound on what a dynamic scheduler would lose.
- Allocation granularities (register allocation unit, warp allocation granularity,
  shared-memory allocation unit) are occupancy-calculator values, not vendor-published
  figures, and the device does not report them.
- With no `--wave-weights`, equal-length waves are assumed, and the output says so.

## Measured window

With `--cycles-per-sm`, each model cell is multiplied by the cycles of the waves that carry
it, which gives lane-slot-cycles. The full-wave shape is the same in every full wave, so the
window needs only the full-wave total and the final-wave cycles. Any capacity no modelled
wave covers is reported as `unattributed`, and the model names no cause for it.

If `--active-lane-cycles` is given, the model's active lane-slot-cycles are compared with the
measured value. The shortfall is a plain difference. Within `--tolerance` the verdict is
`consistent`. Outside it the verdict is `model_overstates_active` (the model calls more lanes
active than were measured) or `model_understates_active`, and the matching
`unexplained_inactive` or `unexplained_active` finding carries the residual. Neither is
assigned a stall cause.

## Observed profile

`observed` joins a timestamp profile to the modelled cells per launch shape. Each row
either gets a model or a reason it cannot be modelled, and the two totals add back to the
profile total exactly. `time_weighted_idle_ns` is the row's observed time minus its
active-cell share of that time, which is the observed time times the modelled idle fraction
up to whole-nanosecond rounding. The row's time is split across cells in proportion to
lane-slot-waves, a model assumption. The observed time is a device event-pair time, so it
includes the launch gap and short kernels are overstated. `time_weighted_idle_ns` is a
prioritisation heuristic for ordering the losses, and the report says so. It is not a
measurement of idle time.

## Honest limits

- No stall causes. The model partitions lane slots; it does not say why a lane is idle
  or inactive. The unexplained residual is reported without a cause.
- No measured occupancy without a profiler. The active-lane integral is a caller input. The
  model does not observe the hardware.
- Time-weighted idle is a heuristic: observed time multiplied by a modelled fraction, used
  for ordering.
- The SM count and register counts are caller inputs unless the probe file or the ptxas
  log supplies them. A wrong SM count rescales every share.
- The sites and fill verbs are static or caller-driven views. Neither measures anything.

## Worked example

One launch of 84 blocks of 256 threads at 137 registers per thread, on a device whose probe
file reports 84 multiprocessors at compute capability 12.0 (48 warps, 65536 registers and
24 blocks per multiprocessor). The register count is from an `nvcc -Xptxas -v` log.

    fak bench lane-ledger --device-props device-props.json --threads 256 --regs 137 --grid 84

The verb reports `blocks/SM=1 limiter=register waves=1 capacity=129024 block_slots_unfilled=0 conserved=true`.
By hand: a block is 8 warps; one warp needs `137 * 32 = 4384` registers, rounded up to the
256-register unit as 4608; `65536 / 4608` fits 14 warps, rounded down to the 4-warp
granularity as 12; 12 warps hold one 8-warp block. Capacity is `1 * 84 * 1536 = 129024`
lane-slot-waves, of which `84 * 256 = 21504` are `active` (share 0.1667) and the remaining
107520 are `resource_capped` (share 0.8333). The register file per multiprocessor splits as
useful 35072, rounding 1792, granularity 9216, headroom 18432 and remainder 1024, which sum
to 65536.

This is a predicted loss on one probe shape, one block per multiprocessor. It is not a
measured occupancy and not a speedup.
