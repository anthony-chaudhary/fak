---
title: "GPU clock cycles: what a kernel costs in the unit the chip counts"
description: "How fak turns per-kernel GPU time into SM clock cycles, how to read the cycle fields in a perf record, and how to reason about cycle budgets per token."
slug: gpu-cycles
keywords:
  - gpu clock cycles
  - cycles per kernel call
  - cycles per token
  - sm clock mhz
  - kernel cost in cycles
  - bytes per cycle
  - FAK_GPU_SM_CLOCK_MHZ
date: 2026-11-24
---

# GPU clock cycles: what a kernel costs in the unit the chip counts

Milliseconds tell you how long something took. Cycles tell you how much of the
chip's work budget it consumed, and they compare across clocks: the same kernel
reads 25 µs at a 2,000 MHz clock and 20 µs at 2,500 MHz, but costs roughly
the same 50,000 cycles at both. When you reason about whether a kernel is near
its floor, think in cycles.

## The conversion

```
cycles = GPU nanoseconds × SM clock MHz / 1000
```

One microsecond at 2,500 MHz is 2,500 cycles. A cycle here is one tick of the
device's SM (shader) clock, elapsed wall-clock cycles. It is **not** summed across
SMs: a kernel that keeps 84 SMs busy for 10,000 cycles costs 10,000 cycles of
elapsed time, and 840,000 SM-cycles of capacity.

## Where agents read it

Every native perf record's kernel table (`native_timing.kernels`) carries cycles
next to time whenever the clock is known:

| Field | Level | Meaning |
|---|---|---|
| `sm_clock_mhz` | table | The clock every cycle field was derived from |
| `clock_source` | table | `observed` (the backend read the device) or `declared` (the operator set it) |
| `gpu_mcycles_per_token` | table | Timed GPU work per decode token, in millions of cycles |
| `cycles_per_call` | kernel row | One launch of this kernel shape, in cycles |
| `mcycles_per_token` | kernel row | This kernel's share of a token, in millions of cycles |

When the clock is unknown the cycle fields are **omitted**, never zero. A missing
`cycles_per_call` means "no clock", not "free".

The code: `compute.SMClockFor` resolves the clock
([`internal/compute/sm_clock.go`](../../internal/compute/sm_clock.go)), and
`perfledger.NewNativeKernels` derives the cycle fields
([`internal/perfledger/kernels.go`](../../internal/perfledger/kernels.go)).

## Getting a clock

A backend that registers `compute.RegisterSMClockReader` supplies an `observed`
clock. Until one does, declare it:

```bash
nvidia-smi -lgc 2617,2617          # lock the SM clock so the declaration is true
export FAK_GPU_SM_CLOCK_MHZ=2617   # then run the profile-ON arm
```

A declared clock is not checked against the device. Without a locked clock the
GPU boosts and throttles, so a declared value is only an estimate. Record
`nvidia-smi --query-gpu=clocks.sm --format=csv` before and after the run next to
the receipt. Values outside 50–10,000 MHz are refused, so a kHz value does not
silently scale every count by 1,000.

## Reasoning in cycles

**The token budget.** At 100 tok/s a token gets 10 ms, which is 26.2 M cycles at
2,617 MHz. `gpu_mcycles_per_token` tells you how much of that the GPU spent working.
The rest is host time, syncs and idle.

**One kernel launch.** A 17.8 µs GEMV at 2,617 MHz is about 46,600 cycles. If that
kernel runs 74 times per token, it costs 3.4 M cycles per token, 13% of the budget
above.

**Bytes per cycle, the floor.** Divide device memory bandwidth by the clock: 960 GB/s
at 2.6 GHz is about 370 bytes per cycle device-wide. A GEMV that must read 8 MiB of
weights cannot finish in fewer than about 22,700 cycles. A measured `cycles_per_call`
far above that floor has headroom; one close to it is bandwidth-bound.

**Fixed costs are cycles too.** The figures below are typical published orders of
magnitude, not measurements from this repository. They vary by architecture, so
measure before relying on any of them:

| Event | Typical order of magnitude |
|---|---|
| Shared-memory access | tens of cycles |
| L2 hit | a few hundred cycles |
| DRAM access latency | several hundred cycles |
| Kernel launch overhead | a few µs, roughly 5,000–15,000 cycles at 2.5 GHz |
| Host round-trip sync | tens of µs, roughly 25,000+ cycles |

A tiny kernel that does 2,000 cycles of work behind a 10,000-cycle launch is
launch-bound. Fusing it or capturing a graph is the lever, not tuning its math.

## What cycles are not

- **Not a speed claim.** The acceptance for a perf change is still the profile-OFF
  token-time delta (decode tok/s, ms/token, prefill tok/s or TTFT). Cycles come
  from the profile-ON arm, like `selected_kernels`, and are secondary evidence.
- **Not hardware counters.** Cycles here are timestamp time × one clock per decode
  window. Per-kernel `sm__cycles_elapsed` from a profiler is a separate,
  finer measurement.

## Checkpoint

1. A kernel row shows `us_per_call: 40` and no `cycles_per_call`. What does it cost
   in cycles? *Unknown: the clock was not known for that window. It is not zero.*
2. The same kernel reads 40 µs at 1,500 MHz and 25 µs at 2,400 MHz. Did it get
   faster per cycle? *No. Both are 60,000 cycles; only the clock changed.*
3. `clock_source: declared` and the GPU was not clock-locked. How far can you trust
   `cycles_per_call`? *As far as the boost clock stayed near the declared value.
   Check the recorded `clocks.sm` samples.*
