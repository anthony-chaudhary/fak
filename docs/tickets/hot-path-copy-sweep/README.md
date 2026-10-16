# Hot-path copy sweep (2026-10-16)

A sweep for the bug class behind the cpu-ref whole-weight copy (fixed in `19eab47217`): per-call copies or conversions of static data, per-token allocations, unguarded tracing, and fast paths that exist but are never enabled.

## How the copy shipped (root cause)

1. `12dc1386da` (2026-06-23) added cpu-ref Q4_K MatMul as a correctness reference. `i8AsBytes` (`make` plus a byte loop) was the only "safe" reinterpretation under the "no unsafe" header in `internal/compute/cpuref.go`. Its witnesses were parity only, with no allocation or bytes witness.
2. `f340b2767e` (2026-08-20) introduced `unsafe.Slice` in the same file, so the header was false from then on, but the decode callers were not revisited. The helper's comment said "reinterprets", which hid the copy.
3. The helper spread to Vulkan uploads (`d26aa23b01`, `2a4d3b7804`, `071ac79241`, `30245c5647`).
4. `3f9a0e3d7` (2026-10-12) found the Vulkan instance with `perf record` on hardware (41.8% of serve CPU) and fixed only those callers. Its guard pins `i8View`, not callers of `i8AsBytes`, and no sibling sweep was filed, so 12 cpu-ref sites survived until `19eab47217`.
5. No gate measures allocations on these paths. `make ci` has no allocation target; benchmarks report but never assert.

The same pattern appears in TICKET-01: a fast path is built and tested, but only tests enable it. **Both are verification gaps.** Correctness was witnessed; cost and production invocation were not.

## Process controls

| Control | Where | Status |
|---|---|---|
| Per-dtype allocation ceiling for every backend MatMul and upload | TICKET-07 | open |
| Detector for gating fields written only by tests | TICKET-07 | open |
| Copying helper moved to test-only code so production cannot call it | `internal/compute` | done (`300a373aed`) |
| A perf fix of an instance must sweep and list its siblings (migrated, or ticket #N) | private landing doctrine | done (doctrine text) |
| Profile receipts must be symbolized (the 10-13 profile has 38% opaque libc) | receipt writer | open |

## Tickets

| # | Ticket | Confidence | ESTIMATED cost |
|---|---|---|---|
| 01 | [V4.1 indexer resident keys never enabled](TICKET-01-v41-indexer-resident-keys-unwired.md) | HIGH | O(context) host copy per token; ~13.6 MB/token at 4K |
| 02 | [Routed-expert fault paths copy instead of borrowing](TICKET-02-v41-expert-fault-owned-copies.md) | HIGH (copies), cost depends on miss rate | up to ~6.6 MB per Q4_K miss |
| 03 | [V4.1 decode-step allocation ceiling](TICKET-03-v41-decode-step-alloc-ceiling.md) | HIGH | ~4-6K allocs and several MB per token |
| 04 | [Sampler sorts the full vocabulary](TICKET-04-inkernel-sampler-full-vocab-sort.md) | HIGH | 15-30 ms per token at V≈129K |
| 05 | [Per-dispatch env lookups and readback copies](TICKET-05-compute-dispatch-getenv-and-readback-copies.md) | MED | thousands of syscalls per token |
| 06 | [Latent per-call weight paths, incl. a layer-shared scratchpad bug](TICKET-06-compute-latent-per-call-weight-paths.md) | HIGH (bug), no live callers | 0 today; a trap when wired |
| 07 | [Allocation-ceiling and test-only-enable gates](TICKET-07-alloc-ceiling-and-test-only-enable-gates.md) | process | n/a |
| 08 | [cpu-ref per-call attention, Q8 and RoPE scratch](TICKET-08-cpuref-per-call-scratch-and-rope.md) | HIGH (allocates), MED (hot) | ~15 MB and ~1.4K allocs per token |

The copying helper is now test-only (`300a373aed`: `i8CopyForTest`), with an allocation ceiling over every packed dtype. The remaining cpu-ref per-call costs are TICKET-08.

All costs are estimates from config arithmetic, not measurements. Each ticket's acceptance requires a red-on-parent test, and a physical receipt for any speed claim.
