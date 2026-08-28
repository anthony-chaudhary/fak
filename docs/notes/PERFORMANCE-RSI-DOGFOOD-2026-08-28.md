# Performance-RSI live repository dogfood — 2026-08-28

**Verdict:** the scorecard now measures 15 of the parent objective's 16 dimensions from committed repository receipts, but it does not prove the 100x objective. Hardware utilization remains unknown, and the measured loop is farthest behind on end-to-end cycle time.

## Scope

- **Centrality:** Core observability for #9752.
- **For:** maintainers choosing the next performance-system intervention.
- **Problem:** four independently landed receipt families had not been exercised together against real repository evidence.
- **Today:** each receipt scored alone, so individual fixtures could mask composition and attribution defects.
- **Better because:** one deterministic scorecard output now exposes cross-receipt compatibility, remaining unknown debt, and the dominant measured bottleneck.
- **Witness:** `docs/_witnesses/issue-9768-performance-rsi-dogfood/`.

## Result

The composed run has 7 `MET`, 8 `BEHIND`, and 1 `UNKNOWN` dimensions. `hardware_utilization` is the sole unknown and remains tracked by #9784. The dominant bottleneck is `cycle_time`: 2 hours against a 0.02-hour target, normalized ratio `0.01`. `experiment_throughput` is tied at `0.01`, with 12 experiments/day against 1,200/day; stable dimension order makes `cycle_time` the reported dominant bottleneck.

`MET` means the observed value meets the dimension's configured target; it is not evidence that the complete system is 100x better. Likewise, the 100x target printed by the scorecard is an explicit unsaturated goal, not a measured multiplier. The artifacts are telemetry and attribution evidence only, not proof of model speed.

## Receipt composition

The run uses the real, privacy-scrubbed receipt sections committed by #9780–#9783. The input chooses the dimension contract owned by each derivation path, then attaches all four receipt sections. This manual selection was necessary because the receipt files disagree on some units: for example, #9780 declares `improvement_yield` as `ratio` while the #9781 derivation requires `percent`; #9782 owns `production_transfer` as elapsed `hours`, while earlier templates declare it as `percent`.

A naive merge using the #9780 dimension array fails with:

```text
fak performance-rsi-scorecard: improvement derivation for improvement_yield: unsupported unit "ratio"
```

That is a real dogfood defect, not an input to normalize silently. #9823 tracks a versioned composition surface with conflict diagnostics.

## Attribution defect

The learning-derived values are real and carry `performance_rsi_learning_receipt` plus `fak-native`, but their rendered source strings remain `fixture/calibration`, `fixture/learning`, and `fixture/compounding`. #9824 tracks replacement of stale template sources with the exact supplying receipt.

## Provenance and limits

- Cycle receipt commit: `fa3ae78b4ee74c75271e5514426adc40b1996a66`.
- Improvement receipt commit: `ff5e3b5bf21d6a7412305acb4137b35d3e4c28a9`.
- Provenance receipt commit: `86a268bc0918aee4c6cafeb7bfc653126042792a`.
- Learning receipt commit: `44d7ecdace862ba43d5be8d1e1448d99d64fcfb3`.
- Scorecard implementation: `cmd/fak@r3523+gcdfa9e36b`, `internal/perfrsiscore@r8+gcdfa9e36b`.

No hardware value was inferred from duration, queue timing, or local-machine state. #9784 remains open until a sanctioned compute route yields a directly measured, sanitized receipt. #9752 also remains open because this dogfood run exposes the improvement gap; it does not close it.
