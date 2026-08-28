# Issue #9768 performance-RSI dogfood witness

This directory captures a real repository run of `fak performance-rsi-scorecard` on August 28, 2026. It composes the committed, privacy-scrubbed receipts from issues #9780–#9783; it is not a synthetic scorecard fixture.

## Reproduce

```bash
# input.json was composed from the four committed receipt sections and their owning
# dimension contracts; see the dogfood note for the composition defect this exposed.
go run ./cmd/fak performance-rsi-scorecard \
  --input docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json \
  --json > /tmp/issue-9768-scorecard.json
cmp /tmp/issue-9768-scorecard.json \
  docs/_witnesses/issue-9768-performance-rsi-dogfood/scorecard.json

go run ./cmd/fak performance-rsi-scorecard \
  --input docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json \
  --markdown > /tmp/issue-9768-scorecard.md
cmp /tmp/issue-9768-scorecard.md \
  docs/_witnesses/issue-9768-performance-rsi-dogfood/scorecard.md
```

## Result

- 15 of 16 dimensions are measured: 7 `MET`, 8 `BEHIND`, and 1 `UNKNOWN`.
- `hardware_utilization` is the only `UNKNOWN`; issue #9784 remains open pending a sanitized sanctioned-hardware receipt.
- `cycle_time` is the dominant bottleneck at a normalized ratio of `0.01` (2 hours observed versus a 0.02-hour target).
- `experiment_throughput` is also at `0.01` (12 experiments/day observed versus 1,200 targeted).
- The scorecard target is explicitly 100x and unsaturated. This run measures the loop; it does **not** prove a 100x result or model speed.

## Source provenance

| Receipt | Commit | Role |
|---|---|---|
| `issue-9780-performance-rsi-cycle.json` | `fa3ae78b4ee74c75271e5514426adc40b1996a66` | cycle time, evaluation latency, throughput, operator-active time |
| `issue-9781-performance-rsi-improvement.json` | `ff5e3b5bf21d6a7412305acb4137b35d3e4c28a9` | useful yield, receipt and quality-gate coverage, attribution |
| `issue-9782-performance-rsi-provenance.json` | `86a268bc0918aee4c6cafeb7bfc653126042792a` | discovery, adaptation, reuse, native production transfer |
| `issue-9783-performance-rsi-learning.json` | `44d7ecdace862ba43d5be8d1e1448d99d64fcfb3` | calibration, retained learning, compounding |

Scorecard implementation at capture: `cmd/fak@r3523+gcdfa9e36b` and `internal/perfrsiscore@r8+gcdfa9e36b`.

## Defects surfaced

- #9823: no canonical deterministic receipt-composition surface; naive merging fails on incompatible dimension contracts.
- #9824: learning-derived rows retain stale `fixture/*` source labels despite real receipt provenance.
