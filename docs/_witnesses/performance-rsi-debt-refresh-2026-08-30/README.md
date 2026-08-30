# Performance-RSI debt refresh — August 30, 2026

**Verdict:** current performance-RSI debt is **6 dimensions**, down from **9** on the August 28 baseline: **3 dimensions / 33.3% retired**. This meets issue #10358's at-least-25% target.

## What changed

- A single unattended PowerShell wrapper started a bounded **receipt-refresh experiment**, found and parsed the three retained receipts, hashed them, checked their schema family, persisted `cycle-result.json`, and read it back before ending.
- The wrapper recorded all six cycle stages contemporaneously in `cycle.json`; it did not reuse the older issue or lane timestamps.
- Total elapsed time was **5.629861 seconds**; execution-to-evaluation was **1.136850 seconds**.
- One second is conservatively charged as operator-active time for launching the wrapper. Consequently, automation coverage improved but remains **BEHIND**; this witness does not claim an automation win.

## Debt result

| Dimension | Baseline | Current | Status |
|---|---:|---:|---|
| `cycle_time` | 2 h | 0.001563850 h | MET |
| `evaluation_latency` | 30 min | 0.018947 min | MET |
| `experiment_throughput` | 12/day | 15346.738/day | MET |
| `automation_coverage` | 75% | 82.238% | BEHIND |

`cycle_time` and `experiment_throughput` are coupled views of the same cycle duration; they are two scorecard dimensions, not two independent operational gains.

## Evidence boundaries

- Reused unchanged: `issue-9781` improvement, `issue-9782` provenance, and `issue-9783` learning receipts. Their SHA-256 values are in `cycle-result.json`.
- Those retained receipts remain stale relative to August 30. Their owned dimensions were not claimed as improved.
- `hardware_utilization` remains **UNKNOWN**. No hardware receipt was introduced and no model-speed claim is made.
- `fak-native/qwen3.8` is the required cycle provenance label. This receipt measures the scorecard refresh path; it is not a Qwen inference benchmark.
- `production_transfer` retains the scorecard's current higher-is-better schema semantics; this refresh does not reinterpret it.

## Reproduce

```powershell
fak.exe performance-rsi-scorecard compose `
  --snapshot issue-10358-performance-rsi-debt-refresh-2026-08-30 `
  docs/_witnesses/performance-rsi-debt-refresh-2026-08-30/cycle-evidence.json `
  docs/_witnesses/issue-9781-performance-rsi-improvement.json `
  docs/_witnesses/issue-9782-performance-rsi-provenance.json `
  docs/_witnesses/issue-9783-performance-rsi-learning.json

fak.exe performance-rsi-scorecard `
  --input docs/_witnesses/performance-rsi-debt-refresh-2026-08-30/input.json `
  --prior docs/_witnesses/performance-rsi-debt-refresh-2026-08-30/baseline.json `
  --json
```

Expected current readout: health **B / 80.7**, debt **6**, measured **15/16**, hardware UNKNOWN.

## Files

- `baseline.json` / `baseline.md`: frozen August 28 scorecard baseline
- `cycle.json`: raw cycle section written by the wrapper
- `cycle-evidence.json`: composable evidence envelope
- `cycle-result.json`: receipt hashes and bounded-cycle result
- `cycle-trace.json`: stage definitions and operator-time basis
- `cycle-metrics.json`: independent timestamp arithmetic
- `input.json`: composed current evidence
- `scorecard.json` / `scorecard.md`: current score and prior comparison
- `manifest.json`: baseline/current summary and artifact digests

Issue: #10358.


