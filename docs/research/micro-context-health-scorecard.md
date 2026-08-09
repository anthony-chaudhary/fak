---
title: "Micro-context health scorecard"
description: "A deterministic grade over the micro-context quality ledger: scale floor, reconciled terminal outcomes, verification rate, and useful-work throughput."
---

# Micro-context health scorecard

`microcontextdemo -health-scorecard` grades the existing quality ledger rather than self-report. The score is deterministic: scale floor, reconciled terminal outcomes, independent verification, and useful-work throughput. Drift remains explicitly `baseline-only` until a second comparable ledger exists.

The controlled native-CUDA 1,000-context ledger grades **A / 100**: 1,000 successes, zero errors/refusals, verification rate 1.0, and 0.9564 independently verified nonempty results/s. Artifact: `experiments/microcontext/s5-gcp-1000-cuda-health-scorecard-2026-08-07.json`.

```powershell
go run ./cmd/microcontextdemo -health-input experiments/microcontext/s5-gcp-1000-cuda-outcomes-2026-08-07.json -health-scorecard -
go run ./cmd/microcontextdemo -verify-health-scorecard experiments/microcontext/s5-gcp-1000-cuda-health-scorecard-2026-08-07.json
```
