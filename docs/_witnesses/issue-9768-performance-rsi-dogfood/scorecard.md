---
title: "Performance RSI dogfood scorecard for issue 9768"
description: "Captured issue 9768 repository-dogfood scorecard, including the explicit improvement target, measured bottleneck, evidence, and follow-up actions."
---

# Performance RSI — issue-9768-live-repository-dogfood-2026-08-28

- Explicit target: **100x** (unsaturated)
- Dominant bottleneck: `cycle_time`
- UNKNOWN debt: **1**

| Dimension | Status | Current | Target | Normalized ratio | Source | Next action |
|---|---:|---:|---:|---:|---|---|
| cycle_time | BEHIND | 2 hours | 0.02 hours | 0.01 | cycle:fak-performance-rsi-cycle/1 | shorten the end-to-end learning cycle |
| improvement_yield | BEHIND | 20 percent | 100 percent | 0.2 | improvement:fak-performance-rsi-improvement/1 | raise yield |
| evaluation_latency | BEHIND | 30 minutes | 5 minutes | 0.166667 | cycle:fak-performance-rsi-cycle/1 | reduce evaluation latency |
| receipt_coverage | MET | 100 percent | 100 percent | 1 | improvement:fak-performance-rsi-improvement/1 | capture receipts |
| quality_gate_coverage | MET | 100 percent | 100 percent | 1 | improvement:fak-performance-rsi-improvement/1 | add quality gates |
| experiment_throughput | BEHIND | 12 experiments/day | 1200 experiments/day | 0.01 | cycle:fak-performance-rsi-cycle/1 | increase completed cycles |
| hypothesis_calibration | MET | 100 percent | 100 percent | 1 | fixture/calibration | calibrate ranking |
| discovery_freshness | BEHIND | 16.0517 hours | 1 hours | 0.0622988 | provenance:fak-performance-rsi-provenance/1 | refresh discovery |
| adaptation_speed | MET | 0.858056 hours | 1 hours | 1.16543 | provenance:fak-performance-rsi-provenance/1 | adapt faster |
| reuse_ratio | MET | 100 percent | 100 percent | 1 | provenance:fak-performance-rsi-provenance/1 | reuse mechanisms |
| learning_retention | MET | 100 percent | 100 percent | 1 | fixture/learning | retain learning |
| production_transfer | BEHIND | 82.5922 hours | 100 hours | 0.825922 | provenance:fak-performance-rsi-provenance/1 | transfer experiments |
| hardware_utilization | UNKNOWN | UNKNOWN percent | 100 percent | UNKNOWN | fixture/hardware | use hardware |
| attribution_quality | MET | 100 percent | 100 percent | 1 | improvement:fak-performance-rsi-improvement/1 | improve attribution |
| automation_coverage | BEHIND | 75 percent | 99 percent | 0.757576 | cycle:fak-performance-rsi-cycle/1 | reduce operator-active time |
| compounding_rate | BEHIND | 98.7276 percent | 100 percent | 0.987276 | fixture/compounding | compound learning |
