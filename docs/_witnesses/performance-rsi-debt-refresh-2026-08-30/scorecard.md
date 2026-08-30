# Performance RSI — issue-10358-performance-rsi-debt-refresh-2026-08-30

- Explicit target: **100x** (unsaturated)
- Loop-health grade: **B** (80.7/100; clean: **false**)
- Performance RSI debt: **6** (5 BEHIND, 1 UNKNOWN; 15/16 measured)
- invocation outcomes: success=1 refusal=0 error=0
- Grade scope: grade describes performance-RSI loop health; it does not prove the explicit target multiplier was achieved
- Dominant bottleneck: `discovery_freshness`

| Dimension | Status | Current | Target | Normalized ratio | Source | Next action |
|---|---:|---:|---:|---:|---|---|
| cycle_time | MET | 0.00156385 hours | 0.02 hours | 12.7889 | cycle:fak-performance-rsi-cycle/1 | shorten the end-to-end learning cycle |
| improvement_yield | BEHIND | 20 percent | 100 percent | 0.2 | improvement:fak-performance-rsi-improvement/1 | raise yield |
| evaluation_latency | MET | 0.000315792 hours | 0.005 hours | 15.8332 | cycle:fak-performance-rsi-cycle/1 | reduce evaluation latency |
| receipt_coverage | MET | 100 percent | 100 percent | 1 | improvement:fak-performance-rsi-improvement/1 | capture receipts |
| quality_gate_coverage | MET | 100 percent | 100 percent | 1 | improvement:fak-performance-rsi-improvement/1 | add quality gates |
| experiment_throughput | MET | 15346.7 experiments/day | 1200 experiments/day | 12.7889 | cycle:fak-performance-rsi-cycle/1 | increase completed cycles |
| hypothesis_calibration | MET | 100 percent | 100 percent | 1 | learning:fak-performance-rsi-learning/1 | calibrate ranking |
| discovery_freshness | BEHIND | 16.0517 hours | 1 hours | 0.0622988 | provenance:fak-performance-rsi-provenance/1 | refresh discovery |
| adaptation_speed | MET | 0.858056 hours | 1 hours | 1.16543 | provenance:fak-performance-rsi-provenance/1 | adapt faster |
| reuse_ratio | MET | 100 percent | 100 percent | 1 | provenance:fak-performance-rsi-provenance/1 | reuse mechanisms |
| learning_retention | MET | 100 percent | 100 percent | 1 | learning:fak-performance-rsi-learning/1 | retain learning |
| production_transfer | BEHIND | 82.5922 hours | 100 hours | 0.825922 | provenance:fak-performance-rsi-provenance/1 | transfer experiments |
| hardware_utilization | UNKNOWN | UNKNOWN percent | 100 percent | UNKNOWN | fixture/hardware | use hardware |
| attribution_quality | MET | 100 percent | 100 percent | 1 | improvement:fak-performance-rsi-improvement/1 | improve attribution |
| automation_coverage | BEHIND | 82.2376 percent | 99 percent | 0.830683 | cycle:fak-performance-rsi-cycle/1 | reduce operator-active time |
| compounding_rate | BEHIND | 98.7276 percent | 100 percent | 0.987276 | learning:fak-performance-rsi-learning/1 | compound learning |

Compared with `issue-9768-live-repository-dogfood-2026-08-28`.
