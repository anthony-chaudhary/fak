# Micro-context S7: mixed-tenant fairness and economics

**Status:** offline scheduler witness captured on 2026-08-06.

S7 adds a tenant-aware weighted scheduler over the micro-context substrate. Each tenant declares weight, queue/concurrency bounds, spend, and rate envelopes. Selection uses normalized served/weight debt, then interactive/deadline tie-breaks. Cancellation is accounted before admission; spend and rate limits fail closed.

## Captured mixed workload

The deterministic fixture mixes a bursty interactive tenant (weight 3) with a bulk tenant (weight 1):

| Measure | Observed |
|---|---:|
| Submitted / scheduled / cancelled | 800 / 794 / 6 |
| Interactive / bulk scheduled | 594 / 200 |
| Interactive / bulk wait p95 | 0.750 / 0.759 ms |
| Maximum weighted lag | interactive 6; bulk 0 |
| Scheduler wall cost | ~1.04 ms |
| Go allocation cost | ~512 KiB |
| Accounted spend | interactive 594; bulk 400 micro-units |
| Duplicated output / failed work / estimated idle model | 0 / 0 / 0 |

```powershell
go test ./internal/microagent -run TestFairScheduler
go run ./cmd/microcontextdemo -verify-fairness experiments/microcontext/s7-local-mixed-tenant-fairness-2026-08-06.json
```

## Net-true economics boundary

The artifact prices scheduler CPU wall time, allocation, declared spend, failed work, duplicated output, and estimated idle model time. It intentionally reports `cost-only; no economic gain claimed without a tuned serving baseline`. Therefore there is no efficiency headline to pass through `fak claim-check`; fabricating a gain against an absent serving baseline would be a strawman. The witness establishes enforcement and measured overhead, not a positive ROI.

The fixture is offline and deterministic. It proves tenant isolation, weighted service, cancellation, deadline/interactive priority, and spend/rate envelopes in the scheduler. It does not prove provider-side fairness, live multi-user TTFT, model throughput, or a dollar saving.
