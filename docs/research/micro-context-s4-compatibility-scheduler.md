# Micro-context S4b: compatibility-class scheduler

**Status:** deterministic controlled-kernel planner witness, 2026-08-06.

The scheduler classifies model turns by model, sampling configuration, advertised tools, prefix identity, prefill/decode phase, and sequence-length bucket. Only exact keys coalesce. Missing classification fails open to the existing singleton path. Per-class queues are bounded, cancelled work is removed, priority ages by queue time, deadlines break ties, and a padding cap shrinks batches instead of hiding wasted prefill.

```powershell
go run ./cmd/microcontextdemo -compatibility-witness experiments/microcontext/s4-local-compatibility-2026-08-06.json
go run ./cmd/microcontextdemo -verify-compatibility experiments/microcontext/s4-local-compatibility-2026-08-06.json
```

The mixed workload submitted 98 turns across three incompatible classes plus one unclassified turn and one cancellation. It scheduled 97 turns in 13 batches, used singleton fallback once, rejected zero, held padding tax to 7.71% under the 10% cap, observed 93.27% batch fill/nominal slot utilization, and recorded a 20 ms maximum synthetic queue age. Tests additionally pin class isolation, queue bounds, cancellation, padding splits, and aging that lets old low-priority work outrank new work.

This is scheduling telemetry, not model telemetry. Batch fill, nominal utilization, and synthetic queue age do not establish GPU occupancy, tokens/sec, TTFT, or a cache gain. S5 must connect the same keys/counters to the controlled endpoint.
