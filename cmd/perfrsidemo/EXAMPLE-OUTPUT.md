# Captured output — perfrsidemo

Captured from `go run ./cmd/perfrsidemo` and `-selfcheck`. The scorecard is a pure deterministic evaluation of the frozen 16-dimension evidence fixture, so this output is reproducible on any machine with Go and requires no API key, network access, or GPU.

## `go run ./cmd/perfrsidemo` (human-readable scorecard)

```text
performance RSI: fixture | target 100x | health C 72.2/100 | performance RSI debt 15
loop health: clean=false | measured 16/16 | BEHIND 15 | UNKNOWN 0
invocation outcomes: success=1 refusal=0 error=0
grade scope: grade describes performance-RSI loop health; it does not prove the explicit target multiplier was achieved
dominant bottleneck: evaluation_latency
cycle_time                BEHIND  current=10 target=5 ratio=0.5 source=fixture/cycles next=shorten cycle
improvement_yield         MET     current=100 target=100 ratio=1 source=fixture/yield next=raise yield
evaluation_latency        BEHIND  current=20 target=5 ratio=0.25 source=fixture/eval next=reduce evaluation latency
receipt_coverage          BEHIND  current=80 target=100 ratio=0.8 source=fixture/receipts next=capture receipts
quality_gate_coverage     BEHIND  current=80 target=100 ratio=0.8 source=fixture/quality next=add quality gates
experiment_throughput     BEHIND  current=8 target=10 ratio=0.8 source=fixture/throughput next=increase safe throughput
hypothesis_calibration    BEHIND  current=80 target=100 ratio=0.8 source=fixture/calibration next=calibrate ranking
discovery_freshness       BEHIND  current=2 target=1 ratio=0.5 source=fixture/discovery next=refresh discovery
adaptation_speed          BEHIND  current=2 target=1 ratio=0.5 source=fixture/adaptation next=adapt faster
reuse_ratio               BEHIND  current=80 target=100 ratio=0.8 source=fixture/reuse next=reuse mechanisms
learning_retention        BEHIND  current=80 target=100 ratio=0.8 source=fixture/learning next=retain learning
production_transfer       BEHIND  current=80 target=100 ratio=0.8 source=fixture/transfer next=transfer experiments
hardware_utilization      BEHIND  current=80 target=100 ratio=0.8 source=fixture/hardware next=use hardware
attribution_quality       BEHIND  current=80 target=100 ratio=0.8 source=fixture/attribution next=improve attribution
automation_coverage       BEHIND  current=80 target=100 ratio=0.8 source=fixture/automation next=automate loop
compounding_rate          BEHIND  current=80 target=100 ratio=0.8 source=fixture/compounding next=compound learning
```

## `go run ./cmd/perfrsidemo -selfcheck` (acceptance selfcheck)

```text
performance RSI: fixture | target 100x | health C 72.2/100 | performance RSI debt 15
loop health: clean=false | measured 16/16 | BEHIND 15 | UNKNOWN 0
invocation outcomes: success=1 refusal=0 error=0
grade scope: grade describes performance-RSI loop health; it does not prove the explicit target multiplier was achieved
dominant bottleneck: evaluation_latency
cycle_time                BEHIND  current=10 target=5 ratio=0.5 source=fixture/cycles next=shorten cycle
improvement_yield         MET     current=100 target=100 ratio=1 source=fixture/yield next=raise yield
evaluation_latency        BEHIND  current=20 target=5 ratio=0.25 source=fixture/eval next=reduce evaluation latency
receipt_coverage          BEHIND  current=80 target=100 ratio=0.8 source=fixture/receipts next=capture receipts
quality_gate_coverage     BEHIND  current=80 target=100 ratio=0.8 source=fixture/quality next=add quality gates
experiment_throughput     BEHIND  current=8 target=10 ratio=0.8 source=fixture/throughput next=increase safe throughput
hypothesis_calibration    BEHIND  current=80 target=100 ratio=0.8 source=fixture/calibration next=calibrate ranking
discovery_freshness       BEHIND  current=2 target=1 ratio=0.5 source=fixture/discovery next=refresh discovery
adaptation_speed          BEHIND  current=2 target=1 ratio=0.5 source=fixture/adaptation next=adapt faster
reuse_ratio               BEHIND  current=80 target=100 ratio=0.8 source=fixture/reuse next=reuse mechanisms
learning_retention        BEHIND  current=80 target=100 ratio=0.8 source=fixture/learning next=retain learning
production_transfer       BEHIND  current=80 target=100 ratio=0.8 source=fixture/transfer next=transfer experiments
hardware_utilization      BEHIND  current=80 target=100 ratio=0.8 source=fixture/hardware next=use hardware
attribution_quality       BEHIND  current=80 target=100 ratio=0.8 source=fixture/attribution next=improve attribution
automation_coverage       BEHIND  current=80 target=100 ratio=0.8 source=fixture/automation next=automate loop
compounding_rate          BEHIND  current=80 target=100 ratio=0.8 source=fixture/compounding next=compound learning
selfcheck: PASS (deterministic performance-rsi scorecard)
```

Observed warm-run wall time: under 1 second.
