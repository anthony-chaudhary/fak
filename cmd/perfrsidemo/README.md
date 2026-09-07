# Performance RSI demo (`perfrsidemo`)

Prerequisite: Go 1.26+ (the repository toolchain setting can fetch it automatically). No API key, network access, live model weights, or GPU is required. On a warm Go build cache the selfcheck completes in under 3 seconds; the first run can take longer while Go fetches the declared toolchain and compiles the package. The fixture is deterministic and reproducible across platforms.

Run the performance RSI scorecard over a deterministic evidence fixture:

```bash
go run ./cmd/perfrsidemo
```

Run the invariant acceptance self-check:

```bash
go run ./cmd/perfrsidemo -selfcheck
```

## Sample run output

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

See [EXAMPLE-OUTPUT.md](EXAMPLE-OUTPUT.md) for captured output.

## What this proves

- The production `internal/perfrsiscore` engine evaluates evidence across all 16 canonical improvement dimensions without an external dependency.
- The dominant bottleneck is calculated deterministically via normalized ratio minimization (`evaluation_latency` at ratio 0.25).
- Unresolved debt is computed honestly as `BEHIND + UNKNOWN`.
- Loop-health scoring maps ratio-credited progress to a letter grade (`C 72.2/100`) while bounding per-dimension credit so overperformance cannot erase debt in another dimension.
- Passing `-selfcheck` asserts schema validity (`fak-performance-rsi-scorecard/1`), 16 canonical dimensions, and the dominant bottleneck invariant before exiting 0.

## What this does not claim

This deterministic fixture demonstrates the scorecard accounting and bottleneck derivation logic. It does not measure live model inference latency, execute hardware benchmarks, perform network calls, or assert that a real checkout's optimization loop has reached its 100x target multiplier. Use `fak performance-rsi-scorecard` with real captured evidence for operational evaluation.
