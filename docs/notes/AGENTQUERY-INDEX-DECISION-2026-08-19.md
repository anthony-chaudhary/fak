# Agent relation index decision � 2026-08-19

## Verdict

Keep the authoritative session journal and bounded in-process fold; do not add an index yet. At 100,000 lifecycle events (50,000 completed sessions), the median direct fold was **403 ms on Windows** and **429 ms on Linux/WSL**, below the declared 1,000 ms interaction threshold. It also remained faster than the `ps`-JSON-plus-`jq`-equivalent transformation on both systems and requires no second process or external dependency.

Reopen this decision when either a representative journal reaches **200,000 events** or the direct path repeatedly exceeds **1,000 ms median**. An index, if justified then, remains disposable and reproducible; the JSONL journal stays authoritative.

## Reproduce

From a clean checkout at the witnessed commit:

```text
fak agents --benchmark --benchmark-sizes 1000,10000,100000 --benchmark-repetitions 5 --json
```

Machine witnesses:

- [`agentquery-benchmark-windows-2026-08-19.json`](../_witnesses/agentquery-benchmark-windows-2026-08-19.json)
- [`agentquery-benchmark-linux-2026-08-19.json`](../_witnesses/agentquery-benchmark-linux-2026-08-19.json)

Each size is a deterministic, content-free journal with equal open/close rows. Every path must produce the same SHA-256 result digest or the command fails.

## Results

| OS | Events | Direct fold | `ps` JSON + `jq` equivalent | Direct JSONL scan |
|---|---:|---:|---:|---:|
| Windows amd64 | 1,000 | 2.64 ms / 2.98 MiB | 3.69 ms / 3.89 MiB | 2.11 ms / 2.52 MiB |
| Windows amd64 | 10,000 | 30.74 ms / 33.57 MiB | 39.44 ms / 44.75 MiB | 28.61 ms / 29.61 MiB |
| Windows amd64 | 100,000 | 402.94 ms / 355.74 MiB | 671.07 ms / 498.80 MiB | 544.04 ms / 316.65 MiB |
| Linux amd64 | 1,000 | 2.37 ms / 2.98 MiB | 2.89 ms / 3.88 MiB | 2.01 ms / 2.52 MiB |
| Linux amd64 | 10,000 | 31.49 ms / 33.57 MiB | 42.37 ms / 44.75 MiB | 34.49 ms / 29.61 MiB |
| Linux amd64 | 100,000 | 428.60 ms / 355.73 MiB | 524.88 ms / 498.78 MiB | 403.09 ms / 316.65 MiB |

Cells are median wall time / median Go allocated bytes over five repetitions. Allocated bytes are not process RSS or peak working set.

## Comparison boundaries

- **Direct fold:** zero process starts, one operator step, no external dependency. Measurement includes JSONL decoding, authoritative lifecycle folding, and aggregation.
- **`fak ps --json | jq` alternative:** two process starts, three operator steps, and an external `jq` dependency. The timed region intentionally measures the equivalent serialization/decode/fold transformation but excludes OS process startup; therefore it is a conservative lower bound for the real shell pipeline, not a claim about full process latency. `fak ps` itself reads live gateway state rather than the historical journal, so an actual pipe cannot answer the same seven-day query without first changing its semantics.
- **Direct JSONL scan:** zero process starts, two operator steps, no external dependency. It decodes and folds rows directly but bypasses the relation's source-health and schema contract.

The benchmark answers the storage decision without pretending unlike interfaces are identical. It does not report peak RSS because the portable in-process witness cannot isolate three path-specific process peaks; it reports Go `TotalAlloc` deltas and labels that scope in machine output.
