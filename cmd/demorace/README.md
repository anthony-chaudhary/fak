# demorace - live reuse race

`demorace` is the live model-backed reuse race: fak versus a tuned warm-cache baseline, plus
a reuse curve. It is the browser sibling of the model-free accounting demos.

## Prerequisites

Go 1.26+ from the repository root. The live race needs a supported exported model on disk;
without one, the server still starts and reports which ladder rungs are missing. Server
startup completes in a few seconds; live race runtime depends on the model and hardware.
The run is deterministic for a fixed model, workload, and hardware budget, but wall-clock
numbers are machine-specific.

## Run it

```bash
go run ./cmd/demorace
go run ./cmd/demorace -jobs 8
FAK_DEMO_BASE_PATH=/demorace go run ./cmd/demorace
```

Open the printed loopback URL, choose a present model rung, run the live race, then build
the curve. `-base-path` or `FAK_DEMO_BASE_PATH` mounts the browser demo behind a preserved
HTTPS reverse-proxy path.

## What you see

```text
demorace fak-demorace-v1 on http://127.0.0.1:8147/ (GOMAXPROCS=8)
ladder present: smollm2-135m
open the URL -> 'Run live race' (HEADLINE fak vs tuned warm-cache, both LIVE) then 'Build curve'
PASS - the server exposes api/ladder, api/race, and api/curve.
```

The browser streams progress over SSE. The API surfaces report which model rungs are
present, so missing weights are visible instead of silently faking a race.

## Scope

This demo is for live serving behavior on the current machine. It does not claim a portable
wall-clock benchmark, does not use missing model rungs as evidence, and does not compare
against cold-cache baselines as the headline. For benchmark framing, see
[`../../docs/benchmarking/README.md`](../../docs/benchmarking/README.md) and
[`../../CLAIMS.md`](../../CLAIMS.md).
