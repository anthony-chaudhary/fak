# tokendemo - tool-call token ledger

`tokendemo` is a browserless demo with two meters: model-context tokens kept out by a
prefiltered bad call, and tool round-trips collapsed by a cached re-read. It replays the
same fixtures through the real kernel path used by the other turn demos.

## Prerequisites

Go 1.26+ from the repository root. No model, GPU, API key, browser, or network is
required. The run is deterministic and usually completes in a few seconds.

## Run it

```bash
go run ./cmd/tokendemo -print
go run ./cmd/tokendemo -print -suite reread-same-file
go run ./cmd/tokendemo -timing
go run ./cmd/tokendemo -timing-json
go run ./cmd/tokendemo -served
go run ./cmd/tokendemo -served-json
go run ./cmd/tokendemo -parallel
go run ./cmd/tokendemo -parallel-json
go run ./cmd/tokendemo -parallel-cold
go run ./cmd/tokendemo -parallel-cold-json
go run ./cmd/tokendemo -json
go run ./cmd/tokendemo -selfcheck
```

Use `-suite prefilter-bad-calls`, `-suite reread-same-file`, or `-suite clean-control`
to focus one fixture. `-json` emits the exact per-call ledger for automation.

## What you see

```text
== tokendemo -selfcheck: replay each suite through the kernel (browserless) ==
prefilter-bad-calls  model-context tokens kept out: 1452  PASS
reread-same-file     round-trips collapsed: 3             PASS
clean-control        overclaim guard: 0 context delta     PASS
OK - all suites reproduced the documented ledger invariants.
```

The `-print` view renders the without-kernel and with-kernel columns. The JSON output
contains every call's class, token accounting, and tool-run count.

Use `-timing` for the concrete cache proof: the raw loop calls the local tool engine
for every read, while the fak loop runs the same trace through `kernel.Syscall` with
the vDSO enabled. Repeated reads show `fak_source=vdso_tier2`,
`fak_engine_call_delta=0`, and a per-call latency next to the raw engine time.

Use `-served` for the served-boundary proof. It starts the gateway handler, repeats a
raw `os.ReadFile` baseline, then sends the same `Read` through HTTP
`POST /v1/fak/syscall` and MCP `fak_syscall` against the real `fakread` filesystem
engine. The first served call reaches `fakread`; repeats show `served_by=vdso`,
`tier=2`, and `/metrics` rows for engine calls, vDSO hits, HTTP, and MCP requests.

Use `-parallel` for a larger hot-cache proof. It prewarms one read per hot file, then
runs hundreds of parallel repeated reads. The output reports raw engine calls, fak
warmup calls, fak hot-phase engine calls, vDSO hits, timing percentiles, and
per-resource cache evidence.

Use `-parallel-cold` for the cold-miss companion to `-parallel`. Where `-parallel`
proves the *warmed* case, this releases N workers at ONE barrier against a NEVER-SEEN
key and counts how many engine calls happen before the vDSO tier-2 fill exists. Each
trial starts from an empty vDSO world and reports `engine_calls_per_key`,
`vdso_hits_per_key`, `cold_fill_races` (`engine_calls_per_key - 1`, clamped at zero),
and raw-vs-fak wall time and per-call p50/p95. It is a measurement first: with no
singleflight on the cold-miss path yet, `MEASURED_RACE` is the expected, honest verdict
(`SINGLEFLIGHT_CONFIRMED` only if every cold key collapses to exactly one engine call).
It is kept as the regression guard for when singleflight is built. This proves cold-fill
coordination only — not the warmed hot-cache hit-rate (`-parallel`) and not any
provider or model-context token saving.

## Scope

This demo separates two meters that are easy to conflate: model-context tokens and tool
executions. It does not claim wall-clock speed, model quality, or that refusing a call is
the same value axis as serving a cached read. For the feature status, see
[`../../docs/supported/features.md`](../../docs/supported/features.md) and
[`../../CLAIMS.md`](../../CLAIMS.md).
