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
go run ./cmd/tokendemo -parallel
go run ./cmd/tokendemo -parallel-json
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

Use `-parallel` for a larger hot-cache proof. It prewarms one read per hot file, then
runs hundreds of parallel repeated reads. The output reports raw engine calls, fak
warmup calls, fak hot-phase engine calls, vDSO hits, timing percentiles, and
per-resource cache evidence.

## Scope

This demo separates two meters that are easy to conflate: model-context tokens and tool
executions. It does not claim wall-clock speed, model quality, or that refusing a call is
the same value axis as serving a cached read. For the feature status, see
[`../../docs/supported/features.md`](../../docs/supported/features.md) and
[`../../CLAIMS.md`](../../CLAIMS.md).
