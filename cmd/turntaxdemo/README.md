# turntaxdemo - turn-tax race

`turntaxdemo` replays tool-call traces through the real turn-tax kernel and compares a
tuned 2026-style loop against fak's one-shot path. The browser animates the turn ladder;
the terminal modes run the same replay without a browser.

## Prerequisites

Go 1.26+ from the repository root. No model, GPU, API key, or network is required. The
run is deterministic and usually completes in a few seconds for `-print` or `-selfcheck`;
the browser server starts in seconds and waits for a local browser.

## Run it

```bash
go run ./cmd/turntaxdemo
go run ./cmd/turntaxdemo -print
go run ./cmd/turntaxdemo -selfcheck
FAK_DEMO_BASE_PATH=/turntax go run ./cmd/turntaxdemo
```

Open the printed loopback URL, pick a suite, then replay through the kernel. `-base-path`
or `FAK_DEMO_BASE_PATH` mounts the browser demo behind a preserved HTTPS reverse-proxy
path.

## What you see

```text
== turntaxdemo -selfcheck: replay each suite through the kernel (browserless) ==
turntax-airline      tuned forced round-trips: 5  fak forced round-trips: 0  PASS
turntax-happy        tuned forced round-trips: 0  fak forced round-trips: 0  PASS
OK - all suites reproduced the documented turn-tax + safety-floor invariants.
```

The `-print` view renders the tuned-SOTA lane and the fak lane side by side. The browser
uses the same `api/run` replay surface.

## Scope

This demo shows forced extra model turns on frozen traces. It does not claim wall-clock
serving speed, model quality, or a universal benchmark across all agent tasks. For the
published turn-tax witness, see [`../../docs/benchmarks/TURN-TAX-RESULTS.md`](../../docs/benchmarks/TURN-TAX-RESULTS.md)
and the honesty ledger in [`../../CLAIMS.md`](../../CLAIMS.md).
