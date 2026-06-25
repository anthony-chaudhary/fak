# ctxdemo - context-reuse accounting

`ctxdemo` shows how much prompt work a multi-agent workload re-reads under cold,
warm-cache, and fak-fused strategies. The browser can run a live model race when model
weights are present; the headless modes are model-free and print exact token accounting.

## Prerequisites

Go 1.26+ from the repository root. `-print` and `-bars` require no model, GPU, API key,
browser, or network and complete in a few seconds. The live browser race is optional and
needs a supported exported model on disk.

## Quick Start

```bash
go run ./cmd/ctxdemo -print
go run ./cmd/ctxdemo -bars
go run ./cmd/ctxdemo
FAK_DEMO_BASE_PATH=/ctxdemo go run ./cmd/ctxdemo
```

Open the printed loopback URL, pick a scenario, then run the live race if a model is
available. `-base-path` or `FAK_DEMO_BASE_PATH` mounts the browser demo behind a preserved
HTTPS reverse-proxy path.

The headless outputs are deterministic: the same scenario catalog and flags produce the
same token counts every run. Only the optional live race has hardware-dependent wall time.

## What you see

```text
fak - ctxdemo - exact, timing-free prefill-token work per scenario module
scenario        C   T   P     warmKV      fak      fak-win
support-bot     5   8   512   39,591      35,495   PASS
deep-research   5   50  512   fak makes the model re-read less
```

The `-bars` view is the 30-second terminal twin of the browser's reuse story. The browser
and the headless table use the same scenario catalog.

## Scope

This demo's headless modes prove token accounting, not live wall-clock speed. The optional
live race depends on local model availability and hardware. It does not claim that fak beats
every warm-cache serving stack by the cold-cache headline ratio. For the longer explanation,
see [`../../docs/explainers/fleet-benchmarks.md`](../../docs/explainers/fleet-benchmarks.md)
and [`../../CLAIMS.md`](../../CLAIMS.md).
