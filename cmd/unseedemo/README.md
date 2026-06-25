# unseedemo - Un-See It

`unseedemo` is the browser version of the KV eviction witness. It drives the real
`ctxmmu` gate, `kvmmu` bridge, and `model.KVCache.Evict` against a synthetic Llama-shaped
model, then animates the poisoned span being removed from the cache.

## Prerequisites

Go 1.26+ from the repository root. No model weights, GPU, API key, or network is required.
The witness is deterministic and usually completes in a few seconds for `-selfcheck`,
`-print`, or `-json`; the browser server starts in seconds and waits for a local browser.

## Run it

```bash
go run ./cmd/unseedemo
go run ./cmd/unseedemo -selfcheck
go run ./cmd/unseedemo -print
go run ./cmd/unseedemo -json
FAK_DEMO_BASE_PATH=/unsee go run ./cmd/unseedemo
```

Open the printed loopback URL and press Play. `-base-path` or `FAK_DEMO_BASE_PATH` mounts
the browser demo behind a preserved HTTPS reverse-proxy path.

## What you see

```text
== unseedemo -selfcheck: drive the real kvmmu bridge / ctxmmu gate / KVCache.Evict (browserless) ==
gate_verdict == QUARANTINE                       QUARANTINE             PASS
evict-vs-never == 0 (bit-identical)              0.000e+00              PASS
too-late-vs-never > 0 (boundary holds)           3.487e-02              PASS
OK - 7/7 invariants reproduced.
```

The browser replays the same event log exposed by `-json`; it is an animation over live
computed witness values, not a prerecorded trace.

## Scope

This demo proves the wiring on a synthetic model witness. It does not claim HuggingFace
oracle equivalence for the eviction proof, live `fak serve` integration of `kvmmu`, or
detection robustness. For the exact fences, see the event log and
[`../../CLAIMS.md`](../../CLAIMS.md).
