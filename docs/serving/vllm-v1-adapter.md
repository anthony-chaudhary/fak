---
title: "vLLM V1 adapter: fak governs a ridden vLLM worker over public surfaces"
description: "Issue #40 driver doc. The registered vllm EngineDriver rides a vLLM V1 worker's OpenAI HTTP, KV-cache-events, and Prometheus surfaces; fak adds routing, trust, and exact-span KV governance on top, degrading honestly to whole-prefix reset."
---

# vLLM V1 adapter — fak governs, vLLM serves

Issue #40 is delivered by the **fak-governs / vLLM-serves** posture, the same
shape as the Dynamo sibling ([`dynamo-interop.md`](dynamo-interop.md)).

`internal/engine.VLLMEngine` registers the `vllm`
[`EngineDriver`](../proofs/engine-seam.md) under `abi.RegisterEngine`
(`internal/engine/vllm.go`, `VLLMEngineID = "vllm"`). It implements the
admit/step/stream [`LifecycleEngine`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/engine_lifecycle.go)
seam (issue #46), so a fak gateway can admit a request, stream vLLM's SSE deltas
token by token, cancel mid-decode, and reclaim the slot. vLLM keeps ownership of
continuous batching, PagedAttention, FP8 KV, tensor parallelism, and speculative
decode; fak adds only what it owns on top: routing, the capability floor, and KV
governance.

## The three public surfaces, and what fak does with each

The adapter touches **only** vLLM's documented public surfaces. No vLLM source is
vendored, forked, or patched. This is not merely a review promise: the
`TestVLLMAdapterConstructsOnlyPublicVLLMEndpoints` guard drives the adapter's real
path constructors and fails closed if any HTTP path it emits leaves the public
allowlist (`/v1/chat/completions`, `/v1/completions`, `/metrics`) or dips below the
public boundary (`/internal`, `/debug`, path traversal) — so an edit that reaches a
forked or internal vLLM endpoint breaks a witness instead of passing silent review.

1. **OpenAI-compatible HTTP** (`/v1/chat/completions` and `/v1/completions`,
   `stream=true`). `Admit` lowers a fak tool call onto the chat or completions
   route, forwards per-request sampling, and pumps the SSE response into the
   request's `Tokens()` channel. `cache_salt` (issue #1841 tenant isolation) and
   an optional priority field (when `FAK_VLLM_PRIORITY_SCHEDULING` advertises the
   V1 priority scheduler) are derived from the fak turn identity, never secret.
2. **KV-cache-events stream.** vLLM publishes `BlockStored` / `BlockRemoved` /
   `AllBlocksCleared` events over ZMQ/msgpack. fak takes no ZMQ or msgpack module, so the
   transport decoder lives outside this leaf and hands the adapter decoded
   `VLLMKVEventBatch` values at the `VLLMKVEventSource` seam
   (`VLLMJSONKVEventSource` is the stdlib NDJSON bridge used by tests).
   `RunKVEventSubscription` folds each batch into the per-worker
   `PrefixResidencyIndex` and the shared `CacheEventRecorder`. This is the
   prefix-residency producer the fleet router's cache-aware routing consumes.
3. **Prometheus scrape.** `ScrapeServingMetrics` reads the worker's `/metrics`
   endpoint and `ParseVLLMPrometheus` normalizes vLLM's
   `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`,
   inter-token-latency, queue, `gpu_cache_usage_perc`, and prefix-cache
   counters into the one shared `ServingMetricsSnapshot` L2 schema, labelled
   `engine="vllm",worker="<id>"`. The normalized rows render as
   `fak_serving_ttft_seconds`, `fak_serving_tpot_seconds`,
   `fak_serving_itl_seconds`, `fak_serving_kv_cache_usage_ratio`, and the rest
   of the `fak_serving_*` family.

## Bring-up

Point the adapter at a vLLM V1 worker launched with `vllm serve`:

```powershell
$env:FAK_VLLM_BASE_URL="http://vllm-host:8000/v1"
$env:FAK_VLLM_MODEL="served-model"
$env:FAK_VLLM_API_KEY="optional-upstream-key"
$env:FAK_VLLM_WORKER_ID="vllm-0"            # defaults to "vllm"
$env:FAK_VLLM_METRICS_URL="http://vllm-host:8000/metrics"   # optional; derived from base if absent
$env:FAK_VLLM_PRIORITY_SCHEDULING="1"       # optional; advertises the V1 priority scheduler
fak run --trace trace.json --engine vllm
```

`FAK_VLLM_METRICS_URL` is optional: when unset, the adapter derives `/metrics`
by stripping the trailing `/v1` from the base URL. Scrape and KV-event
subscription are independent of the request path, so a gateway can serve traffic
through the OpenAI frontend while a control loop observes residency and serving
metrics in parallel.

## Trust boundary — the house-honesty note

This is the honesty call the issue's acceptance requires stated plainly.

fak's bit-exact middle-span KV `Evict` (quarantine a poisoned tool result's span,
re-RoPE the survivors, `max|Δ|=0`) is a **native in-kernel guarantee**. A ridden
vLLM worker does **not** expose it: vLLM's public control plane offers only a
whole-prefix reset (`POST /reset_prefix_cache`). So on the ridden-engine path,
exact-span KV governance **degrades to whole-prefix flush**:

- `enginecache.SupportsExactSpan(EngineVLLM)` returns **false**
  (`internal/enginecache/enginecache.go`, `EngineVLLM = "vllm"`).
- When a quarantine directive names a span, the enginecache client collapses the
  set to one auditable whole-prefix reset and marks the result `Degraded=true`
  with `DegradeReason="exact_span_unsupported_whole_prefix_flush"`.
- If an operator instead requires exact span (`--engine-cache-require-exact-span`),
  the call **fails closed** rather than pretend a precise span was evicted.

The adapter claims nothing it cannot witness. Whole-prefix reset is safe (the
poisoned span is gone) but coarse (every other resident prefix is evicted too).
Bit-exact span eviction over a ridden vLLM is the Track-B native-KV story, not
this adapter; faking it here would be the category error the issue's non-goals
forbid.

No vLLM source is forked or patched. The adapter consumes only the public
OpenAI, KV-event, and Prometheus surfaces.

## Host-independent witnesses, and what is deferred to hardware

The contract above is witnessed on a GPU-free host by the `internal/engine`
tests, which drive faithful vLLM-shaped stubs (an OpenAI SSE frontend, an NDJSON
KV-event stream, and a Prometheus fixture):

```powershell
go test ./internal/engine -run VLLM
```

- `TestVLLMEngineIsRegisteredLifecycleDriver` — the `vllm` id resolves to a
  registered `LifecycleEngine`.
- `TestVLLMHTTPAdapterStreamsChatAndCompletions` — chat and completions routes
  stream SSE deltas through `Tokens()` and assemble a `Result`.
- `TestVLLMKVEventSubscriptionFeedsResidencyAndCacheMetrics` — BlockStored /
  BlockRemoved / AllBlocksCleared fold into the residency index and recorder.
- `TestVLLMPrometheusNormalization` — vLLM counters normalize into the
  `fak_serving_*` L2 schema, per-worker labelled.
- `TestVLLMAdapterConstructsOnlyPublicVLLMEndpoints` — the no-fork / public-surface
  contract is executable: every HTTP path the adapter constructs stays inside the
  documented public vLLM V1 allowlist, with no internal/debug/traversal path.
- `TestVLLMGovernanceResolvesToEngineVLLM` — the cache-control plane resolves a
  directive to the vLLM engine.
- `TestVLLMLiveWorkerSmoke` (`vllm_live_smoke_test.go`) — an env-gated live
  worker smoke that **skips** unless `FAK_VLLM_BASE_URL` and `FAK_VLLM_MODEL`
  point at a real vLLM V1 worker.

Two acceptance items are deliberately **not** claimed here because they require
a live vLLM V1 worker on GPU hardware this dev box does not have:

- End-to-end `/v1/chat/completions` and `/v1/messages` serving through a **real**
  worker (the `/v1/messages` Anthropic-to-OpenAI drop-in is itself witnessed
  GPU-free by `internal/gateway/vllm_messages_dropin_test.go`).
- The measured **fak-fronted-vLLM vs raw-vLLM** parity overhead (added TTFT/TPOT,
  throughput delta) on the sibling parity bench. The issue body says "would need
  measurement until run on real hardware"; that measurement is the bench-harness
  sibling's job, not a number asserted here.

## Generation horizon (`gen/second-next`)

This issue carries `gen/second-next`: an architectural option whose value depends
on assumptions that must be re-checked at promotion time.

- **Promotion evidence (toward `gen/next`).** The blocking seed shipped — the
  admit/step/stream `LifecycleEngine` seam is on `main` (#46, `cbda603`), the
  adapter compiles against it, and the metrics-schema + residency-index siblings
  the adapter feeds are in place. A live-worker pass on GPU hardware is the
  remaining promotion witness.
- **Demotion / retirement evidence.** If the Track-B native engine ships the
  base serving items (continuous batching, paged KV, prefix cache) first, the
  adapter's stated value — "serve many GPU nodes before the native engine is
  ready" — collapses, and it should be demoted toward `gen/future` or retired
  rather than promoted. The adapter stays a useful parity baseline either way.
- **Invalidating assumption in this artifact.** The honesty note assumes vLLM's
  public surface stays whole-prefix-reset-only. If a future vLLM release exposes
  a documented exact-span eviction endpoint, `SupportsExactSpan` for vLLM should
  turn true (or an operator wires `ExactSpanEndpoint`), and the degradation
  described above no longer applies.

## Related

- [Dual-track serving plan](dual-track-serving-plan.md) — the RIDE + NATIVE epic
  this adapter belongs to (Track A).
- [Dynamo interop](dynamo-interop.md) — the sibling ridden-engine adapter.
- [Engine seam proof](../proofs/engine-seam.md) — the exact-span fail-closed
  theorem the degradation above honors.
- [Supported engines](../supported/engines.md) — the base-URL proxy wiring for a
  vLLM worker that does not need the registered adapter.
