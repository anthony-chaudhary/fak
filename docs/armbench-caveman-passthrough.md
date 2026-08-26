---
title: "Caveman × fak six-arm provider benchmark (#6682)"
description: "Verdict: NOT-YET for provider-cache value. The live provider returned zero cache-read and cache-write tokens in every arm."
---
# Caveman × fak six-arm provider benchmark (#6682)

**Verdict: NOT-YET for provider-cache value.** The live provider returned zero cache-read and cache-write tokens in every arm. Therefore this run makes no cache-hit or cost-saving claim. It does establish the six tuned controls, semantic parity, and the clean passthrough path.

Pinned input: `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`, inherited from #6681 with hash checks. Live endpoint/model: the configured real OpenAI-compatible endpoint, `gpt-5.6-sol`. Raw request bodies, complete provider responses, provider usage, outputs, and timing are in [`docs/_witnesses/armbench-caveman-passthrough/live-gpt-5.6-sol/manifest.json`](_witnesses/armbench-caveman-passthrough/live-gpt-5.6-sol/manifest.json).

## Isolation

All six arms use the same prompt corpus, temperature 0, 4096 max output tokens, and three trials. Trial 1 is reported as cold; trials 2–3 are warm. The fak arms traverse an isolated in-process HTTP reverse proxy. Policy, shedding, transforms, routing, and local semantic cache are disabled. Cache-only arms add only a stable `prompt_cache_key`; request content is unchanged. No response is stored locally.

TTFT is `0`/not measured because this endpoint's reliable captured interface was non-streaming; the harness explicitly does not substitute wall latency for TTFT. Fak overhead is also NOT-YET as an independent latency quantity: concurrent real-provider calls prevent a valid paired subtraction. Wall latency is reported per arm without claiming causality.

## Results

All six arms passed 30/30 task-specific semantic gates. See the manifest `Summary` for cold/warm input, output, cache-write, cache-read, median wall latency, TTFT, cost, and overhead cells. Provider cache fields were empty (`prompt_tokens_details: {}`), so cache write/read are both zero. Cost is NOT-YET because this endpoint supplied usage but no bill and this run supplied no explicit price table.

Against tuned direct controls, Caveman still substantially reduces output tokens while retaining correctness. Fak passthrough adds no evidenced token benefit. The cache-only fak arms also add no evidenced benefit after Caveman (or normal) on this provider because provider cache reads were zero. The honest answer is therefore **no demonstrated incremental fak value after Caveman in this run**; the provider-cache claim remains NOT-YET rather than simulated.

## Reproduce

```powershell
$env:FAK_CAVEMAN_PASSTHROUGH_LIVE='1'
go test ./internal/armbench -run '^TestLivePassthrough$' -count=1 -timeout 12m -v
```
