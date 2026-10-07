---
title: "Strix Halo head-to-head: fak serve vs stock llama-server"
description: "Repeatable same-box, same-model, same-prompt comparison of fak serve and stock llama.cpp llama-server on TTFT, prefill, decode and multi-turn cache reuse, with the raw per-request ledger agents can re-fold."
---

# Strix Halo head-to-head: fak serve vs stock llama-server

**Run:** `strix3-20261007-r3` on `strix3` (Ryzen AI MAX+ 395, Radeon 8060S, Vulkan/RADV), 2026-10-07.
**Model:** Qwen3.8-27B-UD-Q2_K_XL, temperature 0, thinking off.
**Raw rows:** [`strix-halo-h2h-latest.jsonl`](strix-halo-h2h-latest.jsonl) (one JSON row per request, schema `fak.bench.h2h.v1`).

All numbers below are WITNESSED: measured by `fak bench h2h` on the box itself, over loopback.

## What was compared

| Arm | Endpoint | What it is |
|---|---|---|
| `fak-serve` | `127.0.0.1:8080` | The box's production `fak serve` gateway (`--engine mock --provider openai`), proxying to the llama-server below |
| `llama-server` | `127.0.0.1:8091` | Stock llama.cpp `llama-server` build `dff6004`, Vulkan, 4 slots, flash-attn, n-gram speculation on |

Both arms reach the same llama-server process, so this run measures what the fak gateway adds or costs in front of llama.cpp. It is not yet a test of fak's native engine (see "Next steps").

## Results

Medians are position-balanced: each arm's median is the mean of its "ran first" and "ran second" medians, because on this server the second of two back-to-back conversations is ~400 ms slower on warm TTFT whichever arm it is.

| Scenario | fak serve | llama-server | Verdict |
|---|---|---|---|
| Cold 512-token prompt, TTFT | 2482 ms | 2386 ms | tie (n=3, within noise) |
| Cold 2k prompt, TTFT / prefill | 8273 ms / 285 tok/s | 8243 ms / 285 tok/s | tie |
| Cold 8k prompt, TTFT / prefill | 32973 ms / 280 tok/s | 33020 ms / 281 tok/s | tie (3 of 6 rows dropped as contended) |
| Decode, 256 tokens | 35.2 tok/s | 35.3 tok/s | tie |
| Short-prompt TTFT (68 tokens) | 518 ms | 493 ms | fak ~25 ms slower (gateway hop) |
| Multi-turn, turns 2-6 warm TTFT | 1697 ms | 1700 ms | tie |
| Multi-turn prompt-cache reuse | 95% | 95% | tie |

`fak bench h2h report` finds no cell where fak serve trails by more than 5%.

## Where fak loses

1. **fak's native engine is not in the race.** The native Vulkan engine (`fak serve --engine inkernel --backend vulkan`) could not be started beside the production llama-server on strix3. Its model load holds about 12 GB of anonymous host RAM, and the kernel OOM killer stopped it; the production services were untouched. The last published native numbers ([STRIX-HALO-BENCHMARK-RESULTS.md](STRIX-HALO-BENCHMARK-RESULTS.md), OBSERVED, not re-measured here) were 49 tok/s prefill and 16.8 tok/s decode. Against the 280 tok/s prefill and 35 tok/s decode measured above, that is about 5.7x behind on prefill and 2x behind on decode.
2. **The gateway costs about 25 ms of TTFT** on a short prompt. This is the whole measurable price of the hop; on prompts of 512 tokens or more it is lost in noise.
3. **fak serve hides llama.cpp's own accounting.** Responses through fak drop llama-server's `timings` block (prompt/decode rate, `draft_n`, `draft_n_accepted`), and non-streaming responses arrive with an empty `prompt_tokens_details`, so a client reading cache hits through fak sees none.

The comparison table in STRIX-HALO-BENCHMARK-RESULTS.md lists llama.cpp at about 48 tok/s prefill with no cross-request prefix cache. On this box today llama-server prefills at about 280 tok/s and reuses 95% of a multi-turn prompt, so that row understates the baseline fak has to beat.

## Next steps

- Run the native arm on a box whose llama-server can be paused for the window, or cut the native load path's host-RAM peak so it fits beside one. Then add `--arm fak-native=http://127.0.0.1:<port>/v1` to the same command.
- Check whether the ~400 ms "second conversation" penalty is llama-server swapping slot KV state. If it is, fak's gateway can win warm TTFT outright by keeping each conversation on its own slot.
- Pass llama-server `timings` and non-streaming `cached_tokens` through the fak gateway so cache and speculation data stays visible to agents.

## Reproduce

Build `fak` for linux/amd64, copy it to the box, and run it there so TTFT has no network hop:

```
fak bench h2h run --host strix3 --model Qwen3.8-27B-UD-Q2_K_XL \
  --arm fak-serve=http://127.0.0.1:8080/v1 --arm llama-server=http://127.0.0.1:8091/v1 \
  --slots-url http://127.0.0.1:8091/slots \
  --cold 512,2048,8192 --reps 3 --decode 256 --turns 6 --system 4096 --turn-tokens 256
fak bench h2h report            # latest run, table plus loss list
fak bench h2h report --json     # same fold for agents (schema fak.bench.h2h-report.v1)
```

Rows append to `.fak/nightrun/bench-h2h.jsonl` beside the gateway perf ledger unless `--ledger` says otherwise. Any arm whose name starts with `fak` is graded against the best other arm. The runner keeps the comparison fair in four ways:

- A per-arm nonce opens every prompt, so one arm never warms the cache for another.
- Arms alternate order request by request.
- A row is marked contended and left out of the medians when the llama-server `/slots` endpoint shows another client, or when the GPU stays busy after a settle wait.
- Rates are computed client-side the same way for every arm.
