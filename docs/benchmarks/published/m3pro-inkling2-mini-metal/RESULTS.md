---
title: "Historical campaign record: Inkling-2-Mini on the 36 GiB M3 Pro (retired vendored Splash port)"
description: "Historical campaign record, not a current fak-native Metal claim: this 2026-11-24 run measured the retired in-process vendored Splash Metal port on a 36 GiB M3 Pro. Pure internal/metalgemm resident receipts are pending."
---

# Historical record: Inkling-2-Mini on the 36 GiB M3 Pro (retired vendored Splash port)

> **Status: HISTORICAL / RETRACTED AS A FAK-NATIVE METAL CLAIM.**
> This 2026-11-24 campaign measured fak's **retired in-process vendored Apache-2.0 Splash source port** at commit `1f31533f2`, **not** fak's pure `internal/metalgemm` resident Metal backend. The vendored Splash port has been retired from `main`. Do not cite the `40.3 tok/s` figure below as a current fak-native Metal result; pure `internal/metalgemm` resident Metal receipts for M3 Pro are pending publication.

The MacBook has 18 GPU cores and runs macOS 27.0.1 on arm64. All engines read the same Mainline_Support UD-Q3_K_XL weights. The checkpoint contains 24 weight layers, 12 recurrent layers and three passes through those recurrent layers, for 48 layer visits per token. The target and DFlash2 sidecar came from owned LAN storage.

| Model file | Bytes | SHA-256 |
| --- | ---: | --- |
| Inkling-2-Mini-UD-Q3_K_XL.gguf | 13,631,779,296 | e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639 |
| Inkling-2-Mini-DFlash2/model.safetensors | 989,975,616 | 217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee |

## Historical provenance

| | |
| --- | --- |
| Measured historical source | public `main` at [1f31533f2](https://github.com/anthony-chaudhary/fak/commit/1f31533f2ccba844c2f03d176bfbb98772913754), committed 2026-11-24 (retired vendored Splash source port) |
| Current status on `main` | Retired. Current `v0.59.0` and `main` run pure `internal/metalgemm` resident Metal for Inkling-2-Mini; see [INSTALL.md](../../../../INSTALL.md) for current installation. |

Models: the target is published as `Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file` in [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The DFlash2 draft is [incoai/Inkling-2-Mini-DFlash2@4f9088ea](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2). Put the draft in an `Inkling-2-Mini-DFlash2/` directory next to the GGUF; Fak picks it up automatically. Check both files against the SHA-256 values above. This run used a 36 GB M3 Pro. The process peaked at about 16 GB.

Only this page and [perf-receipt-v2.json](perf-receipt-v2.json) are published in the public repository. Raw per-run logs, reproduction scripts, and schedules are retained in the maintainer archive and are not linked from this public copy.

## Historical retired vendored Splash port qualification

On this MacBook, the retired vendored Splash Metal port reached 97.5% to 107.4% of Splash 1.7.2’s decode rate on the two natural continuations. It reached 148.1% on repeated text, where verified prompt lookup can supply longer exact runs. Prefill ranged from 98.2% to 107.5% of Splash. These ratios describe warm, single-request throughput for this checkpoint, device, and schedule on the retired vendored port, and do not describe current pure `internal/metalgemm` resident Metal.

Both engines read the same target and DFlash2 weights. The comparison ran four fresh processes in the order Splash, Fak (vendored Splash port), Fak (vendored Splash port), Splash. Each received the same 24 requests, including identical warmups and six measured requests per shape across the two processes. Splash reported zero prompt-cache hits on all requests. Fak reset the request state between prompts. Each cell reports the median observed tokens per second.

| Prompt, input / output tokens | Fak (retired Splash port) prefill | Splash 1.7.2 prefill | Fak (retired Splash port) decode | Splash 1.7.2 decode | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Capitals, 19 / 32 | 160.46 | 149.28 | 40.33 | 37.54 | 1.074× |
| Fibonacci, 20 / 32 | 145.25 | 143.21 | 55.50 | 54.92 | 1.011× |
| Repeated text, 128 / 32 | 320.74 | 323.77 | 177.05 | 119.58 | 1.481× |
| Repeated text, 512 / 1 | 458.75 | 467.37 | — | — | — |
| Capitals, 19 / 128 | 153.05 | 148.19 | 47.76 | 46.49 | 1.027× |
| Fibonacci, 20 / 128 | 145.56 | 148.09 | 63.57 | 65.21 | 0.975× |

For 128-token Fibonacci, the retired port ranged from 58.68 to 67.90 decode tokens/s and Splash from 59.00 to 71.14. Their medians differ by 2.5%. The verifier used a median of 21 cycles and Splash used 19.

**Token agreement.** Each engine matched all 582 short-suite IDs and all 768 Fibonacci IDs against the respective target references. On 128-token capitals, the retired port matched the independent Koosh continuation in three of six runs and Splash in two of six. Both engines first diverged at positions 42, 66 or 112, depending on the run. At position 112 both can choose Koosh’s fifth-ranked token within 0.237 nats. That choice falls outside the existing runner-up-only test. The target-only control matched both 128-token references exactly. A physical test verified continuity when EOS or the generation limit lands inside a speculative block.

**Implementation (retired).** That build ran an in-process source port of the Apache-2.0 Splash Apple GPU operators. It verified candidates against the full target, retained the accepted target KV, short convolution, and draft context on the GPU, and checked the resulting row mask against an independent host-side walk before exposing tokens. An ABBA comparison of the same driver with embedded source versus the upstream precompiled shader library found a 1.5–2.4% decode gain from the precompiled version and a 9.1% loss on fixed verification.

**Follow-up measurements.** For validation we held the verifier at 8 or 16 rows and used the same two 128-token prompts. The 16-row run matched both independent sequences exactly but dropped Fibonacci from 71.87 to 59.86 tokens/s in that trial. The 8-row run reached 72.27 tokens/s and still diverged on capitals. These single warmed observations supported retaining adaptive sizing in that historical port.

**Source and tests.** The historical public revision was [1f31533f2](https://github.com/anthony-chaudhary/fak/commit/1f31533f2ccba844c2f03d176bfbb98772913754). All changed production sources matched the benchmark build byte for byte. The public CLI, agent, loader, and Metal package checks passed, and the changed-path boundary and publication scans passed.

**Installed CLI (historical run).** At revision `1f31533f2ccb`, automatic backend selection reproduced the 32-token raw capitals text. The chat command returned `4` in 5.09 seconds, including loading. A 16-token limit ended before any visible answer. The first installed raw run took 21.99 seconds, including first-use shader compilation. Repeating the same command in a fresh process took 4.43 seconds.

The standard [hardware receipt](perf-receipt-v2.json) records the device, model, kernel, argv, and timing provenance for this historical run.

## Comparison

Mainline llama.cpp b12681 measured 18.9, 18.9 and 19.0 decode tokens/s for capitals, Fibonacci and the repeated 128-token prompt. Koosh0610/llama.cpp-inkling at cf5f8a7 measured 23.4, 23.2 and 23.3. Splash 1.7.2 with DFlash2 measured 39.5, 56.7 and 120.6 in the initial control. Each engine returned the same 32 token IDs on each prompt. The repeated prompt contains long verbatim matches and benefits from prompt lookup.

The paired control used a fresh Splash process and a fresh Fak process on this Mac. Both ran one request at a time and received identical token IDs. Each of the four shapes ran once to warm the device and three times for measurement. Every Splash request had a unique cache salt and reported zero matched prompt tokens. Each Fak request started with empty logical token history, KV, and convolution state. Medians of each phase were reported independently. Prefill throughput divides all input tokens by wall time through the first-token projection. Decode throughput divides the remaining 31 tokens by decode seconds. The 512-token case requests one output and measures only prefill. Cold process launch, weight loading, and shader compilation are separate from warm throughput.

## Measured implementation changes (historical progression)

The first packed Metal implementation returned all 64 short-prompt oracle IDs. Warm decode was 7.3–7.6 tokens/s. Keeping state on the GPU raised the two-prompt paired medians to 18.60 and 19.83. Fusing controller projections raised those medians to 20.60 and 21.28 with the same IDs.

Fak's retired in-process source port of the Apache-2.0 Splash operators owned model loading, tokenization, generation, greedy acceptance, and request state. The Apple Metal device compiled embedded shader source. A transaction test staged a bad continuation, accepted a prefix, and discarded the remaining rows. A second test followed a noncontiguous tree path and checked the following token against full-depth target inference.

The first fused-tree candidate matched 291 of 291 IDs across the 12 measured requests, including the three one-token prefill checks. Its medians against the paired Splash control follow.

| Input shape | Fak (retired Splash port) prefill | Splash prefill | Fak (retired Splash port) decode | Splash decode |
| --- | ---: | ---: | ---: | ---: |
| Capitals, 19 tokens | 130.76 | 144.10 | 31.59 | 37.20 |
| Fibonacci, 20 tokens | 133.20 | 134.18 | 55.87 | 52.79 |
| Repeated text, 128 tokens | 317.33 | 327.95 | 156.67 | 114.85 |
| Repeated text, 512 tokens | 461.72 | 458.09 | — | — |

Splash source was pinned at [2afc7337](https://github.com/incoai/splash/tree/2afc7337aacc044722fa95e4cb3da75c281a19a8). The two llama.cpp references are [mainline b12681](https://github.com/ggml-org/llama.cpp/releases/tag/b12681) and [Koosh's Inkling fork](https://github.com/Koosh0610/llama.cpp-inkling/tree/cf5f8a770eb4ee7ee03e35f8975b35715dbd3536).

## Adaptive verification and held-out continuation

The next candidate reused one idle session arena and reset its logical state between requests. The fused verifier combined short prompt matches with neural proposals and used observed GPU time and acceptance to size the next block. Median session acquisition and teardown overhead fell from 42 ms to 1 ms. Against a second interleaved Splash control it reproduced all 291 short-suite output IDs.

| Input shape | Fak (retired Splash port) prefill | Splash prefill | Fak (retired Splash port) decode | Splash decode |
| --- | ---: | ---: | ---: | ---: |
| Capitals, 19 tokens | 151.48 | 145.96 | 37.89 | 35.74 |
| Fibonacci, 20 tokens | 141.22 | 131.03 | 54.56 | 51.90 |
| Repeated text, 128 tokens | 324.34 | 321.16 | 164.18 | 112.13 |
| Repeated text, 512 tokens | 428.87 | 448.31 | — | — |

Extending the two natural continuations to 128 tokens exposed a separate numerical and throughput question. The adaptive candidate matched 128 of 128 Fibonacci tokens, at 53.16 tokens/s against the initial Splash run's 72.91. Capitals first diverged at the 42nd generated token. The candidate chose token 160146, while Splash and both llama.cpp controls chose 151462 there. The same binary with single-token target verification matched the two llama.cpp capitals continuations for all 128 tokens. Raw short-suite evidence alone cannot establish longer-run greedy equivalence.

## Wide verification

The wide verifier could combine seven neural predictions and up to 15 prompt-lookup proposals in a 16-row transaction. A longer verified prompt copy could use up to 32 rows. All 291 measured short-suite tokens matched. The CLI also reproduced the raw 32-token capitals text with 6.04 seconds elapsed, including loading and teardown.

| Input and output tokens | Fak (retired Splash port) prefill | Splash prefill | Fak (retired Splash port) decode | Splash decode |
| --- | ---: | ---: | ---: | ---: |
| Capitals 19 / 32 | 150.39 | 145.96 | 38.23 | 35.74 |
| Fibonacci 20 / 32 | 133.76 | 131.03 | 54.95 | 51.90 |
| Repeated text 128 / 32 | 322.99 | 321.16 | 161.94 | 112.13 |
| Repeated text 512 / 1 | 463.45 | 448.31 | — | — |
| Capitals 19 / 128 | 142.40 | 166.83 | 45.36 | 52.97 |
| Fibonacci 20 / 128 | 144.01 | 163.70 | 57.73 | 71.36 |

Each 128-output arm included one warmup and three measured repeats. For Fibonacci, the three Fak and three Splash runs matched the same 128 Koosh target tokens. For capitals, one of three Fak runs and two of three Splash runs matched all 128 Koosh tokens. Two Fak repeats chose a separate space and then spelled “Brasilia” where Koosh chose “ Brasília”. At this first mismatch, output index 65, the chosen token was Koosh’s runner-up by 0.086 nats. Repairing the prefix and continuing found a second runner-up at index 111, 0.139 nats behind the top choice. Repairing that prefix reproduced the remaining 16 IDs. This met the existing Inkling oracle’s two-reanchor, runner-up, 0.5-nat criterion. At index 111 one Splash run chose Koosh’s fifth-ranked token, 0.236 nats below its first. Exact-token and near-tie results are reported separately.

A target-only shape control forced the identical 40-token prefix into single rows or fully accepted multirow blocks. The next choice flipped with the block width despite zero rejected tokens. An independent state audit checked 59,424,768 bytes of convolution history and 12,874,752 bytes of retained KV values and scales over 31 commits. Each matched its source row exactly.
