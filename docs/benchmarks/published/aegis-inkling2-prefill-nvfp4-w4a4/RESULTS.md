---
title: "Campaign record: Inkling-2-Mini NVFP4 W4A4 CUDA prefill on an RTX 5080 (rtx5080-host, 2026-11-25)"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-prefill-nvfp4-w4a4 (2026-11-25): Inkling-2-Mini prefill with 4-bit expert weights and activations against the fastest llama.cpp fork on one RTX 5080."
---

# Inkling-2-Mini NVFP4 W4A4 CUDA prefill on an RTX 5080 (rtx5080-host, 2026-11-25)

**[HW-WITNESSED]** One RTX 5080 16 GB (sm_120), CUDA build with `--arch sm_120f`.
Model `Inkling-2-Mini-UD-Q3_K_XL.gguf` (sha256 `e2d03e8c…273639`), the same file for
every engine. Raw logs are held in the maintainer archive.

## Result

HTTP prompt rate on a 512-token prompt (`/v1/chat/completions`, streaming, context 4096,
concurrency 1, empty KV cache). Token time is the sum of prompt tokens over the sum of
server prompt milliseconds. Each arm has 12 timed samples in ABBA order.

| Engine | Prefill tok/s | Answer check |
|---|---:|---|
| **fak, NVFP4 W4A4 prefill** | **5861.0** | 12/12 correct |
| fak, int8 prefill (same binary, control) | 4585.8 | 12/12 correct |
| realugbun llama.cpp fork `2eb2530e` | 3012.4 | 12/12 correct |
| upstream llama.cpp `33aeb17` | 2144.6 | 12/12 correct |

- fak with W4A4 prefill reaches **1.95x** the fastest llama.cpp engine measured (5861.0 vs
  3012.4 tok/s) and is +27.8% over its own int8 path in the same binary.
- The same pair in process (`TestInkling2CUDAPrefillBench`, pp512, 8 warm rows per arm):
  4507.9 tok/s int8, 5638.5 tok/s W4A4 (+25.1%). Both arms pick the same next tokens in
  every row.
- The 2x target (6024.9 tok/s) is not met; W4A4 needs another 2.8%.
- Measured at public commit `d3317d16fc` (landed as `659b4e8219`) with
  `FAK_INKLING2_PREFILL_NVFP4_W4A4=1`.

## Quality gate and default

W4A4 became the default prefill path at public `27355b23dd` after a KL gate on the same
card (trunk `4b5fa3e11f`, sm_120f build):

| Path | Mean KL vs per-token step (nats) | Max | Same top token |
|---|---:|---:|---:|
| int8 | 0.001457 | 0.0456 | 84/85 |
| W4A4 | 0.003728 | 0.105 | 84/85 |

- The gate is mean KL at most 0.01 nats, llama.cpp's own batch-size KL on this file (0.0095).
  W4A4 passes at 39% of it. Five prompts, scored at the prompt end and over 16
  teacher-forced decode steps (85 positions).
- The W4A4 increase sits almost entirely in one low-margin long prompt
  (passphrase-recall-1946); neither path loses a greedy token there.
- The greedy oracle against llama.cpp passes 4/4 on both paths.

## Where this does not reach

- **Prompt rate only.** Median first visible token was 1165.6 ms for fak against
  195.5 ms for the fork, and median full request 1166.2 ms against 952.3 ms. Server prompt
  time is about 87 ms per request, so the rest is non-prefill latency this run did not
  diagnose. Do not quote the 1.95x as an end-to-end latency win.
- **Build-gated.** The W4A4 default only applies to a CUDA image built with
  `--arch sm_120f` or `sm_120a`. A plain `sm_120` or fat build keeps int8 silently.
- **Not in v0.58.0.** The default flip landed on `main` after the v0.58.0 tag.
- One card, one window. A fifth arm (upstream llama.cpp `29b52d39`) returned empty
  completions and is excluded, which is why the harness marked the full panel REFUSED.
