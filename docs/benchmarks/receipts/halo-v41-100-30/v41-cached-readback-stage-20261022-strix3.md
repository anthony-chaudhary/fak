# V4.1 Flash Q2_K on one Strix Halo: cached readback staging cuts the warm decode step 282 -> 219 ms, 3.57 -> 4.61 tok/s, with identical greedy output (2026-10-22)

- **Result.** `[HW-WITNESSED]` `DeepSeek-V4.1-Flash-Q2_K` on one 128 GB AMD Strix Halo
  (Ryzen AI Max+ 395, Radeon 8060S gfx1151, RADV) decodes a warm 128-token greedy request
  at **4.61 tok/s** (219 ms/step), single stream, fixed 45 GiB expert ring. The
  same-session base measured 3.57 tok/s (282 ms/step): **+29%**.
- **Change.** `caa02db68` allocates the shared transfer staging buffer `HOST_CACHED` by
  default (`FAK_VULKAN_STAGE_HOST_CACHED=0` opts out). CPU reads of uncached
  (write-combined) GTT staging run near 1 GB/s on this APU; the D2H copy falls from
  55 to 2.1 ms/step at the same 51.9 MB/step. Bytes moved are unchanged.
- **Parity.** Greedy output sha is equal in 32 of 32 requests across 8 arms.
- **Auto ring.** The auto-sized ring reached 5.18 tok/s (194 ms/step), but its size
  differed between arms (45.6 vs 59.8 GiB), so the fixed-ring figure is the controlled one.
- **Not met.** The single-Halo target is >= 30 tok/s sustained decode and >= 100 tok/s
  prefill. Cold-expert first-pass requests run 1.8-2.0 tok/s and cold prefill is near
  2 tok/s.
- **Next byte levers.** The step still reads back about 52 MB: the attention chain
  (about 525 KB/layer, needs the chain kept on device) and full logits (517 KB/step for
  host greedy argmax).

## Binding

| Field | Value |
| --- | --- |
| Node | strix3, gfx1151, 128 GB, RADV |
| Base build | `73e475e3e` |
| Candidate build | base + the staging default flip, landed as `caa02db68` |
| Model | `DeepSeek-V4.1-Flash-Q2_K` GGUF, 7 shards |
| Invocation | `fak serve --backend vulkan --native-context-tokens 4096 --expert-ring-bytes {auto,45G}`, env `FAK_SESSION_STATE=off` |
| Requests | `/v1/completions`, temperature 0, `max_tokens` 128, batch 1: gold, gold, France, France (warm repeat) |
| Witnesses | `inkernel_chat decode=` (ms/step = decode s / 127), `fak_compute_readback_bytes_total` delta, expert page-in bytes delta, `perf record` |

## Arms (one session, in order A, B, A2, B2, A3, B3)

A = base default staging; B = base with `FAK_VULKAN_STAGE_HOST_CACHED=1`; B2/B3 = candidate,
no env. Cells are ms/step (tok/s).

| Request | A | B | A2 | B2 | A3 (45G) | B3 (45G) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gold first pass | 610 (1.65) | 626 (1.61) | 577 (1.75) | 561 (1.80) | 578 (1.75) | 515 (1.96) |
| gold second pass | 464 (2.17) | 321 (3.14) | 338 (2.98) | 291 (3.46) | 334 (3.01) | 294 (3.43) |
| France | 389 (2.59) | 316 (3.19) | 360 (2.80) | 337 (2.99) | 383 (2.63) | 303 (3.32) |
| France warm | 271 (3.72) | 195 (5.18) | 274 (3.68) | **194 (5.18)** | 282 (3.57) | **219 (4.61)** |
