# DeepSeek V4.1 Flash on one Strix Halo

**Final result: 4.61 decode tok/s** for `DeepSeek-V4.1-Flash-Q2_K` on one 128 GB AMD Strix
Halo (Ryzen AI Max+ 395, Radeon 8060S gfx1151, Vulkan/RADV), single stream, greedy,
warm 128-token request, observed 2026-10-22 `[HW-WITNESSED]`
([receipt](receipts/halo-v41-100-30/v41-cached-readback-stage-20261022-strix3.md)).

The target is ≥ 30 tok/s sustained decode and ≥ 100 tok/s prefill on one box. It is
**not met**.

## Result

| Measure | Value |
| --- | --- |
| Warm decode, fixed 45 GiB expert ring | **4.61 tok/s** (219 ms/step) |
| Same-session base before the change | 3.57 tok/s (282 ms/step), so +29% |
| Warm decode, auto-sized ring | 5.18 tok/s (194 ms/step); ring size differed between arms (45.6 vs 59.8 GiB), so not the controlled figure |
| First pass, cold experts | 1.8–2.0 tok/s |
| Cold prefill | near 2 tok/s |
| Greedy output parity | sha equal in 32/32 requests across 8 arms |

## What moved it

`caa02db68` makes the shared transfer staging buffer `HOST_CACHED` by default
(`FAK_VULKAN_STAGE_HOST_CACHED=0` opts out). CPU reads of uncached, write-combined GTT
staging run near 1 GB/s on this APU, so the per-step device-to-host copy fell from 55 to
2.1 ms at the same 51.9 MB/step. Bytes moved did not change.

## How it was run

```text
fak serve --backend vulkan --native-context-tokens 4096 --expert-ring-bytes 45G
  model: DeepSeek-V4.1-Flash-Q2_K GGUF (7 shards)
  env:   FAK_SESSION_STATE=off
  request: /v1/completions, temperature 0, max_tokens 128, batch 1
```

Decode rate is `inkernel_chat decode=` seconds divided by 127 steps. The receipt has the
full arm table and build binding.

## Remaining levers (not yet measured)

- Keep the attention chain on device: about 525 KB/layer of the ~52 MB/step readback.
- Device-side greedy argmax: full logits are 517 KB/step read back for host argmax.
- This Q2_K checkpoint has no MTP/nextn tensors, so self-speculative decoding does not apply.

Earlier V4.1 receipts for this target live in
[`receipts/halo-v41-100-30/`](receipts/halo-v41-100-30/).
