---
title: "Qwen3.8-27B native Metal witness"
description: "This directory freezes the 2026-08-20 native-fak acceptance of the exact unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe Q4KM checkpoint on..."
---
# Qwen3.8-27B native Metal witness

This directory freezes the 2026-08-20 native-fak acceptance of the exact
`unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
Q4_K_M checkpoint on an Apple M3 Pro MacBook with 36 GiB unified memory.
The 17,106,775,008-byte file had SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

The witness used `internal/model@r448+g8145dc0bea` for persistent streamed
Q4_K Metal weights and `internal/agent@r320+g6bb3c3dd55` for execution identity
and whole-forward Metal serialization. Logs resolved the native path as
`backend=metal`, `forward_path=metal/qwen35-hybrid-session-v1`, and `q4k=true`;
there was no CPU, model, backend, or quant fallback.

## Result

- `/v1/models`, exact `Q38`, strict JSON, and one correlated forced tool call
  admitted `ALLOW` all passed.
- Request p95 was 37.208 s. Full-prefill text/JSON turns decoded at 2.3–2.9
  tok/s; fully cached short text turns varied from 0.4–1.3 tok/s.
- Cold readiness was 103.858 s; a second run with the checkpoint in the OS file
  cache reached readiness in 34.897 s.
- Maximum RSS was 18,920,570,880 B for the functional campaign and
  19,558,662,144 B for the warmup-overlap campaign. Both runs recorded zero
  swaps. The larger `peak memory footprint` values in the summary are the
  macOS `/usr/bin/time -lp` accounting field, not RSS or allocated physical RAM.
- A request launched while `/healthz` reported `warmup_pending` queued behind
  warmup and returned exact `Q38`; a two-request burst also returned two exact
  `Q38` responses. The candidate stopped cleanly and the prior accepted service
  returned healthy after both runs.

This is `PASS_NATIVE_METAL_OPT_IN`, not a llama.cpp-parity claim. The launch
still requires `FAK_METAL_STREAM_Q4K=1` until the conservative capacity
preflight is recalibrated under #8101.

## Reproduce the machine-readable folds

```console
fak model acceptance-gate \
  --input docs/_witnesses/qwen38-27b-2026-08-20/metal-native-acceptance-input.json
fak model readiness-inventory \
  --input docs/_witnesses/qwen38-27b-2026-08-20/metal-native-acceptance-input.json \
  --artifact docs/_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json \
  --artifact-revision internal/model@r448+g8145dc0bea \
  --expected-corpus qwen38-native-metal-acceptance-2026-08-20 \
  --as-of 2026-08-20T22:04:00Z
```

The checked-in summary is scrubbed to hardware class and contains the exact
outputs, performance/resource readings, concurrency result, and rollback result.
It deliberately contains no private endpoint, account, credential, or filesystem
location.
