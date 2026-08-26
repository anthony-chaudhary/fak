---
title: "Issue 9020 — owned Metal session profile"
description: "Captured 2026-08-25 on an Apple M3 Pro with 18 GPU cores and 36 GiB unified memory."
---
# Issue 9020 — owned Metal session profile

Captured 2026-08-25 on an Apple M3 Pro with 18 GPU cores and 36 GiB unified memory. The model was
`unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, artifact size
17,106,775,008 bytes, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The private local artifact path is deliberately rendered as `<exact-gguf>`.

## Capture summary

The capture binary was built from clean detached source revision
`4062891638e99bb6cc40a2328caa5265b0fb9ae2`. Its size was 8,522,050 bytes and SHA-256 was
`f43e784e4cce43486e9ef1709ff6daa64650461c81264723befb70edbda010ed`; `go version -m` reported
that exact `vcs.revision` and `vcs.modified=false`.

```console
FAK_METAL_STREAM_Q4K=1 FAK_Q4K=1 \
  <built-modelbench> -gguf <exact-gguf> -name qwen38:27b -q4k -metal \
  -decode-prompt=32 -decode-steps=64 -prefill-sizes=32 -prefill-reps=1 -decode-reps=1 \
  -native-performance-profile=issue-9020-metal-profile.json
wrote issue-9020-metal-profile.json
wrote issue-9020-metal-profile.receipt.json

<built-modelbench> -native-performance-readback issue-9020-metal-profile.json
native performance readback: PASS issue-9020-metal-profile.json

fak native-performance --profile issue-9020-metal-profile.json
class=synchronization-bound confidence=medium recommended_lever_id=metal.command-buffer-amortization
```

`FAK_Q4K_FREE_CPU`, `FAK_METAL_Q8_UPLOAD`, the Q4_K/Q8 group overrides, worker/budget controls,
and all other behavior-changing FAK controls were absent. The companion receipt binds that full
control map, the host, source, binary, artifact, and loaded model configuration. Its binding digest
is `5c0d8322540b6772dad708efd2b22d6ef16370dd723b6c16cdc25974476bf10b`.

The profile records engine `fak-native`, forward path `metal/qwen35-hybrid-session-v1`, and zero
fallbacks. One `model.Session` owned the P=32 prefill and all T=64 decode calls. Its raw receipt
contains 14,833 ordered Metal command-buffer events across Q4_K, Q6_K, and Q8 GEMM/GEMV/grouped/
fused routes. Every event is committed, waited to completion, read back by the host, and has finite
positive timing. Independent recomputation from those events exactly matches 14,833 command
buffers, 23,025 encoders, 15,322.027541660646 ms GPU duration, and 39,772.66442775726 ms host wait.
The raw-event digest is `fc5a52c4ab0e605b38831ca26f277240241b236e0136f47586076f62a9f4ce47`.
The typed fallback stream is empty, recomputes to zero promised CPU fallbacks, and has digest
`74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b`.

The model resident report supplied 24,595,492,864 resident bytes and process peak RSS supplied the
18,288,771,072-byte working-set reading. The profile SHA-256 is
`2c20871e18bba94de44274a6f5803ace9eab109f082985db857624d4d5fea9f1`; the companion receipt
SHA-256 is `6ef2454f956cae42baefbcdea733a3f2e28c5085866b50835b9dbded561359c8`.

Before the run, the established large Metal listener on port 8090 was resolved by PID and exact
command identity and stopped with TERM. The bounded watcher signaled only that matched owner,
stopped on every exit path, and never used a forced kill. After capture, the approved reversible
recovery command restored the identical command SHA-256
`a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d`; both `/health` and
`/v1/models` returned HTTP 200 through the bounded durability check. No private host, credential,
bridge, binary, or artifact path is retained here.
