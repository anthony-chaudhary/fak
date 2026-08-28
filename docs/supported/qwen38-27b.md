---
title: "Qwen3.8-27B native inference support and acceptance evidence"
description: "Supported Qwen3.8-27B aliases, native Metal and CUDA execution boundaries, acceptance status, and links to captured benchmark evidence."
---
# Qwen3.8-27B

Tracking epic: [#8011](https://github.com/anthony-chaudhary/fak/issues/8011)

Captured product proof: [Qwen3.8-27B acceptance evidence](../_witnesses/issue-8623-qwen38-27b/README.md) records the admitted model identity, native engine path, benchmark envelope, and remaining limitations.

Qwen3.8-27B has exact built-in aliases. An alias is a launch identity, not a
claim that every target machine has passed acceptance:

| Alias | Exact target | Intended path |
|---|---|---|
| `qwen38`, `qwen38:27b` | `hf://unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` | GGUF on Apple Metal or NVIDIA CUDA |
| `qwen38:27b-fp8` | `hf://Qwen/Qwen3.8-27B-FP8` | A100 upstream engine such as vLLM |

Exact frozen identities used by the 2026-08-18/19 campaign:

- base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- GGUF repository: `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- Q4_K_M SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
  (`17,106,775,008` bytes)
- FP8: `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`

The upstream architecture is `Qwen3_5ForConditionalGeneration` / `qwen3_5`: 64
decoder layers, hidden size 5120, 24 attention heads, 4 KV heads, a 3:1
GDN/full-attention cadence, and a 262144-token configured context. It also has a
vision tower. Qwen3.5 architecture compatibility is relevant, but does not
replace exact Qwen3.8 acceptance.

## Support status — 2026-08-28

- The canonical `qwen38:27b` Q4_K_M artifact is pinned in the model registry and
  covered by registry tests, preventing the preferred identity from drifting.
- Native receipt and readmit lineage are test-bound across serving and model-engine
  paths, so a resumed run must retain its native Qwen identity rather than merely
  reporting a successful response.
- The Vulkan path now includes Qwen GDN routing and decode plus host-visible Q4_K
  weight staging. These are implementation and support facts, not an accepted
  performance result.
- AMD/Vulkan remains **AWAITING REMEASURE** until a native full-model run produces a
  quality-complete, comparable hardware receipt. The canonical
  [Qwen performance index](../benchmarks/QWEN-PERFORMANCE-INDEX.md) owns that status.

The [August 27 dogfood snapshot](../notes/QWEN-TRAJECTORY-SNAPSHOT-DOGFOOD-2026-08-27.md)
and [August 28 usage-outcome snapshot](../notes/QWEN-TRAJECTORY-SNAPSHOT-USAGE-OUTCOMES-2026-08-28.md)
add operational and diagnostic context only; neither establishes a new benchmark result.

## First-class default

Qwen3.8-27B Q4_K_M is fak's default model identity for serving demos and local runs.
Inspect the exact identity without downloading weights:

```console
fak model-default
fak model-default --json
```

The machine-readable form reports alias `qwen38:27b`, its exact `hf://` artifact, and
coding/tool capability. `default` resolves through the same registry seam, so
`fak run default` and `fak model pull default` remain aliases for that exact identity.
Neither inspection nor listing starts a download. Hosts that cannot satisfy the roughly
17 GB Q4_K_M artifact should select the explicitly bounded
`qwen2.5-coder:3b` fallback rather than silently running Qwen3.8 on CPU. Promotion to a
universal zero-argument run/serve default remains gated by the MacBook report in #8061
and the cache-value campaign in #8127.


## Resolve and inspect

```console
fak model ls
fak model load qwen38:27b
```

`model load` downloads the exact GGUF and verifies its Hugging Face LFS digest.
The Q4_K_M file is large; check free disk and memory before pulling it.

## Measured support envelope (2026-08-20)

| Route | Hardware | Verdict | Evidence |
|---|---|---|---|
| Native fak Metal GGUF | M3 Pro, 18 GPU cores, 36 GiB unified memory | **PASS_NATIVE_METAL_OPT_IN** | The streamed Q4_K path reaches readiness, identifies `metal/qwen35-hybrid-session-v1`, and passes exact `Q38`, strict JSON, an admitted tool call, warmup/request serialization, and rollback. The functional run used 18,920,570,880 B maximum RSS with zero swaps; request p95 was 37.208 s. Evidence: [`qwen38-27b-2026-08-20`](../_witnesses/qwen38-27b-2026-08-20/README.md). |
| llama.cpp Metal behind fak | same MacBook | **PASS_DELEGATED_METAL** | llama.cpp 9828 (`ebd048fc5`) loads in about 8.15 s at roughly 17 GiB RSS; exact model ID, `Q38`, schema JSON, and an admitted correlated tool call pass through fak. This is a compatibility control, not native-fak Metal acceptance. |
| Native fak CUDA GGUF | 40 GB A100 | **PASS_NATIVE_CUDA_GGUF** | Exact model identity, `Q38`, schema JSON, and an admitted correlated tool call pass through the native `cuda/qwen35-gdn-ssm-decode-v1` forward with no CPU/model/backend fallback. Startup is 52.431 s. Evidence: [#8070](https://github.com/anthony-chaudhary/fak/issues/8070#issuecomment-5337879917). |
| vLLM FP8 behind fak | 2× 40 GB A100, tensor parallel | **PASS_FP8_TP2_THROUGH_FAK** | Exact official FP8 revision boots with vLLM 0.27.1 at about 38,069 MiB per GPU; model identity, `Q38`, strict JSON, correlated tool use, and fak admission pass. One 40GB A100 remains **HOLD_FP8_BOOT**. Evidence: [#8075](https://github.com/anthony-chaudhary/fak/issues/8075#issuecomment-5338921901). |

The native Metal PASS is deliberately scoped to the explicit streamed-Q4_K launch below.
It is not a parity claim: full-prefill turns decoded at 2.3–2.9 tok/s, while fully cached
short turns varied from 0.4–1.3 tok/s; the same machine's delegated llama.cpp control
measured 7.29 tok/s. Cold and warm-file-cache readiness were 103.858 s and 34.897 s.
The default capacity estimator still rejects this route until [#8101](https://github.com/anthony-chaudhary/fak/issues/8101)
is recalibrated from the captured high-water mark, so the environment switch remains part
of the supported recipe.

The A100 GGUF PASS proves exact native CUDA execution plus structured-output and tool-call
acceptance without fallback. One A100 40GB remains outside the measured official-FP8
capacity envelope; the measured supported route is two 40GB devices with tensor parallelism.

## Candidate launch recipes

Native Metal is accepted on the measured 36 GiB host through the streamed-Q4_K route:

```console
fak model pull qwen38:27b
FAK_METAL_STREAM_Q4K=1 FAK_Q4K=1 fak serve \
  --gguf qwen38:27b --model qwen38:27b --metal --context-budget-tokens 4096
```

Wait for `/healthz` to report `"ok":true`, not merely for the socket to bind. During
startup it reports `warmup_pending`; an early request now queues behind that warmup, but
readiness-aware routing avoids charging a user request for the cold promotion. Stop the
candidate and restore the previous accepted service if readiness, any capability probe,
or memory pressure fails.

Native CUDA GGUF is accepted on the measured 40 GB A100 route:

```console
fak model pull qwen38:27b
FAK_Q4K=1 fak serve --gguf qwen38:27b --model qwen38:27b \
  --backend cuda --context-budget-tokens 4096
```

Official FP8 needs a fitting A100 topology. Start vLLM with the frozen FP8
revision and exact served ID, then put the fak OpenAI-compatible gateway in
front of it:

```console
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name qwen38:27b-fp8 --max-model-len 4096 \
  --tensor-parallel-size 2 --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice --reasoning-parser qwen3
fak serve --provider openai --base-url http://127.0.0.1:8000/v1 \
  --model qwen38:27b-fp8
```

Record A100 SKU/count, tensor-parallel topology, driver/CUDA/vLLM versions,
per-rank peak memory, and explicit no-fallback evidence. A single 40GB device is
not a supported FP8 recipe from this campaign.

## Promotion and rollback

Promotion requires immutable reports for deterministic text, structured JSON,
a correlated tool call, exact revision/hash, and performance/resource telemetry.
Fold them through `fak model acceptance-gate`, then join them with
`fak model readiness-inventory`. Missing or failed native Metal, A100 GGUF, or
A100 FP8 evidence remains a typed `HOLD`; do not infer a pass from the alias or
architecture family.

Rollback is explicit:

1. stop the Qwen3.8 candidate listener;
2. restart the previously accepted model/backend on its original port;
3. verify its authenticated `/v1/models` response before restoring traffic; and
4. retain the failed Qwen3.8 logs and exact identity instead of silently falling
   back inside the candidate process.
