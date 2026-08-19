# Qwen3.8-27B

Tracking epic: [#8011](https://github.com/anthony-chaudhary/fak/issues/8011)

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

## Resolve and inspect

```console
fak model ls
fak model load qwen38:27b
```

`model load` downloads the exact GGUF and verifies its Hugging Face LFS digest.
The Q4_K_M file is large; check free disk and memory before pulling it.

## Measured support envelope (2026-08-19)

| Route | Hardware | Verdict | Evidence |
|---|---|---|---|
| Native fak Metal GGUF | M3 Pro, 18 GPU cores, 36 GiB unified memory | **HOLD** | Metal is selected and all 866 tensors load, but startup dies before listener readiness after a measured 65.5 GiB peak footprint. Tracked by [#8067](https://github.com/anthony-chaudhary/fak/issues/8067). |
| llama.cpp Metal behind fak | same MacBook | **PASS_DELEGATED_METAL** | llama.cpp 9828 (`ebd048fc5`) loads in about 8.15 s at roughly 17 GiB RSS; exact model ID, `Q38`, schema JSON, and an admitted correlated tool call pass through fak. This is a compatibility control, not native-fak Metal acceptance. |
| Native fak CUDA GGUF | A100-SXM4-40GB | **HOLD_FUNCTIONAL** | CUDA preflight and exact text pass; strict JSON is malformed and forced tool use returns HTTP 502. Startup is 52.45 s and resident tensors total 24,693.88 MiB. Tracked by [#8070](https://github.com/anthony-chaudhary/fak/issues/8070). |
| vLLM FP8 behind fak | one A100-SXM4-40GB | **HOLD_FP8_BOOT** | vLLM 0.27.1 with a 4096-token cap reaches 38.95 GiB process use, then OOMs requesting another 784 MiB. Use an A100 80GB or supported multi-A100 topology; tracked by [#8075](https://github.com/anthony-chaudhary/fak/issues/8075). |

The delegated Metal PASS proves the checkpoint, tokenizer/template, schema, and
tool-call path can work on the MacBook. It does not promote native Metal. The
A100 GGUF run proves exact CUDA execution without CPU fallback, but text-only
success does not promote structured/tool acceptance. One A100 40GB is outside
the measured FP8 capacity envelope.

## Candidate launch recipes

Native Metal remains a fail-closed acceptance candidate:

```console
fak model pull qwen38:27b
fak serve --gguf qwen38:27b --model qwen38:27b --metal --context-budget-tokens 4096
```

Native CUDA GGUF remains a functional HOLD:

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
  --served-model-name qwen38:27b-fp8 --max-model-len 4096
fak serve --provider openai --base-url http://127.0.0.1:8000 \
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
