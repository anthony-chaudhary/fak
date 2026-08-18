# Qwen3.8-27B

Tracking epic: [#8011](https://github.com/anthony-chaudhary/fak/issues/8011)

Qwen3.8-27B is represented by exact built-in model aliases. The aliases are
launch identities, not a claim that a particular machine has passed acceptance:

| Alias | Exact target | Intended path |
|---|---|---|
| `qwen38`, `qwen38:27b` | `hf://unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` | portable GGUF on Apple Metal or NVIDIA CUDA |
| `qwen38:27b-fp8` | `hf://Qwen/Qwen3.8-27B-FP8` | A100-class upstream engine (for example vLLM) |

The upstream model advertises `Qwen3_5ForConditionalGeneration` / `qwen3_5`:
64 decoder layers, hidden size 5120, 24 attention heads, 4 KV heads, a 3:1
GDN/full-attention cadence, and a 262144-token configured context. It also has a
vision tower. Existing Qwen3.5 architecture compatibility is therefore relevant,
but does **not** replace exact Qwen3.8 acceptance.

## Resolve and inspect

```console
fak model ls
fak model load qwen38:27b
```

`model load` downloads the exact GGUF and verifies its Hugging Face LFS digest.
The Q4_K_M file is large; check free disk and memory before pulling it.

## MacBook Metal candidate

On an Apple-Silicon MacBook with enough unified memory:

```console
fak model pull qwen38:27b
fak serve --gguf qwen38:27b --model qwen38:27b --metal
```

`--metal` is required for the prove-out so an unavailable Metal path fails
closed rather than producing a misleading CPU result. Record the exact fak
commit, macOS/SoC/RAM, resolved file SHA-256, load time, peak memory, TTFT,
prefill/decode throughput, and the raw exact-model acceptance report.

## A100 CUDA candidates

Portable GGUF parity:

```console
fak model pull qwen38:27b
fak serve --gguf qwen38:27b --model qwen38:27b --backend cuda
```

Official FP8 through an upstream engine:

```console
# Start the supported upstream engine with Qwen/Qwen3.8-27B-FP8, preserving
# that exact served model ID, then point fak at its OpenAI-compatible endpoint.
fak serve --base-url http://127.0.0.1:8000/v1 --model qwen38:27b-fp8
```

Record A100 SKU (40/80 GB), count/topology, driver/CUDA versions, tensor
parallelism, per-rank peak memory, and evidence that CUDA—not CPU fallback—ran
the campaign.

## Promotion rule

An alias appearing in `fak model ls` means it is discoverable and exact. It is
not proof of hardware readiness. Promotion requires immutable reports for the
exact resolved revision/hash covering deterministic text, structured JSON, a
correlated tool call, and performance/resource telemetry. Fold those reports
with `fak model acceptance-gate` and join them using
`fak model readiness-inventory`. Missing MacBook or A100 evidence is `HOLD`, not
an inferred pass.
