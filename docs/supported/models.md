---
title: "Models supported by fak — in-kernel architectures and any model you front"
description: "Which models fak supports: every model your serving engine or cloud exposes is fronted through the gateway unchanged, plus the model architectures the in-kernel reference engine runs and proves bit-exact (Llama, Qwen2/Qwen3, Gemma, GLM-MoE, GPT-OSS, SmolLM2)."
---

# Models supported by fak

"Supported" means two different things here, so this page is split in two.

The common case is the gateway. You point `fak serve` at whatever engine or cloud
already serves your tokens, and fak adjudicates the tool calls regardless of which
model produced them. Any model your upstream exposes works unchanged.

The narrower case is the in-kernel reference engine. fak ships a pure-Go forward pass
that runs a model itself, proven bit-exact against HuggingFace. That engine is a
correctness reference, not a production-throughput server, and it covers a fixed set of
architectures.

---

## Layer 1 — Models you front through the gateway

This is the default and the headline rule. `fak serve` is an OpenAI-, Anthropic-, and
MCP-compatible proxy. It adjudicates each tool call your agent proposes and then passes
the request through to the model your upstream serves. **fak does not restrict by model
id.** The model is the upstream's. If your engine or cloud exposes it, fak fronts it.

So the supported model list at this layer is "whatever your upstream serves":

| If your upstream serves… | fak fronts it because… |
|---|---|
| Claude, GPT, Gemini, Grok | the gateway speaks the OpenAI, Anthropic, and Gemini/xAI wires and proxies the request through |
| Llama, Qwen, DeepSeek, Mistral, GLM, any open-weights model | served behind an OpenAI-compatible engine (Ollama, vLLM, SGLang, llm-d, llama.cpp, LM Studio), so it is reached over the OpenAI-compatible wire |
| a local GGUF on your own box | served by your local OpenAI-compatible server and fronted the same way |

The mechanism is one fact: the gateway speaks the wires your agent already speaks
(`/v1/chat/completions`, `/v1/messages`, Gemini/xAI providers, and MCP), so the same
gate sits in front of whichever model serves your tokens. This is the [SHIPPED]
`fak serve` gateway and the `fak guard` front door for it.

Status: [SHIPPED]. Sourced in
[Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) (the
Gateway section), the [Integration index](../integrations/README.md), and the
[Compatibility matrix](../integrations/compatibility-matrix.md). For the exact wires and
endpoints, see [APIs, wires & MCP](apis-and-protocols.md) and the
[Gateway API reference](../fak/api-reference.md). For the providers fak fronts, see
[Clouds & hosted providers](clouds.md) and [Serving engines](engines.md).

When a tighter per-model claim is not sourced, treat it as **fronted via the
OpenAI-compatible wire** — the [Compatibility matrix](../integrations/compatibility-matrix.md)
is the surveyed list (47 harnesses, frameworks, backends, and protocols, each with the
exact repoint key). fak's role at this layer is the governance surface, not tokens per
second.

---

## Layer 2 — Models the in-kernel reference engine runs itself

This is the narrow path: `fak`'s own pure-Go forward pass, selectable with
`--engine inkernel` (the `inkernel` backend), the local-model `--gguf` path, or a real
weight export pointed to by `FAK_MODEL_DIR`. Here the model runs inside the kernel, so
the supported set is a fixed list of architectures.

**Read this first: the in-kernel engine is a correctness reference, not a fast token
server.** The honest scope, claim by claim, is in the
[Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — the
in-kernel forward pass is "correct, not fast" by origin, and the parity lane closed
decode to same-precision-peer parity without disturbing any correctness rung. For a
production token engine you use Layer 1 and front vLLM/SGLang/llama.cpp.

### The proven oracle model

| Model | Architecture | What is proven | Status |
|---|---|---|---|
| **SmolLM2-135M** (134.5M params / 272 tensors) | Llama-family decoder | A pure-Go forward pass runs in-process with the KV cache as a kernel-owned Go structure; every rung is proven against a HuggingFace oracle — embedding exact, per-layer cos=1.000000, final-logits max\|Δ\|≈4.4e-5, KV-decode and KV-quarantine-evict token-for-token identical | [SHIPPED] |

This is the default fixture (`.cache/smollm2-135m`) and the headline witness behind the
whole in-kernel claim. Source: the "In-kernel model" section of
[CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md),
`internal/model` `TestForwardMatchesHFOracle`, and
[`IN-KERNEL-MODEL-RESULTS.md`](../benchmarks/IN-KERNEL-MODEL-RESULTS.md) /
[`MODEL-BASELINE-RESULTS.md`](../benchmarks/MODEL-BASELINE-RESULTS.md).

### Architectures the in-kernel engine runs

The forward pass loads a checkpoint, derives its architecture axes from `model_type` /
`architectures` metadata, and runs the family-specific block topology, normalization,
activation, RoPE, and attention. Two levels of support are distinguished below, because
they are not the same claim:

- **Forward-pass proven** — the family has a HuggingFace oracle witness in
  `internal/model` (`assertForwardMatchesHFOracle`), so its forward output is checked
  bit-faithful against HF. The witness is weight-gated (it skips cleanly when the
  gitignored export is absent), so it is "proven when the export is present," not run on
  every CI box.
- **Config + architecture-axis** — the loader parses the family's metadata and the
  mechanical axes (block topology, norm, activation, RoPE scaling, MoE routing, sliding
  windows) are implemented and unit-tested, but there is no committed HF forward oracle
  for that exact family. Numeric forward correctness for these is not yet asserted by an
  oracle.

| Architecture | model_type / Architectures key | Support level | Witness / source |
|---|---|---|---|
| **Llama** (incl. Llama-3 RoPE scaling + EOS-list) | `llama` | Forward-pass proven | `internal/model` `TestForwardMatchesHFOracle`, `TestOptionalLlama3OracleCoversScalingAndEOSList`; default SmolLM2-135M oracle |
| **Qwen2 / Qwen2.5** | `qwen2` (legacy projection-bias default) | Forward-pass proven | shares the Llama-shape forward; `config_test.go` `TestConfigDerivesQwenLegacyBias…`; oracle path in `internal/model` |
| **Qwen3** (per-head qk-norm) | `qwen3` | Forward-pass proven | `TestOptionalQwen3OracleCoversQKNorm` |
| **Qwen3-MoE** (hybrid dense + sparse layers) | `qwen3moe` | Forward-pass proven | `TestOptionalQwen3MoEOracleCoversHybridDenseSparseLayers` |
| **Gemma2 / Gemma3** (sandwich-norm, (1+w) gain, tanh-GELU, local/global attention) | `gemma2`, `gemma3` | Forward-pass proven (Gemma3 oracle) | `TestOptionalGemma3OracleCoversLocalGlobalAttention`; Gemma axes in `arch.go` + `config_test.go` |
| **GLM-MoE-DSA** (GLM-5.2 lineage; DSA sparse attention + MoE) | `glm_moe_dsa` | Forward-pass proven (cacheless + session-cache oracle); DSA forward is research-grade | `TestOptionalGLMMoeDsaOracleForwardMatchesHFCacheless`, `…SessionCacheMatchesHF` |
| **GPT-OSS** (MoE, yarn RoPE, sliding-window layers, attention sinks) | `gpt_oss` | Config + architecture-axis | `config_test.go` `TestConfigDerivesArchitectureAxesFromMetadata` (gpt-oss case); attention-sink + softcap axes in `arch.go`. No committed HF forward oracle for this family |
| **Mistral** (sliding-window attention) | `mistral` | Forward-pass proven (SWA oracle, when exported) | `TestOptionalMistralSWAOracleNonVacuous` |
| **OLMo2** (post-norm, full-projection-width qk-norm) | `olmo2` | CPU forward numerically witnessed in CI (weight-free oracle) | `family_cpu_oracle_test.go` `TestOlmo2CPUNumericOracle` — the production forward vs an independent HF-semantics scalar reference on a deterministic synthetic fixture, prefill + decode, with a non-vacuity perturbation gate. No real-checkpoint HF export yet |

The mechanical-axis loader also parses several more families' metadata (GPT-NeoX /
Cohere / Falcon parallel-residual, MPT ALiBi, StableLM, DeepSeek-V2 MLA, MiniMax-M3
MSA). Those are wiring-and-config support exercised on synthetic configs in
`config_test.go` and `arch_test.go`; per-family numeric forward correctness needs a
CI numeric oracle (the OLMo2 row's pattern) or a re-exported HF oracle, so they are
not listed above as forward-pass-proven. DeepSeek-V2,
for example, has an oracle that documents the MLA tensor boundary but does not assert a
full forward. Treat any family not in the table above as parse-and-axis support, not a
proven forward.

### Load and quantization formats

The in-kernel engine loads weights in these formats. The reference path is f32; the
narrower-precision paths each have their own status.

| Format | Bytes/param | Support level | Source |
|---|---|---|---|
| **f32** (HuggingFace safetensors / the reference export) | 4 | [SHIPPED] — the proven-correct reference path; every oracle rung is checked against it | `internal/model` oracle; CLAIMS "In-kernel model" |
| **Q8_0 / int8 SIMD** (hand-written AVX2/AVX-512, CPUID-gated, scalar fallback; opt-in `Session.Quant`) | ~1.125 | In-flight increment, **not** [SHIPPED] — witnessed green in the working tree (argmax-exact vs the f32 oracle, decode near-parity with llama.cpp Q8_0), deliberately not given a SHIPPED row until the lane commits | CLAIMS "In-kernel model" (int8/Q8_0 lane note); `MODEL-BASELINE-RESULTS.md` Act 3 |
| **Resident Q4_K** (raw q4_k blocks stay resident, decode streams ~1.8× fewer bytes; the Qwen3.6-27B route) | ~0.5 | Available via `FAK_Q4K` (the resident-Q4_K decode path) | [model-engine-env.md](../model-engine-env.md) (`FAK_Q4K`) |
| **AWQ 4-bit** (activation-aware, symmetric, zero-point 8; safetensors only) | ~0.5625 | Implemented (`model.LoadAWQ`); CUDA kernel near-Q8 throughput, CPU scalar reference; oracle threshold cosine ≥0.95 | [awq-quantization.md](../explainers/awq-quantization.md) |
| **GPTQ 4/8-bit** (AutoGPTQ/GPTQModel `qweight`/`qzeros`/`scales`, optional `g_idx`) | ~0.5 / ~1.0 plus scales | Implemented for CPU-resident in-kernel sessions (`model.LoadGPTQ`, opt-in `Session.GPTQ`); loader supports single-file and sharded safetensors and routes Llama/Mistral-shaped matmul weights through resident GPTQ GEMV. No native packed GPTQ CUDA throughput claim is made here. | `internal/model/gptq.go`; `go test ./internal/model -run TestGPTQ` |

Hardware coverage for these paths (Metal, Vulkan, CUDA Ada and Ampere, the CPU SIMD
tiers) is in the [Hardware matrix](../HARDWARE-MATRIX.md). The `FAK_*` knobs that pick a
load format, residency budget, and SIMD tier are in
[model-engine-env.md](../model-engine-env.md).

---

## How to check what a build serves

The served model id and engine are reported by the gateway, so you never have to guess
what a running build is fronting or running:

- `GET /v1/models` advertises the served model id.
- `GET /healthz` returns `{"ok":true,"model":"…","engine":"…"}` — the `engine` field is
  `inkernel` when the in-kernel reference engine is selected.
- `fak serve --model <id>` (and `--engine inkernel` / `--base-url`) set what the gateway
  serves; the [CLI reference](../cli-reference.md) lists the flags. For inbound tool
  results and the MCP wire shape, see the [MCP tool-result wire](../mcp-tool-result.md).

If a model is not in the Layer 2 table, you are almost certainly on Layer 1 — front it
through the gateway over the OpenAI-compatible wire and it works unchanged. See the
[FAQ](../FAQ.md) for the difference between fronting a model and running one in-kernel.

## Related: the supported-things pages

- [What fak supports (hub)](README.md) — the index of every "supported" page
- [Features](features.md) — every capability with its shipped / simulated / stub status
- [Clouds & hosted providers](clouds.md) — Anthropic, OpenAI, Gemini, xAI, Bedrock, Vertex, Azure, OpenRouter, Together, Groq, Fireworks
- [APIs, wires & MCP](apis-and-protocols.md) — OpenAI Chat/Responses, Anthropic Messages, Gemini, xAI, MCP, fak-native endpoints
- [Agent harnesses & frameworks](agent-harnesses.md) — Claude Code, Cursor, Codex, Aider, Cline, Roo, LangChain, LlamaIndex, CrewAI, …
- [Serving engines](engines.md) — Ollama, vLLM, SGLang, llm-d, llama.cpp, LM Studio, and the in-kernel reference engine

## Reference (the witnessed sources behind this page)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 44 sourced harnesses / frameworks / backends / protocols, each with the exact repoint key
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every capability with one machine-checked tag (shipped / simulated / stub)
- [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) · [CLI reference](../cli-reference.md) · [Hardware matrix](../HARDWARE-MATRIX.md) · [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt)
