---
title: "What fak supports — models, features, clouds, APIs, MCP, harnesses, engines"
description: "The index of fak's supported-things pages: which models, features, clouds and hosted providers, APIs and wires, MCP, agent harnesses and frameworks, and serving engines fak works with — each a dedicated, cross-linked page grounded in the repo and the sourced compatibility matrix."
---

# What fak supports

`fak` is an agent kernel: one Go binary that sits between an AI agent and the tools it
calls. Two facts decide what it supports.

```text
            AI agent (harness / framework)
                        │
                        ▼
        ┌───────────────────────────────────┐
        │     fak — the agent kernel          │
        │  fronts the wires your stack speaks │
        │  (OpenAI · Anthropic · Gemini · MCP │
        │   · xAI); governs, does not generate│
        └───────────────────────────────────┘
                        │
        ┌───────────┬───┴───┬───────────┐
        ▼           ▼       ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌──────┐ ┌──────────────┐
  │ engine  │ │ cloud / │ │ APIs │ │ in-kernel     │
  │ Ollama· │ │ hosted  │ │wires │ │ reference     │
  │ vLLM·   │ │ provider│ │· MCP │ │ engine        │
  │ SGLang· │ │         │ │      │ │ (correctness, │
  │llama.cpp│ │         │ │      │ │  not a server)│
  └─────────┘ └─────────┘ └──────┘ └──────────────┘
   The pages below: Models · Features · Clouds · APIs/MCP ·
   Harnesses · Serving engines — each grounded in the repo
   and the sourced compatibility matrix.
```
*Index map: the kernel fronts the wires, then each page lists one supported category.*

1. **It fronts the wires your stack already speaks** — OpenAI Chat Completions, Anthropic
   Messages, Gemini `generateContent`, and MCP, plus an xAI upstream. Anything that lets
   you set a base URL drops the gate in front with no code change. So the supported set of
   harnesses, clouds, and engines is wide by construction.
2. **It governs, it does not generate.** For production tokens fak fronts an engine
   (Ollama, vLLM, SGLang, llama.cpp, a cloud API). It also ships an in-kernel reference
   engine that runs a model itself, as a correctness reference rather than a fast server.

Each page below is the dedicated list for one category. Every row is grounded in the repo
or in the sourced [compatibility matrix](../integrations/compatibility-matrix.md), and
status follows the witnessed [claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).

## The pages

| Page | What it lists |
|---|---|
| [Models](models.md) | Any model you front through the gateway, plus the architectures the in-kernel engine runs and proves bit-exact (Llama, Qwen2/Qwen3, Gemma, GLM-MoE, GPT-OSS, SmolLM2). |
| [Features](features.md) | Every capability grouped by subsystem with its honest status — shipped, simulated, or stub — mirroring the claims ledger. |
| [Clouds & hosted providers](clouds.md) | Anthropic, OpenAI, Gemini, and xAI as native provider wires, plus AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenRouter, Together, Groq, and Fireworks over the OpenAI-compatible wire. |
| [APIs, wires & MCP](apis-and-protocols.md) | OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Gemini, xAI; MCP over stdio and HTTP; the fak-native endpoints; and the honest interop stance on A2A, AG-UI, ACP, ANP. |
| [Agent harnesses & frameworks](agent-harnesses.md) | Claude Code, Cursor, OpenAI Codex, OpenCode, Aider, Cline, Roo, Goose, Zed, and frameworks like LangChain, LlamaIndex, CrewAI, AutoGen, and the Vercel AI SDK. |
| [Serving engines](engines.md) | The token engines fak fronts — Ollama, vLLM, SGLang, llm-d, llama.cpp, LM Studio — and the in-kernel reference engine. |
| [Silicon backends](silicon-backends.md) | The vendor-neutral backend path for accelerator teams: `compute.Backend`, `Caps`, correctness classes, backend conformance vocabulary, and non-reference gates. |

## Related references (the sourced detail behind these pages)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 47 surveyed harnesses, frameworks, backends, and protocols, each with the wire it speaks, whether it takes a custom base URL, and the exact repoint key, with a source link per row.
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof.
- [Hardware matrix](../HARDWARE-MATRIX.md) — every machine fak has been profiled on: 4 platforms, 2 CPU ISAs, 4 GPU backends (Apple Metal, AMD Vulkan, NVIDIA CUDA Ada + Ampere).
- [Hardware portability via the compute HAL](../explainers/hardware-portability.md) — how accelerator backends bind into fak by registration rather than by forking the forward pass.
- [CLI reference](../cli-reference.md) — every `fak` verb and what it does.
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) · [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) — what is shipped, simulated, or stub, and what is on the critical path.
- [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) — the machine-readable doc map for LLMs and answer engines.
