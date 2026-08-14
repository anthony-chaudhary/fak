---
title: "Clouds and hosted providers fak supports"
description: "The hosted model providers and cloud gateways fak serve sits in front of: Anthropic, OpenAI, Google Gemini, and xAI as native provider wires, plus AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenRouter, Together AI, Groq, and Fireworks AI through the OpenAI-compatible wire."
---

# Clouds and hosted providers fak supports

> **Scope fence — three pages say "cloud" and mean different things.** This page is
> **fronting a hosted model API**: someone else's GPU, someone else's tokens, you point
> `--provider`/`--base-url` at their `/v1`. If instead you **rent a raw GPU box or GPU
> k8s pool** (CoreWeave, Lambda, RunPod, Crusoe, Vast.ai, Nebius) and stand the gateway
> up *on it*, you want [Deploy fak on a rented GPU cloud](../fak/neo-cloud-deploy.md).
> If you *are* the neo-cloud exposing many accelerator backends through one control
> plane, you want the [neo-cloud reference architecture](https://github.com/anthony-chaudhary/fak/blob/main/docs/vendor/neo-cloud-reference-architecture.md).

This page lists the hosted model providers and cloud gateways `fak serve` can sit in front of. `fak serve` is a gateway: it fronts whatever serves your tokens and runs every proposed tool call through the kernel before it reaches the model. So a cloud is "supported" when you can point fak's `--provider` and `--base-url` at it. Two tiers cover the field: native provider wires that fak speaks directly, and any cloud that exposes an OpenAI-compatible endpoint.

## Start here: pick a cloud path and confirm it in one check

If you are an operator, your job is to choose the path for your cloud and confirm fak
fronts it. The choice is one question — does fak speak your cloud's wire natively?

- **Native wire (Tier 1)** — Anthropic, OpenAI, Gemini, or xAI: pass its `--provider`
  value (table below). The accepted values are witnessed in `internal/agent/adapters.go`
  (the `Provider` constants and `ParseProvider`).
- **OpenAI-compatible (Tier 2)** — any other cloud exposing a `/v1` Chat Completions
  endpoint: `--provider openai --base-url <cloud /v1>`. The per-cloud repoint keys are
  witnessed in the [compatibility matrix](../integrations/compatibility-matrix.md).

Either way, start the gateway and confirm the seat is live with one command:

```bash
curl -s http://127.0.0.1:8080/healthz   # -> {"ok":true,"model":"…","engine":"…"}
```

Everything below is the per-provider detail and the exact repoint recipe.

## Tier 1: Native provider wires

These are the `--provider` values `fak serve` and `fak manage` accept. Each value selects a transcript adapter that translates the canonical agent transcript into that provider's request and response wire. The values, wires, and aliases are sourced from `internal/agent/adapters.go` (the `Provider` constants and `ParseProvider`).

| Provider | `--provider` value | Wire | Notes |
|---|---|---|---|
| OpenAI (GPT) | `openai` | OpenAI Chat Completions (`/chat/completions`) | The default when `--provider` is unset. Aliases: `gpt`, `chat-completions`, `openai-compatible`. This is also the wire every Tier 2 cloud below rides. |
| OpenAI Responses | `openai-responses` | OpenAI Responses API (`/responses`) | The item-shaped GPT wire. Aliases: `responses`, `responses-api`. |
| Anthropic (Claude) | `anthropic` | Anthropic Messages API (`/v1/messages`) | Alias: `claude`. Picks `x-api-key` for an `sk-ant-api…` key, or `Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20` for a Claude Pro/Max subscription `sk-ant-oat…` token. |
| Google Gemini | `gemini` | Gemini `generateContent` API | Alias: `google`. Auth via `x-goog-api-key`. Also served to clients as an inbound wire — see [APIs, wires & MCP](apis-and-protocols.md). |
| xAI (Grok) | `xai` | OpenAI-compatible chat completions | Alias: `grok`. Shares the OpenAI chat adapter. |

The native default front door for Claude Code is `fak manage -- claude`, which runs over the `anthropic` wire and uses your logged-in Claude Pro/Max subscription by default, no API key needed. See [Run Claude Code through the fak gateway](../integrations/claude.md).

## Tier 2: Cloud gateways over the OpenAI-compatible wire

Each cloud below serves tokens behind an OpenAI Chat Completions endpoint. You front it with `fak serve --provider openai --base-url <cloud /v1>`, then your agent points at fak instead of the cloud. Every row here is sourced from the "Model backends & gateways" section of the [compatibility matrix](../integrations/compatibility-matrix.md); follow the linked row for the exact upstream base URL and key, which this page does not restate.

| Cloud | How fak fronts it | Custom base URL | Caveat |
|---|---|---|---|
| AWS Bedrock | `--provider openai` at the OpenAI-compatible `/openai/v1` surface, or front the native Converse/InvokeModel path | Partial | Base URL is region-templated, not arbitrary; the native path needs AWS SigV4 or a Bedrock bearer key, not a plain endpoint swap. See the [matrix row](../integrations/compatibility-matrix.md) and its [caveat](../integrations/compatibility-matrix.md#caveats-worth-knowing). |
| Google Vertex AI | `--provider openai` at the OpenAI-compatible Chat Completions route (Gemini / MaaS models) | Partial | Base URL is fully templated by region and project; auth is a short-lived Google OAuth access token, not a static key. Claude on Vertex is the Anthropic Messages wire, not OpenAI. See the [matrix row](../integrations/compatibility-matrix.md) and its [caveat](../integrations/compatibility-matrix.md#caveats-worth-knowing). |
| Azure OpenAI | `--provider openai` at the Azure endpoint (newer `<endpoint>/openai/v1`) | Yes | Azure dialect; deployment-named paths with an `api-version` query. See the [matrix row](../integrations/compatibility-matrix.md). |
| OpenRouter | `--provider openai --base-url https://openrouter.ai/api/v1` | Yes | OpenAI Chat Completions with OpenRouter extensions. See the [matrix row](../integrations/compatibility-matrix.md). |
| Together AI | `--provider openai --base-url https://api.together.xyz/v1` | Yes | OpenAI-compatible chat / completions / embeddings. See the [matrix row](../integrations/compatibility-matrix.md). |
| Groq | `--provider openai --base-url https://api.groq.com/openai/v1` | Yes | OpenAI Chat Completions. See the [matrix row](../integrations/compatibility-matrix.md). |
| Fireworks AI | `--provider openai --base-url https://api.fireworks.ai/inference/v1` | Yes | OpenAI Chat Completions. See the [matrix row](../integrations/compatibility-matrix.md). |
| DeepSeek (V4) | `--provider openai --base-url https://api.deepseek.com` (or `--provider anthropic --base-url https://api.deepseek.com/anthropic`) | Yes | Reasoning model: `thinking` is default-on and `reasoning_effort` takes only `high`/`max`. See the dedicated [DeepSeek V4 page](deepseek.md) for all three routes and the alias-retirement warning. |

Bedrock and Vertex are marked **Partial** because the repoint is templated and the auth is not a plain static key, exactly as the matrix caveats state. The other five expose a custom base URL outright.

If your cloud is not in this table but exposes an OpenAI Chat Completions endpoint, fak fronts it the same way over `--provider openai`. The matrix surveys 47 targets and the rule holds across the field: if your tool or cloud can set a base URL, fak already fronts it.

## How you point fak at a cloud

The pattern mirrors the "Cloud providers" recipe in the [Claude Code guide](../integrations/claude.md): pick the provider wire, set the base URL to the cloud's endpoint, and read the key from an environment variable so the secret is never a command-line argument.

```bash
# A native provider wire (OpenAI here):
fak serve \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4

# Any OpenAI-compatible cloud — same flags, just a different /v1 base URL and key env:
fak serve \
  --provider openai \
  --base-url https://api.groq.com/openai/v1 \
  --api-key-env GROQ_API_KEY \
  --model <cloud-model-id>
```

For a network-facing gateway, add `--require-key-env` for bearer-key auth and tune the timeouts; see [serve config](../serve-config.md).

For a cloud, the win is the governance band in front of the API, not throughput. fak does not make a hosted provider faster. It puts a default-deny capability floor between your agent and the cloud, adjudicates every proposed tool call (allow, deny, repair, quarantine), and can write a hash-chained audit trail of each decision. The KV poison-evictor is a no-op on a proxy seat by design, because the model lives upstream and there is no local KV prefix to drop. See [Run Claude Code through the fak gateway](../integrations/claude.md) for the limits on a proxy seat.

## Related: the supported-things pages

- [What fak supports (hub)](README.md) — the index of every "supported" page
- [Models](models.md) — in-kernel architectures + any model you front
- [Features](features.md) — every capability with its shipped / simulated / stub status
- [APIs, wires & MCP](apis-and-protocols.md) — OpenAI Chat/Responses, Anthropic Messages, Gemini, xAI, MCP, fak-native endpoints
- [Agent harnesses & frameworks](agent-harnesses.md) — Claude Code, Cursor, Codex, Aider, Cline, Roo, LangChain, LlamaIndex, CrewAI, …
- [Serving engines](engines.md) — Ollama, vLLM, SGLang, llm-d, llama.cpp, LM Studio, and the in-kernel reference engine

## Reference (the witnessed sources behind this page)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 44 sourced harnesses / frameworks / backends / protocols, each with the exact repoint key
- [fak + LiteLLM](../integrations/litellm.md) · [Routers & gateways](../integrations/routers.md) — front / behind / route-through topologies for LiteLLM, OpenRouter, Portkey, and the rest of Tier 2
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every capability with one machine-checked tag (shipped / simulated / stub)
- [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) · [CLI reference](../cli-reference.md) · [Hardware matrix](../HARDWARE-MATRIX.md) · [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt)
