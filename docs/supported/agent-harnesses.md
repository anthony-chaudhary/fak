---
title: "Agent harnesses and frameworks fak supports"
description: "The coding agents, IDEs, and agent frameworks fak fronts by repointing one base URL: Claude Code, Cursor, OpenAI Codex, OpenCode, Aider, Cline, Roo Code, Kilo Code, Goose, Zed, Continue.dev, Qwen Code, plus frameworks like LangChain, LlamaIndex, CrewAI, AutoGen, the OpenAI Agents SDK, Pydantic AI, and the Vercel AI SDK."
---

# Agent harnesses and frameworks fak supports

`fak serve` speaks the wires your stack already speaks: OpenAI Chat Completions, Anthropic
Messages, and MCP. So any harness that lets you set a base URL drops the gate in front with
no code change. You point the tool at `fak`, the kernel adjudicates every tool call the
agent proposes, and your agent, model, and prompts stay the same.

This page lists the coding agents, IDEs, and frameworks that work this way. Every row is
drawn from the [compatibility matrix](../integrations/compatibility-matrix.md), which
surveyed 44 targets and found that **38 of them expose a custom base URL outright** (4 more
do so partially). The matrix is the master source — it carries the exact repoint key and a
source link for each row, so this page links there rather than duplicating long config
strings.

For the copy-paste recipe and the 60-second offline proof, start at the
[integration index](../integrations/README.md).

---

## First-class guides

These five have a dedicated walkthrough. Each guide names the wire, the repoint key, and a
worked end-to-end setup.

| Harness | Wire | Repoint key | Guide |
|---|---|---|---|
| Claude Code | Anthropic Messages | `ANTHROPIC_BASE_URL` (or `fak guard -- claude`) | [claude.md](../integrations/claude.md) |
| Cursor | MCP, or OpenAI Chat Completions proxy | MCP server entry, or a custom OpenAI-compatible endpoint | [cursor.md](../integrations/cursor.md) |
| OpenAI Codex | OpenAI Chat Completions | `OPENAI_BASE_URL` | [openai-codex.md](../integrations/openai-codex.md) |
| OpenCode | OpenAI Chat Completions | `OPENAI_BASE_URL` (or `fak guard --provider openai -- opencode`) | [claude.md#opencode](../integrations/claude.md#opencode) |
| Hermes Agent (NousResearch) | OpenAI Chat Completions | `OPENAI_BASE_URL` / `~/.hermes/config.yaml` `model.base_url` (or `fak guard -- hermes`) | [hermes.md](../integrations/hermes.md) |

The one-command front door for Claude Code is `fak guard -- claude`: it starts the gateway
in-process, injects the base URL into the child only, and proxies the real Anthropic API in
passthrough. OpenCode fronts the same way over `--provider openai`. Both are covered in the
[Claude Code guide](../integrations/claude.md).

---

## Coding agents and CLIs

Interactive coding agents and CLIs, sourced row-by-row from the matrix. Each speaks a wire
`fak serve` exposes, so the gate sits in front of whichever model serves the tool. The exact
env var, flag, or config field for every row is in the
[compatibility matrix](../integrations/compatibility-matrix.md).

| Harness | Wire | Custom base URL |
|---|---|---|
| Aider | OpenAI Chat Completions (and others via LiteLLM); Anthropic Messages for Claude models | Yes |
| Hermes Agent (NousResearch) | OpenAI Chat Completions (custom provider; OpenAI tools[] function-calling) | Yes |
| Cline (VS Code) | OpenAI Chat Completions and Anthropic Messages | Yes |
| Roo Code | OpenAI Chat Completions (OpenAI-native tool-calling); also an Anthropic provider | Yes |
| Kilo Code | OpenAI Chat Completions (OpenAI Compatible provider) | Yes |
| Goose (Block) | OpenAI Chat Completions and Anthropic Messages (pluggable provider layer) | Yes |
| Zed editor | OpenAI Chat Completions (native + `openai_compatible`) and Anthropic Messages | Yes |
| Continue.dev | OpenAI Chat Completions; also an `anthropic` provider for Claude | Yes |
| Qwen Code | OpenAI Chat Completions (official OpenAI Node.js SDK) | Yes |
| Gemini CLI (Google) | Gemini (native Generative Language API) | Partial |
| OpenHands | Whatever LiteLLM normalizes to (OpenAI, Anthropic, etc.) | Yes |
| Windsurf (Codeium / Devin Desktop) | Native/proprietary backend (Codeium / Cognition) | No |

Two rows are honestly less than a clean repoint, exactly as the matrix grades them:

- **Gemini CLI — Partial.** `GOOGLE_GEMINI_BASE_URL` repoints the Gemini-protocol endpoint,
  not an arbitrary OpenAI/Anthropic wire. The dedicated base-URL PR was closed unmerged and
  the var is undocumented in the official CLI config (it is read by the underlying SDK). See
  the matrix caveats for the detail.
- **Windsurf — No first-party path.** The official docs route model access through the
  Codeium / Cognition backend and document no user-settable OpenAI- or Anthropic-compatible
  base URL. Third-party proxies exist but are not first-party, so there is no runtime
  boundary `fak` can sit on through a supported config key.

---

## Agent frameworks and SDKs

Libraries you build agents with. Each repoints its OpenAI-compatible client at the gate;
several also speak Anthropic or Gemini natively, which `fak serve` can front too. The exact
constructor arg, env var, or config field for each is in the
[compatibility matrix](../integrations/compatibility-matrix.md).

| Framework | Wire | Custom base URL |
|---|---|---|
| LangChain (`ChatOpenAI`) | OpenAI Chat Completions | Yes |
| LangGraph | OpenAI Chat Completions (via the underlying LangChain model) | Yes |
| LlamaIndex | OpenAI Chat Completions | Yes |
| CrewAI | OpenAI Chat Completions (routed through LiteLLM) | Yes |
| AutoGen / AG2 | OpenAI Chat Completions | Yes |
| OpenAI Agents SDK (Python) | OpenAI Responses API (default) / OpenAI Chat Completions | Yes |
| Pydantic AI | OpenAI Chat Completions / Responses; also native Anthropic, Gemini | Yes |
| smolagents (HuggingFace) | OpenAI Chat Completions (`OpenAIServerModel`); also LiteLLM, InferenceClient | Yes |
| Google ADK | Gemini natively; OpenAI Chat Completions and others via the `LiteLlm` wrapper | Yes |
| AWS Strands Agents | Bedrock Converse natively; OpenAI Chat Completions via `OpenAIModel`; LiteLLM | Yes |
| Microsoft Semantic Kernel | OpenAI Chat Completions / Azure OpenAI; native Anthropic, Gemini | Partial |
| Vercel AI SDK | Provider-abstracted; OpenAI and OpenAI-compatible; native Anthropic, Google | Yes |
| Mastra (TypeScript) | Built on the Vercel AI SDK; OpenAI / OpenAI-compatible plus its own gateways | Yes |
| DSPy | LiteLLM-backed; OpenAI Chat/Text Completions via `openai/<model>` | Yes |

One row is Partial, as the matrix grades it:

- **Semantic Kernel — Partial.** Python has no first-class `base_url` arg on
  `OpenAIChatCompletion`; you inject a pre-built `AsyncOpenAI(base_url=...)` via the
  `async_client` parameter. .NET added an `endpoint` arg later. See the matrix caveats.

For any framework not in the table, the rule still holds: if it lets you set the model's
base URL, `fak` fronts it via the OpenAI-compatible wire. The
[universal recipe](../integrations/README.md#dont-see-your-framework-the-universal-recipe)
in the integration index is the one-paste pattern, and the
[compatibility matrix](../integrations/compatibility-matrix.md) is the sourced list.

---

## Related: the supported-things pages

- [What fak supports (hub)](README.md) — the index of every "supported" page
- [Models](models.md) — in-kernel architectures + any model you front
- [Features](features.md) — every capability with its shipped / simulated / stub status
- [Clouds & hosted providers](clouds.md) — Anthropic, OpenAI, Gemini, xAI, Bedrock, Vertex, Azure, OpenRouter, Together, Groq, Fireworks
- [APIs, wires & MCP](apis-and-protocols.md) — OpenAI Chat/Responses, Anthropic Messages, Gemini, xAI, MCP, fak-native endpoints
- [Serving engines](engines.md) — Ollama, vLLM, SGLang, llm-d, llama.cpp, LM Studio, and the in-kernel reference engine

## Reference (the witnessed sources behind this page)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 44 sourced harnesses / frameworks / backends / protocols, each with the exact repoint key
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every capability with one machine-checked tag (shipped / simulated / stub)
- [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) · [CLI reference](../cli-reference.md) · [Hardware matrix](../HARDWARE-MATRIX.md) · [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt)
