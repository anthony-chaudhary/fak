---
title: "Compatibility matrix — what speaks a wire fak can sit on"
description: "A sourced reference of 44 agent harnesses, frameworks, model backends, and interop protocols, each with the wire it speaks, whether it supports a custom base URL, and the exact key you set to repoint it at fak. fak is the gateway; if your tool can set a base URL, it already works."
---

# Compatibility matrix

`fak serve` adjudicates over the wires your stack already speaks — OpenAI Chat
Completions, Anthropic Messages, and MCP. So the practical question for any tool is
narrow: **does it let you repoint its base URL?** If yes, the gate drops in front with no
code change. This page answers that question for 44 surveyed targets, with the exact key
you set and a link to the docs that prove it.

It's a reference, not a tutorial. For the copy-paste recipe, start at the
[integration index](README.md); for a specific harness, see
[Claude Code](claude.md), [Cursor](cursor.md), or [OpenAI Codex](openai-codex.md). The
universal "set the base URL" pattern those build on is in the
[index's universal recipe](README.md#dont-see-your-framework-the-universal-recipe).

**How to read a row.** *Speaks* is the wire(s) the tool talks — match it to one `fak`
exposes. *Custom base URL* is whether you can point that wire somewhere other than the
vendor default (**Yes** / **Partial** / **No**). *How you repoint it* is the literal env
var, constructor arg, or config field. A **Partial** or **No** means the repoint is
templated, indirect, or undocumented — the [caveats](#caveats-worth-knowing) below say
exactly how.

> Surveyed 2026-06-27 across 45 targets (11 harnesses, 14 frameworks, 13 backends, 7
> protocols). Each row carries a source link; 38 of 45 are high-confidence, the rest
> flagged in the caveats. Wires and config keys drift — when a row looks stale, the source
> link is the ground truth, not this table.

---

### Coding agents & harnesses

Interactive coding agents and CLIs. Almost all let you set a base URL, so the gate drops in front of whichever model serves them.

| Target | Speaks | Custom base URL | How you repoint it |
|---|---|---|---|
| [Aider](https://aider.chat/docs/llms/openai-compat.html) | OpenAI Chat Completions (and others via LiteLLM); also speaks Anthropic Messages for Claude models | Yes | OPENAI_API_BASE env var (or AIDER_OPENAI_API_BASE), CLI flag --openai-api-base, or openai-api-base: in ~/.aider.conf.yml / .env |
| [Cline (VS Code)](https://docs.cline.bot/provider-config/openai-compatible) | OpenAI Chat Completions (OpenAI Compatible provider) and Anthropic Messages (Anthropic provider) | Yes | UI provider settings (gear icon): select 'OpenAI Compatible' provider and fill the 'Base URL' field; for the Anthropic provider check 'Use custom base URL' and enter the URL. Configured via the extension UI, not an env var. |
| [Roo Code](https://roocodeinc.github.io/Roo-Code/providers/openai-compatible) | OpenAI Chat Completions with OpenAI native tool-calling schema (OpenAI Compatible provider); also supports an Anthropic provider | Yes | UI provider settings panel: select 'OpenAI Compatible' as API Provider and enter the 'Base URL' field (plus API Key, Model). Configured via the VS Code extension UI. |
| [Continue.dev](https://docs.continue.dev/customize/model-providers/top-level/openai) | OpenAI Chat Completions (provider: openai); also supports an anthropic provider for Claude (Anthropic Messages) | Yes | apiBase field in ~/.continue/config.yaml (provider: openai, apiBase: http://my-endpoint/v1); also supported in deprecated config.json as "apiBase". |
| [Kilo Code](https://kilo.ai/docs/ai-providers/openai-compatible) | OpenAI Chat Completions (OpenAI Compatible provider); VS Code extension in the Roo/Cline lineage | Yes | UI provider settings panel: select 'OpenAI Compatible' as API Provider and enter the 'Base URL' field (accepts https://api.provider.com/v1 or a full /chat/completions URL), plus API Key and Model ID; optional custom HTTP headers. |
| [Goose (Block)](https://github.com/block/goose/blob/main/documentation/docs/getting-started/providers.md) | OpenAI Chat Completions and Anthropic Messages (plus Bedrock/Vertex/OpenRouter/Databricks/Ollama/LiteLLM); pluggable provider layer | Yes | OPENAI_HOST (OpenAI-compatible host; default https://api.openai.com), OPENAI_BASE_PATH (default v1/chat/completions), ANTHROPIC_HOST for Anthropic-compatible; or a custom provider in ~/.config/goose/config.yaml / custom_providers with base_url |
| [Zed editor (AI/agentic)](https://zed.dev/docs/ai/use-api-access) | OpenAI Chat Completions (native + openai_compatible) and Anthropic Messages (native providers) | Yes | settings.json: language_models.openai_compatible.<ProviderName>.api_url (with available_models[]); API key via <PROVIDER_ID>_API_KEY env (upper snake case) |
| [Windsurf (Codeium / Devin Desktop)](https://docs.devin.ai/desktop/chat/models) | native/proprietary (requests routed through Codeium/Cognition backend to OpenAI and Anthropic flagship models) | No | — |
| [Gemini CLI (Google)](https://github.com/google-gemini/gemini-cli) | Gemini (native Generative Language API via google/genai SDK) | Partial | GOOGLE_GEMINI_BASE_URL env var (consumed by the underlying google/genai SDK); official docs do not document this var, and it has known sandbox-propagation bugs (issue #2168) |
| [OpenHands (formerly OpenDevin)](https://docs.openhands.dev/openhands/usage/llms/llms) | Whatever LiteLLM normalizes to (OpenAI Chat Completions, Anthropic Messages, etc.); LiteLLM is the abstraction layer | Yes | config.toml [llm] base_url (with optional custom_llm_provider, model, api_key); env-var overrides LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (via openhands --override-with-envs), or the Advanced > Base URL field in the UI |
| [Qwen Code](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/) | OpenAI Chat Completions (official OpenAI Node.js SDK; endpoint must accept OpenAI-format requests) | Yes | OPENAI_BASE_URL env var (with OPENAI_API_KEY, OPENAI_MODEL); or ~/.qwen/settings.json modelProviders.openai[].baseUrl; or CLI --openai-base-url / --openaiBaseUrl |

### Agent frameworks & SDKs

Libraries you build agents with. Each repoints its OpenAI-compatible client at the gate; some also speak Anthropic or Gemini natively, which `fak serve` can front too.

| Target | Speaks | Custom base URL | How you repoint it |
|---|---|---|---|
| [LangChain (ChatOpenAI, langchain_openai)](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI) | OpenAI Chat Completions | Yes | ChatOpenAI(base_url=...); falls back to env OPENAI_API_BASE, then OPENAI_BASE_URL |
| [LangGraph](https://docs.langchain.com/oss/python/integrations/chat/openai) | OpenAI Chat Completions (via underlying LangChain chat model) | Yes | Set on the underlying LangChain model, e.g. ChatOpenAI(base_url=...) / env OPENAI_API_BASE; LangGraph has no LLM client of its own |
| [LlamaIndex (OpenAI, llama_index.llms.openai)](https://developers.llamaindex.ai/python/framework-api-reference/llms/openai/) | OpenAI Chat Completions | Yes | OpenAI(api_base=...) constructor arg (note: api_base, not base_url); env OPENAI_API_BASE |
| [CrewAI (LLM class)](https://docs.crewai.com/en/learn/llm-connections) | OpenAI Chat Completions (routed through LiteLLM) | Yes | LLM(model=..., base_url=...) constructor arg; env OPENAI_API_BASE (model via OPENAI_MODEL_NAME) |
| [AutoGen / AG2 (OpenAIChatCompletionClient, autogen_ext.models.openai)](https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.models.openai.html) | OpenAI Chat Completions | Yes | OpenAIChatCompletionClient(model=..., base_url=..., api_key=...) constructor arg (base_url required if model not hosted on OpenAI) |
| [OpenAI Agents SDK (Python)](https://openai.github.io/openai-agents-python/models/) | OpenAI Responses API (default) / OpenAI Chat Completions (via OpenAIChatCompletionsModel) | Yes | set_default_openai_client(AsyncOpenAI(base_url=..., api_key=...)); or OPENAI_BASE_URL env var; or OpenAIChatCompletionsModel(openai_client=AsyncOpenAI(base_url=...)); or MultiProvider(openai_base_url=...) |
| [Pydantic AI](https://ai.pydantic.dev/api/models/openai/) | OpenAI Chat Completions (OpenAIChatModel) / OpenAI Responses; also native Anthropic, Gemini, etc. | Yes | OpenAIProvider(base_url='https://...', api_key=...) passed to OpenAIChatModel(provider=...); or OPENAI_BASE_URL / OPENAI_API_KEY env vars; or OpenAIProvider(openai_client=AsyncOpenAI(base_url=...)) |
| [HuggingFace smolagents](https://huggingface.co/docs/smolagents/en/reference/models) | OpenAI Chat Completions (OpenAIServerModel); also LiteLLMModel, InferenceClientModel, TransformersModel | Yes | OpenAIServerModel(model_id=..., api_base='https://.../v1', api_key=...); extra client params via client_kwargs={...} |
| [Google ADK (Agent Development Kit)](https://google.github.io/adk-docs/agents/models/litellm/) | Gemini / google-genai natively; OpenAI Chat Completions and others via the LiteLlm wrapper | Yes | Use LiteLlm(model='openai/<name>', api_base='https://.../v1', api_key=...) as the LlmAgent model; the api_base/api_key/etc. are passed through to LiteLLM |
| [AWS Strands Agents](https://strandsagents.com/docs/user-guide/concepts/model-providers/openai/) | Amazon Bedrock (Converse) natively; OpenAI Chat Completions via OpenAIModel; LiteLLM via LiteLLMModel | Yes | OpenAIModel(client_args={'api_key': ..., 'base_url': '<URL>'}, model_id=...) in Python; TypeScript new OpenAIModel({ clientConfig: { baseURL: '<URL>' }, ... }) |
| [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.openaichatcompletion?view=semantic-kernel-python) | OpenAI Chat Completions / Azure OpenAI; native connectors for Anthropic, Gemini, etc. | Partial | Python: OpenAIChatCompletion(ai_model_id=..., async_client=openai.AsyncOpenAI(base_url='...', api_key=...)). .NET: AddOpenAIChatCompletion(..., endpoint: new Uri('...')) / OpenAIClientOptions Endpoint |
| [Vercel AI SDK](https://ai-sdk.dev/providers/ai-sdk-providers/openai) | Provider-abstracted; @ai-sdk/openai speaks OpenAI; @ai-sdk/openai-compatible for arbitrary OpenAI-compatible servers; native @ai-sdk/anthropic, @ai-sdk/google, etc. | Yes | createOpenAI({ baseURL: 'https://.../v1', apiKey: ... }) from @ai-sdk/openai; or createOpenAICompatible({ name, baseURL, apiKey }) from @ai-sdk/openai-compatible |
| [Mastra (TypeScript)](https://mastra.ai/models/gateways/custom-gateways) | Built on Vercel AI SDK; OpenAI / OpenAI-compatible plus its own model-router gateways; native Anthropic, Google, etc. | Yes | createOpenAI({ apiKey: ..., baseURL: process.env.OPENAI_BASE_URL }) or createOpenAICompatible({ name, apiKey, baseURL }) passed as the agent model; or a MastraModelGateway subclass returning createOpenAICompatible({ baseURL }) from resolveLanguageModel |
| [DSPy](https://dspy.ai/learn/programming/language_models/) | LiteLLM-backed; OpenAI Chat/Text Completions via 'openai/<model>'; any LiteLLM-supported provider/wire | Yes | dspy.LM('openai/<model>', api_base='https://.../v1', api_key=..., model_type='chat'), then dspy.configure(lm=...) |

### Model backends & gateways

What actually serves the tokens. `fak serve --base-url <here>` puts the gate in front of the engine, then your agent points at `fak` instead of the engine.

| Target | Speaks | Custom base URL | How you repoint it |
|---|---|---|---|
| [Ollama](https://docs.ollama.com/api/openai-compatibility) | OpenAI Chat Completions (plus its own native /api/* REST) | Yes | OpenAI client base_url='http://localhost:11434/v1/' (host/port configurable via OLLAMA_HOST); from fak's side --base-url http://<host>:11434/v1 |
| [vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) | OpenAI Chat Completions / Completions / Embeddings | Yes | Server launched with `vllm serve`; client points at base_url='http://localhost:8000/v1' (host/port set by --host/--port). From fak: --base-url http://<host>:8000/v1 |
| [SGLang](https://docs.sglang.ai/backend/openai_api_completions.html) | OpenAI Chat Completions / Completions / Embeddings (plus SGLang-native extensions) | Yes | Launched via `python3 -m sglang.launch_server ... --host 0.0.0.0 --port 30000`; client base_url='http://<host>:30000/v1'. From fak: --base-url http://<host>:30000/v1 |
| [llama.cpp (llama-server)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | OpenAI Chat Completions (plus llama.cpp-native /completion, /props, etc.) | Yes | `llama-server -m model.gguf --host 0.0.0.0 --port 8080`; client base_url='http://localhost:8080/v1'. From fak: --base-url http://<host>:8080/v1 |
| [LM Studio](https://lmstudio.ai/docs/developer/openai-compat) | OpenAI Chat Completions / Completions / Embeddings / Models (also OpenAI Responses API in recent builds) | Yes | Start the local server in the Developer tab; client base_url='http://localhost:1234/v1' (port configurable in the app). From fak: --base-url http://<host>:1234/v1 |
| [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html) | Native InvokeModel + Converse (AWS SigV4 over bedrock-runtime); ALSO an OpenAI-compatible /openai/v1 Chat Completions surface | Partial | OpenAI SDK base_url='https://bedrock-runtime.<region>.amazonaws.com/openai/v1' with a Bedrock API key; native path uses the AWS SDK (region/credentials, not a free base URL) |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/overview) | Native (Gemini predict/generateContent; Anthropic Messages via rawPredict for Claude); ALSO OpenAI-compatible Chat Completions at .../endpoints/openapi/chat/completions | Partial | OpenAI SDK base_url='https://<location>-aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/endpoints/openapi' + GCP OAuth token as api_key; Claude uses .../publishers/anthropic/models/<model>:rawPredict |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/reference) | OpenAI Chat Completions / Completions / Embeddings (Azure dialect) | Yes | Endpoint 'https://<resource>.openai.azure.com'; path /openai/deployments/<deployment>/chat/completions?api-version=YYYY-MM-DD (newer v1: '<endpoint>/openai/v1'). Use AzureOpenAI client or set azure_endpoint |
| [OpenRouter](https://openrouter.ai/docs/quickstart) | OpenAI Chat Completions (with OpenRouter extensions); also an OpenAI Responses API beta | Yes | OpenAI SDK base_url='https://openrouter.ai/api/v1' + OpenRouter API key. From fak: --base-url https://openrouter.ai/api/v1 |
| [Together AI](https://docs.together.ai/docs/openai-api-compatibility) | OpenAI Chat Completions / Completions / Embeddings / Images | Yes | OpenAI SDK base_url='https://api.together.xyz/v1' (also documented as https://api.together.ai/v1) + Together API key. From fak: --base-url https://api.together.xyz/v1 |
| [Groq](https://console.groq.com/docs/openai) | OpenAI Chat Completions (plus a Responses API) | Yes | OpenAI SDK base_url='https://api.groq.com/openai/v1' + GROQ_API_KEY. From fak: --base-url https://api.groq.com/openai/v1 |
| [Fireworks AI](https://docs.fireworks.ai/tools-sdks/openai-compatibility) | OpenAI Chat Completions / Completions (plus an OpenAI Responses API beta) | Yes | OpenAI SDK base_url='https://api.fireworks.ai/inference/v1' + Fireworks API key. From fak: --base-url https://api.fireworks.ai/inference/v1 |
| [AgentGateway](https://github.com/agentgateway/agentgateway) | OpenAI Chat Completions (unified LLM gateway), MCP, A2A (Linux Foundation project) | Yes | OpenAI SDK base_url='https://<agentgateway-host>:<port>/v1' or via AgentGateway's OpenAI-compatible endpoint; also serves MCP and A2A protocols. A Linux Foundation project (donated 2026) providing connectivity for agent-to-LLM, agent-to-tool, and agent-to-agent communication. |

### Wire & interop protocols

The wires themselves. Three are runtime boundaries a gateway can sit on (MCP-over-HTTP, A2A, the OpenAI Responses API); the rest are stdio-only or static discovery documents with nothing live to adjudicate — noted honestly below.

| Target | Speaks | Custom base URL | How you repoint it |
|---|---|---|---|
| [MCP (Model Context Protocol)](https://modelcontextprotocol.io/specification/2025-11-25) | JSON-RPC 2.0 over stdio or Streamable HTTP (HTTP POST + SSE) | Yes | Client config points at a server URL/command (e.g. mcpServers entry with a "url" for HTTP transport, or "command"/"args" for stdio, in the host's config such as claude_desktop_config.json / .mcp.json) |
| [A2A (Agent2Agent)](https://a2a-protocol.org/latest/) | JSON-RPC 2.0, gRPC, or HTTP+JSON/REST; SSE for streaming | Yes | AgentCard JSON exposes the agent's service endpoint in its "url" field; clients discover/address an agent by that URL (typically published at /.well-known/agent-card.json). v1.0 production standard under Linux Foundation (April 2026) |
| [AG-UI (Agent-User Interaction Protocol)](https://docs.ag-ui.com/concepts/architecture) | Transport-agnostic; default is HTTP POST + Server-Sent Events (also WebSocket, webhook, binary variant) | Yes | Frontend client (e.g. HttpAgent) is constructed with a target agent endpoint URL; it POSTs RunAgentInput and consumes a stream of typed BaseEvents |
| [ACP (Agent Communication Protocol / BeeAI)](https://agentcommunicationprotocol.dev/introduction/welcome) | REST over HTTP (explicitly not JSON-RPC); streaming + await/resume sessions | Yes | REST endpoints; an ACP server hosts one or more agents behind a single HTTP base URL and routes by agent name (OpenAPI-described, e.g. /agents, /runs) |
| [ANP (Agent Network Protocol)](https://agentnetworkprotocol.com/en/specs/07-anp-agent-description-protocol-specification/) | JSON-LD messages over HTTP(S); W3C DID (did:wba) for identity | Yes | Each agent is identified by a DID whose document is hosted at an HTTPS URL; the JSON-LD Agent Description document lists service endpoints |
| [llms.txt](https://llmstxt.org/) | none (static Markdown file served over HTTP at a fixed path) | No | — |
| [OpenAI Responses API](https://github.com/openai/openai-python) | HTTP+JSON at POST /v1/responses; SSE for streaming (typed response.* events) | Yes | OpenAI SDK base_url client parameter, or the OPENAI_BASE_URL environment variable (default https://api.openai.com/v1); Responses API is now the primary OpenAI Python API (June 2026). A gateway exposes an OpenAI-compatible /v1/responses and clients repoint here. |

### Caveats worth knowing

Where a row says **Partial** or **No**, or the repoint has a sharp edge, here's the detail:

- **Windsurf (Codeium / Devin Desktop)** — Official docs (docs.windsurf.com now redirects to docs.devin.ai) describe model access through the Codeium/Cognition backend and do not document any user-settable OpenAI/Anthropic-compatible base URL; third-party proxies/extensions exist but are not first-party. No documented config key found
- **Gemini CLI (Google)** — GOOGLE_GEMINI_BASE_URL repoints the Gemini-protocol endpoint (e.g. a Gemini-compatible proxy), not an arbitrary OpenAI/Anthropic wire; the dedicated GEMINI_BASE_URL PR #2899 was closed unmerged and the var is undocumented in the official CLI config (set in the SDK), and it has known sandbox-propagation bugs (issue #2168)
- **Microsoft Semantic Kernel** — Python has no first-class base_url arg on OpenAIChatCompletion — you must inject a pre-built AsyncOpenAI(base_url=...) via async_client. .NET added an endpoint arg later; older versions could not set a custom OpenAI endpoint (issues #2145/#4152/#5353).
- **Mastra (TypeScript)** — Custom-base-URL support is inherited from the AI SDK providers / Mastra's gateway abstraction rather than a single Mastra-native field; the model-router string form (e.g. 'private/...') requires defining a custom gateway.
- **AWS Bedrock** — Base URL is region-templated, not arbitrary. OpenAI-compat surface is newer/narrower than native Converse/InvokeModel; native path needs SigV4 or a Bedrock bearer key, not a plain endpoint swap.
- **Google Vertex AI** — Base URL is fully templated by region+project, not user-free; auth is a short-lived Google OAuth access token, not a static key. OpenAI-compat route is for Gemini/MaaS models; Claude on Vertex is the Anthropic Messages wire, not OpenAI.
- **AG-UI (Agent-User Interaction Protocol)** — Standardizes the agent<->frontend/UI boundary, not agent-to-agent or agent-to-tool. No formal spec version number published; framed as an established event schema (16 event types) rather than a numbered standard. MIT-licensed, community/CopilotKit-led.
- **ACP (Agent Communication Protocol / BeeAI)** — Pre-alpha/experimental — the docs warn of ongoing breaking changes to protocol/transport/APIs and publish no stable version number. Governed under the Linux Foundation with IBM/BeeAI as reference impl. Source reports conflict on transport (some say JSON-RPC); the official site states REST.
- **ANP (Agent Network Protocol)** — Draft specifications (W3C CG white paper / multiple draft sub-specs), no stable version. Built for decentralized, cross-organization agent-to-agent comms with DID-based mutual auth and end-to-end encryption — a man-in-the-middle governance gateway is awkward by design unless it terminates/holds a DID identity itself.
- **llms.txt** — NOT a wire protocol and not a runtime boundary — it is a static discovery/context document (Markdown) served at the well-known path /llms.txt, path-based like robots.txt/sitemap.xml. No request/response, no streaming, no base URL to repoint. Informal community proposal, no formal version. A governance gateway has nothing live to sit on; at most it could rewrite the served file.
---

## Summary

Of the 45 targets, **39 expose a custom base URL outright** and 4 more do so partially —
because the OpenAI-compatible wire has become the field's lingua franca, and `fak serve`
speaks it. The handful that don't (`llms.txt` is a static file; Windsurf routes through a
closed backend) aren't runtime boundaries a gateway can sit on in the first place.

So the rule from the [index](README.md) holds across the whole field: **if your tool can
set a base URL, fak already fronts it** — your agent, your model, your prompts unchanged,
with a default-deny capability floor in the middle.

## Cross-references

- [Integration index](README.md) — which-agent routing and the universal recipe this matrix backs.
- [fak + LiteLLM](litellm.md) · [Routers & gateways](routers.md) — the dedicated guides for the LiteLLM-backed and router rows above (front / behind / route-through topologies).
- [Claude Code](claude.md) · [Cursor](cursor.md) · [OpenAI Codex](openai-codex.md) — the per-harness guides.
- [Agent memory (mem0 / OpenMemory / MCP)](agent-memory.md) — the gate in front of a memory store.
- [CLAIMS.md](../../CLAIMS.md) — fak's scope, claim by claim (it's the governance band, not the token engine).
