---
title: "APIs, wires, and MCP that fak supports"
description: "The wire protocols fak speaks: OpenAI Chat Completions, the OpenAI Responses API, Anthropic Messages, Gemini generateContent, and xAI as provider wires; MCP over stdio and HTTP; the fak-native syscall / adjudicate / admit endpoints; plus the honest interop stance on A2A, AG-UI, ACP, and ANP."
---

# APIs, wires, and MCP that fak supports

This page lists the protocols `fak serve` actually speaks. There are three groups:
the model wires it speaks (inbound to clients and upstream to providers), the MCP
and fak-native endpoints it serves, and the wider interop protocols it can or cannot
sit on. Every row here is grounded in the gateway source and the
[compatibility matrix](../integrations/compatibility-matrix.md); where the support is
indirect, the row says so plainly rather than overclaim.

`fak` is the governance and gateway band, not the token engine. It fronts whatever
serves your tokens and adjudicates every tool call at the boundary. The throughput
question belongs to the engine; the wire question belongs here.

---

## 1. Model wires fak speaks

`fak serve` exposes three client wires (the surfaces your agent points at) and selects
an upstream provider wire with `--provider`. A client speaks OpenAI Chat Completions,
Anthropic Messages, or Gemini `generateContent` to reach fak, and fak then proxies on
to OpenAI, Anthropic, Gemini, or xAI. The OpenAI Responses wire is upstream-only (no
matching client endpoint yet), and xAI is reached through fak's OpenAI surface because
it speaks the same shape.

The provider names and their wire shapes are defined in
[`internal/agent/adapters.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/agent/adapters.go);
the client surfaces are in the gateway route table
([Gateway API reference](../fak/api-reference.md)).

| Wire | fak surface | `--provider` | Status |
|---|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` (client + upstream) | `openai` | Shipped. The adjudicating chat proxy; the same wire fronts any OpenAI-compatible engine (Ollama, vLLM, SGLang, llama.cpp, LM Studio). |
| OpenAI Responses API | upstream only (`POST /responses`) | `openai-responses` | Upstream provider wire. fak proxies *to* a Responses upstream but exposes Chat Completions and Messages to clients. A Responses-default client connects by selecting the chat-completions model. |
| Anthropic Messages | `POST /v1/messages` (client + upstream) | `anthropic` | Shipped. The Claude-Code-facing proxy; `fak guard -- claude` is the one-command front door. Live token streaming on this wire is synthesized from a buffered, already-adjudicated turn (the live-streaming rung is OpenAI-wire today). |
| Gemini `generateContent` | `POST /v1beta/models/<model>:generateContent` (and `:streamGenerateContent`) (client + upstream) | `gemini` | Shipped (#567, [`internal/gateway/gemini.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/gateway/gemini.go)). The Gemini-CLI / google-genai-facing proxy: repoint a Gemini-native client's base URL from `generativelanguage.googleapis.com` at the fak host, and every proposed `functionCall` is adjudicated before the client sees it. A `:streamGenerateContent` request synthesizes a well-formed Gemini SSE sequence from the buffered, already-adjudicated turn (the same posture as the Anthropic wire). |
| xAI (Grok) | upstream only (OpenAI-compatible `/chat/completions`) | `xai` | Upstream provider wire. xAI uses the OpenAI-compatible chat shape, so the same adapter serves it; clients reach it through fak's OpenAI surface. |

`--provider` aliases match the model family, so `gpt` / `chat-completions` /
`openai-compatible` map to `openai`, `claude` maps to `anthropic`, `google` maps to
`gemini`, and `grok` maps to `xai`
([`ParseProvider`](https://github.com/anthony-chaudhary/fak/blob/main/internal/agent/adapters.go)).
Authentication is wire-correct per provider: a bearer token for the OpenAI wire, an
`x-api-key` (or an `sk-ant-oat` subscription token sent as a bearer with the OAuth
beta flag) for Anthropic, and `x-goog-api-key` for Gemini.

The OpenAI surface also carries two deterministic, self-contained helpers documented
in the [Gateway API reference](../fak/api-reference.md): `POST /v1/embeddings` (a
feature-hash projection, not a learned model) and `POST /v1/moderations` (a lexical
baseline, not a learned safety model). Both are honest baselines for tests and cache
keys, named as such.

Any tool not listed above that lets you set a base URL connects through one of these
wires. That covers most of the field. Rather than restate the list here, see the
[compatibility matrix](../integrations/compatibility-matrix.md), which sources 44
harnesses, frameworks, backends, and protocols, each with the exact repoint key.

---

## 2. MCP and the fak-native endpoints

`fak serve` is also an MCP server. It speaks JSON-RPC 2.0 over two transports:
stdio (`fak serve --stdio`, newline-delimited frames, no listener and no auth
surface) and HTTP (`POST /mcp`, one JSON-RPC message per request). The same dispatch
backs both. MCP clients (Claude Code, Cursor, or any MCP client) use this to ask the
kernel about a call before running it, run a call through the kernel, or screen a
result they ran themselves.

A refusal is a value, not an error. A DENY or QUARANTINE rides inside the tool result
with `isError: false`; a JSON-RPC error is reserved for protocol faults like a bad
frame or an unknown method. The result envelope and the `SyscallResponse` fields are
specified in the [MCP tool-result wire](../mcp-tool-result.md).

### Served routes

The principal served routes, from the Claude Code integration guide and the
[Gateway API reference](../fak/api-reference.md):

| Route | Purpose |
|---|---|
| `POST /v1/messages` | Anthropic Messages API (the Claude Code surface) |
| `POST /v1/chat/completions` | OpenAI-compatible adjudication proxy |
| `POST /v1beta/models/<model>:generateContent` · `:streamGenerateContent` | Gemini `generateContent` proxy (the Gemini CLI / google-genai surface) |
| `POST /v1/fak/syscall` | Adjudicate and execute one tool call (dispatch to the registered engine) |
| `POST /v1/fak/adjudicate` | Get a pre-execution verdict without executing |
| `POST /v1/fak/admit` | Send a client-executed tool result through the result-side floor |
| `GET·POST /v1/fak/changes` | Drain the cross-agent "what changed" feed (vDSO coherence) |
| `GET·POST /v1/fak/events` | Drain the durable decision-journal tail (after a `?since=` cursor) |
| `POST /v1/fak/revoke` | Refute a poisoned or stale world-state witness fleet-wide |
| `POST /mcp` | MCP over HTTP (JSON-RPC 2.0) |
| `GET /v1/models` | Advertise the served model id |
| `GET /metrics` | Prometheus metrics |
| `GET /healthz` | Liveness (the only auth-exempt route) |

The reference also documents the additional fak-native routes `/v1/fak/context/change`
(tombstone a recall page), `/v1/fak/policy/reload`, and `/v1/fak/trace/reset`, plus
`/v1/messages/count_tokens` and `/debug/vars`.

### The fak_* MCP tools

The five MCP tools your agent calls (the `arguments` object mirrors the matching
fak-native request DTO), from [`examples/mcp/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md):

| Tool | What it does | When you call it |
|---|---|---|
| `fak_adjudicate` | Verdict only (ALLOW / DENY / TRANSFORM / REQUIRE_WITNESS), no execution. A DENY carries a disposition; a TRANSFORM carries the repaired canonical arguments. | Before running a tool your own client executes (the production path) |
| `fak_syscall` | Adjudicate and execute through the kernel (dispatch plus context-MMU result admission). Returns the verdict plus the admitted result. | When fak should run the tool for you |
| `fak_admit` | Submit a result your client executed, to screen it through the result-side stack (context-MMU quarantine plus the IFC taint ledger) before it enters context. | After you run a tool, before you trust its output |
| `fak_changes` | Drain the cross-agent "what changed" feed (typed mutations and revocations since your cursor). | To re-plan or evict your cache when another agent changed shared data |
| `fak_revoke` | Refute an external world-state witness found poisoned or stale; every entry admitted under it is evicted fleet-wide. | When you discover a witness you relied on is bad |

A sixth tool, `fak_context_change` (tombstone a recall page), is exposed over the
`/mcp` HTTP transport and documented in the
[MCP tool-result wire](../mcp-tool-result.md); the five above are the ones an agent
reaches for during a normal session.

---

## 3. Interop protocols (the honest grade)

The protocol landscape is wider than the model wire, and the boundaries differ. fak's
position is consistent: it projects its floor, quarantine, and evidence onto the
protocol that owns each boundary instead of reimplementing the protocol. Some of these
are runtime boundaries a gateway can sit on; others are static files or stdio
transports with nothing live to adjudicate. The grades below mirror the
[interoperability stance](../integrations/interoperability.md) and the
[compatibility matrix](../integrations/compatibility-matrix.md) caveats. Do not read a
"needs an adapter" or "different boundary" row as first-party support.

| Protocol | Boundary it owns | Grade | fak's position |
|---|---|---|---|
| MCP | agent ↔ tools/resources | Native / drop-in | fak *is* the stdio server (`fak serve --stdio`) and fronts MCP over HTTP (`POST /mcp`), exposing the `fak_*` adjudication tools. A runtime boundary fak sits on directly. |
| OpenAI Responses | agent ↔ model | Partial | A runtime boundary. fak proxies *to* a Responses upstream (`--provider openai-responses`) but exposes Chat Completions and Messages to clients; a Responses-default client connects by selecting chat-completions. |
| A2A (Agent2Agent) | agent ↔ agent | Needs an adapter | A runtime boundary fak does not yet speak natively. It projects a policy-filtered Agent Card from its reviewed method registry; the live HTTP edge is planned, not shipped. |
| AG-UI | agent ↔ frontend/UI | Different boundary | Standardizes the UI event stream, not the tool-call boundary fak gates. Orthogonal, not blocked. |
| ACP (BeeAI) | agent ↔ agent | Needs an adapter | Pre-alpha with an unsettled transport. fak would front it through the same registry once it stabilizes. |
| ANP | agent ↔ agent (decentralized) | Needs an adapter | DID identity plus end-to-end encryption. A transparent middle proxy is structurally impossible, so fak would terminate the channel and hold its own DID. |
| llms.txt | discovery / answer-engine context | Different boundary | A static Markdown file served at a fixed path, not a runtime wire. fak [ships one](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt); there is nothing live to sit on. |

The agent-to-agent stance has its own design notes, linked from each row in the
[interoperability stance](../integrations/interoperability.md). The short version: fak
exposes the three model client wires above (OpenAI, Anthropic, Gemini) directly; the
remaining gaps are the agent-to-agent protocols (A2A, ACP, ANP), each a tracked
adapter position rather than a closed door.

---

## Related: the supported-things pages

- [What fak supports (hub)](README.md) — the index of every "supported" page
- [Models](models.md) — in-kernel architectures + any model you front
- [Features](features.md) — every capability with its shipped / simulated / stub status
- [Clouds & hosted providers](clouds.md) — Anthropic, OpenAI, Gemini, xAI, Bedrock, Vertex, Azure, OpenRouter, Together, Groq, Fireworks
- [Agent harnesses & frameworks](agent-harnesses.md) — Claude Code, Cursor, Codex, Aider, Cline, Roo, LangChain, LlamaIndex, CrewAI, …
- [Serving engines](engines.md) — Ollama, vLLM, SGLang, llama.cpp, LM Studio, and the in-kernel reference engine

## Reference (the witnessed sources behind this page)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 44 sourced harnesses / frameworks / backends / protocols, each with the exact repoint key
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every capability with one machine-checked tag (shipped / simulated / stub)
- [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) · [CLI reference](../cli-reference.md) · [Hardware matrix](../HARDWARE-MATRIX.md) · [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt)
