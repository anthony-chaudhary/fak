---
title: "fak + the OpenAI API: compatible endpoints and supported choices"
description: "The OpenAI-client route for fak — which /v1/* endpoints the gateway serves on the OpenAI wire (chat completions with tools and streaming, buffered Responses, models, embeddings, moderations), the supported guard/serve and upstream choices, and the honest current limits."
---

# fak + the OpenAI API

**Reader:** an OpenAI client user — the OpenAI SDK, OpenAI Agents SDK, LangChain,
LlamaIndex, or any Chat Completions client — identifying which gateway endpoints are
compatible and what the current limits are.
**Lifecycle:** current · **Generation:** the wire shapes are release-independent; the
per-endpoint status below tracks the current build.
**Authority:** [APIs, wires & MCP that fak supports](../supported/apis-and-protocols.md) ·
[compatibility matrix](compatibility-matrix.md).
**Proof:** `python3 examples/wire-proof/verify.py` (seconds; no key, model, or GPU).

Your client keeps its own loop and SDK. You repoint one base URL at `fak serve`, and
every tool call the model proposes crosses the kernel's capability floor before it
executes — allow, deny, repair, or quarantine, with the verdict attached to the response.

## Start here: one checkable action

Prove the OpenAI wire end to end before touching your client — offline, deterministic,
exit code 0/1:

```bash
python3 examples/wire-proof/verify.py
```

It starts `fak serve` with no upstream (a deterministic offline mock planner), sends a
normal `POST /v1/chat/completions`, and checks that the response is a standard chat
completion carrying the kernel's verdict inline. `PASS` (exit 0) means the gateway
serves your wire on this build. Then repoint your client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
```

## Compatible endpoints on the OpenAI wire

What `fak serve` answers for an OpenAI client, on the current build:

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/chat/completions` | Full: tools + streaming | Proposed tool calls are adjudicated; `stream: true` streams content tokens live when the upstream streams (proposed tool calls are held for adjudication, never streamed raw), and is synthesized from the buffered turn otherwise. |
| `POST /v1/responses` | Buffered | Same served-turn core as the chat wire. `stream: true` is refused with a 400 — a client that needs SSE should use the chat wire or [MCP](mcp.md). |
| `POST /v1/completions` | Legacy text wire | The pre-chat text-completion surface; no tools on this wire. |
| `GET /v1/models` | Served | Advertises the model id fak is fronting. |
| `POST /v1/embeddings` | Deterministic, self-contained | An honest feature-hashing backend — not a learned model. Same text, same vector; good for deterministic tests and smoke checks. |
| `POST /v1/moderations` | Deterministic, self-contained | Lexical backend, per-item results on batched input; no model round-trip. |
| `GET /healthz`, `GET /metrics` | Served | Health JSON and Prometheus metrics. |

The Anthropic wire (`POST /v1/messages`) and the fak-native `/v1/fak/*` endpoints live on
the same gateway but are other routes: [claude.md](claude.md) and
[the wire authority](../supported/apis-and-protocols.md).

## Supported choices

Two ways to put the gateway in front of an OpenAI client:

- **Wrap an agent process:** `fak manage --provider openai --api-key-env OPENAI_API_KEY -- <your agent>`
  starts the gateway in-process and injects `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`
  into the child only (the `/v1` matters — OpenAI clients append `/chat/completions`).
- **Run a long-lived gateway:** `fak serve --addr 127.0.0.1:8080 --provider openai --base-url <upstream>/v1 --model <id>`
  and repoint your client's base URL at it.

Upstream choices behind either entry point:

- **A local OpenAI-compatible server** — Ollama, vLLM, SGLang, llama-server, LM Studio:
  `--base-url http://127.0.0.1:11434/v1`.
- **The OpenAI cloud** — `--base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY`.
- **An OpenAI Responses upstream** — `--provider openai-responses`.
- **No upstream at all** — omit `--base-url` for the deterministic offline mock planner
  (what the proof above uses).

The floor is on by default: `fak manage` loads an embedded secure default policy
(`fak manage --dump-policy` prints it), and `fak serve` without `--policy` uses the
fail-closed default. Author your own with
[POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).

## Current limits

- `/v1/responses` is **buffered**; `stream: true` on that route is refused with a 400.
  Live token streaming is the chat-completions route.
- `/v1/embeddings` and `/v1/moderations` are deterministic self-contained backends, not
  learned models; serving a learned model on those routes is a later seam.
- A harness is first-class only when its exact tool dialect is covered, not merely when
  the wire connects — new launcher claims are gated by the
  [harness integration acceptance checklist](harness-acceptance-checklist.md).

## Not your route?

| You are… | Go to |
|---|---|
| A Codex CLI / IDE-extension user | [openai-codex.md](openai-codex.md) |
| An OpenCode user | [claude.md § OpenCode](claude.md#opencode) |
| A Claude Code / Anthropic SDK user | [claude.md](claude.md) |
| After structured output (JSON schema, Instructor, BAML, …) | [structured-output.md](structured-output.md) |
| Running LiteLLM or a request-level router | [litellm.md](litellm.md) · [routers.md](routers.md) |
| A product feature calling the model directly (no agent) | [embed-in-your-product.md](embed-in-your-product.md) |
