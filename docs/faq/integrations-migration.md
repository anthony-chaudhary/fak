---
title: "fak FAQ — Integrations and migration"
description: "How to connect existing agents and SDKs to fak, migrate endpoints safely, preserve streaming and tools, and roll back cleanly."
---

# Integrations and migration

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Yes — the gateway's read, write, and idle timeouts are each overridable with the `FAK_HTTP_…_TIMEOUT_S` environment variables, and setting one to `0` disables that timeout, which is the knob you want when a slow local CPU decode would otherwise trip the default 90-second write timeout. The defaults are a 10-second read-header timeout, 30-second read, 90-second write, and 120-second idle. The request body is capped at 4 MiB. These are operational dials, not policy: they govern transport, while `--policy` governs which effects are allowed.

## Integrations and migration

Putting `fak` in front of the agent or framework you already run — usually a one-line base-URL change — and moving an existing stack over.

## Do I have to rewrite my agent to put fak in front of it?

No. In almost every case you change exactly one thing — the base URL your agent or framework already points at — and your prompts, tool definitions, and agent loop stay untouched. `fak serve` exposes three wire surfaces on one port, each byte-compatible with a protocol your client already speaks (OpenAI Chat Completions, Anthropic Messages, and fak-native/MCP), so migration is a redirect, not a refactor. Every tool call your model proposes is adjudicated against the capability floor before it reaches your loop, and you can confirm the gate is up with a health check.

```bash
fak serve --addr 127.0.0.1:8080 --base-url <upstream/v1> --model <id> --policy policy.json
curl http://127.0.0.1:8080/healthz
```

## How do I wire Claude Code or the Anthropic SDK to fak?

Point `ANTHROPIC_BASE_URL` at the gateway origin (no `/v1` suffix) and set the API key to any throwaway value for loopback. Claude Code and the Anthropic SDK speak the native Anthropic Messages wire, which `fak serve` serves at `/v1/messages`; the SDK appends `/v1` itself, so you give it the root. Claude Code reads content blocks but not the `fak` response extension, so any drop, repair, or quarantine is also prepended as a leading `[fak] …` text block so you can see what the gate did.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=fak-local
```

## Why does my Anthropic client get a 404 on /v1/v1/messages?

Because Anthropic SDKs append `/v1` themselves, so an Anthropic base URL must point at the gateway origin (`http://127.0.0.1:8080`), not at `.../v1`. Include `/v1` and the SDK turns it into `/v1/v1/messages`, which the gateway doesn't route. This is the single most common wiring mistake and it applies to Claude Code, the Anthropic SDK, `langchain-anthropic`'s `ChatAnthropic`, and any other Anthropic-wire client. OpenAI-wire clients are the opposite — they do include `/v1` in the base URL.

## How do I wire the OpenAI SDK, LangChain, LlamaIndex, or the Vercel AI SDK to fak?

Set the OpenAI base URL to `http://127.0.0.1:8080/v1` and pass any throwaway API key; the framework code stays the same. The exact parameter name differs by client: the OpenAI SDK uses `base_url` and the Vercel AI SDK's `createOpenAI` uses `baseURL`, LangChain's `ChatOpenAI` uses `base_url` (older `langchain-openai` uses `openai_api_base`), and LlamaIndex uses `api_base` (with `OpenAILike` to skip model-name validation for a local model). The OpenAI Agents SDK and any other AsyncOpenAI-based client take the same base URL on the `AsyncOpenAI` you hand the framework.

```python
client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
```

## How do I run fak as an MCP server for Cursor or another MCP client?

Run `fak serve --stdio`, which is an MCP server speaking newline-delimited JSON-RPC over stdin/stdout with no listener and no auth surface. For Cursor, add an `mcpServers` block whose `command` is the absolute path to `fak` with args `["serve","--stdio", …]`; both the `fak` path and any `--policy` path must be absolute. The same stdio dispatch is also reachable over HTTP by starting `fak serve --addr 127.0.0.1:8080` and POSTing to `/mcp`. It exposes adjudication tools including `fak_adjudicate`, `fak_syscall`, `fak_admit`, `fak_changes`, and `fak_revoke`.

## What does the MCP fak_adjudicate tool do versus fak_syscall?

`fak_adjudicate` returns a verdict only and does not execute anything, while `fak_syscall` adjudicates and then executes the call through the kernel. In a typical integration `fak_adjudicate` is the production path: your client asks for a verdict, and if the call is allowed your own code runs the tool. `fak_admit` is the result-side companion that screens a result you already executed through quarantine and taint before it enters context. A DENY is a valid tool result (`isError:false`), not a protocol error — only malformed JSON-RPC produces an error code.

## How do I migrate an existing llama.cpp setup to fak?

Keep your `llama-server` running and point `fak serve --base-url http://127.0.0.1:8131/v1` at it, then move your clients from `:8131/v1` to `:8080/v1`. This is the recommended path: `llama-server` is OpenAI-compatible, so `fak` fronts it as a proxy and you gain the capability floor and result quarantine without touching the engine. There is a second option that drops `--base-url` and passes `--gguf` so `fak` loads the GGUF in-kernel with the embedded tokenizer, but that in-kernel path is a correctness reference, not a production chat engine, so prefer fronting `llama-server` for scale.

## How do I point fak at a hosted provider like OpenAI or Anthropic?

Start `fak serve` with `--provider`, the provider's `--base-url`, and `--api-key-env` naming the environment variable that holds your real upstream key, then move your client's base URL to the gateway. The `--api-key-env` flag names an env var, never a literal key value; `fak` reads it and forwards the real key upstream while your client authenticates to `fak` with a throwaway local key. When the upstream is the real Anthropic API, the gateway can forward the client's original request bytes and its own `x-api-key` as a transparent hop so a real upstream cache hit still reaches the client's accounting.

```bash
fak serve --provider openai --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```

## Will fak break if my model speaks tool calls differently?

`fak` adjudicates the proposed tool calls your upstream model emits, so the upstream must actually produce well-formed tool calls for the gate to act on them. The gateway buffers the whole upstream turn, adjudicates the complete proposed-call set, then re-serializes a well-formed SSE stream, so raw pre-adjudication deltas never pass through. If your upstream announces tool calls but none parse, `fak` fails closed with a `502` rather than forwarding an unverified turn. A self-hosted model that doesn't emit tool calls in its provider's format is a model-side concern, the same as it would be without `fak`.

## How do I prove fak is adjudicating before I migrate my whole agent?

Run a single call against a policy with no server, model, key, or GPU using `fak preflight`, which prints the verdict for one tool call. For an over-the-wire check, start the gateway and POST to `/v1/fak/adjudicate`, which returns a verdict only (no execution) as a `200` carrying the decision. One gotcha on that fak-native route: the JSON key is `arguments`, not `args`, and unknown keys are silently dropped. The repo also ships self-verifying scripts under `examples/` that run the HTTP gate and a real stdio MCP handshake.

```bash
fak preflight --policy policy.json --tool refund_payment --args '{}'
```

## What do I gain on the wire after migrating, and how is a refusal reported?

