---
title: "Put fak in front of the agent you already run"
description: "Integration index for fak — the agent kernel. Any agent or framework that speaks the OpenAI, Anthropic, or MCP wire drops in by repointing one base URL, with no agent-side code change. Guides for Claude Code, Cursor, OpenAI Codex, and any OpenAI/Anthropic SDK or MCP client."
---

# Run your agent through fak

You don't rewrite your agent to adopt `fak`. You point it at `fak serve` — a
kernel-adjudicated gateway — and **every tool call your agent proposes passes through
a default-deny capability floor before it runs**. Dangerous calls are denied by
structure, malformed calls are repaired, and poisoned tool results are quarantined out
of the model's context. Your agent, your model, your prompts — unchanged.

The reason this works for so many agents is one fact: `fak serve` speaks the wires your
agent already speaks.

| Your agent talks… | fak exposes | You change |
|---|---|---|
| **OpenAI** Chat Completions | `POST /v1/chat/completions` | the base URL → `http://127.0.0.1:8080/v1` |
| **Anthropic** Messages | `POST /v1/messages` | `ANTHROPIC_BASE_URL` → `http://127.0.0.1:8080` |
| **MCP** (Model Context Protocol) | `fak serve --stdio`, or `POST /mcp` | add one server entry |

`fak serve` also fronts **Gemini** and **xAI** upstreams (`--provider gemini` / `xai`),
so the *same* gate sits in front of whichever model actually serves your tokens. The
contrast with a fast token engine (vLLM, SGLang, llama.cpp) is **operational surface,
not throughput** — `fak` is the governance + gateway band, in one static Go binary, in
front of the engine. → [One binary is the whole surface](../explainers/one-binary-one-surface.md)

---

## Which agent do you run?

| You run… | Guide |
|---|---|
| **Claude Code** / the Anthropic API or SDK | [`claude.md`](claude.md) |
| **Cursor** (IDE — MCP *or* OpenAI proxy) | [`cursor.md`](cursor.md) |
| **OpenAI Codex** / the OpenAI API or SDK | [`openai-codex.md`](openai-codex.md) |
| **Any MCP client** (one-paste `.mcp.json`) | [`../../examples/mcp/README.md`](../../examples/mcp/README.md) |

**Adopting from outside the repo?** The [adopter playbook](adopter-playbook.md) is the
end-to-end, bare-serve production checklist — prerequisites → `policy.json` → auth-key
env → build/start → `/healthz` → `ANTHROPIC_BASE_URL` wiring — plus the manual Claude
Code MCP-server setup and the CI-embed shape, none of which need the dogfood launcher.

Don't see your exact tool below? Read on — if it lets you set a base URL (almost all
do), it already works.

---

## Don't see your framework? The universal recipe

Most agent frameworks and SDKs let you override the model's base URL. When they do,
`fak` drops in with **no other change** — the gate is invisible to your code, and it
adjudicates the tool calls your framework's agent proposes.

First, start the gate in front of whatever serves your tokens:

```bash
# fronts any OpenAI-compatible upstream (Ollama, vLLM, llama-server, a cloud API)
fak serve --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy floor.json     # omit for the fail-closed default floor
```

Then point your client at it. Pick the wire your framework speaks:

**OpenAI Python SDK** (and anything built on it):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
```

**OpenAI Agents SDK / LangChain / LlamaIndex** — all take the same base URL:

```python
# langchain
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local", model="qwen2.5:1.5b")
```

**Anthropic SDK** (base URL is the gateway root — the SDK appends `/v1/messages`):

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="fak-local")
```

**Vercel AI SDK** (TypeScript) and other JS clients:

```ts
import { createOpenAI } from "@ai-sdk/openai";
const openai = createOpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey: "fak-local" });
```

**Anything that reads the standard env vars** (many CLIs and tools):

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
# or, for Anthropic-wire clients:
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
```

**MCP clients** (the agent *asks* the kernel about a call, rather than being proxied):
run `fak serve --stdio` as the server command. The one-paste setup and the five
`fak_*` tools it exposes are in [`../../examples/mcp/README.md`](../../examples/mcp/README.md).

---

## Prove the gate is real before you wire anything (60 seconds, no key, no model, no GPU)

The capability floor is the same code whether a model is in the loop or not, so you can
watch it deny by structure with nothing installed but [Go 1.26+](https://go.dev/dl/):

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
go run ./cmd/fak agent --offline                                                                                       # injection in context YES->no; destructive op YES->no; task still booked
```

`refund_payment` is refused with a *named reason* — not a model judgment call, a
structural one. Full walkthrough: [`../repro-packet.md`](../repro-packet.md).

---

## See it adjudicate over the wire (same offline gate, the way your agent hits it)

The check above is the CLI; your agent hits the *same* gate over HTTP. Start `fak serve`
with **no `--base-url`** — it serves a deterministic offline mock planner, so this is
still no model, no key, no GPU — and send it a normal OpenAI request. The response comes
back with the kernel's verdict attached, and your agent code never changed:

```bash
fak serve --addr 127.0.0.1:8077 --policy examples/customer-support-readonly-policy.json &
# from a clone, `go run ./cmd/fak serve …` works too

# 1. A normal OpenAI Chat Completions request — exactly what your agent already sends.
curl -s http://127.0.0.1:8077/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"refund my last order"}]}'
```

The model proposes a tool call, and the kernel's inline adjudication rides along in a
`fak` block (response abridged):

```json
{
  "choices": [{ "message": { "tool_calls": [
    { "id": "call_0", "type": "function",
      "function": { "name": "get_user_details", "arguments": "{\"user_id\":\"mia_li_3668\"}" } }
  ] }, "finish_reason": "tool_calls" }],
  "fak": { "adjudications": [
    { "tool_call_id": "call_0", "tool": "get_user_details", "admitted": true,
      "verdict": { "kind": "ALLOW", "by": "monitor" } }
  ] }
}
```

`get_user_details` is on the allow-list, so the kernel **admitted** it and said so inline
— the gate is just *there*, with no agent-side change. Ask it about a tool that is **not**
sanctioned and it refuses by structure:

```bash
# 2. A verdict without executing — the path an MCP client takes before it runs a tool.
curl -s http://127.0.0.1:8077/v1/fak/adjudicate \
  -H 'content-type: application/json' \
  -d '{"tool":"refund_payment","arguments":{"amount":500}}'
# -> {"verdict":{"kind":"DENY","reason":"POLICY_BLOCK","by":"monitor","disposition":"TERMINAL"}, ...}
```

Same gate, two surfaces: transparently in front of the model (the proxy adds the `fak`
block to every response) or asked directly (`/v1/fak/adjudicate`, verdict only — what the
[MCP tools](../../examples/mcp/README.md) expose). Swap the mock for your real engine by
adding `--base-url`; nothing else changes.

**Don't take the snippets on faith — run them.** The same two checks (plus an allow-case)
are a one-command, self-verifying script that starts the offline gate, asserts the
verdicts, and tears it down — `PASS`/`FAIL` with a CI-usable exit code, still no model or
key:

```bash
python3 examples/wire-proof/verify.py   # -> PASS, exit 0
```

→ [`examples/wire-proof/`](../../examples/wire-proof/README.md) (captured output included).

---

## What you get once it's in front

- **A reviewable allow-list** — which tools may run, as a JSON manifest in git, not a
  code edit. Author and check it offline: [`../../POLICY.md`](../../POLICY.md).
- **Result quarantine** — a poisoned or secret-shaped tool result is paged out before it
  reaches the model's context (injection containment by structure).
- **An audit trail** — JSON access logs and an `X-Trace-Id` per call you can ship to a
  SIEM; Prometheus `/metrics` for the fleet.
- **Auth when you need it** — add `--require-key-env FAK_TOKEN` and the gate requires a
  bearer / `x-api-key` on every request. Same binary, one more flag.

The honest fence: `fak` is **not** the fast token engine, and its own in-binary model is
a correctness reference, not a production server. It fronts your engine — the win is the
governance surface, not tokens per second. Full scope, claim by claim:
[`../../CLAIMS.md`](../../CLAIMS.md).

---

## Cross-references

- [Getting started](../../GETTING-STARTED.md) — install the single static binary.
- [Guided tutorial](../fak/tutorial.md) — zero to first adjudicated tool call, real output at every step.
- [Policy / permissions](../../POLICY.md) — author, dump, and review the capability floor.
- [FAQ](../FAQ.md) — what fak is, how it differs from a firewall / guardrails / vLLM, the threat model.
- [llms.txt](../../llms.txt) — a machine-readable map for LLMs and answer engines.
