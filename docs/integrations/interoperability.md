---
title: "fak is unopinionated: bring your own agent, model, and protocol"
description: "fak's interoperability stance — it adopts the agent, model, and framework you already run, and the one opinion it keeps is the capability floor. Reads the field through an honest per-wire grade and defers the full sourced table to the compatibility matrix."
---

# Bring your own agent, model, and protocol

fak does not ask you to adopt its agent, its model, or its way of building agents. It
puts a capability floor in front of the stack you already run. You point one base URL at
`fak serve` (or wrap your agent with `fak guard`), and every tool call your agent
proposes crosses that floor before it runs. Your prompts, your tools, and your framework
stay exactly as they were.

> TL;DR: keep your agent, your model, and your framework. Run `fak guard -- claude`, or
> point one base URL at `fak serve`, and every tool call crosses a default-deny floor
> first. The full sourced table of what connects is the
> [compatibility matrix](compatibility-matrix.md).

That works for so many tools because fak speaks the wires they already speak. A client
that talks OpenAI Chat Completions or Anthropic Messages reaches fak by changing one
setting. fak then proxies on to whatever serves your tokens, whether that is OpenAI,
Anthropic, or a local engine like Ollama or vLLM.

This page is the stance and the map. For the exhaustive, sourced table of which tool
takes which base-URL key, see the [compatibility matrix](compatibility-matrix.md). For
the copy-paste recipe, see the [integration index](README.md). This page explains why fak
stays out of your way, and how to read whether a given tool truly connects.

## The one opinion fak keeps

fak is low-ego on purpose. If your team likes LangGraph, use LangGraph. If you prefer
Aider, or Cursor, or a hand-written SDK loop, fak meets you there. There is no claim here
that one agent framework is the right one, and the gate does not care which one you
picked.

The single opinion fak holds is the capability floor: a default-deny allow-list, result
quarantine, and an audit trail, applied at the tool-call boundary. That opinion is the
reason provider-neutrality is a feature instead of a hedge. fak does not author your
model, so it can referee your model's tool calls with no conflict of interest. A vendor's
own guardrail grades its own homework. fak is the disinterested party in the room.

So the suggested path stays small. Keep your stack, and add the floor. Start from the
built-in fail-closed policy (`fak guard --dump-policy`), narrow it to the tools your agent
genuinely needs, then switch on the audit journal when you want a durable record.
Everything else about how you build the agent is yours.

→ [One binary is the whole surface](../explainers/one-binary-one-surface.md) ·
[Policy in the kernel](../explainers/policy-in-the-kernel.md)

## How the connection works

fak exposes three client wires. Pick the one your tool already speaks and repoint it.

| Your tool speaks | Point it at | Exact setting |
|---|---|---|
| OpenAI Chat Completions | `http://127.0.0.1:8080/v1` | base URL / `OPENAI_BASE_URL` (keep the `/v1`) |
| Anthropic Messages | `http://127.0.0.1:8080` | `ANTHROPIC_BASE_URL` (bare host) |
| MCP (Model Context Protocol) | `fak serve --stdio`, or `POST /mcp` | one server entry |

Those are the wires fak serves to clients. It can proxy on to more than it exposes. The
`--provider` flag selects an upstream of OpenAI, Anthropic, Gemini, or xAI, so the same
gate sits in front of whichever model actually serves your tokens. The asymmetry matters
for honesty. A client that speaks the Gemini wire natively, such as the Gemini CLI or the
`google-genai` SDK, has no fak endpoint to point at yet, even though fak can call Gemini
upstream.

`fak guard -- <agent>` automates the wiring for the agents it recognizes. Name a known
agent and guard injects the right wire and base URL into the child process only, leaving
your shell untouched:

```bash
fak guard -- claude       # Anthropic wire, your Claude Pro/Max subscription
fak guard -- codex        # OpenAI wire, inferred from the agent name
fak guard -- opencode     # OpenAI wire, lowercase-tool floor
fak guard -- aider        # OpenAI wire, via the injected OPENAI_API_BASE
```

An unrecognized agent keeps the Anthropic default, and `--provider` always overrides the
guess. On the OpenAI wire, guard sets both `OPENAI_BASE_URL` and `OPENAI_API_BASE`, so a
client that reads either one connects.

## What "connects" honestly means

The [compatibility matrix](compatibility-matrix.md) answers a narrow question for 44
surveyed tools: does it let you set a base URL? This page adds the sharper one. Can fak
actually adjudicate that wire, and how cleanly? The grades:

- Drop-in: one documented base-URL setting points it at a wire fak exposes.
- Per-wire: it connects on its OpenAI-compatible wire. Route its other wire (often a
  separate Anthropic or Gemini provider) through that one.
- Partial: it connects, but the base URL is region-templated or undocumented, or the
  vendor labels the path unsupported.
- Needs an adapter: fak does not speak this wire transparently today. It projects onto the
  protocol, or it would terminate rather than front it.
- Different boundary: a real protocol, but not the tool-call boundary fak gates.
- No first-party path: no user-settable base URL. The backend is brokered by the vendor.

The highest-confidence rows are fak's own first-party integrations, each with a dedicated
guide (✓):

| Tool | Connects via | Grade |
|---|---|---|
| Claude Code ✓ | `ANTHROPIC_BASE_URL`, or `fak guard -- claude` | Drop-in |
| OpenAI Codex ✓ | `OPENAI_BASE_URL`, or `fak guard -- codex` | Drop-in |
| OpenCode ✓ | `OPENAI_BASE_URL` / `opencode.json`, or `fak guard -- opencode` | Drop-in |
| Cursor ✓ | `fak serve --stdio` (MCP) or an OpenAI-compatible proxy base URL | Drop-in |
| OpenAI / Anthropic SDK (raw) | `base_url=` | Drop-in |

Guides: [Claude Code](claude.md) · [Cursor](cursor.md) · [OpenAI Codex](openai-codex.md) ·
the [agent-framework cookbook](../fak/agent-framework-integration.md) for LangChain,
LlamaIndex, CrewAI, AutoGen, and the rest.

For the other surveyed tools the short version holds. If a tool sets a base URL on an
OpenAI- or Anthropic-compatible wire, it is a drop-in or per-wire connect. That covers
Aider, Cline, Continue, Goose, Zed, OpenHands, Qwen Code, LangChain, LlamaIndex, Pydantic
AI, smolagents, the Vercel AI SDK, Ollama, vLLM, SGLang, llama.cpp, and most of the field.
The templated-URL clouds (Azure OpenAI, AWS Bedrock, Google Vertex) are partial: the base
URL is region- or deployment-locked, and the auth is not a plain static key. The closed
backends have no first-party path:

- Windsurf and GitHub Copilot broker model access through a vendor backend, with no
  user-settable OpenAI or Anthropic endpoint.
- Gemini-native clients speak a wire fak does not serve to clients today. Front a
  Gemini-compatible OpenAI client through the OpenAI wire instead.

Every row's exact key, source link, and caveat is in the
[compatibility matrix](compatibility-matrix.md).

## Protocols: fak projects, it does not reinvent

The protocol landscape is wider than the model wire, and the boundaries differ. fak's
position is consistent. It projects its floor, quarantine, and evidence onto the protocol
that owns each boundary, rather than reimplement the protocol.

| Protocol | Boundary it owns | Grade | fak's position |
|---|---|---|---|
| MCP | agent ↔ tools/resources | Drop-in / native | fak *is* the stdio server (`fak serve --stdio`) and fronts MCP-over-HTTP (`/mcp`), exposing five `fak_*` adjudication tools. stdio MCP is fronted by running fak as the server, not by repointing a URL. |
| OpenAI Responses | agent ↔ model | Partial | fak proxies *to* a Responses upstream (`--provider openai-responses`) but exposes Chat Completions and Messages to clients. A Responses-default client connects by selecting `chat_completions`. |
| A2A (Agent2Agent) | agent ↔ agent | Needs an adapter | fak does not speak A2A's JSON-RPC/gRPC bindings natively. It projects a policy-filtered Agent Card from its reviewed method registry (`tools/fleet_agent_link.py a2a-card`); the HTTP edge is planned. |
| ACP (BeeAI) | agent ↔ agent | Needs an adapter | Pre-alpha, with an unsettled transport. fak would front it through the same registry once it stabilizes. |
| ANP | agent ↔ agent (decentralized) | Needs an adapter | DID identity plus end-to-end encryption. A transparent middle-proxy is structurally impossible, so fak would terminate the channel, holding its own DID. |
| AG-UI | agent ↔ frontend/UI | Different boundary | Standardizes the UI event stream, not the tool-call boundary fak gates. Orthogonal, not blocked. |
| llms.txt | discovery / answer-engine context | Different boundary | A static Markdown file, not a runtime wire. fak [ships one](../../llms.txt); there is nothing live to sit on. |

The agent-to-agent stance has its own design notes. They cover why fak projects onto A2A
instead of shipping another A2A SDK, plus the implementation ladder. See
[A2A value opportunities](../a2a-value-opportunities.md) and the
[agent–machine link design](../agent-machine-link-protocol.md).

## Don't see your tool?

If it lets you set a base URL, it almost certainly works. Read the
[universal recipe](README.md#dont-see-your-framework-the-universal-recipe), point the base
URL at `fak serve`, and prove the gate is real in 60 seconds with no model or key:

```bash
python3 examples/wire-proof/verify.py   # -> PASS, exit 0
```

If your tool speaks a wire fak does not yet expose, such as a Gemini-native client or an
agent-to-agent protocol, that is a tracked gap rather than a closed door. The honest grade
is above, and the adapter position is in the protocol docs linked from each row.

## Cross-references

- [Compatibility matrix](compatibility-matrix.md): the full sourced table of 44 tools, the wire each speaks, the exact repoint key, and a source link.
- [Integration index](README.md): which-agent routing, the universal base-URL recipe, and the 60-second offline proof.
- [Claude Code](claude.md) · [Cursor](cursor.md) · [OpenAI Codex](openai-codex.md): the dedicated harness guides.
- [Agent-framework cookbook](../fak/agent-framework-integration.md): exact per-framework code (proxy and explicit-adjudication modes).
- [A2A value opportunities](../a2a-value-opportunities.md) · [agent–machine link](../agent-machine-link-protocol.md): the agent-to-agent protocol stance.
- [Policy / permissions](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md): author and review the capability floor.
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md): every capability with one machine-checked tag.
