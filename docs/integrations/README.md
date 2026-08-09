---
title: "Put fak in front of the agent you already run"
description: "Integration index for fak — the agent kernel. Any agent or framework that speaks the OpenAI, Anthropic, or MCP wire drops in by repointing one base URL, with no agent-side code change. Guides for Claude Code, Cursor, OpenAI Codex, and any OpenAI/Anthropic SDK or MCP client."
---

# Run your agent through fak

**Reader:** a builder choosing an integration by client, wire, and support status.
**Lifecycle:** current · **Generation:** wire compatibility is release-independent; the `fak guard` subscription default and live token streaming on the OpenAI wire track the current build.
**Authority:** [what fak supports](../supported/README.md) · [compatibility matrix](compatibility-matrix.md). **Proof:** `python3 examples/wire-proof/verify.py` (60s; no key, model, or GPU).

You don't rewrite your agent to adopt `fak`. You point it at `fak serve` — one kernel in
front of the agent you already run — and the same loop comes out **cheaper and
longer-running**: the shared setup (system prompt, tools, KV cache) is computed once and
reused, the provider's prompt-cache discount survives as the session grows, each call can
route to the right model, and repeated reads are served locally. On that same seam, every
tool call also passes through a default-deny capability floor before it runs — dangerous
calls denied by structure, malformed calls repaired, poisoned results quarantined — so the
loop stays controlled at no extra cost. Your agent, your model, your prompts — unchanged.

```text
  Your agent / client          fak serve              Upstream engine
  (repoint one base URL)     (the gateway)            (serves your tokens)

  ┌────────────────────┐
  │ Claude Code  ──────┼──┐  Anthropic Messages
  │  → claude.md       │  │  POST /v1/messages
  └────────────────────┘  │
  ┌────────────────────┐  │   ┌───────────────────────┐    ┌──────────────┐
  │ OpenAI Codex ──────┼──┼──▶│  default-deny          │──▶ │ OpenAI-compat│
  │  → openai-codex.md │  │   │  capability floor      │    │ (Ollama/vLLM │
  └────────────────────┘  │   │  ┌──────────────────┐  │    │  /SGLang/    │
  ┌────────────────────┐  │   │  │ allow · deny ·   │  │    │  llama.cpp)  │
  │ Cursor (MCP /      │  │   │  │ repair ·         │  │    │  Anthropic / │
  │  OpenAI proxy) ────┼──┤   │  │ quarantine       │  │    │  Gemini / xAI│
  │  → cursor.md       │  │   │  └──────────────────┘  │    └──────────────┘
  └────────────────────┘  │   └───────────────────────┘
  ┌────────────────────┐  │  OpenAI Chat Completions
  │ Any MCP client ────┼──┘  POST /v1/chat/completions
  │  → mcp.md          │     MCP: --stdio / POST /mcp
  └────────────────────┘
```

*Every client repoints one base URL; the same gate adjudicates each tool call before it reaches the upstream engine. Each guide above is linked under "Which agent do you run?".*

Wire compatibility is only the first acceptance check. A first-class fak launcher also
has to document its host-tool dialect, danger-bearing argument fields, and stop/continue
behavior after a denied tool; use the
[harness integration acceptance checklist](harness-acceptance-checklist.md) before
advertising a new launcher here.

If you are wrapping a customer workflow rather than a specific agent launcher, start with
the [reusable harness-loop playbook](harness-loop-playbook.md): selector, executor,
witness/closer, and stop policy before autonomous retries or fleet dispatch.

The reason this works for so many agents is one fact: `fak serve` speaks the wires your
agent already speaks.

| Your agent talks… | fak exposes | You change |
|---|---|---|
| **OpenAI** Chat Completions | `POST /v1/chat/completions` | the base URL → `http://127.0.0.1:8080/v1` |
| **Anthropic** Messages | `POST /v1/messages` | `ANTHROPIC_BASE_URL` → `http://127.0.0.1:8080` |
| **MCP** (Model Context Protocol) | `fak serve --stdio`, or `POST /mcp` | add one server entry |

`fak serve` also fronts **Gemini** and **xAI** upstreams (`--provider gemini` / `xai`),
so the *same* gate sits in front of whichever model actually serves your tokens. The
contrast with a fast token engine (vLLM, SGLang, llm-d, llama.cpp) is **operational surface,
not throughput** — `fak` is the governance + gateway band, in one static Go binary, in
front of the engine. → [One binary is the whole surface](../explainers/one-binary-one-surface.md)

---

## Are you comparing fak with a filter, guardrail, gateway, MCP host, or tool router?

Use the [filtering and tooling parity guide](../concepts/filtering-tooling-parity.md) to decide,
capability by capability, whether to replace a component, integrate it with fak, or do neither.
It separates shipped tool-call/result filtering from complementary prompt/output moderation and
names the trust boundary and failure semantics for each integration.

---

## Which agent do you run?

Pick by **client**, the **wire** it speaks, and its **support** status. **Default:** if a
row offers `fak guard`, that is the shortest path.

### Support status: the canonical vocabulary

**This section is authoritative for support status.** The other integration pages — the
[agent-harnesses page](../supported/agent-harnesses.md), the
[interoperability stance](interoperability.md), and the
[compatibility matrix](compatibility-matrix.md) — reference these labels rather than
defining their own. Support has exactly three values:

- **`fak guard`** — a one-command launcher: `fak guard -- <agent>` starts the gateway
  in-process and injects the base URL into the child only. Every `fak guard` tool also
  has a guide, so the tables write it **`fak guard` + guide**.
- **guide** — a dedicated walkthrough you wire by base URL or MCP. (The
  [agent-harnesses page](../supported/agent-harnesses.md) groups these under "harnesses
  with a dedicated guide" and carries the labels below in a *support* column — a heading,
  not a fourth value.)
- **universal recipe** — a tool not listed in the table below uses the
  [universal recipe](#dont-see-your-framework-the-universal-recipe) and is graded,
  sourced, in the [compatibility matrix](compatibility-matrix.md).

Two companion columns on those pages are **wire facts, not support labels**: the matrix's
*custom base URL* (Yes / Partial / No — can the tool repoint its wire) and the
interoperability page's *connect grade* (Drop-in, Per-wire, Partial, Needs an adapter,
Different boundary, No first-party path — how cleanly fak adjudicates that wire). A tool
carries one support-status label from the three above; the wire facts qualify how the
connection is made, and those pages link back here instead of coining new status words.

| You run… | Wire | Support | Guide |
|---|---|---|---|
| **Claude Code** / the Anthropic API or SDK | Anthropic Messages | `fak guard` + guide | [`claude.md`](claude.md) |
| **OpenAI Codex** (CLI / IDE extension) | OpenAI Chat Completions | `fak guard` + guide | [`openai-codex.md`](openai-codex.md) |
| **Any OpenAI SDK or Chat Completions client** (OpenAI / Agents SDK, LangChain, LlamaIndex, …) | OpenAI Chat Completions | guide | [`openai.md`](openai.md) |
| **OpenCode** (terminal agent) | OpenAI Chat Completions | `fak guard` + guide | [`claude.md#opencode`](claude.md#opencode) |
| **Hermes Agent** (NousResearch self-hosted agent) | OpenAI Chat Completions | `fak guard` + guide | [`hermes.md`](hermes.md) |
| **Cursor** (IDE) | MCP *or* OpenAI proxy | guide | [`cursor.md`](cursor.md) |
| **VS Code + GitHub Copilot** (IDE) | OpenAI/Anthropic proxy | guide | [`vscode.md`](vscode.md) |
| **Zed** (AI-native editor) | OpenAI/Anthropic/MCP | guide | [`zed.md`](zed.md) |
| **JetBrains IDEs** (IntelliJ/PyCharm/WebStorm/…) | OpenAI/Anthropic | guide | [`jetbrains.md`](jetbrains.md) |
| **Aider** (CLI pair-programmer) | OpenAI wire | guide | [`aider.md`](aider.md) |
| **Cline** (VS Code) | OpenAI-Compatible provider | guide | [`cline.md`](cline.md) |
| **Continue** (VS Code) | OpenAI (`config.yaml` `apiBase`) | guide | [`continue.md`](continue.md) |
| **Roo Code** (VS Code) | OpenAI-Compatible *or* MCP | guide | [`roo-code.md`](roo-code.md) |
| **Windsurf** (Cascade agent) | OpenAI-compatible proxy | guide | [`windsurf.md`](windsurf.md) |
| **Any MCP client** | MCP | guide (one-paste `.mcp.json`) | [`mcp.md`](mcp.md) |
| **No agent at all — a product feature that calls a model directly** (screener, classifier, drafter; your own loop, your own tools) | OpenAI or Anthropic SDK, direct API call | guide | [`embed-in-your-product.md`](embed-in-your-product.md) |

**On a Claude Pro/Max subscription?** [`fable5-more-usage-for-free.md`](fable5-more-usage-for-free.md)
is the honest, dogfood-witnessed case for how `fak guard -- claude` stretches the seat you
already pay for — the pure-fak, native-cache-excluded slice (context shedding + KV-prefix
reuse), plus the one genuinely Fable-5-specific lever (the capped-Opus → Fable 5 fallover),
with every number labeled by provenance and the shed-per-fire double-count fence carried in full.

**Not running an agent — you have a product feature that calls a model?** A SaaS/API
backend that calls `messages.create(...)` / `chat.completions.create(...)` itself (a
job-application screener, a support-reply drafter, a field extractor) keeps its own loop
and fronts *that call* with `fak serve` — `fak guard` is a child-process launcher and does
not fit a server-side product. The persona guide, including why the default floor denies
your app's own tools and how to author one that doesn't:
[`embed-in-your-product.md`](embed-in-your-product.md).

**Adopting from outside the repo?** The [adopter playbook](adopter-playbook.md) is the
end-to-end, bare-serve production checklist — prerequisites → `policy.json` → auth-key
env → build/start → `/healthz` → `ANTHROPIC_BASE_URL` wiring — plus the manual Claude
Code MCP-server setup and the CI-embed shape, none of which need the dogfood launcher.

**Running a support desk?** The [customer-support playbook](customer-support-playbook.md)
is the vertical adoption guide — threat model, rollout plan, and runbook for the read-only
customer-support policy template.

Don't see your exact tool below? Read on — if it lets you set a base URL (almost all
do), it already works. The [compatibility matrix](compatibility-matrix.md) is the full
sourced list: 47 harnesses, frameworks, backends, and protocols, each with the exact key
you set to repoint it.

---

## Don't see your framework? The universal recipe

Most agent frameworks and SDKs let you override the model's base URL. When they do,
`fak` drops in with **no other change** — the gate is invisible to your code, and it
adjudicates the tool calls your framework's agent proposes.

First, start the gate in front of whatever serves your tokens:

```bash
# fronts any OpenAI-compatible upstream (Ollama, vLLM, llm-d, llama-server, a cloud API)
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
run `fak serve --stdio` as the server command. The one-paste setup and the `fak_*`
tools it exposes are in [`mcp.md`](mcp.md).

**Need the exact key for *your* tool?** The [compatibility matrix](compatibility-matrix.md)
lists 47 surveyed harnesses, frameworks, backends, and protocols — the wire each speaks,
whether it takes a custom base URL, and the literal env var / arg / config field — each
with a source link.

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
[MCP tools](mcp.md) expose). Swap the mock for your real engine by
adding `--base-url`; nothing else changes.

**Don't take the snippets on faith — run them.** The same two checks (plus an allow-case)
are a one-command, self-verifying script that starts the offline gate, asserts the
verdicts, and tears it down — `PASS`/`FAIL` with a CI-usable exit code, still no model or
key:

```bash
python3 examples/wire-proof/verify.py   # -> PASS, exit 0
```

→ [`examples/wire-proof/`](https://github.com/anthony-chaudhary/fak/blob/main/examples/wire-proof/README.md) (captured output included).

---

## What you get once it's in front

- **A reviewable allow-list** — which tools may run, as a JSON manifest in git, not a
  code edit. Author and check it offline: [`../../POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).
- **Result quarantine** — a poisoned or secret-shaped tool result is paged out before it
  reaches the model's context (injection containment by structure).
- **An audit trail** — JSON access logs and an `X-Trace-Id` per call you can ship to a
  SIEM; Prometheus `/metrics` for the fleet.
- **Auth when you need it** — add `--require-key-env FAK_TOKEN` and the gate requires a
  bearer / `x-api-key` on every request. Same binary, one more flag.

The honest fence: `fak` is **not** the fast token engine, and its own in-binary model is
a correctness reference, not a production server. It fronts your engine — the win is the
governance surface, not tokens per second. Full scope, claim by claim:
[`../../CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).

---

## Cross-references

- [What fak supports](../supported/README.md) — the dedicated capability pages: [models](../supported/models.md), [clouds & hosted providers](../supported/clouds.md), [APIs, wires & MCP](../supported/apis-and-protocols.md), [agent harnesses & frameworks](../supported/agent-harnesses.md), and [serving engines](../supported/engines.md).
- [Two runtimes, one binary: gateway vs agent runtime vs client](../explainers/runtime-vs-client.md) — the first-decision naming behind "put fak in front of your agent": `fak serve` (the gateway runtime that governs model traffic) vs `fak serve --native` (the agent application runtime that owns the loop) vs a harness wrapped by `fak guard` (a governed client). `serve` and `guard` are a runtime and a client, not two versions of one thing.
- [More Fable 5 (and every model) out of your Claude seat](fable5-more-usage-for-free.md) — the reader-facing "more usage for free" case for `fak guard -- claude`, reporting only fak's own authored slice (native prompt cache excluded on purpose): 69.7% WITNESSED KV-prefix reuse, ~37k tokens shed per compaction fire, the capped-Opus → Fable 5 fallover, and the honest fences on what "for free" and "Fable-specific" actually mean.
- [Embed fak in your product](embed-in-your-product.md) — the front door for the *non*-agent adopter: a product feature that makes a direct API call keeps its own loop, repoints one base URL at `fak serve`, and authors a capability floor for its own tools (the default floors allow-list a coding harness's tools, not yours).
- [Harness integration acceptance checklist](harness-acceptance-checklist.md) — the model-wire, host-tool dialect, argument-field, deny-behavior, and replay fixture contract for first-class launchers.
- [Reusable harness-loop playbook](harness-loop-playbook.md) — apply the selector/executor/witness/stop pattern to customer workflows such as support queues, data QA, and eval runs.
- [Agent memory (mem0 / OpenMemory / MCP)](agent-memory.md) — put the gate in front of a memory store: oversized and secret-shaped writes refused, a prompt-injected `delete_all` refused, every recalled memory trust-gated before it re-enters context.
- [Add fak to your agent over MCP](mcp.md) — the MCP-client setup route: one `.mcp.json` paste wires `fak serve --stdio` into Claude Code, Cursor, or any MCP client, checked by the deterministic stdio proof; the wire contract and kernel internals stay linked one layer deeper.
- [Harden any MCP server](harden-any-mcp.md) — drop fak in front of any MCP server: a context-MMU quarantines poisoned tool results out of context and a capability allow-list blocks tools you never wired.
- [fak + the OpenAI API](openai.md) — the OpenAI-client route: the compatible `/v1/*` endpoints on the current build (chat completions with tools + streaming, buffered Responses, models, embeddings, moderations), the supported `guard`/`serve` and upstream choices, and the honest current limits.
- [fak + LiteLLM](litellm.md) — the three topologies (fak in front of a LiteLLM proxy, fak as a governed node behind it, and fak's per-aspect routing dispatching through it), and why supporting LiteLLM is one wire, not a hundred adapters.
- [fak + llm-d](llm-d.md) — front the llm-d Gateway API OpenAI-compatible route, or use the registered `llm-d` engine id for syscall/model-route dispatch.
- [Routers & gateways](routers.md) — OpenRouter, Portkey, LiteLLM Router, Unify, Martian: fak as a complement (govern + route per aspect) to request-level routers, with the honest categorical positioning.
- [Structured output through fak](structured-output.md) — a cookbook for the structured-output surfaces (OpenAI SDK, vLLM/SGLang, LangChain, Pydantic AI, Instructor, BAML, LlamaIndex): the posture table for each library, why prompt-only validation is not token-level constrained decoding, and the policy-deny proof that a schema-valid tool call still adjudicates.
- [Interoperability stance](interoperability.md) — why fak adopts whatever agent/model/framework you run (the one opinion kept is the capability floor) and the honest per-wire grade for the flagship harnesses and every interop protocol.
- [Compatibility matrix](compatibility-matrix.md) — 47 harnesses, frameworks, backends, and protocols, each with its wire, custom-base-URL support, and the exact repoint key, sourced.
- [Getting started](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md) — install the single static binary.
- [Guided tutorial](../fak/tutorial.md) — zero to first adjudicated tool call, real output at every step.
- [Debugging a verdict](debugging.md) — why was my call denied/transformed? Reproduce it offline with `fak preflight --explain`, then trace it across the live gateway.
- [Policy / permissions](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — author, dump, and review the capability floor.
- [FAQ](../FAQ.md) — what fak is, how it differs from a firewall / guardrails / vLLM, the threat model.
- [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) — a machine-readable map for LLMs and answer engines.
- [Custom linter subprocess ABI](custom-linters.md) — the versioned `fak-custom-lint/1` wire schema for user- and agent-authored linters on the agent-hook seam, and its host implementation contract.
- [Extension descriptor and local conformance](extension-descriptors.md) — the `fak-extension-descriptor/1` discovery-metadata contract: catalog enumeration never executes an artifact; local `Verify` re-hashes it and re-runs the bounded witness.
- [Slack helper ownership](slack-helpers-canonical.md) — reusable Slack transport and control infrastructure is canonical in the `slack-helpers` repository; what lives there vs in this tree.