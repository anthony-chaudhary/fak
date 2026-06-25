---
title: "fak + OpenAI Codex: MCP first, OpenAI-compatible proxy when the wire fits"
description: "Use fak with OpenAI Codex and OpenAI-compatible coding agents. Current Codex CLI/IDE users should start with fak as an MCP server; OpenAI SDKs and Chat Completions clients can repoint their base URL at fak serve."
---

# fak + OpenAI Codex

fak puts a structural policy gate in front of Codex tool use.

> TL;DR: Use `fak serve --stdio` as an MCP server for current Codex CLI and IDE sessions.

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
```

## Fastest path

Codex is OpenAI's coding agent for software development. Its current surfaces include
the Codex CLI, IDE extension, Codex app, and cloud tasks. This guide keeps those surfaces
separate from the generic OpenAI-compatible API path.

There are two useful fak entry points:

| If you run... | Use this fak path | Why |
|---|---|---|
| Current Codex CLI or IDE extension | `fak serve --stdio` as an MCP server | Codex supports MCP, and fak exposes verdict tools without changing Codex's model wire. |
| OpenAI SDKs, OpenAI Agents SDK, LangChain, LlamaIndex, or any Chat Completions client | `fak serve` as an OpenAI-compatible gateway | The client already calls `/v1/chat/completions`, so you repoint its base URL to fak. |

Honest wire boundary: current Codex model-provider docs are Responses-oriented. fak can
proxy to an OpenAI Responses upstream with `--provider openai-responses`. The public
gateway clients hit today are `/v1/chat/completions`, `/v1/messages`, `/mcp`, and
`/v1/fak/*`. fak does not expose a client-facing `/v1/responses` route. For current
Codex CLI/IDE sessions, use MCP first. For OpenAI-compatible SDKs and Chat Completions
agents, use the base-URL proxy path below.

## Why this matters to Codex

Codex reads `AGENTS.md` before it works in this repo. The repo-level rules already tell
it the build, test, commit, and guardrail contract. fak adds a second layer: the kernel can
adjudicate proposed tool calls and tool results with a default-deny floor that a prompt
cannot talk around.

Use the right path for the job:

- MCP path: Codex keeps its normal model/auth path and gains fak's adjudication tools.
- Proxy path: an OpenAI-compatible client sends chat/tool traffic through fak before the
  upstream model sees it.
- Offline proof: run the preflight commands before any key, model, or GPU is involved.

## 60-second proof before wiring Codex

From the repository root:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
```

Expected shape:

- `refund_payment` is denied with `POLICY_BLOCK`.
- `search_kb` is allowed.
- `fak agent --offline` blocks the injected/destructive path while the task still books.

That proves the capability floor is structural, not a model judgment.

## Path 1: Current Codex CLI or IDE extension via MCP

Build the binary:

```bash
go build -o fak ./cmd/fak
```

Optional self-check for the MCP server:

```bash
python examples/mcp/verify.py
```

Add fak to Codex as a local MCP server:

```bash
codex mcp add fak -- ./fak serve --stdio --policy examples/dev-agent-policy.json
```

Then verify Codex can see it:

```bash
codex exec --json "List the active MCP servers, then summarize AGENTS.md."
```

In the interactive Codex CLI, `/mcp` should show the `fak` server. In the IDE extension,
Codex uses the same `config.toml` MCP configuration as the CLI.

What Codex gets from this path:

| MCP tool surface | What it proves |
|---|---|
| `fak_adjudicate` | Ask the kernel for a verdict before running a call. |
| `fak_syscall` | Let the kernel adjudicate and execute a registered call. |
| `fak_admit` | Screen a tool result before it re-enters model context. |
| `fak_context_change` | Read the "what changed" feed when a shared state surface is present. |

Use this path when you are running Codex itself. It preserves Codex's current model wire and
adds fak as an explicit, inspectable tool boundary.

## Path 2: OpenAI-compatible clients through `fak serve`

Start fak in front of an OpenAI-compatible upstream:

```bash
go build -o fak ./cmd/fak
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder \
  --policy examples/dev-agent-policy.json
```

Then repoint an OpenAI-compatible client:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
```

For Python SDK clients:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="fak-local",
)

response = client.chat.completions.create(
    model="qwen2.5-coder",
    messages=[{"role": "user", "content": "List the Go packages in this repo."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
)
```

For TypeScript SDK clients:

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "fak-local",
});
```

Use this path when a framework already lets you set an OpenAI-compatible base URL.
Good fits include:

- OpenAI Agents SDK in Chat Completions mode.
- LangChain, LlamaIndex, AutoGen, and Pydantic AI Chat Completions models.
- Vercel AI SDK OpenAI-compatible providers and similar clients.

## What the kernel blocks for coding workflows

`examples/dev-agent-policy.json` is the coding-agent floor. It allows ordinary
read/search/list flows plus build and test commands. It blocks publish and
self-modification surfaces.

| Attempt | Kernel result |
|---|---|
| Read/search/list calls | Allowed when the tool is on the allow-list or prefix allow-list. |
| `git_diff`, `git_log`, `git_status`, `go_build`, `go_test`, `run_tests` | Allowed by the dev-agent policy. |
| `git_push`, `git_merge`, `git_tag` | Denied with `POLICY_BLOCK`. |
| Writes to `.git/`, `internal/kernel/`, `internal/policy/`, `VERSION`, or `dos.toml` | Denied by the self-modify floor. |
| Secret-shaped fields such as `api_key`, `token`, or `authorization` | Redacted or quarantined by result-side guards. |

Check one call without launching a model:

```bash
./fak preflight --tool git_push --args "{}" --policy examples/dev-agent-policy.json
```

## Using a Responses upstream

If your upstream model provider is the OpenAI Responses API, fak can still be useful as
the gateway's upstream client:

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai-responses \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --policy examples/dev-agent-policy.json
```

Clients still call fak's supported inbound routes. That means:

- OpenAI-compatible clients call `http://127.0.0.1:8080/v1/chat/completions`.
- Anthropic-wire clients call `http://127.0.0.1:8080/v1/messages`.
- Codex CLI/IDE should use the MCP path unless fak grows a client-facing `/v1/responses`
  route.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Codex cannot see the MCP server | Run `codex mcp --help`, re-add the server, then check `/mcp` in the Codex TUI. |
| `codex exec --json` has no fak events | The MCP server is not enabled for that Codex run, or the task did not call fak. |
| OpenAI SDK gets 404 | OpenAI-compatible clients need the `/v1` suffix: `http://127.0.0.1:8080/v1`. |
| Anthropic SDK gets 404 | Anthropic clients need the origin without `/v1`: `http://127.0.0.1:8080`. |
| Everything is denied | Load a policy with `--policy`; with no policy the floor fails closed. |
| You tried to point default Codex model traffic at fak | Use MCP instead, or use a client/framework path that explicitly speaks Chat Completions to fak. |

## Source alignment

This page was checked against the current OpenAI Codex manual on 2026-06-25:

- [Codex overview](https://developers.openai.com/codex/overview)
- [AGENTS.md guidance](https://developers.openai.com/codex/guides/agents-md)
- [Codex MCP](https://developers.openai.com/codex/mcp)
- [Non-interactive `codex exec`](https://developers.openai.com/codex/noninteractive)
- [Codex configuration](https://developers.openai.com/codex/config-basic)

fak-side references:

- [Integration index](README.md)
- [MCP example](../../examples/mcp/README.md)
- [Policy manifest guide](../../POLICY.md)
- [Supported APIs and protocols](../supported/apis-and-protocols.md)
- [Compatibility matrix](compatibility-matrix.md)
