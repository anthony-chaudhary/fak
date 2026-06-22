---
title: "fak external-adopter integration playbook"
description: "The end-to-end playbook for adopting fak from outside the repo: front your model with a bare `fak serve` (production checklist — prerequisites, policy, auth-key env, build/start, /healthz, ANTHROPIC_BASE_URL wiring), run `fak serve --stdio` as a manual Claude Code MCP server, or embed the kernel in CI. No dogfood launcher required."
---

# fak adopter playbook

This is the playbook for adopting `fak` **from outside this repository** — you have
your own agent, your own model, and your own CI, and you want the kernel's
default-deny capability floor in front of them. It covers the three real integration
shapes, each self-contained:

| Shape | You want… | Jump to |
|---|---|---|
| **A — Front a model** (`serve --base-url`) | every tool call your agent proposes adjudicated transparently, with no agent-side code change | [§A](#a--front-a-model-the-bare-serve-production-path) |
| **B — Manual MCP server** (`serve --stdio`) | your agent to *ask* the kernel about a call before it runs one it executes itself | [§B](#b--fak-as-a-manual-claude-code-mcp-server) |
| **C — Embed in CI** | author and check the capability floor with no model, no key, no GPU | [§C](#c--embed-the-kernel-in-ci) |

The repo's own dogfood launcher (`scripts/dogfood-claude.sh`) wires all of this for a
local model in one command — see [`../../DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md).
This playbook is the **bare-serve** path an external adopter follows by hand, without
that launcher.

> **One binary, every shape.** `fak` is a single static Go binary (zero external
> dependencies, no `go.sum`, no Python, no CUDA toolchain). The same artifact serves
> the HTTP gateway, the stdio MCP server, and the offline CI check — you add flags,
> not components.

---

## A — Front a model (the bare-serve production path)

This is the production checklist for putting `fak serve` transparently in front of a
model you already serve. Your agent talks to `fak`; `fak` adjudicates every proposed
tool call and proxies the rest upstream. Work top to bottom.

### A.0 — Prerequisites

- **The `fak` binary on your `PATH`.** Either a [release binary](../../GETTING-STARTED.md#1-get-the-binary)
  (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, or download from the
  releases page), or build from a clone: `go build -o fak ./cmd/fak` (Go 1.26+).
- **An OpenAI-compatible (or Anthropic/Gemini/xAI) model server** already running and
  reachable — Ollama, vLLM, SGLang, llama-server, LM Studio, or a cloud API. `fak`
  fronts your engine; it is **not** the fast token engine itself.

### A.1 — Author the capability floor (`policy.json`)

With **no policy, the kernel default-denies every tool** — fail-closed. So a real
deployment ships a reviewable allow-list as a JSON manifest in git. Start from a
shipped example and check it offline before it ever binds a listener:

```bash
fak policy --dump > policy.json          # start from the built-in default floor
# …edit policy.json: allow your tools, add deny-by-argument rules, set self-modify guards…
fak policy --check policy.json           # static validation, no model, no listener
fak serve --policy-check --policy policy.json   # the serve-path validator (also binds nothing)
```

Shipped starting points (copy one and adapt): `examples/dev-agent-policy.json`,
`examples/customer-support-readonly-policy.json`, `examples/repo-guard-policy.json`.
Schema and authoring guide: [`../../POLICY.md`](../../POLICY.md).

### A.2 — Set the auth-key env (production)

By default `fak serve` requires **no auth** — fine for a loopback `127.0.0.1` bind, not
for anything reachable. For production, bind a token to an env var and require it on
every request:

```bash
export FAK_TOKEN="$(openssl rand -hex 32)"   # the bearer your clients must present
```

`--require-key-env FAK_TOKEN` (below) then rejects any request without a matching
bearer / `x-api-key`. If your upstream provider needs its own key, hold that in a
separate env var and pass `--api-key-env` (e.g. `--api-key-env OPENAI_API_KEY`).

### A.3 — Build / start the gateway

```bash
fak serve \
  --addr 0.0.0.0:8080 \
  --provider openai \
  --base-url http://127.0.0.1:11434/v1 \   # your upstream model server
  --model qwen2.5-coder:7b \
  --policy policy.json \
  --require-key-env FAK_TOKEN
```

Swap `--provider` for `anthropic`, `gemini`, or `xai` to front those upstreams; the
gate is identical. Drop `--base-url` and `fak` serves a deterministic **offline mock
planner** instead — useful for wiring tests before a real model is in the loop.

### A.4 — Verify health (`/healthz`)

```bash
curl -s http://127.0.0.1:8080/healthz
# {"ok":true,"model":"qwen2.5-coder:7b","engine":"inkernel"}
```

A `200` with `"ok":true` means the gateway is up and the upstream model id is
advertised. Also useful: `GET /v1/models` (the advertised model id) and `GET /metrics`
(Prometheus counters for the fleet).

### A.5 — Wire your client (`ANTHROPIC_BASE_URL`)

Point your agent's base URL at `fak`. Nothing else in your agent changes — the gate is
invisible to your code.

**Claude Code / any Anthropic SDK** (base URL is the gateway root; the SDK appends
`/v1/messages`):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="$FAK_TOKEN"      # the bearer from A.2
claude                                      # Claude Code now runs through the kernel
```

**Any OpenAI-wire client** (OpenAI SDK, LangChain, LlamaIndex, the Agents SDK, Vercel
AI SDK):

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="$FAK_TOKEN"
```

That's the full bare-serve adopter loop: policy → auth → start → `/healthz` → repoint
one base URL. Per-framework snippets (Python, TypeScript, env-var-only clients) are in
the [integration index](README.md#dont-see-your-framework-the-universal-recipe); the
Claude-specific deep dive (architecture, denial table, cloud providers, observability,
troubleshooting) is in [`claude.md`](claude.md).

---

## B — fak as a manual Claude Code MCP server

The complementary shape: instead of proxying the model, your agent **asks the kernel**
about a call it is about to run itself. `fak serve --stdio` is a
[Model Context Protocol](https://modelcontextprotocol.io) server — it speaks
newline-delimited JSON-RPC over stdin/stdout (no listener, no auth surface) and exposes
the kernel's adjudication verbs as MCP tools.

> **fak is wired as a *manual* MCP server — there is no packaged `.claude-plugin/` /
> marketplace entry.** You add it as a standard MCP server with the snippet below; the
> single static binary *is* the server. (If a packaged plugin lands later this section
> is where it will be linked.)

### Copy-pasteable Claude Code config

Either drop this `.mcp.json` at your **project root** (Claude Code discovers it
automatically and offers to enable the server):

```json
{
  "mcpServers": {
    "fak": {
      "command": "fak",
      "args": ["serve", "--stdio", "--policy", "policy.json"],
      "env": {}
    }
  }
}
```

…or register it from the CLI in one line:

```bash
claude mcp add fak -- fak serve --stdio --policy policy.json
```

Open Claude Code in that project and `fak` appears under `/mcp`. Adjust the `--policy`
path to your own floor (see §A.1) or drop `--policy` to run the raw fail-closed kernel.
The same `mcpServers` block works for **Cursor** (`.cursor/mcp.json`) and any other MCP
client; for HTTP transport instead of stdio, run `fak serve --addr 127.0.0.1:8080` and
`POST /mcp`.

### The six tools fak exposes

| Tool | What it does |
|---|---|
| `fak_adjudicate` | Verdict only (ALLOW / DENY / TRANSFORM / REQUIRE_WITNESS), no execution — call it **before** running a tool your client executes. |
| `fak_syscall` | Adjudicate **and** execute through the kernel (dispatch + result admission). |
| `fak_admit` | Screen a result your client already executed through the result-side floor (quarantine + IFC taint) before it enters context. |
| `fak_changes` | Drain the cross-agent "what changed" feed (typed mutations + revocations since your cursor). |
| `fak_revoke` | Refute a poisoned/stale world-state witness; every entry admitted under it is evicted fleet-wide. |
| `fak_context_change` | Record a context-changing event so dependent cached spans are invalidated coherently. |

Full input schemas come back from the MCP `tools/list` discovery call. The in-repo
one-paste example (with a shipped `.mcp.json`) is
[`../../examples/mcp/README.md`](../../examples/mcp/README.md).

---

## C — Embed the kernel in CI

The capability floor is the same code with or without a model, so you can assert it
**denies by structure** in CI with nothing installed but Go 1.26+ — no key, no model,
no GPU. Use it to gate a policy change before it ships:

```bash
# Fail the build if a known-dangerous call is ever ALLOWed by your floor:
fak preflight --policy policy.json --tool refund_payment --args '{}'   # expect: DENY (POLICY_BLOCK)
fak preflight --policy policy.json --tool search_kb      --args '{}'   # expect: ALLOW

# Validate the manifest itself parses and is well-formed:
fak policy --check policy.json
```

`preflight` exits non-zero on a denied verdict, so a CI step that expects `DENY` can
assert it directly. A self-verifying, copy-run example (starts an offline gate, asserts
the verdicts, tears it down, emits a CI-usable exit code) lives at
[`../../examples/wire-proof/`](../../examples/wire-proof/README.md).

---

## Cross-references

- [Integration index](README.md) — the front door: which-agent routing + the universal "set the base URL" recipe.
- [Claude Code / Anthropic API guide](claude.md) — the full Claude deep dive (architecture, denial table, cloud providers, observability).
- [MCP one-paste setup](../../examples/mcp/README.md) — the in-repo `.mcp.json` example and the `fak_*` tools.
- [Getting started](../../GETTING-STARTED.md) — install the single static binary; the four run tiers.
- [Policy / permissions](../../POLICY.md) — author, dump, and review the capability floor.
- [Dogfood launcher](../../DOGFOOD-CLAUDE.md) — the one-command local-model variant of Shape A.
