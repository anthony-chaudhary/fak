# Add fak to your coding agent (MCP)

`fak serve --stdio` is a Model Context Protocol (MCP) server: it speaks
newline-delimited JSON-RPC over stdin/stdout (the MCP stdio convention — no
listener, no auth surface) and exposes the kernel's adjudication verbs as MCP
tools. Your coding agent (Claude Code, Cursor, or any MCP client) can then route a
proposed tool call through the kernel **before** running it, run a tool *through*
the kernel, or screen a tool result it executed itself — each call adjudicated
against a reviewable capability floor.

## One-paste setup (Claude Code)

1. Get the binary onto your `PATH`: `go build -o fak ./cmd/fak` from `fak/`, or a
   [release binary](../../GETTING-STARTED.md#1-get-the-binary).
2. Copy [`.mcp.json`](.mcp.json) to your **project root**. Claude Code discovers a
   project-level `.mcp.json` automatically and offers to enable the server.
3. Open Claude Code in that project — `fak` appears under `/mcp`, and the
   `fak_*` tools below are available.

The shipped [`.mcp.json`](.mcp.json) wires `fak serve --stdio --policy
examples/dev-agent-policy.json` — adjust the policy path to your own floor (see
[`../../POLICY.md`](../../POLICY.md)) or drop `--policy` to run the raw
fail-closed kernel.

## Other agents

| Agent | How |
|---|---|
| **Claude Code** | project-root `.mcp.json` (above), or `claude mcp add fak -- fak serve --stdio` |
| **Cursor** | add the same `mcpServers` block to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) |
| **Any MCP client** | run `fak serve --stdio` as the server command; or, for HTTP transport, `fak serve --addr 127.0.0.1:8080` and `POST /mcp` |

## The tools fak exposes

| Tool | What it does | When your agent calls it |
|---|---|---|
| `fak_adjudicate` | Verdict only (ALLOW / DENY / TRANSFORM / REQUIRE_WITNESS), no execution. A DENY carries a disposition (RETRYABLE / WAIT / ESCALATE / TERMINAL); a TRANSFORM carries the repaired canonical arguments. | **before** running a tool your own client executes — the production path |
| `fak_syscall` | Adjudicate **and** execute through the kernel (dispatch + context-MMU result admission). Returns verdict + admitted result. | when fak should run the tool for you |
| `fak_admit` | Submit a result your client executed, to screen it through the result-side stack (context-MMU quarantine + IFC taint ledger) **before** it enters context. A poisoned/secret-shaped result comes back QUARANTINE with the bytes paged out; the session's taint high-water mark rises so a later egress is gated. | after you run a tool, before you trust its output — arms the exfil floor on the path where YOU run the tool |
| `fak_changes` | Drain the cross-agent "what changed" feed (typed Mutations + Revocations since your cursor). | to re-plan or evict your cache when another agent changed shared data |
| `fak_revoke` | Refute an external world-state witness (a commit / blob hash / lease epoch) found poisoned or stale; every entry admitted under it is evicted fleet-wide. | when you discover a witness you relied on is bad |

The full input schemas are in `tools/list` (the MCP discovery call) — every tool
takes `{tool, arguments, read_only?, trace_id?, witness?}` (or `{tool, result,
trace_id?}` for `fak_admit`). `fak serve` also exposes these over HTTP at
`POST /mcp`, alongside the OpenAI `/v1/chat/completions` and Anthropic
`/v1/messages` adjudication proxies.

## The other way: front your agent's model

MCP tools let your agent *ask* the kernel about a call. The complementary
deployment puts fak **transparently in front of the model** so it adjudicates
every proposed call with no agent-side changes — point your agent's
`ANTHROPIC_BASE_URL` (or OpenAI base URL) at `fak serve`. That path, witnessed
live on macOS + Windows with the real Claude Code CLI, is
[`../../DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md).
