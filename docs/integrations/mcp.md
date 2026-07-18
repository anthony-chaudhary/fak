---
title: "Add fak to your agent over MCP"
description: "User setup for fak as an MCP server: one .mcp.json paste wires fak serve --stdio into Claude Code, Cursor, or any MCP client, checked by a deterministic stdio proof script — kernel internals stay one layer deeper."
---

# Add fak to your agent over MCP

**Reader:** an MCP builder wiring the fak kernel into a client they already run.
**Lifecycle:** current · **Generation:** the stdio transport and the `fak_*` adjudication
verbs are release-independent; `tools/list` on your build is the authoritative tool inventory.
**Authority:** [integration index](README.md) · [APIs, wires & MCP](../supported/apis-and-protocols.md).
**Proof / next action:** `python3 examples/mcp/verify.py` (seconds; deterministic; exit 0/1; no model, key, or GPU).

**Start here: which MCP job is yours?** This page serves the first row — completing
setup here never requires reading kernel implementation history. The other rows route
to their own pages.

| You want to… | Read |
|---|---|
| **Wire fak into your MCP client** (Claude Code, Cursor, any MCP client) | **this page** |
| Put fak in front of another MCP server you already run | [`harden-any-mcp.md`](harden-any-mcp.md) |
| Front your agent's **model** instead — a base-URL proxy, no per-call asking | [`README.md`](README.md); for Claude Code, [`claude.md`](claude.md) |
| Read the wire contract or kernel internals (contributor) | [the deeper layers](#where-the-deeper-layers-live) below |

## What you are wiring

`fak serve --stdio` is a Model Context Protocol server: newline-delimited JSON-RPC 2.0
over stdin/stdout — no listener, no auth surface, no network. It exposes the kernel's
adjudication verbs as MCP tools, so your agent can ask for a verdict **before** running
a call (`fak_adjudicate`), run a tool **through** the kernel (`fak_syscall`), or screen
a result it already executed (`fak_admit`). Every call is adjudicated against a
reviewable capability floor that lives in git as a JSON manifest.

## Setup: Claude Code (one paste)

1. Get the binary onto your `PATH` — `go build -o fak ./cmd/fak` from a clone (the Go
   module is the repo root), or a
   [release binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#1-get-the-binary).
2. Copy [`examples/mcp/.mcp.json`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/.mcp.json)
   to your **project root**:

   ```json
   {
     "mcpServers": {
       "fak": {
         "command": "fak",
         "args": ["serve", "--stdio", "--policy", "examples/dev-agent-policy.json"],
         "env": {}
       }
     }
   }
   ```

3. Open Claude Code in that project — it discovers a project-level `.mcp.json`, offers
   to enable the server, and `fak` appears under `/mcp` with the `fak_*` tools available.

The shipped entry wires the example
[dev-agent floor](https://github.com/anthony-chaudhary/fak/blob/main/examples/dev-agent-policy.json).
Point `--policy` at your own reviewed floor
([POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)), or drop the
flag to run the raw fail-closed kernel (default-deny: every tool refused until you
allow it).

## Setup: other clients

| Client | How |
|---|---|
| **Claude Code** (without the paste) | `claude mcp add fak -- fak serve --stdio` |
| **Cursor** | the same `mcpServers` block in `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) — see [`cursor.md`](cursor.md) |
| **Any MCP client, stdio** | run `fak serve --stdio` as the server command |
| **Any MCP client, HTTP** | `fak serve --addr 127.0.0.1:8080`, then `POST /mcp` |

## Check it worked (the one next action)

From a clone root (the script and the example policy live in the repo, so this one
check needs the clone even if your own project only carries `.mcp.json`):

```bash
python3 examples/mcp/verify.py    # -> PASS / FAIL, exit 0 / 1
```

The script drives the **real stdio transport** — the exact path `.mcp.json` wires — and
is deterministic: the same four checks return the same verdicts on every run, with no
model, no key, no GPU, no network. `PASS` (exit 0) means all four held:

1. the JSON-RPC handshake names the server (`fak-gateway`);
2. `tools/list` discovery lists the `fak_*` adjudication tools;
3. a shared-history mutation (`git_push`) is refused **DENY / POLICY_BLOCK**;
4. a read (`git_status`) is allowed — the floor is live, not a blanket deny.

A captured run with the raw JSON-RPC frames is in
[`examples/mcp/EXAMPLE-OUTPUT.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/EXAMPLE-OUTPUT.md).

## The tools your agent gets

The core verbs are `fak_adjudicate` (verdict only, before your client runs a tool),
`fak_syscall` (adjudicate and execute through the kernel), `fak_admit` (screen a result
you already ran, before it enters context), `fak_read` (kernel-cached file reads),
and `fak_changes` / `fak_revoke` (the cross-agent coherence feed). The per-tool table —
what each does and when your agent calls it — is in
[`examples/mcp/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md#the-tools-fak-exposes);
the full input schemas come from `tools/list` on your build, which is authoritative.

**Scope, honestly:** the verify script exercises the call-side capability gate over MCP
stdio. The result-side stack (context-MMU quarantine, IFC taint ledger) is reached via
`fak_admit` / `fak_syscall`; its claim-by-claim scope is
[CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).

## Where the deeper layers live

Setup does not require these; they are the contributor and maintainer layers behind it.

- [Worked example with captured frames](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md) — the runnable proof directory this page's setup and next action come from.
- [MCP tool-result envelope](../mcp-tool-result.md) — the `SyscallResponse` wire shape and the closed refusal vocabulary (contributor).
- [MCP tool-schema floor baseline](../context-budget/mcp-tool-floor.md) — the measured always-sent schema token budget (contributor).
- [Publishing fak to the Official MCP Registry](../fak/mcp-registry.md) — how the server is listed for discovery (maintainer).
