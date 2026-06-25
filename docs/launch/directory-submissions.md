---
title: "fak directory submissions — copy-paste payloads"
description: "Exact field values for the MCP / AI-tool directories that need a web form or account login. Programmatic registries (Glama, the Official MCP Registry, awesome-list PRs) are handled in-repo and tracked separately."
---

# Directory submissions — the human-gated payloads

These directories require *your* account or email, so you submit them; the payloads below
are copy-paste ready. The programmatic ones are already handled in the repo and need no
form:

- **Glama** — auto-indexes from [`glama.json`](https://github.com/anthony-chaudhary/fak/blob/main/glama.json) (committed; approves in minutes).
- **Official MCP Registry** — wired via [`server.json`](https://github.com/anthony-chaudhary/fak/blob/main/server.json) + the ghcr image; one interactive publish step in [`docs/fak/mcp-registry.md`](../fak/mcp-registry.md).
- **Awesome-list PRs** — already submitted across ~12 lists (don't duplicate).

Reusable description (≈140 chars): *In-process default-deny permission gate for AI agents — fronts OpenAI/Anthropic/MCP wires and adjudicates every tool call like a syscall (prompt-injection / tool-poisoning containment).*

---

## 1. mcpservers.org (wong2's list) — web form

**Form:** <https://mcpservers.org/submit> · *Do NOT open a GitHub PR — the repo refuses them.*

| Field | Value |
|---|---|
| Server Name | `fak` |
| Short Description | In-process default-deny permission gate for AI agents fused with a bit-exact KV cache; fronts OpenAI/Anthropic/MCP wires and ships `fak_*` adjudication tools to contain prompt injection and tool poisoning. |
| Link | `https://github.com/anthony-chaudhary/fak` |
| Category | Development |
| Contact Email | `day24@netrasystems.ai` |

Free listing goes to a manual review queue (a $39 tier skips the wait — not necessary).

## 2. mcp.so — GitHub issue (no form login needed)

**Submit:** open a new issue on `chatmcp/mcpso` (the mcp.so "Submit" button routes here).

- **Title:** `Add MCP server: fak`
- **Body:**
  ```
  **Server Name:** fak (Fused Agent Kernel)
  **Description:** Default-deny permission gate for AI agents — fronts OpenAI/Anthropic/MCP wires and adjudicates every tool call like a syscall (prompt-injection / tool-poisoning containment).
  **GitHub URL:** https://github.com/anthony-chaudhary/fak
  **Homepage:** https://anthony-chaudhary.github.io/fak/
  **Transport:** stdio (Streamable HTTP via `fak serve`)
  **Install:** go install github.com/anthony-chaudhary/fak/cmd/fak@latest
  ```

This one is `gh`-doable if you want it automated — say the word and it can be filed for you
(it posts publicly to a third-party tracker, so it's left for your go-ahead).

## 3. Smithery — account + publish

**Path:** sign in at <https://smithery.ai>, then either:
- **GitHub-connected deploy:** the repo already has [`smithery.yaml`](https://github.com/anthony-chaudhary/fak/blob/main/smithery.yaml) (stdio server). Connect GitHub, claim/add the repo, Deploy.
- **CLI:** `smithery mcp publish <url-or-bundle> -n anthony-chaudhary/fak` (needs a Smithery API key).

Note: Smithery is built around remote HTTPS MCP servers; the stdio `smithery.yaml` lists it,
but a hosted HTTPS endpoint (`fak serve --addr ...` behind TLS) gets the richest listing.

## 4. AlternativeTo — community listing

**Submit:** <https://alternativeto.net> → "Add application" (needs a free account).

| Field | Value |
|---|---|
| Name | fak (Fused Agent Kernel) |
| Category | Development / Self-Hosted / Security |
| Short description | Open-source agent kernel: a default-deny permission gate for AI agents that treats every tool call like a syscall, fused with an addressable bit-exact KV cache. One static Go binary, Apache-2.0. |
| License | Open Source (Apache-2.0) |
| Platforms | Linux, macOS, Windows, Self-Hosted |
| Link | `https://github.com/anthony-chaudhary/fak` |
| List it as an alternative to | LangChain guardrails, NeMo Guardrails, E2B, vLLM (governance layer) |

---

*Every description here is capability-first and traces to [`CLAIMS.md`](../../CLAIMS.md). No
performance multipliers are claimed in any listing — keep it that way if you edit them.*
