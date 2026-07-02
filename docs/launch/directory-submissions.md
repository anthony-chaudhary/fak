---
title: "fak directory submissions — copy-paste payloads"
description: "Exact field values for the MCP / AI-tool directories that need a web form or account login. Programmatic registries (Glama, the Official MCP Registry, awesome-list PRs) are handled in-repo and tracked separately."
---

# Directory submissions — the human-gated payloads

These directories require *your* account or email, so you submit them; the payloads below
are copy-paste ready. The programmatic ones are already handled in the repo and need no
form:

- **Glama** — auto-indexes from [`glama.json`](https://github.com/anthony-chaudhary/fak/blob/main/glama.json) (committed; approves in minutes).
- **Official MCP Registry** — wired via [`server.json`](https://github.com/anthony-chaudhary/fak/blob/main/server.json) + the ghcr image, now at **0.34.0** and matching the published image. As of 2026-06-27 `ghcr.io/anthony-chaudhary/fak:0.34.0` + `:latest` are built and **anonymously pullable** (the prereq is satisfied), so only the one interactive publish step in [`docs/fak/mcp-registry.md`](../fak/mcp-registry.md) remains — see Fresh leads #6 below.
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
| Contact Email | `<your-contact-email>` |

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

## Fresh leads (added 2026-06-27)

Researched + adversarially verified this session. The first is a brand-new
agent/MCP marketplace not in the original campaign; the rest are state changes that
unblock or extend what was already wired.

### 5. Cline MCP Marketplace — GitHub issue (NEW, non-duplicate)

The Cline IDE-agent's marketplace drives installs to a large audience. The sibling
project `DOS` was already submitted ([cline/mcp-marketplace#1794](https://github.com/cline/mcp-marketplace/issues/1794)), but **fak has not been** — confirmed no open `fak` submission. Submission is a GitHub issue (not a PR).

**Submit:** open a new issue on [`cline/mcp-marketplace`](https://github.com/cline/mcp-marketplace/issues/new/choose) with the *Server Submission* template.

- **GitHub Repo URL:** `https://github.com/anthony-chaudhary/fak`
- **Logo:** a **400×400 PNG** attached to the issue. *This is the one missing asset — the repo has wide diagrams (`visuals/*.png`) and a 1200×630 `visuals/social-preview.png`, but no square icon. Make/crop a 400×400 fak icon and drag it onto the issue (GitHub issue image upload is web-UI only).* 
- **Reason for addition (paste):**
  ```
  fak is the Fused Agent Kernel: a default-deny tool-call firewall + result quarantine you put in front of an agent over MCP. Its server (fak serve --stdio) exposes fak_adjudicate / fak_syscall / fak_admit so Cline can screen a proposed tool call BEFORE running it, run one through the kernel, or hold a poisoned tool result out of context entirely — addressing the MCP Top-10 (Tool Poisoning, Memory Poisoning) by structure, not a classifier. One static Go binary, Apache-2.0, zero deps.
  ```
- **README-install confirmation (required checkbox):** TRUE. The repo README + [`examples/mcp/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md) give a README-alone install (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, then `fak serve --stdio` / a project `.mcp.json`). Test it with Cline first, then tick the box.

### 6. Official MCP Registry — the final publish is now unblocked

The blocker the old notes flagged (no OCI artifact) is **gone**: `release-container.yml` ran green for the v0.34.0 tag and `ghcr.io/anthony-chaudhary/fak:0.34.0` + `:latest` are publicly pullable (verified via an anonymous ghcr token). `server.json` is now bumped to 0.34.0 to match. Remaining steps (owner-only, can't be automated):

1. **Make the ghcr `fak` package public** (first publish only) — repo *Packages* tab → set visibility to public.
2. `brew install mcp-publisher` (or the release tarball — see [`docs/fak/mcp-registry.md`](../fak/mcp-registry.md)).
3. `mcp-publisher login github` — interactive GitHub device flow that claims the `io.github.anthony-chaudhary/*` namespace.
4. `mcp-publisher publish` from the repo root (reads `server.json`).

Future releases now keep `server.json` current automatically (`release_bump.py`'s `dist_manifests` target), so step 2-4 is the only recurring cost.

### 7. Claude Code plugin — SHIPPED, just announce + smoke-test

A self-hosted plugin marketplace is now in the repo ([`.claude-plugin/marketplace.json`](https://github.com/anthony-chaudhary/fak/blob/main/.claude-plugin/marketplace.json) + `plugins/fak/`). Users adopt fak in two commands:

```text
/plugin marketplace add anthony-chaudhary/fak
/plugin install fak@fak
```

Smoke-test it once (`/plugin marketplace add` against the live repo, install, `/mcp` shows the `fak` server), then it's a one-paste adopt path you can cite in the README and the social posts.

### 8. Integration / guardrail docs PRs — high value, human-authored

These add fak to the docs of tools fak fronts, reaching THEIR users (not just a backlink). They overlap fak's own interop epic **#1016** (#1017-1020) — coordinate so the outbound PR and the inbound wire land together. Ranked by merge-likelihood:

| Target | PR shape | Why it fits |
|---|---|---|
| [BerriAI/litellm](https://github.com/BerriAI/litellm) | A guardrail-provider doc page under `docs/my-website` (model it on `presidio.md`/`lakera.md`), or register in `SupportedGuardrailIntegrations` / the `litellm-guardrails` registry, or expose fak via the no-PR *Generic Guardrail API*. | LiteLLM already documents third-party guardrails returning BLOCKED/NONE/GUARDRAIL_INTERVENED — exactly fak's default-deny tool gate. Reciprocal: fak already has [`docs/integrations/litellm.md`](../integrations/litellm.md). |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | A runnable example under `examples/model_providers/` repointing `base_url` at `fak serve` for a governed gateway. | The SDK ships `examples/model_providers/` and resolves via custom base_url; fak drops in with zero agent-side change. |
| [block/goose](https://github.com/block/goose) | A docs recipe (custom OpenAI-compatible provider) pointing Goose at `fak serve`. | Model-agnostic CLI agent, any OpenAI-compatible endpoint, community-PR-friendly docs. |
| [vercel/ai](https://github.com/vercel/ai) | A community-provider / `createOpenAICompatible` example. | High-traffic TS audience; community providers are an established category. |

*Not actionable yet:* **awesome-go** — its ≥5-month-commit-history gate is provably failed (repo created 2026-06-21); earliest eligibility ~2026-11-21, and only if Go Report Card grades A-/A/A+ (note: Go Report Card is sunsetting). **Homebrew core** needs ~225 stars; a personal tap (`homebrew-fak`) is available now if Mac reach matters.

---

*Every description here is capability-first and traces to [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md). No
performance multipliers are claimed in any listing — keep it that way if you edit them.*
