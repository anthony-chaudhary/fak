---
title: "fak directory submissions — copy-paste payloads"
description: "Exact field values for the MCP / AI-tool directories that need a web form or account login. Programmatic registries (Glama, the Official MCP Registry, awesome-list PRs) are handled in-repo and tracked separately."
---

# Directory submissions — the human-gated payloads

These directories require *your* account or email, so you submit them; the payloads below
are copy-paste ready. The programmatic ones are already handled in the repo and need no
form:

- **Glama** — auto-indexes from [`glama.json`](https://github.com/anthony-chaudhary/fak/blob/main/glama.json) (committed; approves in minutes).
- **Official MCP Registry** — wired via [`server.json`](https://github.com/anthony-chaudhary/fak/blob/main/server.json) + the ghcr image, now at **0.51.0** (the `server.json` `version` and its `oci` `identifier` both pin `ghcr.io/anthony-chaudhary/fak:0.51.0`; `release-container.yml` pushes `:{version,latest}` on each `v*` tag). The remaining steps are owner-only and interactive — see Fresh leads #6 below.
- **Awesome-list PRs** — already submitted across ~12 lists (don't duplicate).

Reusable description (≈140 chars): *Agent kernel for AI agents: one Go binary that fronts OpenAI/Anthropic/MCP wires, keeps long sessions cache-efficient, routes per call, and audits every tool-call verdict.*

---

## 1. mcpservers.org (wong2's list) — web form

**Form:** <https://mcpservers.org/submit> · *Do NOT open a GitHub PR — the repo refuses them.*

| Field | Value |
|---|---|
| Server Name | `fak` |
| Short Description | Agent kernel for AI agents: one Go binary that fronts OpenAI/Anthropic/MCP wires, keeps long sessions cache-efficient, routes per call, and ships `fak_*` adjudication tools. |
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
  **Description:** Agent kernel for AI agents — fronts OpenAI/Anthropic/MCP wires, keeps long sessions cache-efficient, routes per call, and audits every tool-call verdict.
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
| Category | Development / Self-Hosted / Developer Tools |
| Short description | Open-source agent kernel for AI agents: one static Go binary for long-session cache value, per-call routing, audited tool-call verdicts, and addressable bit-exact KV cache. Apache-2.0. |
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
- **Logo:** a **400×400 PNG** attached to the issue. The repo ships square raster brand assets in visuals/brand/ (fak-icon-512.png and fak-mark-512.png); downscale or crop fak-icon-512.png to 400×400 and drag it onto the issue (GitHub issue image upload is web-UI only).
- **Reason for addition (paste):**
  ```
  fak is the Fused Agent Kernel: one static Go binary you put in front of an agent over MCP. Its server (fak serve --stdio) exposes fak_adjudicate / fak_syscall / fak_admit so Cline can get a kernel verdict for a proposed tool call before running it, run one through the kernel, or hold distrusted tool results out of context. It also keeps long sessions cache-efficient, supports per-call routing, and emits an auditable verdict trail. Apache-2.0, two golang.org/x deps.
  ```
- **README-install confirmation (required checkbox):** TRUE. The repo README + [`examples/mcp/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md) give a README-alone install (`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, then `fak serve --stdio` / a project `.mcp.json`). Test it with Cline first, then tick the box.

### 6. Official MCP Registry — the final publish is now unblocked

The blocker the old notes flagged (no OCI artifact) is **gone**: `release-container.yml` builds + pushes `ghcr.io/anthony-chaudhary/fak:{version,latest}` on each `v*` tag (the v0.34.0 image was confirmed anonymously pullable via an anonymous ghcr token when this was first wired on 2026-06-27). `server.json` now tracks the current release — **0.51.0** — with its `oci` `identifier` pinned to `ghcr.io/anthony-chaudhary/fak:0.51.0`. Remaining steps (owner-only, can't be automated; re-confirm the `:0.51.0` image is anonymously pullable before publishing):

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

### 9. Hugging Face Space — the offline in-browser demo (NEW, owner-gated)

Spaces are the one directory where a visitor **runs** a claim instead of reading it. fak ships
no weights (so no model card — that's the fence, not a gap), but the committed
[`spaces/hf-demo/`](https://github.com/anthony-chaudhary/fak/tree/main/spaces/hf-demo) **Docker
Space** runs three offline witnesses in-browser: policy DENY/ALLOW, provable deletion
(`max|Δ|=0`), and the turn tax — no key, no GPU. Runnable source: [spaces/hf-demo/](https://github.com/anthony-chaudhary/fak/blob/main/spaces/hf-demo/README.md).

- **Create:** <https://huggingface.co/new-space> → SDK **Docker**, name `fak-demo`, license `apache-2.0`, public.
- **Push:** the three files in `spaces/hf-demo/` (`README.md` carries the HF `sdk: docker` front-matter, plus `Dockerfile` and `app.py`) to the Space git remote — it builds automatically.
- **Short description (paste):** `Adjudicate every tool call like a syscall; provably evict a poisoned result. Three witnesses, no key, no GPU.`
- **Tags:** `agents`, `llm-security`, `prompt-injection`, `kv-cache`, `mcp`
- **Fence:** fak uploads no weights — the Space runs fak's own binaries; the HF-oracle **numeric** parity (cos=1.000000, KV-evict `max|Δ|=0`) is `go test ./internal/model`, not the Space itself.

---

*Every description here is agent-kernel-first and traces to [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md). No
performance multipliers are claimed in any listing — keep it that way if you edit them.*

<!-- Freshness review 2026-09-06: directory availability, submission status, and copy remain current against repository artifacts and current release v0.51.0. -->
