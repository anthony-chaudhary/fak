---
title: "OpenAI internal pilot pitch - fak for Codex, Agents SDK, MCP, and runtime tool governance"
description: "A one-page OpenAI-facing pilot memo for adopting fak as an externalized tool-call governance boundary around Codex, OpenAI Agents SDK tool guardrails, MCP servers, and enterprise managed configuration."
---

# OpenAI internal pilot pitch - fak for Codex, Agents SDK, MCP, and runtime tool governance

**Audience:** Codex CLI/IDE/app-server, Agents SDK, Codex Security, enterprise governance.
**Ask:** run a one-day pilot where `fak` is added as a Codex MCP server and mapped to
Agents SDK tool guardrails, proving "unsafe call denied, useful work continues."

## Why this is attractive to OpenAI

OpenAI already has strong control surfaces: Codex sandboxing, approvals, MCP
configuration, managed configuration, and security workflows. The OpenAI-facing pitch
should therefore avoid "replace your controls." The better pitch is:

> `fak` is the external, evidence-bearing tool-call kernel that complements Codex and
> Agents SDK guardrails when a team needs a portable policy floor, result admission, and
> audit trail across MCP servers and OpenAI-compatible clients.

Current OpenAI docs make the fit concrete:
[Codex CLI](https://developers.openai.com/codex/cli),
[agent approvals and security](https://developers.openai.com/codex/agent-approvals-security),
[Codex MCP](https://developers.openai.com/codex/mcp),
[managed configuration](https://developers.openai.com/codex/enterprise/managed-configuration),
[Codex Security](https://developers.openai.com/codex/security), and the Agents SDK
[guardrails](https://openai.github.io/openai-agents-python/guardrails/),
[MCP](https://openai.github.io/openai-agents-python/mcp/), and
[tracing](https://openai.github.io/openai-agents-python/tracing/) docs.

## One-day pilot

Start with the deterministic MCP proof. It needs no model, key, or network:

```bash
go build -o fak ./cmd/fak
python examples/mcp/verify.py
```

Expected evidence:

| Check | Evidence |
|---|---|
| MCP handshake works | `verify.py` initializes the `fak-gateway` server over stdio. |
| Codex-visible tools exist | `tools/list` includes `fak_adjudicate`, `fak_syscall`, `fak_admit`, `fak_changes`, `fak_revoke`, and `fak_context_change`. |
| Dangerous shared-history mutation is denied | `git_push` returns `DENY` with `POLICY_BLOCK`. |
| Safe read is not blanket-blocked | `git_status` returns `ALLOW`. |

Then register `fak` as a Codex MCP server:

```bash
codex mcp add fak -- ./fak serve --stdio --policy examples/dev-agent-policy.json
codex exec --json "List active MCP servers, then summarize AGENTS.md."
```

For a non-Codex OpenAI-compatible agent, use the proxy path:

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder \
  --policy examples/dev-agent-policy.json

export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
```

## Agents SDK guardrail mapping

The adapter should be tiny. It should call the kernel before a tool executes and after a
tool returns, then translate `fak` verdicts into the SDK's guardrail outcomes.

| `fak` surface | Agents SDK hook | Mapping |
|---|---|---|
| `fak_adjudicate` | Tool input guardrail | `ALLOW` lets the tool run; `DENY` rejects the tool call; `REQUIRE_WITNESS` raises or pauses for approval; `TRANSFORM` returns canonical arguments to the wrapper. |
| `fak_admit` | Tool output guardrail | `ALLOW` admits the result; `QUARANTINE` replaces model-visible content with a safe message and records the held reference; `DENY` raises a tripwire. |
| `trace_id` | Tracing span metadata | Attach the `fak` trace id, verdict kind, reason, and policy digest to the tool span. |
| `fak_revoke` | Incident response utility | Evict context/cache tied to a bad world-state witness. |

This lets OpenAI teams keep the Agents SDK programming model while testing a
default-deny, portable policy floor.

## Runtime boundary against Codex controls

| OpenAI control | What it already does | What `fak` adds |
|---|---|---|
| Sandbox | Contains local command execution and filesystem/network reach. | Decides whether a proposed tool/effect should run at all, before execution. |
| Approval policy | Routes boundary-crossing actions to a user or reviewer. | Produces machine-readable refusal reasons and can require witnesses before a tool runs. |
| MCP server config | Connects Codex to external tools and context providers. | Governs calls into those MCP tools and screens results before context admission. |
| Managed config | Lets admins set requirements/defaults. | Gives admins a portable policy manifest plus audit rows to enforce and inspect. |
| Codex Security | Finds and validates likely code vulnerabilities. | Governs runtime tool execution and result admission during agent work. |

## Residual risks to name up front

- `fak` does not replace Codex sandboxing or approvals; it is a complementary runtime
  policy plane.
- Current `fak` docs say Codex should start with MCP. A client-facing `/v1/responses`
  route is not the primary public inbound path today.
- The Agents SDK adapter is still a needed artifact. Until it exists, the guardrail
  mapping is a design and can be piloted through MCP or direct HTTP calls.
- The OpenAI-specific cache/fleet value should be proven with OpenAI-compatible
  telemetry before it is used as the headline argument.

## Decision OpenAI can make

If the pilot passes, OpenAI can adopt the idea without adopting the repo:

> Standardize a tool-governance profile for Codex, MCP, and Agents SDK guardrails:
> pre-execution verdict, post-result admission, quarantine, revocation, and audit.

`fak` then becomes one open reference implementation and conformance generator, not a
replacement for Codex or the Agents SDK.
