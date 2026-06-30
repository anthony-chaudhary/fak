---
title: "Anthropic internal pilot pitch - fak for Claude Code, MCP, and enterprise policy"
description: "A one-page Anthropic-facing pilot memo for adopting fak as an externalized tool-call governance boundary for Claude Code, Anthropic Messages, MCP, and enterprise managed-policy workflows."
---

# Anthropic internal pilot pitch - fak for Claude Code, MCP, and enterprise policy

**Audience:** Claude Code product/SDK, MCP ecosystem, API tools, enterprise security.
**Ask:** run a one-day pilot where `fak` governs Claude Code tool calls, records a
hash-chained audit trail, and proves "unsafe call denied, useful work continues."

## Why this is attractive to Anthropic

Claude Code and the Claude API already expose the surfaces `fak` is built to govern:
tools, MCP, hooks, subagents, skills, permissions, sessions, and managed settings.
Anthropic's own docs frame these as product and enterprise control points:
[Claude Code overview](https://docs.anthropic.com/en/docs/claude-code/overview),
[MCP in Claude Code](https://docs.anthropic.com/en/docs/claude-code/mcp),
[hooks](https://docs.anthropic.com/en/docs/claude-code/hooks),
[security](https://docs.anthropic.com/en/docs/claude-code/security), and
[enterprise IAM / managed settings](https://docs.anthropic.com/en/docs/claude-code/iam).

`fak` is complementary to those controls. It is not a Claude replacement and not a new
agent harness. It is a kernel-style tool-call boundary:

| Anthropic surface | `fak` value |
|---|---|
| Claude Code local tools | Default-deny policy floor, structured refusal reasons, and audit for proposed tool calls. |
| Claude Code MCP servers | `fak serve --stdio` exposes `fak_adjudicate`, `fak_admit`, and revocation tools to any MCP-capable client. |
| Anthropic Messages tool use | `fak serve` can sit on the Anthropic Messages wire and adjudicate tool calls at the gateway boundary. |
| Enterprise managed settings | `fak` policy files become the reviewable capability floor enterprises can distribute and audit. |
| Security / compliance | Quarantine keeps poisoned or secret-shaped tool results out of model-visible context while preserving evidence. |

## One-day pilot

Run the live Claude Code guard path with an audit journal:

```bash
go build -o fak ./cmd/fak

FAK_AUDIT_JOURNAL="$PWD/fak-audit.jsonl" \
  ./fak guard --log "$PWD/gw.log" --anthropic-oauth -- \
  claude -p "Run: echo hello-from-guard" \
    --allowedTools "Bash(echo:*)" \
    --output-format json
```

Expected evidence:

| Check | Evidence |
|---|---|
| Claude completed useful work | Claude JSON result contains `hello-from-guard` and `is_error:false`. |
| The request crossed the gateway | `gw.log` contains a `POST /v1/messages` row with status `200`. |
| A tool call crossed the floor | `fak-audit.jsonl` contains a `DECIDE` row with `tool:"Bash"` and `verdict:"ALLOW"`. |
| The audit is tamper-evident | Consecutive audit rows carry `prev_hash` and `hash`. |

Then prove the safety floor without a live model:

```bash
./fak preflight \
  --policy examples/dogfood-claude-policy.json \
  --tool Bash \
  --args '{"command":"rm -rf <tmp>/fak-pilot"}'
# expected: DENY, reason POLICY_BLOCK
```

The pilot succeeds only if both facts are true: useful work continues, and the dangerous
call is refused by structure.

## Managed rollout shape

| Layer | Pilot setting | Enterprise shape |
|---|---|---|
| Capability floor | `examples/dogfood-claude-policy.json` | Organization-reviewed policy manifest distributed with managed settings or a vetted plugin bundle. |
| MCP | project `.mcp.json` or `claude mcp add fak -- fak serve --stdio --policy policy.json` | Managed allow-list of MCP servers plus a standard `fak` server entry. |
| Hooks | Optional local hook for audit export or policy checks | Managed hook that ships audit rows to SIEM and blocks unapproved policy drift. |
| Logs | `--log gw.log` and `FAK_AUDIT_JOURNAL=fak-audit.jsonl` | Central log sink plus hash-chain verifier. |
| Rollback | Remove the MCP entry or stop using `fak guard` | Disable managed server/plugin and restore direct Claude Code execution. |

## Residual risks to name up front

- On the subscription/proxy path, Anthropic owns the upstream model KV cache. `fak`
  governs tool calls and result admission, but it does not evict upstream KV state.
- The Anthropic `/v1/messages` streaming path may be more buffered than a direct client
  path depending on the route. Measure time-to-first-token in the pilot.
- The durable audit journal is opt-in unless `FAK_AUDIT_JOURNAL` is required by managed
  rollout.
- The Messages API MCP-connector architecture still needs a dedicated reference diagram
  showing whether `fak` is the MCP server, the gateway in front of a server, or both.

## Decision Anthropic can make

If the pilot passes, Anthropic does not need to adopt `fak` wholesale. The actionable
decision is narrower:

> Bless a governed-tool boundary profile for Claude Code and MCP deployments, with `fak`
> as one reference implementation and conformance target.

That makes the work useful even if Anthropic later implements the same profile in a
native managed-policy layer.
