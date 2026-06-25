---
title: "Agent Tool Governance Gateway profile"
description: "A vendor-neutral profile for governing agent tool calls: pre-execution verdicts, result admission, quarantine, revocation, and audit fields, with fak as one reference implementation."
---

# Agent Tool Governance Gateway profile

**Status:** draft profile, 2026-06-25.
**Goal:** make the `fak` adoption wedge standard-shaped: any vendor, MCP server, IDE
agent, SDK guardrail, or gateway can implement this profile without adopting `fak`'s
internal API.

## Goals

- Define the minimum wire behavior a governed agent-tool boundary should provide.
- Let clients treat refusals and quarantines as successful adjudications, not transport
  failures.
- Preserve useful work: a denied unsafe call should not collapse the whole task.
- Make audit, quarantine, and revocation observable enough for enterprise rollout.
- Keep the profile implementable over MCP, OpenAI-compatible clients, Anthropic Messages
  gateways, SDK guardrails, or direct HTTP.

## Non-goals

- This is not a replacement for MCP, OpenAI Responses, Anthropic Messages, or an SDK's
  native guardrail API.
- This does not standardize a model API or a tool schema language.
- This does not require `fak`; `fak` is one reference implementation.

## Actors

| Actor | Role |
|---|---|
| Agent/client | Proposes a tool call or holds a tool result. |
| Governance gateway | Adjudicates the call/result against policy and state. |
| Tool executor | Runs the allowed or transformed tool call. It may be the client, the gateway, or an MCP server. |
| Audit consumer | Reads verdict rows, counters, quarantine references, and revocations. |

## Required operations

### 1. Pre-execution adjudication

Before a tool call executes, the gateway must be able to return a verdict over:

```json
{
  "tool": "git_push",
  "arguments": {},
  "read_only": false,
  "trace_id": "atgg-deny-1",
  "witness": "git:HEAD"
}
```

Required verdict kinds:

| Kind | Meaning |
|---|---|
| `ALLOW` | The call may execute as proposed. |
| `DENY` | The call must not execute. A machine reason is required. |
| `TRANSFORM` | The call may execute only with repaired canonical arguments. |
| `REQUIRE_WITNESS` | The call is blocked until an external witness, approval, or read-back is supplied. |

### 2. Result admission

After a tool returns, the gateway must be able to screen the result before it enters
model-visible context:

```json
{
  "tool": "web_fetch",
  "result": {
    "content": "ATGG_FAKE_SECRET_MARKER for conformance only"
  },
  "trace_id": "atgg-quarantine-1",
  "witness": "https://example.invalid/page"
}
```

Required result-admission verdict kinds:

| Kind | Meaning |
|---|---|
| `ALLOW` | The result may enter model-visible context. |
| `QUARANTINE` | The result is held out of model-visible context; a recovery/audit reference is preserved. |
| `DENY` | The result is refused and no model-visible substitute is required. |

### 3. Revocation

The gateway must be able to revoke trust in an external witness and evict dependent
cache/context entries:

```json
{ "witness": "sha256:1111111111111111111111111111111111111111111111111111111111111111" }
```

The response must state that the witness is revoked and expose enough counters for an
operator to see whether entries were evicted locally.

## Required verdict fields

| Field | Required | Meaning |
|---|---|---|
| `kind` | yes | `ALLOW`, `DENY`, `TRANSFORM`, `REQUIRE_WITNESS`, `QUARANTINE`, or a stricter implementation-specific value. |
| `reason` | for refusals/quarantine | Stable machine token such as `POLICY_BLOCK`, `SELF_MODIFY`, `SECRET_EXFIL`, `MALFORMED`, or a vendor-specific token. |
| `disposition` | for refusals | Loop behavior: `RETRYABLE`, `WAIT`, `ESCALATE`, or `TERMINAL`. |
| `by` | should | Which adjudicator or policy layer decided. |
| `detail` | optional | Bounded, non-secret diagnostic metadata. |

## Audit row

Every decision should emit or expose an audit row with:

| Field | Required | Notes |
|---|---|---|
| `trace_id` | yes | Correlates the model turn, tool call, result admission, and audit row. |
| `timestamp` | yes | Wall-clock or monotonic time from the gateway. |
| `operation` | yes | `adjudicate`, `syscall`, `admit`, or `revoke`. |
| `tool` | for call/result operations | Tool name as proposed. |
| `verdict.kind` | yes | Decision kind. |
| `verdict.reason` | for refusals/quarantine | Stable reason token. |
| `policy_id` | should | Policy digest, version, or deployment id. |
| `args_digest` | should | Digest of arguments, not raw secret-bearing args. |
| `result_digest` | for result operations | Digest of result bytes when present. |
| `prev_hash` / `hash` | should | Hash chain for tamper-evident logs. |

## Conformance fixtures

The draft fixtures live under [`fixtures/`](fixtures/):

| Fixture | Proves |
|---|---|
| [`allow-read-only.json`](fixtures/allow-read-only.json) | A safe read-only call is allowed. |
| [`deny-policy-block.json`](fixtures/deny-policy-block.json) | A dangerous call is denied with a closed reason. |
| [`transform-canonical-args.json`](fixtures/transform-canonical-args.json) | Malformed but repairable arguments produce a transformed canonical call. |
| [`quarantine-result.json`](fixtures/quarantine-result.json) | A poisoned/secret-marked result is held out of context. |
| [`revoke-witness.json`](fixtures/revoke-witness.json) | A witness can be revoked and dependent state evicted. |

A conforming implementation does not need to use these exact tool names. It must provide
semantically equivalent cases and produce the required verdict fields.

## fak mapping

| Profile operation | fak surface |
|---|---|
| Pre-execution adjudication | `fak_adjudicate` over MCP or `POST /v1/fak/adjudicate` |
| Adjudicate and execute | `fak_syscall` over MCP or `POST /v1/fak/syscall` |
| Result admission | `fak_admit` over MCP or `POST /v1/fak/admit` |
| Revocation | `fak_revoke` over MCP or `POST /v1/fak/revoke` |
| Audit | Gateway logs, `/metrics`, `/debug/vars`, and `FAK_AUDIT_JOURNAL` |

For the current `fak` MCP tool-result wire, see
[`docs/mcp-tool-result.md`](../mcp-tool-result.md).
