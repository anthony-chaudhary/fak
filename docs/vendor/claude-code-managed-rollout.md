---
title: "Claude Code managed rollout - fak as a governed MCP and audit boundary"
description: "Enterprise rollout guide for using fak with Claude Code managed settings, MCP controls, hooks, audit logging, and rollback."
---

# Claude Code managed rollout - fak as a governed MCP and audit boundary

**Audience:** Anthropic Claude Code product/SDK, enterprise admins, security teams.
**Purpose:** turn the Anthropic pilot pitch into a managed rollout shape: central MCP
configuration, managed-policy restrictions, optional hooks, audit export, and rollback.

## Fit with Claude Code controls

Claude Code already gives enterprises managed settings, MCP server controls, hook
controls, and security policies. The rollout below uses those surfaces rather than
asking users to bypass them.

Current Anthropic docs relevant to this rollout:

- Settings: managed settings can restrict MCP servers, hooks, permission rules, plugin
  sources, and customization surfaces.
  <https://docs.anthropic.com/en/docs/claude-code/settings>
- MCP: organizations can deploy a fixed server set, restrict servers with
  `allowedMcpServers` and `deniedMcpServers`, and use a managed MCP configuration.
  <https://docs.anthropic.com/en/docs/claude-code/mcp>
- Hooks: managed policy settings and plugin hooks can be controlled centrally; admins
  can allow only managed hooks.
  <https://docs.anthropic.com/en/docs/claude-code/hooks>
- Security: Claude Code treats MCP servers as part of the security boundary and
  recommends trusted servers and permission configuration.
  <https://docs.anthropic.com/en/docs/claude-code/security>

## Rollout model

| Layer | Admin-controlled setting | fak role |
|---|---|---|
| MCP server | Managed `fak` MCP server entry running `fak serve --stdio --policy <policy>` | Exposes `fak_adjudicate`, `fak_admit`, revocation, and context-change tools. |
| MCP allowlist | `allowedMcpServers` / `allowManagedMcpServersOnly` | Ensures only the vetted `fak` server identity is loaded. |
| Hooks | Managed hook or plugin hook | Ships audit rows, verifies hash chains, or blocks unapproved local policy drift. |
| Permissions | Managed permission rules | Keeps Claude Code's native filesystem/shell boundary in place; fak governs tool-call intent and result admission. |
| Audit | `FAK_AUDIT_JOURNAL` and gateway log sink | Produces per-decision evidence: trace id, tool, verdict, reason, hash chain. |
| Rollback | Disable managed `fak` server or remove policy bundle | Claude Code falls back to native tool and MCP controls. |

## Pilot configuration

Start with a project-level proof. This keeps the pilot reversible:

```json
{
  "mcpServers": {
    "fak": {
      "command": "fak",
      "args": ["serve", "--stdio", "--policy", "examples/dogfood-claude-policy.json"],
      "env": {
        "FAK_AUDIT_JOURNAL": "./fak-audit.jsonl"
      }
    }
  }
}
```

Validate the same server outside Claude Code first:

```bash
python examples/mcp/verify.py
```

Then run a live `fak guard` turn when a Claude Code seat is available:

```bash
FAK_AUDIT_JOURNAL="$PWD/fak-audit.jsonl" \
  fak guard --log "$PWD/gw.log" --anthropic-oauth -- \
  claude -p "Run: echo hello-from-guard" \
    --allowedTools "Bash(echo:*)" \
    --output-format json
```

The pilot passes only when the useful command completes and a dangerous command is
denied by policy:

```bash
fak preflight \
  --policy examples/dogfood-claude-policy.json \
  --tool Bash \
  --args '{"command":"rm -rf <tmp>/fak-pilot"}'
# expected: DENY, reason POLICY_BLOCK
```

## Managed policy shape

The exact managed-settings distribution mechanism depends on the enterprise deployment
channel. The policy intent is:

```json
{
  "allowManagedMcpServersOnly": true,
  "allowedMcpServers": [
    { "serverName": "fak" }
  ],
  "allowManagedHooksOnly": true,
  "allowManagedPermissionRulesOnly": true,
  "strictPluginOnlyCustomization": ["hooks", "mcp"]
}
```

This keeps the pilot narrow:

- users can keep using Claude Code normally;
- the vetted `fak` MCP server remains available;
- user/project MCP and hook sources cannot silently replace the governed boundary;
- Claude Code's native permission rules still govern local effects.

## Audit contract

Each fak decision row should be exportable to the enterprise log sink with:

| Field | Why it matters |
|---|---|
| `trace_id` | Correlates the Claude turn, MCP call, gateway decision, and audit row. |
| `tool` | Names the proposed effect. |
| `verdict` / `reason` | Shows the machine refusal or admission reason. |
| `by` | Names the deciding layer. |
| `args_digest` | Avoids logging raw secrets while preserving evidence. |
| `prev_hash` / `hash` | Makes missing or edited audit rows detectable. |

Keep raw tool outputs out of the central log unless the organization explicitly approves
that data flow. The digest is the default evidence.

## Rollback

1. Disable or remove the managed `fak` MCP server entry.
2. Stop setting `FAK_AUDIT_JOURNAL` and gateway log export variables.
3. Remove `fak guard` from any launcher wrappers.
4. Keep Claude Code's native managed settings, permission rules, and MCP allowlist in
   place.

Rollback should not require changing user prompts or code repositories. That is the
point of placing `fak` at the tool-governance boundary.

## Residual risks

- A proxy/subscription `fak guard -- claude` run governs tool calls and result admission
  but does not own Anthropic's upstream KV cache.
- Audit rows are durable only when `FAK_AUDIT_JOURNAL` or an equivalent sink is required.
- A managed MCP entry gives Claude access to governance tools; it does not automatically
  force every other MCP server to call `fak_adjudicate`. Use allowlists and wrapper
  patterns for that.
- Pilot commands should measure latency and streaming behavior on the actual deployment
  route before broad rollout.
