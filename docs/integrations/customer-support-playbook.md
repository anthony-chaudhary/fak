---
title: "Customer-support vertical adoption playbook"
description: "Threat model, rollout, and runbook for adopting the read-only customer-support policy template — a vertically focused guide for support teams using AI agents."
---

# Customer-support vertical adoption playbook

This playbook is for adopting the read-only customer-support policy template (`examples/customer-support-readonly-policy.json`) — a vertical-specific capability floor designed for AI agents that handle customer support inquiries but must not take irreversible account actions or exfiltrate customer data.

## The threat model

### What the attacker controls

In the customer-support vertical, the threat model assumes an attacker can control everything the agent reads:

- **Customer-provided prompts** — adversarial text embedded in support requests
- **Retrieved documents** — poisoned knowledge base entries
- **Tool results** — malicious data returned from external systems

The agent (the language model) is the untrusted component. The question is not "did the model get fooled?" but "can a fooled model still pull an irreversible lever or exfiltrate customer data?"

### What the capability floor bounds

The customer-support template enforces three structural guarantees:

| Guarantee | How it's enforced |
|---|---|
| **No irreversible account actions** | High-impact tools (`delete_account`, `refund_payment`, `transfer_funds`, `rotate_credentials`, `send_customer_email`) are explicitly denied with `POLICY_BLOCK` — refused regardless of prompt context. |
| **No customer data exfiltration** | `export_customer_data` is blocked; outbound data must flow through `transfer_to_human_agents` (the only safe sink), which escalates to a human. |
| **Secret hygiene** | Fields matching `password`, `secret`, `api_key`, `token`, `authorization`, `ssn` are redacted (`[REDACTED]`) before any tool call, preventing secrets from being echoed back or logged. |

The floor does NOT trust the model to "refuse" these actions — a tool that is not allow-listed is refused structurally, before the model's output is ever evaluated.

### What is NOT in scope

- **The model's internal reasoning** — the floor gates tool calls, not thoughts.
- **Result content analysis** — detecting "bad" content in tool results is the detector layer; the floor only gates which tools run.
- **Argument-level injection within allow-listed tools** — if `read_customer_record` is allowed, the model can request any record ID. Data access controls (RBAC, PII masking) are your application's responsibility, not the floor's.

---

## Rollout checklist

### Phase 0 — Verify the template works offline

Before integrating with your model, prove the floor behaves as expected with no network, no key, no GPU:

```bash
# Clone and build fak
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak

# Validate the manifest parses
./fak policy --check examples/customer-support-readonly-policy.json
# OK  examples/customer-support-readonly-policy.json

# Test dangerous calls are denied
./fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool refund_payment \
  --args '{}'
# → DENY (POLICY_BLOCK)

./fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool export_customer_data \
  --args '{}'
# → DENY (POLICY_BLOCK)

# Test allowed calls pass
./fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool search_kb \
  --args '{}'
# → ALLOW

./fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool create_support_ticket \
  --args '{"body":"test"}'
# → ALLOW
```

These four commands prove: (1) the manifest is valid, (2) dangerous tools are blocked, (3) read-only work passes, and (4) escalation paths work.

### Phase 1 — Adapt the template to your tool names

Copy the template and map it to your actual tools:

```bash
cp examples/customer-support-readonly-policy.json my-support-policy.json
# Edit my-support-policy.json — rename tools, adjust deny list, customize redact_fields
./fak policy --check my-support-policy.json
```

**Mapping the template to your tools:**

| Template tool | Your equivalent (example) |
|---|---|
| `read_customer_record` | `get_customer_profile`, `lookup_user` |
| `read_corp_kb` | `search_docs`, `query_knowledge_base` |
| `create_support_ticket` | `open_case`, `log_interaction` |
| `transfer_to_human_agents` | `escalate_to_tier2`, `handoff_to_human` |
| `refund_payment` (DENY) | `process_refund`, `issue_credit` |
| `delete_account` (DENY) | `close_account`, `terminate_user` |

If your tool names don't match the `allow_prefix` list (`read_`, `get_`, `search_`, `list_`, `lookup_`, `find_`), add them explicitly to `allow`.

### Phase 2 — Add argument rules for your constraints

Use `arg_rules` to bound what allow-listed tools can do:

```json
{
  "arg_rules": [
    {
      "tool": "create_support_ticket",
      "arg": "body",
      "max_bytes": 4000,
      "reason": "OVERSIZE"
    },
    {
      "tool": "read_customer_record",
      "arg": "customer_id",
      "deny_regex": "\\d{16,}",
      "reason": "POLICY_BLOCK"
    }
  ]
}
```

Example: deny lookups that look like full credit card numbers (16+ digits) even though `read_customer_record` is allowed.

### Phase 3 — Deploy the gateway

#### Option A — Front a cloud model (no code change to your agent)

```bash
export FAK_TOKEN="$(openssl rand -hex 32)"

./fak serve \
  --addr 0.0.0.0:8080 \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --api-key-env ANTHROPIC_API_KEY \
  --policy my-support-policy.json \
  --require-key-env FAK_TOKEN
```

Point your agent's base URL at `http://127.0.0.1:8080`. The agent sees no difference — tool calls are silently adjudicated, allowed calls proxy upstream, dangerous calls are refused.

#### Option B — Use `fak guard` with Claude Code (local dev)

```bash
./fak guard --policy my-support-policy.json -- claude
```

Every tool call Claude proposes crosses the customer-support floor before execution.

#### Option C — Manual MCP server (ask before executing)

Create `.mcp.json` at your project root:

```json
{
  "mcpServers": {
    "fak-support": {
      "command": "fak",
      "args": ["serve", "--stdio", "--policy", "my-support-policy.json"]
    }
  }
}
```

Your agent calls `fak_adjudicate` before running a tool it executes itself.

### Phase 4 — Verify health and observability

```bash
curl -s http://127.0.0.1:8080/healthz
# {"ok":true,"model":"claude-sonnet-4-20250514","engine":"inkernel"}

curl -s http://127.0.0.1:8080/metrics | grep -E 'fak_adjudicate|fak_verdict'
```

Enable the audit journal for a tamper-evident record:

```bash
export FAK_AUDIT_JOURNAL=/var/log/fak-support-audit.jsonl
./fak serve ... --policy my-support-policy.json
```

---

## Runbook: common tasks

### Add a new read-only tool

1. Add the tool name to `allow` (if not covered by `allow_prefix`).
2. If the tool takes customer IDs, consider an `arg_rules` regex to block patterns that look like sensitive data (e.g., full credit card numbers).
3. Validate: `fak policy --check my-support-policy.json`
4. Test: `fak preflight --policy my-support-policy.json --tool YOUR_TOOL --args '{}'`

### Add a new irreversible tool (e.g., `reset_password`)

**Do NOT add it to `allow`.** Instead, add it to the safe sink escalation path:

```json
{
  "safe_sinks": ["transfer_to_human_agents", "reset_password_via_human"]
}
```

The agent can call `reset_password_via_human`, which your application routes to a human operator. The model never runs the dangerous tool directly.

### Respond to a denial incident

1. Find the refused call in the audit journal:
   ```bash
   grep '"verdict":"DENY"' /var/log/fak-support-audit.jsonl | tail -1
   ```
2. Check the reason (`POLICY_BLOCK`, `SELF_MODIFY`, `RATE_LIMITED`, etc.).
3. If the refusal was correct (dangerous tool), no action — the floor worked.
4. If the refusal was incorrect (legitimate tool blocked), add the tool to `allow` or adjust `arg_rules`, then hot-reload:
   ```bash
   curl -X POST http://127.0.0.1:8080/v1/fak/policy/reload \
     -H "Authorization: Bearer $FAK_TOKEN"
   ```

### Add rate limiting for cost control

Add a `rate_limit` block to your policy:

```json
{
  "rate_limit": {
    "max_calls": 1000,
    "max_cost": 500000,
    "key": "trace",
    "retry_after_ms": 60000
  }
}
```

`max_cost` is cumulative argument bytes (≈ token-equivalent). Over-limit calls return `RATE_LIMITED` with an advisory `retry_after`.

---

## Cross-references

- [General adopter playbook](adopter-playbook.md) — the full three-shape integration guide (front a model, MCP server, CI embed).
- [Policy schema](../../POLICY.md) — the `fak-policy/v1` manifest documentation.
- [Claude Code integration](claude.md) — full Claude-specific guide (denial table, cloud providers, observability).
- [Security and threat model](../fak/security.md) — the two-gate containment architecture and the general threat model.