# fak — front-page overflow (moved off README.md)

These sections used to live on the front page. They were moved here on
2026-06-28 to keep `README.md` focused on what the lowest-common-denominator
reader needs first — performance and cost, the no-key demo, and the one-command
wrap. Nothing here is deprecated; it is narrower-audience or deep-dive material
that earns a link from the front page rather than a place on it.

Each claim still carries the same authority it did on the front page — every
number traces to [BENCHMARK-AUTHORITY.md](../BENCHMARK-AUTHORITY.md), and every
tagged claim to [CLAIMS.md](../CLAIMS.md).

---

## Why Now

The agent stack has moved from demos into operations. Coding agents now have
plugins and background agents. They also have MCP servers, prompt caches, long
sessions, and live tool permissions.

Recent public tooling points in the same direction. MCP reliability, auth, and
observability work is active. Claude Code is shipping MCP and sandbox permission
fixes. Security writing has moved toward runtime tool poisoning alongside prompt
wording.

That changes the useful first screen for `fak`. The value is:

- Make prompt-cache and routing decisions explicit enough to test — and keep the
  cache discount alive across a long session instead of busting it.
- Preserve a traceable, privacy-conscious audit trail of every tool call.
- Put a default-deny floor under the tools your agent already has.
- Keep poisoned tool output and secret-shaped results out of model context.

Relevant external signals: [Claude Code changelog](https://code.claude.com/docs/en/changelog),
[MCP stateless/auth discussion](https://dev.to/alexmercedcoder/ai-weekly-codex-goes-long-mcp-goes-stateless-584d),
and [MCP tool-poisoning/security analysis](https://www.cybedefend.com/en/blog/mcp-security-tool-poisoning).

## Use Cases By Domain

Each row is a starter policy floor: a reviewable allow-list you copy, trim, and run
`fak preflight` against to watch the floor bite. Point your agent at one with
`fak guard --policy examples/<file>` (or `fak serve --policy …` for a gateway). The
full catalogue, with a witness command per floor, is in
[examples/README.md](../examples/README.md).

| Domain | Starter floor | The dangerous action it denies |
|---|---|---|
| Coding agent | [`presets/coding-agent-safe.json`](../examples/presets/coding-agent-safe.json) | force-push, `git add -A`, out-of-tree writes, destructive shell |
| Coding agent (push feature branches) | [`protected-push-floor-policy.json`](../examples/protected-push-floor-policy.json) | a `git_push` whose ref is `main`/`release/*`, by argument value |
| PR-review bot | [`code-review-bot-policy.json`](../examples/code-review-bot-policy.json) | `merge_pull_request`, `git_push`, `workflow_dispatch` |
| Customer support | [`customer-support-readonly-policy.json`](../examples/customer-support-readonly-policy.json) | `refund_payment`, direct account or email action |
| Open-web research | [`research-agent-policy.json`](../examples/research-agent-policy.json) | `send_email`, shell, upload, arbitrary note path |
| Browsing / scraping | [`browser-web-agent-policy.json`](../examples/browser-web-agent-policy.json) | `submit_form`, `execute_script`, a `file:`/`javascript:` URL |
| Email / calendar | [`email-calendar-assistant-policy.json`](../examples/email-calendar-assistant-policy.json) | `send_email`, `forward_email`, `invite_external_guest` |
| Infra / DevOps review | [`devops-dryrun-policy.json`](../examples/devops-dryrun-policy.json) | `terraform_apply`, exec, delete, production deploy |
| Flight booking | [`flight-booking-agent-policy.json`](../examples/flight-booking-agent-policy.json) | `refund_payment`, `export_pnr`, a `$10k+` fare |
| Trading / brokerage | [`finance-trading-agent-policy.json`](../examples/finance-trading-agent-policy.json) | `withdraw_funds`, a six-figure order, a `short` side |
| Clinical / PHI | [`healthcare-phi-policy.json`](../examples/healthcare-phi-policy.json) | `export_patient_data`, `email_phi`, record delete |
| BI / SQL analyst | [`sql-analyst-policy.json`](../examples/sql-analyst-policy.json) | a `DROP`/`INSERT` inside an allowed read-query tool |

Each denied action escalates to that floor's human safe sink instead of failing
silently. Every refusal cites a closed reason code you can assert on, such as
`POLICY_BLOCK`, `OVERSIZE`, or `SECRET_EXFIL`.

## vCache: Provider Cache As A Budget Signal

A provider's prompt cache is not memory you control. You cannot ask it to evict a span
or prove a prefix is resident. You just get telemetry after the request. So `fak vcache`
treats a cache hit as a realized rebate, never something the answer depends on. It
proves or refutes each saving from the provider's own usage counters.

```bash
./fak vcache status
./fak vcache prove
```

Evidence from two live traces:

- Claude Code prefix probe: **13,141.5 input-token equivalents saved** over four
  sibling turns, **4.73%**.
- Codex/OpenAI session telemetry: **9,147,340.8 token equivalents saved** over
  68 token-count events, **85.98%**.

Those are provider-cache accounting proofs on those traces: `fak` supplies the
accounting and control plane. The design contract, the full command set, and the
causality fences are on the
[vCache page](notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md); the Codex/OpenAI
probe is written up in
[the probe note](../experiments/agent-live/VCACHE-CODEX-OPENAI-PROBE-2026-06-25.md).

## Model Routing And Router Fusion

Most routers pick one model for a whole request. `fak route` routes an aspect
instead. The unit can be the request or one tool call. It can also be a
sub-query, reasoning step, or tagged stage.

An ensemble is a first-class plan. Supported reductions include `vote` and
`best_of`. They also include `first`, `concat`, and scalar `all_reduce`.

Try it offline:

```bash
./fak route --aspect tool_call --tool write_file --simulate "approve,deny,deny"
./fak route --aspect step --complexity high
./fak routebench
```

The router is useful because it sits at the same point as the security floor. A
write-shaped call can route to a guard ensemble. An easy read can route to a
cheap model. A tenant-sensitive payload bound for a remote route is denied by
the residency floor.

Read [docs/model-routing.md](model-routing.md) and
[docs/integrations/litellm.md](integrations/litellm.md).

## Three axes of the same kernel

`fak`'s invariants repeat along three dimensions:

- **Scale axis** (vertical): tool call → turn → session → fleet → RSI. How much of
  the stack lives in one address space. The same observe → decide → act → verify
  shape and trust invariant recur at every ring.
- **Depth axis** (downward): CPU reference → CUDA → Vulkan → Metal. Which silicon
  runs the matmul. New backends plug in via registration against the compute HAL.
- **Deployment-substrate axis** (across): IoT → edge → laptop → hyperscaler.
  What *kind of box* and how big. The same workload shape (agent loop proposing
  tool calls) and the same invariants (default-deny, quarantine, bit-exact reuse,
  tamper-evident audit) do not change with the box.

The crossing point is one kernel present at the most scales, depths, and substrate
targets, carrying the same invariants through all of them — the way Linux runs on
the phone in your pocket and the rack training the model on it. See [the
cross-platform spine](explainers/cross-platform-spine.md).

---

The current front page is [README.md](../README.md).
