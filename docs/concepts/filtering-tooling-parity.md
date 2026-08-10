---
title: "How fak fits with filters, guardrails, gateways, MCP, and tool routers"
description: "Decide whether fak replaces or complements prompt filters, guardrails, policy engines, gateways, MCP hosts, tool routers, caching, and audit systems."
---

# Is fak a filter, guardrail, proxy, policy engine, gateway, MCP host, or tool router?

**Short answer:** fak is an agent-kernel gateway with a structural policy and result-admission
checkpoint. It has some filter, guardrail, proxy, routing, and MCP middleware capabilities,
but those labels describe individual seams, not interchangeable products. Keep a specialist
filter or framework guardrail when it owns a surface fak does not inspect; put fak at the tool
and model boundary where it can enforce capabilities, quarantine tool results, route calls,
reuse cache state, and leave a witnessed decision trail.

**Reader:** a builder deciding whether to replace a component, integrate it with fak, or do
neither. **Lifecycle:** current. This page distinguishes shipped behavior from limited or
planned behavior; it does not turn an integration seam into a blanket replacement claim.

## Should I replace my existing filter or integrate it?

Use the narrowest answer that matches the row below:

- **Replace** a component only when the shipped fak row covers the same wire, decision, and
  failure semantics you require.
- **Integrate** when the systems inspect different surfaces—for example, a provider content
  filter checks generated prose while fak gates tool capabilities and inbound tool results.
- **Do neither** when you do not need a governed tool/model boundary or when fak cannot sit on
  the wire your client uses.

“Parity” below means fak implements the named concern at its own checkpoint. “Complement” means
fak and the named system protect or optimize different checkpoints. “Non-goal” means fak should
not be selected as a replacement for that concern.

## Capability matrix: filter, guardrail, policy engine, gateway, MCP, and routing parity

| Concern | Relationship | Where the systems meet | Current maturity | Evidence and replacement decision |
|---|---|---|---|---|
| **Prompt and output filters** | **Complement; narrow parity for tool-boundary content.** fak screens proposed tool calls and admits or quarantines inbound `tool_result` content. It is not a general classifier for every user prompt or every token of assistant prose. | Run a prompt/output safety service in the client or provider path; run the same request through `fak serve` for tool-call and tool-result enforcement. | **Shipped** for tool-call structural checks and inbound-result quarantine; **non-goal** as a universal prose moderation replacement. | [`TestChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend`](../../internal/gateway/gateway_test.go) and [`TestResearchEgressResultInjectionQuarantined`](../../internal/adjudicator/decide_egress_normgate_test.go). **Integrate** unless tool-boundary filtering is your entire requirement. |
| **Structural policy / capability guardrail** | **Parity at the tool checkpoint.** The compiled policy floor can allow, deny, repair, or require confirmation from tool name, arguments, capability, and runtime context; unknown tools fail closed. | Supply a policy to `fak preflight`, `fak serve`, or `fak guard`; the decision happens before tool execution. | **Shipped.** | [Policy in the kernel](../explainers/policy-in-the-kernel.md), [`TestEmptyPolicyDefaultDeny`](../../internal/adjudicator/adjudicator_test.go), and the [60-second proof](../repro-packet.md). **Replace** a tool-call allow/deny shim if these semantics cover its full contract; otherwise integrate. |
| **Model adjudication** | **Complement.** fak can add model-assisted adjudication behind structural checks, but a model cannot override a hard structural refusal. This is a decision rung, not a substitute for provider moderation or application-specific review. | Configure the adjudicator at the kernel decision seam while retaining upstream/downstream safety classifiers as separate principals. | **Shipped but optional**; the no-model structural path remains the baseline. | [Adjudication demo](../../examples/adjudication-demo/README.md) and [`TestComplainDoesNotDowngradeHardRefusal`](../../internal/adjudicator/complain_test.go). **Integrate** when probabilistic review adds value; do not replace hard policy with it. |
| **MCP and tool middleware** | **Parity for interception; complement for hosting.** fak can sit between an MCP client and tools, apply policy, and expose kernel tools. It is not a catalog, marketplace, or business-tool implementation. | Connect the client to the MCP endpoint/config while the real tool servers remain behind or alongside fak. | **Shipped** for the documented MCP path; tool-specific coverage varies. | [Runnable MCP example](../../examples/mcp/README.md) and [MCP integration guide](../integrations/mcp.md). **Replace** a pass-through policy shim; **integrate** with the tool servers themselves. |
| **Model proxy, router, or API gateway** | **Parity for supported model wires and routing; complement for fleet infrastructure.** fak proxies OpenAI/Anthropic-compatible traffic and can choose an engine/model. It is not a general HTTP ingress, service mesh, billing system, or provider control plane. | Point the client base URL at `fak serve`; fak then calls the configured upstream engine. Keep infrastructure gateways outside that process. | **Shipped** on the wires listed by the compatibility matrix; support is wire-specific. | [Integration compatibility matrix](../integrations/compatibility-matrix.md) and [routing explainer](../model-routing.md). **Replace** a model-only pass-through router when the listed wire/features suffice; otherwise chain them. |
| **Caching and context handling** | **Parity for agent-loop cache/context work; complement for provider caches.** fak reuses shared setup, preserves provider cache-friendly prefixes, serves eligible repeated reads locally, and manages context pressure. It does not replace the provider's KV cache or an application's durable data cache. | The kernel sits before the provider; provider prompt/KV caching remains enabled upstream. Tool-result filtering occurs before admitted content joins that context. | **Shipped**, with engine- and path-specific levers. | [Cache architecture](../explainers/cache.md), [context management](../explainers/context.md), and [`TestChatProxyAdmitsInboundToolResultBeforeUpstream`](../../internal/gateway/admit_test.go). **Integrate** with provider caching; replace only duplicated agent-loop cache glue. |
| **Audit, receipts, and witness behavior** | **Parity for kernel decisions; complement for enterprise audit systems.** fak records policy/result decisions and can bind claims to independent witnesses. It is not a SIEM, compliance certification, or organization-wide event store. | Export or retain the hash-chained journal and proof artifacts; forward them to your existing audit pipeline if required. | **Shipped** for kernel journals and repository witnesses; external retention/export remains deployment-specific. | [Verification ladder](../notes/verification-ladder-epics.md) and [`TestProxyResultQuarantineJoinsOriginCallSeq`](../../internal/gateway/result_quarantine_forensics_test.go). **Integrate** with compliance storage and alerting. |

## Integration pattern 1: fak in front of an existing tool or filter stack

Use this when an MCP server, tool router, DLP filter, or command sandbox already owns the actual
tool implementation.

```text
agent/client -> fak guard or fak serve -> existing filter/router -> tool server
                    |                         |
                    | capability decision     | domain-specific filtering/execution
                    + result admission <------+
```

1. Point the agent's supported model or MCP wire at fak using the [integration chooser](../integrations/README.md).
2. Keep the existing stack's domain policy—for example, SQL row filtering or a DLP detector—enabled.
3. Give fak the least-capability tool manifest. It rejects structurally forbidden calls before
   the downstream stack receives them.
4. Return the downstream result through the same conversation wire. fak admits, redacts, or
   quarantines it before upstream model delivery; a quarantined result is not silently converted
   to an allow.
5. Correlate fak's decision journal with the downstream system's execution receipt rather than
   treating either log as proof of the other system's action.

The [MCP example](../../examples/mcp/README.md) is the runnable connection proof. The gateway
result tests above are the admission proof. This pattern removes a redundant pass-through
allowlist, but it does **not** remove tool-specific validation or execution sandboxing.

## Integration pattern 2: fak alongside provider or framework guardrails

Use this when a provider or agent framework already moderates prompts and generated text.

```text
user -> framework prompt guard -> agent -> fak -> model provider output guard
                                      |
                                      +-> tool policy -> tool -> result admission
```

1. Leave provider/framework prompt and output moderation enabled.
2. Repoint the supported model base URL to `fak serve`, or launch the client with `fak guard`.
3. Put deterministic capability rules in the fak policy rather than asking a prose classifier to
   infer whether a tool is allowed.
4. Treat refusal classes independently: a provider content refusal, a fak `POLICY_BLOCK`, and a
   quarantined tool result are different outcomes and should stay distinguishable in the UI and
   audit trail.
5. Fail closed if either required layer is unavailable. Do not retry around fak directly to the
   provider, and do not reinterpret a provider refusal as permission to execute a tool locally.

The [customer-support read-only policy](../../examples/customer-support-readonly-policy.json)
and [reproduction packet](../repro-packet.md) provide the no-key structural proof. Add your
provider/framework guardrail's own test corpus for the prose surfaces it owns.

## What are the trust boundaries and failure semantics?

### The model is not the policy authority

The model proposes calls. The compiled floor decides whether they can proceed. Optional model
adjudication can narrow or explain a decision but cannot talk past a hard structural refusal.

### A tool result is untrusted input

A successful tool process does not make its output trustworthy. The result crosses a separate
admission boundary before it reaches the model context. Suspicious output is quarantined or
redacted with a typed note; it is not treated as ordinary content merely because the call was
allowed.

### The provider remains a separate principal

fak does not attest that a provider moderated output, retained no data, honored a geographic
boundary, or executed a requested model. Keep provider controls and receipts when those claims
matter.

### Bypass is failure, not graceful degradation

If policy/result admission is required, a direct fallback from the client to the upstream model
changes the security contract. Configure clients so a dead kernel fails closed. Likewise, an MCP
or tool-server failure remains a tool failure; fak must not invent a successful result.

### Evidence is scoped

A fak journal proves the kernel decision it records. A tool receipt proves the tool-side effect it
records. A provider receipt proves the provider event it records. Join those records explicitly;
none is a blanket witness for the others.

## Which choice should I make?

| Your requirement | Choice |
|---|---|
| Deterministic allow/deny/repair for tool calls on a supported wire | **Replace** a simpler pass-through policy shim with fak. |
| General prompt toxicity, PII, copyright, or assistant-prose moderation | **Integrate** a specialist/provider filter; fak is not its replacement. |
| Existing MCP or business-tool server plus a shared capability floor | **Integrate**: keep the server, put fak at the interception boundary. |
| Model routing plus cache/context management on a supported wire | **Replace or integrate** depending on whether your current gateway also owns general ingress, billing, or fleet operations. |
| Enterprise compliance archive, SIEM, or certification | **Integrate** fak's scoped journal; do not replace the archive. |
| No tool calls, no supported proxy/MCP wire, and no need for kernel cache/context controls | **Do neither** until there is a boundary fak can actually govern. |

For client-specific wiring, continue to the [integration chooser](../integrations/README.md). For
the exact support label and wire limitations, use the [compatibility matrix](../integrations/compatibility-matrix.md).
