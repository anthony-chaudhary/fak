---
title: "fak architecture — external system view and trust boundaries"
description: "A current external-builder view of fak's interfaces, request flow, kernel checkpoint, effects boundary, and deeper implementation authorities."
---

# fak architecture: the external system view

**Primary audience:** an external builder deciding where fak belongs in an agent system before reading implementation internals.

`fak` is one host-side agent-kernel binary between an agent or compatible client and the model and tool effects it uses. Calls that enter through a fak-managed interface cross one checkpoint where fak can reuse setup, choose execution, return repeats locally, adjudicate proposed tool calls, and admit results before the next model turn.

```text
agent or compatible client
          |
          v
+---------------- fak-managed interface ----------------+
| managed loop | OpenAI-compatible HTTP | MCP / adapter |
+-------------------------+------------------------------+
                          |
                          v
+------------------- kernel checkpoint ------------------+
| normalize request -> route/reuse -> model proposal     |
|                         |                               |
|                  policy verdict                        |
|                   /          \                         |
|              deny             allow                    |
|              |                   |                     |
|        refusal result       tool effect                |
|                                  |                     |
|                         result admission               |
|                    (accept / redact / quarantine)       |
+-------------------------+------------------------------+
                          |
                          v
                admitted continuation + evidence
```

The checkpoint is a **mediation boundary, not an operating-system sandbox**. It governs traffic routed through the selected fak interface. Tool paths, credentials, or side channels that bypass fak remain outside this boundary; deployment isolation and tool authorization still apply.

## Choose the interface

| Builder job | Interface | fak owns | Start here |
|---|---|---|---|
| Prove the full managed loop deterministically | `fak agent --offline` | Planner fixture, proposal handling, policy verdicts, result admission, and continuation | [Reproduction packet](repro-packet.md) |
| Manage one local agent process | `fak manage` | Process lifecycle plus the configured hook and policy boundary | [One-agent guide](../README.md#manage-one-local-agent-fak-guard) |
| Point compatible clients at a shared endpoint | `fak serve` | HTTP gateway, routing, policy checkpoint, engines, and service evidence | [Server quickstart](fak/server-quickstart.md) |
| Embed or extend the managed runtime | Go ABI, MCP, or an adapter | Only the seams selected by the embedding application | [Runtime ownership and flow](explainers/agent-runtime.md) |

**Default:** begin with `fak agent --offline`. It exercises the end-to-end checkpoint without a credential, live model, service deployment, or accelerator and therefore separates architecture validation from provider quality and infrastructure availability.

**Next action:** run `fak agent --offline` and verify that the task completes while the poisoned result and destructive operation are blocked.

## End-to-end request flow

1. **Enter and normalize.** A managed loop, gateway, MCP transport, or adapter converts the client request into the selected kernel contract.
2. **Route or reuse.** The kernel chooses the configured engine/model path and may serve an eligible repeat from its local fast path. This optimization does not grant a tool capability.
3. **Adjudicate the proposal.** Before a proposed tool effect runs, the capability floor and adjudicator emit an allow or deny verdict with a closed-vocabulary reason.
4. **Execute only an allowed effect.** The host/tool adapter performs the effect. A denial becomes a structured result rather than a hidden tool execution.
5. **Admit the result.** Result-admission and context-management seams accept, transform, or quarantine returned material before it can influence continuation.
6. **Continue with evidence.** The managed loop returns admitted state to the model/client and exposes verdict and runtime evidence through the selected interface.

## Ownership and trust boundaries

| Boundary | fak owns on the managed path | The integrator still owns |
|---|---|---|
| Client edge | Protocol handling and request normalization for the selected interface | Client identity, endpoint exposure, and traffic sent outside fak |
| Model path | Configured routing, engine invocation, repeat handling, and managed context | Provider credentials, model suitability, quotas, and provider availability |
| Tool-call checkpoint | Capability-floor evaluation, adjudication, refusal reason, and pre-effect verdict | Accurate tool declarations and ensuring effects cannot bypass the checkpoint |
| Result boundary | Admission, redaction/quarantine decisions, and continuation input on the managed path | Tool correctness, external data provenance, and downstream consumers outside fak |
| Operations | fak metrics, traces, decision evidence, and process behavior exposed by the chosen mode | Host hardening, secret storage, network policy, persistence, backup, and incident response |

## Lifecycle and support context

This page describes the **current generation's public architecture**. Runtime code and tests are authoritative; mode-specific API, configuration, and deployment guides define their narrower contracts. Experimental, simulated, stubbed, superseded, and dated research pages do not expand this support boundary.

The deterministic offline path is the default architecture witness. Live HTTP serving requires a configured engine/provider and production deployment controls. Accelerator-backed and private-control routes have environment-specific prerequisites and are not implied by the general diagram.

## Deeper implementation authorities

Read these only after choosing the applicable interface:

- [Managed runtime responsibilities and flow](explainers/agent-runtime.md) — who proposes, adjudicates, executes, admits, and continues.
- [Frozen ABI and registry architecture](../ARCHITECTURE.md) — contributor-facing package seams and additive registration contract.
- [Agent integration architecture](fak/agent-integration-architecture.md) — detailed gateway, ABI, policy, extension, and observability reference.
- [Serving architecture and engines](serving/README.md) — engine selection and service composition.
- [API reference](fak/api-reference.md), [server configuration](fak/server-config.md), and [deployment guide](fak/deployment-guide.md) — mode-specific operational contracts.
