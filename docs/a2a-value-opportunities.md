---
title: "fak + A2A: the evidence layer behind the front door"
description: "How fak projects its capability floor, quarantine, and task evidence onto the Agent2Agent protocol surface instead of reinventing it."
---

# Fleet A2A Value Opportunities

Date: 2026-06-19

Status: strategy memo plus first wedge. This is not a claim that Fleet has
shipped an A2A HTTP edge adapter yet. The current shipped control-plane artifact
is `tools/fleet_agent_link.py`: a stdio JSON-RPC method registry that can now
project an A2A Agent Card and lint it against the reviewed Fleet registry.

## Thesis

A2A makes cross-agent discovery, task lifecycle, streaming, and multi-tenant
routing a commodity interface. Fleet's value should not be "we also speak A2A."
Fleet's value is the evidence layer behind an A2A surface:

- advertised skills come from a reviewed method registry, not prose;
- callers and tenants get skill-scoped authorization before work starts;
- every task transition has an audit row and artifact provenance;
- poisoned or policy-sensitive artifacts are contained before they enter model
  context;
- private workstations can participate without becoming public HTTP daemons;
- read-heavy multi-agent work can be deduplicated behind a standard task API.

The concise product line is:

> A2A gives agents a standard front door. Fleet gives that front door a
> capability floor, quarantine, task evidence, and workload reuse.

## Current Fleet Base

The repo already has the right boundary shape:

- `docs/agent-machine-link-protocol.md` says Fleet should use an internal method
  registry, a stdio JSON-RPC machine link, and an optional A2A edge adapter.
- `tools/fleet_agent_link.py` exposes `agent.info`, `agent.ping`,
  `protocol.manifest`, and reviewed `laptop.*` methods with `read` / `act`
  policy scopes and no generic `exec`.
- `fak` already carries the strongest differentiated mechanisms: default-deny
  tool policy, closed reason codes, quarantine, replayable evidence, MCP/OpenAI
  gateway work, and read-heavy fleet reuse projections.

That means A2A work should project existing Fleet semantics into A2A, not invent
an A2A-first backend. The first concrete step is now in-tree: `a2a-card` emits a
card from the registry, and `a2a-lint` checks that advertised skills map back to
registered methods and scopes.

## A2A Facts That Matter

Checked against the official A2A materials on 2026-06-19:

- The latest released specification is A2A `1.0.0`.
- A2A is explicitly for independent, potentially opaque agents to discover
  capabilities, exchange messages, manage tasks, and collaborate without sharing
  internal memory, tools, or resource access.
- The key public object is the Agent Card: identity, capabilities, skills,
  service endpoint, authentication requirements, supported interfaces, and
  optional signatures.
- Core operations include `SendMessage`, `SendStreamingMessage`, `GetTask`,
  `ListTasks`, `CancelTask`, `SubscribeToTask`, push notification configuration,
  and `GetExtendedAgentCard`.
- A2A v1.0 leans into enterprise deployment: HTTPS/TLS, standard HTTP auth,
  per-skill authorization, least privilege, trace context, audit trails, and API
  management.
- Multi-tenancy can be expressed by URL path, authentication header routing, or a
  request `tenant` field echoed from the selected Agent Card interface.
- A2A and MCP are complementary: A2A is agent-to-agent collaboration; MCP is
  agent-to-tool/resource integration. Fleet should use both at the right
  boundary.
- The A2A community roadmap names validation as an ecosystem need, including
  Inspector and TCK work. That leaves room for Fleet to contribute policy and
  security evidence fixtures rather than another generic SDK.

## Ranked Opportunities

| Rank | Opportunity | Fleet-specific value | A2A surface | First proof |
|---:|---|---|---|---|
| 1 | Policy-filtered Agent Cards | Skills are generated from the Fleet method registry and policy scopes, so the public card cannot advertise unreviewed power. | Agent Card `skills`, `security`, `securitySchemes`, signed cards, extended card | Add a command that emits an A2A Agent Card from `protocol.manifest`; test that unknown methods and generic exec never appear. |
| 2 | A2A task admission and audit flight recorder | `SendMessage` becomes a reviewed call envelope or task creation, and every state transition has caller, tenant, method, params hash, artifact paths, and reason codes. | `SendMessage`, `GetTask`, `ListTasks`, `CancelTask`, streaming updates | Add persistent task ids plus an append-only audit log before any HTTP server. |
| 3 | Skill-scoped authorization gateway | Fleet can enforce `read` / `act` scopes, tenant routing, local operator exceptions, and policy refusals before a task reaches a machine. | Agent Card security declarations, per-skill authorization, tenant field | Fixture: a caller with read scope can call `laptop.status` but receives a closed denial for `laptop.accept`. |
| 4 | Quarantine-aware artifacts | A2A standardizes artifacts and parts; Fleet can attach provenance and keep suspicious content out of downstream context. | Message/Artifact `Part`s, task status, extensions or metadata | Replay an A2A trace where a poisoned artifact is withheld or scrubbed and the task result names the quarantine reason. |
| 5 | Private machine relay | A shared A2A coordinator can advertise skills while reaching a laptop over one-shot stdio JSON-RPC via SSH/Tailscale SSH. The laptop does not need a public listener. | Agent Card endpoint for coordinator; internal Fleet Agent Link to target machine | Coordinator skill `fak_laptop_proof` calls `laptop.status` over the existing `serve-once` path. |
| 6 | A2A/MCP policy preservation bridge | When A2A agents are exposed as MCP resources/tools, or MCP-backed capabilities are advertised as A2A skills, Fleet can preserve deny/quarantine metadata across the boundary. | A2A Agent Card plus MCP tool/resource descriptions | One fixture where the same policy refusal is visible through both A2A and MCP shapes. |
| 7 | Agent Card and TCK security linter | The ecosystem will need validation beyond schema shape: signed-card verification, scope/card drift, tenant echo rules, overbroad skill descriptions, and missing audit promises. | Agent Card, supported interfaces, signatures, version negotiation | `tools/fleet_agent_link.py a2a-lint` reads a card and emits pass/fail rows with Fleet-specific policy checks. |
| 8 | Fleet scheduler behind A2A tasks | A2A clients see ordinary task state while Fleet deduplicates repeated read-heavy work, shares witnessed world state, and routes to the right machine or model backend. | Task lifecycle, streaming, artifacts | Run two A2A task submissions over a shared Fleet working set and report avoided duplicated calls/artifacts. |
| 9 | A2A incident replay packet | Security teams can submit an A2A message/task/artifact trace and get allow/deny/quarantine rows, raw JSON, and residuals. | Messages, tasks, artifacts, Agent Cards | Extend the current replay/adoption packet with an A2A trace fixture. |
| 10 | Human review and secondary authorization | `act` methods can become explicit approval or secondary-credential states instead of hidden prompt negotiation. | `input_required` / authorization-required flows, task status updates | `laptop.accept` starts as `input_required` unless the caller has an approved act scope. |

## Best First Wedge

Build the Agent Card projection and linter before the A2A HTTP adapter. The
projection/linter piece is now implemented; the task store and HTTP edge are
still next.

This is the smallest useful artifact because it turns the current Fleet method
registry into something an A2A client can inspect, without introducing daemon
lifecycle, public routing, or streaming. It also creates an immediate regression
gate: if a method is not in the registry and policy map, it cannot appear as an
A2A skill.

Concrete deliverables and current status:

1. Done: add a projection function that converts the Fleet registry into an A2A
   Agent Card structure.
2. Done: include method scope, manifest digest, confirmation requirement, and
   evidence pointer in skill metadata. Use a normal metadata field first; reserve
   a formal A2A extension until there is a concrete interoperability need.
3. Done: add tests that prove `read` and `act` skills are labeled,
   `generic_exec` never appears, and `supportedInterfaces[]` carries protocol
   version and optional tenant consistently.
4. Done: add a linter that checks advertised skill missing from registry, scope
   mismatch, unsigned card when signing is required, missing auth scheme, and
   tenant/interface inconsistency.
5. Next: add persistent task ids and audit.
6. Later: add the HTTP+JSON or JSON-RPC A2A server wrapper over the same
   registry and task store.

## Implementation Ladder

### Step 0: Current state

`tools/fleet_agent_link.py` exposes a manifest, local JSON-RPC dispatch, an A2A
Agent Card projection, and an A2A card linter. A2A HTTP remains planned.

### Step 1: Card projection

Generate a card for a coordinator, not for every laptop checkout:

```powershell
py -3 tools\fleet_agent_link.py a2a-card `
  --url https://fleet.example.com/a2a `
  --scope read `
  --tenant workspace-a |
  py -3 tools\fleet_agent_link.py a2a-lint
```

```json
{
  "name": "Fleet Coordinator",
  "description": "Policy-filtered Fleet control-plane agent.",
  "supportedInterfaces": [
    {
      "url": "https://fleet.example.com/a2a",
      "protocolBinding": "HTTP+JSON",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": false, "pushNotifications": false},
  "skills": [
    {
      "id": "fleet_laptop_status",
      "name": "Laptop Status",
      "description": "Run tools/fak_laptop_test.py status against saved proof reports.",
      "tags": ["fleet", "agent-link", "laptop", "read"],
      "inputModes": ["application/json"],
      "outputModes": ["application/json"],
      "metadata": {
        "fleet_method": "laptop.status",
        "fleet_policy_scope": "read",
        "fleet_requires_confirmation": false,
        "fleet_manifest_schema": "fleet.agent-link.v1",
        "fleet_evidence_method": "protocol.manifest"
      }
    }
  ]
}
```

### Step 2: Task store and audit log

Add stable task ids before network service work. The minimum task event should
include:

- task id and optional A2A context id;
- caller identity from transport/auth headers;
- tenant/routing key;
- Fleet method and policy scope;
- params hash, not raw secrets;
- state transition;
- artifact paths and quarantine/provenance metadata;
- denial reason when rejected.

### Step 3: Minimal A2A edge

Implement only these mappings first:

| A2A operation | Fleet mapping |
|---|---|
| `SendMessage` | parse a single skill invocation, validate params, dispatch short method or create task |
| `GetTask` | read task store by id |
| `ListTasks` | list tasks by context/caller/tenant |
| `CancelTask` | mark cancellable tasks canceled |
| `GetExtendedAgentCard` | return the authenticated/private card when allowed |

Streaming and push notifications should wait until the task store has useful
state changes to stream.

### Step 4: Signed card and tenant policy

Add signing only once the generated card is stable. Signing proves the card bytes
were issued by Fleet; it does not prove the caller is authorized. Authorization
still comes from transport identity plus Fleet policy.

Tenant rules should be boring:

- URL path routes major agent surfaces;
- auth credentials identify caller/org;
- A2A `tenant` selects a workspace only when the selected Agent Card interface
  declared one;
- policy decides which skills the caller can use in that tenant.

### Step 5: Replay and compatibility packet

Ship one no-key packet:

1. a sample Agent Card;
2. a policy manifest;
3. an allowed `laptop.status` task;
4. a denied `laptop.accept` task;
5. a quarantined artifact trace;
6. raw JSON and a rerun command.

That packet is more valuable to A2A maintainers, security teams, and framework
authors than a broad claim that Fleet "supports A2A."

## Non-Wedges

- A generic A2A SDK is not differentiated. The A2A project already has
  multi-language SDK momentum.
- A public HTTP listener on every laptop is the wrong first primitive. Use a
  coordinator plus one-shot Fleet Agent Link for private machines.
- Agent Card signatures are not authorization. They are card-integrity evidence.
- Skill descriptions are not policy. They must be projections of reviewed methods
  and scopes.
- A2A should not bypass `fak` or Fleet policy. The adapter is another entry point
  into the same registry, task store, and quarantine boundary.
- Do not make A2A replace MCP. A2A is the peer-agent protocol; MCP remains the
  model/tool/context boundary.

## Source Links

- A2A latest specification: https://a2a-protocol.org/latest/specification/
- A2A v1.0 announcement: https://a2a-protocol.org/latest/announcing-1.0/
- A2A enterprise features: https://a2a-protocol.org/latest/topics/enterprise-ready/
- A2A multi-tenancy: https://a2a-protocol.org/latest/topics/multi-tenancy/
- A2A and MCP comparison: https://a2a-protocol.org/latest/topics/a2a-and-mcp/
- A2A roadmap: https://a2a-protocol.org/latest/roadmap/
