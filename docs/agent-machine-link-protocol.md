---
title: "Fleet Agent Link: the fak machine-link protocol"
description: "Design for a stdio JSON-RPC control plane that reaches a remote agent checkout via one shared method registry, with A2A and MCP as edge adapters."
---

# Fleet Agent Link Design

Date: 2026-06-19

Fleet needs one missing control-plane layer: a way to ask an agent-capable
checkout on another machine what it is, what it can do, and to run a reviewed
Fleet operation there. Existing repo surfaces solve adjacent problems:

- `tools/claude_agent_chat.py` starts a new Claude Code session from the account
  roster, and `tools/install_agent_command.py` installs the durable
  `fleet-agent` command for that launcher. This is a launcher, not a live remote
  messaging protocol.
- `tools/qwen36_node_packet.py` and `tools/qwen36_node_server.py` package and
  run model-serving nodes over Tailscale/OpenAI-compatible HTTP. They are model
  data-plane helpers, not the agent control plane.
- `fak` already has direct HTTP/MCP syscall compatibility for tool admission and
  quarantine. That is the enforcement boundary for tools, not the machine link
  for reaching a specific laptop/workstation checkout.

The design below keeps those layers separate.

## Decision

Use three layers, sharing one semantic method registry:

```text
Fleet in-memory bus
  -> stdio JSON-RPC machine link
  -> A2A HTTP edge adapter
  -> MCP tool/context adapter where useful
```

The first implementation target should be the stdio JSON-RPC machine link,
carried locally by a subprocess or remotely by SSH/Tailscale SSH. A2A should be
built into Fleet as an edge adapter over the same method registry, not as the
internal "kernel" protocol.

This deliberately invents as little as possible. Fleet Agent Link is not a new
internet protocol. It is a narrow Fleet ABI: JSON-compatible method calls, task
records, and audit events that can be wrapped by existing protocols.

## Shared Method Contract

Every adapter should eventually converge on this internal shape:

```json
{
  "schema": "fleet.call.v1",
  "id": "01J...",
  "caller": {"kind": "operator", "name": "anthony"},
  "target": {"machine": "laptop", "checkout": "fleet-laptop-proof"},
  "method": "laptop.status",
  "params": {"cpu_only": true},
  "policy": {"scope": "read", "requires_confirmation": false}
}
```

Short calls return:

```json
{
  "schema": "fleet.result.v1",
  "id": "01J...",
  "status": "completed",
  "result": {"ok": true}
}
```

Long calls return a task:

```json
{
  "schema": "fleet.result.v1",
  "id": "01J...",
  "status": "accepted",
  "task": {
    "task_id": "task_01J...",
    "state": "working",
    "artifact_paths": []
  }
}
```

The `fleet.call.v1` and `fleet.result.v1` schemas are per-message envelope
schemas: one names a call, the other names either an immediate result or a
task-acceptance result. The protocol-level descriptor below uses
`fleet.agent-link.v1` for the Fleet Agent Link ABI and adapter surface.

Task states should be a deliberately small subset that maps cleanly to A2A:
`submitted`, `working`, `input_required`, `completed`, `failed`, `canceled`,
and `rejected`.

This is the core contract. JSON-RPC carries it over stdio/SSH. A2A maps it to
messages, tasks, and artifacts. MCP can expose selected methods as tools if the
operation is tool-like and sufficiently stateless.

## Shared State Contract

The method contract above is the control-plane half. The state half should use the
same vocabulary as [Shared state ladder](shared-state-ladder.md):

- **live updates** are messages or task events;
- **live shared objects** are named cells with versions and conflict rules;
- **durable state** is a task/audit/artifact record that survives the process or
  context window;
- **disaggregated state** is a digest-verified value whose bytes may live outside
  the local process or engine;
- **user-level collaboration** is a patch/edit with author, base digest, scope,
  durability, and a typed verdict.

That distinction matters for adapters. A2A can project task events and artifacts;
MCP can expose selected tools/resources; a UI can show patches and conflicts. None
of those adapters should create a second definition of what counts as shared state.

## Layer 1: Fleet In-Memory Bus

This is the right meaning of "in-kernel" for Fleet: an in-process dispatch table
with typed handlers, JSON-compatible params/results, and an audit event for every
call. It should not require HTTP, sockets, or a background daemon.

Initial method families:

- `agent.info`: host, repo, toolchain, identity, and advertised methods.
- `agent.ping`: cheap liveness check with monotonic timestamp.
- `laptop.check`: run the laptop proof preflight.
- `laptop.status`: inspect saved laptop proof reports.
- `laptop.verify`: verify saved laptop proof reports.
- `laptop.accept`: run the laptop acceptance lane.
- Later: `session.launch`, `session.status`, `node.serve`, `node.report`,
  `task.cancel`, and `task.tail`.

The in-memory bus should be synchronous for short calls and task-backed for long
calls. Long calls return a task id immediately, then append status/output events
to an audit log.

Tradeoff: this is not interoperable by itself, but it is fast, testable, and
keeps Fleet's core semantics independent from any one network protocol.

Do not call this "A2A in memory" in the implementation. The better phrase is
"Fleet task bus with an A2A projection." The A2A projection can preserve A2A
concepts such as messages, tasks, artifacts, and skills, but the in-process
dispatch path should stay simple enough to unit test without a web server.

## Layer 2: Stdio JSON-RPC Machine Link

JSON-RPC 2.0 is the best local/remote machine ABI because it is a small
request/response envelope and is explicitly transport-agnostic. Fleet can use
the same request object through:

- local subprocess stdin/stdout;
- `ssh`;
- `tailscale ssh`;
- future named pipes or Unix sockets, if needed.

The v1 endpoint is `tools/fleet_agent_link.py serve-once`. It reads one JSON-RPC
request from stdin and writes one JSON-RPC response to stdout. It calls the same
in-memory method registry used by the local CLI path.

Example request:

```json
{"jsonrpc":"2.0","id":"info","method":"agent.info","params":{}}
```

Example remote transport:

```powershell
@'
{"jsonrpc":"2.0","id":"status","method":"laptop.status","params":{"cpu_only":true}}
'@ | tailscale ssh anthony@<laptop-tailnet-name> `
  "cd C:\path\to\fleet-laptop-proof; py -3 tools\fleet_agent_link.py serve-once"
```

Tradeoff: stdio JSON-RPC does not provide discovery, streaming, auth policy, or
task semantics by itself. Fleet should add those in its method registry and task
log, then expose them through adapters.

The first remote call does not need a resident process on the laptop. SSH starts
the endpoint, the endpoint handles one request, and the process exits. That is
slower than a daemon but easier to trust, easier to update with `git fetch`, and
less likely to leave stale privileged code running on a personal machine.

Implemented helper commands:

```powershell
py -3 tools\fleet_agent_link.py request agent.info
py -3 tools\fleet_agent_link.py call-local agent.ping
py -3 tools\fleet_agent_link.py remote-command `
  --cwd 'C:\path\to\fleet-laptop-proof' `
  --shell powershell
py -3 tools\fleet_agent_link.py a2a-card `
  --url https://fleet.example.com/a2a `
  --scope read
```

Local laptop status call:

```powershell
py -3 tools\fleet_agent_link.py request laptop.status --params '{"cpu_only":true}' |
  py -3 tools\fleet_agent_link.py serve-once
```

## Layer 3: A2A Edge Adapter

A2A should be built into Fleet at the boundary where independent agents need to
discover Fleet and delegate work to it. That means:

- publish an Agent Card for each Fleet agent surface;
- advertise selected Fleet methods as A2A skills;
- map A2A `SendMessage` into one method call or task creation;
- map `GetTask`, `ListTasks`, `CancelTask`, and streaming updates onto Fleet's
  task store and audit log;
- keep auth and tenant routing in HTTP headers/Agent Card metadata, not inside
  arbitrary method params.

An A2A adapter for the laptop proof should advertise a skill like
`fak_laptop_proof`. The implementation should call the internal
`laptop.accept`/`laptop.status` methods rather than shelling out from the A2A
handler.

Tradeoff: A2A gives interoperability, discovery, task lifecycle, and streaming,
but it also brings HTTP service management, version negotiation, Agent Card
validation, auth policy, and more surface area. That is worthwhile for external
agents, but too heavy for the first laptop-to-checkout control path.

The A2A edge should be optional per machine. A laptop proof checkout should not
need to run an HTTP listener just to answer a private Fleet operator. A shared
Fleet coordinator or hosted relay can run A2A when there is an actual
inter-agent discovery requirement.

For the product wedge and implementation ladder, see
[`docs/a2a-value-opportunities.md`](a2a-value-opportunities.md). The short
version: Agent Card projection and linting now exist; build task ids and audit
next, then the HTTP edge.

## MCP Position

MCP remains important for Fleet, but at a different boundary. MCP is the right
protocol when Fleet wants to expose tools, resources, prompts, or context to an
LLM host. It should not become the primary machine-to-machine agent control
plane.

Practical rule:

- Use MCP when a model host asks "what tools/resources can I use?"
- Use Fleet Agent Link when an operator or peer agent asks "what can that machine
  run, and can it run this reviewed Fleet operation?"
- Use A2A when an independent agent asks "what agent are you, what skills do you
  have, and can I delegate a task to you?"

## Security Model

The protocol must not expose a generic `exec` method. Every method maps to a
reviewed Fleet entry point with explicit params.

Required controls:

- allowlist methods by name;
- validate params before dispatch;
- record `caller`, `target`, `method`, params hash, start time, end time,
  exit/status, and artifact paths;
- separate short results from artifact/log paths;
- cap stdout/stderr tails;
- default to read/status methods before run/accept methods;
- make long-running methods cancellable through a task id;
- do not let A2A or MCP adapters bypass the same method policy.

Tailscale SSH is a good first transport for a single-user laptop because it uses
tailnet identity and avoids opening a public port. Its access model is still
coarser than per-method authorization, so Fleet must enforce method-level policy
inside the endpoint.

Run/accept methods should be split into two policy classes:

- `read`: `agent.info`, `agent.ping`, `laptop.status`, report inspection.
- `act`: `laptop.check`, `laptop.verify`, `laptop.accept`, session launch, model
  server start/stop.

The first implementation can allow both classes for the local operator, but the
audit schema should name the class from day one so A2A/MCP adapters can enforce
stricter policy later.

## Versioning

Use a Fleet-level protocol schema separate from A2A and MCP versions:

```json
{
  "schema": "fleet.agent-link.v1",
  "methods": ["agent.info", "laptop.status"],
  "adapters": {
    "stdio_jsonrpc": true,
    "a2a": "agent-card-projection-only",
    "a2a_http": "planned-edge-adapter",
    "mcp": false
  }
}
```

Compatibility promise for v1:

- JSON-RPC request/response framing remains stable.
- Method params/results are additive by default.
- Breaking method changes require a new method name or `schema` version.
- A2A and MCP adapters can evolve independently as long as they preserve the
  internal method contract.

## Implementation Boundary

Implemented v1:

1. In-memory method registry in Python for the current Fleet tools.
2. `tools/fleet_agent_link.py serve-once` JSON-RPC stdio adapter.
3. `request`, `call-local`, and `remote-command` helper commands.
4. A2A Agent Card projection and `a2a-lint` registry drift checks.
5. Hermetic tests for request validation, method dispatch, card projection, and laptop runner
   argument construction.
6. Laptop usage docs with local and `tailscale ssh` examples.

Next boundary:

1. Persistent task ids and an audit log for long-running methods.
2. Cancellation and task-tail methods.
3. A2A adapter after the method registry has task ids and an audit log.

This order prevents the common failure mode: building an HTTP/A2A facade first
and then discovering that the underlying machine actions are still ad hoc shell
commands.

## Rejected Alternatives

- Full A2A first: too much web-service surface before Fleet has a stable method
  registry and task log. Good second adapter, poor first primitive.
- MCP first: useful for exposing Fleet capabilities as tools, but it frames the
  caller as an LLM host using tools. The laptop control path is about a machine
  endpoint running reviewed Fleet operations.
- REST-only HTTP: easy to debug, but would invent routing, errors, discovery,
  and task semantics that JSON-RPC/A2A already cover better.
- gRPC-first: strong schemas and streaming, but heavier cross-platform bootstrap
  and worse fit for shell/SSH stdin piping.
- WebSocket-first: useful for streaming, but unnecessary for single request
  status and acceptance calls. Add it only if the task event stream needs it.
- Message broker first: NATS/Redis/MQTT-style brokers solve durable fanout, not
  the initial "reach this laptop checkout now" problem.
- Always-on local daemon: faster after startup, but introduces lifecycle,
  upgrade, and privilege questions before the one-shot transport is proven.

## Non-Goals

- No arbitrary remote shell.
- No OS-kernel networking protocol.
- No always-on laptop daemon as the first requirement.
- No claim that A2A replaces MCP.
- No claim that MCP replaces A2A.
- No model-serving changes; Qwen/OpenAI-compatible serving remains a separate
  data-plane concern.

## Sources

- JSON-RPC 2.0: https://www.jsonrpc.org/specification
- A2A latest specification: https://a2a-protocol.org/latest/specification/
- Fleet A2A value opportunities: ./a2a-value-opportunities.md
- MCP specification: https://modelcontextprotocol.io/specification/
- Tailscale SSH: https://tailscale.com/docs/features/tailscale-ssh
