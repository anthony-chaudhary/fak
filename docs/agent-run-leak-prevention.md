---
title: "AgentRun leak-prevention boundary"
description: "Normative public spec for AgentRun identity, parent-child propagation, PolicyDigest pinning, ToolCallID linkage, inherited capability envelopes, unmanaged-spawn default deny, and the witness suite for agent-run leak prevention."
---

# AgentRun leak-prevention boundary

Status: public contract spec for issue #2355. This page defines the vocabulary
that implementation leaves must share before they add broker code, launch
adapters, policy gates, or red-team fixtures.

The goal is not a detector for leaked processes. The goal is a boundary every
agent, subagent, tool call, spawn attempt, and descendant process must cross
before it can produce effects or return bytes to model context.

## Non-goals

- Do not replace `internal/toolproc` or `internal/toolprocgate`.
- Do not create a second process table.
- Do not implement SpawnBroker, process-tree containment, egress controls, or
  red-team fixtures in this spec.
- Do not add a Python tool or move code.

## Terms

The keywords MUST, MUST NOT, SHOULD, and MAY are normative.

`AgentRun` is the authority snapshot for one running agent loop. A root
AgentRun is minted at a request/session boundary. A child AgentRun is minted
only through an allowed `SpawnAttempt`.

`SpawnAttempt` is the pre-exec record for a proposed child agent, subagent,
background job, remote job, or process-like helper. It is not the child itself:
it is the admission event that either denies the launch or authorizes one child
AgentRun with a bounded inherited envelope.

`ToolCallID` is the non-empty tool-call identity that binds launch, lifecycle,
PID binding, and result admission. In the current ABI this is
`abi.ToolCall.TraceID`; on the Anthropic wire the `tool_use_id` maps into the
same logical ToolCallID before result admission.

`PolicyDigest` is the content identity of the effective policy snapshot that
authorized the run. The digest MUST be over canonical policy data, not over an
operator's prose label. Until a canonical serializer is implemented, adapters
MUST treat an absent or unverified PolicyDigest as not capable of widening a
child run.

## AgentRun Record

An implementation MAY store the record in Go structs rather than JSON, but the
fields below are the contract. Unknown additive fields are allowed; missing
required fields are not.

```json
{
  "schema": "fak.agent-run.v1",
  "agent_run_id": "ar_child_001",
  "parent_agent_run_id": "ar_parent_001",
  "root_agent_run_id": "ar_root_001",
  "session_id": "sess_001",
  "tenant_id": "tenant_001",
  "spawn_attempt_id": "spawn_001",
  "launch_tool_call_id": "toolu_001",
  "policy_digest": "sha256:0123456789abcdef...",
  "capability_envelope_digest": "sha256:abcdef0123456789...",
  "state": "running"
}
```

Required identity fields:

| Field | Requirement |
|---|---|
| `agent_run_id` | Non-empty, unique in the process table horizon. It is the owner key used for descendant tool lifetimes. |
| `parent_agent_run_id` | Empty only for a root AgentRun. Non-root runs MUST name the parent that authorized them. |
| `root_agent_run_id` | Non-empty. Root runs point to themselves; descendants preserve the root unchanged. |
| `session_id` | Non-empty session or lease boundary used for operator control and orphan handling. |
| `tenant_id` | Non-empty when the deployment has a tenant boundary; otherwise an explicit local/default tenant token. |
| `spawn_attempt_id` | Empty only for a root AgentRun. Non-root runs MUST name the SpawnAttempt that authorized them. |
| `launch_tool_call_id` | Empty only when the root run was created by an external request rather than a tool call. Non-root runs MUST carry the ToolCallID that launched them. |
| `policy_digest` | PolicyDigest of the effective policy snapshot for this run. A child MAY use the same digest as its parent or a stricter child digest; it MUST NOT use a digest that grants more authority. |
| `capability_envelope_digest` | Digest of the effective inherited envelope described below. It is the child-run authority set, not a descriptive label. |
| `state` | Closed set: `running`, `ended`, `revoked`. |

The AgentRunID is not a substitute for ToolCallID. AgentRunID owns a running
agent loop; ToolCallID owns one proposed effect or process-like launch inside
that run.

## SpawnAttempt Record

Every child run or process-like helper MUST be represented by a SpawnAttempt
before exec, before remote dispatch, and before a background id is accepted as
live.

```json
{
  "schema": "fak.spawn-attempt.v1",
  "spawn_attempt_id": "spawn_001",
  "parent_agent_run_id": "ar_parent_001",
  "root_agent_run_id": "ar_root_001",
  "tool_call_id": "toolu_001",
  "requested_kind": "subagent",
  "requested_policy_digest": "sha256:requested...",
  "granted_policy_digest": "sha256:granted...",
  "granted_capability_envelope_digest": "sha256:envelope...",
  "decision": "allow",
  "reason": "",
  "child_agent_run_id": "ar_child_001"
}
```

Required SpawnAttempt fields:

| Field | Requirement |
|---|---|
| `spawn_attempt_id` | Non-empty, unique for the parent AgentRun horizon. |
| `parent_agent_run_id` | Non-empty and currently running. A spawn from an ended or unknown parent is refused. |
| `root_agent_run_id` | Must equal the parent's root. |
| `tool_call_id` | Non-empty ToolCallID for the call that requested the spawn. |
| `requested_kind` | Closed set owned by the broker implementation, for example `subagent`, `background_tool`, `remote_job`, `mcp_tool`. |
| `requested_policy_digest` | The requested child policy, if the caller supplied one. Empty means "same as parent, possibly narrowed by requested envelope." |
| `granted_policy_digest` | The PolicyDigest actually granted. It must be parent-equal or stricter. |
| `granted_capability_envelope_digest` | Digest of the final inherited capability envelope. |
| `decision` | Closed set: `allow`, `deny`. There is no observe-only launch. |
| `reason` | Closed refusal token when `decision=deny`; empty on allow. |
| `child_agent_run_id` | Non-empty only when `decision=allow` and the broker minted the child. |

The SpawnAttempt is the only place a launch can become a child AgentRun. A later
"we spawned one" report is a self-report, not authority.

## Propagation Rules

1. The root request/session boundary mints exactly one root AgentRun with a
   PolicyDigest and capability envelope.
2. A tool call that can launch a child run, background job, remote job, or
   process-like helper MUST have a non-empty ToolCallID before admission.
3. The spawn broker computes the child envelope as an intersection of the
   parent's effective envelope and the requested child envelope.
4. The broker MUST deny when it cannot prove the child envelope is equal to or
   narrower than the parent envelope.
5. The broker MUST deny when the parent AgentRun is unknown, ended, revoked, or
   missing a verified PolicyDigest.
6. A child AgentRun MUST preserve `root_agent_run_id`, `tenant_id`, and
   `session_id` unless a stricter tenant/session boundary is explicitly created
   by the broker.
7. A child AgentRun MUST carry `parent_agent_run_id`, `spawn_attempt_id`, and
   `launch_tool_call_id`. A descendant without that chain is unmanaged and is
   denied by default.
8. Policy hot reload MUST NOT widen an already-running AgentRun. It MAY narrow
   live descendants by revalidation or revocation, but a child cannot claim new
   authority because the global process later loaded a wider policy.
9. Result admission MUST use the launch ToolCallID, not a child-supplied label.
   A child cannot relabel a result to escape revocation.

## Inherited Capability Envelope

An inherited capability envelope is the authority set a child may use. It is
monotone: each child can keep or drop authority, never add it.

The envelope includes at least these axes:

| Axis | Inheritance rule |
|---|---|
| Tool allow set | Child allow set MUST be a subset of the parent allow set after arg rules are applied. |
| Explicit denies | Child deny set MUST include all parent denies and MAY add more. |
| Arg rules | Child rules MUST be at least as restrictive for every inherited tool. Unknown rule semantics fail closed. |
| Runtime envelope | Child `deadline_ms` MUST be no greater than the parent deadline when the parent declares one. Child `heartbeat_every_ms` MUST be no greater than the parent cadence when the parent declares one. A child may add a positive deadline or heartbeat where the parent omitted one because that narrows runtime authority. |
| Rate limits | Child limits MUST fit inside remaining parent budget and MUST NOT reset parent counters. |
| Egress | Child egress allow lists MUST be subsets; deny lists union. |
| Isolation | Child backend/trust placement MUST be equal or stronger than the parent placement. Unknown placement fails closed. |
| Result admission | Child MUST inherit the registered result-admission chain. It cannot disable `toolprocgate`, secret gates, normalization gates, quarantine gates, or IFC gates. |
| Share scope | Child result/data scope MUST be no wider than the parent scope unless the broker proves an explicit declassification path. |

The `policy.Runtime.ToolRuntime.EnvelopeFor(tool)` table is the current source
for per-tool runtime grants. It is one axis of the inherited envelope, not a
separate permission system.

## Mapping To Existing Mechanisms

AgentRun is a contract layer above the existing tool process table.

`internal/toolprocgate.Supervisor` remains the live lifecycle engine:

- `Supervisor.Spawn(callID, tool, session, deadlineMS, heartbeatEveryMS, nowMS, cancel)`
  MUST be called with `callID = ToolCallID` and `session = AgentRunID` for
  process-like tool work.
- `Supervisor.BindPID(ToolCallID, pid)` binds the launched OS process tree to
  the ToolCallID that admitted it.
- `Supervisor.SessionEnd(AgentRunID, nowMS)` marks the AgentRun orphan boundary.
- `Supervisor.Tick(nowMS)` folds `internal/toolproc` and acts on kill/reap
  advice by cancelling, revoking, and recording the kill.
- `toolprocgate.Kill(ToolCallID, reason)` arms result admission for the same
  ToolCallID.
- The rank-2 `toolprocgate.Gate` quarantines a late completion with
  `TOOL_RESULT_AFTER_KILL`. The payload is dropped, not held for page-in.

This mapping is deliberately one-to-many:

| AgentRun concept | Current mechanism |
|---|---|
| AgentRun owner | `toolproc.Event.Session` / `Supervisor.SessionEnd` owner key |
| ToolCallID | `abi.ToolCall.TraceID`, `toolproc.Event.CallID`, revocation table key |
| SpawnAttempt runtime grant | `policy.Runtime.ToolRuntime.EnvelopeFor` plus broker envelope intersection |
| Descendant PID | `Supervisor.BindPID(ToolCallID, pid)` |
| Late result after revocation | `toolprocgate.Gate` -> `TOOL_RESULT_AFTER_KILL` |

## Unmanaged Spawn Default Deny

Any surface that can create an agent, subagent, shell, remote job, background
job, MCP-progressing job, or process-like helper MUST route through
SpawnAttempt before launch.

If the surface cannot name a live parent AgentRun and a non-empty ToolCallID,
the default verdict is deny. A detector that finds an unmanaged child after it
already started MUST NOT convert it into an allowed child by observation. It
must either:

- bind it to the nearest known parent only long enough to kill/reap it and
  revoke its ToolCallID, or
- quarantine/drop its result path if the process cannot be safely reached.

The admission refusal MAY cite the existing `POLICY_BLOCK` reason until a child
implementation leaf adds a more specific closed reason. A completion that
arrives after revocation is already covered by `TOOL_RESULT_AFTER_KILL`.

## Surfaces That Must Mint Or Propagate

| Surface | Obligation |
|---|---|
| Gateway proposed tool calls | Mint or preserve ToolCallID; attach parent AgentRun; run SpawnAttempt for launch-capable tools. |
| Anthropic `tool_use` / `tool_result` | Preserve `tool_use_id` as ToolCallID through result admission. |
| MCP server/client adapters | Map request, progress, cancellation, and completion to ToolCallID plus AgentRun owner. |
| `fak guard` hook adapter | Stamp hook-observed launches with parent AgentRun and ToolCallID before they enter the toolproc journal. |
| Native agent loop | Mint root AgentRun at loop start and child AgentRuns through SpawnAttempt only. |
| Remote or background launch adapters | Report SpawnAttempt before dispatch, then bind returned pid/job id to ToolCallID. |

## Witness Suite Obligations

Implementation leaves that claim this boundary is enforced MUST ship witnesses
for these properties:

| Obligation | Minimum witness |
|---|---|
| AgentRun identity | A root and child record carry all required fields; missing parent, root, policy_digest, or launch ToolCallID is refused. |
| Parent-child propagation | Child preserves root/session/tenant, points to parent, and names the SpawnAttempt and ToolCallID that authorized it. |
| PolicyDigest pinning | Same policy yields stable PolicyDigest; policy changes change the digest; a child cannot use a wider digest than its parent. |
| Envelope monotonicity | A child can drop tools, tighten arg rules, shorten deadlines, add heartbeats, narrow egress, and strengthen isolation; attempts to widen any axis are denied. |
| Unmanaged spawn deny | A launch path with no live parent AgentRun or no ToolCallID is denied before exec. A discovered unmanaged descendant is killed/reaped or its result path is quarantined. |
| toolprocgate bridge | Spawn -> BindPID -> deadline/orphan Tick -> Kill -> late completion is quarantined by `toolprocgate.Gate` with `TOOL_RESULT_AFTER_KILL`. |
| Result relabel resistance | A child cannot change ToolCallID on result re-entry to escape a parent revocation. |
| Adapter coverage | Gateway, guard hooks, MCP, and native loop adapters either mint/propagate AgentRun or explicitly deny unmanaged launch. |
| Red-team leak case | A denied parent capability cannot be recovered by spawning a subagent, background shell, remote job, or late result. |

The standing source witness for this spec is:

```bash
rg -n "AgentRun|SpawnAttempt|PolicyDigest|TOOL_RESULT_AFTER_KILL|toolprocgate" docs internal/toolprocgate internal/toolproc internal/policy
```

The standing acceptance gate for the existing enforcement spine is:

```bash
go test ./internal/toolproc ./internal/toolprocgate ./internal/policy
```

On the Windows dev host, native `go test` may be blocked by OS application
control as described in `AGENTS.md`; in that case the report must say the test
gate was not run on this host and name the blocked command.

## Honest Scope

This spec defines the AgentRun boundary and inherited capability contract. It
does not claim live spawn containment is implemented. The current shipped
pieces it relies on are the policy runtime envelope, `internal/toolproc`, and
`internal/toolprocgate`; child leaves must wire the broker and adapters without
inventing incompatible IDs or a second process table.
