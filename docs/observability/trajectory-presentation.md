# Trajectory interpretation and presentation

A trajectory is evidence. A trajectory **view** is a derived product for an audience. They need separate contracts.

Today `internal/trajectory` records a compact `Turn` row and can ingest native chat exports. That is a useful canonical spine, but a growing agent product needs more than “render the JSONL.” It needs an interpretation pipeline that can preserve source truth while producing several intentionally different views.

## Pipeline

```text
source bytes
  -> source adapter
  -> canonical events
  -> semantic enrichment
  -> audience projection
  -> renderer
  -> interaction events (written back to the trajectory)
```

### 1. Source adapter

Decode a provider or harness format without erasing source identity. Preserve:

- source type, source session/event ID, source timestamp, and ordering key;
- the raw-event digest and adapter version;
- unknown fields or an explicit loss report;
- parse warnings rather than guessed semantics.

Adapters answer “what did this source emit?” They do not decide what an operator should see.

### 2. Canonical events

Represent facts at their natural granularity, rather than forcing every fact into a completed turn. The minimum event families are:

- run/session lifecycle;
- user, assistant, system, and tool messages, including streaming deltas;
- tool call proposed, started, progressed, completed, or failed;
- approval requested, decided, expired, or bypassed;
- state snapshot/delta and checkpoint/fork;
- artifact created/changed/referenced;
- model route, usage, cache, latency, and cost observations;
- policy, hook, error, retry, cancellation, and intervention.

Each canonical event carries stable identity, parent/causal links, source provenance, visibility class, and payload schema version. The existing `trajectory.Turn` remains a valid compact projection; it should not become the only parse target.

### 3. Semantic enrichment

Derived records explain a trajectory without impersonating source facts. Examples include phase boundaries, retry clusters, stalls, decisions, outcomes, goal progress, and artifact lineage. Every derivation needs:

- derivation kind and version;
- input event IDs;
- confidence or deterministic rule identity;
- creation time and producer;
- invalidation behavior when inputs change.

A model-generated label such as “debugging phase” is an annotation, not a rewrite of the underlying calls.

### 4. Audience projection

A projection is a declarative view specification over canonical and derived records:

```json
{
  "schema": "fak-trajectory-view/1alpha1",
  "audience": "operator",
  "include": ["message.user", "tool.*", "approval.*", "artifact.*", "error.*"],
  "collapse": ["tool.progress", "message.assistant.delta"],
  "group_by": ["retry_cluster", "phase"],
  "redaction_policy": "operator-local",
  "density": "compact",
  "live_controls": ["approve", "deny", "steer", "pause"]
}
```

The view compiler returns selected records plus a transform receipt: source trajectory digest, view-spec digest, redaction policy/version, derivation versions, omitted counts by reason, and renderer-independent ordering. This makes a screenshot or report reproducible and prevents “the UI hid it” from becoming an unauditable state.

Useful built-in audiences differ materially:

| Audience | Default emphasis | Default suppression |
|---|---|---|
| End user | request, visible progress, result, requested approvals | internal routing, repetitive reads, private diagnostics |
| Operator | active phase, blockers, approvals, spend, retries, artifacts | token deltas and routine successes |
| Developer | full event ordering, payloads, causal links, adapter warnings | secrets only |
| Auditor | policy decisions, authority changes, evidence digests, exports | presentation-only animation/progress noise |
| Manager | goals, milestones, outcomes, elapsed time, intervention points | raw tool payloads and implementation chatter |

### 5. Renderer

Terminal, web, Slack, Markdown, JSON, and accessibility renderers consume the same projection. Theme, layout, density, keybindings, animation, notification routing, and screen-reader labels live here. They never change event meaning or redaction policy.

### 6. Interaction loop

Approve, deny, steer, pause, retry, and fork are not ephemeral UI gestures. The renderer submits a typed control action; the runtime records the resulting intervention event. A later view can therefore explain both what the agent did and how a person changed its course.

## Invariants

1. **Raw evidence is append-only.** Parsing or rendering never edits source bytes.
2. **No silent loss.** Unknown/unparsed material is retained or counted in a loss report.
3. **Meaning precedes visibility.** Parse and enrich before audience filtering.
4. **Redaction precedes rendering.** A renderer never receives fields its audience may not see.
5. **Views are reproducible.** Every output identifies trajectory, transform, policy, and renderer versions.
6. **Controls round-trip.** User interventions become canonical events.
7. **Partial order survives.** Concurrent events keep causal links; a renderer may linearize only with an explicit ordering rule.
8. **Derived claims stay distinguishable.** Confidence-bearing interpretation never masquerades as emitted fact.

## What this changes in fak

The immediate architectural seam is additive:

- keep `trajectory.Turn` as the stable compact record and compatibility projection;
- add canonical event envelopes beside it rather than widening one row indefinitely;
- let current Slack/status/TUI surfaces consume named projections instead of each inventing filters;
- make export receipts identify omitted/redacted material and all transform versions;
- register source adapters independently so Codex, Claude, OpenAI export, AG-UI, and future formats can report fidelity.

This document is the design spine, not a claim that those runtime pieces already ship. The structured gap/status record is [`agent-customization-index.json`](../research/agent-customization-index.json): source adapters and semantic events are partial; policy-aware parsing redaction and first-class audience/selection views are absent.

## Acceptance witness for a first implementation

A minimal implementation is real when one fixture containing a user message, streamed assistant text, a tool call, an approval, a tool result, and a human steer can be:

1. ingested with source IDs and a raw digest;
2. projected into operator and end-user views with observably different event sets;
3. redacted before either renderer receives secret material;
4. rendered twice with identical projection receipts;
5. round-tripped so the steer appears as a canonical intervention event;
6. exported back to the compact `Turn` shape without changing existing readers.
