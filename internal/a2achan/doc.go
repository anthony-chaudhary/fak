// Package a2achan is the in-kernel agent-to-agent message channel: a generic,
// capability-floored, Ref-backed mailbox that lets one agent hand a value to
// ANOTHER agent — and the SAME default-deny floor that gates a tool call gates
// the message.
//
// # Why it exists (the gap it fills)
//
// fak already had every HALF of a channel but no channel. abi.Ref carries the
// Taint + ShareScope a cross-agent message needs ("never shared more widely than
// its scope; sharing a result shares its taint") but never ROUTES anywhere. The
// async Submit/Reap seam is 1:1 (the same goroutine submits and reaps; there is
// no recipient identity, queue, or notify). internal/session moves per-session
// DRIVE state, internal/recall moves read-only memory images, the vDSO/emitter
// fan-out is read-only cache-invalidation broadcast — none DELIVERS an addressed
// value from agent A to a different agent B. And tools/fleet_agent_link.py (the
// fleet "Agent Link" / A2A HTTP edge, see docs/a2a-value-opportunities.md) is an
// OUT-OF-KERNEL control plane with no kernel adjudication. a2achan is the missing
// in-kernel substrate that edge can later project.
//
// # The model (one key shape, three locales)
//
//   - A ChannelKey is (Locale, ID). Locale ∈ {InKernel, Session, Window}. The SAME
//     Send/Recv serve all three; only the key differs — sessions and windows are
//     the same mailbox addressed by a different ID, not three mechanisms:
//     InKernel : ID is a rendezvous name within ONE process — two concurrent
//     goroutine-agents meet here (the Reap-is-goroutine-safe locale).
//     Session  : ID is a peer's ToolCall.TraceID — the session identity already on
//     every call, so no new identity is invented (cross-session handoff).
//     Window   : ID is a continuation id minted on context-window compaction — the
//     summarizing window SENDs its handoff before teardown; the resuming
//     window RECVs it (an EXPLICIT, adjudicated handoff where today's
//     continuity is implicit via recall page-back).
//   - A Message is a Ref body (its Taint+Scope ride unchanged) plus its From/To
//     addressing and a per-bus monotonic Seq (deterministic delivery order).
//   - The Bus is a process-global, mutex+condvar mailbox: one ordered queue per
//     ChannelKey. Recv blocks (ctx-aware) until a peer Sends — async rendezvous
//     between concurrent in-kernel agents with no new kernel code.
//
// # The capability floor on messages (the fak-native part)
//
// A message is not a memcpy; it is an adjudicated transfer. Send and Recv are
// gated by a REGISTERED kernel adjudicator (a2aGate) + ingress admitter
// (a2aIngress) — the same registries the kernel walks for every tool call, so the
// message floor is first-class in the kernel, not a side library:
//
//   - SEND fails CLOSED. Without the CapA2ASend capability negotiated → deny. A
//     TaintQuarantined body → deny (a poisoned message never leaves). A ScopeAgent
//     body addressed to a channel that is NOT the sender's own (To.ID != From) →
//     deny: a private payload cannot cross the agent boundary. To share, the
//     sender must EXPLICITLY widen the body's Scope to ScopeFleet/ScopeTenant — an
//     auditable act, exactly "never shared more widely than its scope." The
//     fail-closed default (Tainted, ScopeAgent) is therefore undeliverable across
//     agents by construction.
//   - RECV re-screens on ingress through a2aIngress (the dual of the context-MMU's
//     result admission): a TaintQuarantined message is HELD, never admitted to the
//     receiver's context; an admitted message KEEPS its Taint (sharing a result
//     shares its taint), so the receiver cannot re-share it past its Scope.
//   - The refusal cites the closed core vocabulary (abi.ReasonTrustViolation for a
//     scope/taint violation; abi.ReasonDefaultDeny for an un-negotiated cap) — no
//     new reason is minted into the 12-reason set.
//
// # Honest scope (the wedge)
//
// The InKernel locale is REAL and shipped: a process-global, adjudicated,
// Ref-backed mailbox with ctx-aware blocking Recv, proven by the package tests
// (determinism, fail-closed default, taint/scope enforcement, async rendezvous,
// ingress quarantine-hold). The Session and Window locales share the identical
// code path keyed differently and work in-process today; DURABLE cross-process
// persistence (a session-image-backed mailbox, a compaction trigger that mints the
// Window continuation id) is the named next rung — it is NOT claimed here. The
// primitive deliberately does NOT register an abi engine (abi.Engine("") picks the
// lowest-id engine, so a "comm" engine would silently hijack the process default)
// and does NOT depend on the registered-but-unwalked abi Op table (a pure-Op design
// would never execute). It rides the registries the kernel actually walks.
//
// Tier: mechanism (2) — see internal/architest. Imports only abi + stdlib; an
// upward import fails the architest gate. See AGENTS.md and internal/architest for
// the layering contract.
package a2achan
