---
title: "In-flight agents understanding each other: A2A claim-verification SOTA vs. fak's no-claimed-field invariant (2026-07-07)"
description: "A July-2026 field scan of agent-to-agent status/coordination protocols, dogfood-witnessed against what fak already ships, isolating one concrete gap: fak's verified-status discipline (no `claimed` field) holds in-kernel but is dropped at the A2A HTTP edge — the exact surface a foreign in-flight agent reads to understand ours."
---

# In-flight agents understanding each other

**Date:** 2026-07-07 · **Kind:** research note + one parked, seam-grounded issue body.
**House rule (this note):** every claim about *fak* is read directly from HEAD with a
file cite. Every claim about the *field* is search-surfaced (URLs in Sources) and labelled
as such — this note did not fetch-and-read each paper, so treat the SOTA half as a map, not
a settled reading. No throughput/latency number is asserted.

## The question

The prompt: *in-flight agents need to understand each other better.* Concretely — when
agent B is running, how does agent A read B's **status** (is it live? what has it actually
done? is it safe to interrupt?) in a way A can **trust**? This note maps the 2026 field to
what fak already ships and isolates the one honest gap.

## 1. The field, July 2026 (search-surfaced)

**Google A2A (Agent2Agent), the de-facto transport.** Open-sourced April 2025, now under
the Linux Foundation; **v1.0** landed ~April 2026 (search-surfaced: ~22K GitHub stars,
5-language SDKs, 150+ orgs). It standardizes: the **Agent Card** (a JSON capability/identity
manifest at a well-known URL), **JSON-RPC 2.0** transport (plus gRPC / HTTP+JSON bindings),
SSE streaming + push notifications, and an **eight-state Task lifecycle** —
`submitted · working · input_required · auth_required · completed · failed · canceled ·
rejected`. The headline v1.0 addition is **Signed Agent Cards**: a cryptographic signature
so a receiver can verify the card was issued by the claimed domain owner. A2A and MCP are
positioned as complementary (A2A = agent↔agent; MCP = agent↔tool/resource).

**The frameworks (AutoGen/AG2, LangGraph, CrewAI, OpenAgents).** Coordination is natural-
language message passing over a shared graph/blackboard or role handoff. A recurring finding
in the 2026 comparisons: these frameworks *"lack formal coordination guarantees — no
characterization of when agents' plans are compatible, no metric for coordination
complexity, and no principled mechanism for detecting or resolving inter-agent conflicts."*
As of mid-2026, OpenAgents advertises native MCP+A2A; CrewAI added A2A; LangGraph/AutoGen
lean on community integrations.

**The trust critique (the part that matters here).** A cluster of 2026 papers/commentary
converges on one point — the same point fak was built around:

- *"Self-reported records are not evidence. They are claims. The log has a fundamental
  problem: it was written by the same entity that performed the action — the equivalent of
  a company auditing itself."*
- **Pramana** — a protocol-layer treatment of **claim verification** in autonomous agent
  networks. **"Capability Advertisement as a Market for Lemons"** — a trust layer for
  heterogeneous agent networks (advertised capability ≠ delivered capability).
- Security threat-modeling across **MCP / A2A / Agora / ANP** and the interoperability
  survey both flag: A2A signs *who the agent is* (identity), but there is **no protocol-
  level mechanism for a receiving agent to verify a peer's reported task status/claims**.
  Even the A2A project's own **#1672** proposes *identity* verification for Agent Cards —
  not claim verification.

**The distinction that organizes everything:** A2A v1.0 (signed cards) hardens the
**identity layer** — *is this really agent X?* It does not touch the **claim layer** —
*did agent X actually do what its task status says?* The in-flight-understanding problem
lives almost entirely in the claim layer.

## 2. What fak already ships (dogfood-witnessed at HEAD)

fak's whole answer to the claim layer is one invariant — *a decision no participant can
move by narrating a number* — carried at every scale (`docs/standards/agent-grammar.md`,
the closed `status`/`verify`/`arbitrate` verbs). Witnessed:

- **The no-`claimed`-field status digest.** `dos_status` folds four adjudicated kernel
  verdicts — liveness · **ledger-VERIFIED** progress · held-lease region · resume plan —
  into one **A2A-shaped, peer-readable** record whose load-bearing property is that it has
  **no `claimed` key by construction**: a peer *structurally cannot* pick up a self-report
  it is never handed. Its in-tree twin is `internal/relay/progress.go`
  (`VerifiedProgress`/`ReadVerifiedProgress`): progress is a **re-verifiable cursor** into
  the intent ledger (`ledger_ref` → rows carrying a durable `Ref`: a commit SHA, `#1234`, a
  memory slug), never a percentage a closing leg asserted. `dos_verify` is the same
  discipline aimed at one claim (confirm from git evidence, not the worker's word).

- **The in-kernel A2A message channel** (`internal/a2achan/`, `go run ./cmd/a2ademo`):
  a capability-floored, `Ref`-backed mailbox. A message is an *adjudicated transfer*, not a
  memcpy — `Send`/`Recv`/`Publish` fold the **same** registered adjudicator + ingress screen
  the kernel walks for a tool call. Default `Ref` `(Tainted, ScopeAgent)` is undeliverable
  across an agent boundary by construction; poison is quarantine-held on ingress. Three
  locales (`InKernel`/`Session`/`Window`) — the `Window` one makes a context-compaction
  handoff *explicit and adjudicated*.

- **The bounded worker-correction handshake** (`internal/a2achan/correction.go`): the
  mid-flight steering path. A worker exposes a live `WorkerStatus` whose `Digest()` a peer
  **must cite**; a stale or fabricated view is refused `UNWITNESSED` before it reaches the
  worker inbox, out-of-scope corrections are `TRUST_VIOLATION`, and the worker's ack must
  **reflect** the correction id in its next action. This is "understand each other" with a
  freshness proof baked in.

- **Coordination-conflict detection** — the exact thing the frameworks were faulted for
  lacking: `dos_arbitrate` decides whether two in-flight workers may run concurrently
  without colliding on the same files (disjoint-lease admission), and the coordination
  primitives ride one floor (`internal/comm`, `docs/multi-agent-coordination-protocol.md`).

So on the claim layer, fak is *ahead* of the surveyed field: verified status, freshness-
gated correction, and conflict arbitration all exist as shipped, test-witnessed kernel
packages.

## 3. The one honest gap (witnessed, file-cited)

**fak's no-`claimed`-field guarantee stops at the in-kernel boundary. The A2A HTTP edge —
the one surface a *foreign* in-flight agent actually reads — re-introduces a self-reported
task state.**

Read `internal/gateway/a2a.go` at HEAD:

- The `a2aTask` struct (`:23-37`) carries `State string` and `Result interface{}` and
  **no** binding to a verified-progress cursor: no `run_id`, no `ledger_ref`, no
  `dos_status` digest — nothing a reader could re-verify.
- `handleA2ASendMessage` sets status **imperatively**, and today line **`:333`** literally
  reads `// Simulate method execution` and then unconditionally sets
  `task.State = "completed"` with `Result{"success": true}` and logs a `completed`
  transition (`:333-349`).
- `handleA2AGetTaskByID` (`:467-492`) — the peer's read path — returns that `state` +
  `result` verbatim.

So a foreign agent doing `GET /a2a/v1/tasks/{id}` to understand ours receives a **claim**
(`state: completed, success: true`), not a re-verifiable pointer — precisely the "company
auditing itself" failure the 2026 literature names, at precisely the boundary where it
matters most (a peer that *cannot* see our kernel). The audit log (`a2aAuditLog`) records
transitions, but the **state a peer reads is not evidence-bound**. (Minor drift also worth a
line: the edge's ad-hoc `created/running/completed/failed/canceled` string set is not the
A2A v1.0 eight-state lifecycle — no `input_required`/`auth_required`/`rejected`.)

This is the missing rung of the value ladder already sketched in
`docs/a2a-value-opportunities.md` (Rank-2 "A2A task admission and audit flight recorder",
Ladder Step 2): the task store exists, but its **status projection is not yet the
`dos_status` digest**.

## 4. The seam to close it

Carry the invariant across the edge: bind an `a2aTask`'s state/result to a **`dos_status`-
shaped digest** (liveness + ledger-VERIFIED progress + region + resume), keyed on the run,
so `GetTask` hands a peer a **re-verifiable cursor** (`run_id` + `ledger_ref` + verified
steps) instead of a self-reported enum. `internal/relay/progress.go` already produces the
in-tree shape; the work is projecting it onto `a2aTask`/`handleA2AGetTaskByID`, not
inventing a format. First checkable step: a `GetTask` response for a still-running task
carries `progress.verdict ∈ {verified, unknown}` with steps drawn only from the ledger, and
a test asserts the response type tree has **no `claimed`/`success` self-report field** for
in-flight state (the reflective invariant `relay.TestVerifiedProgressHasNoClaimedField`
already pins for the in-kernel twin).

## Parked issue body (contract shape — not yet filed)

> **`feat(gateway): project the no-claimed-field dos_status digest onto the A2A task edge`**
> The in-kernel status discipline (`dos_status`, `relay.VerifiedProgress`) guarantees a peer
> is never handed a self-report. The A2A HTTP projection drops it: `a2aTask` (`internal/
> gateway/a2a.go:23`) has no verified-progress binding and `handleA2ASendMessage:333` sets a
> hardcoded `completed/success:true`, which `handleA2AGetTaskByID:467` returns to a foreign
> peer. **Change:** add a `run_id`/`ledger_ref` (or embedded `VerifiedProgress`) to
> `a2aTask` and have `GetTask` return the ledger-verified cursor for in-flight state.
> **Acceptance:** a running task's `GetTask` body carries `progress{verdict, steps[]}` read
> from the intent ledger; a test asserts no `claimed`/self-reported `success` field for
> in-flight state; existing A2A edge tests stay green. **Seam:** `internal/gateway/a2a.go`,
> reusing `internal/relay/progress.go`. Anchors to the coordination-protocol RFC
> (`docs/multi-agent-coordination-protocol.md`) and the value ladder Step 2
> (`docs/a2a-value-opportunities.md`).

## Honest fences

- The SOTA half is a **search-surfaced map**: paper claims (Pramana, Market-for-Lemons, the
  MCP/A2A/Agora/ANP threat model, the interoperability survey) and the A2A v1.0 stats
  (stars/orgs/state names) come from the Sources below and were **not** individually
  fetch-verified in this pass; the A2A-v1.0 web write-ups broadly agree but should be pinned
  to the primary spec before any of it is quoted as fact.
- The fak half is read directly at HEAD and is the load-bearing part; the gap is a real,
  file-cited observation, not a claim that any fix shipped. The parked issue body is a
  proposal, not a landed change.
- fak's claim-layer lead is over the *surveyed* field; it is not a completeness claim.

## Sources

Field (search-surfaced 2026-07-07):
- A2A protocol spec: https://a2a-protocol.org/latest/specification/ · v1.0 announcement:
  https://a2a-protocol.org/latest/announcing-1.0/ · A2A↔MCP:
  https://a2a-protocol.org/latest/topics/a2a-and-mcp/
- IBM, "What Is Agent2Agent (A2A)": https://www.ibm.com/think/topics/agent2agent-protocol
- A2A #1672, "Agent Identity Verification for Agent Cards":
  https://github.com/a2aproject/A2A/issues/1672
- "Pramana: A Protocol-Layer Treatment of Claim Verification in Autonomous Agent Networks":
  https://arxiv.org/pdf/2605.20312
- "Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneous Agent
  Networks": https://arxiv.org/pdf/2606.03034
- "Security Threat Modeling for Emerging AI-Agent Protocols (MCP, A2A, Agora, ANP)":
  https://arxiv.org/pdf/2602.11327
- "A Survey of Agent Interoperability Protocols (MCP, ACP, A2A, ANP)":
  https://arxiv.org/html/2505.02279v1
- "Hallucination as Context Drift: Synchronization Protocols for Multi-Agent LLM Systems":
  https://arxiv.org/pdf/2606.21666
- Framework comparisons (LangGraph/CrewAI/AutoGen/OpenAgents), 2026:
  https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared

In-tree (read at HEAD):
- `internal/gateway/a2a.go` (the A2A edge + `a2aTask`), `internal/a2achan/` (channel +
  `correction.go`), `internal/relay/progress.go` (`VerifiedProgress`),
  `docs/a2a-in-kernel-channel.md`, `docs/a2a-value-opportunities.md`,
  `docs/multi-agent-coordination-protocol.md`, `docs/standards/agent-grammar.md`; the
  `dos_status` MCP verb.
</content>
</invoke>
