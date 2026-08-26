---
title: "Agent scale: macro, baseline, sub, and micro"
description: "A lifecycle hierarchy for agents that stays independent from model size, compute, and fleet volume."
---

# Agent scale: macro, baseline, sub, and micro

**Agent size means the span of identity and state that must remain coherent. It does not mean
model size, intelligence, importance, cost, or number of copies.**

This document defines four lifecycle classes:

> **macro → baseline → sub → micro**

The arrow moves from the broadest durable identity and history toward the smallest useful
agentic execution and state boundary. It is not a chain of command and it is not a quality
ranking.

This hierarchy is one coordinate of a larger design space. Every deployment should describe
at least three independent coordinates:

1. **Lifecycle/state scale:** macro, baseline, sub, or micro.
2. **Model/compute scale:** the model, reasoning budget, hardware, and latency budget selected
   for each operation.
3. **Fleet scale/topology:** how many actors exist and how they branch, join, compete, vote,
   or share work.

Keeping these coordinates independent is the central invariant. A macro agent can live for a
year while handling most events with a tiny model. A micro agent can live for one turn while
using the largest model and an entire accelerator. A fleet can contain one macro agent, one
micro agent, or 100,000 micro agents with shared prefixes and cache reuse.

A compact description can use the scale vector **A/M/F**: agent lifecycle class, model/compute
choice, and fleet cardinality/topology. **A/M/F is descriptive notation, not a shipped schema or
API.**

| Descriptor | Lifecycle | Model/compute | Fleet/topology |
|---|---|---|---|
| `micro/frontier/1` | Micro | Frontier | One actor |
| `micro/small/100k-fanout` | Micro | Small | 100,000-way fan-out with shared-prefix cache reuse |

Both configurations are coherent. The descriptor records a deployment decision, not a permanent
property of the actor; model and fleet coordinates can change on the next operation while the
macro identity remains the same.

## The lifecycle hierarchy

| Class | Identity and lifetime | State boundary | Addressability | Typical ownership boundary | Termination |
|---|---|---|---|---|---|
| **Macro agent** | Human-like durable identity; months, years, or open-ended | Layered history, commitments, relationships, preferences, authority, and resumable goals survive many sessions and processes | Stable asynchronous address such as an email address, mailbox, queue, or agent URI; the sender need not keep a session open | Owns a durable charter and may create or supervise many sessions and descendants | Retirement, succession, charter end, or explicit deletion—not the end of a chat or process |
| **Baseline agent** | The familiar ordinary interactive agent; usually one task, conversation, or bounded session | Working history and task state remain coherent for that session; selected results may be promoted upward | Usually a live session, run, or synchronous endpoint; durable inbox is optional | Owns one user-visible task or conversation | Task completion, session close, reset, or context-policy boundary |
| **Sub-agent** | Delegated child actor; shorter-lived than the parent relationship that created it | Receives an explicit subset of context, capability, budget, and objective | Reached through its parent, task ID, lease, or result channel | Owns a delegated subproblem and owes a typed result or receipt to a parent | Result returned, lease expires, cancellation, or delegated objective ends |
| **Micro agent** | Smallest useful agentic execution/state unit; one context window, turn, hardware scheduling unit, activation-bounded contribution, or similarly narrow boundary | Minimal prompt/state needed for one bounded decision, effect, check, or contribution | Usually invocation-scoped; durable public identity is unnecessary | Owns one narrow outcome and emits a compact result/effect receipt | Immediately after its bounded outcome or budget boundary |

The class is determined by what must remain the **same actor** across time, not by how many
tokens, parameters, GPUs, tools, or child processes it uses.

## Fast boundary tests

Use these tests in order:

1. **Identity test:** Would a person reasonably say “contact the same one next month,” and
   expect its commitments, relationships, and authority to continue? That is macro behavior.
2. **Session test:** Does the identity only need to stay coherent for this user-visible task or
   conversation? That is baseline behavior.
3. **Delegation test:** Does the actor exist because a parent assigned a narrower objective,
   with inherited limits and a required return edge? That is sub-agent behavior.
4. **Minimum-unit test:** Can the actor disappear after one bounded decision, effect, check,
   or contribution without losing user-visible identity? That is micro-agent behavior.

A long transcript alone does not make a macro agent. Macro requires durable identity semantics:
restart continuity, layered history, stable reachability, ongoing authority, and explicit
retirement or succession. Likewise, a one-turn child is not automatically a micro agent: it is
a sub-agent when the important contract is delegation and return-to-parent. The labels can be
combined when useful—for example, “a micro-scale sub-agent”—but the system should record the
lifecycle class and the ownership edge separately.

## Three independent coordinates

### 1. Lifecycle/state scale

This document's hierarchy answers: **How much identity and state must remain coherent, and for
how long?**

The retained state should become more selective as lifetime grows. A macro agent must not replay
months of raw transcripts on every turn. It needs layers such as:

- current working context;
- episodic records and receipts;
- durable semantic memory and preferences;
- commitments, authority, relationships, and unresolved goals;
- provenance, retention policy, corrections, and deletion history.

Lower levels receive projections of that state, not automatic copies of the whole biography.

### 2. Model and compute scale

This coordinate answers: **What inference and hardware should this operation use?** It may vary
on every operation without changing agent class.

| Lifecycle class | Small-model example | Large-model example |
|---|---|---|
| Macro | A year-long support persona triages routine inbox mail on a small model and escalates rarely | The same identity uses a frontier model for a consequential negotiation or deep retrospective |
| Baseline | A normal chat handles routine edits cheaply | A single session uses a large reasoning model for a difficult design review |
| Sub | Thousands of delegated classifiers use small models | One delegated theorem-checking child uses the largest model available |
| Micro | 100,000 one-turn extractors run on the smallest acceptable model | One one-turn judge spends a large model and accelerator on a high-stakes decision |

Therefore, never infer model size from the words macro, baseline, sub, or micro. Routing should
consider task difficulty, quality target, risk, latency, cost, privacy, hardware locality, and
cache state independently from identity lifetime.

### 3. Fleet scale and topology

This coordinate answers: **How many actors run, and how are their outcomes composed?** Relevant
properties include:

- cardinality: one, tens, thousands, or 100,000+ actors;
- topology: fan-out/fan-in, tree, market, tournament, pipeline, swarm, or long-lived society;
- concurrency and admission limits;
- shared immutable prefix versus per-actor delta;
- cache sharing and invalidation boundaries;
- result aggregation, voting, deduplication, and conflict resolution;
- budget, cancellation, backpressure, and straggler policy.

Fleet count does not change lifecycle class. A single micro agent remains micro. A 100,000-way
micro fleet remains a fleet of micro agents. A macro agent may be singular while supervising a
large transient fleet, and many macro agents may form a durable organization.

At high fan-out, the economic unit is not merely “tokens per agent.” Net cost includes shared
setup, cacheable prefix construction, branch-specific deltas, scheduling, verification,
aggregation, failed branches, and recovery. Micro agents become compelling when the kernel can
do shared work once, preserve cache locality, bound each branch, and reconcile only useful
results.

## Composition and ownership rules

The hierarchy composes rather than replacing one level with another:

1. A **macro agent** accepts an asynchronous event under a stable identity.
2. It opens or resumes a **baseline agent** session for the user-visible task.
3. The baseline agent delegates bounded work to **sub-agents** with explicit context,
   capability, budget, and return contracts.
4. A sub-agent or baseline agent may invoke many **micro agents** for minimal decisions,
   effects, checks, or speculative branches.
5. Compact receipts and selected outcomes flow upward. Raw lower-level histories do not become
   durable macro memory by default.

Every downward edge should narrow authority or make an intentional grant explicit. Every upward
edge should carry provenance. Parent cancellation should have defined propagation semantics,
while a macro identity must survive ordinary child, session, process, and machine failures.

A macro agent's stable address is a routing handle, not proof that a process is always running.
An email-like contract implies asynchronous delivery, durable queueing, authentication,
deduplication or idempotency, thread correlation, reply semantics, abuse controls, and an audit
trail. The identity can wake on demand, choose a model and fleet plan for the event, persist the
result, and return to dormancy.

## Worked example: a six-month release steward

`release-steward@example.invalid` names one macro agent responsible for a product's release
readiness for six months.

- **Macro lifecycle:** its charter, contacts, decisions, unresolved commitments, and correction
  history survive restarts and individual chats. Maintainers can send it mail while it is
  dormant. It can be retired or succeeded without pretending that a session ended the identity.
- **Baseline session:** an incoming “prepare the September release” thread opens one coherent
  interactive task with a bounded working set.
- **Sub-agents:** the session delegates changelog review, compatibility analysis, benchmark
  checking, and rollout planning with separate capability and budget limits.
- **Micro fleet:** compatibility analysis fans out 100,000 narrow checks over artifacts. Most
  use the smallest model that meets the quality threshold, reuse a shared policy/tool prefix through shared-prefix cache reuse,
  and retain only branch deltas. A few ambiguous checks escalate to a large model. One
  high-stakes final micro judge may use the largest model for a single response.
- **Return path:** aggregators deduplicate results and return receipts to their parents. The
  baseline session presents a release recommendation. Only durable decisions, commitments,
  evidence references, and corrected preferences are promoted into macro history.

The macro agent is “large” because its identity and obligations span months, not because every
turn uses a large model or because every event launches a huge fleet.

## What this hierarchy is not

- It is not a model-size ladder.
- It is not an intelligence, autonomy, trust, or importance score.
- It is not an organizational rank; a sub-agent may outperform its parent on its specialty.
- It is not a context-length ladder; durable history should be indexed, compacted, and selected.
- It is not a mandatory four-process architecture; the boundaries are semantic and may share a
  runtime.
- It does not claim that activation-scale agents are shipped today. “Activation-bounded” is a
  possible lower research boundary only when the unit has an objective, bounded state or
  budget, and an observable result or contribution. Raw activations alone are not agents.

## Current fak status

**Shipped:** fak already supports bounded sessions, child registration, policy and capability
floors, budgets, cache-aware execution, micro-agent demos, effect receipts, and guarded worker
orchestration. The concrete micro-agent walkthrough and evidence are in
[`micro-agents.md`](micro-agents.md).

**Concept spine:** the four lifecycle classes, human-like macro identity, stable asynchronous
address, durable layered history, and the three-coordinate vocabulary are definitions in this
document. They are not a claim that fak currently ships a production macro-agent mailbox or a
complete durable-identity runtime.

**Research and follow-on implementation:** durable identity schemas, mailbox protocols,
retention and correction semantics, succession, cross-session authority, macro-level
observability, high-cardinality fleet economics, and activation-bounded execution require
separate designs and witnesses.

## Naming rule

Describe a system with coordinates rather than an overloaded adjective:

> **one macro agent, small-model default with frontier escalation, supervising a 100,000-way
> cache-sharing micro fleet**

That sentence says substantially more than “a big agent” or “a small agent,” while preserving
the useful macro → baseline → sub → micro hierarchy.
