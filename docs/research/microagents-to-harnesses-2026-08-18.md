---
title: "Bounded microagents as harness-building primitives"
description: "A source-pinned design map from 1-3-turn microagents to recursive, context-clean harness construction, with the shipped demo spine and falsifiable next measurements."
status: working
last_updated: 2026-08-18
---

# Bounded microagents as harness-building primitives (2026-08-18)

## Verdict first

A one-turn tool call is a useful mental model, but the production primitive should be a **bounded delegated computation**: a typed task enters with a capability, lineage, and aggregate budget; it may use one to three model turns; it exits through a verified evidence receipt. The parent admits only that receipt, never the child's full transcript. A harness is then a compiled graph of those receipts: runtime, model, tools, policy, workflow, completion proof, and operator surface.

The recursion is valuable only while four quantities remain bounded together: **depth/fanout, spend, authority, and context admitted upward**. A child must request grandchildren through the same host admission seam; handing it a raw spawn handle would turn decomposition into privilege and spend amplification.

## Value frame

- **For:** an operator or master agent constructing a task-specific harness.
- **Problem:** architecture, tool, policy, workflow, and proof decisions pollute one long context and become hard to verify independently.
- **Today:** a monolithic planner reasons across every concern, or a fleet returns prose whose budgets and provenance are weakly coupled.
- **Better because:** narrow children work in disposable contexts and return typed, independently checked receipts that compose deterministically.
- **Witness:** `go run ./cmd/microharnessdemo -selfcheck` runs 1-, 2-, and 3-turn children, admits descendants only in a second host wave, and shows `full child transcripts in root=false`.
- **Outcome counters:** pass `-ledger <path>` to append privacy-safe invocation rows, then query weekly completed/failed and live/fixture counts with `-fold-usage`. A captured fixture readout is:
  ```json
  [{"week":"2026-W35","invocations":1,"completed":1,"failed":0,"live":0,"fixture":1,"prompt_tokens":0,"completion_tokens":0}]
  ```
  Reproduce it with `go run ./cmd/microharnessdemo -selfcheck -ledger <temp.jsonl>` followed by `go run ./cmd/microharnessdemo -fold-usage -ledger <temp.jsonl>`.

## What shipped as the smallest spine

`cmd/microharnessdemo` takes the concrete goal “Build a local coding harness that can edit this repository and prove its work.” An architecture child runs for two turns and proposes two depth-2 children. The host re-admits those requests; a tool child runs one turn and a proof child runs three. Their compact receipts compose a three-field harness plan. The captured render and test prove the depth/turn envelope and receipt-only root fold.

This is intentionally an offline deterministic fixture. It proves the control shape, not live-model quality, arbitrary-goal decomposition, token savings, or production security.

## The recursive contract

Each delegated task should carry:

1. `goal` and explicit done condition;
2. references to immutable input context, not an inherited transcript;
3. requested capabilities intersected with the parent's capabilities;
4. max turns, tokens, cost, deadline, depth, and fanout reservation;
5. expected receipt schema and verifier;
6. lineage and dedupe/cycle fingerprint.

Each receipt should carry:

1. decision or artifact reference;
2. evidence and uncertainty;
3. consumed budget and effects;
4. unresolved questions;
5. child requests, which are data submitted for host admission rather than direct spawns;
6. verifier outcome and provenance.

The host owns scheduling. The parent owns decomposition intent. The verifier owns done. Harness composition owns conflict detection and deterministic assembly.

## Scaling model

The useful progression is:

`one-turn query` -> `bounded 1-3-turn worker` -> `verified receipt` -> `parallel receipt set` -> `dependency graph` -> `compiled harness/workflow` -> `harness that can invoke the same primitive`

At the last step recursion becomes operational: a generated harness may delegate, but every nested request re-enters the kernel with inherited lineage and a strictly smaller or equal envelope. The root reserves aggregate budget before children run. Depth alone is not sufficient: a depth-2 tree with unbounded fanout still explodes.

Good one-turn classes are closed choices, extraction, ranking against an explicit rubric, capability selection, and witness selection. Two or three turns are justified when a tool result must return, a verifier names a concrete gap, or two receipts need arbitration. Coupled product tradeoffs, irreversible effects, and ambiguous goal negotiation should remain master-only until evidence shows otherwise.

## How this changes harness construction

Harness construction stops being a single generative act and becomes a typed build:

- discover constraints;
- dispatch bounded architecture/model/tool/policy/proof decisions;
- verify receipts independently;
- detect conflicts and arbitrate only the disputed fields;
- compile a workflow graph and harness artifact;
- selfcheck the artifact on the original goal;
- retain the decision graph so regeneration can reuse low-volatility receipts.

The master context stays clean only if raw child transcripts remain out-of-band. “Summarize every child into prose” is not enough: summaries can carry prompt injection, erase provenance, and grow linearly. Receipts must be schema-constrained data, quoted as evidence rather than interpolated into control instructions.

## External field evidence, pinned 2026-08-18

### OpenAI Swarm — `openai/swarm@6af0b4caf37dca4526dfd98e9fbd8ce36e7eeb22`

Swarm makes handoff a function return and exposes explicit context variables. Borrow the small, inspectable handoff seam and the rule that transfer is a typed event. Do not borrow unbounded conversational transfer as the root-context boundary. Classification: **RECIPE** for handoff ergonomics; fak still needs aggregate budgets, receipts, and verification.

### LangGraph — `langchain-ai/langgraph@644815f9e5bc52ad8f7a5227a456227e9c3e639b`

LangGraph's dynamic `Send` fanout, subgraphs, reducers, checkpointing, and recursion limits show how a workflow graph can create children from state. Borrow graph-shaped state and reducers, but make budget conservation and authority intersection kernel invariants rather than graph conventions. Classification: **OPTIONAL-MODULE** semantics behind fak-native contracts.

### mini-swe-agent — `SWE-agent/mini-swe-agent@25941c89cfbc91eb40b3f8756348c91d9977d57e`

mini-swe-agent reinforces that the model/tool loop can remain small while step and cost limits stay explicit. Borrow scaffold minimality and hard termination. Do not infer that every harness decision benefits from delegation. Classification: **DEFAULT** design pressure toward the smallest child loop.

### AutoGen — `microsoft/autogen@027ecf0a379bcc1d09956d46d12d44a3ad9cee14`

AutoGen teams expose handoff messages, termination conditions, nested team composition, and state save/load. Borrow explicit termination and resumable team state. Avoid making a free-form handoff message the trusted composition unit. Classification: **RECIPE** plus a compatibility study.

## Failure modes that decide whether recursion helps

- **Budget multiplication:** every child believes it owns the root budget.
- **Privilege amplification:** a child requests broader tools or paths than its parent.
- **Context laundering:** untrusted transcript text is folded into root instructions.
- **False decomposition:** tightly coupled decisions are split and reconciled by arbitrary last-writer wins.
- **Verification recursion:** every verifier spawns another verifier without a terminal evidence rule.
- **Duplicate subtrees:** semantically equivalent tasks consume repeated turns.
- **Cheap-agent quality collapse:** context is clean but the harness is wrong.
- **Latency tail:** serial 1-3-turn calls make a simple decision slower than one master turn.
- **Cache defeat:** tiny heterogeneous prompts lose prefix reuse and cost more net-true tokens.

## Measurements required before a default

Compare a monolithic harness planner with receipt-only recursive variants on at least three real harness classes. Report goal success, human/independent verifier score, wall time, input/output tokens, provider cache reuse, root-context bytes, receipt bytes, duplicate work, effects, and dollar cost. Sweep depth, fanout, one-turn-first escalation, and model tier. The default wins only if quality remains within the declared band and net cost or context pressure improves; clean-looking context alone is not a win.

## Current-fak classification

- **PRESENT:** shared in-process microagent host, bounded queue/workers, cancellation, retries, verification, session state, lineage helpers, capability/egress floors, and harness composition packages.
- **PARTIAL:** bounded multi-turn examples, RPC subagent schema, recursive lineage, harness artifact/protocol work, and context compaction exist but are not one admitted recursive harness-building path.
- **ABSENT:** host-enforced per-child turn ceiling, host-mediated recursive spawn contract, aggregate tree budget, receipt-to-harness compiler, live model verb, recursion scorecard, and quality/cost comparison.

## Immediate backlog shape

The high-leverage order is: hard turn/depth/fanout and aggregate-budget invariants; typed task/receipt plus host-mediated child requests; capability intersection and verifier gate; receipt-to-harness composition; live model adapter; then comparative benchmarks and real harness dogfood. The broad QA, product, observability, integration, docs, and release envelope is generated from the shipped spine with `fak-dev issue fanout`.
