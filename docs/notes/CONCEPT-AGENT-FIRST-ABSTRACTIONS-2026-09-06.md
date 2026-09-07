# Agents are ephemeral; work, effects, and evidence are durable

Research and design proposal, observed 2026-09-06. No implementation or scale result is claimed. Names below are illustrative contracts over existing FAK seams, not a proposal for six new packages or another agent framework.

Durable study receipt: `study_119ac6a29b6acc8201063affb98ee9d7b3e5ffe4d71588921b694f4ed084d96e`. Coordinator read-back anchors the local candidate seams to FAK commit `78c3d77fc7ce3fe0453edc7acb3a3f5d0e5b4ebb`.

The next useful abstraction is not a better conversation between agents. It is work that remains intelligible when the conversation, model, process, and supervisor have all changed. A worker should be replaceable without losing what was requested, what was authorized, what happened outside the system, or why a result deserves acceptance.

At small scale, a coordinator can remember these distinctions. At large scale, its memory becomes a bottleneck and an accidental database. A thousand workers amplify ambiguous completion, stale assumptions, duplicate spending, and noisy notification. More agents become useful only when each additional worker carries a bounded obligation and returns something another process can check.

Thousands of workers change the communication contract. All-to-all conversation creates quadratic message opportunities; a central inbox merely relocates the overload. Selective subscriptions and bounded fan-in should let a worker observe only changes affecting its obligations. Claims need explicit dependencies so the system can identify who must reconsider, rather than asking everyone to reread everything. Truth does not emerge from consensus among workers sharing the same stale premise. Independent checks and explicit applicability remain necessary even when every agent agrees.

For developers operating FAK's integrated serving, harness, and memory product, the problem is continuity of useful work across parallel execution and failure. Today, FAK already has recursive budgets, capability admission, durable effect deduplication, and verified compact receipts. The better outcome is to compose those mechanisms around explicit identities and validity boundaries. The witness is a small, deterministic three-worker failure experiment before any claim about massive scale. This is Core work for the all-in-one product, with standalone harness value; it does not require replacing native inference or introducing a heavyweight distributed platform.

## What the open-source systems actually teach

Temporal makes nondeterministic observations replayable. At the pinned Go SDK revision, [internal/internal_event_handlers.go:1054](https://github.com/temporalio/sdk-go/blob/f502fbabc233f5ed7719f43752dfd6da1b993000/internal/internal_event_handlers.go#L1054) either retrieves a recorded SideEffect result or evaluates and records a new one. Mutable side effects distinguish call occurrences at `:1125–1184`; version markers at `:958` support workflow evolution. The transferable lesson is that a result needs a position in an execution history, not merely a friendly name. Remote model calls still need an appropriate durable activity boundary; a short synchronous SideEffect callback is not a universal LLM wrapper.

Ray exposes a different boundary: process restart and task retry are separate decisions. Its [python/ray/actor.py:1773](https://github.com/ray-project/ray/blob/317c2888eade3c294c4fdb46eff9d9ec290b08f2/python/ray/actor.py#L1773) describes restart/retry budgets, per-handle pending-call limits, and concurrency ordering. The fault-tolerance documentation explicitly admits that an error can arrive after a task successfully executed. Restarting an actor reruns its constructor; application-state recovery is a separate obligation. An available process is not necessarily continuous work.

A2A provides useful task vocabulary, but not an authority system. Its proto defines task identity, status, artifacts, and history; its [lifecycle documentation:83](https://github.com/a2aproject/A2A/blob/98853be376c88df25e1704771cd3ea9ef8823a96/docs/topics/life-of-a-task.md#L83) says terminal tasks do not restart and leaves artifact mutation/version lineage to clients. A transport task can project an execution attempt where semantics match. It cannot silently become the complete identity of a durable objective.

LangGraph's pending-write handling shows why preserving successful sibling work matters. Its [runtime:1530](https://github.com/langchain-ai/langgraph/blob/81bf17b23123e4ef8b9d5f49fa09a0122fc2edd1/libs/langgraph/langgraph/pregel/_loop.py#L1530) orders particular persistence futures before checkpoints, and its tests exercise a successful sibling alongside a failing/retried one. This is valuable recovery structure, not proof of arbitrary external effects executing once. These systems offer complementary mechanisms; none of these bounded readings establishes an integrated solution to all the problems below.

## Six composable objects

| Object | Owns | Deliberately separate |
|---|---|---|
| WorkRef | Objective, input basis, acceptance contract, accepted result | A replaceable execution Attempt |
| ScopeRef | Authority, resource reservation, deadline, cancellation lifecycle | Transport connectivity and worker liveness |
| DecisionRef | One recorded nondeterministic choice and its versioned basis | A timeless instruction to reuse the answer |
| EffectRef | Semantic intent, deduplication identity, observed outcome, reconciliation | A successful network response |
| ClaimRef | Proposition, evidence references, validity dependencies | Artifact authenticity alone |
| DemandRef | A bounded request for a capability/result and its rendezvous | Broadcasting every event to every agent |

**WorkRef** lets a goal outlive attempts. Its acceptance contract is immutable within a version: changing the required artifact or quality condition creates a new version, rather than retroactively declaring an old attempt successful. Two workers can race to satisfy one version, but acceptance chooses an explicitly witnessed result. FAK's durable-logical-agent epic [#10633](https://github.com/anthony-chaudhary/fak/issues/10633) already names much of this direction. First-class child handles in [#10491](https://github.com/anthony-chaudhary/fak/issues/10491) are the immediate composability seam. Neither deserves a duplicate “add durable agents” ticket.

**ScopeRef** combines attenuation with accounting. A child receives a subset of authority and a reservation from a shared resource ceiling, never a fresh copy of its parent's budget. FAK's `internal/microagent/spawn_budget.go:14–34` already represents reservations, spent amounts, and ancestry; `:73` admits children. The narrow question is what survives controller death. A durable reservation must not disappear on restart or be charged twice when a late child report arrives. Cancellation also needs settlement: requested cancellation, stopped admission, acknowledged quiescence, and a late completed effect are distinguishable facts. Temporal's cancellation tests explicitly allow completion after a cancel command.

**DecisionRef** records an occurrence, its input basis, and the selected output. Replaying work consumes the recorded choice; asking a new model to reconsider creates a new decision or branch. The cost is retention and version management, so retain compact references rather than whole transcripts by default. Replay determinism does not mean the original decision was correct. A changed dependency can make a faithfully replayed decision unsuitable for current work.

**EffectRef** protects the external boundary. FAK already implements durable PENDING, UNKNOWN_APPLIED, APPLIED, and PROVEN_ABSENT in `internal/idempotency/idempotency.go:85–125`. Its store blocks ambiguous effects pending operation-specific readback. The proposal must preserve that existing work. The narrower gap is visible in `internal/microagent/effectsafe.go:15` and `:59`: the intent includes resource and capability, but the key passed to the store is derived from operation and token. Bind that identity to a canonical semantic digest and authority domain. Same token with different resource or request semantics should conflict, not replay an unrelated result. Making a new digest into a new key would miss the conflict: the original key must remember which semantics it accepted.

**ClaimRef** says what evidence supports and under which dependencies it remains applicable. FAK's `internal/microagent/receiptverify.go:33` already bounds a completion receipt and provenance; `:212` routes it through an independent verifier before folding into root context. This is a strong base. The proposed axis is dependency validation at the first parent fold: reject or request correction if the input, policy, schema, or artifact basis changed while the child was working. Authentic bytes establish that a receipt has not changed; they do not establish that its conclusion is still valid. Existing materialized-state work [#8803](https://github.com/anthony-chaudhary/fak/issues/8803) supplies adjacent identity/freshness vocabulary; avoid a parallel cache or evidence system.

**DemandRef** would let topology follow unmet needs. A worker publishes a bounded need with capability constraints, deadline, capacity, and an acceptable result shape. Eligible workers rendezvous with that need; results wake interested consumers. This is INFERENCE/WATCH, not a claim borrowed from the inspected subtrees. Ray's per-handle queue cap offers a useful warning: multiplying callers can multiply queued work unless admission is aggregate. Start with a single existing dispatcher, a bounded queue, and observable rejection. A distributed market or global task exchange is unnecessary for the first experiment.

These objects should be references across existing seams. A small task may use only WorkRef and a verified receipt; an irreversible operation needs EffectRef. Requiring every request to populate a universal ontology would recreate the coordination overhead this design is meant to remove.

## A future worth testing

Imagine an organization of short-lived specialists assembled by demand. A failed verification opens a narrow demand; a fresh evidence item satisfies it; only dependent claims wake. A thousand idle workers do not need a thousand active conversations. The scheduler spends compute where an unresolved obligation justifies it. Test this against a tuned fixed worker pool on one repeated workload, counting queue latency, coordinator tokens, wake storms, indexing, and verification. Reject the design if bookkeeping consumes the saved work. Maintenance includes expiry, fairness, poison demands, and schema compatibility.

Now add reversible speculation. Several workers propose patches against one explicit state basis. Their branches converge through checked patches and acceptance rules, not through an average of their prose. Only authority-bearing effects cross a serialized resource boundary. This is INFERENCE/WATCH: managed detached worktrees already supply part of the experiment. Compare two bounded speculative candidates with a sequential baseline, including wasted inference, merge work, tests, and discarded artifacts. Keep speculation optional if value appears only under narrow latency constraints.

Finally, preserve institutional memory of falsified assumptions. Store “this inference failed under these conditions” with its witness and refresh trigger. A future planner retrieves the scoped counterexample rather than a permanent ban on a technique. This is INFERENCE/WATCH with a cheap test: repeat a failure-prone task class and measure whether a retrieved counterexample prevents the same mistake without blocking valid changed conditions. Curating validity windows, contradictory evidence, privacy, and retirement costs is part of the feature, not free storage.

There is no need for one global total order. Per-effect serialization, local event order, and explicit dependency edges can serve different contracts. CRDT convergence means replicas settle on the same state; it does not make a merged plan semantically correct. Transport authentication identifies a peer; it does not delegate authority. A valid signature does not prove a fresh premise. Durable retry does not confer exactly-once effects on an endpoint lacking appropriate idempotency or reconciliation.

## One experiment with three workers

Worker A produces a candidate from input digest X. Worker B applies one external change. Worker C verifies A's claim before the parent accepts it. Run the sink in a separate process with deterministic fault hooks and a durable append log.

First reserve B's budget, then restart the controller before settlement. The reservation must remain occupied. Let the sink commit B's effect and kill B before its local result is persisted. Recovery must replay a proven result or retain UNKNOWN and reconcile; it must not infer “nothing happened” from process death. Submit the same effect key with changed semantic intent and require a conflict before apply. Meanwhile replace input X with Y before C folds A's receipt. Authentic evidence about X must not be accepted as evidence about Y. Deliver a late completion after cancellation and ensure it remains auditable without granting new authority or overwriting an accepted newer result.

Capture reservations, commands, sink writes, receipts, and verifier decisions as one integration artifact. Run a no-crash baseline and charge persistence, recovery, and verification overhead. This proves bounded three-worker properties. It does not prove thousand-agent scalability or exactly-once behavior for arbitrary services.

## Four atomic next steps

Existing-work disposition: closed #7809 covers live recursive accounting, closed #7814 covers receipt verification, and closed #8284 covers durable ambiguity handling. Custom receipt verifiers may already check freshness; the proposed shared contract names dependency snapshots at first fold. No generic primitive is claimed absent.

Identity discipline: bind effects to the actual principal/tenant namespace. ContextID groups conversation, not authority. Retries across Attempts preserve effect identity; changed semantics under the original key conflict. Hashing a changed payload into a fresh key would silently permit another effect and violate the intended contract.

1. **Persist recursive reservations across restart.** Extend the existing SpawnBudget lifecycle, building on [#7809](https://github.com/anthony-chaudhary/fak/issues/7809). One witness restarts after admission and proves replayed settlement neither frees capacity early nor double-charges. Keep ancestry/scheduler redesign out of scope.
2. **Validate dependency basis at the first parent fold.** Extend the existing verified receipt boundary with a bounded dependency comparison. One witness changes the basis after child production and requires rejection before root context acceptance. Do not bundle full invalidation propagation or a knowledge graph.
3. **Bind durable store keys to semantic intent.** Reject same-key binding mismatches after reopening storage, with one direct witness. Keep this leaf in the idempotency package and preserve UNKNOWN/readback behavior.
4. **Wire EffectIntent to the admitted binding.** After step 3, adapt the microagent coordinator with one witness proving changed resource/request semantics cannot reach apply. Keep this leaf in microagent. Remote protocol reconciliation remains open [#10427](https://github.com/anthony-chaudhary/fak/issues/10427).

| Axis or mechanism | Local finding | Disposition |
|---|---|---|
| Live recursive accounting; verified receipts; durable UNKNOWN/readback | PRESENT in inspected seams | Preserve and compose |
| Reservation recovery across controller restart | PARTIAL; live accounting exists | WATCH pending crash witness |
| Explicit dependency snapshot at first parent fold | PARTIAL; custom verification exists | WATCH pending mutation witness |
| Durable effect semantic binding and coordinator integration | PARTIAL; durable op/token identity exists | WATCH; two dependent leaves |
| Demand-driven topology, speculative convergence, falsified-assumption memory | INFERENCE; coverage incomplete | WATCH with bounded experiments |

The broader framing belongs with [#11190](https://github.com/anthony-chaudhary/fak/issues/11190), the existing high-volume architecture epic. These are candidate issue scopes, not implementation claims. Recheck duplicate coverage and current source before filing.

Companions: [field-borrow](https://github.com/anthony-chaudhary/fak/blob/78c3d77fc7ce3fe0453edc7acb3a3f5d0e5b4ebb/.claude/skills/field-borrow/SKILL.md), [study-repo](https://github.com/anthony-chaudhary/fak/blob/78c3d77fc7ce3fe0453edc7acb3a3f5d0e5b4ebb/.claude/skills/study-repo/SKILL.md), and the durable study receipt above.

P1 is advanced by retaining compact identities and evidence instead of reconstructing transcripts; dependency changes must invalidate reuse honestly. P2 requires comparison with existing FAK orchestration, including storage, failed attempts, verification, and operator work; no efficiency gain is established here. P3 requires additive, versioned contracts and a reversible rollout at existing seams. P4 requires exercising the real spawn/effect/fold paths and exposing unresolved reservations, UNKNOWN effects, and stale claims to operators. A helper-only test is insufficient for the integrated scenario.

## Source ledger and limits

Temporal: [sdk-go@f502fbabc233f5ed7719f43752dfd6da1b993000](https://github.com/temporalio/sdk-go/tree/f502fbabc233f5ed7719f43752dfd6da1b993000), commit 2026-09-06T04:24:36Z, main snapshot; latest release observed v1.48.0 dated 2026-08-18. Exact LICENSE is MIT with Temporal/Uber notices. Read workflow API, event handlers, and command-state-machine tests, including cancellation `:219–276`. Closed [#1014](https://github.com/temporalio/sdk-go/issues/1014) documented same-ID occurrence replay failure; closed [#2109](https://github.com/temporalio/sdk-go/issues/2109) showed test-environment equality semantics diverging from runtime, fixed by PR #2516 on 2026-08-04. These are reasons to test real replay.

Ray: [ray@317c2888eade3c294c4fdb46eff9d9ec290b08f2](https://github.com/ray-project/ray/tree/317c2888eade3c294c4fdb46eff9d9ec290b08f2), commit 2026-09-06T05:02:20Z; master snapshot, latest release observed 2.58.0 dated 2026-08-23. Exact root LICENSE is Apache-2.0 with additional component notices. Read actor API, actor fault-tolerance documentation, and `python/ray/tests/test_actor_failures.py:1187–1264` process-kill tests. Documentation `actors.rst:56–62` and API `actor.py:1806` disagree about retry ordering; universal ordering is unresolved in this study.

A2A: [98853be376c88df25e1704771cd3ea9ef8823a96](https://github.com/a2aproject/A2A/tree/98853be376c88df25e1704771cd3ea9ef8823a96), commit 2026-09-01T08:22:53Z, Apache-2.0, main snapshot rather than a release claim. Read `specification/a2a.proto:163–207` and `docs/topics/life-of-a-task.md:83–134`. Open #1992 contains lifecycle proposals, not settled guarantees.

LangGraph: [81bf17b23123e4ef8b9d5f49fa09a0122fc2edd1](https://github.com/langchain-ai/langgraph/tree/81bf17b23123e4ef8b9d5f49fa09a0122fc2edd1), commit 2026-09-03T15:23:25Z. Exact `libs/langgraph/LICENSE` is MIT; separate checkpoint-package licenses were not reviewed. Read `libs/langgraph/langgraph/pregel/_loop.py:415–498,1530–1547` and `libs/langgraph/tests/test_pregel.py:891–935`. Open #8039 reports persistence-ordering failure; it was not reproduced here. SDK release metadata is not evidence of runtime release status.

ADAPT is a plausible reuse disposition for these bounded mechanisms with required attribution; no expressive code was copied. Imported, generated, vendored, and separate-package obligations require checking before a port. Tests were read, not executed. Server persistence, Ray C++ scheduling, LangGraph backend/async internals, full discussions and RFCs, and performance scaling remain outside coverage. Local FAK findings are read-back of a shared working tree, not a newly verified release. Source changes, relevant issue resolution, or implementation planning trigger refresh. The next checkable step is the bound-identity store witness in #11950, followed by coordinator integration #11954; reservation recovery #11951 and receipt snapshot validation #11952 can progress independently under their lane rules.

## Filed follow-ons

The design is published in [#11949](https://github.com/anthony-chaudhary/fak/issues/11949). Open/closed duplicate searches and source read-back preceded filing.

- [#11950: Bind retry tokens to immutable semantic request identity](https://github.com/anthony-chaudhary/fak/issues/11950).
- [#11951: Recover lineage budget reservations after restart](https://github.com/anthony-chaudhary/fak/issues/11951).
- [#11952: Validate input snapshot before child receipt fold](https://github.com/anthony-chaudhary/fak/issues/11952).
- [#11954: Enforce bound request identity in EffectCoordinator](https://github.com/anthony-chaudhary/fak/issues/11954), dependent on #11950.

These are experimental research proposals, not implemented guarantees. The contract review returns `triage_only / ISSUE_OPERATING_ENVELOPE_UNDER_TARGET` while implementation witnesses are pending. Unknown parent completion estimates were not fabricated. Broader demand-driven topology, reversible speculation, and falsified-assumption memory remain WATCH hypotheses in this design, with the falsifiers above; they are not dispatchable implementation leaves.
