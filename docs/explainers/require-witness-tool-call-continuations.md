---
title: "REQUIRE_WITNESS as a durable tool-call continuation"
description: "Design for suspending a proposed tool call until a bound attestation arrives, then rechecking policy and resuming with exactly-once execution."
---

# REQUIRE_WITNESS as a durable tool-call continuation

**Status:** design note; current ABI has the verdict, but the durable held-call queue described here is not yet implemented end to end.

## Short answer

Yes: one useful meaning of `REQUIRE_WITNESS` is that the agent's proposed tool call becomes a **durable suspended continuation**. The kernel records the exact call and the condition that must be witnessed, puts a witness request on a queue, and does not execute the tool. Another process, agent, human, or service reads the request and returns a bound attestation. The kernel verifies that attestation, rechecks policy and freshness, executes the call at most once, admits or quarantines the result, and resumes the waiting agent from its checkpoint.

What pauses is the **causal tool-call continuation**, not necessarily an OS process or an in-memory model context. A synchronous adapter may block a request for a short approval, but the scalable form parks the turn, releases the model/GPU/HTTP worker, and later rehydrates it. This is closer to `await durable_future` than to keeping one context window resident in a queue.

The existing pieces make this direction natural but do not yet constitute that service:

- `abi.VerdictRequireWitness` and `abi.WitnessPayload{Claim: ...}` name the gate.
- the gateway wire projection reports `REQUIRE_WITNESS` with `disposition: ESCALATE`.
- `internal/sessionctl` already has durable park/resume concepts for sessions.
- no current gateway path persists a held call, publishes a witness request, consumes an attestation, and resumes execution as one exactly-once protocol.

That last bullet is the implementation gap. Until it lands, `REQUIRE_WITNESS` means "stop and route outward," not "the kernel already owns a complete wait queue."

## The object that waits

Do not queue a mutable chat transcript or a live context window. Queue a small, immutable **held-call record**:

```text
HeldCall {
  held_id                 // stable public handle
  session_id, turn_id, call_id
  tool, canonical_args_digest
  policy_version, plugin_set_digest
  principal, capability_snapshot
  witness_contract       // what fact/approval/read-back is required
  continuation_ref       // durable checkpoint, not raw provider state
  created_at, expires_at
  state, attempt, idempotency_key
}
```

The witness request can reveal a bounded presentation of that record, but its response must bind at least `held_id`, the call digest, the witness contract, the witness identity/class, a decision or evidence digest, and an expiry. A statement like "approved" without those bindings is replayable and must not release a call.

A witness is independent relative to the claim. It may be:

1. **Human approval** — a person accepts a concrete, bounded action.
2. **Peer adjudication** — another agent or model checks a proposal under a separately identified principal and policy.
3. **Read-back** — an independently sourced observation proves a precondition or an earlier effect. A post-effect read-back cannot authorize the same held call, because that call has not run yet; it can witness a prerequisite or release a later dependent call.
4. **External attestation** — CI, git ancestry, a database read, a policy engine, or hardware-backed signer proves the requested fact.

"A second agent said yes" is not by itself independence. The witness contract must specify an acceptable source and the verifier must establish that source did not merely echo the proposing agent.

## Process model: a reconciled graph, not a pipeline

The arrows in a simple state diagram are useful for orientation, but they are not the
process model. A held call is a **durable resource reconciled from facts and events**.
Evidence can arrive twice, in either order, or after cancellation; policy can change while
approval is pending; a witness can revoke an attestation; an execution worker can die after
the effect but before recording completion; result delivery can fail while the effect remains
complete. Treating those cases as one linear `WAITING -> VERIFYING -> EXECUTING -> RESUME`
program creates races and duplicate effects.

Do not make one enum carry all of this. Keep orthogonal observed facts on the held-call
resource:

```text
HeldCallStatus {
  intent_generation       // increments if tool/args/contract changes
  lifecycle               // open | cancelled | expired
  evidence[]              // pending | valid | rejected | revoked, each source-bound
  witness_rule            // all | any | quorum(n) | expression over witness classes
  policy_epoch_checked
  readiness               // unsatisfied | satisfied | stale
  execution               // none | leased | effect-recorded | outcome-unknown
  result_admission        // pending | admitted | quarantined
  continuation_delivery   // pending | delivered | acknowledged
}
```

Each durable event is appended idempotently, then a reducer updates observed facts and a
reconciler asks **what work is enabled now?** It may schedule zero, one, or several actions.
Workers claim actions with leases and report new facts; they do not advance a global program
counter.

```text
on event(call_id, event_id):
    append_once(event_id)
    facts = reduce(all_events_for(call_id))

    repeat until no local fact changes:
        desired = derive_desired_state(spec, facts, current_policy)
        actions = diff(desired, facts)
        enqueue_idempotently(actions)
        facts = reduce(new_local_events)
```

The repeat is a bounded fixed-point calculation, not an unbounded busy loop. External work
returns through new events and starts another reconciliation tick. A per-call generation and
event IDs make old work harmless after edits or cancellation.

### The loops that cooperate

There is no requirement that one process own a held call for its lifetime. At least four
independent loops can cooperate through the durable record:

1. **Call controller loop** — folds proposal, policy, user edits, cancellation, time, and
   evidence into desired readiness. It can move `unsatisfied -> satisfied -> stale ->
   unsatisfied` rather than only forward.
2. **Witness loops** — one adapter per eligible witness route publishes requests and ingests
   attestations or revocations. Routes can run concurrently; `all`, `any`, quorum, veto, and
   fallback-after-timeout are data in the witness rule.
3. **Execution recovery loop** — acquires the effect lease only for the current generation,
   revalidates immediately before dispatch, and resolves crashes using the tool's idempotency
   key or an independent effect read-back. `outcome-unknown` is a real state; it must not be
   converted into a blind retry.
4. **Continuation delivery loop** — admits/quarantines the result and retries delivery to an
   offline or moved session. Delivery can loop independently after execution is complete and
   can be acknowledged separately.

Notification, expiry, garbage collection, and audit projection can be additional observer
loops. They do not authorize execution.

### Common non-linear paths

```text
                              policy/config changes
                           +-------------------------+
                           v                         |
PROPOSED -> HELD <-> EVIDENCE_UNSATISFIED -> SATISFIED
                ^       ^          |              |
                |       |          | revoke/veto  | pre-exec revalidation
 edit/transform |       +----------+              v
(new generation)                            READY_FOR_LEASE
                ^                              |       |
                |                         lease lost   | effect dispatched
                |                              |       v
                +------------------------------+  EFFECT_RECORDED
                                                   |          |
                                          quarantine      admit result
                                                   |          v
                                                   +----> DELIVERY_PENDING
                                                            ^       |
                                                     retry  |       v
                                                            +-- DELIVERED

Any open pre-effect node --cancel/expiry--> TERMINAL_NO_EFFECT
Any post-dispatch ambiguity -----------> OUTCOME_UNKNOWN --read-back--> effect/no-effect
```

This is deliberately a graph with cycles:

- a policy epoch change makes previously sufficient evidence stale and sends the call back
  for reconciliation;
- a transform or user edit creates a new generation, invalidating attestations bound to the
  old call digest;
- one witness may approve while another vetoes, or a quorum may become satisfied before a
  late duplicate arrives;
- cancellation wins before the execution lease, while cancellation after dispatch records
  user intent but cannot pretend an irreversible effect did not happen;
- continuation delivery may retry many times without rerunning the tool;
- a dependent call may become held when an upstream result changes, while unrelated calls
  continue.

### Joins, forks, and nested holds

A witness obligation is often a dependency graph rather than one request. For example,
`transfer_money` may require `(human approval AND fraud check) OR break-glass operator`, with
a compliance veto. The controller forks witness requests, folds responses under that
expression, and joins only when the rule is satisfied. A witness provider may itself propose
a governed tool call; that nested call receives its own `held_id` and dependency edge rather
than blocking the parent worker in memory. Cycle detection and a maximum dependency depth
must fail closed, because `A waits for B waits for A` is otherwise a durable deadlock.

The same graph permits speculative work without violating ordering. The scheduler may run
calls proven independent of the held result, but any call with a dependency edge remains
unready. "Pause the agent" is therefore a policy/UI projection of the dependency graph, not a
kernel-wide process suspension.

### Invariants across every loop

- **No witness executes the tool.** It only adds or removes source-bound evidence; the kernel
  remains the sole disposer.
- **No auto-allow after waiting.** Every execution lease is generation-, digest-, principal-,
  plugin-set-, and policy-epoch-bound and is revalidated at dispatch.
- **At-most-once effect, at-least-once reconciliation.** Events and work delivery may repeat;
  effects may not. A compare-and-swap lease plus tool idempotency/read-back closes the crash
  window.
- **No first-response-wins shortcut.** The declared join/veto rule folds concurrent and
  conflicting attestations deterministically.
- **Cancellation and expiry are facts, not queue deletion.** Late events remain auditable but
  cannot enable stale work.
- **Result admission is independent.** Witnessing a call never trusts its output;
  `QUARANTINE` remains a result-side decision.
- **Resume is idempotent and separable from execution.** Replaying a delivery notification
  cannot replay the effect.

## Does the whole agent pause?

That is a dependency and user-preference question, not part of the verdict itself.

- **Strict turn ordering (safe default):** park the turn until this call resolves. This preserves the normal tool-call/result sequence expected by most model APIs.
- **Independent-call concurrency:** other calls may proceed only when the planner supplied an explicit dependency graph proving they do not depend on the held result and cannot invalidate its witness contract.
- **Session backgrounding:** park this turn while the user starts or continues unrelated sessions. When the witness arrives, surface a resumable notification rather than stealing foreground focus.
- **Short synchronous wait:** a UI may hold an HTTP stream briefly for low-latency approval, then convert to a durable `202 Accepted`-style handle after a configured threshold. This is a transport optimization, not a different semantic.

The provider's KV cache may survive separately, but correctness must not depend on it. The durable continuation is the checkpoint plus references needed to reconstruct the next turn.

## Tool-call verbs: dispositions, not plugin names

The core vocabulary is easiest to understand as kernel dispositions over two checkpoints:

| Checkpoint | Verdict | Meaning |
|---|---|---|
| proposal | `ALLOW` | Execute the canonical proposal now. |
| proposal | `DENY` | Never execute this proposal. |
| proposal | `TRANSFORM` | Replace tool and/or arguments with a kernel-approved proposal, then adjudicate/execute that exact replacement. |
| proposal | `REQUIRE_WITNESS` | Persist and await an independently verifiable condition. |
| proposal | `DEFER` / `INDETERMINATE` | This adjudicator cannot decide; climb to another layer or fail closed. |
| result | `QUARANTINE` | Keep returned bytes out of model-visible context. |

These are not mutually interchangeable extension APIs. In particular:

- `TRANSFORM` is a **clean semantic result** for a plugin, but it is not itself the plugin entry point.
- `DEFER` means "ask the next adjudicator," while `REQUIRE_WITNESS` means "the decision is known: execution must wait for external evidence."
- `REQUIRE_WITNESS` is not a slow `ALLOW`; it creates durable state and an externally addressable obligation.
- registered open-range verdict kinds extend the wire vocabulary with fail-closed fallback; they do not authorize loading arbitrary plugin code or bypassing the core floor.

## Where plugins fit

A plugin system should expose narrow, typed stages around the stable verdict ABI rather than let extensions own execution:

```text
proposal
  -> canonicalizers / proposal transforms
  -> adjudicator chain
  -> witness-contract providers (only when held)
  -> kernel executor
  -> result admission filters
  -> audit/notification sinks
```

### 1. Proposal transform plugin

A transform plugin can normalize paths, redact forbidden fields, substitute a sanctioned tool, add a preview flag, or map a product-specific tool call into a canonical kernel call. It returns a proposed `TransformPayload`; the kernel then validates the payload, detects transform loops, applies capability checks to the replacement tool, and re-adjudicates the transformed call.

This makes `TRANSFORM` a good **meeting point** between plugins and the kernel, not a trust boundary. A plugin cannot transform `delete_everything` into an unreviewed shell escape and inherit the original call's authority.

Recommended contract:

- deterministic over declared inputs;
- versioned and content-digested;
- bounded output and runtime;
- no direct tool execution;
- declares which tool/argument fields it may change;
- produces an audit diff, not only replacement bytes;
- reaches a fixed point or fails on a bounded transform count.

### 2. Adjudicator plugin

An organization may add DLP, cost, compliance, ownership, or product-specific policy. It returns a core verdict or a registered restrictive verdict. Folding is monotone: an extension may narrow authority, request a witness, or defer, but cannot override the kernel's default-deny capability floor.

### 3. Witness-provider plugin

A witness provider translates a typed witness contract into an external request and verifies the response: GitHub review, Slack approval, CI status, git ancestry, a database read-back, another model, or an enterprise policy engine. The plugin never marks the held call `READY` directly; it submits an attestation to the kernel verifier.

### 4. Result-admission plugin

A result plugin can identify secrets, prompt injection, unsupported provenance, or schema violations. It may redact through a separately adjudicated transform or return `QUARANTINE`; it must not silently inject hidden instructions into model context.

### 5. Notification and audit plugin

These plugins project state changes to user surfaces. They are observers: failure to notify cannot change the authorization outcome, and replaying a notification cannot replay an effect.

## User-centric policy and preferences

Users should be able to choose **how stricter gates are experienced**, while policy owners retain control over the minimum floor.

Useful preferences include:

| Preference | Examples |
|---|---|
| Witness route | me, team reviewers, CI, named policy service, independent model class |
| Wait mode | block briefly, always background, notify-only, cancel on disconnect |
| Timeout | fail closed after 10 minutes; re-request after policy change |
| Scope | always witness money movement; witness writes outside workspace; never send source to third-party reviewers |
| Transform posture | auto-apply formatting/path canonicalization; preview semantic rewrites; forbid tool substitution |
| Disclosure | exact args to local approver; redacted summary to remote service |
| Resume UX | return to foreground, inbox item, webhook, continue only on explicit user click |
| Cost/latency | local verifier first, frontier model only if local defers |

The merge rule should be explicit:

```text
effective behavior = mandatory kernel floor
                   + organization policy
                   + project policy
                   + user preferences that only narrow or route
                   + per-call choice that only narrows or cancels
```

A user can say "ask me before every write" or "use CI rather than a reviewer for this claim." A user preference cannot say "ignore the organization's secret-exfiltration denial." Administrators may delegate selected widening choices through a witnessed policy-change syscall, but that is itself governed work, not a preferences toggle.

Plugin selection is also policy. A profile should pin plugin identity, version/digest, permissions, data egress, allowed stages, timeout, fallback, and precedence. Marketplace popularity is not authority.

## Minimal external protocol

A practical first spine needs only four durable command/query operations. These mutate or read the held-call resource; they do not encode a linear workflow:

```http
POST /tool-calls
  -> 200 result | 403 denial | 202 { held_id, witness_request, expires_at }

GET /held/{held_id}
  -> current state and bounded status

POST /held/{held_id}/attestations
  -> accepted-for-verification, duplicate, stale, or invalid

POST /held/{held_id}/cancel
  -> idempotent cancellation
```

A queue consumer and a polling client are then two adapters over the same reconciliation graph. A later event stream (`held.created`, `attestation.accepted`, `held.ready`, `call.completed`, `result.quarantined`) improves latency but should not be the source of truth.

The smallest end-to-end witness is: a fake `money_transfer` call returns `202`; a second principal submits a call-digest-bound approval; the kernel revalidates and executes it once; duplicate and out-of-order approval delivery does not duplicate the effect; a policy-epoch change sends a ready call back to evidence reconciliation; and result-delivery retry gives the parked session exactly one logical admitted result without rerunning the tool. A crash-after-effect/read-back case closes the most important non-linear window. Those tests are the proof that this concept has become execution rather than vocabulary.

## Design choices to keep separate

1. **Approval versus evidence.** Both can satisfy `REQUIRE_WITNESS`, but their contracts and verifier logic differ.
2. **Transform versus witness.** A transform may remove the need for a witness (for example, turn a destructive call into a preview), but the transformed call must be adjudicated from the start; a plugin cannot mutate a held call under an existing approval.
3. **Session resume versus effect execution.** The call can complete while the UI is offline. Resuming presentation is retriable; executing the effect is not.
4. **Protocol extension versus implementation plugin.** Registered verdict kinds preserve wire compatibility. Plugin loading, isolation, signing, capabilities, and lifecycle require a separate host contract.
5. **Context residency versus semantic continuity.** Keeping a provider cache warm is a performance optimization. The held-call ledger is the correctness mechanism.

## Recommendation

Treat `REQUIRE_WITNESS` as the kernel's durable `await` primitive, implemented by convergent reconciliation loops rather than a blocking pipeline, and `TRANSFORM` as one typed output available to proposal-stage plugins. Build the held-call event ledger, reducer/reconciler, and exact-once effect plus idempotent-delivery spine before adding a broad plugin marketplace. Then expose plugins at typed stages with pinned identities and monotone authority, and expose user preferences primarily as routing, latency, disclosure, and stricter-policy choices.
