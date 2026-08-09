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

## State machine

```text
PROPOSED
   |
   | adjudication = REQUIRE_WITNESS
   v
HELD --publish--> WAITING
  |                  |
  | cancel/expiry    | attestation arrives
  v                  v
CANCELLED/EXPIRED  VERIFYING
                       |
             invalid / denied --> REJECTED
                       |
                    valid
                       v
                 REVALIDATING
                  /         \
     policy/input stale     still valid
              |                 |
           REJECTED              v
                              READY
                                |
                         CAS execution lease
                                v
                            EXECUTING
                                |
                    result admission gate
                       /                \
                 QUARANTINED          COMPLETED
                       \                /
                        RESUME CONTINUATION
```

Important properties:

- **No witness executes the tool.** It only satisfies a named gate; the kernel remains the sole disposer.
- **No auto-allow after waiting.** Policy, principal authority, plugin versions, argument digest, and time-sensitive preconditions are revalidated immediately before execution. A stale approval returns to `HELD` with a new request or fails closed.
- **At-most-once effect.** A compare-and-swap execution lease plus an idempotency key prevents two witness consumers or retries from running the same call twice.
- **One terminal decision.** Conflicting attestations fold fail-closed according to policy; the first arrival does not automatically win.
- **Cancellation propagates.** Cancelling the session or held call makes late attestations inert.
- **Result admission still applies.** Witnessing the call does not trust its output; `QUARANTINE` remains a separate result-side verdict.

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

A practical first spine needs only four durable operations:

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

A queue consumer and a polling client are then two adapters over the same state machine. A later event stream (`held.created`, `attestation.accepted`, `held.ready`, `call.completed`, `result.quarantined`) improves latency but should not be the source of truth.

The smallest end-to-end witness is: a fake `money_transfer` call returns `202`; a second principal submits a call-digest-bound approval; the kernel revalidates and executes it once; duplicate approval and replay do not duplicate the effect; the parked session receives exactly one admitted result. That test is the proof that this concept has become execution rather than vocabulary.

## Design choices to keep separate

1. **Approval versus evidence.** Both can satisfy `REQUIRE_WITNESS`, but their contracts and verifier logic differ.
2. **Transform versus witness.** A transform may remove the need for a witness (for example, turn a destructive call into a preview), but the transformed call must be adjudicated from the start; a plugin cannot mutate a held call under an existing approval.
3. **Session resume versus effect execution.** The call can complete while the UI is offline. Resuming presentation is retriable; executing the effect is not.
4. **Protocol extension versus implementation plugin.** Registered verdict kinds preserve wire compatibility. Plugin loading, isolation, signing, capabilities, and lifecycle require a separate host contract.
5. **Context residency versus semantic continuity.** Keeping a provider cache warm is a performance optimization. The held-call ledger is the correctness mechanism.

## Recommendation

Treat `REQUIRE_WITNESS` as the kernel's durable `await` primitive and `TRANSFORM` as one typed output available to proposal-stage plugins. Build the held-call ledger and exact-once resume spine before adding a broad plugin marketplace. Then expose plugins at typed stages with pinned identities and monotone authority, and expose user preferences primarily as routing, latency, disclosure, and stricter-policy choices.
