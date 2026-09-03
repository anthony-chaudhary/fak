---
title: "Progress-state defaults for plans and reports"
description: "Report delivery, evidence maturity, and next movement as independent axes without weakening acceptance gates or erasing shipped implementation."
---

# Progress-state defaults

Use these defaults for plans, issue rollups, and operator reports. They separate what shipped, how mature the evidence is, and what can move next. A failed or missing performance receipt changes the evidence axis; it does not erase delivered implementation or force the whole item into a static `HOLD`.

This vocabulary changes progress reporting only. It does **not** weaken acceptance gates, convert provisional results into claims, or authorize a fallback engine.

## Report three independent axes

Every active or recently completed item reports one value on each axis:

`PRODUCT / EVIDENCE / QUEUE`

Do not collapse the axes into one overall `KEEP`, `REJECT`, or `HOLD` label. Keep those terms in evidence history when they describe an actual past campaign decision.

### Product state

| State | Meaning |
|---|---|
| `NOT_STARTED` | No reusable implementation or campaign spine has shipped. |
| `SPINE_SHIPPED` | A bounded validator, harness, receipt schema, or end-to-end campaign path has shipped. |
| `IMPLEMENTATION_SHIPPED` | The intended product-path implementation has shipped in its stated envelope. |
| `PRODUCTION_READY` | Implementation, operating safeguards, and required acceptance evidence are complete for the stated envelope. |

Product state describes delivered scope, not benchmark success.

### Evidence state

| State | Meaning |
|---|---|
| `NO_EVIDENCE` | No qualifying contract or runtime evidence exists. |
| `CONTRACT_VALIDATED` | Deterministic tests or validators prove the bounded contract. |
| `RUNTIME_READY` | Exact artifact, command, host envelope, validator, outputs, and thresholds are prepared for execution. |
| `CAMPAIGN_RUNNING` | A sanctioned run is live and has an authoritative handle or receipt stream. |
| `RESULT_PROVISIONAL` | Real runtime evidence exists but misses a named qualification dimension. It may choose the next experiment, but cannot support a benchmark claim. |
| `RESULT_QUALIFIED` | The evidence satisfies every gate required for the stated claim and envelope. |

`BOUNDED_ACCEPTANCE` is an evidence annotation used with `CONTRACT_VALIDATED`: a named validator, correctness, or mechanism scope is fully accepted while a broader runtime or product campaign remains incomplete. Always name the accepted boundary and the missing broader evidence.

### Queue state

| State | Meaning |
|---|---|
| `ACTIVE_PROBE` | A bounded fak-native experiment can reduce material uncertainty now. |
| `DEPENDENCY_ADVANCING` | The target cannot run yet, but a concrete dependency edge is actively being removed. |
| `READY_TO_RUN` | The complete campaign packet is prepared; only sanctioned execution or capacity remains. |
| `QUEUED_BEHIND <issue>` | An ordered predecessor owns the next movement. |
| `CAPACITY_BLOCKED` | The packet is complete, but no currently available sanctioned node meets its declared envelope. Keep the dispatch command ready. |
| `EXTERNAL_BLOCK` | Local, fleet, and dependency-clearing routes are exhausted and movement needs an external state change. Name the reactivation condition. |
| `PARKED_LOW_VALUE` | Work is reversibly deprioritized because another action has higher expected value. Name the reactivation condition. |
| `COMPLETE` | No additional movement is required for the stated scope. Broader follow-on work, if any, has a separate owner. |

A queue label must identify movement, not merely explain why work stopped. `CAPACITY_BLOCKED`, `EXTERNAL_BLOCK`, and `PARKED_LOW_VALUE` require a concrete reactivation condition.

## Delivery credit and performance credit

Track these ledgers separately.

**Delivery credit** may be awarded for a witnessed artifact within its stated scope, including:

- validator or receipt spine shipped;
- correctness or implementation shipped;
- exact campaign packet prepared;
- dependency edge removed;
- sanctioned run launched with an authoritative handle;
- evidence independently reproduced.

**Performance credit** is awarded only when the complete claim gate passes. For native-inference work that means, at minimum:

- exact artifact and quantization identity;
- `engine=fak-native` for the product arm;
- fallback count `0`;
- strict quality acceptance;
- inclusive request-boundary accounting, including setup, recovery, and verification overhead;
- a one-variable matched comparison in the declared operating envelope.

llama.cpp and MLX may be explicit comparison or reference arms only. They never become fak product fallbacks and never earn fak-native performance credit. A shipped mechanism, accepted validator, provisional result, issue closure, or historical `KEEP`/`REJECT`/`HOLD` label cannot substitute for the full gate.

## Movement defaults

1. Choose the smallest next action that can change the evidence state or remove a dependency.
2. Keep at most three items in `ACTIVE_PROBE` across one plan. Finish, queue, or park one before opening another.
3. Allow any number of complete `READY_TO_RUN` packets; readiness is useful inventory, not active WIP.
4. Use `DEPENDENCY_ADVANCING` only when a named edge and owner are moving now. Otherwise use `QUEUED_BEHIND`, `EXTERNAL_BLOCK`, or `PARKED_LOW_VALUE` honestly.
5. Reserve `EXTERNAL_BLOCK` for a true impasse after local, fleet, and dependency-clearing routes are exhausted.
6. Use `PARKED_LOW_VALUE` instead of indefinite `HOLD`; record why another action is better and what would reactivate this item.
7. A provisional result must name the missing qualification dimension and the next discriminating test.
8. Historical campaign decisions stay in the execution log. Current state reports the three axes and the next movement.

## Plan rendering

In **Current state**, render each phase or row with:

| Item | Product | Evidence | Queue | Next movement |
|---|---|---|---|---|
| Example | `SPINE_SHIPPED` | `CONTRACT_VALIDATED` (`BOUNDED_ACCEPTANCE`: validator only) | `READY_TO_RUN` | Dispatch the pinned packet on the named sanctioned node. |

In **Execution log**, preserve dated facts such as a campaign entering `HOLD`, a result being rejected, or a mechanism landing. Historical decisions remain evidence; they no longer masquerade as the whole current state.
