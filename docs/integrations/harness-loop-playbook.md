---
title: "Reusable harness-loop playbook for customer workflows"
description: "How to wrap a customer workflow in a safe agent loop: selector, executor, witness/closer, and budgeted stop policy, with examples for support, data QA, and eval runs."
---

# Reusable Harness-Loop Playbook

A harness loop is the smallest useful product shape for agent automation. It lets an
agent repeat a workflow only after the previous tick produced evidence strong enough
to continue. The payload can be support tickets, data batches, eval cases, content
reviews, compliance checks, or any queue with a checkable outcome.

Do not start with "let the agent keep going." Start with this contract:

| Part | Question it answers | Customer artifact |
|---|---|---|
| Selector | Which single unit is ready now? | Queue query, ranker, or issue/ticket filter |
| Executor | What bounded action may run on that unit? | Tool allow-list plus one action recipe |
| Witness / closer | What proves the action landed? | Receipt, diff, validator result, audit row, or human approval |
| Stop policy | When must the loop stop, retry, or ask? | Budget envelope plus closed refusal reasons |

The loop is useful because these parts stay separate. A selector is not a proof. An
executor's final message is not a witness. A retry is not allowed just because a model
wants another turn.

## Minimal Config

Before live automation, write one small record like this for each workflow:

```yaml
loop: support-refund-review
surface: support_tickets
selector:
  query: status=open priority>=high missing_human_gate=false
  ordering: oldest_sla_first
executor:
  action: draft_refund_response
  allowed_tools:
    - search_kb
    - get_order_status
    - draft_ticket_reply
witness:
  close_on:
    - ticket_reply_created
    - policy_check=pass
  hold_on:
    - refund_payment_required
    - identity_mismatch
    - policy_check=unknown
stop_policy:
  max_retries_per_ticket: 1
  max_tokens_per_ticket: 8000
  max_wall_seconds: 300
  ask_human_on:
    - missing_authority
    - irreversible_action
    - ambiguous_policy
```

This is deliberately plain. The important property is that every field can be checked
outside the model's self-report.

## Worked Example: Support Reply Loop

| Field | Value |
|---|---|
| Selector | Pick the oldest open priority ticket whose account context is complete and whose requested action is allowed by the support policy. |
| Executor | Search the KB, inspect order status, and draft a reply. Do not execute refunds or account changes. |
| Witness / closer | Close only when the ticket system records a draft reply and the policy checker marks it `pass`. |
| Stop policy | Stop on missing account authority, refund-required class, identity mismatch, policy uncertainty, one failed retry, or token budget breach. |

What can be automated:

- selecting the next ticket;
- gathering allowed read-only context;
- drafting a response;
- marking a ticket ready for human send when the witness is present.

What must refuse or ask:

- any irreversible account action outside the allow-list;
- any reply whose policy check is missing or unknown;
- any ticket whose identity or authorization evidence is incomplete.

This is the same loop shape as the dev fleet, but the witness is a ticket-system and
policy receipt rather than a commit.

## Three Customer Workflow Patterns

| Workflow | Selector | Executor | Witness / closer | Stop policy |
|---|---|---|---|---|
| Support-ticket actions | Next SLA-risk ticket with complete account context | Read account/order state, search KB, draft reply or prepare allowed action | Ticket diff, policy result, CRM receipt, optional human send approval | Stop on irreversible action, missing authority, unknown policy, retry/token cap |
| Data-ingestion QA | Next failed batch ranked by severity and freshness | Normalize fields, enrich from allowed source, quarantine bad rows | Schema validator, row counts, checksum, sampled review row | Stop on source ambiguity, validator regression, repeated row-class failure |
| Eval or benchmark runs | Next stale case, failing slice, or changed model route | Run one bounded benchmark slice | Stored result row, score/parity gate, artifact checksum | Stop on budget breach, score regression, missing artifact, dirty baseline |

The loop can be local and single-worker first. Fleet dispatch comes later, after the
selector can prove that many ready units are independent.

## Evidence Checklist Before Live Retries

Use this checklist before turning on autonomous retry or multi-worker dispatch:

- The selector excludes units already in flight.
- The executor has a tool allow-list and one bounded action recipe.
- The witness is produced by a system other than the executor's final message.
- The stop policy has token, wall-clock, retry, and human-ask limits.
- Every refusal reason is closed vocabulary, not free-form prose.
- Re-running the same unit is idempotent or has a duplicate-prevention key.
- A failed witness creates a hold state instead of another blind retry.
- The loop records selected unit, action, witness, cost, and outcome per tick.

When this checklist is incomplete, keep the loop in dry-run or human-review mode.

## When to Add a Fleet

A fleet loop is useful only when the bottleneck is real parallel work. Do not raise
worker count until the limiting term is known:

```text
effective_workers = min(requested_workers, host_cap, seat_cap, lease_cap, ready_work_cap)
```

If `ready_work_cap` is low, improve the selector. If `lease_cap` is low, partition
the surface. If `seat_cap` is low, adding workers increases contention. If witness
latency dominates, improve the closer before increasing worker count.

## When to Add a Super Loop

Add a super loop when the hard question changes from "which unit is next?" to
"which loop deserves attention next?" A super loop is a read-first selector over
loops. It walks member loops, ranks dark/stale/high-debt members first, and points
the operator at the next loop to enter.

That is different from a worker launcher. A launcher starts executors after admission
passes. A super loop should be safe to run as orientation because it mutates nothing
at its own altitude.

Use this when a customer has several loops, such as support, ingestion, evals, and
compliance review, and needs one control surface for the portfolio.

## Where fak Fits

`fak` provides the boundary and evidence surfaces that make this loop shape practical:

- [`../dispatch-loop.md`](../dispatch-loop.md) shows a fleet loop over GitHub issues:
  select, spawn, witness, close.
- [`../super-loops.md`](../super-loops.md) explains the read-first super loop that
  chooses which member loop to enter.
- [`../loops-user-value.md`](../loops-user-value.md) explains the end-user value:
  token/cost controls, fleet efficiency, harness reliability, and super-loop steering.
- [`../managed-context-continuous-usage.md`](../managed-context-continuous-usage.md)
  explains how long runs preserve objectives, recoverable history, and budget
  envelopes across resets.
- [`../fak/dojo.md`](../fak/dojo.md) shows the prediction -> run -> measure -> score
  loop for token-saving levers.
- [`../cache-value-rollup.md`](../cache-value-rollup.md) keeps witnessed fak reuse and
  observed provider savings in separate accounts.

## Honest Scope

This playbook is not a new generic harness runtime, and it does not make every
workflow safe to automate. It is a design contract for deciding what can run, what
must be witnessed, and when the system must stop. If the witness is absent, the
correct state is `not yet`, not "agent says done."
