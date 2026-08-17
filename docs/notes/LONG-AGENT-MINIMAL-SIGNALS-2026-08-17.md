---
title: "Minimal signals for supervising long-running agents"
description: "A small operator-first projection that separates healthy work, questionable motion, and decisions that need a human."
---

# Minimal signals for supervising long-running agents

A long-running-agent surface should answer one question first:

> **Do I need to act now?**

Everything else is drill-down. Operators should not have to infer health from token counts,
transcript motion, process uptime, or a wall of per-agent telemetry.

## The minimum projection

Show four fields, in this order:

| Field | Question answered | Evidence |
|---|---|---|
| **Attention** | Must a human act now? | `needs-human`, `watch`, or `none`, derived from typed refusal, expired lease, failed witness, or an explicit pending decision. |
| **Outcome** | Is useful work landing? | Latest independently witnessed artifact plus age: commit, closed issue, passing required gate, durable checkpoint, or declared task output. |
| **Current move** | What is the agent doing now? | One bounded activity label plus age: planning, tool call, waiting, testing, landing, or idle. |
| **Next check** | When should this be reconsidered? | A concrete condition or deadline: tool return, lease expiry, next watchdog tick, gate completion, or operator decision. |

The compact row is therefore:

```text
ATTENTION | OUTCOME + AGE | CURRENT MOVE + AGE | NEXT CHECK
```

Examples:

```text
none        | commit b6fe106 4m ago | testing 28s       | check when gate exits
watch       | no outcome for 24m    | repeated test 11m | intervene at 30m
needs-human | checkpoint 7m ago     | waiting 6m        | choose retry or stop
```

The first field must be visually dominant. A healthy row can collapse; an attention row must
name the exact action rather than merely saying `blocked`.

## Why these four survive compression

- **Attention** prevents the operator from polling every worker.
- **Outcome** distinguishes delivered progress from expensive motion. `fak progress` already
  applies this boundary at repository scope: uncommitted WIP alone is `WIP_ONLY`, not progress.
- **Current move** provides enough liveness context to avoid interrupting healthy work.
- **Next check** makes waiting bounded. Without it, `working` and `watch` become indefinite states.

No scalar percent complete is required. Agents rarely have a calibrated denominator, and a
plausible percentage can hide a missing witness or a repeated failure. If a plan has countable
criteria, show `3/5 criteria witnessed` only as outcome detail, never as the primary health signal.

## Evidence hierarchy

Use the strongest available signal and label weaker ones honestly:

1. **Witnessed outcome** — externally readable artifact or gate result.
2. **Durable checkpoint** — declared completed step with a resumable address.
3. **Tool transition** — current bounded operation and start time.
4. **Heartbeat** — process is alive, but progress is unknown.
5. **Transcript or token motion** — diagnostic only; never evidence of progress.

A heartbeat can disprove death. It cannot prove useful work. Likewise, token throughput and tool-call
volume belong in drill-down for cost or loop diagnosis, not in the default row.

## Derived states

Keep the state vocabulary closed and actionable:

- **needs-human** — an explicit decision, permission, credential, policy exception, or irreversible
  choice is pending. Show the choice and safe default.
- **watch** — liveness exists but witnessed outcome is stale, the same operation is repeating, a
  lease/checkpoint is near expiry, or a required gate is failing. Show the intervention threshold.
- **none** — the latest outcome is fresh or the current bounded operation is still inside its stated
  wait window.
- **unknown** — required evidence is missing. Never silently map unknown to healthy or stuck.

Thresholds should come from the operation's contract (timeout, lease TTL, retry budget, watchdog
cadence), not one global "idle for N minutes" constant. Ten minutes in a compile and ten minutes at a
permission prompt are different states.

## Default and drill-down

The default fleet view should show:

1. attention rows first;
2. changed rows since the previous look;
3. one folded healthy count, such as `12 healthy; next check <= 4m`.

Expand a row to show objective, owner, lane, parent/child lineage, recent tool transitions, retries,
cost/tokens, logs, and transcript. These are useful for diagnosis but too expensive and noisy for the
default control surface.

The operator brief and live-agent view serve different cadences:

- `fak operator brief` answers portfolio-level choices and changes across reports.
- the live projection above answers whether a running worker needs attention now.
- `fak progress` answers whether repository-level outcomes landed in a selected time window.

They should share the attention/outcome boundary instead of merging into one giant dashboard.

## Display invariants

1. Never label WIP, heartbeats, token motion, or tool volume as delivered progress.
2. Never show `blocked` without the blocker, owner, and next action.
3. Never show `working` without the current move's age and bounded next check.
4. Never hide unknown or stale evidence behind a green aggregate.
5. Preserve the last witnessed outcome while a new operation is in flight.
6. Prefer change since last review over replaying unchanged detail.
7. Keep healthy workers compressible to a count; keep exceptions individually addressable.

## Witness plan

The smallest implementation witness is a captured renderer test with four workers:

- fresh witnessed outcome plus active tool call -> `none`;
- live heartbeat but stale outcome -> `watch`;
- explicit operator decision -> `needs-human` with the choice;
- missing timestamps -> `unknown`, not healthy.

The test should assert the compact row contains only the four default fields and that transcript,
token, and tool-count details appear only after expansion. A live capture should then show the same
ordering against actual leases/checkpoints before this projection is treated as the default UI.
