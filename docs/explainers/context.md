---
title: "Context management"
description: "The current builder route for fak context placement: the default behavior, supported controls, lifecycle boundaries, and one dry-run check."
slug: context
keywords:
  - fak context management
  - context planning
  - context shedding
  - resident context
  - context budget
  - context restore
  - managed context
  - long agent sessions
date: 2026-07-16
---

# Context management

**For builders running agents through fak:** fak's current default path plans a bounded
resident view, keeps the objective pinned, sheds compactible history without rewriting it
as a summary, and leaves recoverable handles for cold spans. Start by inspecting that
placement decision locally:

```console
fak debug --cmd context-plan-preview
```

The dry run needs no provider key. Its `PINNED`, `RECENT`, `DEEP`, `ELIDED`, and
`QUERY-NEEDED` groups show what would remain resident and what would be recoverable by
handle. That preview is the one next check for this page; use the deeper operator and proof
routes below only when the readout raises a specific question.

## Current default

On the `fak guard` and `fak serve` path, context placement is automatic:

1. Fak upgrades the cache horizon and plans a resident view under a token budget.
2. It carries the objective as a verbatim pin and preserves a stable warm prefix.
3. When compactible history exceeds its budget, it sheds suffix-shaped spans and records
   `fak_context_restore` handles instead of asking a model to summarize them.
4. It elides oversized tool results, can quarantine or page out results at write time, and
   restores recoverable content on demand or resume.

The shipped conservative defaults are an 8K planned resident view
(`--ctx-view-budget`) and history shedding at roughly 48K tokens
(`--compact-history-budget`). They are placement budgets, not claims that the provider's
advertised window is only that large. The selection doctrine, output reserve, and
provenance rules live in [Long-context defaults](../long-context-defaults.md).

This behavior reduces manual prompt housekeeping; it does not make every host fully
kernel-owned. On the Claude Code wire, the host can still trigger its own compaction. Fak
observes that event and keeps its own operations cache-preserving, but does not currently
suppress the host compactor. Treat the detailed wiring inventory and that limitation in
[You Never Manage the Context Window](you-never-manage-the-context-window.md) as the
current implementation boundary.

## Choose the operating mode

| Need | Supported choice | What changes |
|---|---|---|
| Ordinary agent run | Keep the automatic default. | Fak plans, pins, sheds, and restores without a context flag. |
| A measured task needs a different resident envelope | Set `--ctx-view-budget <tokens>`; change `--compact-history-budget <tokens>` only with a task-local witness. | The explicit budgets replace the conservative defaults for that run. |
| An operator needs a hard session envelope or reset policy | Use the `fak guard -h -all` budget and reset controls, then inspect the session readout. | Session governance is explicit; it is separate from per-turn placement. |
| Flat-context handoff with no transcript compaction | Follow the relay contract as design guidance only. | Relay mode is planned, not a shipped enforcement mode. Do not describe it as active. |

A larger number is not automatically better. Select the smallest resident envelope that
preserves the task's minimum viable context and output reserve; use a same-task witness
before presenting a larger envelope as a default.

## What the lifecycle preserves

A span can move from resident to cold without becoming durable memory. Fak retains a
digest or restore handle so the span can be queried again; promotion into durable memory
is a separate, evidence-bearing decision. Across a managed reset, continuity requires a
pinned objective, remaining budget, a deterministic reset record, and independently
queryable evidence—not a model-written recap treated as fact.

The current continuity contract and minimum operator readout are in
[Managed-context continuous usage semantics](../managed-context-continuous-usage.md).
The mechanics of pins, tombstones, restore handles, and value measurement are in
[Context shedding](context-shedding.md).

## Generation and support boundary

This page is the **current (`gen/now`) builder route**. It describes behavior wired into
current `fak guard`/`serve` execution plus CLI controls that are available now. Runtime
authority lives in the [`ctxplan`](../../internal/ctxplan/) placement code,
[`ctxmmu`](../../internal/ctxmmu/) admission code, and gateway message pipeline; the
focused proof ledger is [D3 · ctxmmu](../proofs/ctxmmu.md).

Dated notes, concept pages, and relay proposals explain why the design evolved, but they
do not override this route or the runtime. Promote a proposed behavior here only after it
is wired and witnessed. Demote a statement when a newer runtime authority or supported
mode supersedes it. For provider-specific limits, host integration status, or deployment
support, use the relevant integration page rather than inferring support from the generic
context mechanism.

## Deeper routes

- [Long-context defaults](../long-context-defaults.md) — budget selection and provenance.
- [Managed-context continuous usage semantics](../managed-context-continuous-usage.md) — reset and continuity contract.
- [Context shedding](context-shedding.md) — drop, pin, tombstone, restore, and value details.
- [You Never Manage the Context Window](you-never-manage-the-context-window.md) — doctrine, wiring survey, and historical rationale.
- [D3 · ctxmmu](../proofs/ctxmmu.md) — scoped implementation proof.

### Named, lazy, queryable sources

Managed placement answers which spans are resident. The complementary
programming interface is [Context as a variable](context-as-a-variable.md): a
task-scoped name resolves to immutable source bytes, then demand separately
fetches, queries/materializes, and admits a bounded derived view. The explainer
also distinguishes page/blob, plan, derived-view, call-outcome, and provider-KV
caches, and specifies why dereferencing a call-result binding never refreshes or
reissues the call.
