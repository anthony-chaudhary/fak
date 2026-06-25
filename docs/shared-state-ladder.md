---
title: "fak shared state ladder"
description: "The five-rung vocabulary for shared state in fak: live messages, shared objects, durable handoff, disaggregated state, and human-agent collaboration."
---

# Shared state ladder

This is the vocabulary for shared state in fak. It keeps five related but
different things from being treated as one feature:

| Rung | Meaning | Shipped shape |
|---|---|---|
| Live addressed message | One value delivered now between agents | `internal/a2achan` `Send`/`Recv` and `Publish`/`Subscribe` over scoped `Ref` bodies |
| Live shared object | A named mutable cell during one run | Planned first-class region/window work; today compose from refs and messages |
| Durable handoff | State that survives a process/session/window boundary | Session image and snapshot primitives; `internal/sharedtask` journal is a task-local materialized handoff |
| Disaggregated state | Bytes live outside the task record | Shared task refs carry digest, taint, scope, store, and deletion certificate |
| User-level collaboration | A human and agents co-edit task state | Shared task record contract plus in-memory fold, conflicts, scoped views, and scoped live topics |

## Collaboration Spine

The collaborative task surface uses a single record with explicit patch
semantics:

1. A task has a stable `task_id`, current `rev`, scoped body ref, artifact refs,
   note refs, and open decisions.
2. A human or agent submits a patch against the `rev` it saw.
3. Append-only notes, artifacts, and open decisions can commute when their ids
   are new.
4. Scalar edits such as `replace /title` and `replace /state` require the
   current base and return a typed conflict when stale.
5. Disaggregated refs must carry digest shape and deletion witnesses before the
   record advances.
6. Adapters read `View` and `EventsView` so scoped readers get the same
   redaction behavior for snapshots and catch-up event history.
7. Live adapters use `SubscribeScopedView` and scoped event topics so future
   tenant-scoped events do not land in a fleet-scoped inbox.

## Honest Scope

Shipped today: in-process `a2achan` delivery, scoped/tainted refs, the
`internal/sharedtask` in-memory patch fold, scoped read and event-history
projection, scoped live event fanout, disaggregated ref admission, and a
materialized task journal. Not claimed here: a durable cross-process mailbox, a
networked task-store daemon, external L3 artifact transport, or a browser/editor
UI.
