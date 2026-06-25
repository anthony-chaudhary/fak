---
title: "fak shared task record contract"
description: "Defines fak's adapter-neutral shared-task JSON envelopes, merge rules, and in-memory reference fold for humans and agents co-editing a task board."
---

# Shared task record contract

This is the concrete adapter-neutral contract for user-level shared task state.
It is the rung where humans and agents can co-edit a task board or plan without
turning edits into unstructured chat.

The runtime reference is `internal/sharedtask`; the executable fixture validator
is `tools/shared_task_contract.py`.

## Envelopes

Task record:

```json
{
  "schema": "fak.shared-task.v1",
  "task_id": "task_shared_demo",
  "rev": "sha256:taskrev001",
  "state": "working",
  "title": "Coordinate the shared release checklist",
  "body_ref": {
    "kind": "cas",
    "digest": "sha256:body001",
    "bytes": 512,
    "taint": "tainted",
    "scope": "fleet",
    "durability": "session"
  },
  "artifacts": [],
  "notes": [],
  "open_decisions": [],
  "updated_by": {"kind": "agent", "id": "planner"},
  "updated_at": "2026-06-25T00:00:00Z"
}
```

Patch:

```json
{
  "schema": "fak.shared-patch.v1",
  "task_id": "task_shared_demo",
  "base_rev": "sha256:taskrev001",
  "actor": {"kind": "human", "id": "editor"},
  "scope": "fleet",
  "durability": "session",
  "ops": [
    {"op": "replace", "path": "/title", "value": "Coordinate the scoped release checklist"}
  ],
  "message": "Rename the collaborative task."
}
```

Accepted result:

```json
{
  "schema": "fak.shared-patch-result.v1",
  "task_id": "task_shared_demo",
  "base_rev": "sha256:taskrev001",
  "current_rev": "sha256:taskrev002",
  "verdict": "accepted",
  "reason": "",
  "event_id": "evt_title_001",
  "record_ref": "sha256:taskrev002"
}
```

Event:

```json
{
  "schema": "fak.shared-event.v1",
  "event_id": "evt_title_001",
  "task_id": "task_shared_demo",
  "prev_event": "",
  "event_kind": "patch_accepted",
  "actor": {"kind": "human", "id": "editor"},
  "base_rev": "sha256:taskrev001",
  "next_rev": "sha256:taskrev002",
  "scope": "fleet",
  "durability": "session",
  "taint": "tainted",
  "patch_digest": "sha256:patchtitle001",
  "verdict": "accepted",
  "reason": "",
  "ts": "logical:1"
}
```

Disaggregated artifact ref:

```json
{
  "schema": "fak.shared-artifact-ref.v1",
  "artifact_id": "art_remote_trace",
  "ref": "sha256:remoteartifact001",
  "media_type": "application/json",
  "taint": "tainted",
  "scope": "tenant",
  "store": "l3-kv",
  "deletion_certificate": "sha256:deleteartifact001"
}
```

Materialized journal:

```json
{
  "schema": "fak.shared-task-journal.v1",
  "task_id": "task_shared_demo",
  "initial": {
    "schema": "fak.shared-task.v1",
    "task_id": "task_shared_demo",
    "rev": "sha256:taskrev001",
    "state": "working",
    "title": "Coordinate the shared release checklist",
    "body_ref": {
      "kind": "cas",
      "digest": "sha256:body001",
      "bytes": 512,
      "taint": "tainted",
      "scope": "fleet",
      "durability": "session"
    },
    "artifacts": [],
    "notes": [],
    "open_decisions": [],
    "updated_by": {"kind": "agent", "id": "planner"},
    "updated_at": "2026-06-25T00:00:00Z"
  },
  "entries": [],
  "digest": "sha256:journal001"
}
```

## Merge Rules

| Operation | Auto-merge? | Notes |
|---|---:|---|
| append note | yes | note id must be new; body is a scoped ref |
| append artifact | yes | artifact id must be new enough for the adapter's policy |
| append open decision | yes | decision id must be new |
| replace `/title` or `/state` | no | requires current base; stale writers get a conflict |
| replace `/body_ref` | no | external refs need deletion certificate |
| replace open decision state | no | stale or missing decisions conflict |

## Runtime Reference Fold

`internal/sharedtask` ships the in-memory reference behavior:

- accepted patches advance the materialized record revision and emit an event row;
- `replace /title` and `replace /state` are current-base scalar edits;
- stale non-commuting writes return a typed conflict body with base, current, and
  proposed values;
- stale append-only notes, artifacts, and open decisions can merge;
- decision resolution is `replace /open_decisions/<decision_id>/state`;
- disaggregated artifact, note-body, and task-body refs need digest-shaped
  deletion witnesses;
- `View` redacts task snapshots by reader scope and quarantine policy;
- `EventsView` applies the same policy to historical event catch-up;
- `PublishEventScoped`, `ApplyAndPublishScoped`, and `SubscribeScopedView` keep
  future live events on per-reader-scope topics;
- `Journal` and `LoadJournal` move one task's initial record plus accepted event
  snapshots across a process boundary as data, not as a hosted service.

## Validate

```bash
python tools/shared_task_contract.py validate-doc docs/shared-task-record-contract.md
python tools/shared_task_contract.py validate-sequence examples/shared-task-record
python tools/shared_task_contract.py validate-verdicts examples/shared-task-record-verdicts
```

## Honest Scope

This is a contract document plus a small in-memory reference fold. It is not a
networked task-store daemon, not a durable mailbox, not an external L3 transport,
and not a browser/editor UI.
