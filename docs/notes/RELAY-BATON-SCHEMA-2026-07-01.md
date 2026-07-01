---
title: "Relay baton schema"
description: "Data-only field specification for fak.relay.baton.v1."
date: 2026-07-01
---

# Relay baton schema

Status: data-only spec. No Go type or CLI consumes this format yet. This file
pins the first relay-baton wire contract so the future writer, reader, verifier,
and status surfaces build against one shape.

`fak.relay.baton.v1` is the handoff object a relay leg writes at a safe point for
its successor. It is deliberately pointer-only: it may point at commits, ledger
rows, issue numbers, memory slugs, and file globs, but it must not carry transcript
bytes or a model-written recap.

## Invariants

- **Closed shape.** Unknown fields are invalid. Field names are case-sensitive.
- **Pointer-only.** A baton carries handles into durable stores, not source
  payloads or transcript excerpts.
- **No claimed field.** A baton is invalid if any object contains a field named
  `claimed`. Progress is represented only by re-verifiable cursors.
- **Re-verify at read.** A reader must verify the objective digest, git/ledger
  cursors, artifact refs, tombstone reason, and held region before using the baton.
- **Stable JSON.** Emitters should write deterministic JSON object fields in the
  order documented below and stable-sort arrays whose order is semantically a set.

## Top-level Fields

| Field | Type | Required | Meaning and validation |
|---|---|---:|---|
| `schema` | string | yes | Must equal `fak.relay.baton.v1`. |
| `relay_id` | string | yes | Stable id for the whole relay. Non-empty; unchanged across every leg. |
| `leg` | integer | yes | Closing leg number. Must be `>= 0` and monotonic; equals `session.Generation` for the closing leg. |
| `parent_trace` | string | yes | Trace id of the closing leg. The successor records this as lineage, not as trusted progress. |
| `objective` | object | yes | The active `ctxplan.ObjectivePin`, carried verbatim. See below. |
| `done_when` | string | yes | One-line durable-store predicate the successor evaluates before doing more work. |
| `progress_cursor` | object | yes | Re-verifiable progress anchors. Contains no progress percentage and no self-report. See below. |
| `next_action` | string | yes | One line naming the next atomic action. Must not be a recap. |
| `open_questions` | array of strings | yes | Durable pointers or short labels for unresolved decisions. Empty array is valid. |
| `artifacts` | array of objects | yes | Durable pointers the successor may inspect. Empty array is valid only when no artifact exists yet. |
| `do_not_rederive` | array of strings | yes | Durable pointers to closed dead ends the successor should not retry. Empty array is valid. |
| `tombstone` | object | yes | The closing leg's typed exit record. See below. |

## `objective`

`objective` is exactly the JSON shape of `ctxplan.ObjectivePin`.

| Field | Type | Required | Meaning and validation |
|---|---|---:|---|
| `pin_id` | string | yes | Stable objective identity. Non-empty. |
| `text` | string | yes | Human-readable objective text. |
| `digest` | string | yes | `sha256` hex over `pin_id + NUL + text`, as produced by `ctxplan.ObjectivePin`. |
| `step` | integer | no | Audit ordering only; omitted when unknown. |
| `source_span_id` | string | no | Optional span id the objective was extracted from. |

A reader must run the same digest check as `ObjectivePin.Verify`. A mismatch is not
objective drift; it is corrupt baton input and must fail closed before execution.

## `progress_cursor`

`progress_cursor` is a set of anchors the successor re-reads. It never states how
much work is done.

| Field | Type | Required | Meaning and validation |
|---|---|---:|---|
| `start_sha` | string | yes | Git commit the closing leg used as its progress anchor. A reader verifies it exists and is an ancestor or the configured base. |
| `ledger_ref` | string | no | Intent ledger, run ledger, or DOS row to re-read for verified progress. If present, it must resolve. |
| `held_region` | array of strings | yes | Lane/path region the successor must re-acquire before writing. Empty is invalid for a write-capable relay. |

If the cursor no longer matches git, ledger, or the lease region, the read outcome
is `RELAY_BATON_STALE`; the successor re-derives from durable state instead of
trusting the baton.

## `artifacts`

Each artifact row is a pointer into a durable store.

| Field | Type | Required | Meaning and validation |
|---|---|---:|---|
| `kind` | string | yes | One of `commit`, `issue`, `memory`, `ledger`, or `file`. |
| `ref` | string | yes | Store-native reference: commit SHA, `#1234`, memory slug, ledger id, or repo-relative path/glob. |

Artifact rows do not carry bytes. A `file` artifact points to a path or glob; it
does not embed the file content.

## `tombstone`

| Field | Type | Required | Meaning and validation |
|---|---|---:|---|
| `reason` | string | yes | One of the relay reason tokens in [`RELAY-REASON-VOCABULARY-2026-07-01.md`](RELAY-REASON-VOCABULARY-2026-07-01.md). |
| `at_sha` | string | yes | Git commit observed when the closing leg wrote the baton. A reader verifies it exists. |
| `note` | string | no | Short operator note. It is display-only and must not be consumed as progress. |

`note` can explain why the reason fired, but it cannot substitute for
`progress_cursor`, `artifacts`, or `done_when`.

## Example

This example is syntactically valid and exercises every required field. The refs
are illustrative; a real reader must still re-verify them against git, the ledger,
and the issue tracker.

```json
{
  "schema": "fak.relay.baton.v1",
  "relay_id": "RLY-20260701-0001",
  "leg": 7,
  "parent_trace": "trace-relay-leg-7",
  "objective": {
    "pin_id": "pin-relay-schema",
    "text": "Ship the relay baton schema and close #1863.",
    "digest": "42e5a2de4275c715f5464585d135b13aa426e34e9829c22dd8f0eb72bf45a348",
    "step": 3,
    "source_span_id": "span:user:3"
  },
  "done_when": "A pushed commit resolves issue #1863 and dos commit-audit passes.",
  "progress_cursor": {
    "start_sha": "0123456789abcdef0123456789abcdef01234567",
    "ledger_ref": ".dos/runs/relay-demo.jsonl#L12",
    "held_region": [
      "INDEX.md",
      "docs/notes/**"
    ]
  },
  "next_action": "Run the schema doc witnesses and close #1863.",
  "open_questions": [
    "issue:#1870 decides the Go package for Baton"
  ],
  "artifacts": [
    {
      "kind": "issue",
      "ref": "#1863"
    },
    {
      "kind": "file",
      "ref": "docs/notes/RELAY-BATON-SCHEMA-2026-07-01.md"
    }
  ],
  "do_not_rederive": [
    "memory:relay-schema-freeform-draft"
  ],
  "tombstone": {
    "reason": "RELAY_ROTATED",
    "at_sha": "0123456789abcdef0123456789abcdef01234567",
    "note": "schema handoff ready; successor must verify refs"
  }
}
```

## Reader Contract

The minimum read ladder is:

1. Parse JSON and require the exact top-level field set.
2. Reject any field named `claimed` anywhere in the object tree.
3. Check `schema == "fak.relay.baton.v1"`.
4. Verify `objective.digest` from `pin_id` and `text`.
5. Check `tombstone.reason` against the relay reason vocabulary.
6. Verify git refs in `progress_cursor.start_sha`, `tombstone.at_sha`, and any
   `commit` artifacts.
7. Resolve `ledger_ref` and artifact refs that are present.
8. Re-acquire or refuse the `held_region` before any write.
9. Evaluate `done_when` before continuing; a done relay ends with
   `RELAY_GOAL_DONE` rather than launching another leg.

Any stale cursor, missing artifact, or objective mismatch fails closed. A future
reader may render the baton for humans, but it must not continue work from display
text alone.
