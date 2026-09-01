---
title: "Headless run protocol v1"
description: "pkg/harnesskit exposes fak.harness.run/v1, a product-neutral event and input contract. Provider request, response, SDK,"
---
# Headless run protocol v1

This page is authoritative for run-event semantics. For layer selection, async ownership, queue acknowledgement, and public/private boundaries, see [Builder contract ladder](builder-contract-ladder.md).

`pkg/harnesskit` exposes `fak.harness.run/v1`, a product-neutral event and input contract. Provider request, response, SDK, and transport objects stop at the adapter boundary; public envelopes contain only semantic payloads declared in `protocol.go`.

## Ordering and replay

- A run is one totally ordered log. `sequence` starts at 1, increases by exactly one, and is immutable. `event_id` is stable and unique within the run. Consumers reject gaps, reordering, version changes, and cross-run records.
- `correlation_id` groups a request, tool call, approval, or input. `causation_id` names the event/input that directly caused a record. Neither substitutes for ordering.
- Inputs carry a caller-generated `input_id`, idempotent within `run_id`. A repeated ID acknowledges the original effect and never performs it twice.
- Consumers persist the exclusive `Cursor` only after applying through `sequence`. Reconnect calls `Resume(cursor, credit)`; the producer authenticates `checkpoint`, returns only records after the cursor, and returns the next cursor. Empty cursor means sequence zero.
- Credit bounds each delivery batch. Zero means the currently available tail for trusted in-process callers; transports should issue explicit `flow.credit` inputs. Blocking adapters obey `context.Context` cancellation.
- `run.cancel` is idempotent. Once accepted, no ordinary output may be appended; `run.canceled` and a final checkpoint are the terminal semantic records.

## Compatibility and unknown events

The major protocol string versions schemas and ordering semantics. Within v1, optional fields and new event types are additive. Consumers must validate the envelope, advance the cursor, and ignore an unknown event payload. They must not render it as prose, execute it, infer approval, or fail the run. Removing fields, changing meanings/order, or making an optional field required needs a new major version.

## Approval and sensitive-data boundaries

`approval.requested` creates the only valid target for `approval.resolve`; the decision must name that ID and be `approve` or `deny`. Approval grants only its declared scope and is not reusable authority. Unknown events can never grant approval. Adapters must classify envelopes as `public`, `private`, or `secret` before publication. `secret` payloads are always redacted at an untrusted consumer boundary; `private` requires an authenticated private consumer. Event metadata and artifact URIs must themselves be safe to expose. Artifacts carry references, not secret bytes.

## Semantic coverage and consumers

V1 covers run lifecycle, messages and deltas, structured UI blocks, tool lifecycle, plans, approvals, artifact references, typed errors, usage, cancellation, checkpoints, reconnect, and resume. First-party `fak harness protocol project --input FILE --view cli|tui|json` reads JSONL envelopes and projects CLI and TUI views from the same semantic reducer. It never parses terminal prose.

The independent recorded fixture is `docs/_witnesses/harness-protocol/roundtrip-events.jsonl`; `roundtrip-witness.json` records producer split/resume and independent CLI/TUI projection hashes.
