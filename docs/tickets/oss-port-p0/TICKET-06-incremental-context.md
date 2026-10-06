# feat(agent): update only the changed context layer and reuse the rest

## Parent context

Coordinates with parent #10701 (https://github.com/anthony-chaudhary/fak/issues/10701).

Direct port/adapt of the incremental context update studied in opencode at revision
`4eb29a6` (MIT). The concept is the layering: a long agent session holds context in
several layers — a stable Source layer (system prompt and agent scaffold), a Registry
layer (accumulated tool/turn history), and a Snapshot layer (the rendered, tokenized
form actually fed to the model). Sending the whole context again when only one layer
changed is the dominant cost of a long session; the update path must send the changed
layer and reuse the rest.

<!-- fak-agent-key: oss-port-p0-incremental-context -->
<!-- fak-cross-key: oss-port-p0-incremental-context -->
<!-- fak-public-issue: anthony-chaudhary/fak#10701 -->

```routing
lane: agent/incremental-context
paths: ["internal/agent/inkernel_planner.go", "docs/tickets/oss-port-p0/TICKET-06-incremental-context.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#10701
cross_key: oss-port-p0-incremental-context
```

Process cause: verification-gap

## Scope class

single-package

## Current state

`internal/agent/inkernel_planner.go` builds the whole request prompt from the whole
message list on every call: line 418 renders every message through
`renderInKernelChatMLRequest(messages, tools, p.m.Cfg, ...)` into one fresh string, and
the encode step in `internal/agent/prompt_encoding.go` (line 80) renders the same full
message list again before tokenizing it. Nothing between those two points records which
layer of the context was already rendered and tokenized for the previous turn, so the
render and the tokenize both restart from length zero on each turn of the same
conversation.

There is reuse machinery below this point: the KV radix path already reports
`cacheable`, `matched` and `computed` counts and `TestInKernelReuseMultiTurnPrefillSavings`
covers it. That reuse spares prefill *compute* under the model. It does not spare the
render and tokenize *above* it, which is what this ticket's seam owns.

## For

Long agent sessions on the native engine path, where a conversation accumulates many
turns and only the newest turn changes between requests.

## Problem

Every request re-renders and re-tokenizes the entire accumulated context, so the cost of
a session grows with the total conversation length even though the per-turn delta is a
single new message.

## Today

Turn N of a session pays a full render plus a full tokenize of turns 1..N. The stable
layers are recomputed identically to the previous request.

## Better

The stable layers of a session's context are held once and reused, and only the layer
that actually changed between consecutive requests is re-rendered and re-tokenized.

## Why this is next

The concept is ported directly from a permissive upstream (opencode @ `4eb29a6`, MIT) and
the seam is a single well-known function on the hottest planner path. It is the cheapest
unit that makes per-turn cost proportional to the turn rather than to the session, and
it is disjoint from every other lane in this wave.

## Working spine

Consecutive requests on one session -> the stable context layers are rendered and
tokenized once and held -> a request that changes only the newest layer re-renders and
re-tokenizes only that layer -> the assembled prompt is byte-identical to what a full
rebuild would have produced.

## Core through-line

Reuse the unchanged context layers across consecutive requests, and prove the reuse by
comparing the incrementally assembled prompt against a full rebuild rather than asserting
a counter went down.

## Gold-plating boundary

No cross-request prompt cache eviction policy, no change to the KV radix reuse path that
already exists below this seam, no message compaction or elision rework, no new
rendering template, no token-savings percentage target, and no configuration flag. This
leaf adds the layer memo and its proof; it does not re-architect context assembly.

## Verifiable Witness

```bash
go test ./internal/agent/... -run 'TestInKernelIncrementalContext' -count=1
```

The test must fail against the parent tree. It asserts that two consecutive requests
whose context differs only in the newest layer produce a prompt identical to a full
rebuild of the second request, while only the changed layer is re-rendered and
re-tokenized.

## Done condition / witness

Witness: `go test ./internal/agent/... -run 'TestInKernelIncrementalContext' -count=1`.

- [ ] [SW-VERIFIED] The incremental path exists at `internal/agent/inkernel_planner.go`
  and names the Source, Registry and Snapshot layers it holds and reuses.
- [ ] [SW-VERIFIED] A test asserts the incrementally assembled prompt is byte-identical
  to a full rebuild of the same request, and the test FAILS against the parent tree and
  PASSES after the change — a test that only counts calls does not satisfy this ticket.
- [ ] [SW-VERIFIED] A test asserts that a change confined to one layer re-renders only
  that layer, naming the reuse path rather than the component in isolation.
- [ ] [HW-WITNESSED] A per-turn prompt-token saving is measured on a live engine.
  `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue or
the resolving commit. The test named above does not exist yet; adding it is part of this
ticket's work.

## Acceptance gate

The focused agent test is green, it fails against the parent tree, and the routing fence
and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `agent` leaf in
its commit subject.

## Witness

The focused agent test above is the completion witness for the layer memo and its
equivalence proof. The per-turn saving measurement is a separate hardware read against a
live engine and must cite its own receipt; a green software test is not that measurement.

## Likely files

- `internal/agent/inkernel_planner.go` — the seam that re-renders and re-tokenizes the
  full message list on every request.
- `docs/tickets/oss-port-p0/TICKET-06-incremental-context.md` — routing and acceptance
  contract.

## Lane

agent/incremental-context

## Blast radius and affected lanes

One package: `internal/agent`. The KV radix reuse path, the model backends and the
gateway remain outside the leaf. `internal/agent` is a serialized lane; a concurrent
change in that package must land first or this leaf waits for it.

## Quarantined fallback mechanism

The current full-render path continues unchanged while this leaf is absent, and it stays
the fallback for any request whose context does not cleanly decompose into the held
layers. If an incremental assembly cannot be proven byte-identical to a full rebuild, the
planner must fall back to the full rebuild rather than emit a prompt that differs.

- Centrality: Enabling (a long agent session re-renders and re-tokenizes only its changed context layer)
- P1 Context: advanced — the repeated per-turn full-context cost is removed at the seam that owns it.
- P2 Net value: advanced — per-turn cost becomes proportional to the turn, not the session.
- P3 Adaptation: preserved — existing render templates and the KV radix path are unchanged.
- P4 Operations: advanced — yields one bounded dispatchable P0 unit.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #10701.

## Completion standard

development

## Target operating envelope

- re-encoded prompt tokens per turn: not applicable (development leaf; the per-turn saving is a separate hardware receipt)

## Witnessed operating envelope

- re-encoded prompt tokens per turn: not applicable (no incremental context path exists yet, so no saving can be witnessed)
