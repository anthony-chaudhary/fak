# feat(trajectory): deterministically reconstruct trace state and report the first divergence

```routing
lane: trajectory
paths:
  - pkg/traceabi/**
  - internal/trajectory/**
expected_steps: 8
```

<!-- fak-trace-foundation-key: deterministic-state-reconstruction-v1 -->

## Parent context

#6053. Reusable replay primitive with hard prerequisites `typed-causal-envelope-v1` and `many-to-many-value-lineage-v1`. It consumes their public `pkg/traceabi` envelope and lineage types through an internal adapter rather than redefining them. `bounded-private-evidence-policy-v1` is also required so incomplete capture, truncation, and evidence ceilings survive replay. It is narrower than #6053's global coordinator replay and broader than #6528's context-view replay; consumers supply domain reducers.

## Why this is next

Capturing and querying a trace still leaves correctness unproven. A pure reconstruction contract must exist before domain tracers claim replayability with incompatible local meanings.

## Problem

For an agent investigating a regression, displaying recorded events is not enough: it must reconstruct the state those events imply and identify the first point where observed and replayed state disagree. Today fak has several narrow replay paths with different inputs and assertions. Better because one deterministic fold contract makes replay honest, side-effect-free, schema-pinned, and able to distinguish missing evidence from true nondeterminism. Witness: ordered and shuffled equivalent histories reconstruct the same checkpoints, while one tampered transition reports the exact first divergent event and expected/actual digests.

## Centrality and P1-P4

**Centrality: Enabling correctness.** This makes tracer evidence reproducible rather than merely viewable.

- P1 Context: advanced - checkpointed reconstruction retrieves only the state slice needed for diagnosis.
- P2 Net value: preserved - replay is offline and side-effect-free; consumed events/bytes and unsupported rows are counted.
- P3 Adaptation: advanced - reducers are versioned/pure, ordering is canonical, side effects are forbidden, and gaps remain indeterminate.
- P4 Operations: advanced - decision, tensor, context, cache, and agent-flow tracers can share one replay/result vocabulary.

## Current state

At `f21eb416d92f03286747f84e888fb7dd1038c7a9`:

- `cmd/fak/audit_replay.go:21-43` folds repeated tool+args-digest decisions to detect inconsistent verdicts, but explicitly cannot reconstruct raw args or re-execute state.
- `internal/sessionreplay/replay.go:14-41` re-adjudicates one captured tool call under one regime-specific policy.
- `internal/compute/kvreplay_trace.go:74-102` replays one KV eviction corpus through policy/oracle simulators.
- `internal/trajectory/event.go:128-157` strictly encodes/decodes events, while `:166-203` projects message events into turns; there is no generic reducer/checkpoint/divergence contract.

The capability is **PARTIAL**: deterministic domain replays exist, but no shared reconstruction kernel or first-divergence receipt exists.

## Dependencies and schedule

- Hard prerequisite: `typed-causal-envelope-v1`.
- Hard prerequisite: `many-to-many-value-lineage-v1`.
- Hard prerequisite: `bounded-private-evidence-policy-v1`.
- Schedule after TICKET-01 through TICKET-04 because TICKET-04 and this issue both extend `pkg/traceabi/**`.

## Core through-line

Add public, data-only `Reducer`, `ReplayManifest`, `StateCheckpoint`, and `ReplayResult` contracts to `pkg/traceabi`, plus a pure replay engine in `internal/trajectory` that operates on those contracts. A versioned reducer consumes canonical envelope, lineage, capture, and loss records and returns canonical state bytes/digest. Replay validates schema/reducer/config/source digests, orders only by explicit causal/sequence rules, records periodic state checkpoints, and compares optional recorded `before_state`/`after_state` digests. Results are `reconstructed`, `diverged`, `incomplete`, or `unsupported`, with the first divergent event/span, expected/actual digests, missing prerequisites, counts, and final digest. The public contract enables private domain reducers without importing `internal/*`; it must not introduce alternative causal IDs, lineage operations, evidence classes, or capture accounting.

## Working spine

1. Define stdlib-only `Reducer`, `ReplayManifest`, `StateCheckpoint`, and `ReplayResult` contracts in `pkg/traceabi` with an external-consumer test.
2. Require reducer/config/schema/source digests and a deterministic initial state.
3. Topologically order causal events with stable sequence/ID tie-breaks.
4. Reject cycles/duplicate IDs; report gaps and unknown kinds as incomplete/unsupported.
5. Compare before/after digests and stop at the first divergence.
6. Bound events, input bytes, state bytes, checkpoints, and wall-independent step count.
7. Provide one tiny reference reducer over synthetic unpack/repack/contract state.
8. Prove repeat, shuffle, tamper, missing-parent, unknown-kind, and no-side-effect fixtures.

## Gold-plating boundary

Do not re-run live tools/providers, persist raw payloads, promise bitwise hardware replay, add a workflow engine, or migrate existing domain replay commands. This issue supplies the pure contract, engine, receipt, and one synthetic reducer.

## Dedupe

- #6053 owns replay of global coordination decisions and action witnesses; this is its generic pure reducer dependency.
- #6528 owns replay/explain of derived context views.
- #8609 (closed) owns compute-event capture/summary for kernel replay.
- #11028 (closed) owns agent-trajectory regression evaluation.
- Existing `fak audit replay`, `sessionreplay`, and KV replay remain domain-specific consumers.

Searches for `deterministic replay`, `reconstruction`, and `first divergence` across open/closed public issues on 2026-09-08 found no shared trace-state fold contract.

## Prior-art ledger

- **Source:** Temporal Go SDK `test/replaytests/replay_test.go:121-163,182-194,241-250,380-383@3686024a610787c7b36edacc109f9b969fe44f2a`.
- **Source event/state:** commit `3686024a610787c7b36edacc109f9b969fe44f2a`, 2026-09-08T16:00:58Z; shipped tests on `main`.
- **Observed at:** 2026-09-08T16:10:19Z. **License:** MIT.
- **Disposition:** **ADAPT / DEFAULT** — reuse history replay as a compatibility/nondeterminism witness; adapt to typed fak events, explicit state digests, bounded pure reducers, and incomplete-evidence states. No code copied.
- **Refresh trigger:** Temporal replay-contract change or a fak domain requiring nondeterministic external inputs.

## Done condition

The trajectory package can deterministically reconstruct canonical state, refuse insufficient histories, and pinpoint the first digest divergence without executing effects.

## Definition of done

- [ ] Manifest pins reducer, schema, config, and source digests.
- [ ] Replay order is deterministic under equivalent input permutation.
- [ ] Cycles, duplicates, missing parents, and unknown kinds have typed outcomes.
- [ ] Before/after state digests are checked at every available transition.
- [ ] First-divergence receipt names event/span and expected/actual digests.
- [ ] Event/input/state/checkpoint bounds are enforced and receipted.
- [ ] Replay API exposes no tool/provider side-effect capability.
- [ ] Identical histories produce byte-identical checkpoints and result.

## Verifiable Witness

```bash
go test ./pkg/traceabi ./internal/trajectory -run 'TestExternalReplayConsumer|TestReplayReconstruct|TestReplayFirstDivergence|TestReplayBounds' -count=1
```

## Witness

Run the deterministic `go test ./pkg/traceabi ./internal/trajectory -run 'TestExternalReplayConsumer|TestReplayReconstruct|TestReplayFirstDivergence|TestReplayBounds' -count=1` gate above.

## Acceptance gate

Two permutations of the same causal history yield identical checkpoint/final digests; changing one transform output yields `diverged` at that exact event; deleting one parent yields `incomplete`, never a reconstructed success.

## Likely files

- `pkg/traceabi/replay.go`
- `pkg/traceabi/replay_external_test.go`
- `internal/trajectory/replay.go`
- `internal/trajectory/replay_test.go`
- `internal/trajectory/testdata/replay-transform-chain.json`

## Lane

trajectory

Schedule after TICKET-01 through TICKET-04; the public contracts are hard prerequisites and TICKET-04 shares `pkg/traceabi/**`.

## Expected steps

8

## Closure binding

The resolving commit cites this issue and uses `(fak trajectory)`. Report `pkg/traceabi@rev`, `internal/trajectory@rev`, fixture/final digests, and the first-divergence witness.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12262
