# fix(agent): bound native sequence prefill cancellation at completed chunks

GitHub: https://github.com/anthony-chaudhary/fak/issues/12691

<!-- fak-agent-key: native-sequence-prefill-cooperative-cancellation -->

```routing
lane: agent-runtime
paths: ["internal/agent/inkernel_decode.go", "internal/agent/inkernel_prefill_cancel_test.go"]
expected_steps: 4
```

## Working spine

Canceled request context -> qualified native sequence prefill -> completed bounded chunk -> context checkpoint -> stop before the next chunk -> ordinary session cleanup.

## Current state

At public commit `3fc587c07`, `InKernelPlanner.prefillDivergentSuffix` checks the context only between non-final chunks. Its short/monolithic branch and final logits-producing `Prefill` return without a context check. More significantly, `generateReusedContextWithBias` sends the cached-prefix checkpoint directly to `s.Prefill`, bypassing the helper and its chunk boundaries entirely. A canceled request can therefore continue until a whole native prefill call returns.

The chunk route is restricted to the CPU Q4_K hybrid condition. Device backends that implement the complete native sequence-prefill contract do not receive bounded cooperative checkpoints even when they expose the exact sequence-prefill identity and embedding-row marker needed by the same path.

## Why this is next

Request cancellation needs a bounded place to take effect during long native prompt ingestion. The existing chunk API supplies that boundary without changing backend APIs or attempting unsafe device-side interruption.

## Parent context

Standalone public runtime correctness leaf; no parent issue.

## Core through-line

Route each qualified Qwen hybrid Q4_K native sequence backend through the existing bounded chunk helper. Qualification requires `model.Qwen35SequencePrefillBackend`, the exact `compute.Qwen35SequencePrefillPath` identity, and `compute.Qwen35SequenceEmbeddingRowsBackend` with the exact rows-path marker. This capability gate may admit Vulkan or CUDA implementations that satisfy the same contracts.

Check `ctx.Err()` before prefill, after every completed `PrefillNoLogits` chunk, and after the final logits-producing `Prefill`. Route the cached-prefix checkpoint through the same helper before snapshot publication. Cancellation remains cooperative: an active GPU submission retains its resources and reaches its normal synchronization/return boundary before cancellation is reported.

## Gold-plating boundary

Do not add device-side preemption, command-buffer destruction, exported APIs, scheduler changes, service routes, or per-layer cancellation. Preserve monolithic behavior for backends that do not satisfy the complete qualified sequence-prefill contract.

## Verifiable Witness

Required deterministic helper witness:

```bash
go test ./internal/agent -run 'TestInKernelPrefillCancellation|TestInKernelPrefillChunkRouting' -count=1
```

The pre-fix source witness is:

```bash
git show 3fc587c07:internal/agent/inkernel_decode.go
```

At issue creation, the package build was independently blocked by production Metal ICB types being defined only in a test file. That dependency was repaired in `7a62f603d`. Independent testing then passed the new cancellation/routing tests and existing bounded-prefill, configured-partition, cancellation, and unrelated-path tests without an overlay: `ok internal/agent 0.043s`, exit 0. A separate source review confirmed that the cached-prefix checkpoint calls the helper and returns on cancellation before `PrefixSnapshot` and admission. Live full-checkpoint Vulkan/CUDA cancellation latency and parity are not claimed by this software leaf.

## Witness

Run `go test ./internal/agent -run 'TestInKernelPrefillCancellation|TestInKernelPrefillChunkRouting' -count=1`, then review `generateReusedContextWithBias` to verify that checkpoint prefill reaches `prefillDivergentSuffix` before snapshot publication.

## Definition of done

Completion requires every item in Scoped Acceptance Criteria below and the focused witness.

## Scoped Acceptance Criteria

- [x] Qualified native sequence prefill uses bounded chunks under the existing append contract; full-checkpoint hardware qualification remains separate.
- [x] Context is checked before work, after each completed non-final chunk, and after the final call.
- [x] Cancellation after a completed chunk prevents submission of the next chunk.
- [x] Cached-prefix checkpoint prefill uses the helper before snapshot publication (source review).
- [x] Generic and partially capable backends retain their historical monolithic behavior.
- [x] Active device work keeps resource ownership until its prefill call returns; no immediate GPU-interrupt claim is made.
- [x] The focused fake-session tests pass without hardware.

## Acceptance gate

The focused test command passes, source review proves the cached-prefix call-site route, and the implementation changes no exported API.

## Closure binding

The resolving commit cites this issue and carries the `agent-runtime` lane.

## Lane

agent-runtime

## Likely files

- `internal/agent/inkernel_decode.go:225-244` (`generateReusedContextWithBias`) — cached-prefix checkpoint and remaining suffix.
- `internal/agent/inkernel_decode.go:358-385` (`InKernelPlanner.prefillDivergentSuffix`) — context checkpoints and chunk dispatch.
- `internal/agent/inkernel_prefill_cancel_test.go` — deterministic helper cancellation and capability-routing witness.

## Expected steps

4

## Blast radius and affected lanes

Only `internal/agent` changes. The fake session makes the witness independent of GPU availability. Other runtime and backend packages remain unaffected.

## Quarantined fallback mechanism

Backends missing any required capability marker keep the existing monolithic prefill path and existing request-level context checks.

- Centrality: Core
- P1 Context: advanced — bounded cancellation makes long prompt ingestion operator-controllable.
- P2 Net value: preserved — no exported API or token-semantic change.
- P3 Adaptation: advanced — capability markers select the qualified route.
- P4 Operations: advanced — canceled work stops at a completed chunk boundary.

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/8 points for native request lifecycle control.

## Completion standard

demo
