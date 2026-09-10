<!-- fak-agent-key: evaluated-generation-continuation-cache -->
<!-- fak-public-issue: anthony-chaudhary/fak#12716 -->
# perf(agent): reuse evaluated generation state on continuation turns

```routing
lane: agent
paths: ["internal/agent/inkernel_decode.go", "internal/agent/inkernel_continuation_cache_test.go", "internal/agent/inkernel_reuse_test.go"]
expected_steps: 6
```

## Parent context

Parent: #11572, native local-agent performance campaign.

## Why this is next

The ordinary agent loop discards useful generation state despite already having
complete prefix snapshots; kernel-only copy reductions do not close this gap.

## Working spine

Generated token -> successful native forward -> scoped checkpoint -> next-turn
cache hit -> fewer evaluated prompt tokens with unchanged outputs.

The ordinary call path is `Complete` -> render and encode -> native generation ->
append assistant message and tool results -> `Complete`. Reuse still requires an
exact token prefix. Retokenization, whitespace/reasoning normalization, canonical
tool-call rendering, prompt shrinking, or changed tools can shorten that prefix;
this change does not claim a hit for every reconstructed conversation.

## Current state

At public source `80089bc92b5e81666eb2052a68386c7b518eeedf`,
`internal/agent/inkernel_decode.go:270-300` snapshots only the incoming prompt.
`generateReusedContextWithBias` then decodes and closes the session at lines
302-355 without retaining the evaluated assistant continuation. A following
agent/tool turn with an identical token prefix must prefill that generated
content again. Existing Vulkan KV copy-on-write and recurrent checkpoint work
reduce snapshot costs but do not admit this missing cache entry.

## Problem Frame

- Centrality: Core native agent context reuse through the ordinary serving path.
- P1 Context: advanced; emitted tokens and evaluated model-state tokens differ.
- P2 Net value: advanced; avoid repeated prefill of already evaluated output.
- P3 Adaptation: advanced; keep CPU and complete device snapshots, logits, and
  scoped cache identity aligned across serial and batched decoding.
- P4 Operations: advanced; preserve cancellation, ownership, cache budgets, and
  exact output semantics without introducing another model forward.

## Core through-line

Prompt -> native decode -> successful evaluated-token boundary -> one bounded
continuation snapshot with matching logits -> ordinary scoped prefix lookup ->
suffix-only prefill on the following turn -> deterministic work/parity witness.

## Gold-plating boundary

- No tokenization rewrite, speculative decoding change, or stop-policy change.
- No GPU kernel, cache precision, allocator, paging, or scheduler redesign.
- No additional model step to consume the final emitted token just for caching.
- No hardware speedup or comparative performance claim from fixture tests.

## Done condition

- [x] Successful eligible generation admits at most one continuation beyond
  the existing prompt, covering only tokens actually forwarded by the model.
- [x] Max-token and emit-stop completion exclude the unforwarded final token;
  a later sampled stop token preserves earlier evaluated output.
- [x] Cached logits correspond exactly to the snapshot boundary, and following
  matching turns reuse those tokens while preserving generated outputs.
- [x] Serial and batched paths retain the same evaluated-token contract.
- [x] Cancellation and failed generation do not admit continuation state;
  optional admission failure does not change a completed response.
- [x] Scoped admission preserves tenant/agent isolation and existing budgets.
- [x] No additional inference step is performed to build a cache entry.

## Witness

The command below must exit zero, restore the evaluated continuation boundary,
and preserve outputs relative to the same cache-disabled fixture.

### Verifiable Witness

```text
go test ./internal/agent -run TestInKernelContinuationCache -count=1
go test -race ./internal/agent -run TestInKernelContinuationCache -count=1
```

The independent regression first establishes the missing continuation entry on
the base revision, then checks next-turn matched/evaluated tokens and output
parity on the candidate. Tests are software evidence, not Halo latency evidence.
Any physical performance receipt must additionally bind source, model, backend,
driver, power/thermal state, sample count, P50/P90 and confidence intervals, plus
the snapshot/admission overhead paid by the first turn.

## Likely files

- `internal/agent/inkernel_decode.go`
- `internal/agent/inkernel_continuation_cache_test.go` (independent regression)
- `internal/agent/inkernel_reuse_test.go` (quarantine also evicts the generated descendants)

### File:Line seams

`internal/agent/inkernel_decode.go:270-355` owns admission and generation lifetime;
`internal/agent/inkernel_decode.go:629-780` owns successful forward boundaries.

## Blast Radius

Primary package: `internal/agent`. Reuses the existing `model.PrefixSnapshot`
and `radixkv` contracts without changing either API. Compute kernels, model
numerics, private serving policy, installer, and peer lanes are unaffected.

## Quarantined Fallback

The pre-decode prompt entry remains available. Ineligible, canceled, failed,
or unadvanced requests retain existing behavior. An optional continuation
snapshot failure releases its ownership and leaves the completed response valid.

## Lane

`agent`, six bounded steps, one production package.

## Definition of done

All software acceptance checkboxes above are witnessed by independent tests,
the relevant agent regression suite passes, and the verified change is landed.

## Acceptance gate

The focused regression and race run exit zero, with no extra inference step and
no output divergence or cross-tenant hit. Physical latency remains unclaimed
unless separately measured with complete snapshot provenance.

## Closure binding

The resolving commit cites the tracking issue and uses the `(fak agent)` trailer.

## Verification on 2026-09-10

The focused continuation regressions and the full `internal/agent` package pass.
The focused continuation suite also passes with the race detector. Independent
tests cover serial/batched boundaries, current logits, complete simulated hybrid
HAL restoration, cancellation, a coalesced failure after forward, tenant isolation,
and quarantine of generated descendants while preserving a benign sibling.

The original implementation fails the independent negative control by retaining
only the incoming prompt at the queried continuation boundary. The two-turn
software fixture separately accounts for existing prompt reuse and newly reused
generated tokens. No physical Halo speedup is claimed; the latency of ordinary
retokenized tool turns still requires a separate hardware measurement.
