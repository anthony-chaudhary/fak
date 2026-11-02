# Structured output in native linear speculation

The ordinary `InKernelPlanner.Complete` and `CompleteStream` routes compose an
already-configured generic speculative engine with request-local structured
constraints for greedy CPU decoding. No new drafter is automatically enabled.
Supported masks include JSON schema, JSON object, regex and guided choices.

Each verification round forks the current grammar history, reasoning phase and
notification state. Compiled masks may share synchronized admission caches;
request histories and scratch rows never alias. Every target choice, including
the boundary and correction, is masked **before** comparing it to a proposal.
Only the matching draft prefix advances the temporary matcher. Rejected suffixes
are removed from target KV; the parent grammar advances only for output tokens
actually emitted. An output-budget or callback stop retains exactly that prefix,
including one emitted correction. EOS is neither emitted nor added to history.
Empty or failed proposals continue with masked target sampling. Tree proposals
disable drafting for the request remainder rather than using linear semantics.

## Supported envelope

The first envelope is ordinary host KV, model types `llama`, `qwen2`, `qwen3`,
and the generic empty-type configuration. Unknown architectures, hybrid/recurrent
layer layouts, MLA/compressed-state configurations, device execution, Metal,
Q4_K-specialized execution and resident MTP remain target-only. Stochastic,
seeded, top-logprob, vision and thinking-budget/reasoning requests also remain target-only.
Existing tools-plus-structured and DFlash2 refusals are unchanged. The verifier
checks suffix-eviction support before any provisional forward.

This is software correctness coverage, not hardware qualification or a throughput
claim. `VerifyForward` can use serial target steps (including quantized CPU
sessions); a proposed block is not necessarily one target operation.

## Provenance and verification

Adapted from DeepInfer at `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`:
`deepinfer/structure_output/matcher.py` (accept/rollback state) and
`deepinfer/model_executor/layers/sampler.py` (per-position masking before
verification). See [the attribution](../third_party/deepinfer-grammar-transactions/NOTICE).
DeepInfer's `copy_without_status` starts at the initial position; it is not the
current-position fork used here. No XGrammar implementation was copied.

Bounded CPU regression command:

```text
go test ./internal/agent -run '^TestNativeStructuredSpeculative' -count=1 -timeout=3m
```

The fixtures compare actual Complete/stream results to constrained serial output,
assert proposal invocation, inspect the serving session's retained KV and
continuation after early emission stops, and check fork/request isolation and
masked correction. Synthetic CPU evidence does not establish model quality or
performance on a physical accelerator.
