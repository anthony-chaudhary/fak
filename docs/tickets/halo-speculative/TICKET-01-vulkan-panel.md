<!-- fak-model-key: qwen38-vulkan-speculative-target-panel-v1 -->
<!-- fak-public-issue: anthony-chaudhary/fak#12687 -->

# perf(model): verify K<=4 Qwen3.8 drafts in one resident Vulkan panel

Parent: #12184. Coordinate with #12538, which separately owns resident Vulkan
execution of the retained MTP draft head.

```routing
lane: model
paths: ["internal/compute/qwen35_sequence_contract.go", "internal/compute/vulkan_qwen35_sequence.go", "internal/compute/vulkan_qwen35_sequence_test.go", "internal/model/verify.go", "internal/model/speculative.go", "internal/model/qwen35_hal.go", "internal/model/qwen35_vulkan_speculative_test.go"]
expected_steps: 6
```

## Current state

At public revision `6776c2bbfd53d7cc9e7f03c9b44b339431c2ed8a`, the
generic speculative serving loop already accepts `ProposalGenerator` implementations,
including the prompt n-gram generator, and calls `ParallelVerifyKernel` from
`internal/agent/inkernel_planner.go:797-970`.

The target-verification seam does not batch the deployed Qwen hybrid Vulkan shape.
`internal/model/speculative.go:451-465` selects `Session.VerifyForward` only when
`verifyForwardBatchedOK` admits the target. `internal/model/verify.go:351-362`
rejects every device backend outside the Metal-specific Qwen envelope, so Vulkan
executes one `Session.Step` for each proposed token.

The existing Qwen sequence operation is the closest real device primitive:
`internal/model/qwen35_hal.go:925-1007` builds the complete model request and
`internal/compute/vulkan_qwen35_sequence.go:460-629` executes a token panel
layer-major on Vulkan. It currently projects only the final token when
`NeedLogits` is set, so speculative verification cannot obtain one target-logit
row per proposed token from that operation.

The older diagnostic in `internal/model/qwen35_hal.go:1022-1161` must not be used
as evidence: it calls `Session.Step` four times at lines 1112-1120, reports
`SinglePass: true`, and derives `ThroughputTokS` from a fixed baseline at lines
1138-1145. This leaf removes or repairs that false accounting.

An initial physical full-checkpoint A/B reached the device-verification path but
did not pass the exact token-parity gate on the novel-control workload. After the
causal Vulkan attention prerequisite was corrected, a one-sample full-generation
diagnostic passed exact token parity on both workloads. That N=1 diagnostic does
not qualify performance; the N>=5 hardware criterion and any performance
conclusion remain open.

Centrality: Core

Portfolio tier: 2 - fak serving only. Priority: P0.

- P1 Context: advanced - supplies the missing target-side operation needed by any
  useful speculative proposer on the supported Qwen3.8 Vulkan route.
- P2 Net Value: advanced - verifies up to four proposed tokens in one batched
  target invocation instead of repeating full target steps.
- P3 Adaptation: preserved - the target model remains the exactness boundary and
  unsupported shapes continue through explicit serial target decode.
- P4 Operations: advanced - reports observed operation, transfer, accepted-token,
  rollback, and timing counters without modeled throughput fields.

## Parent context

Public issue #12184 owns the broad native Qwen3.8 K=4 speculative-verification
objective. This ticket is its bounded Vulkan target-panel leaf. It is immediately
usable with prompt n-gram proposals while #12538 remains open for MTP-only drafting.

## Why this is next

The generic proposer, exact acceptance loop, and Vulkan Qwen sequence operation
already exist. Connecting those shipped seams is the smallest vertical change
that replaces repeated target steps with one batched target invocation and makes
current prompt n-gram proposals useful on the Halo serving path.

## Working spine

Prompt n-gram proposal -> generic speculative engine -> one resident Vulkan
Qwen target panel -> per-position logits -> exact accept/correct/rollback ->
truthful operation and transfer receipt.

## Core through-line

Linear proposal of one to four token IDs -> one Qwen3.8 sequence operation on the
resident Vulkan target -> one target-logit row per proposed position -> existing
exact greedy acceptance and KV/recurrent transaction -> commit the accepted prefix
and correction token or restore the pre-verification state -> measured receipt.

Extend the Qwen sequence contract so the Vulkan implementation can return all
verification logits for a bounded K<=4 panel. Admit only the exact Qwen hybrid
Vulkan envelope in `VerifyForwardOneOperation` / `ParallelVerifyKernel`. Reuse the
existing speculative transaction semantics; do not add a second acceptance or
rollback implementation.

## Implementation scope

- Return all K target-logit rows from one bounded Qwen hybrid Vulkan sequence call.
- Admit that operation through the existing model verifier for linear K<=4 proposals.
- Exercise the entry with the existing prompt n-gram generator and exact transaction.
- Replace the older serial-as-single-pass diagnostic and its modeled throughput.
- Record only observed operation, transfer, timing, acceptance, and rollback counters.

## Gold-plating boundary

Do not implement the retained MTP draft head (#12538), a new proposer, branching
tree verification, K>4, adaptive depth, multi-request batching, a new public HTTP
schema, default-on serving, or a new shader family. Do not change ordinary prompt
prefill or serial `Session.Step` behavior. Do not report estimated throughput or
claim a Halo gain from software tests.

## Done condition

For a linear K=1..4 prompt n-gram proposal whose first token matches the known
target boundary, the generic speculative engine executes one actual Qwen3.8
Vulkan sequence invocation, obtains every target-logit row, and commits or rolls
back the existing transaction with token-identical output and truthful measured
counters. A known-boundary first-token rejection executes zero target operations
or decode steps, returns the exact correction, and leaves state unchanged. The
capability helper returns a typed downgrade for unsupported envelopes. The
compatibility transaction may instead complete the existing serial verifier and
must then report `OneOperation=false`, the ordinary target path, and a reason.
Malformed device results are fatal after committed state is restored and never
claim device execution.

## Verifiable Witness

The source repro and regression oracle in the Witness section below bind the
change to one executed device operation, exact results, and state restoration.

## Witness

Source repro:

```bash
rg -n "verifyForwardBatchedOK|VerifyForward\(|NeedLogits|SinglePass|ThroughputTokS" internal/model/speculative.go internal/model/verify.go internal/model/qwen35_hal.go internal/compute/qwen35_sequence_contract.go internal/compute/vulkan_qwen35_sequence.go
```

Regression oracle:

```bash
go test ./internal/compute -run "TestQwen35SequenceEmbeddingContract" -count=1
go test ./internal/model -run "Test(Qwen35DeviceVerification|VerifyGreedyDeviceDraft|Qwen35DeviceTargetTransaction|SpecDecodeGreedyDevice)" -count=1
go test ./internal/agent -run "Test(GreedySpeculative|SpeculativeDevice|VerifyGreedySpeculativeRound|NGramSpeculative)" -count=1
go test ./internal/gateway -run "^TestNGramSpeculative" -count=1
go test -tags vulkan ./internal/model -run "Test(VulkanQwen35Device|Qwen35DeviceSpeculativeRealCheckpointAB)" -count=1
```

The Vulkan test must execute rather than skip when a declared Vulkan ICD is
available. A CPU oracle supplies the serial reference; exact greedy token IDs,
accepted length, correction token, and final KV/recurrent state must match.

## Scoped Acceptance Criteria

- [ ] [SW-VERIFIED] K=1, K=2, K=3, and K=4 linear proposals whose first token
  matches the known target boundary each execute exactly one Qwen sequence target
  operation and zero `Session.Step` verification calls.
- [ ] [SW-VERIFIED] A known-boundary first-token rejection executes zero target
  operations and decode steps, reports the explicit boundary-reject path with
  `OneOperation=false` and zero rollback, returns the boundary argmax correction,
  and leaves target state unchanged.
- [ ] [SW-VERIFIED] The operation returns K complete finite target-logit rows;
  their greedy token IDs match serial target verification exactly.
- [ ] [SW-VERIFIED] Full acceptance, rejection at every position, and zero
  acceptance preserve the existing correction-token, KV, recurrent-state, and
  sampler-count transaction semantics.
- [ ] [SW-VERIFIED] A real `NGramProposalGenerator` reaches the new verifier entry;
  no retained MTP tensors or draft-head implementation are required.
- [ ] [SW-VERIFIED] Receipts count observed target operations, serial steps,
  H2D/D2H bytes, accepted and rejected tokens, rollback positions, and measured
  elapsed time. `OneOperation` is true only for the actual single device call.
- [ ] [SW-VERIFIED] The serial-as-single-pass Qwen diagnostic is removed or made
  truthful; no fixed tok/s baseline, clamp, or estimated throughput remains.
- [ ] [SW-VERIFIED] Unsupported backend, quantization, shape, or tree proposal
  returns a typed capability downgrade without mutating committed state; the
  compatibility transaction may explicitly run the serial verifier only with
  `OneOperation=false`, the ordinary target path, and a reason.
- [ ] [SW-VERIFIED] A malformed device result returns a fatal error after restoring
  committed state and never claims device execution.
- [ ] [HW-WITNESSED] A source-bound target-device A/B with one warm-up and N>=5
  matched repetitions per arm records exact token parity, `fallback_count=0`, all
  samples, and lower median target-verification time than serial verification.

## Blast radius and affected lanes

The target-verification mechanism remains routed to `internal/compute` and
`internal/model`. The completed vertical integration also changes
`internal/agent` to preserve device prefix snapshots, request semantics, and
truthful speculative statistics, plus `internal/gateway` to admit explicit
`FAK_SPECULATIVE=ngram` selection only on the supported Qwen Vulkan envelope.
Metal, CUDA, CPU, ordinary prefill, external providers, and HTTP schemas remain
unaffected.

## Quarantined fallback mechanism

The new verifier remains opt-in through the existing speculative engine. The
capability helper returns a typed downgrade for unsupported envelopes before
committed state changes. The compatibility transaction may explicitly run the
serial target path with `OneOperation=false`, the ordinary target path, and a
reason. A malformed device result is fatal after committed state is restored and
never claims device execution. The old diagnostic may remain callable only if its
receipt truthfully says it used serial steps.

## Likely files

- `internal/compute/qwen35_sequence_contract.go:132-166`
- `internal/compute/vulkan_qwen35_sequence.go:460-629`
- `internal/model/verify.go:155-362`
- `internal/model/speculative.go:407-510`
- `internal/model/qwen35_hal.go:1022-1161`
- `internal/model/qwen35_sequence_embedding_rows.go`
- `internal/model/qwen35_verify_panel_device.go`
- `internal/model/speculative_device.go`
- `internal/model/qwen35_verify_panel_device_test.go`
- `internal/model/qwen35_verify_panel_device_vulkan_test.go`
- `internal/model/qwen35_speculative_device_ab_vulkan_test.go`
- `internal/agent/inkernel_planner.go`
- `internal/agent/inkernel_speculative_device_test.go`
- `internal/gateway/chat_completions.go`
- `internal/gateway/gateway.go`
- `internal/gateway/speculative_ngram_test.go`

## Lane

model

## Expected steps

8: extend the sequence result contract; gather bounded embedding rows; return
all-position logits; bind the exact transaction and boundary fast reject; preserve
planner prefix/cancellation/measurement semantics; admit the explicit gateway
n-gram route; replace false diagnostic accounting; run software and physical
witnesses.

## Acceptance gate

Both regression commands pass, the Vulkan-selected test records one target
operation and zero serial verification steps, and the committed-state oracle is
identical for every rejection cut point. Hardware promotion remains pending until
the source-bound A/B criterion is satisfied.

## Closure binding

The resolving public commit references this issue and `Ref #12184`, contains the
routed target mechanism plus its bounded agent/gateway tracer integration, and
keeps #12538 open for resident MTP draft-head work.

## Required scale stages

1. Deterministic CPU and injected-backend exactness tests for every rejection cut point.
2. Local Vulkan execution proving one operation, K rows, and zero serial verification steps.
3. Physical target-device N>=5 matched serial/panel repetitions with source-bound receipts.

## Work estimate

Estimate: 5 points.

## Overall completion contribution

Contribution: 5/8 points toward #12184's target-verification objective. Resident
MTP drafting and serving promotion remain separate work.

## Completion standard

development
