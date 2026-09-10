<!-- public-issue: https://github.com/anthony-chaudhary/fak/issues/12714 -->

<!-- fak-model-key: qwen38-vulkan-partial-recurrent-replay-v1 -->

# perf(model): commit partial Vulkan drafts without full target replay

Parent: #12687.

## Parent context

The parent added the resident Vulkan target panel. This bounded leaf completes
its partial-acceptance path without changing the proposer or serving APIs.

```routing
lane: model
paths: ["internal/compute/qwen35_sequence_contract.go", "internal/compute/vulkan_qwen35_partial_commit.go", "internal/compute/vulkan_qwen35_sequence.go", "internal/model/qwen35_partial_commit_test.go", "internal/model/qwen35_partial_commit_vulkan_test.go", "internal/model/qwen35_speculative_device_ab_vulkan_test.go", "internal/model/qwen35_verify_panel_device.go", "internal/model/qwen35_verify_panel_device_test.go", "internal/model/qwen35_verify_panel_device_vulkan_test.go", "internal/model/specbind.go", "internal/model/speculative_state.go", "internal/compute/vulkan_qwen35_partial_commit_test.go"]
expected_steps: 6
```

## Current state

At public commit `19d160b9fdac5203c518a713f09ceda93f5a5b68`, the Vulkan
target verifies K<=4 tokens in one sequence invocation. Full acceptance adopts
that state. Partial acceptance clones the pre-round snapshot, restores it, and
calls ordinary `Session.Step` for every accepted token.

The concrete call sites are `internal/model/specbind.go:229-263`
(`qwen35MTPTargetTransaction.Commit`) and
`internal/model/qwen35_verify_panel_device_test.go:455-498`
(`TestQwen35DeviceTargetTransactionAcceptRollbackAndFailureAtomicity`). The
existing test explicitly expects those replay steps. This establishes the
missing mechanism; it does not establish a hardware speedup.

Every linear layer already computes the complete GDN recurrence inputs
`mixed/z/beta/alpha` in `internal/compute/vulkan_qwen35_sequence.go:552-556`.
The existing native operation consumes their causal prefix sequentially and
updates only convolution/recurrent state. Its inputs can be retained until the
target chooses an accepted length, avoiding repeated attention and dense
projections for accepted tokens.

Centrality: Core

Portfolio tier: 2 - fak serving only. Priority: P1.

- P1 Context: advanced - removes full-model recomputation after partial target acceptance.
- P2 Net Value: advanced - reuses already computed device projections rather than rereading target weights for accepted tokens.
- P3 Adaptation: preserved - exact greedy acceptance and ordinary fallback remain authoritative.
- P4 Operations: advanced - distinguishes full target replay steps from recurrent-only repair in receipts.

## Core through-line

One admitted device target panel -> retain per-layer GDN projections -> choose
accepted prefix -> restore only recurrent inputs from the intact pre-round
snapshot -> run recurrence for the accepted rows -> truncate attention KV tail
metadata -> return the accepted boundary logits -> durable exact-state receipt.

## Working spine

`Session.VerifyGreedyDeviceDraft` -> target transaction -> resident sequence
checkpoint -> accepted-prefix recurrent repair -> exact committed session.

## Why this is next

The target has already calculated the accepted tokens during verification.
Retaining the recurrence inputs avoids executing their dense projections and
attention a second time while preserving the existing rollback snapshot.

## Gold-plating boundary

No new shaders, native ABI, draft model, tree proposals, multi-request batching,
default enablement, or tuned acceptance policy. Do not export normalized hidden
states as raw target-hidden history. Requests that capture target hidden states
must return a typed downgrade before live mutation. This leaf does not add raw
hidden capture to the all-device ordinary Step path. Do not change generic KV eviction or add host KV transfers.

## Definition of Done

An admitted partial device draft commits its accepted prefix with zero full
target replay steps, identical state and continuation to ordinary decoding,
and independently owned failure rollback. Unsupported replay backends retain serial execution. Hidden-capture requests
return a typed downgrade while preserving the prior live state and history. Hardware performance remains separately qualified by the
complete end-to-end receipt below.

## Done condition

- [x] [SW-VERIFIED] An optional backend capability owns retained projections independently of transient request retirement and closes exactly once.
- [x] [SW-VERIFIED] Partial cuts 1..K-1 commit the same KV, convolution, recurrent state, token lineage, metadata, and next logits as ordinary target execution, with zero full target replay steps.
- [x] [SW-VERIFIED] Full acceptance, abort, and cancellation preserve existing behavior; unsupported replay backends retain full replay; hidden-capture requests return a typed downgrade without live mutation.
- [x] [SW-VERIFIED] All shape, alias, ownership, and KV-tail validation precedes mutation; reset/replay/fence failure restores the original prefix from the untouched snapshot and releases retained buffers.
- [x] [SW-VERIFIED] Receipts count full target replay and recurrent-only repair separately; the real-checkpoint harness reads those observed counts rather than inferring replay from a partial acceptance.
- [x] [HW-WITNESSED] Required-device synthetic tests pass every partial cut and continued decoding against the ordinary target with exact token parity and source/artifact binding on RX 7600. This is separate from Halo full-checkpoint performance qualification.
- [ ] [HW-WITNESSED] A clean-source, isolated N>=5 end-to-end A/B reports P50/P90 and confidence intervals for the same useful copy edit and novel control, including all snapshot, verification, rejection, and repair costs. A speed claim remains unchecked until supported by this receipt.

## Witness

Current behavioral witness (passes while confirming the existing replay contract):

```bash
go test ./internal/model -run '^TestQwen35DeviceTargetTransactionAcceptRollbackAndFailureAtomicity$' -count=1 -v
```

Independent regression against that base:

```bash
go test ./internal/model -run '^TestQwen35DevicePartialCommitAvoidsFullTargetReplay$' -count=1 -v
```

Observed failures before implementation:

```text
cut1: full target replay steps=1, want 0
cut2: full target replay steps=2, want 0
cut3: full target replay steps=3, want 0
FAIL (0.031s)
```

The regression is at internal/model/qwen35_partial_commit_test.go. The final
fixture exercises the optional capability and retains the old replay behavior
for unsupported backends. Independent software regressions passed, including
lifetime/error cleanup, hidden-capture typed downgrade, and Read panic cleanup.

On 2026-09-10 UTC, required RX 7600 execution passed at source checkpoint
e06cd0d24f853265f1f04cb58121a20106643273 on base
19d160b9fdac5203c518a713f09ceda93f5a5b68. The model test binary SHA-256 is
cd3589e7f3cb29f7d6725cf95c31ecd1f48bc8fccf1865b7df5cf063562b626b; the compute
test binary is 4941fcc0acba2d2fac11661955d483d23297715dd6d91d778b7e0ed15a20a0c4.
The source/shader manifest SHA-256 is
e3fc9907ec5ed5b54a27e52e726efbe409e03144f98d2ab9a0079787484cdc7b.
This was a working-tree build with exact source binding, not clean-commit
Halo speed qualification.

Required-device selectors:

    go test -tags vulkan ./internal/model -run '^(TestVulkanQwen35PartialCommitMatchesOrdinaryWithoutTargetReplay|TestVulkanQwen35DeviceVerificationAllRowsAndContinuationMatchCPU|TestVulkanQwen35DeviceTransactionAcceptsOnlyCommittedPrefix|TestSpeculativeABBootstrapCI95IsDeterministicAndN1IsUnmeasured)$' -count=1 -v
    go test -tags vulkan ./internal/compute -run '^TestVulkanQwen35PrefixReplayTruncatesEveryActiveKVPlane$' -count=1 -v

Set FAK_VULKAN_REQUIRE_DEVICE=1 and FAK_VULKAN_EXPECT_DEVICE to the actual
device name, with a freshly built matching native library and shader bundle.
All partial cuts 1/2/3 passed with zero full-model replay steps. The compute
test checks exact active K/Kraw/V prefixes and unchanged spare slots; state
and float logits checks retain their documented numerical tolerances.
All selected tests ran without skips. These are N=1 synthetic correctness
invocations; no inference throughput claim is made.

The first physical attempt rejected valid spare KV slots during preflight.
The second exposed an unsupported host-KV-export assumption in the model
test. The final production validation checks only active attention slots;
the independent compute test supplies the exact KV-plane witness.

Four unrelated affected-package failures reproduce on the unmodified base:
CPU Q2_K continuation NaNs (#12433), GDN scalar/batched bit parity (#12038),
and resumed-state parity (#11987). #12702 independently records the same
set. Focused regressions are green; full-package green is not claimed.

## Verifiable Witness

The regression must observe target Step calls, complete state, and subsequent
logits; a receipt field alone is insufficient. The original prefix snapshot
must remain independently restorable until commit completes. Physical results
require the actual GPU identity, source SHA, native library and shader digests,
system state, sample count, and exact receipt path.

## Likely files

- `internal/compute/qwen35_sequence_contract.go`
- `internal/compute/vulkan_qwen35_partial_commit.go`
- `internal/compute/vulkan_qwen35_sequence.go`
- `internal/model/qwen35_partial_commit_test.go`
- `internal/model/qwen35_partial_commit_vulkan_test.go`
- `internal/model/qwen35_speculative_device_ab_vulkan_test.go`
- `internal/model/qwen35_verify_panel_device.go`
- `internal/model/qwen35_verify_panel_device_test.go`
- `internal/model/qwen35_verify_panel_device_vulkan_test.go`
- `internal/model/specbind.go`
- `internal/model/speculative_state.go`
- `internal/compute/vulkan_qwen35_partial_commit_test.go`

## Acceptance gate

The independent partial-commit regression and focused model/compute tests pass;
required-device state/continuation checks pass; boundary and public leak audits
pass. Known base package failures are recorded separately and are not waived
as green results. Hardware performance is reported only from the complete bound A/B receipt,
with unmeasured criteria left unchecked.

## Closure binding

Land by explicit paths with independent implementation/test provenance and a
reference to this issue and #12687. Close this leaf only when its acceptance
criteria are witnessed; parent #12687 remains open for any unmet end-to-end
performance qualification.

## Lane

Public core runtime, `internal/compute` and `internal/model`. Agent/gateway APIs
and private factory packages are unaffected. Unsupported backends use the
existing correct serial replay, so this optional extension leaves trunk usable
without hardware qualification.
