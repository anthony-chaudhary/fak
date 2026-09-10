<!-- fak-model-key: qwen38-vulkan-mtp-raw-target-history-v1 -->
# Connect raw Vulkan target history to native Qwen MTP

GitHub tracker: https://github.com/anthony-chaudhary/fak/issues/12720

```routing
lane: model
paths: ["internal/model/hal.go", "internal/model/qwen35_hal.go", "internal/model/qwen35_verify_panel_device.go", "internal/model/qwen35_mtp_draft.go", "internal/model/qwen35_mtp_target_vulkan.go", "internal/model/qwen35_mtp_target_test.go", "internal/model/qwen35_mtp_target_vulkan_test.go", "internal/model/qwen35_sequence_prefill_test.go", "internal/compute/qwen35_sequence_contract.go", "internal/compute/qwen35_raw_hidden_contract.go", "internal/compute/vulkan_qwen35_sequence.go", "tools/concept_disambiguation_scorecard.data/rows-support-maturity-halo-mtp-target.json", "docs/fak/concept-glossary.md"]
expected_steps: 8
```

## Current state

At public commit `644bd060041a89caa7e1024b2cf1540b058cce16`, the resident
Vulkan MTP draft from #12538 and recurrent partial commit from #12714 are
implemented. The production target cannot use the draft: the depth-N target
validator rejects backend/quant sessions, and the device verification path
rejects target-hidden capture. In addition, `prefillHAL` records normalized
`LastHidden` as raw target history, retaining only the last prompt row.
MTP requires each committed token's residual before final normalization.

Centrality: Core. Portfolio tier: 2 (serving). Priority: P0.

- P1 Context: advanced; joins the real target and retained draft.
- P2 Net value: advanced; removes the explicit production admission blocker.
- P3 Adaptation: preserved; exact target verification controls acceptance.
- P4 Operations: advanced; typed capability refusal and witnessed state lineage.

Source repro at that revision:

```text
rg -n 'only the native f32 target|target-hidden capture|result.LastHidden|last := norm' internal/model/qwen35_mtp_draft.go internal/model/qwen35_verify_panel_device.go internal/model/hal.go internal/compute/vulkan_qwen35_sequence.go
```

The output includes the F32-only refusal at `qwen35_mtp_draft.go:703`,
the hidden-capture refusal at `qwen35_verify_panel_device.go:60-62`,
`rememberTargetHidden(...Read(result.LastHidden))` at `hal.go:985`, and
`last := norm(lastRaw, req.OutputNorm, 1)` at
`vulkan_qwen35_sequence.go:623`. These are source witnesses, not hardware
performance measurements.

## Core through-line

Explicit raw-history request -> Vulkan pre-final-norm residual rows ->
validated positional target history -> retained Vulkan MTP draft -> existing
bounded target verification and accepted-prefix transaction -> durable receipt.
Keep the existing normalized `LastHidden` contract unchanged. Capture is
optional and must not add readbacks to ordinary decode. Admit only the
supported Vulkan capability and tensor layout; preserve explicit CPU callers.

## Gold-plating boundary

No new shaders, external inference engine, sampling changes, automatic
performance promotion, hardware throughput estimate, or global default flag.
Request-scoped agent/gateway MTP integration is a subsequent seam. This leaf
does not claim faster whole-model decode from component correctness alone.

## Done condition

- [x] [SW-VERIFIED] Sequence capture returns every raw residual row and preserves
      normalized `LastHidden`; capture disabled adds no raw-history retention.
- [x] [SW-VERIFIED] Scalar steps, prompt prefill, and verification populate
      raw history with exact input-token lineage.
- [x] [SW-VERIFIED] Full, partial, and zero acceptance preserve the matching
      hidden prefix and continued-decode correctness.
- [x] [SW-VERIFIED] The model depth-N entry selects the resident Vulkan draft
      for eligible Vulkan targets; unsupported capabilities refuse explicitly.
- [x] [SW-VERIFIED] Independent behavioral tests and affected checks pass;
      unrelated baseline failures are reported separately.
- [x] [HW-WITNESSED] Required-device Vulkan tests execute without skipping,
      comparing raw rows and continuation with the native reference and
      observing the resident draft. No throughput claim follows from this check.

## Witness

```text
go test ./internal/model ./internal/compute -run 'TestQwen35MTPVulkanTarget|TestQwen35VulkanMTPRawHidden' -count=1 -v
go test -tags vulkan ./internal/model -run 'TestQwen35VulkanMTPRawHidden' -count=1 -v
```

For the physical command set `CGO_ENABLED=1`, require the actual Vulkan device,
and retain build revision, device identity, invocation and non-skipped results.
The CPU command alone cannot establish physical execution.

Validation on 2026-09-10: the focused model MTP/sequence/device/partial-commit
regression suite and compute Qwen3.5/Qwen3.8 contract tests pass. The required
Vulkan binary ran all six selected top-level tests without skips on an AMD
Radeon RX 7600. It witnessed raw prompt rows, three scalar continuations, the
resident draft, and accepted device verification with exact greedy output
comparison. Full/partial/zero rollback is independently exercised through an
injected transaction backend; these three modes are software witnesses.

The broad compute short suite reports a separate unchanged timing-sensitive
`TestDoorbellSub100usExchange` failure. This is a simulated scheduling assertion,
not hardware bandwidth evidence. This leaf claims neither a whole-suite pass
nor real-model Halo performance. Request-scoped planner/gateway integration is
tracked separately in #12721.

## Verifiable Witness

The independent oracle distinguishes pre-norm residuals from normalized output,
checks multiple prompt positions and a subsequent decode position, exercises
accepted-prefix transactions, and verifies actual Vulkan draft construction.
Retain failures and missing evidence. Do not inject historical benchmark values.

## Likely files

The routing block is the exhaustive source/test write surface. Key seams are
`hal.go:984` (prefill result), `qwen35_hal.go:934` (request admission),
`qwen35_verify_panel_device.go:60` (capture rejection),
`qwen35_mtp_draft.go:676` (target admission),
`qwen35_sequence_contract.go:197` (result contract), and
`vulkan_qwen35_sequence.go:618` (raw-to-normalized boundary).

## Lane

Public model with a disjoint compute contract leaf. The affected packages are
`internal/model` and `internal/compute`. Private serving, installer, other
backends, and unrelated model paths remain independent.

## Quarantined fallback mechanism

The new capability is explicit and opt-in. Unsupported targets keep the current
typed refusal and ordinary native decode path. No broad quarantine or external
engine fallback is introduced.

## Dependencies and execution

Builds on #12538, #12714 and #12687; contributes to #12184 and #11572.
Eight steps: claim paths; expose raw rows; record prompt/step history; preserve
verification history; bind resident draft; author independent tests; validate;
land source and receipt. Implementation and tests use separate contexts.

## Definition of done

The scoped raw-history and resident-target admission behavior is committed,
the independent behavioral and required-device correctness witnesses pass,
and the durable receipt records their exact source and execution scope.
Whole-model performance and normal request-path promotion remain separate gates.

## Working spine

Native model entry -> Vulkan target residual capture -> retained Vulkan draft
-> existing device verification -> committed hidden lineage and receipt.

## Parent context

Parent #12184; prerequisites #12538 and #12714 already have landed mechanisms.

## Why this is next

The current backend and capture refusals prevent the newly landed resident
draft from using the native Vulkan target. Fixing this observed handoff makes
the existing implementations composable before more kernel tuning.

## Acceptance gate

The named focused tests run and pass, with required-device execution for the
physical correctness criterion and clean boundary and publication checks.

## Closure binding

The resolving signed public commit references this issue and local ticket,
carries `(fak model)`, and binds the independent test receipt to its source.

## Done condition / witness

The named behavioral and required-device witness commands exit zero with actual
matching test execution, validating the acceptance checklist above. Attach the
source-bound receipt before closing this leaf.
