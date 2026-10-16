<!-- fak-model-key: v41-indexer-resident-keys-unwired -->
```routing
lane: model
paths: ["internal/model/v41_gather_ops_device.go", "internal/model/v41_indexer_score_device.go", "internal/model/v41_layer_step.go", "internal/model/v41_indexer_projection.go", "internal/model/v41_indexer_score_device_test.go"]
expected_steps: 6
```

# perf(model): wire the resident V4.1 indexer-key path so decode stops repacking O(context) keys every token

GitHub: [#13870](https://github.com/anthony-chaudhary/fak/issues/13870)

Process cause: verification-gap

## Value

- **For:** fak-native DeepSeek V4.1 Flash decode on the Halo Vulkan path.
- **Problem:** `residentIndexerScore` only runs when `gatherOps.indexerKeyBudgetBytes > 0` (`internal/model/v41_layer_step.go:773`, `internal/model/v41_indexer_score_device.go:120`). Only `v41_indexer_score_device_test.go:274,322,328` sets that field, so production always takes the host repack at `internal/model/v41_indexer_projection.go:137-146`: a fresh `make` and an append of every index key, on every index-source layer, every token.
- **Today:** ESTIMATED from the V4.1 config (head_dim 512, compress ratios 2/1): about 3.3 KB x pos per token, so about 13.6 MB/token at 4K context and about 109 MB/token at 32K. That is O(n^2) bytes over a generation, plus a likely re-upload of the same bytes to the device. Unmeasured.
- **Better because:** the device-resident scorer already exists and is tested; giving it a production budget removes the context-proportional host copy from the decode step.
- **Witness:** an allocation-scaling test (pos 32 vs 4096) on a reduced V4.1 fixture that is red today, plus a matched Halo A/B decode receipt.

## Problem frame

- Centrality: Core
- P1 advanced: advanced - removes an O(context) per-token host copy from the native decode critical path.
- P2 preserved: preserved - fak keeps ownership of the scorer, budget and fallback.
- P3 advanced: advanced - reuses the existing resident scorer instead of building a new one.
- P4 preserved: preserved - promotion requires parity plus a physical receipt; the software estimate is not a speed claim.

## Why this is next

This is the largest per-token copy that the 2026-10-16 hot-path copy sweep found, and its cause is a presence-vs-invokability gap: tested code that no production configuration turns on. The resident scorer and its tests exist, so the remaining work is wiring plus a budget policy.

## Current state

Classification: **PARTIAL**.

PRESENT: the resident indexer scorer, its budget accounting, and device tests (`v41_indexer_score_device.go`, `v41_indexer_score_device_test.go`).

ABSENT: a production writer of `indexerKeyBudgetBytes`, a budget policy tied to device memory admission, and any allocation-scaling ceiling over the V4.1 incremental step.

## Core through-line

Serve-time V4.1 load -> device memory admission computes an indexer-key budget -> `gatherOps.indexerKeyBudgetBytes` set > 0 -> `prepareV41LayerStepRole` takes `residentIndexerScore` -> no host `flat` repack -> same top-k selection as the host path.

## Working spine

V4.1 session load on Vulkan -> decode step at growing pos -> resident scorer selected -> allocation-scaling test green -> Halo A/B receipt.

## Scope

Production wiring of the indexer-key budget, its admission policy, and one allocation-scaling regression test.

## Gold-plating boundary

- No change to the scorer kernel, top-k semantics, or compressor math.
- No new device memory manager; reuse the existing admission inputs.
- No other per-token allocation fixes (tracked in TICKET-03).

## Done condition

- [ ] A production code path sets `indexerKeyBudgetBytes` from device admission (not only tests).
- [ ] Host fallback stays available when the budget is zero or exhausted, with a named reason.
- [ ] A reduced-V4.1 allocation-scaling test (pos 32 vs 4096) fails on the parent and passes on the candidate.
- [ ] Top-k index parity between host and resident paths over a multi-token run.
- [ ] Matched Halo A/B decode receipt at two context lengths.

## Definition of done

The resident path is the default on admitted Vulkan devices, the scaling test pins bytes/step growth, and a physical receipt reports the decode delta.

## Witness

`go test ./internal/model/ -run 'V41.*Indexer|V41.*AllocScaling' -count=1`, plus a matched Halo A/B decode receipt.

## Likely files

- `internal/model/v41_gather_ops_device.go`
- `internal/model/v41_layer_step.go`
- `internal/model/v41_indexer_projection.go`
- `internal/model/v41_indexer_score_device.go`
- `internal/model/v41_indexer_score_device_test.go`

## Parent context

Parent campaign: #12640 (DeepSeek V4.1 native decode performance). Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

A parity test plus a scaling test that is red on the parent and green on the candidate; a physical receipt is required before any tok/s claim.

## Closure binding

Close with the landing commit, the scaling test name, and the Halo receipt path.

## Work estimate

Estimate: 5 points (medium) - about 1-2 days, including the budget policy and the Halo A/B.

## Overall completion contribution

5/26 points of the hot-path copy sweep.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13870
