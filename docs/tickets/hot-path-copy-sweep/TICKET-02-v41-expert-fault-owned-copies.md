<!-- fak-model-key: v41-expert-fault-owned-copies -->
```routing
lane: model
paths: ["internal/model/gguf_expert_source.go", "internal/model/expert_pagecache.go", "internal/model/expert_checkpoint_tier.go", "internal/model/gguf_expert_mapped_range_test.go"]
expected_steps: 7
```

# perf(model): borrow the mapped GGUF on every routed-expert fault path, not only Q2_K/Q3_K with host budget 0

GitHub: [#13871](https://github.com/anthony-chaudhary/fak/issues/13871)

Process cause: scoping-failure

## Value

- **For:** fak-native MoE decode (DeepSeek V4.1 and other routed-expert models) whenever an expert misses the ring.
- **Problem:** public `e284f6e3e` borrows the mapped expert range only for Q2_K/Q3_K, host budget 0, no overlay, page cache off (gate at `internal/model/expert_checkpoint_tier.go:979`). Every other fault path still copies one whole expert projection into the Go heap:
  - `internal/model/gguf_expert_source.go:170-171`: `make(stride)` + `ReadAt` for Q4_K/Q5_K/Q6_K experts, or whenever the gate above fails.
  - `internal/model/expert_pagecache.go:305-306`: a cold read allocates and preads although `st.data` already holds the mapping.
  - `internal/model/expert_checkpoint_tier.go:566-576`: a borrowed page on any backend other than Vulkan (only `vulkan_borrowed_expert_upload.go:11` opts in) is `append`-copied at staging.
  - `internal/model/expert_checkpoint_tier.go:998-1001`: a compact copy when host budget > 0.
- **Today:** ESTIMATED about 3.9 MB per projection (Q2_K) and 6.6 MB (Q4_K); 40 layers x 6 experts x 3 projections = 720 stagings/token, so a 10% miss rate on Q4_K is about 475 MB/token of heap copy. Unmeasured; the ring miss rate decides real cost.
- **Better because:** the alias mechanism exists and is proven byte-identical; extending it removes the copy class rather than one instance.
- **Witness:** per-path `AllocsPerRun` ceilings (0 owned bytes for a borrowable fault) plus a matched A/B on a Q4_K MoE model.

## Problem frame

- Centrality: Core
- P1 advanced: advanced - removes per-miss whole-projection heap copies from MoE decode.
- P2 preserved: preserved - fak owns the expert tier, page cache and borrow contract.
- P3 advanced: advanced - extends the landed `e284f6e3e` borrow path instead of adding a new mechanism.
- P4 preserved: preserved - allocation ceilings are the software gate; speed needs a physical receipt.

## Why this is next

A 2026-10-12 same-day hardware A/B measured +3.6-4.0% decode from borrowing on the one gated path. The same copy remains on four sibling paths that the gate excludes.

## Current state

Classification: **PARTIAL**.

PRESENT: `mmapReaderAt.ReadOnlyRange` alias, `SupportsBorrowedExpertUploads` on Vulkan, borrowed-read alloc test (`gguf_expert_mapped_range_test.go:47`).

ABSENT: borrow on Q4_K/Q5_K/Q6_K, page-cache cold reads, non-Vulkan staging and host-budget compaction; allocation ceilings for those four paths.

## Core through-line

Routed-expert miss -> tier fault -> mapped range alias (any K-quant, any budget) -> backend stages from the alias -> no owned heap copy unless a named reason (overlay, unmapped reader) requires it.

## Working spine

MoE model on a mapped GGUF -> forced ring miss -> fault path -> allocation ceiling test per path -> A/B decode receipt.

## Scope

The four owned-copy fault paths above, each either borrowing or recording a named reason it cannot.

## Gold-plating boundary

- No ring sizing, prefetch, or eviction policy change.
- No new backend upload API beyond the existing borrowed-upload capability.
- No change to overlay semantics.

## Done condition

- [ ] Q4_K/Q5_K/Q6_K faults borrow when the reader is mapped.
- [ ] Page-cache cold reads alias `st.data` instead of a fresh pread buffer.
- [ ] Non-Vulkan backends either accept borrowed pages or copy once with a recorded reason.
- [ ] Host-budget compaction copies only bytes it must retain.
- [ ] `AllocsPerRun` ceiling per path, red on the parent for at least the Q4_K path.

## Definition of done

Every routed-expert fault path borrows or names why not, and the ceilings pin it.

## Witness

`go test ./internal/model/ -run 'Expert.*(Borrow|Mapped|Alloc)' -count=1`, plus a Halo or host A/B decode receipt on a Q4_K MoE model.

## Likely files

- `internal/model/gguf_expert_source.go`
- `internal/model/expert_pagecache.go`
- `internal/model/expert_checkpoint_tier.go`
- `internal/model/gguf_expert_mapped_range_test.go`

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Per-path allocation ceilings green; at least one red on the parent; byte parity of staged experts.

## Closure binding

Close with the landing commit, ceiling test names, and the A/B receipt path.

## Work estimate

Estimate: 5 points (medium) - about 2 days.

## Overall completion contribution

5/26 points of the hot-path copy sweep.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13871
