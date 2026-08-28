# FlashInfer current-source refresh: two attention mechanisms and two provenance contracts

**Verdict:** the source-pinned refresh at `flashinfer-ai/flashinfer@93f4f2642e1b3680a52ebb51cf68e0fdad237796` finds two decision-changing attention mechanisms—striped context-parallel speculative decode and fixed-shape per-query variable windows—plus two provenance contracts: filtered artifacts need derived identity, and dispatch eligibility must separate hardware capability, compiler emit capability, and op/shape applicability. No FAK performance gain, runtime dependency, or automatic fallback is claimed.

Observed at: `2026-08-28T13:48:46Z`
Forge cutoff: `2026-08-28T13:46:58Z`
Durable receipt: `study_6a93ae6773023dd3e547b7e25d2b705701dbc77f6b875a9cc150b744edcfdc57`

## Source and integrity

The new forge capture contains 5,248 records: 3,543 pulls, 1,232 issues, 360 releases, 87 labels, 24 discussions, and 2 milestones. `fak dev study-forge validate --receipt docs/research/inventory/flashinfer-ai-flashinfer-corpus-2026-08-28.jsonl` validates the exact 17,501,344 bytes. The capture is complete but sequential across endpoints; the embedded receipt's accepted identity deltas are an honesty boundary, not an atomic-snapshot claim.

The generated tree inventory now pins 3,121 files at the same main revision. The refresh preserves the 2026-08-22 note and 2026-08-27 research directory; it supersedes their current-state decisions rather than mutating historical observations.

## Eight-commit main delta

All eight commits after `39b484f1ce2fff086c66f9a899a0a58ba7f0ec3e` are enumerated in [`main-delta.json`](../research/flashinfer-2026-08-28/main-delta.json). The decision-changing rows are:

- **DCP speculative attention:** `flashinfer/cake_dcp.py:249-257,439-671@93f4f264...` makes empty ranks valid, validates split assignment, binds caller workspace, and carries exact rank-local merge semantics. FAK is **ABSENT on-axis**: `internal/compute/compute.go:405-407` exposes no DCP primitive and `internal/model/tensor_parallel_forward.go:1-95` covers standard GQA TP while failing closed on unsupported decompositions. This is `WATCH` for a Blackwell/Qwen3.8 hardware envelope; it is not a speed claim.
- **Per-query variable windows:** `flashinfer/attention/prims_ts/context.py:231-242,661-787@93f4f264...` validates positive causal windows and plans the geometry. FAK is **PARTIAL** because `internal/compute/compute.go:292-300` carries only per-layer `WindowPerLayer`.
- **Variable-length top-k:** `csrc/cub_topk.cu:42-52,157-258@93f4f264...` records the compile-time segment bound and two-phase workspace flow. FAK is **PARTIAL**: it has a discrete-set equality gate at `internal/compute/cuda_accuracy_gates.go:142-144`, not a general variable-length top-k API/workspace.
- **Three-axis eligibility:** `flashinfer/cute_dsl/utils.py:105-231@93f4f264...` proves that hardware support is not compiler-emission support. The calling op still owns shape/applicability. FAK has architecture membership checks but no single typed tuple joining all three axes.
- **Derived artifact identity:** `flashinfer/artifacts.py:138-199@93f4f264...` packages several architectures together; `flashinfer/jit/trtllm_gen_metainfo.py:72-168@93f4f264...` filters the manifest before building. The filtered bytes—not only the raw package pin—must name the derived build input.

The other three commits (per-token NVFP4 ReLU2 MoE, fused Blackwell SwiGLU+MXFP8, and bias-aware cuBLASLt fallback) remain candidate rows with explicit seams and refresh triggers. Their upstream measurements do not become FAK claims.

## Divergent `v0.6.18rc10`

The tag resolves to release commit `e62941a1da605fb9b3c8c50b23c9720df12cf6b4`, whose merge base with main is `61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd`; the symmetric history is 66 main-only versus 45 release-only commits. Therefore a main-tree delta is not a release review.

Two release-only findings change the evidence boundary:

1. `9ac0f5dbad7503afc45fc18a9f97c41c28e22e75` filters the TRTLLM manifest per architecture to restore build time, reinforcing derived filtered-manifest identity.
2. `f0ff7f3502bb4f4eb2338b460207ad8c171386c9` batches five SM107 fixes and reports release-specific CI failures; these are negative knowledge, not main-tree proof.

The release tag also pins `3rdparty/cccl@876867684f7fac130e0f5911236e0a92a970d4fd`, while main after the CUB top-k commit pins `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116`. [`release-divergence.json`](../research/flashinfer-2026-08-28/release-divergence.json) records all 45 release-only commits and this gitlink difference.

## License and ownership

The exact revision carries Apache-2.0 at `LICENSE:1@93f4f264...`, NVIDIA attribution in `NOTICE:1@93f4f264...`, four bundled license files (CUTLASS, FlashAttention-3, fmt, spdlog), and four gitlinks (CCCL, CUTLASS, NIXL, spdlog). Hashes, URLs, revisions, and the CCCL change are in [`license-provenance.json`](../research/flashinfer-2026-08-28/license-provenance.json).

No source, test, comment, or asset was copied in this research pass. Candidate-specific `ADAPT`, `INSPIRE-ONLY`, and `DO-NOT-USE` decisions remain bounded by file-level provenance review at implementation time. FlashInfer stays prior art/oracle only; FAK retains the engine, kernels, memory, scheduling, cache, adaptation, and operations.

## Self-query and dedup

[`self-query.json`](../research/flashinfer-2026-08-28/self-query.json) preserves the exact `fak capabilities`, moved `fak dev index ...`, and raw-grep commands. Plans/workspaces, artifact identity, attention portfolio, and variable-length top-k are `PARTIAL`; DCP attention is `ABSENT`; bounded tracing and offline tuning are `PRESENT` through closed #8609 and #8608.

The current issue read-back is also not the 2026-08-27 map: #3088, #4184, #4202, #4356, #6042, #8608, and #8609 are closed outcomes and must not be treated as active landing targets. Active gaps remap to their existing issues. No new implementation issue survived: DCP remains a hardware-envelope `WATCH` within research issue #9757.

## Refresh triggers

Re-run when any of these changes:

- main advances or `v0.6.18` gains another release candidate/final tag;
- release-only fixes merge or are cherry-picked to main;
- the CCCL, multi-arch artifact, or filtered-manifest pin changes;
- FAK ships DCP attention, per-query windows, a typed eligibility tuple, or derived artifact identity;
- a FAK-native Qwen3.8/Blackwell matched-envelope receipt supplies quality, memory, setup, and end-to-end timing evidence.

## Companions

- [`docs/research/flashinfer-2026-08-28/README.md`](../research/flashinfer-2026-08-28/README.md)
- [`docs/research/inventory/flashinfer-ai-flashinfer.json`](../research/inventory/flashinfer-ai-flashinfer.json)
- [`docs/notes/flashinfer-study-2026-08-22.md`](flashinfer-study-2026-08-22.md) (historical predecessor)
- [Issue #9757](https://github.com/anthony-chaudhary/fak/issues/9757)
