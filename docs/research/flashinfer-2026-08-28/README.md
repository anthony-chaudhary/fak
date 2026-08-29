---
title: "FlashInfer current-source refresh - August 28, 2026"
description: "Pinned FlashInfer source refresh covering repository state, releases, issues, pull requests, and implications for fak-native inference work."
---

# FlashInfer current-source refresh — 2026-08-28

**Verdict:** pinned main advanced eight commits to `93f4f2642e1b3680a52ebb51cf68e0fdad237796`. The synchronized forge corpus validates at cutoff `2026-08-28T13:46:58Z` with **5,248 records** (3,543 pulls, 1,232 issues, 360 releases, 87 labels, 24 discussions, 2 milestones). `v0.6.18rc10` is divergent: 45 release-only commits and a different CCCL gitlink. The refresh preserves the earlier 2026-08-22/27 observations and supersedes their current-state decision surfaces without rewriting history.

## What changed the decision

1. **DCP speculative attention:** striped context-parallel decode carries caller-owned workspace, exact empty-rank behavior, split validation, and state merge semantics (`flashinfer/cake_dcp.py:249-257,439-671@93f4f2642e1b3680a52ebb51cf68e0fdad237796`). FAK is **ABSENT on-axis** at `internal/compute/compute.go:405-407` and `internal/model/tensor_parallel_forward.go:1-95`; it remains a Blackwell/Qwen3.8 operating-envelope **WATCH**, not a gain claim.
2. **Per-query windows:** PrimTS validates and plans a fixed-shape variable left window per query (`flashinfer/attention/prims_ts/context.py:231-242,661-787@93f4f2642e1b3680a52ebb51cf68e0fdad237796`). FAK's `WindowPerLayer` is **PARTIAL** (`internal/compute/compute.go:292-300`).
3. **Artifact identity and dispatch gates:** a multi-arch package is filtered before build (`flashinfer/jit/trtllm_gen_metainfo.py:72-168@93f4f2642e1b3680a52ebb51cf68e0fdad237796`), so the filtered bytes need derived identity. Hardware capability, compiler emit capability, and op/shape applicability are separate facts (`flashinfer/cute_dsl/utils.py:105-231@93f4f2642e1b3680a52ebb51cf68e0fdad237796`).
4. **Variable-length top-k:** CUB's two-phase workspace contract and per-segment bounds are explicit (`csrc/cub_topk.cu:42-52,157-258@93f4f2642e1b3680a52ebb51cf68e0fdad237796`); FAK is PARTIAL and keeps discrete-set correctness as the gate.

## Receipts

- Durable decision receipt: `study_6a93ae6773023dd3e547b7e25d2b705701dbc77f6b875a9cc150b744edcfdc57`.
- Forge corpus: `../inventory/flashinfer-ai-flashinfer-corpus-2026-08-28.jsonl`, 17,501,344 bytes, SHA-256 `8a989301e7f4a1fd02fe0e24afd57451eec29ed68a34b3fe7bfa0607c464d7fd`.
- Validator: `fak dev study-forge validate --receipt docs/research/inventory/flashinfer-ai-flashinfer-corpus-2026-08-28.jsonl`.
- Sequential endpoint capture is not called atomic; the receipt records the accepted bounded identity delta.

## Audit map

- `main-delta.json` — all eight post-`39b484f1...` main commits and changed paths.
- `release-divergence.json` — rc10 tag/commit, 45 release-only commits, review findings, and CCCL divergence.
- `candidate-ledger.json` — 30 dated candidates, exact anchors, self-query verdicts, seams, license and portfolio dispositions, refresh triggers, and dedup target.
- `self-query.json` — PRESENT/PARTIAL/ABSENT commands and null-result evidence.
- `license-provenance.json` — root LICENSE, NOTICE, four bundled licenses, `.gitmodules`, and all four gitlinks.
- `dedup-map.json` — live GitHub read-back, including closed outcomes #3088/#4184/#4202/#4356/#6042/#8608/#8609.
- `source-completeness.json` — source-class accounting and honest limitations.

## Boundary

No upstream source was copied, no FlashInfer runtime dependency or automatic engine fallback was added, and no FAK performance gain is claimed. Any performance promotion must run fak-native on an explicit hardware/model/shape envelope with quality and total setup/recovery/verification cost included.
