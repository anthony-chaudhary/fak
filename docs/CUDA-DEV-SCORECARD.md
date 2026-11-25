---
title: "fak CUDA-dev-process scorecard — the process-debt measuring stick"
description: "fak's deterministic CUDA-dev-process scorecard: KPIs across the five stages of the kernel development loop — author, build, validate, gate, onboard — folded into a composite score and the headline process-debt metric, re-derived from the git-tracked tree."
---

# CUDA-dev-process scorecard — how hard is it to develop a kernel in fak

<!-- cuda-dev-scorecard: 2026-11-24 · process: tools/cuda_dev_scorecard.py -->

This grades the **CUDA development loop**: changing `cuda_kernels.cu` / `cuda_backend.h` / `cuda.go`, building it, proving it correct, and keeping it from rotting — a loop made unusually painful because the canonical dev host has no CUDA toolkit and a walled GPU, so the loop spans a remote GPU node. Every number is re-derived from the git-tracked tree by `tools/cuda_dev_scorecard.py` — no hand-entry. The headline metric is **process-debt**: the count of concrete, mechanical defects that make the loop slower or less safe — a missing local gate, no automatic CI compile check, no one-command witness, no dev guide.

> Regenerate: `python tools/cuda_dev_scorecard.py --markdown --stamp DATE > docs/CUDA-DEV-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Process-debt (total HARD defects)** | **19** |
| Composite score | 83.0/100 (grade B) |
| Dev loop | author 51 · build 72 · validate 100 · gate 100 · onboard 88 |
| Advisory (soft) signals | 0 |
| Debt by stage | author:16 · build:2 · validate:0 · gate:0 · onboard:1 |

## The five stages of the kernel loop

14 KPIs, each 0–100, grouped by the loop stage they gate. `debt` = units of HARD process-debt.

| Stage | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| author | `abi_parity` | 0 | 15 | 15 ABI mismatch(es) |
| build | `task_runner` | 30 | 1 | 3 cuda verb(s) missing/undelegating |
| author | `local_static_check` | 70 | 1 | 1 missing local-check affordance(s) |
| build | `build_portable` | 78 | 1 | 1 portability gap(s) |
| onboard | `dev_guide` | 82 | 1 | 1 content gap(s) in the dev guide |
| author | `cpuref_parity_coverage` | 100 | 0 | 7/7 device op families have a cpuref-parity witness |
| build | `toolchain_pinned` | 100 | 0 | CUDA version pinned + arch override documented |
| validate | `witness_coverage` | 100 | 0 | every recorded floor names an on-disk acceptance witness |
| validate | `witness_aggregator` | 100 | 0 | tools/cuda_acceptance.sh runs every witness with one verdict |
| validate | `floor_honesty` | 100 | 0 | every recorded floor carries the recorded-not-measured caveat |
| gate | `cgo_typecheck_gate` | 100 | 0 | `go vet -tags cuda` runs automatically on every push/PR |
| gate | `nvcc_compile_gate` | 100 | 0 | an automatic job compiles + links the cuda variant (no GPU) |
| gate | `pure_go_guard` | 100 | 0 | an automatic gate asserts the default build is pure-Go |
| onboard | `entrypoint_surfaced` | 100 | 0 | the dev guide is linked from the orientation path and resolves |

## Process-debt work-list

### `abi_parity` (author) — 15 defect(s), score 0
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_attend() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_bucket() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_combine() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_conv() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_ctrl_apply() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_ctrl_down() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_expert_iq() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_expert_nvfp4() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_gather_pairs() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_gemm_f32() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_max_tiles() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_mix() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_qkv_post() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_route() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable
- definition with no prototype: internal/compute/cuda_kernels.cu, internal/compute/cuda_nccl.cu, internal/compute/cuda_nccl_pg.cu, internal/compute/cuda_deepgemm_wkv.cu, internal/compute/cuda_flashmla_sparse_decode.cu, internal/compute/cuda_flashmla_sparse_prefill.cu exports fcuda_ink2p_scatter_pairs() (extern "C") but the ABI header closure and every cgo preamble declare no prototype — the seam is undocumented / unreachable

### `local_static_check` (author) — 1 defect(s), score 70
- no `make cuda-check` target whose recipe runs tools/cuda_abi_parity.py --check (the one-command local gate)

### `build_portable` (build) — 1 defect(s), score 78
- the GPU arch is not in an executable context (no FAK_CUDA_ARCH override + -arch="$ARCH" nvcc line) — the arch matrix is a comment, not a build path

### `task_runner` (build) — 1 defect(s), score 30
- the Makefile exposes no cuda build/test/accept verb that delegates to the real scripts (missing/undelegating: cuda-build, cuda-test, cuda-accept) — a dev has to know `go run ./cmd/fak-dev build-cuda legacy` by heart

### `dev_guide` (onboard) — 1 defect(s), score 82
- docs/cuda-dev.md names an artifact that does not exist: internal/compute/cuda_archive_stamp_gen.go (doc-rot — a dev who pastes it hits a missing-file error)

