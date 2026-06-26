---
title: "fak CUDA-dev-process scorecard — the process-debt measuring stick"
description: "fak's deterministic CUDA-dev-process scorecard: KPIs across the five stages of the kernel development loop — author, build, validate, gate, onboard — folded into a composite score and the headline process-debt metric, re-derived from the git-tracked tree."
---

# CUDA-dev-process scorecard — how hard is it to develop a kernel in fak

<!-- cuda-dev-scorecard: 2026-06-26 · process: tools/cuda_dev_scorecard.py -->

This grades the **CUDA development loop**: changing `cuda_kernels.cu` / `cuda_backend.h` / `cuda.go`, building it, proving it correct, and keeping it from rotting — a loop made unusually painful because the canonical dev host has no CUDA toolkit and a walled GPU, so the loop spans a remote GPU node. Every number is re-derived from the git-tracked tree by `tools/cuda_dev_scorecard.py` — no hand-entry. The headline metric is **process-debt**: the count of concrete, mechanical defects that make the loop slower or less safe — a missing local gate, no automatic CI compile check, no one-command witness, no dev guide.

> Regenerate: `python tools/cuda_dev_scorecard.py --markdown --stamp DATE > docs/CUDA-DEV-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Process-debt (total HARD defects)** | **0** |
| Composite score | 100.0/100 (grade A) |
| Dev loop | author 100 · build 100 · validate 100 · gate 100 · onboard 100 |
| Advisory (soft) signals | 0 |
| Debt by stage | author:0 · build:0 · validate:0 · gate:0 · onboard:0 |

## The five stages of the kernel loop

14 KPIs, each 0–100, grouped by the loop stage they gate. `debt` = units of HARD process-debt.

| Stage | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| author | `local_static_check` | 100 | 0 | ABI checker + make cuda-check + ci.ps1 mirror all present |
| author | `abi_parity` | 100 | 0 | 34 prototypes in full parity (0 standby advisory) |
| author | `cpuref_parity_coverage` | 100 | 0 | 7/7 device op families have a cpuref-parity witness |
| build | `build_portable` | 100 | 0 | host matrix (WSL · GPU server · cloud · native Windows) + executable arch override covered |
| build | `toolchain_pinned` | 100 | 0 | CUDA version pinned + arch override documented |
| build | `task_runner` | 100 | 0 | cuda-build/cuda-test/cuda-accept delegate to the real scripts |
| validate | `witness_coverage` | 100 | 0 | every recorded floor names an on-disk acceptance witness |
| validate | `witness_aggregator` | 100 | 0 | tools/cuda_acceptance.sh runs every witness with one verdict |
| validate | `floor_honesty` | 100 | 0 | every recorded floor carries the recorded-not-measured caveat |
| gate | `cgo_typecheck_gate` | 100 | 0 | `go vet -tags cuda` runs automatically on every push/PR |
| gate | `nvcc_compile_gate` | 100 | 0 | an automatic job compiles + links the cuda variant (no GPU) |
| gate | `pure_go_guard` | 100 | 0 | an automatic gate asserts the default build is pure-Go |
| onboard | `dev_guide` | 100 | 0 | dev guide present, names real artifacts, covers the loop |
| onboard | `entrypoint_surfaced` | 100 | 0 | the dev guide is linked from the orientation path and resolves |

## Process-debt work-list

No process-debt: the CUDA dev loop has a local gate, automatic CI coverage, a one-command witness, and a documented path. 🎉

