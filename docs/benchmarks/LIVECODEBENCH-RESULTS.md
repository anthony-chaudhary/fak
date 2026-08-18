---
title: "LiveCodeBench results"
description: "Provenance-bound LiveCodeBench results through fak, with explicit workload and evidence boundaries."
---

# LiveCodeBench Results

Status: **bounded official-workload result published**. The full-window matched-arm
leaderboard campaign remains pending; the result below closes the narrower real-model
→ exact-generation → official-evaluator spine.

Upstream: [LiveCodeBench](https://github.com/livecodebench/livecodebench) (official
`lcb_runner` harness). Repo epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085).
Runbook: [LIVECODEBENCH-RUNBOOK.md](LIVECODEBENCH-RUNBOOK.md).
Submission gate: [LIVECODEBENCH-SUBMISSION-PACKET.md](LIVECODEBENCH-SUBMISSION-PACKET.md).

## Proper Official-Workload Spine (2026-08-18)

| Model / backend | Official workload | Sampling | Official pass@1 | Evidence |
|---|---|---:|---:|---|
| Qwen3.6-27B-Q4_K_M / fak in-kernel CUDA on A100 40 GB | deterministic 8-problem `release_v6` easy subset | n=1, temperature=0, seed=42 | **0.25 (2/8)** | [`RESULT.md`](../../experiments/livecodebench/runs/qwen36-release_v6-easy8-2026-08-18/RESULT.md) |

These solutions came from a real model over pinned official LiveCodeBench problems;
the exact saved generations were graded by the official evaluator (`28fef95…`). This
is a bounded subset result, not a full-release leaderboard score. Four outputs hit the
512-token cap and remained failures; no repair or selection was applied. The recovered 2026-08-17 device campaign is a **LiveCodeBench-shaped device workload**, not an official
LCB score, and is intentionally absent from this result table.

## Full-Window Result Ledger

| Field | Value |
|---|---|
| Benchmark release | pending run |
| Contest date window | pending run |
| Scenario | pending run |
| Model / serving backend | pending run |
| Engine (pure-kernel arm) | `pending GPU run` — records `engine=inkernel` + backend once served; see [runbook §3a](LIVECODEBENCH-RUNBOOK.md#3a-pure-kernel-codegen-arm-no-external-engine-in-the-path) |
| Raw arm generation artifact | pending run |
| fak arm generation artifact | pending run |
| Official grading artifact | pending run |
| Evidence class | pending run |
| Promotion status | pending run |
| pass@1 | pending run |
| pass@5 | pending run |

## Bounded Pure-Kernel Witness (2026-07-14)

| Field | Witness |
|---|---|
| Evidence class | `official-lcb-runner-graded` (bounded contest-day slice; not a full-release score) |
| fak model path | `engine=inkernel`, CPU reference backend, no upstream `--base-url` |
| Model | `SmolLM2-135M-Instruct-Q8_0`, SHA-256 `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83` |
| Dataset | `livecodebench/code_generation_lite`, `release_v2`, `codegeneration`, contest date `2023-12-16` |
| Coverage | 8 real problems, `n=1`, temperature `0`; 4,218 prompt tokens + 3,235 completion tokens |
| Official evaluator | upstream LiveCodeBench `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, exit `0` |
| Result | bounded pass@1 `0.0` (0/8); this small model proves the live wiring, not coding quality |
| Generation report | SHA-256 `9d51b2710b1d52780f8ebe9e9e8517c5b61b572d8c184eb1b5e94e24560eee64` |
| Official `eval_all` | SHA-256 `21c1f060cb3c81e43f7119104656f826c5a3e03d1138504e4b1a074f1799339a` |
| Witness manifest | `fak.livecodebench-pure-kernel-witness.v1`, SHA-256 `6ce4dd0b970d63c62f1e79a8e54d275ff8c6fa474f3816a1f2747674ab8d9df7` |

This retires the narrower claim that no LCB codegeneration pass rate has ever been produced through fak's own decode. It does not fill the full-window leaderboard table: promotion there still requires the complete pinned release/date window and matched baseline arms. The remote serve remained live at read-back with `/healthz` reporting `engine=inkernel`; its launch command contained `--gguf` and `--engine inkernel` and no proxy/base URL.

## Methodology: Release And Date Window

Every LiveCodeBench result must state both the benchmark release and the contest-date
window it was scored over. LiveCodeBench is designed as a time-aware coding benchmark:
the upstream project publishes versioned releases and supports date-windowed scoring.
Those two fields are not metadata decoration; they are the contamination boundary.

The minimum reportable identity for a fak LCB run is:

| Required field | Why it is required |
|---|---|
| `release_version` | Fixes the dataset snapshot instead of relying on a moving `release_latest`. |
| `scenario` | Separates code generation, self-repair, test-output prediction, and code execution. |
| `start_date` / `end_date` | States which contest-publication window was scored. |
| Model identity and training-cutoff statement | Makes the contamination risk legible; unknown cutoff stays an explicit residual. |
| Generation artifact digest | Binds raw and fak arms to the exact completions later graded by the official harness. |
| Official grading command and artifact | Prevents a local proxy metric from being reported as an LCB result. |

No score is promoted from this scaffold until the same saved generations have been graded
by the official LiveCodeBench evaluator. A fak-only dry run, a local smoke, or a harness
preflight remains `pending run` here.

For the pure-kernel arm (codegen served by `fak serve --gguf --engine inkernel`, no
external engine — [runbook §3a](LIVECODEBENCH-RUNBOOK.md#3a-pure-kernel-codegen-arm-no-external-engine-in-the-path)),
the `Engine (pure-kernel arm)` row records `engine=inkernel` and the serving backend once
the model is served, but the `pass@1` / `pass@5` cells stay `pending GPU run`: a pure-kernel
LCB codegen pass rate is only reportable after a real GPU run through the in-kernel path
produces one from the official evaluator.

## Fill Procedure

1. Run the LiveCodeBench flow from [LIVECODEBENCH-RUNBOOK.md](LIVECODEBENCH-RUNBOOK.md).
2. Record the exact `release_version`, `scenario`, and date window used for scoring.
3. Attach the raw-arm and fak-arm generation artifacts with digests.
4. Attach the official grading output produced by `lcb_runner`.
5. Replace only the corresponding `pending run` cells above.

## Links

- Epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085)
- Results scaffold child: [#2119](https://github.com/anthony-chaudhary/fak/issues/2119)
- Runbook child: [#2118](https://github.com/anthony-chaudhary/fak/issues/2118)
- Submission packet child: [#2115](https://github.com/anthony-chaudhary/fak/issues/2115)
- Upstream harness: [LiveCodeBench](https://github.com/livecodebench/livecodebench)

