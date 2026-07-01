# LiveCodeBench Results Scaffold

Status: **pending run**. This page is an authority placeholder only; it records the
fields a real LiveCodeBench result must carry, but it does not report any pass rate.

Upstream: [LiveCodeBench](https://github.com/livecodebench/livecodebench) (official
`lcb_runner` harness). Repo epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085).
Runbook: [LIVECODEBENCH-RUNBOOK.md](LIVECODEBENCH-RUNBOOK.md).

## Result Ledger

| Field | Value |
|---|---|
| Benchmark release | pending run |
| Contest date window | pending run |
| Scenario | pending run |
| Model / serving backend | pending run |
| Raw arm generation artifact | pending run |
| fak arm generation artifact | pending run |
| Official grading artifact | pending run |
| Evidence class | pending run |
| Promotion status | pending run |
| pass@1 | pending run |
| pass@5 | pending run |

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
- Upstream harness: [LiveCodeBench](https://github.com/livecodebench/livecodebench)
