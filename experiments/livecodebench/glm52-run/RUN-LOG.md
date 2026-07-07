# LiveCodeBench — actual grading run (release_v2)

**Issue:** #3060 (GLM-5.2 LiveCodeBench) · **Date:** 2026-07-07 · **Host:** Windows 11 dev box

## What actually ran

The **official** LiveCodeBench code-generation grader
(`lcb_runner.evaluation.compute_code_generation_metrics.codegen_metrics`,
harness commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`) was executed
end-to-end over **real `release_v2` problems** loaded through the official
`load_code_generation_dataset` loader (full 1.25 GB dataset downloaded and
decoded locally).

Two candidate generations per problem — one correct, one deliberately wrong —
so the pass@1 is discriminating, never a trivial 1.0.

| generation | verdict (official grader) | tests |
|---|---|---|
| `1873_A` correct | **PASS** | 5/5 |
| `1873_A` wrong (always `YES`) | **FAIL** | 0/5 |
| `1899_A` correct | **PASS** | 13/13 |
| `1899_A` wrong (always `First`) | **FAIL** | 0/13 |

**Official `pass@1` = 0.5** (`official-grading-result.json`). The grader
correctly distinguished correct from incorrect solutions on real problems.

## The honest boundary — this is NOT a GLM-5.2 score

- **No model endpoint is reachable from this host.** GLM-5.2 is served on the
  lab machines; the fak gateway (`http://127.0.0.1:8080/v1`) is down here, and
  Docker's daemon is not running. So the **generation** half could not use a
  model — the candidate solutions above were **hand-authored**.
- Therefore this run **validates the grading harness + the fak→official
  grading seam execute end-to-end over real problems**. It does **not** and
  **cannot** stand in for the GLM-5.2 benchmark result.
- `result_claim_allowed` stays **false** in `run-contract.json`. A published
  GLM-5.2 pass@1 requires the two-arm run in the contract, executed on the lab
  gateway, graded by the official evaluator.

## Environment (so the run is reproducible)

- Python **3.11.15** via `uv python install 3.11` (lcb_runner pins 3.11).
- `datasets==2.21.0` (5.x dropped the script-based loader the LCB lite repo ships).
- Windows shim: `sitecustomize.py` provides `signal.SIGALRM`/`signal.alarm` as
  no-ops (Unix-only syscalls `run_test` expects). The **real** wall-clock
  timeout still fires — `check_correctness` enforces it via
  `multiprocessing.Process` + `p.join(timeout=...)`, which is cross-platform.
  No official grading logic was modified.

## To get the real GLM-5.2 number (on the lab machine)

1. Bring up the fak gateway routed to the GLM-5.2 serving backend.
2. Run **both arms** in `run-contract.json` (`raw-lcb_runner` and `fak-native`)
   over the pinned `release_v2` window, same `problem_selection`.
3. Grade **both** with the official evaluator per the contract's `grading`
   block. Only that evaluator's output may back a published pass@1.

## Artifacts in this directory

- `run-contract.json` / `run-contract.md` — the two-arm official-run contract (#2110), `status: READY_FOR_OFFICIAL_RUN`, pinned to the real `release_v2` ids.
- `suite-real.json` — the real `release_v2` candidate problems (ids `1873_A`, `1873_B`, `1873_D`, `1883_B`, `1883_C`, `1899_A`).
- `official-grading-result.json` — the official grader output (pass@1 = 0.5).
- `real-problems-manifest.json` — id/platform/difficulty/test-count of the loaded problems.
