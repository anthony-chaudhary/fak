# Ponytail non-token gates (#6687)

## Verdict

**NOT-YET.** The gate runner and reproducible evidence are present, but the 2026-08-14 provider run found real category failures. Token or aggregate results must not override any row below.

Comparator: `DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3`.
Provider identity reference: configured account `aug8-netra`; model alias `haiku`. No credential material is recorded.

## Reproduce

```powershell
fak armbench ponytail-gates --checkout $env:TEMP\ponytail-2ed6c52-6687 `
  --out docs/_witnesses/armbench-ponytail-gates-dryrun-2026-08-14.json

fak armbench ponytail-gates --checkout $env:TEMP\ponytail-2ed6c52-6687 `
  --live --account aug8-netra --model haiku `
  --out docs/_witnesses/armbench-ponytail-gates-live-final-2026-08-14.json
```

Dry-run exit 3 is intentional: required provider cells are `not_run`. Live exit 3 means at least one arm/category gate failed. `--replay <prior.json>` re-scores captured provider outputs with the pinned graders and calls the provider only for missing/failed prior cells.

## Inventory and judging

The JSON witness records path, bytes, and SHA-256 for all eleven comparator inputs. Stable IDs cover:

- `up.behavior.*`: all 3 tests from `behavior.yaml`, scored by unchanged `behavior.js`.
- `up.correctness.*`: all 5 prompt-matrix tests, scored by unchanged `correctness.js`.
- `up.robustness.*`: all 16 `TASKS` in `robustness-audit.js`, scored by its unchanged `pyBlock` and `checkPy` exports.
- `up.correctness-regression.*`: all 4 named fixtures in `correctness.test.js`, executed with `node --test`; raw TAP is preserved.

No LLM judge is used. The live upstream audit's `n=20` loop is an execution setting, not an additional scenario; this witness uses one sample per cell and makes no repeat-rate claim. The requested arm expansion evaluates baseline, pinned Caveman, and pinned Ponytail even where upstream behavior/audit scripts originally list only two arms.

The separate `extensions` array contains detector-success examples for instruction leakage, malformed/no-code output, and over-compression. They are not counted as upstream cells or acceptance results.

## Captured result

| arm | behavior | correctness | robustness |
|---|---:|---:|---:|
| baseline | 1/3 | 3/5 | 16/16 |
| Caveman | 1/3 | 4/5 | 15/16 |
| Ponytail | 2/3 | 4/5 | 15/16 |

Pinned correctness regression fixtures: **4/4**.

Ponytail failures are explicit in the raw witness: hardware calibration (pinned behavior heuristic), React countdown (provider timeout), and IPv4 (pinned robustness instrument). Other arms also have failures. Therefore `overall_pass=false`; #6687 must remain open and no comparator acceptance is inferred.

## Evidence

- Dry fail-closed witness: [`_witnesses/armbench-ponytail-gates-dryrun-2026-08-14.json`](_witnesses/armbench-ponytail-gates-dryrun-2026-08-14.json)
- Complete provider witness: [`_witnesses/armbench-ponytail-gates-live-final-2026-08-14.json`](_witnesses/armbench-ponytail-gates-live-final-2026-08-14.json)

Intermediate provider/re-score files are intentionally not authoritative; the final witness contains the complete output set and assumptions.
