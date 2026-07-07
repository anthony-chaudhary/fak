# LiveCodeBench Official-Run Contract

- Generated: `2026-07-07T19:09:32Z`
- Issue: `#3060`
- Benchmark: `LiveCodeBench official-run contract (raw lcb_runner vs fak-native)`
- Status: `READY_FOR_OFFICIAL_RUN`
- Evidence class: `OFFICIAL_RUN_CONTRACT`
- Result claim allowed: `false`
- Boundary: Official-run contract only: it pins the run constants, the raw and fak generation commands, the shared problem/prompt/model/release requirements, and the official grading handoff. It performs no run and claims no pass-rate; result_claim_allowed stays false until the official evaluator grades the exact saved generations (#2113).

## Constants

| Constant | Value |
|---|---|
| Release | `release_v2` (selector `release_v2`) |
| Scenario | `codegeneration` |
| Date window | `2024-09-01 .. 2024-11-20` |
| Model | `glm-5.2` |
| Serving backend | `SGLang W4AFP8` |
| Gateway | `http://127.0.0.1:8080/v1` |
| Run dir | `experiments/livecodebench/glm52-run` |

## Problem Selection

- Candidate suite: `experiments/livecodebench/glm52-run/suite-real.json`
- Candidate question ids: `1873_A`, `1873_B`, `1873_D`, `1883_B`, `1883_C`, `1899_A`

## Arms

### `raw-lcb_runner` (official-lcb_runner-through-fak-gateway)

- Output: `experiments/livecodebench/glm52-run/raw/codegeneration`
- Notes: requires an lcb_runner model registration pointing at the fak gateway; --evaluate produces the official pass@1 directly.

```bash
python -m lcb_runner.runner.main --model glm-5.2 --scenario codegeneration --release_version release_v2
```

### `fak-native` (fak-livecodebench-generate-through-fak-gateway)

- Output: `experiments/livecodebench/glm52-run/fak/codegeneration`
- Notes: generates only; the SAME problem_ids/prompt_hash/model/release/scenario as the raw arm, exported to the official custom-evaluator shape.

```bash
fak livecodebench generate --gateway http://127.0.0.1:8080/v1 --model glm-5.2 --release-version release_v2 --scenario codegeneration --start-date 2024-09-01 --end-date 2024-11-20 --out experiments/livecodebench/glm52-run/fak/codegeneration
```

```bash
go run ./cmd/livecodebench export --format custom-evaluator --fixture experiments/livecodebench/glm52-run/fak-codegeneration-fixture.json --out experiments/livecodebench/glm52-run/fak-codegeneration-custom.json
```

## Grading (sole result authority: official LiveCodeBench lcb_runner)

```bash
# raw arm direct grade
python -m lcb_runner.runner.main --model glm-5.2 --scenario codegeneration --evaluate --release_version release_v2

# fak arm: grade the exported generations
python -m lcb_runner.runner.custom_evaluator --custom_output_file experiments/livecodebench/glm52-run/fak-codegeneration-custom.json --release_version release_v2

# date-windowed score
python -m lcb_runner.evaluation.compute_scores --eval_all_file experiments/livecodebench/glm52-run/<official-eval-all-file> --start_date 2024-09-01 --end_date 2024-11-20
```

Grading is the sole result-bearing authority. The exact saved generations from each arm are graded by the official evaluator; only then may LIVECODEBENCH-RESULTS.md be filled and result_claim_allowed flip true (#2113).

## Gates

| Gate | OK | Detail |
|---|:---:|---|
| `release_pinned_explicit` | yes | resolved release_v2 |
| `scenario_known` | yes | codegeneration |
| `date_window_recorded` | yes | 2024-09-01 .. 2024-11-20 |
| `model_recorded` | yes | glm-5.2 |
| `serving_backend_recorded` | yes | SGLang W4AFP8 |
| `fak_gateway_recorded` | yes | http://127.0.0.1:8080/v1 |
| `candidate_problem_ids` | yes | 6 candidate question ids from the pinned suite |
| `same_problem_ids_required` | yes | raw and fak arms must score the identical question_ids |
| `same_prompt_hash_required` | yes | raw and fak arms must send the identical rendered prompt per problem (SamePromptHash) |
| `same_release_required` | yes | both arms must use the identical release_version |
| `official_grading_required` | yes | the exact saved generations must be graded by the official lcb_runner evaluator before any claim |

## Required Before Any Result Claim

- release_version, scenario, start_date, and end_date recorded
- model identity, serving backend, and model training-cutoff statement or explicit residual
- raw-arm saved generations + SHA256 digest
- fak-arm saved generations + SHA256 digest
- SameProblemIDs and SamePromptHash asserted between raw and fak arms over the same release
- official grading command + eval_all artifact digest for each arm
- result_claim_allowed flips true only after official grading (#2113); LIVECODEBENCH-RESULTS.md cells filled for both arms
