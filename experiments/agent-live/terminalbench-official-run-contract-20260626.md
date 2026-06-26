# Terminal-Bench Official-Run Contract

- Generated: `2026-06-26T03:13:56Z`
- Benchmark: `Terminal-Bench command-boundary official-run contract`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- Evidence class: `EXTERNAL_RUN_CONTRACT`
- Result claim allowed: `false`
- Local fixture artifact: `experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json`
- Boundary: External-run contract only: fixes the raw/fak Terminal-Bench command shape, shared task/model/image/budget requirements, evidence paths, and promotion gates. It is not an official result until benchmark-native tb run task logs, test output, and a raw-vs-fak compare artifact are checked in.

## Task Selection

- Candidate suite: `testdata/terminalbench/command_boundary_smoke.json`
- Candidate task ids: `go-cli-help-benign, python-config-fix-danger-after-tests`
- Official dataset: `terminal-bench-core==0.1.1`
- Concurrent tasks: `1`

| Candidate | Image | Budget turns | Test oracle |
|---|---|---:|---|
| `go-cli-help-benign` | `golang:1.26` | 2 | `fixture-recorded-go-test` |
| `python-config-fix-danger-after-tests` | `python:3.12-slim` | 4 | `fixture-recorded-pytest` |

## Arms

| Arm | Harness | Output | Command |
|---|---|---|---|
| `raw-terminalbench` | `benchmark-native` | `experiments/agent-live/terminalbench-official-raw-20260626` | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset terminal-bench-core==0.1.1 --agent terminus --model shared-agent-model --task-id $task --n-concurrent 1 } |
| `fak-terminalbench` | `benchmark-native-through-fak-gateway` | `experiments/agent-live/terminalbench-official-fak-20260626` | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset terminal-bench-core==0.1.1 --agent terminus-through-fak --model shared-agent-model --task-id $task --n-concurrent 1 } |

## Gates

| Gate | OK | Detail |
|---|:---:|---|
| `candidate_task_ids` | yes | 2 candidate ids from local Terminal-Bench-shaped smoke suite |
| `official_dataset_pin` | yes | terminal-bench-core==0.1.1 |
| `same_task_ids_required` | yes | raw and fak official runs must use the same benchmark-native Terminal-Bench task ids |
| `same_image_required` | yes | raw and fak official runs must use the same benchmark-provided image or environment setup for each task |
| `same_budget_required` | yes | raw and fak official runs must use the same task budget and retry policy |
| `same_model_required` | yes | shared-agent-model |
| `raw_arm_command` | yes | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset terminal-bench-core==0.1.1 --agent terminus --model shared-agent-model --task-id $task --n-concurrent 1 } |
| `fak_arm_command` | yes | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset terminal-bench-core==0.1.1 --agent terminus-through-fak --model shared-agent-model --task-id $task --n-concurrent 1 } |
| `official_harness_required` | yes | external tb run output with benchmark-native test results is required before promotion |

## Required Before Any Result Claim

- benchmark-native Terminal-Bench task ids for the selected fixed subset
- Terminal-Bench image or environment setup manifest for each selected task
- raw-arm tb run directory with command log and official test output over those exact task ids
- fak-arm tb run directory with command log and official test output over those exact task ids
- proof that raw and fak arms used the same model, task ids, image or environment, budget, concurrency, and retry policy
- fak per-command verdict/evidence log linked to the corresponding tb run command log and test output
- raw/fak compare artifact reporting benchmark-native solve separately from safe resolve, blocked dangerous actions, unnecessary blocks, runtime, cost or token budget, and evidence completeness
