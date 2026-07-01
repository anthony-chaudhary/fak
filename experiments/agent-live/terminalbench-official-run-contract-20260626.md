# Terminal-Bench Official-Run Contract

- Generated: `2026-07-01T15:42:19Z`
- Benchmark: `Terminal-Bench Harbor/Codex official-run contract`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- Evidence class: `EXTERNAL_RUN_CONTRACT`
- Result claim allowed: `false`
- Local fixture artifact: `experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json`
- Boundary: External-run contract only: fixes the raw/fak Harbor command shape, shared task/model/image/budget requirements, fak gateway adapter wiring, evidence paths, and promotion gates. It is not an official result until benchmark-native Harbor task logs, test output, gateway witness, and a raw-vs-fak compare artifact are checked in.

## Task Selection

- Candidate suite: `testdata\terminalbench\command_boundary_smoke.json`
- Candidate task ids: `go-cli-help-benign, python-config-fix-danger-after-tests`
- Official dataset: `terminal-bench/terminal-bench-2-1`
- Concurrent tasks: `1`
- Shared model: `gpt-5.5`
- Terminal-Bench 2.1 top-agent model: `gpt-5.5`
- Public agent label: `codex-cli`

| Candidate | Image | Budget turns | Test oracle |
|---|---|---:|---|
| `go-cli-help-benign` | `golang:1.26` | 2 | `fixture-recorded-go-test` |
| `python-config-fix-danger-after-tests` | `python:3.12-slim` | 4 | `fixture-recorded-pytest` |

## Arms

| Arm | Harness | Output | Command |
|---|---|---|---|
| `raw-terminalbench` | `benchmark-native` | `experiments/agent-live/terminalbench-official-raw-20260626` | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { harbor run -d terminal-bench/terminal-bench-2-1 -a codex -m gpt-5.5 --task-id $task --n-concurrent 1 } |
| `fak-terminalbench` | `benchmark-native-through-fak-gateway` | `experiments/agent-live/terminalbench-official-fak-20260626` | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { harbor run -d terminal-bench/terminal-bench-2-1 -a codex -m gpt-5.5 --task-id $task --n-concurrent 1 --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal } |

## Score Evidence Link

- Required: `true`
- Official test artifacts: `experiments/agent-live/terminalbench-official-raw-20260626/harbor-test-results.json`, `experiments/agent-live/terminalbench-official-raw-20260626/harbor-command-log.jsonl`, `experiments/agent-live/terminalbench-official-fak-20260626/harbor-test-results.json`, `experiments/agent-live/terminalbench-official-fak-20260626/harbor-command-log.jsonl`
- fak command evidence files: `experiments/agent-live/terminalbench-official-fak-20260626/fak-command-evidence.jsonl`, `experiments/agent-live/terminalbench-official-fak-20260626/raw-fak-command-join.json`, `experiments/agent-live/terminalbench-official-fak-20260626/fak-gateway-witness.json`
- Join keys: `task_id`, `command_index`, `command`, `cwd`, `evidence_id`, `state_hash`
- Detail: The official compare artifact must join Harbor test/pass rows and command logs to the mediated fak command verdict and evidence checkpoint for the same task command.

## Gates

| Gate | OK | Detail |
|---|:---:|---|
| `candidate_task_ids` | yes | 2 candidate ids from local Terminal-Bench-shaped smoke suite |
| `official_dataset_pin` | yes | terminal-bench/terminal-bench-2-1 |
| `same_task_ids_required` | yes | raw and fak official runs must use the same benchmark-native Terminal-Bench task ids |
| `same_image_required` | yes | raw and fak official runs must use the same benchmark-provided image or environment setup for each task |
| `same_budget_required` | yes | raw and fak official runs must use the same task budget and retry policy |
| `same_agent_required` | yes | raw=codex fak=codex |
| `same_model_required` | yes | gpt-5.5 |
| `top_agent_model_current` | yes | gpt-5.5 |
| `harbor_codex_adapter` | yes | Harbor adapter name must be codex; codex-cli is only the public leaderboard label |
| `raw_arm_command` | yes | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { harbor run -d terminal-bench/terminal-bench-2-1 -a codex -m gpt-5.5 --task-id $task --n-concurrent 1 } |
| `fak_arm_command` | yes | $env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { harbor run -d terminal-bench/terminal-bench-2-1 -a codex -m gpt-5.5 --task-id $task --n-concurrent 1 --agent-env OPENAI_BASE_URL=http://host.docker.internal:18080/v1 --agent-env OPENAI_API_BASE=http://host.docker.internal:18080/v1 --agent-env 'OPENAI_API_KEY={{FAK_GATEWAY_KEY}}' --allow-agent-host host.docker.internal } |
| `fak_gateway_agent_env` | yes | fak Harbor arm must pass OPENAI_BASE_URL, OPENAI_API_BASE, and OPENAI_API_KEY through --agent-env |
| `fak_gateway_host_allowlist` | yes | fak Harbor arm must allow the Docker agent to reach the host fak gateway |
| `official_harness_required` | yes | external Harbor run output with benchmark-native test results is required before promotion |

## Required Before Any Result Claim

- benchmark-native Terminal-Bench task ids for the selected fixed subset
- Terminal-Bench image or environment setup manifest for each selected task
- raw-arm Harbor run directory with command log and official test output over those exact task ids
- fak-arm Harbor run directory with command log and official test output over those exact task ids
- proof that raw and fak arms used the same Harbor codex adapter, model, task ids, image or environment, budget, concurrency, and retry policy
- fak per-command verdict/evidence log linked to the corresponding Harbor command log and test output
- gateway witness proving at least one structured model HTTP success and one gateway inference-turn event from the Dockerized Codex agent
- raw/fak compare artifact reporting benchmark-native solve separately from safe resolve, blocked dangerous actions, unnecessary blocks, runtime, cost or token budget, and evidence completeness
