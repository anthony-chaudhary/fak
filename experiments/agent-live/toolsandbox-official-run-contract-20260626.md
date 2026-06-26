# ToolSandbox/tau3 Official-Run Contract

- Generated: `2026-06-26T03:03:22Z`
- Benchmark: `ToolSandbox/tau3 policy-state official-run contract`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- Result claim allowed: `false`
- Local fixture artifact: `experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json`
- Boundary: External-run contract only: fixes the raw/fak command shape, shared model and simulator requirements, evidence paths, and promotion gates for a benchmark-native tau3 or ToolSandbox run. It is not an official result until external task definitions, raw/fak outputs, and native grader summaries are checked in.

## Task Selection

- Candidate suite: `testdata/toolsandbox/policy_state_smoke.json`
- Candidate task ids: `retail-refund-policy-minefield`
- Official harness: `tau3`
- Official domain: `retail`
- Trials: `1`

## Arms

| Arm | Harness | Output | Command |
|---|---|---|---|
| `raw-toolsandbox` | `benchmark-native` | `experiments/agent-live/toolsandbox-official-raw-20260626` | $env:TAU3_TASK_IDS='<space-separated official task ids>'; tau2 run --domain retail --agent-llm shared-agent-model --user-llm shared-agent-model --num-trials 1 --task-ids ($env:TAU3_TASK_IDS -split ' ') |
| `fak-toolsandbox` | `benchmark-native-through-fak-gateway` | `experiments/agent-live/toolsandbox-official-fak-20260626` | $env:TAU3_TASK_IDS='<space-separated official task ids>'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; tau2 run --domain retail --agent-llm shared-agent-model --user-llm shared-agent-model --num-trials 1 --task-ids ($env:TAU3_TASK_IDS -split ' ') |

## Gates

| Gate | OK | Detail |
|---|:---:|---|
| `candidate_task_ids` | yes | 1 candidate id from local smoke suite |
| `same_task_ids_required` | yes | raw and fak official runs must use the same benchmark-native task ids |
| `same_model_required` | yes | shared-agent-model |
| `same_user_simulator_required` | yes | shared-agent-model |
| `raw_arm_command` | yes | $env:TAU3_TASK_IDS='<space-separated official task ids>'; tau2 run --domain retail --agent-llm shared-agent-model --user-llm shared-agent-model --num-trials 1 --task-ids ($env:TAU3_TASK_IDS -split ' ') |
| `fak_arm_command` | yes | $env:TAU3_TASK_IDS='<space-separated official task ids>'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; tau2 run --domain retail --agent-llm shared-agent-model --user-llm shared-agent-model --num-trials 1 --task-ids ($env:TAU3_TASK_IDS -split ' ') |
| `official_harness_required` | yes | external tau3 or Apple ToolSandbox task/grader output is required before promotion |

## Required Before Any Result Claim

- benchmark-native task manifest or scenario ids for the selected tau3 or ToolSandbox subset
- raw-arm benchmark-native output over those exact task ids
- fak-arm benchmark-native output over those exact task ids
- benchmark-native grader or result summary for both arms
- proof that raw and fak arms used the same model, user simulator, task ids, budget, and retry policy
- fak verdict/evidence log linked to each mediated tool call
- raw/fak compare artifact reporting task success separately from policy compliance and benign utility preservation
