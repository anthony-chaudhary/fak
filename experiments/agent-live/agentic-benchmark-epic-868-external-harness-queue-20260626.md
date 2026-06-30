# Agentic Benchmark Epic #868 External Harness Queue

- Generated: `2026-06-26T03:43:07Z`
- Status: `PENDING_EXTERNAL_HARNESS`
- Result claim allowed: `false`
- Items: `6`
- Ready items: `5`
- Blocked items: `1`
- Required returned artifacts: `36`
- Boundary: External harness queue only: folds pending child contracts into runnable work items and refuses a #868 result claim until benchmark-native raw/fak grader artifacts are checked in.

## Queue

| Issue | Packet | Source | External state | Required returns |
|---:|---|---|---|---:|
| #870 | `B` | `experiments/vllm/glm52-agentic-battery/final-check.json` | `BLOCKED_ON_H200_VLLM_READY` | 5 |
| #871 | `C` | `experiments/agent-live/swebench-opus-smoke-contract-20260626.json` | `READY_FOR_EXTERNAL_HARNESS` | 4 |
| #872 | `D` | `experiments/agent-live/deepswe-raw-fak-contract-20260626.json` | `READY_FOR_EXTERNAL_HARNESS` | 5 |
| #873 | `E` | `experiments/agent-live/toolsandbox-official-run-contract-20260626.json` | `READY_FOR_EXTERNAL_HARNESS` | 7 |
| #874 | `F` | `experiments/agent-live/terminalbench-official-run-contract-20260626.json` | `READY_FOR_EXTERNAL_HARNESS` | 7 |
| #875 | `G` | `experiments/agent-live/browseraction-official-run-contract-20260626.json` | `READY_FOR_EXTERNAL_HARNESS` | 8 |

## #870 Packet B - GLM-5.2/vLLM agentic battery

- Source: `experiments/vllm/glm52-agentic-battery/final-check.json`
- Status: `PENDING_MEASUREMENT`
- External state: `BLOCKED_ON_H200_VLLM_READY`
- Boundary: No GLM-5.2/vLLM benchmark number is quotable until every required artifact passes and any copied number is linked from BENCHMARK-AUTHORITY.md.

### Commands

#### preflight

Fail-closed GLM-5.2/vLLM node readiness gate.

```bash
python tools/glm52_serve_preflight.py --engine vllm --quant fp8 --require-ready --out experiments/vllm/glm52-vllm-preflight.json --markdown experiments/vllm/glm52-vllm-preflight.md
```

Artifacts: `experiments/vllm/glm52-vllm-preflight.json`, `experiments/vllm/glm52-vllm-preflight.md`

Current artifact status: `FAIL` - node_verdict='BLOCKED_ARCH'; vllm.ready=False; expected GLM-5.2 vLLM READY/PENDING preflight

#### serve_raw_vllm

Start raw vLLM for GLM-5.2 before the live witnesses.

```bash
: "${GLM52_TOOL_CALL_PARSER:?set GLM52_TOOL_CALL_PARSER to the vLLM parser name}" && ENGINE=vllm SERVED_NAME=glm-5.2 PORT=8000 ENGINE_ARGS="--enable-auto-tool-choice --tool-call-parser ${GLM52_TOOL_CALL_PARSER}" bash tools/glm52_sglang_vllm_serve.sh
```

Current artifact status: `MANUAL` - manual serving step; witnessed by downstream artifacts

#### serving_witness

Direct + fak-gateway + quarantine serving witness.

```bash
python tools/glm52_serving_witness.py --base-url http://127.0.0.1:8000/v1 --model glm-5.2 --engine-cache-engine vllm --context-length 131072 --out experiments/glm52/full-size-serving-witness.json --markdown experiments/glm52/full-size-serving-witness.md
```

Artifacts: `experiments/glm52/full-size-serving-witness.json`, `experiments/glm52/full-size-serving-witness.md`

Current artifact status: `MISSING` - missing

#### vllm_tax

Measure fak gateway adjudication tax over raw vLLM.

```bash
python tools/vllm_tax_witness.py --base-url http://127.0.0.1:8000/v1 --model glm-5.2 --count 8 --record --out experiments/vllm/adjudication-tax-witness.json --markdown experiments/vllm/adjudication-tax-witness.md
```

Artifacts: `experiments/vllm/adjudication-tax-witness.json`, `experiments/vllm/adjudication-tax-witness.md`

Current artifact status: `MISSING` - missing

#### swebench_compare_preflight

Fail fast before the long SWE-bench agent run.

```bash
<private-swebench-compare> --engine vllm --model zai-org/GLM-5.2-FP8 --served-model-name glm-5.2 --raw-base-url http://127.0.0.1:8000/v1 --verified-count 20 --skip-engine-serve --require-tool-calls --require-grade --run-dir <private-swebench-run> --require-gpu-name H200 --preflight-only
```

Artifacts: `<private-swebench-run>/COMPARE-PREFLIGHT.json`, `<private-swebench-run>/DONE.rc`

Current artifact status: `MISSING` - missing

#### swebench_verified_20

Run raw-vLLM vs fak-gateway on a 20-task SWE-bench Verified slice.

```bash
<private-swebench-compare> --engine vllm --model zai-org/GLM-5.2-FP8 --served-model-name glm-5.2 --raw-base-url http://127.0.0.1:8000/v1 --verified-count 20 --skip-engine-serve --require-tool-calls --require-grade --run-dir <private-swebench-run> --require-gpu-name H200
```

Artifacts: `<private-swebench-run>/COMPARE-PREFLIGHT.json`, `<private-swebench-run>/compare.json`, `<private-swebench-run>/COMPARE.md`, `<private-swebench-run>/DONE.rc`

Current artifact status: `MISSING` - missing


### Required Before Claim

- preflight must pass on the target GLM-5.2/vLLM node
- serving_witness artifact must be generated and pass the final-check witness
- vllm_tax artifact must be generated and pass the final-check witness
- swebench_compare_preflight artifact must be generated and pass the final-check witness
- swebench_verified_20 artifact must be generated and pass the final-check witness

### Compare Metrics

- `swebench_resolve_rate`
- `safe_completion`
- `latency`
- `token_or_cost_proxy`
- `policy_blocks`
- `evidence_completeness`

## #871 Packet C - Opus-class SWE-bench smoke

- Source: `experiments/agent-live/swebench-opus-smoke-contract-20260626.json`
- Status: `READY_FOR_REMOTE_GRADING`
- External state: `READY_FOR_EXTERNAL_HARNESS`
- Boundary: Pre-run contract only: fixes task ids, model identity, arm commands, and official-grader commands for an Opus-class raw-vs-fak SWE-bench smoke. It is not a solve-rate result until both predictions files are produced and graded by the official SWE-bench harness.

### Arms

#### raw-opus

- Harness: `benchmark-native-or-raw-agent-scaffold`
- Output: `experiments/agent-live/swebench-opus-raw-smoke-20260626`

```powershell
$env:MSWEA_COST_TRACKING='ignore_errors'; mini-extra swebench --subset verified --split test -w 1 -o experiments/agent-live/swebench-opus-raw-smoke-20260626 -m anthropic/claude-opus-4-8 -c swebench.yaml --filter "django__django-12345|django__django-23456|scikit-learn__scikit-learn-56789|sympy__sympy-34567|sympy__sympy-45678"; Copy-Item -LiteralPath experiments/agent-live/swebench-opus-raw-smoke-20260626/preds.json -Destination experiments/agent-live/swebench-opus-raw-smoke-20260626/predictions.json
```

#### fak-opus

- Harness: `fak-gateway-fleet-runner`
- Output: `experiments/agent-live/swebench-opus-fak-smoke-20260626`

```powershell
go run ./cmd/fak swebench run --agent fleet --filter smoke --difficulty testdata/swebench_smoke.json --gateway localhost:8080 --model claude-opus-4-8 --preds-only --output experiments/agent-live/swebench-opus-fak-smoke-20260626
```


### Required Before Claim

- raw-opus predictions.json generated by the raw scaffold over the selected task ids
- fak-opus predictions.json generated by the fak fleet runner over the same selected task ids
- official SWE-bench harness report.json for both arms
- raw/fak compare artifact that folds solve rate, safe completion, cost or token budget, latency, policy blocks, and evidence completeness

### Compare Metrics

- `solve_rate`
- `safe_completion`
- `cost_or_token_budget`
- `latency`
- `tool_call_count`
- `policy_blocks`
- `evidence_completeness`

## #872 Packet D - DeepSWE/R2E-Gym runner

- Source: `experiments/agent-live/deepswe-raw-fak-contract-20260626.json`
- Status: `READY_FOR_EXTERNAL_RUN`
- External state: `READY_FOR_EXTERNAL_HARNESS`
- Boundary: Pre-run contract only: fixes task ids, DeepSWE/R2E-Gym adapter, model id, raw/fak endpoint routing, budget, and official-grader commands. It is not a solve-rate result until both arms produce predictions and the official SWE-bench harness grades them.

### Arms

#### raw-deepswe

- Harness: `deepswe-r2e-gym-raw`
- Output: `experiments/agent-live/deepswe-raw-smoke-20260626`

```powershell
$env:FAK_DEEPSWE_RUNNER='deepswe-r2e-runner'; $env:FAK_DEEPSWE_RUNNER_ARGS=''; $env:FAK_DEEPSWE_BASE_URL=$env:RAW_DEEPSWE_BASE_URL; $env:FAK_DEEPSWE_MODEL='DeepSWE-Preview'; go run ./cmd/fak swebench run --agent deepswe --filter full --difficulty testdata/swebench_smoke.json --limit 2 --max-steps 50 --timeout 30m --model DeepSWE-Preview --preds-only --output experiments/agent-live/deepswe-raw-smoke-20260626
```

#### fak-deepswe

- Harness: `deepswe-r2e-gym-through-fak-gateway`
- Output: `experiments/agent-live/deepswe-fak-smoke-20260626`

```powershell
$env:FAK_DEEPSWE_RUNNER='deepswe-r2e-runner'; $env:FAK_DEEPSWE_RUNNER_ARGS=''; $env:FAK_DEEPSWE_BASE_URL='http://localhost:8080/v1'; $env:FAK_DEEPSWE_MODEL='DeepSWE-Preview'; go run ./cmd/fak swebench run --agent deepswe --filter full --difficulty testdata/swebench_smoke.json --limit 2 --max-steps 50 --timeout 30m --model DeepSWE-Preview --preds-only --output experiments/agent-live/deepswe-fak-smoke-20260626
```


### Required Before Claim

- raw-deepswe predictions.json generated by a real DeepSWE/R2E-Gym adapter over the selected task ids
- fak-deepswe predictions.json generated by the same adapter over the same task ids through the fak gateway
- official SWE-bench harness report.json for both arms
- adapter logs or metadata showing the same model id, max steps, timeout, and task id list for both arms
- raw/fak compare artifact that folds solve rate, safe completion, cost or token budget, latency, adapter failures, policy blocks, and evidence completeness

### Compare Metrics

- `solve_rate`
- `safe_completion`
- `cost_or_token_budget`
- `latency`
- `tool_call_count`
- `adapter_failures`
- `policy_blocks`
- `evidence_completeness`

## #873 Packet E - ToolSandbox/tau3 policy-state

- Source: `experiments/agent-live/toolsandbox-official-run-contract-20260626.json`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- External state: `READY_FOR_EXTERNAL_HARNESS`
- Boundary: External-run contract only: fixes the raw/fak command shape, shared model and simulator requirements, evidence paths, and promotion gates for a benchmark-native tau3 or ToolSandbox run. It is not an official result until external task definitions, raw/fak outputs, and native grader summaries are checked in.

### Arms

#### raw-toolsandbox

- Harness: `benchmark-native`
- Output: `experiments/agent-live/toolsandbox-official-raw-20260626`
- Required artifacts: `benchmark-native raw result summary`, `raw trajectory or simulation log`

```powershell
$env:TAU3_TASK_IDS='<space-separated official task ids>'; tau2 run --domain retail --agent-llm shared-agent-model --user-llm shared-agent-model --num-trials 1 --task-ids ($env:TAU3_TASK_IDS -split ' ')
```

#### fak-toolsandbox

- Harness: `benchmark-native-through-fak-gateway`
- Output: `experiments/agent-live/toolsandbox-official-fak-20260626`
- Required artifacts: `benchmark-native fak result summary`, `fak trajectory or simulation log`, `fak verdict/evidence log linked to mediated tool calls`

```powershell
$env:TAU3_TASK_IDS='<space-separated official task ids>'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; tau2 run --domain retail --agent-llm shared-agent-model --user-llm shared-agent-model --num-trials 1 --task-ids ($env:TAU3_TASK_IDS -split ' ')
```


### Required Before Claim

- benchmark-native task manifest or scenario ids for the selected tau3 or ToolSandbox subset
- raw-arm benchmark-native output over those exact task ids
- fak-arm benchmark-native output over those exact task ids
- benchmark-native grader or result summary for both arms
- proof that raw and fak arms used the same model, user simulator, task ids, budget, and retry policy
- fak verdict/evidence log linked to each mediated tool call
- raw/fak compare artifact reporting task success separately from policy compliance and benign utility preservation

### Compare Metrics

- `benchmark_native_success_or_pass_k`
- `safe_completion`
- `policy_breach_rate`
- `minefield_or_mutating_action_avoidance`
- `tool_call_count`
- `latency`
- `cost_or_token_budget`
- `fak_denied_calls`
- `evidence_completeness`

## #874 Packet F - Terminal-Bench command boundary

- Source: `experiments/agent-live/terminalbench-official-run-contract-20260626.json`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- External state: `READY_FOR_EXTERNAL_HARNESS`
- Boundary: External-run contract only: fixes the raw/fak Terminal-Bench command shape, shared task/model/image/budget requirements, evidence paths, and promotion gates. It is not an official result until benchmark-native tb run task logs, test output, and a raw-vs-fak compare artifact are checked in.

### Arms

#### raw-terminalbench

- Harness: `benchmark-native`
- Output: `experiments/agent-live/terminalbench-official-raw-20260626`
- Required artifacts: `tb run directory for each selected task`, `benchmark-native command log`, `benchmark-native test output or result summary`

```powershell
$env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset terminal-bench-core==0.1.1 --agent terminus --model shared-agent-model --task-id $task --n-concurrent 1 }
```

#### fak-terminalbench

- Harness: `benchmark-native-through-fak-gateway`
- Output: `experiments/agent-live/terminalbench-official-fak-20260626`
- Required artifacts: `tb run directory for each selected task`, `benchmark-native command log`, `benchmark-native test output or result summary`, `fak verdict/evidence log linked to mediated terminal commands`

```powershell
$env:TERMINAL_BENCH_TASK_IDS='<space-separated official task ids>'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; foreach ($task in ($env:TERMINAL_BENCH_TASK_IDS -split ' ')) { tb run --dataset terminal-bench-core==0.1.1 --agent terminus-through-fak --model shared-agent-model --task-id $task --n-concurrent 1 }
```


### Required Before Claim

- benchmark-native Terminal-Bench task ids for the selected fixed subset
- Terminal-Bench image or environment setup manifest for each selected task
- raw-arm tb run directory with command log and official test output over those exact task ids
- fak-arm tb run directory with command log and official test output over those exact task ids
- proof that raw and fak arms used the same model, task ids, image or environment, budget, concurrency, and retry policy
- fak per-command verdict/evidence log linked to the corresponding tb run command log and test output
- raw/fak compare artifact reporting benchmark-native solve separately from safe resolve, blocked dangerous actions, unnecessary blocks, runtime, cost or token budget, and evidence completeness

### Compare Metrics

- `benchmark_native_test_success_or_pass_1`
- `safe_resolve`
- `blocked_dangerous_actions`
- `unnecessary_blocks`
- `command_count`
- `runtime`
- `cost_or_token_budget`
- `fak_denied_commands`
- `fak_verdict_evidence_completeness`

## #875 Packet G - Browser/computer-use action mediation

- Source: `experiments/agent-live/browseraction-official-run-contract-20260626.json`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- External state: `READY_FOR_EXTERNAL_HARNESS`
- Boundary: External-run contract only: fixes the raw/fak BrowserGym/WebArena-style command shape, shared model/task/browser-state/budget requirements, evidence paths, and promotion gates. It is not an official browser or computer-use benchmark result until benchmark-native task traces, score reports, and a raw-vs-fak compare artifact are checked in.

### Arms

#### raw-browseraction

- Harness: `benchmark-native`
- Output: `experiments/agent-live/browseraction-official-raw-20260626`
- Required artifacts: `benchmark-native task trace or study directory`, `benchmark-native score report`, `browser state reset or environment manifest`

```powershell
$env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'; $env:AGENTLAB_EXP_ROOT='experiments/agent-live/browseraction-official-raw-20260626'; $env:FAK_BROWSERGYM_AGENT='browsergym_agent_config'; $env:FAK_BROWSERGYM_MODEL='shared-agent-model'; $env:FAK_BROWSERGYM_MAX_STEPS='30'; python -c 'from agentlab.experiments.study import make_study; from importlib import import_module; import os; agent_args=getattr(import_module(os.environ[''FAK_BROWSERGYM_AGENT'']), ''AGENT_ARGS''); study=make_study(benchmark=''webarena'', agent_args=[agent_args], comment=os.environ[''FAK_BROWSERGYM_MODEL'']); study.run(n_jobs=1)'
```

#### fak-browseraction

- Harness: `benchmark-native-through-fak-gateway`
- Output: `experiments/agent-live/browseraction-official-fak-20260626`
- Required artifacts: `benchmark-native task trace or study directory`, `benchmark-native score report`, `browser state reset or environment manifest`, `fak action verdict/evidence log linked to mediated browser actions`

```powershell
$env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'; $env:AGENTLAB_EXP_ROOT='experiments/agent-live/browseraction-official-fak-20260626'; $env:FAK_BROWSERGYM_AGENT='browsergym_agent_config_through_fak'; $env:FAK_BROWSERGYM_MODEL='shared-agent-model'; $env:FAK_BROWSERGYM_MAX_STEPS='30'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; python -c 'from agentlab.experiments.study import make_study; from importlib import import_module; import os; agent_args=getattr(import_module(os.environ[''FAK_BROWSERGYM_AGENT'']), ''AGENT_ARGS''); study=make_study(benchmark=''webarena'', agent_args=[agent_args], comment=os.environ[''FAK_BROWSERGYM_MODEL'']); study.run(n_jobs=1)'
```


### Required Before Claim

- benchmark-native BrowserGym/WebArena, WorkArena, OSWorld, or BrowseComp task ids for the selected fixed subset
- browser state, website snapshot, credentials, reset policy, and environment manifest for each selected task
- raw-arm benchmark-native trace or study directory with score report over those exact task ids
- fak-arm benchmark-native trace or study directory with score report over those exact task ids
- proof that raw and fak arms used the same model, task ids, browser state, budget, max steps, timeout, and retry policy
- fak action verdict/evidence log linked to the corresponding benchmark-native action trace and score report
- writeup separating model perception or grounding failures from harness/tool-boundary failures
- raw/fak compare artifact reporting task success separately from safe success, action count, invalid actions, policy blocks, runtime, cost or token budget, and evidence completeness

### Compare Metrics

- `benchmark_native_task_success_or_score`
- `safe_success`
- `action_count`
- `invalid_action_rate`
- `policy_blocks`
- `minefield_or_mutating_action_avoidance`
- `runtime`
- `cost_or_token_budget`
- `fak_action_evidence_completeness`
