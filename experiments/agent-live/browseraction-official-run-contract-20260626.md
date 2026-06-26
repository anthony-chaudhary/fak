# Browser Action Official-Run Contract

- Generated: `2026-06-26T03:24:10Z`
- Benchmark: `Browser/computer-use action official-run contract`
- Status: `READY_FOR_EXTERNAL_HARNESS`
- Evidence class: `EXTERNAL_RUN_CONTRACT`
- Result claim allowed: `false`
- Local fixture artifact: `experiments/agent-live/browser-action-mediation-smoke-20260625.json`
- Boundary: External-run contract only: fixes the raw/fak BrowserGym/WebArena-style command shape, shared model/task/browser-state/budget requirements, evidence paths, and promotion gates. It is not an official browser or computer-use benchmark result until benchmark-native task traces, score reports, and a raw-vs-fak compare artifact are checked in.

## Task Selection

- Candidate suite: `testdata/webbench/action_mediation_smoke.json`
- Candidate task ids: `knowledgebase-search-benign, shopping-address-delete-minefield`
- Official harness: `BrowserGym/AgentLab`
- Official benchmark: `webarena`
- Max steps: `30`

| Candidate | Benchmark | Domain | Budget turns | Source |
|---|---|---|---:|---|
| `knowledgebase-search-benign` | `browser-agent` | `help.example` | 3 | `https://help.example` |
| `shopping-address-delete-minefield` | `browser-agent` | `shop.example` | 4 | `https://shop.example/account` |

## Arms

| Arm | Harness | Output | Command |
|---|---|---|---|
| `raw-browseraction` | `benchmark-native` | `experiments/agent-live/browseraction-official-raw-20260626` | $env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'; $env:AGENTLAB_EXP_ROOT='experiments/agent-live/browseraction-official-raw-20260626'; $env:FAK_BROWSERGYM_AGENT='browsergym_agent_config'; $env:FAK_BROWSERGYM_MODEL='shared-agent-model'; $env:FAK_BROWSERGYM_MAX_STEPS='30'; python -c 'from agentlab.experiments.study import make_study; from importlib import import_module; import os; agent_args=getattr(import_module(os.environ[''FAK_BROWSERGYM_AGENT'']), ''AGENT_ARGS''); study=make_study(benchmark=''webarena'', agent_args=[agent_args], comment=os.environ[''FAK_BROWSERGYM_MODEL'']); study.run(n_jobs=1)' |
| `fak-browseraction` | `benchmark-native-through-fak-gateway` | `experiments/agent-live/browseraction-official-fak-20260626` | $env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'; $env:AGENTLAB_EXP_ROOT='experiments/agent-live/browseraction-official-fak-20260626'; $env:FAK_BROWSERGYM_AGENT='browsergym_agent_config_through_fak'; $env:FAK_BROWSERGYM_MODEL='shared-agent-model'; $env:FAK_BROWSERGYM_MAX_STEPS='30'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; python -c 'from agentlab.experiments.study import make_study; from importlib import import_module; import os; agent_args=getattr(import_module(os.environ[''FAK_BROWSERGYM_AGENT'']), ''AGENT_ARGS''); study=make_study(benchmark=''webarena'', agent_args=[agent_args], comment=os.environ[''FAK_BROWSERGYM_MODEL'']); study.run(n_jobs=1)' |

## Gates

| Gate | OK | Detail |
|---|:---:|---|
| `candidate_task_ids` | yes | 2 candidate ids from local browser-action smoke suite |
| `official_harness_pin` | yes | BrowserGym/AgentLab / webarena |
| `same_task_ids_required` | yes | raw and fak official runs must use the same benchmark-native BrowserGym task ids |
| `same_browser_state_required` | yes | raw and fak official runs must use the same browser profile, website snapshot, credentials, and reset policy |
| `same_budget_required` | yes | raw and fak official runs must use the same max steps, retry policy, and timeout |
| `same_model_required` | yes | shared-agent-model |
| `raw_arm_command` | yes | $env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'; $env:AGENTLAB_EXP_ROOT='experiments/agent-live/browseraction-official-raw-20260626'; $env:FAK_BROWSERGYM_AGENT='browsergym_agent_config'; $env:FAK_BROWSERGYM_MODEL='shared-agent-model'; $env:FAK_BROWSERGYM_MAX_STEPS='30'; python -c 'from agentlab.experiments.study import make_study; from importlib import import_module; import os; agent_args=getattr(import_module(os.environ[''FAK_BROWSERGYM_AGENT'']), ''AGENT_ARGS''); study=make_study(benchmark=''webarena'', agent_args=[agent_args], comment=os.environ[''FAK_BROWSERGYM_MODEL'']); study.run(n_jobs=1)' |
| `fak_arm_command` | yes | $env:BROWSERGYM_TASK_IDS='<space-separated BrowserGym env ids>'; $env:AGENTLAB_EXP_ROOT='experiments/agent-live/browseraction-official-fak-20260626'; $env:FAK_BROWSERGYM_AGENT='browsergym_agent_config_through_fak'; $env:FAK_BROWSERGYM_MODEL='shared-agent-model'; $env:FAK_BROWSERGYM_MAX_STEPS='30'; $env:OPENAI_BASE_URL='http://localhost:8080/v1'; $env:OPENAI_API_BASE='http://localhost:8080/v1'; python -c 'from agentlab.experiments.study import make_study; from importlib import import_module; import os; agent_args=getattr(import_module(os.environ[''FAK_BROWSERGYM_AGENT'']), ''AGENT_ARGS''); study=make_study(benchmark=''webarena'', agent_args=[agent_args], comment=os.environ[''FAK_BROWSERGYM_MODEL'']); study.run(n_jobs=1)' |
| `official_harness_required` | yes | external BrowserGym or AgentLab run output with benchmark-native score and trace logs is required before promotion |

## Required Before Any Result Claim

- benchmark-native BrowserGym/WebArena, WorkArena, OSWorld, or BrowseComp task ids for the selected fixed subset
- browser state, website snapshot, credentials, reset policy, and environment manifest for each selected task
- raw-arm benchmark-native trace or study directory with score report over those exact task ids
- fak-arm benchmark-native trace or study directory with score report over those exact task ids
- proof that raw and fak arms used the same model, task ids, browser state, budget, max steps, timeout, and retry policy
- fak action verdict/evidence log linked to the corresponding benchmark-native action trace and score report
- writeup separating model perception or grounding failures from harness/tool-boundary failures
- raw/fak compare artifact reporting task success separately from safe success, action count, invalid actions, policy blocks, runtime, cost or token budget, and evidence completeness
