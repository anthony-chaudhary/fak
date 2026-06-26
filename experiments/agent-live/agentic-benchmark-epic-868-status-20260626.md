# Agentic Benchmark Epic #868 Rollup

- Generated: `2026-06-26T03:43:07Z`
- Status: `PENDING_EXTERNAL_HARNESS`
- Result claim allowed: `false`
- Children parsed: `8/8`
- Result-claim artifacts: `0`
- Result packets: `0` passed / `0` total
- Boundary: Parent rollup only: reads committed child artifacts and refuses a #868 result claim until benchmark-native raw/fak grader evidence exists for at least one live lane.

## Children

| Issue | Packet | Artifact | Gate | Status | Detail |
|---:|---|---|---|---|---|
| #869 | `A` | `experiments/agent-live/agentdojo-fak-fullstack-20260625.json` | `PASS_LOCAL` | `PASS` | local structural floor PASS; full-stack ASR successes 0; benign controls 2 |
| #870 | `B` | `experiments/vllm/glm52-agentic-battery/final-check.json` | `PENDING_EXTERNAL_HARNESS` | `PENDING_MEASUREMENT` | GLM final-check PENDING_MEASUREMENT; required artifacts 6/11 passed |
| #871 | `C` | `experiments/agent-live/swebench-opus-smoke-contract-20260626.json` | `PENDING_EXTERNAL_HARNESS` | `READY_FOR_REMOTE_GRADING` | READY_FOR_REMOTE_GRADING; raw/fak predictions and official SWE-bench reports still required |
| #872 | `D` | `experiments/agent-live/deepswe-raw-fak-contract-20260626.json` | `PENDING_EXTERNAL_HARNESS` | `READY_FOR_EXTERNAL_RUN` | READY_FOR_EXTERNAL_RUN; raw/fak DeepSWE predictions and official SWE-bench reports still required |
| #873 | `E` | `experiments/agent-live/toolsandbox-official-run-contract-20260626.json` | `PENDING_EXTERNAL_HARNESS` | `READY_FOR_EXTERNAL_HARNESS` | READY_FOR_EXTERNAL_HARNESS; benchmark-native tau3/ToolSandbox raw/fak outputs and grader summaries still required |
| #874 | `F` | `experiments/agent-live/terminalbench-official-run-contract-20260626.json` | `PENDING_EXTERNAL_HARNESS` | `READY_FOR_EXTERNAL_HARNESS` | READY_FOR_EXTERNAL_HARNESS; benchmark-native Terminal-Bench raw/fak run dirs, command logs, and official test summaries still required |
| #875 | `G` | `experiments/agent-live/browseraction-official-run-contract-20260626.json` | `PENDING_EXTERNAL_HARNESS` | `READY_FOR_EXTERNAL_HARNESS` | READY_FOR_EXTERNAL_HARNESS; benchmark-native browser/computer-use raw/fak traces, score reports, and linked fak action evidence still required |
| #876 | `authority` | `BENCHMARK-AUTHORITY.md` | `PASS_LOCAL` | `AUTHORITY_SHAPE_PRESENT` | authority rows and promotion gate text are present |

## Result Packet Intake

- Directory: `experiments/agent-live/agentic-benchmark-result-packets/*.json`
- Schema: `fak.agentic-benchmark-result-packet.v1`
- Required gates: `benchmark_native`, `same_task_ids`, `same_model`, `same_budget`, `official_grader.available`, raw/fak arms, checked-in artifacts, and metric categories `task_success`, `safe_success`, `cost_or_token_budget`, `latency`, `policy_events`, `evidence_completeness`.

## Acceptance Gates

| Gate | OK | Detail |
|---|---:|---|
| `child_artifacts_parse` | true | 8/8 child artifacts parsed; failed=none |
| `authority_entry_shape` | true | BENCHMARK-AUTHORITY carries local rows plus the promotion-gate shape |
| `result_packet_graduated` | false | 0 result-claim-enabled artifact(s); result packets passed=0/0 failed=0; pending live children=#870, #871, #872, #873, #874, #875 |
| `external_harness_grading` | false | requires benchmark-native raw/fak grader output for open live lanes |
| `compare_metrics_complete` | false | requires solve/safe/cost-or-token/latency/policy/evidence metrics from a result-bearing compare artifact |
| `final_writeup_ready` | false | final #868 writeup waits for at least one real raw-vs-fak result packet |
