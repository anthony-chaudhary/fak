---
title: "Top 100 Autonomous Factory Tools Inventory & Migration Catalog"
description: "Authoritative inventory and ranking of the top 100 dev process tools migrating from Python to Go in fak-private"
---

# Top 100 Autonomous Factory Tools Inventory & Migration Catalog

In accordance with [`docs/dev-process-private-boundary.md`](dev-process-private-boundary.md) and issue #6220, this catalog tracks the top 100 autonomous factory tools migrating from legacy Python scripts (`tools/*.py`) to native Go platform packages under `fak-private/platform/`.

## 1. Domain Breakdown & Target Packages

| Domain | Count | Target Package in `fak-private` | Description |
|---|---|---|---|
| **Watchdogs & Reapers** | 20 | `platform/watchdogs` | Session heartbeats, stall detection, autoheal hooks, process resource limits, orphan sweeps |
| **Autonomous Dispatch** | 19 | `platform/dispatch` | Contract lease queues, ticket DAG routing, worker worktree sandboxing, admission gates |
| **Scorecards & Telemetry** | 41 | `platform/scorecards` | Scorecard registry, control panes, regression ratchets, fleet bottleneck metrics |
| **Cluster & Lab Control** | 18 | `tools/dgxbridge`, `platform/cluster` | GPU server cluster RPC, GCP GPU probes, remote serving witnesses, node benching |
| **Agent Memory & Context** | 7 | `platform/memsync` | Memory mirroring, recall audits, context co-travel, tape extraction |
| **Release & Promotion** | 15 | `platform/release` | Version bumping, release status, promotion gates, staging locks |
| **Total** | **120** | | Ranked cohort of autonomous factory tools |

---

## 2. Ranked Top 100 Migration Catalog

### Cohort 1: Watchdogs & Session Control (`platform/watchdogs`)
1. `tools/crash_audit.py` $\rightarrow$ `platform/watchdogs/crash_audit.go`
2. `tools/dos_supervisor_watchdog.py` $\rightarrow$ `platform/watchdogs/supervisor.go`
3. `tools/fleet_dos_dispatch_watchdog.py` $\rightarrow$ `platform/watchdogs/dispatch_watchdog.go`
4. `tools/fleet_resume_watchdog.py` $\rightarrow$ `platform/watchdogs/resume_watchdog.go`
5. `tools/fleet_session_signals.py` $\rightarrow$ `platform/watchdogs/signals.go`
6. `tools/fleet_sessions.py` $\rightarrow$ `platform/watchdogs/sessions.go`
7. `tools/fleet_supervisor_watchdog.py` $\rightarrow$ `platform/watchdogs/supervisor_watchdog.go`
8. `tools/gen_session_effectiveness_svg.py` $\rightarrow$ `platform/watchdogs/effectiveness.go`
9. `tools/guard_hop_bench.py` $\rightarrow$ `platform/watchdogs/guard_hop_bench.go`
10. `tools/guard_hop_rsi.py` $\rightarrow$ `platform/watchdogs/guard_hop_rsi.go`
11. `tools/peek_session.py` $\rightarrow$ `platform/watchdogs/peek.go`
12. `tools/proc_resource_guard.py` $\rightarrow$ `platform/watchdogs/resource_guard.go`
13. `tools/resume_resolver.py` $\rightarrow$ `platform/watchdogs/resume_resolver.go`
14. `tools/resume_sweep.py` $\rightarrow$ `platform/watchdogs/resume_sweep.go`
15. `tools/resume_watch.py` $\rightarrow$ `platform/watchdogs/resume_watch.go`
16. `tools/session0_orphan_sweep.py` $\rightarrow$ `platform/watchdogs/orphan_sweep.go`
17. `tools/session_checkpoint.py` $\rightarrow$ `platform/watchdogs/checkpoint.go`
18. `tools/stale_work_watchdog.py` $\rightarrow$ `platform/watchdogs/stale_work.go`
19. `tools/stopped_sessions.py` $\rightarrow$ `platform/watchdogs/stopped_sessions.go`
20. `tools/vcache_codex_session_extract.py` $\rightarrow$ `platform/watchdogs/vcache_extract.go`

### Cohort 2: Autonomous Dispatch & Worktrees (`platform/dispatch`)
21. `tools/dispatch_account_topup.py` $\rightarrow$ `platform/dispatch/topup.go`
22. `tools/dispatch_glm_docs.py` $\rightarrow$ `platform/dispatch/glm_docs.go`
23. `tools/dispatch_log_audit.py` $\rightarrow$ `platform/dispatch/log_audit.go`
24. `tools/dispatch_preflight.py` $\rightarrow$ `platform/dispatch/preflight.go`
25. `tools/dispatch_status.py` $\rightarrow$ `platform/dispatch/status.go`
26. `tools/dispatch_throughput.py` $\rightarrow$ `platform/dispatch/throughput.go`
27. `tools/dispatch_worker.py` $\rightarrow$ `platform/dispatch/worker.go`
28. `tools/issue_dispatch.py` $\rightarrow$ `platform/dispatch/issue_dispatch.go`
29. `tools/issue_gardener_worker.py` $\rightarrow$ `platform/dispatch/gardener.go`
30. `tools/issue_lane_router.py` $\rightarrow$ `platform/dispatch/lane_router.go`
31. `tools/issue_resolve_dispatch.py` $\rightarrow$ `platform/dispatch/resolve_dispatch.go`
32. `tools/issue_worker_prompt.py` $\rightarrow$ `platform/dispatch/prompt_renderer.go`
33. `tools/lane_core.py` $\rightarrow$ `platform/dispatch/lane_core.go`
34. `tools/lane_yield.py` $\rightarrow$ `platform/dispatch/lane_yield.go`
35. `tools/launch_admission.py` $\rightarrow$ `platform/dispatch/launch_admission.go`
36. `tools/learning_debt_dispatch.py` $\rightarrow$ `platform/dispatch/learning_debt.go`
37. `tools/tier_launch.py` $\rightarrow$ `platform/dispatch/tier_launch.go`
38. `tools/worker_worktree.py` $\rightarrow$ `platform/dispatch/worker_worktree.go`
39. `tools/worktree_doctor.py` $\rightarrow$ `platform/dispatch/worktree_doctor.go`

### Cohort 3: Scorecard Control Panes & Regression Ratchets (`platform/scorecards`)
40. `tools/behavior_contract_scorecard.py` $\rightarrow$ `platform/scorecards/behavior_contract.go`
41. `tools/bench_dx_scorecard.py` $\rightarrow$ `platform/scorecards/bench_dx.go`
42. `tools/bench_signal.py` $\rightarrow$ `platform/scorecards/bench_signal.go`
43. `tools/claim_repro_scorecard.py` $\rightarrow$ `platform/scorecards/claim_repro.go`
44. `tools/code_quality_scorecard.py` $\rightarrow$ `platform/scorecards/code_quality.go`
45. `tools/code_slop_scorecard.py` $\rightarrow$ `platform/scorecards/code_slop.go`
46. `tools/commit_quality_scorecard.py` $\rightarrow$ `platform/scorecards/commit_quality.go`
47. `tools/concept_disambiguation_scorecard.py` $\rightarrow$ `platform/scorecards/concept_disambiguation.go`
48. `tools/cuda_dev_scorecard.py` $\rightarrow$ `platform/scorecards/cuda_dev.go`
49. `tools/demo_quality_scorecard.py` $\rightarrow$ `platform/scorecards/demo_quality.go`
50. `tools/demo_robustness_scorecard.py` $\rightarrow$ `platform/scorecards/demo_robustness.go`
51. `tools/dispositions.py` $\rightarrow$ `platform/scorecards/dispositions.go`
52. `tools/doc_appeal_scorecard.py` $\rightarrow$ `platform/scorecards/doc_appeal.go`
53. `tools/docs_scorecard.py` $\rightarrow$ `platform/scorecards/docs.go`
54. `tools/fleet_accounts.py` $\rightarrow$ `platform/scorecards/fleet_accounts.go`
55. `tools/fleet_bottleneck.py` $\rightarrow$ `platform/scorecards/fleet_bottleneck.go`
56. `tools/fleet_control_pane.py` $\rightarrow$ `platform/scorecards/fleet_control_pane.go`
57. `tools/fleet_top.py` $\rightarrow$ `platform/scorecards/fleet_top.go`
58. `tools/fleet_trend.py` $\rightarrow$ `platform/scorecards/fleet_trend.go`
59. `tools/gate_signal.py` $\rightarrow$ `platform/scorecards/gate_signal.go`
60. `tools/industry_scorecard.py` $\rightarrow$ `platform/scorecards/industry.go`
61. `tools/intent_literal_scorecard.py` $\rightarrow$ `platform/scorecards/intent_literal.go`
62. `tools/learning_scorecard.py` $\rightarrow$ `platform/scorecards/learning.go`
63. `tools/observability_scorecard.py` $\rightarrow$ `platform/scorecards/observability.go`
64. `tools/persona_fit_scorecard.py` $\rightarrow$ `platform/scorecards/persona_fit.go`
65. `tools/persona_readiness_scorecard.py` $\rightarrow$ `platform/scorecards/persona_readiness.go`
66. `tools/popularization_readiness_scorecard.py` $\rightarrow$ `platform/scorecards/popularization.go`
67. `tools/product_scorecard.py` $\rightarrow$ `platform/scorecards/product.go`
68. `tools/release_readiness_scorecard.py` $\rightarrow$ `platform/scorecards/release_readiness.go`
69. `tools/repo_hygiene_scorecard.py` $\rightarrow$ `platform/scorecards/repo_hygiene.go`
70. `tools/rsi_maturity_scorecard.py` $\rightarrow$ `platform/scorecards/rsi_maturity.go`
71. `tools/score_signal.py` $\rightarrow$ `platform/scorecards/score_signal.go`
72. `tools/scorecard_control_pane.py` $\rightarrow$ `platform/scorecards/control_pane.go`
73. `tools/scorecard_since.py` $\rightarrow$ `platform/scorecards/since.go`
74. `tools/skill_slop_scorecard.py` $\rightarrow$ `platform/scorecards/skill_slop.go`
75. `tools/sota_coverage_scorecard.py` $\rightarrow$ `platform/scorecards/sota_coverage.go`
76. `tools/stability_scorecard.py` $\rightarrow$ `platform/scorecards/stability.go`
77. `tools/steerability_scorecard.py` $\rightarrow$ `platform/scorecards/steerability.go`
78. `tools/tooling_quality_scorecard.py` $\rightarrow$ `platform/scorecards/tooling_quality.go`
79. `tools/trajctl_signal.py` $\rightarrow$ `platform/scorecards/trajctl_signal.go`
80. `tools/vcache_scorecard_gate.py` $\rightarrow$ `platform/scorecards/vcache_gate.go`

### Cohort 4: Cluster & Lab Hardware Nodes (`tools/dgxbridge`, `platform/cluster`)
81. `tools/dgx_swebench_compare.py` $\rightarrow$ `platform/cluster/swebench_compare.go`
82. `tools/gcp_accel.py` $\rightarrow$ `platform/cluster/gcp_accel.go`
83. `tools/gcp_bench.py` $\rightarrow$ `platform/cluster/gcp_bench.go`
84. `tools/gcp_gpu_probe.py` $\rightarrow$ `platform/cluster/gcp_gpu_probe.go`
85. `tools/gcp_quota_request.py` $\rightarrow$ `platform/cluster/gcp_quota.go`
86. `tools/glm52_serve_preflight.py` $\rightarrow$ `platform/cluster/glm52_preflight.go`
87. `tools/glm52_serving_witness.py` $\rightarrow$ `platform/cluster/glm52_witness.go`
88. `tools/glm52_vllm_agentic_battery.py` $\rightarrow$ `platform/cluster/glm52_vllm.go`
89. `tools/glm_throughput_record.py` $\rightarrow$ `platform/cluster/glm_throughput.go`
90. `tools/glm_witness_record.py` $\rightarrow$ `platform/cluster/glm_witness.go`
91. `tools/qwen36_node_packet.py` $\rightarrow$ `platform/cluster/node_packet.go`
92. `tools/qwen36_node_reports.py` $\rightarrow$ `platform/cluster/node_reports.go`
93. `tools/qwen36_node_server.py` $\rightarrow$ `platform/cluster/node_server.go`
94. `tools/qwen36_perf_gate.py` $\rightarrow$ `platform/cluster/perf_gate.go`
95. `tools/qwen36_standalone_readiness.py` $\rightarrow$ `platform/cluster/readiness.go`
96. `tools/qwen36_surface_smoke.py` $\rightarrow$ `platform/cluster/surface_smoke.go`
97. `tools/qwen36_watch_nodes.py` $\rightarrow$ `platform/cluster/watch_nodes.go`
98. `tools/receive_node_bench.py` $\rightarrow$ `platform/cluster/receive_bench.go`

### Cohort 5: Agent Memory & Context Flow (`platform/memsync`)
99. `tools/context_tape.py` $\rightarrow$ `platform/memsync/context_tape.go`
100. `tools/ctxcost.py` $\rightarrow$ `platform/memsync/ctxcost.go`
