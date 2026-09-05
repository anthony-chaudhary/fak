---
parent_goal: goals/GOAL-agent-centric-mlx-mac-observability.md
sub_step: 4_verification_and_stress_tests
witness: "go test -v -count=1 ./internal/macobs/... && go test -v -count=1 ./cmd/fak -run TestRunMacObs"
target_files:
  - internal/macobs/macobs_test.go
  - internal/macobs/analyzer_test.go
  - internal/macobs/mlx_metrics_test.go
  - internal/macobs/headroom_test.go
  - internal/macobs/hardware_darwin_test.go
  - cmd/fak/macobs_test.go
---
# Sub-Goal Objective
Expand and harden test coverage across `internal/macobs` and `cmd/fak`:
1. Verify 100% boundary testing on:
   - High memory pressure and swap thrashing scenarios
   - Thermal throttling conditions (Serious/Critical and CPU speed downclocking)
   - Heavy queue contention vs prefix cache saturation
   - Radical edge cases (zero memory, single token context, massive concurrency requests)
   - Malformed XML and sysctl output robustness (ensuring zero panic, graceful degradation)
2. Add end-to-end multi-agent scenario test in `internal/macobs`:
   - Simulate a coordinator planning a wave of 8 subagents with 4096-token shared preamble and 2048-token private turns
   - Assert exact calculation of `ConcurrencyAdvantage`, `AvailableKVMBytes`, `Verdict`, and `Remediation`
3. Verify live Darwin execution on this Apple Silicon host and ensure clean fallback.
