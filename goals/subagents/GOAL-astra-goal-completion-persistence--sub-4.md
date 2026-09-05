---
parent_goal: goals/GOAL-astra-goal-completion-persistence.md
sub_step: 4
witness: "go test -v ./internal/orchestration/... ./cmd/fak/... -run 'TestSelectSOLRoute|TestOrchestrationLaunch|TestGuardOperatorDirected'"
target_files:
  - internal/orchestration/solroute.go
  - internal/orchestration/solroute_test.go
  - cmd/fak/guard_operator_directed.go
  - cmd/fak/guard_operator_directed_test.go
---
# Sub-Goal Objective
Enhance Astra manager subagent delegation and goal persistence across cmd/fak and internal/orchestration by adding subagent signals to parallel routing and tailoring premature surrender continuation messages.

# Scope Fence
- Work exclusively in: internal/orchestration/solroute.go, internal/orchestration/solroute_test.go, cmd/fak/guard_operator_directed.go, cmd/fak/guard_operator_directed_test.go
- Prohibited: Do not touch root configs, go.mod, go.sum, dos.toml, or sibling packages.

# Witness Command
go test -v ./internal/orchestration/... -run TestSelectSOLRoute
