---
parent_goal: goals/GOAL-astra-goal-completion-persistence.md
sub_step: 2
witness: "go test -v ./internal/stopgate/... -run 'TestStopgatePrematureSurrender|TestStopgateGoalPersistence'"
target_files:
  - internal/stopgate/types.go
  - internal/stopgate/boundary.go
  - internal/stopgate/boundary_test.go
---
# Sub-Goal Objective
Implement premature give-up / surrender refusal and goal persistence in internal/stopgate so that an agent cannot easily give up on active goals without verified terminal boundary evidence, and instead receives actionable continuation guidance to subdivide and delegate.

# Scope Fence
- Work exclusively in: internal/stopgate/types.go, internal/stopgate/boundary.go, internal/stopgate/boundary_test.go
- Prohibited: Do not touch root configs, go.mod, go.sum, dos.toml, or sibling packages.

# Witness Command
go test -v ./internal/stopgate/... -run 'TestStopgatePrematureSurrender|TestStopgateGoalPersistence'
