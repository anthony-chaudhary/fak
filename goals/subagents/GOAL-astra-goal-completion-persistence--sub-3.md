---
parent_goal: goals/GOAL-astra-goal-completion-persistence.md
sub_step: 3
witness: "go test -v ./internal/agent/... -run 'TestStopgatePrematureSurrender|TestStopgateGoalAnchor'"
target_files:
  - internal/agent/loop_turn.go
  - internal/agent/goal_anchor.go
  - internal/agent/goal_anchor_test.go
  - internal/agent/loop_stopgate_test.go
---
# Sub-Goal Objective
Wire GoalAnchor and premature surrender into the agent loop turn boundary (internal/agent) so that an agent cannot give up on active goals, receives subagent delegation reminders upon recovery, and continues to completion.

# Scope Fence
- Work exclusively in: internal/agent/loop_turn.go, internal/agent/goal_anchor.go, internal/agent/goal_anchor_test.go, internal/agent/loop_stopgate_test.go
- Prohibited: Do not touch root configs, go.mod, go.sum, dos.toml, or sibling packages.

# Witness Command
go test -v ./internal/agent/... -run 'TestStopgatePrematureSurrender|TestStopgateGoalAnchor'
