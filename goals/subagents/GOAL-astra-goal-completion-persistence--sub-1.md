---
parent_goal: goals/GOAL-astra-goal-completion-persistence.md
sub_step: 1
witness: "go test -v ./internal/headlesslint/... -run TestScanClassifies"
target_files:
  - internal/headlesslint/headlesslint.go
  - internal/headlesslint/headlesslint_test.go
---
# Sub-Goal Objective
Add PrematureSurrender detection to internal/headlesslint to detect when an agent gives up or prematurely surrenders in its final turn, classifying it with actionable resolution to persist, decompose, and delegate to subagents.

# Scope Fence
- Work exclusively in: internal/headlesslint/headlesslint.go, internal/headlesslint/headlesslint_test.go
- Prohibited: Do not touch root configs, go.mod, go.sum, dos.toml, or sibling packages.

# Witness Command
go test -v ./internal/headlesslint/... -run TestScanClassifies
