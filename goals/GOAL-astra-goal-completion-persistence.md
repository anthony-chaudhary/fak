---
loop: goal
goal_slug: astra-goal-completion-persistence
witness: "go test -v ./internal/headlesslint/... ./internal/stopgate/... ./internal/agent/... -run 'TestPrematureSurrender|TestStopgate|TestGoalAnchor'"
budget: { max_iters: 20 }
lane: multi
---
# Objective
Make Astra (via fak) 10x better at completing goals by preventing premature surrender/giving up, enforcing goal persistence, and empowering subagent delegation.

# Non-Goals
- Do not edit frozen ABI (`internal/abi`).
- Do not touch `go.mod`, `go.sum`, `dos.toml`, or unrelated subsystems.
- Do not commit peer WIP or use `git add -A`.
- Do not create shell scripts (`.sh` / `.ps1`).

# Plan
- [x] 1. Establish baseline reproduction showing unhindered premature give-up in headlesslint, stopgate, and agent loop
- [x] 2. Implement PrematureSurrender detection in internal/headlesslint
- [x] 3. Implement premature surrender refusal and goal persistence in internal/stopgate
- [x] 4. Wire GoalAnchor persistence and subagent delegation guidance in internal/agent
- [x] 5. Enhance Astra manager subagent delegation and goal persistence in cmd/fak and orchestration
- [x] 6. Verify all affected packages with deterministic test witnesses

# Results and Verification Evidence
- **Step 1 (Baseline Repro)**: Identified that agents saying "giving up", "cannot complete", or stopping without tool calls under an active goal previously fell through to `DispCleanCompletion` (exit 0) and `headlesslint.Clean`.
- **Step 2 (Headlesslint)**: Added `PrematureSurrender Class = "PREMATURE_SURRENDER"` in `internal/headlesslint/headlesslint.go` with regex patterns and actionable resolution. Verified with `go test -v ./internal/headlesslint -run TestScanClassifies` (PASS).
- **Step 3 (Stopgate)**: Added `IsSurrenderNote`, `NotedSurrender`, `SurrenderNote`, `GoalActive`, `GoalObjective` in `internal/stopgate/`. `EvaluateBoundary` blocks premature surrender on active goals with `ActionContinue` (exit 2) and actionable persistence guidance. Verified with `go test -v ./internal/stopgate/... -run 'TestStopgatePrematureSurrender|TestStopgateGoalPersistence'` (PASS).
- **Step 4 (Agent Loop)**: Wired `GoalAnchor` persistence into `BoundaryInput` in `internal/agent/loop_turn.go` and reinforced subagent delegation in `FormatRecoveryReinforcement`. Verified with `go test -v ./internal/agent/... -run 'TestStopgate|TestGoalAnchor'` (PASS).
- **Step 5 (Orchestration & cmd/fak)**: Added subagent signals to `parallelSignals` in `internal/orchestration/solroute.go` to route tasks mentioning subagents to `SOLUltra` multi-agent delegation, and updated `guardOperatorDirectedContinueMessage` in `cmd/fak/guard_operator_directed.go` to tailor continuation instructions for premature surrender. Verified with `go test -v ./internal/orchestration -run TestSelectSOLRouteSubagentsSignal` and `go test -v ./cmd/fak -run TestGuardOperatorDirected` (PASS).
- **Step 6 (Comprehensive Witness)**: Verified all targeted packages cleanly with `go test -v ./internal/headlesslint/... ./internal/stopgate/... ./internal/agent/... ./internal/orchestration/...` and `go vet`.

# Scratch / last-refusal
All 6 milestones verified with deterministic tests and green witnesses. Premature surrender eliminated on active goals with subagent delegation reinforcement.
