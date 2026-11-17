# SubagentSupervisor: deliberately tested-but-unwired (decision record)

- Date: 2026-11-17
- Lane: agent
- Resolves: `anthony-chaudhary/fak-private#3286` ("NewSubagentSupervisor and
  TeardownParent have no production caller")
- Follow-up: [fak#12632](https://github.com/anthony-chaudhary/fak/issues/12632)
  (OPEN) — "fix(agent): propagate context cancellation and active thread
  teardown in SubagentSupervisor"

## Question

`NewSubagentSupervisor` and `TeardownParent`
(`internal/agent/subagent_supervisor.go`, landed in commit `4b9bf91fc`) have
no production caller. The acceptance gate offers two exits: wire the
primitive into a real production entrypoint with proof-of-use, or record a
decision that the primitive is deliberately unwired.

## Evidence

1. **No production caller.** Grep for `SubagentSupervisor|TeardownParent`
   across `cmd/`, `pkg/`, and non-test `internal/` sources returns only the
   primitive's own file and `subagent_supervisor_test.go`.
2. **The production subagent task path is `TaskState`**
   (`internal/agent/tasktools.go`, #11414/#11840). The run defers
   `closeChildTaskState` (`internal/agent/loop.go`), which calls
   `TaskState.closeAndWait` — the run-owned cleanup boundary that owns
   executor cancellation and join.
3. **The supervisor is an observer, not an executor.**
   `ErrSubagentAbandoned` is explicitly "observer teardown, not executor
   cancellation." Wiring the supervisor into the task path would create a
   second source of truth for subagent lifecycle (duplicating `TaskState`
   tracking) or regress cancellation semantics.
4. **Collision check.** `modelengine.SubagentState*`
   (`internal/modelengine/subagent_sched.go`: waiting/prefilling/active/
   yielded/completed/cancelled) is a distinct decode-lane FSM in a distinct
   package; no name collision with `agent.SubagentState*`
   (running/completed/failed/timed_out/abandoned). `modelengine.
   SubagentScheduler` is itself unwired — out of scope for this decision.

## Decision

Keep `SubagentSupervisor` as a **deliberately tested-but-unwired** primitive:
exported, covered by deterministic tests
(`go test ./internal/agent/ -run TestSubagentSupervisor -count=1`), and
documented as such in its doc comment. No speculative entrypoint is invented
solely to consume the primitive, and teardown semantics are not redesigned.
The primitive becomes production-relevant through #12632, which will make the
observer's cancellation/teardown behavior real; the wiring decision is
revisited at that point with proof-of-use from the entrypoint that owns it.
