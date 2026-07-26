package agent

import "context"

// governed_arm.go — the exported entry a server endpoint needs to drive ONE
// kernel-governed arm of the owned loop AND collect its per-call decision trace
// (#3258, epic #3256). RunArm already records every adjudicated call, but its
// trace log parameter is the unexported []traceEvent, so an external caller
// (internal/gateway's agent-runtime spine endpoint) could observe the ArmMetrics
// only — never the turn/tool/verdict rows the loop witnessed. This wrapper keeps
// the log internal and hands back the SAME exported CallTrace rows the A/B
// artifact embeds (RunResult.Calls), so a served session streams the identical
// decision trace a `fak run` artifact carries.

// RunGovernedArm drives the kernel-governed (fak) arm of the owned agent loop
// for one goal and returns the arm's witnessed metrics plus the per-call
// decision trace. It is RunArm(fak=true) with the trace log collected and
// converted to the exported CallTrace rows; options thread through unchanged,
// so the session gate / route manifest / steer bus wiring a host installs
// applies identically. On an error the calls recorded up to the failure are
// still returned, so a partial run remains debuggable from its trace.
func RunGovernedArm(ctx context.Context, p Planner, goal string, maxTurns int, opts ...RunOption) (ArmMetrics, []CallTrace, error) {
	var log []traceEvent
	m, err := RunArm(ctx, p, goal, true, maxTurns, &log, opts...)
	return m, toCallTraces(log), err
}
