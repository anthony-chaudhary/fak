package agent

import (
	"context"
	"errors"
)

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

// RunGovernedArmStream is the streaming twin of RunGovernedArm: it drives the
// same kernel-governed (fak) arm through RunArmStream so natural-language
// content is delivered incrementally to sink as each model turn produces it,
// while tool calls remain held until adjudication exactly as the buffered path
// does. It returns the SAME (ArmMetrics, []CallTrace) contract. When the
// planner cannot stream (does not implement StreamingPlanner, or reports
// StreamingSupported()==false, or its wire fails with ErrStreamingUnsupported),
// it falls back cleanly to the buffered RunGovernedArm for that arm so offline/
// mock planners never break; buffered turns emit nothing to sink.
func RunGovernedArmStream(ctx context.Context, p Planner, goal string, maxTurns int, sink StreamSink, opts ...RunOption) (ArmMetrics, []CallTrace, error) {
	var log []traceEvent
	m, err := RunArmStream(ctx, p, goal, true, maxTurns, sink, &log, opts...)
	if err != nil && errors.Is(err, ErrStreamingUnsupported) {
		return RunGovernedArm(ctx, p, goal, maxTurns, opts...)
	}
	return m, toCallTraces(log), err
}
