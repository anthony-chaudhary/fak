package resume

import "github.com/anthony-chaudhary/fak/internal/trajctl"

type TrajectoryWatchdogAction string

const (
	TrajectoryLeaveAlone   TrajectoryWatchdogAction = "LEAVE_ALONE"
	TrajectoryNudge        TrajectoryWatchdogAction = "NUDGE"
	TrajectoryReviveAnchor TrajectoryWatchdogAction = "REVIVE_WITH_FRESH_ANCHOR"
)

type TrajectoryWatchdogInput struct {
	Alive          bool
	Signal         trajctl.Signal
	NudgeAttempted bool
}

type TrajectoryWatchdogDecision struct {
	Action TrajectoryWatchdogAction `json:"action"`
	Reason string                   `json:"reason"`
}

// DecideTrajectoryWatchdog is the clock/IO-free intervention core. Liveness
// dominates: dead sessions revive from the independently-read fresh anchor.
// Living healthy/slow sessions are untouched; a living stall gets the cheapest
// reversible intervention first and only escalates after that nudge was tried.
func DecideTrajectoryWatchdog(in TrajectoryWatchdogInput) TrajectoryWatchdogDecision {
	if !in.Alive {
		return TrajectoryWatchdogDecision{Action: TrajectoryReviveAnchor, Reason: "session is dead; revive from the fresh witnessed anchor"}
	}
	if in.Signal == trajctl.SignalStall {
		if !in.NudgeAttempted {
			return TrajectoryWatchdogDecision{Action: TrajectoryNudge, Reason: "session is alive but its witnessed curve is stalled; nudge before revive"}
		}
		return TrajectoryWatchdogDecision{Action: TrajectoryReviveAnchor, Reason: "session remained stalled after a nudge; revive from the fresh witnessed anchor"}
	}
	return TrajectoryWatchdogDecision{Action: TrajectoryLeaveAlone, Reason: "session is alive and its witnessed curve is healthy or merely slow"}
}
