package sessionintent

import "time"

// Progress is the executor-owned measurement snapshot used for a stop decision.
type Progress struct {
	Active    time.Duration
	Elapsed   time.Duration
	Completed bool
	Failed    bool
	Cancelled bool
}

// StopState explains whether work must continue, may finish, or must stop.
type StopState string

const (
	StopContinue  StopState = "continue"
	StopEligible  StopState = "eligible"
	StopTimeout   StopState = "timeout"
	StopComplete  StopState = "complete"
	StopFailed    StopState = "failed"
	StopCancelled StopState = "cancelled"
)

// StopDecision is deterministic and receipt-ready.
type StopDecision struct {
	State  StopState `json:"state"`
	Reason string    `json:"reason"`
}

// DecideStop applies terminal signals, hard maxima, completion, then minimum eligibility.
// Target bounds never decide a stop: they are planner observations.
func (i Intent) DecideStop(p Progress) StopDecision {
	if p.Cancelled {
		return StopDecision{State: StopCancelled, Reason: "operator_cancelled"}
	}
	if p.Failed {
		return StopDecision{State: StopFailed, Reason: "executor_failed"}
	}
	for _, b := range i.Effort {
		if b.Kind == BoundMaximum && measured(b.Clock, p) >= b.Duration {
			return StopDecision{State: StopTimeout, Reason: string(b.Clock) + "_maximum_reached"}
		}
	}
	for _, b := range i.Effort {
		if b.Kind == BoundMinimum && measured(b.Clock, p) < b.Duration {
			return StopDecision{State: StopContinue, Reason: string(b.Clock) + "_minimum_not_reached"}
		}
	}
	if p.Completed {
		return StopDecision{State: StopComplete, Reason: "completion_evidence_satisfied"}
	}
	return StopDecision{State: StopEligible, Reason: "minimums_satisfied"}
}

func measured(clock Clock, p Progress) time.Duration {
	if clock == ClockActive {
		return p.Active
	}
	return p.Elapsed
}
