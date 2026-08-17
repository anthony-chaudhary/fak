package superloop

// NoProgressStage is the bounded recovery rung selected from consecutive witnessed
// zero-yield cycles. Progress always resets to dispatch; repeated failures climb and cap.
type NoProgressStage struct {
	Name    string `json:"name"`
	Command string `json:"command,omitempty"`
	Reason  string `json:"reason"`
}

// EscalateNoProgress maps a consecutive no-progress streak onto existing recovery
// surfaces. It is pure so an overnight simulation can prove the whole transition table.
func EscalateNoProgress(streak int) NoProgressStage {
	switch {
	case streak <= 0:
		return NoProgressStage{Name: "dispatch", Command: "go run ./cmd/fak dispatch sweep", Reason: "fresh or reset cycle: dispatch actionable work"}
	case streak == 1:
		return NoProgressStage{Name: "retry", Command: "go run ./cmd/fak dispatch sweep", Reason: "one zero-yield cycle: retry after transient capacity or lease movement"}
	case streak == 2:
		return NoProgressStage{Name: "replan", Command: "dos replan --workspace . --json", Reason: "two zero-yield cycles: refresh the plan portfolio from evidence"}
	case streak == 3:
		return NoProgressStage{Name: "unblock", Command: "dos promote --workspace . --json", Reason: "three zero-yield cycles: surface typed holds and safe unblock moves"}
	case streak == 4:
		return NoProgressStage{Name: "unstick", Command: "dos unstick --workspace . --json", Reason: "four zero-yield cycles: cluster the recurring wedge and propose a structural fix"}
	default:
		return NoProgressStage{Name: "operator-decision", Command: "dos decisions --workspace . --json", Reason: "recovery ladder exhausted: route a bounded decision instead of spinning or declaring drain"}
	}
}
