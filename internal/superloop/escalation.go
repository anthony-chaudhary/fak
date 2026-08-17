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
		return NoProgressStage{Name: "dispatch", Command: "go run ./cmd/fak dispatch sweep --live", Reason: "fresh or reset cycle: dispatch actionable work"}
	case streak == 1:
		return NoProgressStage{Name: "retry", Command: "go run ./cmd/fak dispatch sweep --live", Reason: "one zero-yield cycle: retry after transient capacity or lease movement"}
	case streak == 2:
		return NoProgressStage{Name: "replan", Command: "dos plan --workspace . --once --json", Reason: "two zero-yield cycles: refresh the canonical plan portfolio from evidence"}
	case streak == 3:
		return NoProgressStage{Name: "unblock", Command: "go run ./cmd/fak-dev issue repair --live --json", Reason: "three zero-yield cycles: apply canonical safe issue-contract repairs"}
	case streak == 4:
		return NoProgressStage{Name: "unstick", Command: "go run ./cmd/fak stale-work loop --live-issues --live-launch --json", Reason: "four zero-yield cycles: recover stale work through the guarded issue and launch path"}
	default:
		return NoProgressStage{Name: "operator-decision", Command: "dos decisions --workspace . --json", Reason: "recovery ladder exhausted: route a bounded decision instead of spinning or declaring drain"}
	}
}
