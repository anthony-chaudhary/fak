package main

import (
	"strings"
)

// Phase identifiers for worker progression.
const (
	WorkerPhaseRedReproduction     = "RED_REPRODUCTION"
	WorkerPhaseGreenImplementation = "GREEN_IMPLEMENTATION"
	WorkerPhaseUnlocked            = "UNLOCKED"
)

// WorkerPhaseState tracks whether a worker is in a mandatory red reproduction
// phase or has unlocked code edits.
type WorkerPhaseState struct {
	IssueKind              string `json:"issue_kind,omitempty"`
	Phase                  string `json:"phase,omitempty"`
	HasFailingTestProof    bool   `json:"has_failing_test_proof"`
	ImplementationUnlocked bool   `json:"implementation_unlocked"`
	Reason                 string `json:"reason,omitempty"`
}

// InitWorkerPhase initializes the worker phase state based on the issue kind.
// Bug issues ("bug", "kind:bug") require a red reproduction phase proving test
// failure before code edits are permitted. Other issue kinds ("feat", "docs",
// "refactor", etc.) initialize unlocked.
func InitWorkerPhase(issueKind string) WorkerPhaseState {
	norm := strings.ToLower(strings.TrimSpace(issueKind))
	switch norm {
	case "bug", "kind:bug":
		return WorkerPhaseState{
			IssueKind:              issueKind,
			Phase:                  WorkerPhaseRedReproduction,
			ImplementationUnlocked: false,
			Reason:                 "Bug fix requires reproduction test proving failure before code tree is unlocked.",
		}
	default:
		return WorkerPhaseState{
			IssueKind:              issueKind,
			Phase:                  WorkerPhaseUnlocked,
			ImplementationUnlocked: true,
			Reason:                 "Non-bug issue does not require red reproduction phase.",
		}
	}
}

// ValidateWorkerRedPhase validates the outcome of a reproduction test run.
// For issues in RED_REPRODUCTION:
//   - A failing test (testRan && testExitCode != 0) establishes proof of defect,
//     transitioning to GREEN_IMPLEMENTATION and unlocking code edits.
//   - A passing test (testRan && testExitCode == 0) is rejected as tautological.
//   - If no test ran (!testRan), code tree remains locked.
//
// If the worker is already unlocked or not in the red phase, returns (true, "").
func ValidateWorkerRedPhase(state *WorkerPhaseState, testExitCode int, testRan bool) (unlocked bool, reason string) {
	if state == nil {
		return false, "nil worker phase state"
	}
	if state.ImplementationUnlocked || state.Phase != WorkerPhaseRedReproduction {
		return true, ""
	}

	if !testRan {
		state.ImplementationUnlocked = false
		state.Reason = "No reproduction test executed."
		return false, state.Reason
	}

	if testExitCode == 0 {
		state.ImplementationUnlocked = false
		state.Reason = "Test passed without fix; reproduction test must fail on unfixed codebase."
		return false, state.Reason
	}

	state.Phase = WorkerPhaseGreenImplementation
	state.ImplementationUnlocked = true
	state.HasFailingTestProof = true
	state.Reason = ""
	return true, ""
}
