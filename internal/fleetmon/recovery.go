package fleetmon

import "time"

// RecoveryAction is one ordered, non-destructive step for a wedged worker.
type RecoveryAction string

const (
	RecoveryPark     RecoveryAction = "PARK_CHECKPOINT"
	RecoveryReclaim  RecoveryAction = "RECLAIM_PROCESS_AND_LEASE"
	RecoveryReplace  RecoveryAction = "START_REPLACEMENT"
	RecoveryEscalate RecoveryAction = "ESCALATE"
)

// RecoveryRequest is the independently witnessed state needed to recover one
// productive-liveness wedge. ExistingReplacement and Attempts make retries
// idempotent and bounded.
type RecoveryRequest struct {
	Worker              PlanWorker
	Progress            ProgressState
	Checkpointed        bool
	ExistingReplacement string
	Attempts            int
	MaxAttempts         int
	Now                 time.Time
}

// RecoveryDecision is an ordered recovery contract. It never removes a
// worktree: parking/checkpointing precedes process/lease reclamation.
type RecoveryDecision struct {
	Eligible           bool             `json:"eligible"`
	Reason             string           `json:"reason"`
	ReplacementSession string           `json:"replacement_session,omitempty"`
	Actions            []RecoveryAction `json:"actions,omitempty"`
}

// EvaluateWedgedRecovery plans one bounded, idempotent recovery.
func EvaluateWedgedRecovery(req RecoveryRequest) RecoveryDecision {
	if req.Progress != Wedged {
		return RecoveryDecision{Reason: "worker is not WEDGED"}
	}
	if req.ExistingReplacement != "" {
		return RecoveryDecision{Eligible: true, Reason: "replacement already exists; no duplicate launch", ReplacementSession: req.ExistingReplacement}
	}
	max := req.MaxAttempts
	if max <= 0 {
		max = 1
	}
	if req.Attempts >= max {
		return RecoveryDecision{Reason: "recovery budget exhausted", Actions: []RecoveryAction{RecoveryEscalate}}
	}
	session := EvaluateReplace(ReplaceRequest{Worker: req.Worker, Class: ClassStaleTranscript, Index: req.Attempts + 1, Now: req.Now}).NewSession
	actions := []RecoveryAction{}
	if !req.Checkpointed {
		actions = append(actions, RecoveryPark)
	}
	actions = append(actions, RecoveryReclaim, RecoveryReplace)
	return RecoveryDecision{Eligible: true, Reason: "wedged worker has recovery budget", ReplacementSession: session, Actions: actions}
}
