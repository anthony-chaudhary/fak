package agent

import (
	"context"
	"errors"
	"time"
)

// SpeculativeAttemptCost counts one outcome's actual proposal attempts. Its
// inclusive span contains drafting; the two durations must not be added.
type SpeculativeAttemptCost struct {
	Attempts           int           `json:"attempts"`
	EmittedTokens      int           `json:"emitted_tokens"`
	MeasuredAttempts   int           `json:"measured_attempts"`
	UnmeasuredAttempts int           `json:"unmeasured_attempts"`
	DraftElapsed       time.Duration `json:"draft_elapsed_ns"`
	InclusiveElapsed   time.Duration `json:"inclusive_elapsed_ns"`
}

// SpeculativeDecodeCost covers request-owned proposal-through-emission-and-
// advance attempts, excluding prefill and unassisted serial remainder.
type SpeculativeDecodeCost struct {
	Scope             string                 `json:"scope"`
	Attempts          int                    `json:"attempts"`
	Completed         SpeculativeAttemptCost `json:"completed"`
	ProposalError     SpeculativeAttemptCost `json:"proposal_error"`
	EmptyProposal     SpeculativeAttemptCost `json:"empty_proposal"`
	VerificationError SpeculativeAttemptCost `json:"verification_error"`
	Cancelled         SpeculativeAttemptCost `json:"cancelled"`
	Incomplete        SpeculativeAttemptCost `json:"incomplete"`
}

type speculativeAttemptOutcome uint8

const (
	speculativeAttemptIncomplete speculativeAttemptOutcome = iota
	speculativeAttemptCompleted
	speculativeAttemptProposalError
	speculativeAttemptEmptyProposal
	speculativeAttemptVerificationError
)

type speculativeAttemptObservation struct {
	totals       *SpeculativeDecodeCost
	start        time.Time
	genAtOpen    int
	draftElapsed time.Duration
	draftKnown   bool
	outcome      speculativeAttemptOutcome
	active       bool
	cancelled    bool
}

func (a *speculativeAttemptObservation) begin(start time.Time, gen int) {
	if a.totals == nil {
		a.totals = &SpeculativeDecodeCost{Scope: "proposal-through-emission-and-advance"}
	}
	a.start, a.genAtOpen = start, gen
	a.draftElapsed, a.outcome = 0, speculativeAttemptIncomplete
	a.draftKnown = false
	a.active, a.cancelled = true, false
}

func speculativeAttemptCancelled(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func (a *speculativeAttemptObservation) draftFinished(elapsed time.Duration, err error) {
	a.draftElapsed = elapsed
	a.draftKnown = true
	if err != nil {
		a.outcome = speculativeAttemptProposalError
		a.cancelled = speculativeAttemptCancelled(err)
	}
}

func (a *speculativeAttemptObservation) emptyProposal() {
	if a.outcome != speculativeAttemptProposalError {
		a.outcome = speculativeAttemptEmptyProposal
	}
}

func (a *speculativeAttemptObservation) verificationFailed(err error) {
	a.outcome = speculativeAttemptVerificationError
	a.cancelled = speculativeAttemptCancelled(err)
}

func (a *speculativeAttemptObservation) close(gen int, err, ctxErr error, now func() time.Time) {
	if !a.active {
		return
	}
	a.active = false
	bucket := &a.totals.Incomplete
	if a.cancelled || speculativeAttemptCancelled(err) || speculativeAttemptCancelled(ctxErr) {
		bucket = &a.totals.Cancelled
	} else {
		switch a.outcome {
		case speculativeAttemptCompleted:
			if err == nil {
				bucket = &a.totals.Completed
			}
		case speculativeAttemptProposalError:
			bucket = &a.totals.ProposalError
		case speculativeAttemptEmptyProposal:
			bucket = &a.totals.EmptyProposal
		case speculativeAttemptVerificationError:
			bucket = &a.totals.VerificationError
		}
	}
	a.totals.Attempts++
	bucket.Attempts++
	bucket.EmittedTokens += max(0, gen-a.genAtOpen)
	if now == nil || !a.draftKnown {
		bucket.UnmeasuredAttempts++
		return
	}
	inclusive := now().Sub(a.start)
	if inclusive <= 0 || a.draftElapsed < 0 || inclusive < a.draftElapsed {
		bucket.UnmeasuredAttempts++
		return
	}
	bucket.MeasuredAttempts++
	bucket.DraftElapsed += a.draftElapsed
	bucket.InclusiveElapsed += inclusive
}
