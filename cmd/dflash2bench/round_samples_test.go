package main

import (
	"github.com/anthony-chaudhary/fak/internal/model"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2RoundSamplesReconcile(t *testing.T) {
	makeArm := func() arm {
		return arm{Rounds: 1, DraftCalls: 1, RequestedDraftPositions: 7,
			ProposedDraftTokens: 7, VerifiedDraftTokens: 7, CommittedDraftTokens: 1, CommittedTargetAnchors: 1,
			TokenIDs: []int{0, 0}, ProposalSeconds: 1, VerifySeconds: 2, PrepareSeconds: 4, CommitSeconds: 3,
			RoundSamples: []roundSample{{Index: 0, RequestedDraftPositions: 7, ProposedDraftTokens: 7,
				VerifiedDraftTokens: 7, CommittedDraftTokens: 1, CommittedTargetAnchors: 1, StopTruncatedTokens: 6,
				ProposalRan: true, PrepareComplete: true, Complete: true,
				ProposalSeconds: 1, VerifySeconds: 2, PrepareSeconds: 4, CommitSeconds: 3,
				Commit: model.DFlash2CommitObservation{Complete: true, ReceiptAvailable: true,
					Verification: model.TargetVerificationReceipt{DraftTokens: 0, AcceptedTokens: 2, RejectedTokens: 6}}}}}
	}
	a := makeArm()
	if err := reconcileRoundSamples(&a); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*arm){
		"partial":            func(a *arm) { a.RoundSamples[0].Complete = false },
		"commit unavailable": func(a *arm) { a.RoundSamples[0].Commit.ReceiptAvailable = false },
		"tokens":             func(a *arm) { a.CommittedDraftTokens++ },
		"timing":             func(a *arm) { a.RoundSamples[0].CommitSeconds++ },
		"stop":               func(a *arm) { a.RoundSamples[0].StopTruncatedTokens++ },
	} {
		t.Run(name, func(t *testing.T) {
			a := makeArm()
			mutate(&a)
			if err := reconcileRoundSamples(&a); err == nil {
				t.Fatal("mismatched or partial sample accepted")
			}
		})
	}
}
