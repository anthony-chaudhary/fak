package model

import (
	"context"
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime medium est=2s lane=default
func TestDFlash2PostCommitObservation(t *testing.T) {
	for _, accepted := range []int{0, 2, 8} {
		t.Run(fmt.Sprint(accepted), func(t *testing.T) {
			q, _ := dflash2RequestFixture(t, true)
			if err := q.Bootstrap(context.Background(), []int{3, 7, 11}); err != nil {
				t.Fatal(err)
			}
			round, timing, err := q.PrepareTimed(context.Background(), 8)
			if err != nil {
				t.Fatal(err)
			}
			if !timing.Complete || len(round.Tokens()) != timing.Verification.Rows {
				t.Fatal("fixture tokens disagree with completed verification work")
			}
			tx := round.cp.tx.tx
			tokens, observation, err := round.CommitObserved(context.Background(), accepted)
			if err != nil {
				t.Fatal(err)
			}
			if !observation.Complete || !observation.ReceiptAvailable || len(tokens) != accepted {
				t.Fatalf("invalid observation: %+v", observation)
			}
			receipt := observation.Verification
			if receipt.DraftTokens != 0 {
				t.Fatalf("serial receipt DraftTokens=%d, want original zero", receipt.DraftTokens)
			}
			if !reflect.DeepEqual(receipt, tx.VerificationReceipt()) {
				t.Fatal("post-commit receipt was not preserved exactly")
			}
			if got := receipt.MissingCostCategories(); !reflect.DeepEqual(got, []string{"drafting", "rejection", "recovery", "memory"}) {
				t.Fatalf("missing costs=%v", got)
			}
			if receipt.AcceptedTokens != accepted || receipt.RejectedTokens != 8-accepted || receipt.TargetDecodeSteps != 8 {
				t.Fatalf("wrong token receipt: %+v", receipt)
			}
			wantReplay := 0
			if accepted < 8 {
				wantReplay = accepted
			}
			if receipt.FullTargetReplaySteps != wantReplay || receipt.RecurrentRepairTokens != 0 {
				t.Fatalf("wrong replay receipt: %+v", receipt)
			}
			if !receipt.Accounting.TargetVerification.Measured || !receipt.Accounting.Synchronization.Measured || receipt.Accounting.Drafting.Measured || receipt.EndToEndMeasured() {
				t.Fatalf("cost availability lost: %+v", receipt.Accounting)
			}
			_, stale, err := round.CommitObserved(context.Background(), 0)
			if err == nil || stale.Complete || stale.ReceiptAvailable {
				t.Fatal("stale round produced successful diagnostic")
			}
			if observation.Verification.AcceptedTokens != accepted {
				t.Fatal("receipt changed after stale call")
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2PostCommitObservationCanceled(t *testing.T) {
	q, _ := dflash2RequestFixture(t, true)
	if err := q.Bootstrap(context.Background(), []int{3, 7, 11}); err != nil {
		t.Fatal(err)
	}
	round, _, err := q.PrepareTimed(context.Background(), 8)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tokens, observation, err := round.CommitObserved(ctx, 2)
	if err == nil || observation.Complete || tokens != nil {
		t.Fatal("canceled round counted as successful")
	}
	if !observation.ReceiptAvailable {
		t.Fatal("partial transaction evidence discarded")
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2PostCommitObservationVerifiedPrefix(t *testing.T) {
	q, _ := dflash2RequestFixture(t, false)
	if err := q.Bootstrap(context.Background(), []int{3, 7, 11}); err != nil {
		t.Fatal(err)
	}
	round, timing, err := q.PrepareTimed(context.Background(), 8)
	if err != nil {
		t.Fatal(err)
	}
	accepted := len(round.Tokens())
	tokens, observation, err := round.CommitObserved(context.Background(), accepted)
	if err != nil {
		t.Fatal(err)
	}
	if !observation.Complete || len(tokens) != accepted || observation.Verification.AcceptedTokens != accepted || observation.Verification.AcceptedTokens+observation.Verification.RejectedTokens != timing.ProposedDraftTokens+1 {
		t.Fatalf("acceptance receipt disagrees: %+v", observation)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2PostCommitObservationActualEOS(t *testing.T) {
	q, plain := dflash2RequestFixture(t, true)
	q.ref.target.M.Cfg.EOSTokenID = 0
	q.ref.target.M.Cfg.EOSTokenIDs = []int{0}
	prompt := []int{3, 7}
	if err := q.Bootstrap(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	want := dflash2RequestPlain(prompt, plain)
	if argmaxF32(want) != 0 || !q.ref.target.M.Cfg.IsEOS(0) {
		t.Fatal("EOS fixture is not active")
	}
	round, timing, err := q.PrepareTimed(context.Background(), 8)
	if err != nil {
		t.Fatal(err)
	}
	verified := round.Tokens()
	if !reflect.DeepEqual(verified, []int{0}) || !timing.Complete || timing.ProposalRan || timing.ProposedDraftTokens != 0 {
		t.Fatalf("unexpected EOS proposal: %v %+v", verified, timing)
	}
	// Exercise the same stop-before-commit decision as the diagnostic runner.
	chosen := len(verified)
	for i, token := range verified {
		if q.ref.target.M.Cfg.IsEOS(token) {
			chosen = i
			break
		}
	}
	tx := round.cp.tx.tx
	tokens, observation, err := round.CommitObserved(context.Background(), chosen)
	if err != nil {
		t.Fatal(err)
	}
	if !observation.Complete || !observation.ReceiptAvailable || len(tokens) != 0 || len(verified)-chosen != 1 {
		t.Fatalf("EOS observation=%+v tokens=%v", observation, tokens)
	}
	rc := observation.Verification
	if rc.DraftTokens != 0 || rc.AcceptedTokens != 0 || rc.RejectedTokens != 1 || rc.TargetDecodeSteps != 1 || rc.FullTargetReplaySteps != 0 || rc.RecurrentRepairTokens != 0 {
		t.Fatalf("EOS receipt=%+v", rc)
	}
	if !reflect.DeepEqual(rc, tx.VerificationReceipt()) || !reflect.DeepEqual(rc.MissingCostCategories(), []string{"drafting", "rejection", "recovery", "memory"}) {
		t.Fatal("EOS receipt or missing categories changed")
	}
	dflash2RequestEqual(t, q, plain, want)
}
