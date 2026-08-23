package fleetmon

import (
	"reflect"
	"testing"
	"time"
)

func TestEvaluateWedgedRecoveryIsOrderedBoundedAndIdempotent(t *testing.T) {
	worker := PlanWorker{Issue: 7208, Session: "issue-7208"}
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)

	first := EvaluateWedgedRecovery(RecoveryRequest{Worker: worker, Progress: Wedged, MaxAttempts: 1, Now: now})
	want := []RecoveryAction{RecoveryPark, RecoveryReclaim, RecoveryReplace}
	if !first.Eligible || first.ReplacementSession != "issue-7208-replacement-1" || !reflect.DeepEqual(first.Actions, want) {
		t.Fatalf("first recovery = %#v, want ordered park/reclaim/replace", first)
	}

	deduped := EvaluateWedgedRecovery(RecoveryRequest{Worker: worker, Progress: Wedged, ExistingReplacement: first.ReplacementSession, MaxAttempts: 1, Now: now})
	if !deduped.Eligible || len(deduped.Actions) != 0 || deduped.ReplacementSession != first.ReplacementSession {
		t.Fatalf("retry must dedupe replacement: %#v", deduped)
	}

	exhausted := EvaluateWedgedRecovery(RecoveryRequest{Worker: worker, Progress: Wedged, Attempts: 1, MaxAttempts: 1, Now: now})
	if exhausted.Eligible || !reflect.DeepEqual(exhausted.Actions, []RecoveryAction{RecoveryEscalate}) {
		t.Fatalf("repeated wedge must escalate: %#v", exhausted)
	}

	progressing := EvaluateWedgedRecovery(RecoveryRequest{Worker: worker, Progress: Progressing, MaxAttempts: 1, Now: now})
	if progressing.Eligible || len(progressing.Actions) != 0 {
		t.Fatalf("progressing worker must be preserved: %#v", progressing)
	}
}

func TestEvaluateWedgedRecoveryPreservesCheckpointedWorktree(t *testing.T) {
	got := EvaluateWedgedRecovery(RecoveryRequest{
		Worker:       PlanWorker{Issue: 7208, Session: "issue-7208"},
		Progress:     Wedged,
		Checkpointed: true,
		MaxAttempts:  1,
	})
	want := []RecoveryAction{RecoveryReclaim, RecoveryReplace}
	if !reflect.DeepEqual(got.Actions, want) {
		t.Fatalf("checkpointed recovery = %v, want %v", got.Actions, want)
	}
}
