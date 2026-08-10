package main

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/idempotency"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"path/filepath"
	"testing"
	"time"
)

func TestEffectBatchSelfcheck(t *testing.T) {
	p := filepath.Join(t.TempDir(), "effects.json")
	if e := runEffectBatchSelfcheck(p); e != nil {
		t.Fatal(e)
	}
	if e := verifyEffectBatchArtifact(p); e != nil {
		t.Fatal(e)
	}
}
func TestSelectorCannotGrantEffectAuthority(t *testing.T) {
	s, _ := idempotency.Open(filepath.Join(t.TempDir(), "l"), time.Hour)
	state := &fixtureEffects{values: map[string]string{}, applies: map[string]int{}}
	r := executeSelectedEffect(context.Background(), microagent.NewEffectCoordinator(s), state, selectedEffect{ContextID: "x", Stage: "set-label", Capability: "issue.delete", Resource: "issue:1", Operation: "delete", IdempotencyKey: "x", Approved: true}, []string{"issue.label.write"})
	if r.Status != "denied" || state.calls.Load() != 0 {
		t.Fatalf("authority escaped: %+v", r)
	}
}
func TestUnknownCannotEnterConfirmedFold(t *testing.T) {
	r := effectBatchReport{Schema: effectBatchSchema, Confirmed: 6, ReplayedConfirmed: 3, Denied: 5, Conflicts: 1, Failed: 1, CancelledBeforeDispatch: 1, UnknownPendingReadback: 1, UnknownLaterConfirmed: 1, ApprovalNotRun: 1, DryRuns: 1, BreakerNotRun: 2, FoldedUnknown: 0}
	if verifyEffectBatch(r) == nil {
		t.Fatal("hidden unknown accepted")
	}
}
func TestVerifierRejectsDuplicateApply(t *testing.T) {
	r := effectBatchReport{Schema: effectBatchSchema, Confirmed: 6, ReplayedConfirmed: 3, Denied: 5, Conflicts: 1, Failed: 1, CancelledBeforeDispatch: 1, UnknownPendingReadback: 1, UnknownLaterConfirmed: 1, ApprovalNotRun: 1, DryRuns: 1, BreakerNotRun: 2, FoldedUnknown: 1, DuplicatePhysicalApplies: 1}
	if verifyEffectBatch(r) == nil {
		t.Fatal("duplicate apply accepted")
	}
}
