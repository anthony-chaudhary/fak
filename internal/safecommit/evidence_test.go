package safecommit

import (
	"encoding/json"
	"testing"
	"time"
)

func TestCommitEvidenceGoldenMatrix(t *testing.T) {
	passed := &BuildCheckResult{Outcome: BuildCheckPassed}
	compileRed := &BuildCheckResult{Outcome: BuildCheckFailed, CompileEvidence: EvidenceFailed, TestEvidence: EvidenceUnrun}
	testRed := &BuildCheckResult{Outcome: BuildCheckFailed, CompileEvidence: EvidencePassed, TestEvidence: EvidenceFailed}
	skipped := &BuildCheckResult{Outcome: BuildCheckSkippedInfra, Compiled: false, FailedOpen: true}

	cases := []struct {
		name      string
		base      Result
		contract  EvidenceContract
		verified  bool
		delivery  bool
		compile   EvidenceOutcome
		test      EvidenceOutcome
		push      EvidenceOutcome
		closure   EvidenceOutcome
		wantScore int
		velScored bool
	}{
		{"green local", Result{Committed: true, Verified: true, BuildCheck: passed}, EvidenceContract{}, true, true, EvidencePassed, EvidencePassed, EvidenceUnrun, EvidenceUnrun, 100, true},
		{"compile red", Result{Committed: true, Verified: true, BuildCheck: compileRed}, EvidenceContract{}, false, false, EvidenceFailed, EvidenceUnrun, EvidenceUnrun, EvidenceUnrun, 55, false},
		{"compile skipped infra", Result{Committed: true, Verified: true, BuildCheck: skipped}, EvidenceContract{}, false, false, EvidenceSkipped, EvidenceUnrun, EvidenceUnrun, EvidenceUnrun, 55, false},
		{"test red", Result{Committed: true, Verified: true, BuildCheck: testRed}, EvidenceContract{}, false, false, EvidencePassed, EvidenceFailed, EvidenceUnrun, EvidenceUnrun, 55, false},
		{"test unrun", Result{Committed: true, Verified: true, BuildCheck: &BuildCheckResult{Outcome: BuildCheckPassed, CompileEvidence: EvidencePassed, TestEvidence: EvidenceUnrun}}, EvidenceContract{}, false, false, EvidencePassed, EvidenceUnrun, EvidenceUnrun, EvidenceUnrun, 55, false},
		{"push green", Result{Committed: true, Verified: true, Pushed: true, BuildCheck: passed}, EvidenceContract{RequirePush: true}, true, true, EvidencePassed, EvidencePassed, EvidencePassed, EvidenceUnrun, 100, true},
		{"push unrun required", Result{Committed: true, Verified: true, BuildCheck: passed}, EvidenceContract{RequirePush: true}, false, false, EvidencePassed, EvidencePassed, EvidenceUnrun, EvidenceUnrun, 55, false},
		{"closure bound", Result{Committed: true, Verified: true, BuildCheck: passed}, EvidenceContract{RequireClosure: true, ClosureBound: true}, true, true, EvidencePassed, EvidencePassed, EvidenceUnrun, EvidencePassed, 100, true},
		{"closure unbound", Result{Committed: true, Verified: true, BuildCheck: passed}, EvidenceContract{RequireClosure: true}, false, false, EvidencePassed, EvidencePassed, EvidenceUnrun, EvidenceUnrun, 55, false},
		{"record only", Result{Committed: true, Verified: true, BuildCheck: &BuildCheckResult{Outcome: BuildCheckDisabled}}, EvidenceContract{CompletionClass: CompletionRecordOnly}, true, false, EvidenceUnrun, EvidenceUnrun, EvidenceUnrun, EvidenceUnrun, 85, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := ScoreCommitVelocity(tc.base, time.Millisecond, time.Millisecond, DefaultVelocityBudgets)
			tc.base.Velocity = &v
			got := ScoreResult(FinalizeEvidence(tc.base, tc.contract))
			if got.Verified != tc.verified || got.DeliveryVerified() != tc.delivery {
				t.Fatalf("verified=%v delivery=%v, want %v/%v; evidence=%+v", got.Verified, got.DeliveryVerified(), tc.verified, tc.delivery, got.Evidence)
			}
			if got.Evidence.Schema != EvidenceSchema || got.Evidence.Compiled.Outcome != tc.compile || got.Evidence.Tested.Outcome != tc.test || got.Evidence.Pushed.Outcome != tc.push || got.Evidence.ClosureBound.Outcome != tc.closure {
				t.Fatalf("axes=%+v; want compile=%s test=%s push=%s closure=%s", got.Evidence, tc.compile, tc.test, tc.push, tc.closure)
			}
			if got.Score != tc.wantScore {
				t.Fatalf("score=%d notes=%v, want %d", got.Score, got.ScoreNotes, tc.wantScore)
			}
			if scored := got.Velocity.Local.Status == VelocityScored; scored != tc.velScored {
				t.Fatalf("velocity=%+v, scored=%v want %v", got.Velocity.Local, scored, tc.velScored)
			}
		})
	}
}

func TestCommitEvidenceLegacyCompatibility(t *testing.T) {
	legacy := Result{Committed: true, Verified: true}
	if !legacy.DeliveryVerified() {
		t.Fatal("schema-less Result must retain the legacy Verified contract")
	}
	encoded, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip Result
	if err := json.Unmarshal(encoded, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if !roundTrip.Verified || roundTrip.Evidence != nil {
		t.Fatalf("legacy round trip changed: %+v", roundTrip)
	}
}

func TestSkippedInfraReceiptCannotClaimVerified(t *testing.T) {
	res := FinalizeEvidence(Result{
		Committed:  true,
		Verified:   true,
		BuildCheck: &BuildCheckResult{Outcome: BuildCheckSkippedInfra, FailedOpen: true},
	}, EvidenceContract{})
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatal(err)
	}
	if wire["verified"] != false {
		t.Fatalf("contradictory receipt survived: %s", b)
	}
}
