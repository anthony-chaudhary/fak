package ultracodedogfood

import (
	"encoding/json"
	"os"
	"testing"
)

func TestLifecycleSessionObservedReplay(t *testing.T) {
	b, err := os.ReadFile("testdata/issue8678-lifecycle-session.json")
	if err != nil {
		t.Fatal(err)
	}
	r, err := EvaluateLifecycleSession(b)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" {
		t.Fatalf("verdict=%s reason=%s", r.Verdict, r.Reason)
	}
	if r.ScopeAvoidedTokens != 1621 {
		t.Fatalf("scope avoided=%d", r.ScopeAvoidedTokens)
	}
}

func TestLifecycleSessionFailsClosedOnAmbiguousBoundary(t *testing.T) {
	b, err := os.ReadFile("testdata/issue8678-lifecycle-session.json")
	if err != nil {
		t.Fatal(err)
	}
	var s LifecycleSession
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	s.Cells[2].BoundaryEvidence = nil
	b, _ = json.Marshal(s)
	r, err := EvaluateLifecycleSession(b)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "ABSTAIN" || r.Cells[2].Status != "UNKNOWN" {
		t.Fatalf("report=%+v", r)
	}
}

func TestLifecycleSessionRejectsFalseRecoveryAndUnequalOutcome(t *testing.T) {
	b, _ := os.ReadFile("testdata/issue8678-lifecycle-session.json")
	var s LifecycleSession
	_ = json.Unmarshal(b, &s)
	zero := int64(0)
	s.Cells[3].ProviderCacheReadTokens = &zero
	mut, _ := json.Marshal(s)
	if _, err := EvaluateLifecycleSession(mut); err == nil {
		t.Fatal("accepted cache recovery without provider reads")
	}
	_ = json.Unmarshal(b, &s)
	s.Cells[6].OutcomeDigest = "different"
	mut, _ = json.Marshal(s)
	if _, err := EvaluateLifecycleSession(mut); err == nil {
		t.Fatal("accepted unequal outcome")
	}
}
