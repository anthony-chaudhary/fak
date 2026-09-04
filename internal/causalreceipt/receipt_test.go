package causalreceipt

import (
	"errors"
	"testing"
	"time"
)

func fixture(outcome string) Receipt {
	now := time.Unix(1, 0)
	return Receipt{Schema: Schema, IDs: IDs{Work: "work-1", Turn: "turn-1", Graph: "graph-1", Request: "req-1", ModelSession: "session-1"}, Phases: []Phase{{ID: "agent", Kind: "agent", Engine: "fak-native", Backend: "offline", Outcome: "completed", Started: now, Ended: now.Add(time.Millisecond), OperationIDs: []string{"tool-1"}}, {ID: "serve", ParentID: "agent", Kind: "model", Engine: "fak-native", Backend: "metal", Outcome: outcome, Reason: "policy_block", Tokens: 12, Bytes: 100, CacheReuseBytes: 40, QueueNS: 2, LoadNS: 3, VerificationNS: 5, ResourceIDs: []string{"weights"}}}, Resources: []Resource{{ID: "weights", Kind: "model_weights", State: "released", PlannedLocality: "device", ActualLocality: "device", Bytes: 100, Released: true}}, Decisions: []Decision{{Kind: "route", ID: "route-1", Reason: "resident", Planned: "metal", Actual: "metal"}, {Kind: "policy", ID: "policy-1", Reason: "policy_block", Planned: "evaluate", Actual: "denied"}}, ModuleVersions: map[string]string{"internal/modelroute": "r1+gabc"}}
}
func TestOfflineAndServedShareCausalVocabulary(t *testing.T) {
	for _, outcome := range []string{"completed", "denied"} {
		r := fixture(outcome)
		m, err := DeriveMetrics(r)
		if err != nil {
			t.Fatal(err)
		}
		if m.Tokens != 12 || m.Bytes != 100 || m.CacheReuseBytes != 40 || m.OverheadNS != 10 {
			t.Fatalf("metrics=%+v", m)
		}
		labels := MetricLabels(r)
		if len(labels) != 3 {
			t.Fatalf("high-cardinality labels: %#v", labels)
		}
		answers := IncidentAnswers(r)
		if len(answers) == 0 {
			t.Fatal("missing incident answers")
		}
	}
}
func TestRejectsContentAndOrphans(t *testing.T) {
	r := fixture("completed")
	r.Attributes = map[string]string{"prompt_text": "secret"}
	if !errors.Is(Validate(r), ErrInvalid) {
		t.Fatal("content admitted")
	}
	r = fixture("completed")
	r.Phases[1].ParentID = "missing"
	if !errors.Is(Validate(r), ErrInvalid) {
		t.Fatal("orphan admitted")
	}
	r = fixture("completed")
	r.Resources[0].Released = false
	if !errors.Is(Validate(r), ErrInvalid) {
		t.Fatal("leaked lifecycle admitted")
	}
}

func BenchmarkValidate(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(r); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSummarize(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Summarize(r); err != nil {
			b.Fatal(err)
		}
	}
}
