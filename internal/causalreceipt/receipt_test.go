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

func TestMissingCausalIdentity(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Receipt)
	}{
		{"wrong schema", func(r *Receipt) { r.Schema = "wrong" }},
		{"missing work", func(r *Receipt) { r.IDs.Work = "" }},
		{"missing turn", func(r *Receipt) { r.IDs.Turn = "" }},
		{"missing graph", func(r *Receipt) { r.IDs.Graph = "" }},
		{"missing request", func(r *Receipt) { r.IDs.Request = "" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := fixture("completed")
			tc.mutate(&r)
			if err := Validate(r); !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected ErrInvalid, got %v", err)
			}
			if _, err := DeriveMetrics(r); !errors.Is(err, ErrInvalid) {
				t.Fatalf("DeriveMetrics: expected ErrInvalid, got %v", err)
			}
		})
	}
}

func TestNativeEngineIntegrity(t *testing.T) {
	r := fixture("completed")
	r.Phases[0].Engine = "llama-native" // ambiguous native engine
	if err := Validate(r); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid for ambiguous native engine, got %v", err)
	}

	r = fixture("completed")
	r.Phases[0].Engine = "" // empty engine
	if err := Validate(r); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid for empty engine, got %v", err)
	}
}

func TestDecisionValidation(t *testing.T) {
	r := fixture("completed")
	r.Decisions = append(r.Decisions, Decision{ID: "", Kind: "policy", Actual: "denied"})
	if err := Validate(r); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid for empty decision ID, got %v", err)
	}

	r = fixture("completed")
	r.Decisions = append(r.Decisions, Decision{ID: "d-1", Kind: "", Actual: "denied"})
	if err := Validate(r); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid for empty decision Kind, got %v", err)
	}

	r = fixture("completed")
	r.Decisions = append(r.Decisions, Decision{ID: "d-1", Kind: "policy", Actual: ""})
	if err := Validate(r); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid for empty decision Actual, got %v", err)
	}
}

func TestSensitiveAttributePatterns(t *testing.T) {
	sensitiveKeys := []string{
		"prompt", "Prompt", "PROMPT_TEXT",
		"output", "turn_output",
		"argument", "tool_argument",
		"result", "call_result",
		"content", "body_content",
		"screenshot", "screenShot_png",
		"filepath", "file_path", "FILE-PATH",
	}
	for _, k := range sensitiveKeys {
		r := fixture("completed")
		r.Attributes = map[string]string{k: "value"}
		if err := Validate(r); !errors.Is(err, ErrInvalid) {
			t.Fatalf("expected ErrInvalid for sensitive key %q, got %v", k, err)
		}
	}
}

func TestIncidentAnswersAndMetricLabels(t *testing.T) {
	r := fixture("completed")
	r.Phases[1].TransferNS = 100
	r.Resources = append(r.Resources, Resource{
		ID:       "old-kv",
		Kind:     "kv_cache",
		State:    "evicted",
		Released: true,
	})
	answers := IncidentAnswers(r)
	wantSubstrings := []string{"evicted:old-kv", "reason:policy_block", "reload:serve", "transfer:serve"}
	if len(answers) != len(wantSubstrings) {
		t.Fatalf("expected %d answers, got %d: %v", len(wantSubstrings), len(answers), answers)
	}
	for i, want := range wantSubstrings {
		if answers[i] != want {
			t.Errorf("answer[%d] = %q, want %q", i, answers[i], want)
		}
	}

	// Test empty phases for MetricLabels
	emptyR := Receipt{Schema: Schema}
	labels := MetricLabels(emptyR)
	if len(labels) != 1 || labels["schema"] != Schema {
		t.Fatalf("unexpected labels for empty receipt: %v", labels)
	}
}
