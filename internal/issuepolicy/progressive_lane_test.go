package issuepolicy

import (
	"strings"
	"testing"
)

func TestProgressiveLane_NaturalLanguageTitle(t *testing.T) {
	// Natural language title without paths or regex
	title := "Fix race condition in gateway request multiplexer"
	res := InferLaneProgressive(title, "", nil)
	if res.Lane != "gateway" && res.Lane != "provisional:gateway" {
		t.Fatalf("res.Lane = %q, want 'gateway' or 'provisional:gateway'", res.Lane)
	}
	if res.Confidence < 0.80 {
		t.Fatalf("res.Confidence = %.2f, want >= 0.80", res.Confidence)
	}

	// Verify dispatchability through ReviewCandidate
	c := Candidate{
		Key:            "FAK-12015",
		Title:          title,
		ParentRef:      "#12000",
		CurrentState:   "Gateway multiplexer has a data race under concurrent streams.",
		WhyNow:         "Hangs active client connections.",
		WorkingSpine:   "gateway -> request multiplexer -> thread safety",
		InScope:        "Fix race in request multiplexer.",
		OutOfScope:     "Do not change public API.",
		DoneCondition:  "Multiplexer tests pass under race detector.",
		Witness:        "go test -race ./internal/gateway/...",
		AcceptanceGate: "green race test",
		ClosureBinding: "commit",
		ProblemFrame:   completeProblemFrame(),
	}
	review := ReviewCandidate(c, Options{})
	if review.Dispatchability != Dispatchable {
		t.Fatalf("review.Dispatchability = %q, want %q (verdict: %q, reasons: %v)", review.Dispatchability, Dispatchable, review.Verdict, review.Reasons)
	}
	if review.Lane != "gateway" && review.Lane != "provisional:gateway" {
		t.Fatalf("review.Lane = %q, want 'gateway' or 'provisional:gateway'", review.Lane)
	}
	if review.LaneConfidence < 0.80 {
		t.Fatalf("review.LaneConfidence = %.2f, want >= 0.80", review.LaneConfidence)
	}
}

func TestProgressiveLane_HighConfidenceAutoAssignment(t *testing.T) {
	title := "Fix memory leak in kvmmu cache manager when evicting blocks"
	body := "The kvmmu block manager fails to free allocated page tables during cache eviction."
	res := InferLaneProgressive(title, body, nil)
	if res.Confidence < 0.85 {
		t.Fatalf("expected confidence >= 0.85, got %.2f", res.Confidence)
	}
	if res.Provisional {
		t.Fatalf("expected canonical auto-assignment (Provisional=false), got Provisional=true")
	}
	if res.Lane != "kvmmu" {
		t.Fatalf("expected lane 'kvmmu', got %q", res.Lane)
	}
}

func TestProgressiveLane_ModerateConfidenceProvisional(t *testing.T) {
	title := "Improve cache metadata lookup performance"
	body := "Indexing optimization for entries"
	res := InferLaneProgressive(title, body, nil)
	if res.Confidence < 0.40 || res.Confidence >= 0.85 {
		t.Fatalf("expected confidence in [0.40, 0.85), got %.2f", res.Confidence)
	}
	if !res.Provisional {
		t.Fatalf("expected Provisional=true, got false")
	}
	if res.Lane != "provisional:cachemeta" {
		t.Fatalf("expected lane 'provisional:cachemeta', got %q", res.Lane)
	}
}

func TestProgressiveLane_LowConfidenceSuggestions(t *testing.T) {
	title := "Update contributor guide and formatting style"
	body := "General documentation update for welcoming new developers."
	res := InferLaneProgressive(title, body, nil)
	if res.Confidence >= 0.40 {
		t.Fatalf("expected confidence < 0.40, got %.2f", res.Confidence)
	}
	if res.Lane != "" {
		t.Fatalf("expected empty lane, got %q", res.Lane)
	}
	if res.Provisional {
		t.Fatalf("expected Provisional=false, got true")
	}
	if len(res.SuggestedLanes) < 3 {
		t.Fatalf("expected at least 3 suggested lanes, got %v", res.SuggestedLanes)
	}
}

func TestProgressiveLane_PreserveExactMatching(t *testing.T) {
	// 1. Regex scope in title
	res1 := InferLaneProgressive("fix(gateway): handle multiplexer reset", "", nil)
	if res1.Lane != "gateway" || res1.Provisional || res1.Confidence != 1.0 {
		t.Fatalf("res1 = %+v, want lane 'gateway', provisional false, conf 1.0", res1)
	}

	// 2. Bracket scope in title
	res2 := InferLaneProgressive("[kvmmu] fix block leak", "", nil)
	if res2.Lane != "kvmmu" || res2.Provisional || res2.Confidence != 1.0 {
		t.Fatalf("res2 = %+v, want lane 'kvmmu', provisional false, conf 1.0", res2)
	}

	// 3. Path prefix matching
	res3 := InferLaneProgressive("Fix race condition", "", []string{"internal/adjudicator/decide.go"})
	if res3.Lane != "adjudicator" || res3.Provisional || res3.Confidence != 1.0 {
		t.Fatalf("res3 = %+v, want lane 'adjudicator', provisional false, conf 1.0", res3)
	}
}

func TestProgressiveLane_ReviewCandidateProvisionalAdmitted(t *testing.T) {
	c := Candidate{
		IssueNumber:    12015,
		Key:            "FAK-12015",
		Title:          "Improve cache metadata lookup performance",
		Body:           "Indexing optimization for entries",
		ParentRef:      "#12000",
		CurrentState:   "Cache entries lookup overhead is high.",
		WhyNow:         "Bottleneck in hot path.",
		WorkingSpine:   "cachemeta -> indexing -> lookup speed",
		InScope:        "Optimize metadata entries indexing.",
		OutOfScope:     "No changes to storage engine.",
		DoneCondition:  "Lookup benchmarks improve by 20%.",
		Witness:        "go test -v ./internal/cachemeta/...",
		AcceptanceGate: "green benchmark",
		ClosureBinding: "commit",
		ProblemFrame:   completeProblemFrame(),
	}
	review := ReviewCandidate(c, Options{})
	if review.Dispatchability != Dispatchable {
		t.Fatalf("review.Dispatchability = %q, want %q (verdict: %q, reasons: %v)", review.Dispatchability, Dispatchable, review.Verdict, review.Reasons)
	}
	if !review.ProvisionalLane {
		t.Fatalf("expected ProvisionalLane=true")
	}
	if !strings.HasPrefix(review.Lane, "provisional:") {
		t.Fatalf("expected lane to start with 'provisional:', got %q", review.Lane)
	}
	// Check advisory note in Coordination
	foundAdvisory := false
	for _, note := range review.Coordination {
		if strings.Contains(note, "[advisory]") && strings.Contains(note, "assigned provisional lane") {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Fatalf("expected advisory note in Coordination, got: %v", review.Coordination)
	}
}
