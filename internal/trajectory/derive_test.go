package trajectory

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestDeriveSignalsBindsPhasesRetriesStallsAndOutcomes(t *testing.T) {
	base := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	events := []Event{
		derivedFixture("phase-start", EventRunLifecycle, "started", base, 1, `{"phase":"research"}`),
		derivedFixture("tool-fail", EventTool, "failed", base.Add(time.Second), 2, `{"tool":"search"}`),
		derivedFixture("tool-retry", EventTool, "started", base.Add(2*time.Second), 3, `{"tool":"search"}`),
		derivedFixture("phase-end", EventRunLifecycle, "completed", base.Add(12*time.Second), 4, `{"phase":"research"}`),
		derivedFixture("outcome", EventOutcome, "completed", base.Add(13*time.Second), 5, `{"status":"success"}`),
	}
	original, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}

	first, err := DeriveSignals(events, DeriveOptions{StallThreshold: 5 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveSignals(events, DeriveOptions{StallThreshold: 5 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("derivation is not deterministic:\n%#v\n%#v", first, second)
	}
	if err := first.Validate(); err != nil {
		t.Fatal(err)
	}
	if first.InputDigest == "" || first.OutputDigest == "" || first.InputDigest == first.OutputDigest {
		t.Fatalf("bad receipt digests: %#v", first)
	}
	if got := countDerived(first.Signals, DerivedPhase); got != 1 {
		t.Fatalf("phases=%d, want 1: %#v", got, first.Signals)
	}
	if got := countDerived(first.Signals, DerivedRetry); got != 1 {
		t.Fatalf("retries=%d, want 1: %#v", got, first.Signals)
	}
	if got := countDerived(first.Signals, DerivedStall); got != 1 {
		t.Fatalf("stalls=%d, want 1: %#v", got, first.Signals)
	}
	if got := countDerived(first.Signals, DerivedOutcome); got != 1 {
		t.Fatalf("outcomes=%d, want 1: %#v", got, first.Signals)
	}
	for _, signal := range first.Signals {
		if len(signal.SourceEventIDs) == 0 || signal.Rule == "" || signal.ConfidenceBasis == "" {
			t.Fatalf("signal lacks provenance: %#v", signal)
		}
	}
	after, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	if string(original) != string(after) {
		t.Fatal("derivation mutated canonical events")
	}
}

func TestDeriveSignalsLeavesUnsupportedEvidenceExplicit(t *testing.T) {
	event := derivedFixture("message", EventMessage, "completed", time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC), 1, `{"role":"assistant","text":"done"}`)
	receipt, err := DeriveSignals([]Event{event}, DeriveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got := receipt.UnmatchedKinds[EventMessage]; got != 1 {
		t.Fatalf("unsupported message count=%d, want 1", got)
	}
	if len(receipt.Signals) != 0 {
		t.Fatalf("unsupported evidence became a signal: %#v", receipt.Signals)
	}
}

func TestDeriveSignalsKeepsOpenPhaseBoundedToObservedWindow(t *testing.T) {
	event := derivedFixture("phase-start", EventRunLifecycle, "started", time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC), 1, `{"phase":"execution"}`)
	receipt, err := DeriveSignals([]Event{event}, DeriveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.Signals) != 1 || receipt.Signals[0].End != nil || receipt.Signals[0].Confidence != "medium" {
		t.Fatalf("open phase not explicit: %#v", receipt.Signals)
	}
}

func derivedFixture(id string, kind EventKind, action string, at time.Time, sequence uint64, payload string) Event {
	return Event{Schema: EventSchema, ID: id, ConversationID: "conversation-1", Kind: kind, Action: action, Timestamp: at, Sequence: sequence, Visibility: VisibilityOperator, Source: EventSource{Type: "fixture", Adapter: "test", AdapterVersion: "1"}, Payload: json.RawMessage(payload)}
}

func countDerived(signals []DerivedSignal, kind DerivedKind) int {
	count := 0
	for _, signal := range signals {
		if signal.Kind == kind {
			count++
		}
	}
	return count
}

func TestDeriveSignalsCoversExplicitSemanticEventsAndCausalLinks(t *testing.T) {
	base := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	checkpoint := derivedFixture("goal", EventCheckpoint, "saved", base, 1, `{"goal":"ship parser","progress":"working"}`)
	decision := derivedFixture("decision", EventIntervention, "approve", base.Add(time.Second), 2, `{"reason":"operator approved"}`)
	decision.ParentIDs = []string{"goal"}
	artifact := derivedFixture("artifact", EventArtifact, "created", base.Add(2*time.Second), 3, `{"path":"report.json"}`)
	cost := derivedFixture("cost", EventObservation, "usage", base.Add(3*time.Second), 4, `{"input_tokens":12,"output_tokens":4}`)
	receipt, err := DeriveSignals([]Event{checkpoint, decision, artifact, cost}, DeriveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []DerivedKind{DerivedGoal, DerivedControlAction, DerivedArtifact, DerivedCost, DerivedCausalLink} {
		if got := countDerived(receipt.Signals, kind); got != 1 {
			t.Fatalf("%s count=%d, want 1: %#v", kind, got, receipt.Signals)
		}
	}
	if len(receipt.UnmatchedKinds) != 0 {
		t.Fatalf("unexpected unsupported evidence: %#v", receipt.UnmatchedKinds)
	}
}

func TestDerivationReceiptRejectsSignalTampering(t *testing.T) {
	event := derivedFixture("outcome", EventOutcome, "completed", time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC), 1, `{"status":"success"}`)
	receipt, err := DeriveSignals([]Event{event}, DeriveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	receipt.Signals[0].Label = "tampered"
	if err := receipt.Validate(); err == nil {
		t.Fatal("tampered receipt validated")
	}
}
