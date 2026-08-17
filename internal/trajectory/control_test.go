package trajectory

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestApplyViewControlRoundTripsCanonicalIntervention(t *testing.T) {
	at := time.Date(2026, 8, 17, 15, 0, 0, 0, time.UTC)
	history := []Event{derivedFixture("approval-1", EventApproval, "requested", at, 7, `{"tool":"deploy"}`)}
	request := ControlRequest{Action: "approve", ActorID: "operator-7", TargetEventID: "approval-1", Payload: json.RawMessage(`{"reason":"reviewed"}`), Timestamp: at.Add(time.Second)}

	event, receipt, err := ApplyViewControl(OperatorView(), history, request)
	if err != nil {
		t.Fatal(err)
	}
	again, againReceipt, err := ApplyViewControl(OperatorView(), history, request)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(event, again) || !reflect.DeepEqual(receipt, againReceipt) {
		t.Fatal("control round trip is not deterministic")
	}
	if event.Kind != EventIntervention || event.Action != "approve" || event.Sequence != 8 {
		t.Fatalf("bad intervention: %#v", event)
	}
	if !reflect.DeepEqual(event.ParentIDs, []string{"approval-1"}) {
		t.Fatalf("bad causal target: %#v", event.ParentIDs)
	}
	if receipt.Schema != ControlReceiptSchema || receipt.EventDigest == "" || receipt.HistoryDigest == "" || receipt.RequestDigest == "" || receipt.ViewSpecDigest == "" {
		t.Fatalf("incomplete receipt: %#v", receipt)
	}
	if got, _ := event.Digest(); got != receipt.EventDigest {
		t.Fatalf("receipt event digest=%q, want %q", receipt.EventDigest, got)
	}
	if len(history) != 1 || history[0].ID != "approval-1" {
		t.Fatal("control mutated canonical history")
	}
}

func TestApplyViewControlRefusesActionsOutsideView(t *testing.T) {
	at := time.Date(2026, 8, 17, 15, 0, 0, 0, time.UTC)
	history := []Event{derivedFixture("message-1", EventMessage, "completed", at, 1, `{"role":"assistant","text":"ready"}`)}
	for _, request := range []ControlRequest{
		{Action: "approve", ActorID: "user", Timestamp: at.Add(time.Second)},
		{Action: "delete", ActorID: "user", Timestamp: at.Add(time.Second)},
		{Action: "steer", Timestamp: at.Add(time.Second)},
		{Action: "steer", ActorID: "user", TargetEventID: "missing", Timestamp: at.Add(time.Second)},
	} {
		event, receipt, err := ApplyViewControl(EndUserView(), history, request)
		if err == nil {
			t.Fatalf("request %#v unexpectedly accepted", request)
		}
		if event.ID != "" || receipt.Schema != "" {
			t.Fatalf("refusal emitted evidence: event=%#v receipt=%#v", event, receipt)
		}
	}
}

func TestApplyViewControlSupportsClosedVocabulary(t *testing.T) {
	at := time.Date(2026, 8, 17, 15, 0, 0, 0, time.UTC)
	history := []Event{derivedFixture("message-1", EventMessage, "completed", at, 1, `{"role":"assistant","text":"ready"}`)}
	spec := OperatorView()
	spec.LiveControls = []string{"approve", "deny", "steer", "pause", "retry", "branch"}
	for i, action := range spec.LiveControls {
		event, _, err := ApplyViewControl(spec, history, ControlRequest{Action: action, ActorID: "operator", Timestamp: at.Add(time.Duration(i+1) * time.Second)})
		if err != nil {
			t.Fatalf("%s: %v", action, err)
		}
		if event.Action != action {
			t.Fatalf("action=%q, want %q", event.Action, action)
		}
	}
}

func TestApplyViewControlRedactsBeforeCanonicalEvent(t *testing.T) {
	at := time.Date(2026, 8, 17, 15, 0, 0, 0, time.UTC)
	history := []Event{derivedFixture("message-1", EventMessage, "completed", at, 1, `{"role":"assistant","text":"ready"}`)}
	event, _, err := ApplyViewControl(EndUserView(), history, ControlRequest{Action: "steer", ActorID: "user", Payload: json.RawMessage(`{"text":"continue","api_key":"top-secret"}`), Timestamp: at.Add(time.Second)})
	if err != nil {
		t.Fatal(err)
	}
	if string(event.Payload) == "" || containsJSONText(event.Payload, "top-secret") {
		t.Fatalf("secret reached canonical intervention: %s", event.Payload)
	}
	if !containsJSONText(event.Payload, "[REDACTED]") {
		t.Fatalf("redaction marker missing: %s", event.Payload)
	}
}

func containsJSONText(raw json.RawMessage, wanted string) bool {
	return strings.Contains(string(raw), wanted)
}
