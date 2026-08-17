package trajectory

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestEventRoundTripDigestAndLoss(t *testing.T) {
	ts := time.Date(2026, 8, 17, 16, 0, 0, 0, time.UTC)
	events := []Event{{
		Schema: EventSchema, ID: "evt-1", ConversationID: "conv-1", Kind: EventTool,
		Action: "completed", Timestamp: ts, Sequence: 2, ParentIDs: []string{"evt-0"},
		Visibility: VisibilityDeveloper,
		Source:     EventSource{Type: "codex-jsonl", SessionID: "s1", EventID: "native-9", OrderingKey: "9", RawDigest: "sha256:raw", Adapter: "codex", AdapterVersion: "1"},
		Payload:    json.RawMessage(`{"name":"shell","ok":true}`),
		Loss:       &LossReport{UnknownFields: []string{"future_field"}, OmittedBytes: 12, Reason: "unsupported attachment"},
	}}

	first, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeEvents(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := EncodeEvents(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("non-deterministic round trip\n%s\n%s", first, second)
	}
	if decoded[0].Loss == nil || decoded[0].Loss.OmittedBytes != 12 {
		t.Fatalf("loss report not preserved: %#v", decoded[0].Loss)
	}
	d1, err := events[0].Digest()
	if err != nil {
		t.Fatal(err)
	}
	d2, err := decoded[0].Digest()
	if err != nil {
		t.Fatal(err)
	}
	if d1 != d2 {
		t.Fatalf("digest changed: %s != %s", d1, d2)
	}
}

func TestEventValidationRejectsUnknownSemantics(t *testing.T) {
	e := validEvent()
	e.Kind = "guess"
	if err := e.Validate(); err == nil || !strings.Contains(err.Error(), "unknown trajectory event kind") {
		t.Fatalf("err=%v", err)
	}
	e = validEvent()
	e.Payload = json.RawMessage(`{"broken"`)
	if err := e.Validate(); err == nil || !strings.Contains(err.Error(), "valid JSON") {
		t.Fatalf("err=%v", err)
	}
}

func TestCompactTurnsKeepsStableShapeAndOrdering(t *testing.T) {
	ts := time.Date(2026, 8, 17, 16, 0, 0, 0, time.UTC)
	events := []Event{
		messageEvent("assistant", "done", "evt-a", "turn-1", ts, 2, []string{"evt-u"}),
		messageEvent("user", "please inspect", "evt-u", "turn-1", ts, 1, nil),
		{Schema: EventSchema, ID: "evt-steer", ConversationID: "conv-1", Kind: EventIntervention, Action: "steer", Timestamp: ts, Sequence: 3, ParentIDs: []string{"evt-a"}, Visibility: VisibilityOperator, Source: testSource(), Payload: json.RawMessage(`{"text":"also test it"}`)},
	}
	turns, err := CompactTurns(events)
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 2 {
		t.Fatalf("turns=%d", len(turns))
	}
	if turns[0].TraceID != "conv-1" || turns[0].Seq != 1 || turns[0].Query != "please inspect" || turns[0].Labels["role"] != "user" {
		t.Fatalf("user projection=%#v", turns[0])
	}
	if turns[1].Seq != 2 || turns[1].Query != "done" || turns[1].Labels["role"] != "assistant" {
		t.Fatalf("assistant projection=%#v", turns[1])
	}
	if len(events[0].ParentIDs) != 1 || events[0].ParentIDs[0] != "evt-u" {
		t.Fatal("projection mutated causal links")
	}
}

func validEvent() Event {
	return Event{Schema: EventSchema, ID: "evt", ConversationID: "conv", Kind: EventRunLifecycle, Action: "started", Timestamp: time.Unix(1, 0).UTC(), Visibility: VisibilityOperator, Source: testSource(), Payload: json.RawMessage(`{}`)}
}

func testSource() EventSource {
	return EventSource{Type: "fixture", Adapter: "fixture", AdapterVersion: "1"}
}

func messageEvent(role, text, id, turnID string, ts time.Time, sequence uint64, parents []string) Event {
	payload, _ := json.Marshal(messageEventPayload{Role: role, Text: text, TurnID: turnID})
	return Event{Schema: EventSchema, ID: id, ConversationID: "conv-1", Kind: EventMessage, Action: "completed", Timestamp: ts, Sequence: sequence, ParentIDs: parents, Visibility: VisibilityPublic, Source: testSource(), Payload: payload}
}
