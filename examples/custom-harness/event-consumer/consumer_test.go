package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestReplaySafeConsumer(t *testing.T) {
	store := NewStore()
	consumer := NewConsumer(store)
	events := demoEvents()

	committed, err := consumer.Consume(events, 2)
	if err != nil {
		t.Fatal(err)
	}
	if committed != 2 || store.Cursor.Sequence != 2 {
		t.Fatalf("bounded batch = (%d, %d), want (2, 2)", committed, store.Cursor.Sequence)
	}

	// Simulate disconnect: the broker redelivers the entire original batch.
	committed, err = consumer.Consume(events, len(events))
	if err != nil {
		t.Fatal(err)
	}
	if committed != len(events) || store.Cursor.Sequence != 6 {
		t.Fatalf("resume = (%d, %d), want (%d, 6)", committed, store.Cursor.Sequence, len(events))
	}
	if got := store.Projection.Messages["m-1"]; got != "ready" {
		t.Fatalf("message projection = %q", got)
	}
	if _, leaked := store.Projection.Messages["secret"]; leaked {
		t.Fatal("sensitive payload was persisted")
	}
	if store.Projection.RedactedEvents != 1 || store.Projection.IgnoredEvents != 1 {
		t.Fatalf("redacted/ignored = %d/%d", store.Projection.RedactedEvents, store.Projection.IgnoredEvents)
	}
	if store.Projection.ToolEffects["call-1"] != 1 {
		t.Fatalf("tool effect count = %d, want 1", store.Projection.ToolEffects["call-1"])
	}
	if store.Projection.RunStatus != "completed" {
		t.Fatalf("run status = %q", store.Projection.RunStatus)
	}

	// A second redelivery cannot duplicate a committed domain effect.
	if _, err := consumer.Consume(events, len(events)); err != nil {
		t.Fatal(err)
	}
	if store.Projection.ToolEffects["call-1"] != 1 || store.Projection.RedactedEvents != 1 {
		t.Fatalf("redelivery changed projection: %+v", store.Projection)
	}
}

func TestProjectionFailureDoesNotAdvanceCursor(t *testing.T) {
	store := NewStore()
	consumer := NewConsumer(store)
	bad := envelope(1, "evt-bad", harnesskit.EventMessageCompleted, harnesskit.SensitivityPublic, map[string]any{"message_id": 7})
	if _, err := consumer.Consume([]harnesskit.Envelope{bad}, 1); err == nil {
		t.Fatal("expected payload decode failure")
	}
	if store.Cursor.Sequence != 0 || len(store.Projection.Messages) != 0 {
		t.Fatalf("partial commit: cursor=%d projection=%+v", store.Cursor.Sequence, store.Projection)
	}
}

func TestInputIDIsIdempotent(t *testing.T) {
	store := NewStore()
	store.Cursor = harnesskit.Cursor{Version: harnesskit.ProtocolVersion, RunID: "run-demo", Sequence: 1}
	consumer := NewConsumer(store)
	input := harnesskit.Input{
		Version: harnesskit.ProtocolVersion, RunID: "run-demo", InputID: "input-1",
		Type: harnesskit.InputMessage, Message: &harnesskit.MessageInput{Text: "continue"},
	}
	first, err := consumer.AcceptInput(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := consumer.AcceptInput(input)
	if err != nil {
		t.Fatal(err)
	}
	if !first || second || len(store.Projection.AcceptedInputs) != 1 {
		t.Fatalf("accepted first/second/count = %v/%v/%d", first, second, len(store.Projection.AcceptedInputs))
	}
}

func TestEventIDIsIdempotentAcrossNewSequence(t *testing.T) {
	store := NewStore()
	consumer := NewConsumer(store)
	original := envelope(1, "evt-tool", harnesskit.EventToolCompleted, harnesskit.SensitivityPublic, harnesskit.ToolPayload{CallID: "call-1", Name: "book", Status: "completed"})
	duplicate := envelope(2, "evt-tool", harnesskit.EventToolCompleted, harnesskit.SensitivityPublic, harnesskit.ToolPayload{CallID: "call-1", Name: "book", Status: "completed"})
	if _, err := consumer.Consume([]harnesskit.Envelope{original, duplicate}, 2); err != nil {
		t.Fatal(err)
	}
	if store.Cursor.Sequence != 2 || store.Projection.ToolEffects["call-1"] != 1 || len(store.Projection.AppliedEvents) != 1 {
		t.Fatalf("event idempotency failed: cursor=%d projection=%+v", store.Cursor.Sequence, store.Projection)
	}
}
