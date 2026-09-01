package main

import (
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func main() {
	store := NewStore()
	consumer := NewConsumer(store)
	events := demoEvents()

	first, err := consumer.Consume(events, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("first batch: committed=%d cursor=%d\n", first, store.Cursor.Sequence)

	// A disconnect redelivers the full transport batch. The semantic cursor makes
	// committed records no-ops and resumes with the first sequence after it.
	second, err := consumer.Consume(events, len(events))
	if err != nil {
		panic(err)
	}
	fmt.Printf("redelivery: observed=%d cursor=%d messages=%v tool_effects=%d redacted=%d ignored=%d\n",
		second, store.Cursor.Sequence, messageIDs(store.Projection), len(store.Projection.ToolEffects),
		store.Projection.RedactedEvents, store.Projection.IgnoredEvents)
}

func demoEvents() []harnesskit.Envelope {
	return []harnesskit.Envelope{
		envelope(1, "evt-1", harnesskit.EventRunStarted, harnesskit.SensitivityPublic, harnesskit.RunPayload{Status: "running"}),
		envelope(2, "evt-2", harnesskit.EventMessageCompleted, harnesskit.SensitivityPublic, harnesskit.MessagePayload{MessageID: "m-1", Role: "assistant", Text: "ready"}),
		envelope(3, "evt-3", harnesskit.EventToolCompleted, harnesskit.SensitivityPublic, harnesskit.ToolPayload{CallID: "call-1", Name: "book", Status: "completed"}),
		envelope(4, "evt-4", harnesskit.EventType("builder.future"), harnesskit.SensitivityPublic, map[string]string{"additive": "field"}),
		envelope(5, "evt-5", harnesskit.EventMessageCompleted, harnesskit.SensitivitySecret, harnesskit.MessagePayload{MessageID: "secret", Text: "must not persist"}),
		envelope(6, "evt-6", harnesskit.EventRunCompleted, harnesskit.SensitivityPublic, harnesskit.RunPayload{Status: "completed"}),
	}
}

func envelope(sequence uint64, eventID string, eventType harnesskit.EventType, sensitivity harnesskit.Sensitivity, payload any) harnesskit.Envelope {
	encoded, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	return harnesskit.Envelope{
		Version: harnesskit.ProtocolVersion, RunID: "run-demo", Sequence: sequence,
		EventID: eventID, Type: eventType, Sensitivity: sensitivity, Payload: encoded,
	}
}
