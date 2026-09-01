package main

import (
	"context"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// offlineRunner keeps the learning example deterministic and runnable without
// a provider key, network, model, or GPU. Replace only this type to connect the
// adapter to a generated harness runtime.
type offlineRunner struct{}

func (offlineRunner) Run(ctx context.Context, prompt string) ([]harnesskit.Envelope, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	text := "You can update the address before the package ships."
	payload, err := json.Marshal(harnesskit.MessagePayload{
		MessageID: "reply-1",
		Role:      "assistant",
		Text:      text,
	})
	if err != nil {
		return nil, err
	}
	completed, err := json.Marshal(harnesskit.RunPayload{Status: "completed"})
	if err != nil {
		return nil, err
	}

	return []harnesskit.Envelope{
		{
			Version: harnesskit.ProtocolVersion, RunID: "embedded-demo", Sequence: 1,
			EventID: "event-1", Type: harnesskit.EventRunStarted,
		},
		{
			Version: harnesskit.ProtocolVersion, RunID: "embedded-demo", Sequence: 2,
			EventID: "event-2", Type: harnesskit.EventMessageDelta, Payload: payload,
		},
		{
			Version: harnesskit.ProtocolVersion, RunID: "embedded-demo", Sequence: 3,
			EventID: "event-3", Type: harnesskit.EventRunCompleted, Payload: completed,
		},
	}, nil
}
