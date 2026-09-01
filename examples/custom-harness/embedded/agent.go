package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// eventRunner is the narrow boundary to a generated fak harness. A real
// adapter can stream these same semantic envelopes from its chosen transport.
type eventRunner interface {
	Run(context.Context, string) ([]harnesskit.Envelope, error)
}

// harnessAgent translates harness events into the one value the host needs.
type harnessAgent struct {
	runner eventRunner
}

func (a harnessAgent) Suggest(ctx context.Context, ticket Ticket) (string, error) {
	prompt := "Draft one helpful sentence about " + ticket.Topic + "."
	events, err := a.runner.Run(ctx, prompt)
	if err != nil {
		return "", err
	}

	var runID string
	var lastSequence uint64
	var text strings.Builder
	completed := false
	for _, event := range events {
		if err := event.Validate(); err != nil {
			return "", fmt.Errorf("invalid harness event: %w", err)
		}
		if runID == "" {
			runID = event.RunID
		}
		if event.RunID != runID || event.Sequence <= lastSequence {
			return "", errors.New("harness events are not one ordered run")
		}
		lastSequence = event.Sequence

		switch event.Type {
		case harnesskit.EventMessageDelta:
			var payload harnesskit.MessagePayload
			if err := event.DecodePayload(&payload); err != nil {
				return "", err
			}
			text.WriteString(payload.Text)
		case harnesskit.EventRunCompleted:
			completed = true
		}
	}
	if !completed || text.Len() == 0 {
		return "", errors.New("harness run did not produce a completed suggestion")
	}
	return text.String(), nil
}
