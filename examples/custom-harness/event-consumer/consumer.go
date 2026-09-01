package main

import (
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// Projection is the builder-owned domain view derived from semantic events.
type Projection struct {
	Messages       map[string]string
	ToolEffects    map[string]int
	RunStatus      string
	RedactedEvents int
	IgnoredEvents  int
	AcceptedInputs map[string]int
	AppliedEvents  map[string]int
}

// Store models one atomic persistence boundary for the projection and semantic cursor.
// A production adapter should implement the same transaction in its database.
type Store struct {
	Projection Projection
	Cursor     harnesskit.Cursor
}

func NewStore() *Store {
	return &Store{Projection: Projection{
		Messages:       make(map[string]string),
		ToolEffects:    make(map[string]int),
		AcceptedInputs: make(map[string]int),
		AppliedEvents:  make(map[string]int),
	}}
}

// Consumer applies at most credit records. The returned count is transport-facing:
// a broker may acknowledge those messages only after Consume returns successfully.
type Consumer struct{ store *Store }

func NewConsumer(store *Store) *Consumer { return &Consumer{store: store} }

func (c *Consumer) Consume(events []harnesskit.Envelope, credit int) (int, error) {
	if credit < 0 {
		return 0, errors.New("credit must not be negative")
	}
	committed := 0
	for _, event := range events {
		if committed == credit {
			break
		}
		// A committed cursor is exclusive: records at or below it are redelivery.
		if event.RunID == c.store.Cursor.RunID && event.Sequence <= c.store.Cursor.Sequence {
			committed++
			continue
		}
		if err := event.Validate(); err != nil {
			return committed, err
		}
		if c.store.Cursor.RunID != "" && event.RunID != c.store.Cursor.RunID {
			return committed, fmt.Errorf("run changed from %q to %q", c.store.Cursor.RunID, event.RunID)
		}

		// Clone, project, then commit projection and cursor together. A failure before
		// this assignment leaves both unchanged and makes redelivery safe.
		next := cloneProjection(c.store.Projection)
		if err := project(&next, event); err != nil {
			return committed, err
		}
		c.store.Projection = next
		c.store.Cursor = harnesskit.Cursor{
			Version:  harnesskit.ProtocolVersion,
			RunID:    event.RunID,
			Sequence: event.Sequence,
		}
		committed++
	}
	return committed, nil
}

// AcceptInput demonstrates input-id idempotency independently from event replay.
func (c *Consumer) AcceptInput(input harnesskit.Input) (bool, error) {
	if err := input.Validate(); err != nil {
		return false, err
	}
	if c.store.Cursor.RunID != "" && input.RunID != c.store.Cursor.RunID {
		return false, fmt.Errorf("input run %q does not match %q", input.RunID, c.store.Cursor.RunID)
	}
	if _, exists := c.store.Projection.AcceptedInputs[input.InputID]; exists {
		return false, nil
	}
	next := cloneProjection(c.store.Projection)
	next.AcceptedInputs[input.InputID] = 1
	c.store.Projection = next
	return true, nil
}

func project(dst *Projection, event harnesskit.Envelope) error {
	if _, applied := dst.AppliedEvents[event.EventID]; applied {
		return nil
	}
	if event.Sensitivity == harnesskit.SensitivityPrivate || event.Sensitivity == harnesskit.SensitivitySecret {
		dst.RedactedEvents++
		dst.AppliedEvents[event.EventID] = 1
		return nil
	}
	if !event.Known() {
		dst.IgnoredEvents++
		dst.AppliedEvents[event.EventID] = 1
		return nil
	}
	switch event.Type {
	case harnesskit.EventMessageCompleted:
		var payload harnesskit.MessagePayload
		if err := event.DecodePayload(&payload); err != nil {
			return err
		}
		dst.Messages[payload.MessageID] = payload.Text
	case harnesskit.EventToolCompleted:
		var payload harnesskit.ToolPayload
		if err := event.DecodePayload(&payload); err != nil {
			return err
		}
		// CallID is the domain-effect idempotency key. Cursor replay protection is
		// necessary but not sufficient when effects span external systems.
		if _, done := dst.ToolEffects[payload.CallID]; !done {
			dst.ToolEffects[payload.CallID] = 1
		}
	case harnesskit.EventRunStarted:
		dst.RunStatus = "running"
	case harnesskit.EventRunCompleted:
		dst.RunStatus = "completed"
	}
	dst.AppliedEvents[event.EventID] = 1
	return nil
}

func cloneProjection(src Projection) Projection {
	dst := src
	dst.Messages = cloneMap(src.Messages)
	dst.ToolEffects = cloneMap(src.ToolEffects)
	dst.AcceptedInputs = cloneMap(src.AcceptedInputs)
	dst.AppliedEvents = cloneMap(src.AppliedEvents)
	return dst
}

func cloneMap[K comparable, V any](src map[K]V) map[K]V {
	dst := make(map[K]V, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func messageIDs(p Projection) []string {
	ids := make([]string, 0, len(p.Messages))
	for id := range p.Messages {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
