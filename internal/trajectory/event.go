package trajectory

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

const EventSchema = "fak-trajectory-event/1alpha1"

type EventKind string

const (
	EventRunLifecycle EventKind = "session"
	EventMessage      EventKind = "message"
	EventTool         EventKind = "tool"
	EventApproval     EventKind = "approval"
	EventState        EventKind = "state"
	EventCheckpoint   EventKind = "checkpoint"
	EventArtifact     EventKind = "artifact"
	EventObservation  EventKind = "observation"
	EventOutcome      EventKind = "policy"
	EventError        EventKind = "error"
	EventIntervention EventKind = "intervention"
)

var validEventKinds = map[EventKind]struct{}{
	EventRunLifecycle: {}, EventMessage: {}, EventTool: {}, EventApproval: {}, EventState: {},
	EventCheckpoint: {}, EventArtifact: {}, EventObservation: {}, EventOutcome: {},
	EventError: {}, EventIntervention: {},
}

type Visibility string

const (
	VisibilityPublic     Visibility = "public"
	VisibilityOperator   Visibility = "operator"
	VisibilityDeveloper  Visibility = "developer"
	VisibilityRestricted Visibility = "restricted"
)

var validVisibilities = map[Visibility]struct{}{
	VisibilityPublic: {}, VisibilityOperator: {}, VisibilityDeveloper: {}, VisibilityRestricted: {},
}

// EventSource preserves the native identity needed to audit an adapter's output.
type EventSource struct {
	Type           string `json:"type"`
	SessionID      string `json:"session_id,omitempty"`
	EventID        string `json:"event_id,omitempty"`
	OrderingKey    string `json:"ordering_key,omitempty"`
	RawDigest      string `json:"raw_digest,omitempty"`
	Adapter        string `json:"adapter"`
	AdapterVersion string `json:"adapter_version"`
}

// LossReport makes adapter omissions explicit instead of silently discarding source data.
type LossReport struct {
	UnknownFields []string `json:"unknown_fields,omitempty"`
	UnknownKinds  []string `json:"unknown_kinds,omitempty"`
	OmittedBytes  int      `json:"omitted_bytes,omitempty"`
	Reason        string   `json:"reason,omitempty"`
}

// Event is the provenance-preserving record beside the compact Turn projection.
type Event struct {
	Schema         string          `json:"schema"`
	ID             string          `json:"id"`
	ConversationID string          `json:"conversation_id"`
	Kind           EventKind       `json:"kind"`
	Action         string          `json:"action"`
	Timestamp      time.Time       `json:"timestamp"`
	Sequence       uint64          `json:"sequence"`
	ParentIDs      []string        `json:"parent_ids,omitempty"`
	Visibility     Visibility      `json:"visibility"`
	Source         EventSource     `json:"source"`
	Payload        json.RawMessage `json:"payload"`
	Loss           *LossReport     `json:"loss,omitempty"`
}

func (e Event) Validate() error {
	if e.Schema != EventSchema {
		return fmt.Errorf("trajectory event schema %q, want %q", e.Schema, EventSchema)
	}
	if strings.TrimSpace(e.ID) == "" || strings.TrimSpace(e.ConversationID) == "" {
		return errors.New("trajectory event requires id and conversation_id")
	}
	if _, ok := validEventKinds[e.Kind]; !ok {
		return fmt.Errorf("unknown trajectory event kind %q", e.Kind)
	}
	if strings.TrimSpace(e.Action) == "" {
		return errors.New("trajectory event requires action")
	}
	if e.Timestamp.IsZero() {
		return errors.New("trajectory event requires timestamp")
	}
	if _, ok := validVisibilities[e.Visibility]; !ok {
		return fmt.Errorf("unknown trajectory event visibility %q", e.Visibility)
	}
	if strings.TrimSpace(e.Source.Type) == "" || strings.TrimSpace(e.Source.Adapter) == "" || strings.TrimSpace(e.Source.AdapterVersion) == "" {
		return errors.New("trajectory event source requires type, adapter, and adapter_version")
	}
	if len(e.Payload) == 0 || !json.Valid(e.Payload) {
		return errors.New("trajectory event payload must be valid JSON")
	}
	return nil
}

// Digest binds the canonical encoding of one event, including its provenance and loss report.
func (e Event) Digest() (string, error) {
	if err := e.Validate(); err != nil {
		return "", err
	}
	b, err := json.Marshal(e)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func EncodeEvents(events []Event) ([]byte, error) {
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	for i := range events {
		if err := events[i].Validate(); err != nil {
			return nil, fmt.Errorf("event %d: %w", i, err)
		}
		if err := enc.Encode(events[i]); err != nil {
			return nil, err
		}
	}
	return out.Bytes(), nil
}

func DecodeEvents(data []byte) ([]Event, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var events []Event
	for dec.More() {
		var event Event
		if err := dec.Decode(&event); err != nil {
			return nil, fmt.Errorf("decode trajectory event %d: %w", len(events), err)
		}
		if err := event.Validate(); err != nil {
			return nil, fmt.Errorf("event %d: %w", len(events), err)
		}
		events = append(events, event)
	}
	return events, nil
}

type messageEventPayload struct {
	Role   string `json:"role"`
	Text   string `json:"text"`
	TurnID string `json:"turn_id,omitempty"`
}

// CompactTurns projects completed message events into the stable Turn shape.
// Event ordering stays causal: sequence is a tie-breaker only after timestamp.
func CompactTurns(events []Event) ([]Turn, error) {
	ordered := append([]Event(nil), events...)
	for i := range ordered {
		if err := ordered[i].Validate(); err != nil {
			return nil, fmt.Errorf("event %d: %w", i, err)
		}
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Timestamp.Equal(ordered[j].Timestamp) {
			return ordered[i].Sequence < ordered[j].Sequence
		}
		return ordered[i].Timestamp.Before(ordered[j].Timestamp)
	})

	turns := make([]Turn, 0)
	for _, event := range ordered {
		if event.Kind != EventMessage || event.Action != "completed" {
			continue
		}
		var payload messageEventPayload
		if err := json.Unmarshal(event.Payload, &payload); err != nil {
			return nil, fmt.Errorf("message event %s: %w", event.ID, err)
		}
		turns = append(turns, Turn{
			TraceID:    event.ConversationID,
			Seq:        int(event.Sequence),
			TSUnixNano: event.Timestamp.UnixNano(),
			Query:      payload.Text,
			Labels: map[string]string{
				"event_id": event.ID,
				"role":     payload.Role,
				"turn_id":  payload.TurnID,
			},
		})
	}
	return turns, nil
}
