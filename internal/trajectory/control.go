package trajectory

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

const ControlReceiptSchema = "fak-trajectory-control/1alpha1"

var validControlActions = map[string]struct{}{
	"approve": {}, "deny": {}, "steer": {}, "pause": {}, "retry": {}, "branch": {},
}

type ControlRequest struct {
	Action        string          `json:"action"`
	ActorID       string          `json:"actor_id"`
	TargetEventID string          `json:"target_event_id,omitempty"`
	Payload       json.RawMessage `json:"payload,omitempty"`
	Timestamp     time.Time       `json:"timestamp"`
}

type ControlReceipt struct {
	Schema         string `json:"schema"`
	ViewSpecDigest string `json:"view_spec_digest"`
	HistoryDigest  string `json:"history_digest"`
	RequestDigest  string `json:"request_digest"`
	EventDigest    string `json:"event_digest"`
	InterventionID string `json:"intervention_id"`
}

// ApplyViewControl validates a view action and returns the canonical intervention to append.
// It never mutates history; a refusal returns no event or receipt.
func ApplyViewControl(spec ViewSpec, history []Event, request ControlRequest) (Event, ControlReceipt, error) {
	if err := spec.Validate(); err != nil {
		return Event{}, ControlReceipt{}, fmt.Errorf("view spec: %w", err)
	}
	action := strings.ToLower(strings.TrimSpace(request.Action))
	if _, ok := validControlActions[action]; !ok {
		return Event{}, ControlReceipt{}, fmt.Errorf("unsupported view control %q", request.Action)
	}
	if !containsControl(spec.LiveControls, action) {
		return Event{}, ControlReceipt{}, fmt.Errorf("view control %q is not allowed for audience %q", action, spec.Audience)
	}
	if strings.TrimSpace(request.ActorID) == "" {
		return Event{}, ControlReceipt{}, errors.New("view control requires actor_id")
	}
	if request.Timestamp.IsZero() {
		return Event{}, ControlReceipt{}, errors.New("view control requires timestamp")
	}
	if len(request.Payload) == 0 {
		request.Payload = json.RawMessage(`{}`)
	}
	if !json.Valid(request.Payload) {
		return Event{}, ControlReceipt{}, errors.New("view control payload must be valid JSON")
	}

	ordered := append([]Event(nil), history...)
	var conversationID string
	var maxSequence uint64
	for i := range ordered {
		if err := ordered[i].Validate(); err != nil {
			return Event{}, ControlReceipt{}, fmt.Errorf("history event %d: %w", i, err)
		}
		if conversationID == "" {
			conversationID = ordered[i].ConversationID
		}
		if ordered[i].ConversationID != conversationID {
			return Event{}, ControlReceipt{}, errors.New("view control history spans multiple conversations")
		}
		if ordered[i].Sequence > maxSequence {
			maxSequence = ordered[i].Sequence
		}
	}
	if conversationID == "" {
		return Event{}, ControlReceipt{}, errors.New("view control requires non-empty canonical history")
	}
	if request.TargetEventID != "" && !hasEventID(ordered, request.TargetEventID) {
		return Event{}, ControlReceipt{}, fmt.Errorf("target event %q is not in canonical history", request.TargetEventID)
	}

	specBytes, _ := json.Marshal(spec)
	historyBytes, err := EncodeEvents(ordered)
	if err != nil {
		return Event{}, ControlReceipt{}, err
	}
	requestBytes, _ := json.Marshal(request)
	safeInput, _, err := redactPayload(request.Payload, spec.RedactFields)
	if err != nil {
		return Event{}, ControlReceipt{}, fmt.Errorf("redact control payload: %w", err)
	}
	eventPayload, _ := json.Marshal(map[string]any{
		"actor_id": request.ActorID, "target_event_id": request.TargetEventID,
		"view_audience": spec.Audience, "view_spec_digest": viewDigest(specBytes), "input": safeInput,
	})
	parentIDs := []string(nil)
	if request.TargetEventID != "" {
		parentIDs = []string{request.TargetEventID}
	}
	idSeed, _ := json.Marshal([]any{conversationID, action, request.ActorID, request.TargetEventID, request.Timestamp.UTC(), request.Payload, viewDigest(specBytes), digestBytes(historyBytes)})
	event := Event{
		Schema: EventSchema, ID: "intervention:" + strings.TrimPrefix(digestBytes(idSeed), "sha256:")[:20], ConversationID: conversationID,
		Kind: EventIntervention, Action: action, Timestamp: request.Timestamp.UTC(), Sequence: maxSequence + 1, ParentIDs: parentIDs,
		Visibility: VisibilityOperator, Source: EventSource{Type: "trajectory-view", SessionID: conversationID, OrderingKey: fmt.Sprint(maxSequence + 1), Adapter: "view-control", AdapterVersion: "1"}, Payload: eventPayload,
	}
	if err := event.Validate(); err != nil {
		return Event{}, ControlReceipt{}, err
	}
	eventDigest, err := event.Digest()
	if err != nil {
		return Event{}, ControlReceipt{}, err
	}
	return event, ControlReceipt{Schema: ControlReceiptSchema, ViewSpecDigest: viewDigest(specBytes), HistoryDigest: digestBytes(historyBytes), RequestDigest: digestBytes(requestBytes), EventDigest: eventDigest, InterventionID: event.ID}, nil
}

func containsControl(controls []string, wanted string) bool {
	for _, control := range controls {
		if strings.EqualFold(strings.TrimSpace(control), wanted) {
			return true
		}
	}
	return false
}

func hasEventID(events []Event, wanted string) bool {
	for _, event := range events {
		if event.ID == wanted {
			return true
		}
	}
	return false
}
