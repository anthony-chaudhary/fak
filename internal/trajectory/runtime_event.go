package trajectory

import (
	"encoding/json"
	"fmt"
	"time"
)

const RuntimeEventSchema = "fak-runtime-event/1"

type RuntimeSource struct {
	Component string `json:"component"`
	Instance  string `json:"instance"`
	Runtime   string `json:"runtime"`
}

type RuntimeEvent struct {
	Schema    string          `json:"schema"`
	EventID   string          `json:"event_id"`
	SessionID string          `json:"session_id"`
	TurnID    string          `json:"turn_id"`
	TraceID   string          `json:"trace_id"`
	Sequence  uint64          `json:"sequence"`
	Timestamp time.Time       `json:"timestamp"`
	Kind      string          `json:"kind"`
	Source    RuntimeSource   `json:"source"`
	Payload   json.RawMessage `json:"payload"`
}

// ToolCallEvents emits the minimum canonical lifecycle for one tool decision.
// A denied call has one verdict and no result; an admitted call has exactly one result.
func ToolCallEvents(sessionID, turnID, traceID, callID, tool string, admitted bool, result json.RawMessage, at time.Time, source RuntimeSource) ([]RuntimeEvent, error) {
	if sessionID == "" || turnID == "" || traceID == "" || callID == "" || tool == "" || source.Component == "" || source.Instance == "" || source.Runtime == "" || at.IsZero() {
		return nil, fmt.Errorf("runtime event identity, source, and timestamp are required")
	}
	payloads := []struct {
		kind string
		body any
	}{
		{"turn_started", map[string]any{}},
		{"tool_proposed", map[string]any{"call_id": callID, "tool": tool}},
		{"tool_verdict", map[string]any{"call_id": callID, "admitted": admitted}},
	}
	if admitted {
		if len(result) == 0 || !json.Valid(result) {
			return nil, fmt.Errorf("admitted tool result must be valid JSON")
		}
		payloads = append(payloads, struct {
			kind string
			body any
		}{"tool_result_admitted", map[string]any{"call_id": callID, "result": result}})
	}
	payloads = append(payloads, struct {
		kind string
		body any
	}{"turn_terminal", map[string]any{"status": map[bool]string{true: "completed", false: "denied"}[admitted]}})
	out := make([]RuntimeEvent, 0, len(payloads))
	for i, p := range payloads {
		b, err := json.Marshal(p.body)
		if err != nil {
			return nil, err
		}
		out = append(out, RuntimeEvent{Schema: RuntimeEventSchema, EventID: fmt.Sprintf("%s:%d", traceID, i+1), SessionID: sessionID, TurnID: turnID, TraceID: traceID, Sequence: uint64(i + 1), Timestamp: at.Add(time.Duration(i) * time.Nanosecond), Kind: p.kind, Source: source, Payload: b})
	}
	return out, nil
}

// AsTrajectoryEvents adapts runtime wire events into the canonical strict trajectory stream.
func AsTrajectoryEvents(in []RuntimeEvent) ([]Event, error) {
	out := make([]Event, 0, len(in))
	for _, r := range in {
		if r.Schema != RuntimeEventSchema {
			return nil, fmt.Errorf("runtime event schema %q", r.Schema)
		}
		b, err := json.Marshal(r)
		if err != nil {
			return nil, err
		}
		out = append(out, Event{Schema: EventSchema, ID: r.EventID, ConversationID: r.SessionID, Kind: EventTool, Action: r.Kind, Timestamp: r.Timestamp, Sequence: r.Sequence, Visibility: VisibilityOperator, Source: EventSource{Type: "runtime", SessionID: r.SessionID, EventID: r.EventID, OrderingKey: fmt.Sprint(r.Sequence), Adapter: "runtime-event", AdapterVersion: "1"}, Payload: b})
	}
	return out, nil
}
