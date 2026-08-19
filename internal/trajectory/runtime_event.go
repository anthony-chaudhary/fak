package trajectory

import (
	"encoding/json"
	"fmt"
	"time"
)

const RuntimeEventSchema = "fak-runtime-event/1"
const RuntimeEventDescriptorSchema = "fak-runtime-event-schema/1"

type RuntimeEventKind = string

const (
	RuntimeTurnStarted     RuntimeEventKind = "turn_started"
	RuntimeToolProposed    RuntimeEventKind = "tool_proposed"
	RuntimeVerdict         RuntimeEventKind = "tool_verdict"
	RuntimeToolResult      RuntimeEventKind = "tool_result_admitted"
	RuntimeContextChanged  RuntimeEventKind = "context_changed"
	RuntimeCostDebited     RuntimeEventKind = "cost_debited"
	RuntimeTerminalWitness RuntimeEventKind = "terminal_witness"
	RuntimeTerminal        RuntimeEventKind = "turn_terminal" // retained for tool-call wire compatibility
	RuntimeError           RuntimeEventKind = "error"
)

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
type RuntimeEventDescriptor struct {
	Schema        string             `json:"schema"`
	EventSchema   string             `json:"event_schema"`
	Kinds         []RuntimeEventKind `json:"kinds"`
	Required      []string           `json:"required"`
	PayloadPolicy string             `json:"payload_policy"`
	Transport     []string           `json:"transport"`
}

func RuntimeEventSchemaDescriptor() RuntimeEventDescriptor {
	return RuntimeEventDescriptor{Schema: RuntimeEventDescriptorSchema, EventSchema: RuntimeEventSchema, Kinds: RuntimeEventKinds(), Required: []string{"event_id", "session_id", "turn_id", "trace_id", "sequence", "timestamp", "kind", "source", "payload"}, PayloadPolicy: "payload must be one valid JSON value; content-bearing payloads require upstream ctxmmu screening and taint stamping", Transport: []string{"ndjson", "sse-data-json"}}
}

func RuntimeEventKinds() []RuntimeEventKind {
	return []RuntimeEventKind{RuntimeTurnStarted, RuntimeToolProposed, RuntimeVerdict, RuntimeToolResult, RuntimeContextChanged, RuntimeCostDebited, RuntimeTerminalWitness, RuntimeTerminal, RuntimeError}
}

func NewRuntimeEvent(eventID, sessionID, turnID, traceID string, sequence uint64, at time.Time, kind RuntimeEventKind, source RuntimeSource, payload json.RawMessage) (RuntimeEvent, error) {
	e := RuntimeEvent{Schema: RuntimeEventSchema, EventID: eventID, SessionID: sessionID, TurnID: turnID, TraceID: traceID, Sequence: sequence, Timestamp: at, Kind: kind, Source: source, Payload: payload}
	if err := ValidateRuntimeEvent(e); err != nil {
		return RuntimeEvent{}, err
	}
	return e, nil
}

func ValidateRuntimeEvent(e RuntimeEvent) error {
	if e.Schema != RuntimeEventSchema {
		return fmt.Errorf("runtime event schema %q", e.Schema)
	}
	if e.EventID == "" || e.SessionID == "" || e.TurnID == "" || e.TraceID == "" || e.Source.Component == "" || e.Source.Instance == "" || e.Source.Runtime == "" || e.Timestamp.IsZero() || e.Sequence == 0 {
		return fmt.Errorf("runtime event identity, source, sequence, and timestamp are required")
	}
	known := false
	for _, kind := range RuntimeEventKinds() {
		if e.Kind == kind {
			known = true
			break
		}
	}
	if !known {
		return fmt.Errorf("unknown runtime event kind %q", e.Kind)
	}
	if len(e.Payload) == 0 || !json.Valid(e.Payload) {
		return fmt.Errorf("runtime event payload must be valid JSON")
	}
	return nil
}

func ToolCallEvents(sessionID, turnID, traceID, callID, tool string, admitted bool, result json.RawMessage, at time.Time, source RuntimeSource) ([]RuntimeEvent, error) {
	if sessionID == "" || turnID == "" || traceID == "" || callID == "" || tool == "" || source.Component == "" || source.Instance == "" || source.Runtime == "" || at.IsZero() {
		return nil, fmt.Errorf("runtime event identity, source, and timestamp are required")
	}
	payloads := []struct {
		kind string
		body any
	}{
		{RuntimeTurnStarted, map[string]any{}},
		{RuntimeToolProposed, map[string]any{"call_id": callID, "tool": tool}},
		{RuntimeVerdict, map[string]any{"call_id": callID, "admitted": admitted}},
	}
	if admitted {
		if len(result) == 0 || !json.Valid(result) {
			return nil, fmt.Errorf("admitted tool result must be valid JSON")
		}
		payloads = append(payloads, struct {
			kind string
			body any
		}{RuntimeToolResult, map[string]any{"call_id": callID, "result": result}})
	}
	payloads = append(payloads, struct {
		kind string
		body any
	}{RuntimeTerminal, map[string]any{"status": map[bool]string{true: "completed", false: "denied"}[admitted]}})
	out := make([]RuntimeEvent, 0, len(payloads))
	for i, p := range payloads {
		b, err := json.Marshal(p.body)
		if err != nil {
			return nil, err
		}
		event, err := NewRuntimeEvent(fmt.Sprintf("%s:%d", traceID, i+1), sessionID, turnID, traceID, uint64(i+1), at.Add(time.Duration(i)*time.Nanosecond), p.kind, source, b)
		if err != nil {
			return nil, err
		}
		out = append(out, event)
	}
	return out, nil
}

// AsTrajectoryEvents adapts runtime wire events into the canonical strict trajectory stream.
func AsTrajectoryEvents(in []RuntimeEvent) ([]Event, error) {
	out := make([]Event, 0, len(in))
	for _, r := range in {
		if err := ValidateRuntimeEvent(r); err != nil {
			return nil, err
		}
		b, err := json.Marshal(r)
		if err != nil {
			return nil, err
		}
		out = append(out, Event{Schema: EventSchema, ID: r.EventID, ConversationID: r.SessionID, Kind: EventTool, Action: r.Kind, Timestamp: r.Timestamp, Sequence: r.Sequence, Visibility: VisibilityOperator, Source: EventSource{Type: "runtime", SessionID: r.SessionID, EventID: r.EventID, OrderingKey: fmt.Sprint(r.Sequence), Adapter: "runtime-event", AdapterVersion: "1"}, Payload: b})
	}
	return out, nil
}
