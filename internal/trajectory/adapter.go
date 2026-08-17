package trajectory

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

const FidelityReceiptSchema = "fak-trajectory-fidelity/1alpha1"

// Adapter turns one native transcript format into canonical trajectory events.
type Adapter interface {
	Name() string
	Version() string
	SourceType() string
	Ingest([]byte) ([]Event, FidelityReceipt, error)
}

// AdapterRegistry makes source selection explicit; it never guesses from bytes.
type AdapterRegistry struct {
	bySource map[string]Adapter
}

func NewAdapterRegistry(adapters ...Adapter) (*AdapterRegistry, error) {
	r := &AdapterRegistry{bySource: make(map[string]Adapter)}
	for _, adapter := range adapters {
		if adapter == nil || strings.TrimSpace(adapter.SourceType()) == "" {
			return nil, errors.New("trajectory adapter requires source type")
		}
		if _, exists := r.bySource[adapter.SourceType()]; exists {
			return nil, fmt.Errorf("duplicate trajectory adapter for %q", adapter.SourceType())
		}
		r.bySource[adapter.SourceType()] = adapter
	}
	return r, nil
}

func DefaultAdapterRegistry() *AdapterRegistry {
	r, _ := NewAdapterRegistry(CodexJSONLAdapter{}, AGUIJSONLAdapter{})
	return r
}

func (r *AdapterRegistry) Sources() []string {
	out := make([]string, 0, len(r.bySource))
	for source := range r.bySource {
		out = append(out, source)
	}
	sort.Strings(out)
	return out
}

func (r *AdapterRegistry) Ingest(source string, data []byte) ([]Event, FidelityReceipt, error) {
	adapter, ok := r.bySource[source]
	if !ok {
		return nil, FidelityReceipt{}, fmt.Errorf("no trajectory adapter for %q; available: %s", source, strings.Join(r.Sources(), ", "))
	}
	return adapter.Ingest(data)
}

// FidelityReceipt accounts for every native record without claiming unsupported semantics.
type FidelityReceipt struct {
	Schema          string         `json:"schema"`
	SourceType      string         `json:"source_type"`
	SourceDigest    string         `json:"source_digest"`
	Adapter         string         `json:"adapter"`
	AdapterVersion  string         `json:"adapter_version"`
	InputRecords    int            `json:"input_records"`
	EmittedEvents   int            `json:"emitted_events"`
	UnknownKinds    map[string]int `json:"unknown_kinds,omitempty"`
	MalformedRecord int            `json:"malformed_records,omitempty"`
	SyntheticTimes  int            `json:"synthetic_times,omitempty"`
	Warnings        []string       `json:"warnings,omitempty"`
	EventDigest     string         `json:"event_digest,omitempty"`
}

func (r FidelityReceipt) Validate() error {
	if r.Schema != FidelityReceiptSchema || r.SourceType == "" || r.SourceDigest == "" || r.Adapter == "" || r.AdapterVersion == "" {
		return errors.New("incomplete trajectory fidelity receipt identity")
	}
	if r.InputRecords < 0 || r.EmittedEvents < 0 || r.MalformedRecord < 0 || r.MalformedRecord > r.InputRecords {
		return errors.New("invalid trajectory fidelity receipt counts")
	}
	return nil
}

func finishReceipt(receipt *FidelityReceipt, events []Event) error {
	receipt.EmittedEvents = len(events)
	encoded, err := EncodeEvents(events)
	if err != nil {
		return err
	}
	receipt.EventDigest = digestBytes(encoded)
	return receipt.Validate()
}

func newReceipt(sourceType, adapter, version string, data []byte) FidelityReceipt {
	return FidelityReceipt{Schema: FidelityReceiptSchema, SourceType: sourceType, SourceDigest: digestBytes(data), Adapter: adapter, AdapterVersion: version, UnknownKinds: make(map[string]int)}
}

func digestBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

type CodexJSONLAdapter struct{}

func (CodexJSONLAdapter) Name() string       { return "codex-jsonl" }
func (CodexJSONLAdapter) Version() string    { return "1" }
func (CodexJSONLAdapter) SourceType() string { return "codex-jsonl" }

type nativeEnvelope struct {
	Timestamp string          `json:"timestamp"`
	Type      string          `json:"type"`
	Payload   json.RawMessage `json:"payload"`
}

func (a CodexJSONLAdapter) Ingest(data []byte) ([]Event, FidelityReceipt, error) {
	receipt := newReceipt(a.SourceType(), a.Name(), a.Version(), data)
	var events []Event
	sessionID := "codex-import"
	err := scanJSONL(data, func(index int, raw []byte) error {
		receipt.InputRecords++
		var envelope nativeEnvelope
		if err := json.Unmarshal(raw, &envelope); err != nil {
			receipt.MalformedRecord++
			return fmt.Errorf("codex-jsonl record %d: %w", index, err)
		}
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(envelope.Payload, &payload); err != nil {
			receipt.MalformedRecord++
			return fmt.Errorf("codex-jsonl record %d payload: %w", index, err)
		}
		if envelope.Type == "session_meta" {
			sessionID = stringField(payload, "id", sessionID)
		}
		kind, action, known := codexSemantics(envelope.Type, stringField(payload, "type", ""))
		if !known {
			key := envelope.Type
			if subtype := stringField(payload, "type", ""); subtype != "" {
				key += "/" + subtype
			}
			receipt.UnknownKinds[key]++
			return nil
		}
		ts, synthetic := nativeTime(envelope.Timestamp, index)
		if synthetic {
			receipt.SyntheticTimes++
		}
		sourceEventID := stringField(payload, "id", stringField(payload, "call_id", ""))
		eventID := canonicalEventID("codex", index, sourceEventID)
		parents := compactStrings(stringField(payload, "parent_id", ""), stringField(payload, "call_id", ""))
		events = append(events, Event{Schema: EventSchema, ID: eventID, ConversationID: sessionID, Kind: kind, Action: action, Timestamp: ts, Sequence: uint64(index), ParentIDs: parents, Visibility: VisibilityDeveloper, Source: EventSource{Type: a.SourceType(), SessionID: sessionID, EventID: sourceEventID, OrderingKey: strconv.Itoa(index), RawDigest: digestBytes(raw), Adapter: a.Name(), AdapterVersion: a.Version()}, Payload: append(json.RawMessage(nil), envelope.Payload...)})
		return nil
	})
	if err != nil {
		return nil, receipt, err
	}
	if receipt.SyntheticTimes > 0 {
		receipt.Warnings = append(receipt.Warnings, fmt.Sprintf("%d record(s) lacked a source timestamp; deterministic ordering timestamps were used", receipt.SyntheticTimes))
	}
	if len(receipt.UnknownKinds) > 0 {
		receipt.Warnings = append(receipt.Warnings, "unsupported native kinds were counted and omitted")
	}
	if err := finishReceipt(&receipt, events); err != nil {
		return nil, receipt, err
	}
	return events, receipt, nil
}

func codexSemantics(envelopeType, subtype string) (EventKind, string, bool) {
	switch envelopeType + "/" + subtype {
	case "session_meta/":
		return EventRunLifecycle, "started", true
	case "event_msg/user_message":
		return EventMessage, "completed", true
	case "event_msg/agent_message":
		return EventMessage, "completed", true
	case "event_msg/task_started":
		return EventRunLifecycle, "started", true
	case "event_msg/task_complete":
		return EventRunLifecycle, "completed", true
	case "response_item/message":
		return EventMessage, "completed", true
	case "response_item/function_call", "response_item/custom_tool_call":
		return EventTool, "proposed", true
	case "response_item/function_call_output", "response_item/custom_tool_call_output":
		return EventTool, "completed", true
	case "response_item/reasoning":
		return EventObservation, "recorded", true
	default:
		return "", "", false
	}
}

type AGUIJSONLAdapter struct{}

func (AGUIJSONLAdapter) Name() string       { return "ag-ui-jsonl" }
func (AGUIJSONLAdapter) Version() string    { return "1" }
func (AGUIJSONLAdapter) SourceType() string { return "ag-ui-jsonl" }

func (a AGUIJSONLAdapter) Ingest(data []byte) ([]Event, FidelityReceipt, error) {
	receipt := newReceipt(a.SourceType(), a.Name(), a.Version(), data)
	var events []Event
	err := scanJSONL(data, func(index int, raw []byte) error {
		receipt.InputRecords++
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			receipt.MalformedRecord++
			return fmt.Errorf("ag-ui-jsonl record %d: %w", index, err)
		}
		nativeKind := stringField(fields, "type", "")
		kind, action, known := aguiSemantics(nativeKind)
		if !known {
			receipt.UnknownKinds[nativeKind]++
			return nil
		}
		sessionID := stringField(fields, "threadId", stringField(fields, "runId", "ag-ui-import"))
		sourceEventID := stringField(fields, "eventId", stringField(fields, "messageId", stringField(fields, "toolCallId", "")))
		eventID := canonicalEventID("agui", index, sourceEventID)
		ts, synthetic := nativeTime(stringField(fields, "timestamp", ""), index)
		if synthetic {
			receipt.SyntheticTimes++
		}
		parents := compactStrings(stringField(fields, "parentMessageId", ""), stringField(fields, "parentRunId", ""))
		events = append(events, Event{Schema: EventSchema, ID: eventID, ConversationID: sessionID, Kind: kind, Action: action, Timestamp: ts, Sequence: uint64(index), ParentIDs: parents, Visibility: VisibilityDeveloper, Source: EventSource{Type: a.SourceType(), SessionID: sessionID, EventID: sourceEventID, OrderingKey: strconv.Itoa(index), RawDigest: digestBytes(raw), Adapter: a.Name(), AdapterVersion: a.Version()}, Payload: append(json.RawMessage(nil), raw...)})
		return nil
	})
	if err != nil {
		return nil, receipt, err
	}
	if receipt.SyntheticTimes > 0 {
		receipt.Warnings = append(receipt.Warnings, fmt.Sprintf("%d record(s) lacked a source timestamp; deterministic ordering timestamps were used", receipt.SyntheticTimes))
	}
	if len(receipt.UnknownKinds) > 0 {
		receipt.Warnings = append(receipt.Warnings, "unsupported native kinds were counted and omitted")
	}
	if err := finishReceipt(&receipt, events); err != nil {
		return nil, receipt, err
	}
	return events, receipt, nil
}

func aguiSemantics(nativeKind string) (EventKind, string, bool) {
	switch nativeKind {
	case "RUN_STARTED":
		return EventRunLifecycle, "started", true
	case "RUN_FINISHED":
		return EventRunLifecycle, "completed", true
	case "RUN_ERROR":
		return EventError, "reported", true
	case "TEXT_MESSAGE_START":
		return EventMessage, "started", true
	case "TEXT_MESSAGE_CONTENT":
		return EventMessage, "delta", true
	case "TEXT_MESSAGE_END":
		return EventMessage, "completed", true
	case "TOOL_CALL_START":
		return EventTool, "started", true
	case "TOOL_CALL_ARGS":
		return EventTool, "delta", true
	case "TOOL_CALL_END", "TOOL_CALL_RESULT":
		return EventTool, "completed", true
	case "STATE_SNAPSHOT":
		return EventState, "snapshot", true
	case "STATE_DELTA", "MESSAGES_SNAPSHOT":
		return EventState, "delta", true
	case "ACTIVITY_SNAPSHOT", "ACTIVITY_DELTA":
		return EventObservation, "recorded", true
	case "STEP_STARTED":
		return EventRunLifecycle, "step-started", true
	case "STEP_FINISHED":
		return EventRunLifecycle, "step-completed", true
	case "CUSTOM":
		return EventObservation, "custom", true
	default:
		return "", "", false
	}
}

func canonicalEventID(prefix string, index int, sourceEventID string) string {
	if sourceEventID == "" {
		return fmt.Sprintf("%s-%d", prefix, index)
	}
	return fmt.Sprintf("%s-%d-%s", prefix, index, sourceEventID)
}

func scanJSONL(data []byte, consume func(int, []byte) error) error {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	index := 0
	for scanner.Scan() {
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 {
			continue
		}
		index++
		if err := consume(index, append([]byte(nil), raw...)); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func nativeTime(value string, index int) (time.Time, bool) {
	if value != "" {
		if ts, err := time.Parse(time.RFC3339Nano, value); err == nil {
			return ts.UTC(), false
		}
		if millis, err := strconv.ParseInt(value, 10, 64); err == nil {
			return time.UnixMilli(millis).UTC(), false
		}
	}
	return time.Unix(0, int64(index)).UTC(), true
}

func stringField(fields map[string]json.RawMessage, name, fallback string) string {
	raw, ok := fields[name]
	if !ok {
		return fallback
	}
	var value string
	if json.Unmarshal(raw, &value) == nil {
		return value
	}
	return fallback
}

func compactStrings(values ...string) []string {
	out := make([]string, 0, len(values))
	seen := make(map[string]struct{})
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
