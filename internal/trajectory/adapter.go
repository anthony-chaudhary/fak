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
	r, _ := NewAdapterRegistry(CodexJSONLAdapter{}, AGUIJSONLAdapter{}, ClaudeCodeJSONLAdapter{}, OpenAIChatExportAdapter{}, PiJSONLAdapter{}, NativeReceiptAdapter{})
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

const (
	nativeReceiptSchema      = "fak.agent.native.v1"
	nativeReceiptCallsSchema = "fak.agent.native.calls.v1"
)

// NativeReceiptAdapter imports the bounded decision trace carried by a native
// agent receipt. It deliberately does not infer result bodies, source times, or
// per-call usage from the receipt's aggregate metrics.
type NativeReceiptAdapter struct{}

func (NativeReceiptAdapter) Name() string       { return "fak-agent-native-receipt-json" }
func (NativeReceiptAdapter) Version() string    { return "1" }
func (NativeReceiptAdapter) SourceType() string { return "fak-agent-native-receipt-json" }

type nativeReceiptImport struct {
	Schema string `json:"schema"`
	Calls  *struct {
		Schema  string            `json:"schema"`
		Entries []json.RawMessage `json:"entries"`
	} `json:"calls"`
}

type nativeCallImport struct {
	Turn    int    `json:"turn"`
	Tool    string `json:"tool"`
	Verdict string `json:"verdict"`
}

func (a NativeReceiptAdapter) Ingest(data []byte) ([]Event, FidelityReceipt, error) {
	receipt := newReceipt(a.SourceType(), a.Name(), a.Version(), data)
	receipt.InputRecords = 1
	var native nativeReceiptImport
	if err := json.Unmarshal(data, &native); err != nil {
		receipt.MalformedRecord = 1
		return nil, receipt, fmt.Errorf("native receipt: %w", err)
	}
	if native.Schema != nativeReceiptSchema {
		_ = finishReceipt(&receipt, nil)
		return nil, receipt, fmt.Errorf("unsupported native receipt schema %q", native.Schema)
	}
	if native.Calls == nil {
		receipt.Warnings = append(receipt.Warnings, "aggregate-only native receipt has no call records and cannot establish a fully audited coding trajectory")
		if err := finishReceipt(&receipt, nil); err != nil {
			return nil, receipt, err
		}
		return nil, receipt, nil
	}
	if native.Calls.Schema != nativeReceiptCallsSchema {
		_ = finishReceipt(&receipt, nil)
		return nil, receipt, fmt.Errorf("unsupported native calls schema %q", native.Calls.Schema)
	}

	sessionID := "native:" + strings.TrimPrefix(receipt.SourceDigest, "sha256:")[:16]
	events := make([]Event, 0, len(native.Calls.Entries))
	for i, raw := range native.Calls.Entries {
		var call nativeCallImport
		if err := json.Unmarshal(raw, &call); err != nil {
			receipt.MalformedRecord = 1
			return nil, receipt, fmt.Errorf("native receipt call %d: %w", i+1, err)
		}
		sequence := i + 1
		action := "adjudicated"
		switch strings.ToUpper(call.Verdict) {
		case "ALLOW":
			action = "admitted"
		case "DENY":
			action = "denied"
		}
		event := Event{
			Schema:         EventSchema,
			ID:             canonicalEventID("native-call", sequence, ""),
			ConversationID: sessionID,
			Kind:           EventTool,
			Action:         action,
			Timestamp:      time.Unix(0, int64(sequence)).UTC(),
			Sequence:       uint64(sequence),
			Visibility:     VisibilityRestricted,
			Source: EventSource{
				Type:           a.SourceType(),
				SessionID:      sessionID,
				OrderingKey:    strconv.Itoa(sequence),
				RawDigest:      digestBytes(raw),
				Adapter:        a.Name(),
				AdapterVersion: a.Version(),
			},
			Payload: append(json.RawMessage(nil), raw...),
			Loss: &LossReport{
				Reason: "native receipt omits source timestamp, tool result body, and exact per-call usage",
			},
		}
		if err := event.Validate(); err != nil {
			return nil, receipt, fmt.Errorf("native receipt call %d: %w", sequence, err)
		}
		events = append(events, event)
	}
	receipt.SyntheticTimes = len(events)
	receipt.Warnings = append(receipt.Warnings,
		"native call records omit source timestamps; deterministic synthetic timestamps were used",
		"native call records omit tool result bodies",
		"native receipt exposes aggregate usage only; exact per-call usage is unavailable",
	)
	if err := finishReceipt(&receipt, events); err != nil {
		return nil, receipt, err
	}
	return events, receipt, nil
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

// PiJSONLAdapter ingests the versioned Pi session JSONL log (one JSON object per
// line). Pi records are a parent-linked tree: `session` opens the run with a schema
// version, `model_change`/`thinking_level_change` are state, and `message` records
// carry a role and typed content blocks (`text`, `toolCall`). Tool results arrive in
// a follow-up `message` with role `toolResult` joined by `toolCallId`. The adapter
// preserves session/entry/parent IDs, raw digests, tool-call/result linkage, usage
// provenance (the whole message payload, including `usage`), and counts unknown
// kinds rather than silently flattening them.
type PiJSONLAdapter struct{}

func (PiJSONLAdapter) Name() string       { return "pi-jsonl" }
func (PiJSONLAdapter) Version() string    { return "1" }
func (PiJSONLAdapter) SourceType() string { return "pi-jsonl" }

// supportedPiSessionVersion is the pinned Pi session schema version this adapter
// understands. A session that declares another version cannot qualify a run.
const supportedPiSessionVersion = 3

type piEnvelope struct {
	Type          string          `json:"type"`
	ID            string          `json:"id"`
	ParentID      string          `json:"parentId"`
	Timestamp     string          `json:"timestamp"`
	Version       int             `json:"version"`
	Cwd           string          `json:"cwd"`
	Provider      string          `json:"provider"`
	ModelID       string          `json:"modelId"`
	ThinkingLevel string          `json:"thinkingLevel"`
	Message       json.RawMessage `json:"message"`
}

type piMessage struct {
	Role       string          `json:"role"`
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Content    json.RawMessage `json:"content"`
	Usage      json.RawMessage `json:"usage"`
}

type piContentBlock struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Text      string          `json:"text"`
	Arguments json.RawMessage `json:"arguments"`
}

func (a PiJSONLAdapter) Ingest(data []byte) ([]Event, FidelityReceipt, error) {
	receipt := newReceipt(a.SourceType(), a.Name(), a.Version(), data)
	var events []Event
	sessionID := ""
	unsupportedVersion := 0
	versionUnsupported := false
	toolResultsSeen := map[string]bool{}

	err := scanJSONL(data, func(index int, raw []byte) error {
		receipt.InputRecords++
		var envelope piEnvelope
		if err := json.Unmarshal(raw, &envelope); err != nil {
			receipt.MalformedRecord++
			return fmt.Errorf("pi-jsonl record %d: %w", index, err)
		}
		if envelope.Type == "session" {
			sessionID = stringFieldRaw(envelope.ID, sessionID)
			if envelope.Version != supportedPiSessionVersion {
				unsupportedVersion = envelope.Version
				versionUnsupported = true
				receipt.UnknownKinds[fmt.Sprintf("session_version=%d", envelope.Version)]++
			}
		}
		if sessionID == "" {
			sessionID = "pi-import:" + strings.TrimPrefix(receipt.SourceDigest, "sha256:")[:16]
		}
		stamp, synthetic := nativeTime(envelope.Timestamp, index)
		if synthetic {
			receipt.SyntheticTimes++
		}
		emit := func(kind EventKind, action string, sourceEventID string, parents []string, payload json.RawMessage) error {
			event := Event{
				Schema: EventSchema, ID: canonicalEventID("pi", len(events), sourceEventID), ConversationID: sessionID,
				Kind: kind, Action: action, Timestamp: stamp, Sequence: uint64(len(events) + 1), ParentIDs: parents,
				Visibility: VisibilityDeveloper,
				Source:     EventSource{Type: a.SourceType(), SessionID: sessionID, EventID: sourceEventID, OrderingKey: strconv.Itoa(index), RawDigest: digestBytes(raw), Adapter: a.Name(), AdapterVersion: a.Version()},
				Payload:    payload,
			}
			if err := event.Validate(); err != nil {
				return err
			}
			events = append(events, event)
			return nil
		}

		switch envelope.Type {
		case "session":
			return emit(EventRunLifecycle, "started", envelope.ID, parentIDs(envelope.ParentID), json.RawMessage(raw))
		case "model_change", "thinking_level_change":
			return emit(EventObservation, "recorded", envelope.ID, parentIDs(envelope.ParentID), json.RawMessage(raw))
		case "message":
			var msg piMessage
			if len(envelope.Message) == 0 || json.Unmarshal(envelope.Message, &msg) != nil {
				receipt.MalformedRecord++
				return fmt.Errorf("pi-jsonl record %d message: malformed", index)
			}
			switch msg.Role {
			case "user", "assistant":
				payload, normErr := normalizedPiMessage(msg)
				if normErr != nil {
					return normErr
				}
				if err := emit(EventMessage, "completed", envelope.ID, parentIDs(envelope.ParentID), payload); err != nil {
					return err
				}
				// A tool call is an action the assistant proposed within its turn;
				// preserve each block as its own linked event.
				for _, block := range piBlocks(msg.Content) {
					if block.Type != "toolCall" {
						continue
					}
					callID := compactStrings(block.ID, envelope.ID)[0]
					callPayload := block.Arguments
					if len(callPayload) == 0 || !json.Valid(callPayload) {
						callPayload = json.RawMessage(`{}`)
					}
					if err := emit(EventTool, "proposed", callID, compactStrings(envelope.ID, block.ID), callPayload); err != nil {
						return err
					}
				}
				return nil
			case "toolResult":
				if msg.ToolCallID != "" {
					if toolResultsSeen[msg.ToolCallID] {
						// Duplicate terminal record: count it, do not create a second
						// tool effect for the same call.
						receipt.Warnings = append(receipt.Warnings, "duplicate tool result for call "+msg.ToolCallID+" was not emitted twice")
						return nil
					}
					toolResultsSeen[msg.ToolCallID] = true
				}
				resultPayload := json.RawMessage(`{}`)
				if b, mErr := json.Marshal(map[string]any{"toolCallId": msg.ToolCallID, "toolName": msg.ToolName, "content": json.RawMessage(msg.Content)}); mErr == nil {
					resultPayload = b
				}
				return emit(EventTool, "completed", envelope.ID, parentIDs(envelope.ParentID, msg.ToolCallID), resultPayload)
			default:
				receipt.UnknownKinds["message_role="+msg.Role]++
				return emit(EventObservation, "recorded", envelope.ID, parentIDs(envelope.ParentID), json.RawMessage(raw))
			}
		default:
			receipt.UnknownKinds[envelope.Type]++
			payload := json.RawMessage(raw)
			event := Event{
				Schema: EventSchema, ID: canonicalEventID("pi", len(events), envelope.ID), ConversationID: sessionID,
				Kind: EventObservation, Action: envelope.Type, Timestamp: stamp, Sequence: uint64(len(events) + 1), ParentIDs: parentIDs(envelope.ParentID),
				Visibility: VisibilityDeveloper,
				Source:     EventSource{Type: a.SourceType(), SessionID: sessionID, EventID: envelope.ID, OrderingKey: strconv.Itoa(index), RawDigest: digestBytes(raw), Adapter: a.Name(), AdapterVersion: a.Version()},
				Payload:    payload, Loss: &LossReport{UnknownKinds: []string{envelope.Type}, Reason: "native kind preserved as observation"},
			}
			if err := event.Validate(); err != nil {
				return err
			}
			events = append(events, event)
			return nil
		}
	})
	if err != nil {
		receipt.EmittedEvents = len(events)
		_ = finishReceipt(&receipt, events)
		return events, receipt, err
	}
	if receipt.SyntheticTimes > 0 {
		receipt.Warnings = append(receipt.Warnings, fmt.Sprintf("%d record(s) lacked a source timestamp; deterministic ordering timestamps were used", receipt.SyntheticTimes))
	}
	if len(receipt.UnknownKinds) > 0 {
		receipt.Warnings = append(receipt.Warnings, "unsupported native kinds were counted and omitted")
	}
	if err := finishReceipt(&receipt, events); err != nil {
		return events, receipt, err
	}
	if versionUnsupported {
		return events, receipt, fmt.Errorf("pi-jsonl session schema version %d is unsupported (want %d); the run cannot be qualified", unsupportedVersion, supportedPiSessionVersion)
	}
	return events, receipt, nil
}

// normalizedPiMessage produces the stable message payload CompactTurns reads
// (`role`/`text`) while preserving the original content blocks and the usage,
// provider and model provenance a consumer needs. It never drops the native
// blocks: they ride under `content`.
func normalizedPiMessage(msg piMessage) (json.RawMessage, error) {
	var textParts []string
	for _, block := range piBlocks(msg.Content) {
		if block.Type == "text" && strings.TrimSpace(block.Text) != "" {
			textParts = append(textParts, block.Text)
		}
	}
	object := map[string]any{"role": msg.Role, "text": strings.Join(textParts, "\n")}
	if len(msg.Content) > 0 && json.Valid(msg.Content) {
		object["content"] = json.RawMessage(msg.Content)
	}
	if msg.ToolCallID != "" {
		object["tool_call_id"] = msg.ToolCallID
	}
	if msg.ToolName != "" {
		object["tool_name"] = msg.ToolName
	}
	if len(msg.Usage) > 0 && json.Valid(msg.Usage) {
		object["usage"] = json.RawMessage(msg.Usage)
	}
	normalized, err := json.Marshal(object)
	if err != nil {
		return nil, err
	}
	return normalized, nil
}

// piBlocks decodes a Pi message's typed content blocks; a non-list content yields none.
func piBlocks(content json.RawMessage) []piContentBlock {
	if len(content) == 0 {
		return nil
	}
	var blocks []piContentBlock
	if json.Unmarshal(content, &blocks) != nil {
		return nil
	}
	return blocks
}

// parentIDs returns the non-empty parent identifiers in order.
func parentIDs(values ...string) []string { return compactStrings(values...) }

// stringFieldRaw returns value when non-blank, else fallback.
func stringFieldRaw(value, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
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
