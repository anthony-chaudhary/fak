package trajectory

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ClaudeCodeJSONLAdapter struct{}

func (ClaudeCodeJSONLAdapter) Name() string       { return "claude-code-jsonl" }
func (ClaudeCodeJSONLAdapter) Version() string    { return "1" }
func (ClaudeCodeJSONLAdapter) SourceType() string { return "claude-code-jsonl" }
func (a ClaudeCodeJSONLAdapter) Ingest(data []byte) ([]Event, FidelityReceipt, error) {
	return ingestExportJSONL(a.Name(), a.Version(), a.SourceType(), data, map[string]exportMeaning{
		"user": {EventMessage, "completed"}, "assistant": {EventMessage, "completed"}, "system": {EventMessage, "completed"},
		"tool_use": {EventTool, "proposed"}, "tool_call": {EventTool, "proposed"}, "tool_result": {EventTool, "completed"},
		"summary": {EventCheckpoint, "saved"}, "checkpoint": {EventCheckpoint, "saved"},
	}, true)
}

type OpenAIChatExportAdapter struct{}

func (OpenAIChatExportAdapter) Name() string       { return "openai-chat-export-jsonl" }
func (OpenAIChatExportAdapter) Version() string    { return "1" }
func (OpenAIChatExportAdapter) SourceType() string { return "openai-chat-export-jsonl" }
func (a OpenAIChatExportAdapter) Ingest(data []byte) ([]Event, FidelityReceipt, error) {
	return ingestExportJSONL(a.Name(), a.Version(), a.SourceType(), data, map[string]exportMeaning{
		"message": {EventMessage, "completed"}, "chat.message": {EventMessage, "completed"},
		"tool_call": {EventTool, "proposed"}, "response.function_call_arguments.done": {EventTool, "proposed"},
		"tool_result": {EventTool, "completed"}, "function_call_output": {EventTool, "completed"},
		"response.completed": {EventRunLifecycle, "completed"},
	}, false)
}

type exportMeaning struct {
	kind   EventKind
	action string
}

type exportEnvelope struct {
	Type           string          `json:"type"`
	ID             string          `json:"id"`
	SessionID      string          `json:"session_id"`
	ConversationID string          `json:"conversation_id"`
	Timestamp      string          `json:"timestamp"`
	Payload        json.RawMessage `json:"payload"`
	Message        json.RawMessage `json:"message"`
}

func ingestExportJSONL(name, version, sourceType string, data []byte, meanings map[string]exportMeaning, inferRole bool) ([]Event, FidelityReceipt, error) {
	receipt := newReceipt(sourceType, name, version, data)
	var events []Event
	sessionID := ""
	err := scanJSONL(data, func(index int, line []byte) error {
		var envelope exportEnvelope
		if err := json.Unmarshal(line, &envelope); err != nil {
			receipt.InputRecords++
			receipt.MalformedRecord++
			return err
		}
		receipt.InputRecords++
		if sessionID == "" {
			sessionID = firstExportValue(envelope.SessionID, envelope.ConversationID)
		}
		if sessionID == "" {
			sessionID = sourceType + ":" + strings.TrimPrefix(receipt.SourceDigest, "sha256:")[:16]
		}
		meaning, supported := meanings[envelope.Type]
		if !supported {
			meaning = exportMeaning{EventObservation, envelope.Type}
			receipt.UnknownKinds[envelope.Type]++
		}
		payload := envelope.Payload
		if len(payload) == 0 || string(payload) == "null" {
			payload = envelope.Message
		}
		if len(payload) == 0 || string(payload) == "null" {
			payload = json.RawMessage(`{}`)
		}
		if inferRole && meaning.kind == EventMessage {
			payload = normalizeExportMessage(envelope.Type, payload)
		}
		stamp, synthetic := nativeTime(envelope.Timestamp, index)
		if synthetic {
			receipt.SyntheticTimes++
		}
		event := Event{Schema: EventSchema, ID: canonicalEventID(name, index, envelope.ID), ConversationID: sessionID, Kind: meaning.kind, Action: meaning.action, Timestamp: stamp, Sequence: uint64(index + 1), Visibility: VisibilityOperator, Source: EventSource{Type: sourceType, SessionID: sessionID, EventID: envelope.ID, OrderingKey: fmt.Sprint(index + 1), RawDigest: digestBytes(line), Adapter: name, AdapterVersion: version}, Payload: payload}
		if !supported {
			event.Loss = &LossReport{UnknownKinds: []string{envelope.Type}, Reason: "native kind preserved as observation"}
		}
		if err := event.Validate(); err != nil {
			return err
		}
		events = append(events, event)
		return nil
	})
	if err != nil {
		receipt.EmittedEvents = len(events)
		_ = finishReceipt(&receipt, events)
		return events, receipt, err
	}
	if err := finishReceipt(&receipt, events); err != nil {
		return events, receipt, err
	}
	return events, receipt, nil
}

func normalizeExportMessage(role string, payload json.RawMessage) json.RawMessage {
	var object map[string]any
	if json.Unmarshal(payload, &object) != nil {
		return payload
	}
	if nested, ok := object["message"].(map[string]any); ok {
		if value, ok := nested["role"]; ok {
			object["role"] = value
		}
		if value, ok := nested["content"]; ok {
			object["text"] = value
		}
	}
	if _, ok := object["role"]; !ok {
		object["role"] = role
	}
	if _, ok := object["text"]; !ok {
		if value, ok := object["content"]; ok {
			object["text"] = value
		}
	}
	normalized, err := json.Marshal(object)
	if err != nil {
		return payload
	}
	return normalized
}

func firstExportValue(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
