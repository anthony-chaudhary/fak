// Adapted from openai/codex rollout record types (Apache-2.0),
//
// Component parser and round-trip converter for Codex-style rollout JSONL
// session logs. Imported sessions remain read-only analysis artifacts: this
// file does not alter fak's own SessionCheckpoint writes and does not resume
// foreign sessions.

package agent

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

const rolloutMaxLineBytes = 32 * 1024 * 1024

var (
	// ErrRolloutMissingSessionMeta indicates a Codex rollout lacks a leading session_meta record.
	ErrRolloutMissingSessionMeta = errors.New("agent: rollout missing leading session_meta line")
	// ErrRolloutMalformedLine indicates a Codex rollout contains invalid JSON or an invalid line envelope.
	ErrRolloutMalformedLine = errors.New("agent: rollout malformed JSONL line")
)

// RolloutItemType identifies the supported typed item families parsed from a Codex rollout.
type RolloutItemType string

const (
	RolloutItemMessage            RolloutItemType = "message"
	RolloutItemFunctionCall       RolloutItemType = "function_call"
	RolloutItemFunctionCallOutput RolloutItemType = "function_call_output"
	RolloutItemCompacted          RolloutItemType = "compacted"
)

// SessionTrajectoryKind classifies one record in an imported fak session trajectory.
type SessionTrajectoryKind string

const (
	SessionTrajectoryMessage         SessionTrajectoryKind = "message"
	SessionTrajectoryToolAction      SessionTrajectoryKind = "tool_action"
	SessionTrajectoryToolObservation SessionTrajectoryKind = "tool_observation"
	SessionTrajectoryMarker          SessionTrajectoryKind = "compaction_marker"
)

// RolloutSessionMeta carries the leading session_meta payload from a Codex rollout.
type RolloutSessionMeta struct {
	ID            string          `json:"id,omitempty"`
	Timestamp     time.Time       `json:"timestamp,omitempty"`
	CWD           string          `json:"cwd,omitempty"`
	Originator    string          `json:"originator,omitempty"`
	CLIVersion    string          `json:"cli_version,omitempty"`
	ModelProvider string          `json:"model_provider,omitempty"`
	Source        json.RawMessage `json:"source,omitempty"`
	Instructions  string          `json:"instructions,omitempty"`
}

// RolloutContentPart represents one structured content block in a Codex message or output.
type RolloutContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Refusal  string `json:"refusal,omitempty"`
}

// RolloutMessage holds a parsed Codex response_item of type "message".
type RolloutMessage struct {
	ID              string               `json:"id,omitempty"`
	Role            string               `json:"role"`
	Content         []RolloutContentPart `json:"content,omitempty"`
	Text            string               `json:"-"`
	ContentIsString bool                 `json:"-"`
	Phase           string               `json:"phase,omitempty"`
}

// RolloutFunctionCall holds a parsed Codex response_item of type "function_call".
type RolloutFunctionCall struct {
	ID        string `json:"id,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	CallID    string `json:"call_id"`
	Status    string `json:"status,omitempty"`
}

// RolloutFunctionCallOutput holds a parsed Codex response_item of type "function_call_output".
type RolloutFunctionCallOutput struct {
	ID        string          `json:"id,omitempty"`
	CallID    string          `json:"call_id"`
	Output    string          `json:"-"`
	OutputRaw json.RawMessage `json:"-"`
}

// RolloutCompacted holds a parsed Codex "compacted" rollout item.
type RolloutCompacted struct {
	Message               string            `json:"message,omitempty"`
	ReplacementHistory    []RolloutRecord   `json:"-"`
	ReplacementHistoryRaw []json.RawMessage `json:"-"`
	FirstWindowID         string            `json:"first_window_id,omitempty"`
	PreviousWindowID      string            `json:"previous_window_id,omitempty"`
	WindowID              string            `json:"window_id,omitempty"`
	WindowNumber          int               `json:"window_number,omitempty"`
	WrappedInResponse     bool              `json:"-"`
}

// RolloutRecord is one typed item parsed in stream order from a Codex rollout JSONL file.
type RolloutRecord struct {
	Line               int                        `json:"line,omitempty"`
	Timestamp          time.Time                  `json:"timestamp,omitempty"`
	Type               RolloutItemType            `json:"type"`
	Message            *RolloutMessage            `json:"message,omitempty"`
	FunctionCall       *RolloutFunctionCall       `json:"function_call,omitempty"`
	FunctionCallOutput *RolloutFunctionCallOutput `json:"function_call_output,omitempty"`
	Compacted          *RolloutCompacted          `json:"compacted,omitempty"`
}

// CodexRollout is the parsed representation of one Codex-style session JSONL file.
type CodexRollout struct {
	MetaTimestamp time.Time          `json:"meta_timestamp,omitempty"`
	Meta          RolloutSessionMeta `json:"meta"`
	Records       []RolloutRecord    `json:"records"`
	Skipped       int                `json:"skipped,omitempty"`
	UnknownTypes  map[string]int     `json:"unknown_types,omitempty"`
}

// SessionTrajectoryRecord is one ordered record inside a fak imported session trajectory.
type SessionTrajectoryRecord struct {
	Seq       int                   `json:"seq"`
	Timestamp time.Time             `json:"timestamp,omitempty"`
	Kind      SessionTrajectoryKind `json:"kind"`

	ItemID          string               `json:"item_id,omitempty"`
	Role            string               `json:"role,omitempty"`
	Content         string               `json:"content,omitempty"`
	ContentParts    []RolloutContentPart `json:"content_parts,omitempty"`
	ContentIsString bool                 `json:"content_is_string,omitempty"`
	Phase           string               `json:"phase,omitempty"`

	CallID    string          `json:"call_id,omitempty"`
	ToolName  string          `json:"tool_name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Status    string          `json:"status,omitempty"`
	Output    string          `json:"output,omitempty"`
	OutputRaw json.RawMessage `json:"output_raw,omitempty"`

	MarkerKind         string            `json:"marker_kind,omitempty"`
	Summary            string            `json:"summary,omitempty"`
	ReplacementCount   int               `json:"replacement_count,omitempty"`
	ReplacementHistory []RolloutRecord   `json:"replacement_history,omitempty"`
	RawReplacement     []json.RawMessage `json:"-"`
	FirstWindowID      string            `json:"first_window_id,omitempty"`
	PreviousWindowID   string            `json:"previous_window_id,omitempty"`
	WindowID           string            `json:"window_id,omitempty"`
	WindowNumber       int               `json:"window_number,omitempty"`
	WrappedInResponse  bool              `json:"wrapped_in_response,omitempty"`
}

// SessionTrajectory holds the fak-normalized trajectory converted from a Codex rollout.
type SessionTrajectory struct {
	SessionID     string                    `json:"session_id"`
	CWD           string                    `json:"cwd,omitempty"`
	Provider      string                    `json:"model_provider,omitempty"`
	Originator    string                    `json:"originator,omitempty"`
	CLIVersion    string                    `json:"cli_version,omitempty"`
	CreatedAt     time.Time                 `json:"created_at,omitempty"`
	MetaTimestamp time.Time                 `json:"meta_timestamp,omitempty"`
	MetaSource    json.RawMessage           `json:"meta_source,omitempty"`
	Instructions  string                    `json:"instructions,omitempty"`
	Records       []SessionTrajectoryRecord `json:"records"`
}

type rolloutLineEnvelope struct {
	Timestamp string          `json:"timestamp"`
	Type      string          `json:"type"`
	Payload   json.RawMessage `json:"payload"`
}

// ParseCodexRollout parses a Codex-style rollout JSONL buffer into typed records.
// The first non-empty line must be a session_meta line; unknown subsequent item types
// are skipped and counted rather than failing the import.
func ParseCodexRollout(data []byte) (*CodexRollout, error) {
	return ParseCodexRolloutReader(bytes.NewReader(data))
}

// ParseCodexRolloutReader streams and parses a Codex-style rollout JSONL reader.
func ParseCodexRolloutReader(r io.Reader) (*CodexRollout, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil reader", ErrRolloutMissingSessionMeta)
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), rolloutMaxLineBytes)

	out := &CodexRollout{
		Records:      make([]RolloutRecord, 0),
		UnknownTypes: make(map[string]int),
	}
	lineNum := 0
	metaSeen := false

	for sc.Scan() {
		lineNum++
		raw := bytes.TrimSpace(sc.Bytes())
		if len(raw) == 0 {
			continue
		}
		if !bytes.HasPrefix(raw, []byte("{")) {
			return nil, fmt.Errorf("%w: line %d is not a JSON object", ErrRolloutMalformedLine, lineNum)
		}
		var env rolloutLineEnvelope
		if err := json.Unmarshal(raw, &env); err != nil {
			return nil, fmt.Errorf("%w: line %d: %v", ErrRolloutMalformedLine, lineNum, err)
		}
		ts, err := parseRolloutTimestamp(env.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("%w: line %d timestamp: %v", ErrRolloutMalformedLine, lineNum, err)
		}

		if !metaSeen {
			if env.Type != "session_meta" {
				return nil, fmt.Errorf("%w: first non-empty line %d has type %q", ErrRolloutMissingSessionMeta, lineNum, env.Type)
			}
			meta, err := parseRolloutSessionMeta(env.Payload)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNum, err)
			}
			out.MetaTimestamp = ts
			out.Meta = meta
			metaSeen = true
			continue
		}

		switch env.Type {
		case "session_meta":
			// Subagent rollouts embed copied parent history after the file-local
			// session_meta header. Ignore subsequent headers without replacing Meta.
			continue
		case "response_item":
			rec, unknownKey, err := parseRolloutResponseItem(env.Payload, lineNum, ts)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNum, err)
			}
			if unknownKey != "" {
				out.Skipped++
				out.UnknownTypes[unknownKey]++
				continue
			}
			out.Records = append(out.Records, rec)
		case "compacted":
			comp, err := parseRolloutCompacted(env.Payload, false)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNum, err)
			}
			out.Records = append(out.Records, RolloutRecord{
				Line:      lineNum,
				Timestamp: ts,
				Type:      RolloutItemCompacted,
				Compacted: &comp,
			})
		default:
			key := env.Type
			if strings.TrimSpace(key) == "" {
				key = "<missing>"
			}
			out.Skipped++
			out.UnknownTypes[key]++
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("agent: read rollout stream: %w", err)
	}
	if !metaSeen {
		return nil, fmt.Errorf("%w: empty rollout stream", ErrRolloutMissingSessionMeta)
	}
	return out, nil
}

func parseRolloutTimestamp(raw string) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, nil
	}
	if ts, err := time.Parse(time.RFC3339Nano, raw); err == nil {
		return ts.UTC(), nil
	}
	if ts, err := time.Parse(time.RFC3339, raw); err == nil {
		return ts.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("invalid RFC3339 timestamp %q", raw)
}

func parseRolloutSessionMeta(raw json.RawMessage) (RolloutSessionMeta, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return RolloutSessionMeta{}, fmt.Errorf("%w: session_meta payload is missing", ErrRolloutMissingSessionMeta)
	}
	if !bytes.HasPrefix(trimmed, []byte("{")) {
		return RolloutSessionMeta{}, fmt.Errorf("%w: session_meta payload is not a JSON object", ErrRolloutMalformedLine)
	}
	var payload struct {
		ID            string          `json:"id"`
		SessionID     string          `json:"session_id"`
		Timestamp     string          `json:"timestamp"`
		CWD           string          `json:"cwd"`
		Originator    string          `json:"originator"`
		CLIVersion    string          `json:"cli_version"`
		ModelProvider string          `json:"model_provider"`
		Source        json.RawMessage `json:"source"`
		Instructions  string          `json:"instructions"`
	}
	if err := json.Unmarshal(trimmed, &payload); err != nil {
		return RolloutSessionMeta{}, fmt.Errorf("%w: session_meta payload: %v", ErrRolloutMalformedLine, err)
	}
	ts, err := parseRolloutTimestamp(payload.Timestamp)
	if err != nil {
		return RolloutSessionMeta{}, fmt.Errorf("%w: session_meta payload timestamp: %v", ErrRolloutMalformedLine, err)
	}
	id := payload.ID
	if strings.TrimSpace(id) == "" {
		id = payload.SessionID
	}
	var sourceCopy json.RawMessage
	if src := bytes.TrimSpace(payload.Source); len(src) > 0 && !bytes.Equal(src, []byte("null")) {
		sourceCopy = append([]byte(nil), src...)
	}
	return RolloutSessionMeta{
		ID:            id,
		Timestamp:     ts,
		CWD:           payload.CWD,
		Originator:    payload.Originator,
		CLIVersion:    payload.CLIVersion,
		ModelProvider: payload.ModelProvider,
		Source:        sourceCopy,
		Instructions:  payload.Instructions,
	}, nil
}

func parseRolloutResponseItem(raw json.RawMessage, lineNum int, ts time.Time) (RolloutRecord, string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || !bytes.HasPrefix(trimmed, []byte("{")) {
		return RolloutRecord{}, "", fmt.Errorf("%w: response_item payload must be a JSON object", ErrRolloutMalformedLine)
	}
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(trimmed, &head); err != nil {
		return RolloutRecord{}, "", fmt.Errorf("%w: response_item payload: %v", ErrRolloutMalformedLine, err)
	}
	switch head.Type {
	case string(RolloutItemMessage):
		msg, err := parseRolloutMessage(trimmed)
		if err != nil {
			return RolloutRecord{}, "", err
		}
		return RolloutRecord{
			Line:      lineNum,
			Timestamp: ts,
			Type:      RolloutItemMessage,
			Message:   &msg,
		}, "", nil
	case string(RolloutItemFunctionCall):
		fc, err := parseRolloutFunctionCall(trimmed)
		if err != nil {
			return RolloutRecord{}, "", err
		}
		return RolloutRecord{
			Line:         lineNum,
			Timestamp:    ts,
			Type:         RolloutItemFunctionCall,
			FunctionCall: &fc,
		}, "", nil
	case string(RolloutItemFunctionCallOutput):
		fco, err := parseRolloutFunctionCallOutput(trimmed)
		if err != nil {
			return RolloutRecord{}, "", err
		}
		return RolloutRecord{
			Line:               lineNum,
			Timestamp:          ts,
			Type:               RolloutItemFunctionCallOutput,
			FunctionCallOutput: &fco,
		}, "", nil
	case string(RolloutItemCompacted):
		comp, err := parseRolloutCompacted(trimmed, true)
		if err != nil {
			return RolloutRecord{}, "", err
		}
		return RolloutRecord{
			Line:      lineNum,
			Timestamp: ts,
			Type:      RolloutItemCompacted,
			Compacted: &comp,
		}, "", nil
	default:
		sub := head.Type
		if strings.TrimSpace(sub) == "" {
			sub = "<missing>"
		}
		return RolloutRecord{}, "response_item/" + sub, nil
	}
}

func parseRolloutMessage(raw []byte) (RolloutMessage, error) {
	var wire struct {
		ID      string          `json:"id"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
		Phase   string          `json:"phase"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return RolloutMessage{}, fmt.Errorf("%w: message payload: %v", ErrRolloutMalformedLine, err)
	}
	parts, text, isStr, err := decodeRolloutContent(wire.Content)
	if err != nil {
		return RolloutMessage{}, fmt.Errorf("%w: message content: %v", ErrRolloutMalformedLine, err)
	}
	return RolloutMessage{
		ID:              wire.ID,
		Role:            wire.Role,
		Content:         parts,
		Text:            text,
		ContentIsString: isStr,
		Phase:           wire.Phase,
	}, nil
}

func decodeRolloutContent(raw json.RawMessage) ([]RolloutContentPart, string, bool, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, "", false, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, "", false, err
		}
		return []RolloutContentPart{{Type: "text", Text: s}}, s, true, nil
	}
	if trimmed[0] == '[' {
		var parts []RolloutContentPart
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return nil, "", false, err
		}
		var texts []string
		for _, part := range parts {
			if part.Text != "" {
				texts = append(texts, part.Text)
			} else if part.Refusal != "" {
				texts = append(texts, part.Refusal)
			}
		}
		return parts, strings.Join(texts, "\n"), false, nil
	}
	return nil, "", false, errors.New("content must be string or array")
}

func parseRolloutFunctionCall(raw []byte) (RolloutFunctionCall, error) {
	var wire struct {
		ID        string          `json:"id"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		CallID    string          `json:"call_id"`
		Status    string          `json:"status"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return RolloutFunctionCall{}, fmt.Errorf("%w: function_call payload: %v", ErrRolloutMalformedLine, err)
	}
	args, err := decodeRolloutJSONStringOrRaw(wire.Arguments)
	if err != nil {
		return RolloutFunctionCall{}, fmt.Errorf("%w: function_call arguments: %v", ErrRolloutMalformedLine, err)
	}
	return RolloutFunctionCall{
		ID:        wire.ID,
		Name:      wire.Name,
		Arguments: args,
		CallID:    wire.CallID,
		Status:    wire.Status,
	}, nil
}

func parseRolloutFunctionCallOutput(raw []byte) (RolloutFunctionCallOutput, error) {
	var wire struct {
		ID     string          `json:"id"`
		CallID string          `json:"call_id"`
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return RolloutFunctionCallOutput{}, fmt.Errorf("%w: function_call_output payload: %v", ErrRolloutMalformedLine, err)
	}
	text, rawCopy, err := decodeRolloutOutput(wire.Output)
	if err != nil {
		return RolloutFunctionCallOutput{}, fmt.Errorf("%w: function_call_output output: %v", ErrRolloutMalformedLine, err)
	}
	return RolloutFunctionCallOutput{
		ID:        wire.ID,
		CallID:    wire.CallID,
		Output:    text,
		OutputRaw: rawCopy,
	}, nil
}

func decodeRolloutJSONStringOrRaw(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", err
		}
		return s, nil
	}
	return string(trimmed), nil
}

func decodeRolloutOutput(raw json.RawMessage) (string, json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", nil, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", nil, err
		}
		return s, nil, nil
	}
	rawCopy := append([]byte(nil), trimmed...)
	if trimmed[0] == '[' {
		var parts []RolloutContentPart
		if err := json.Unmarshal(trimmed, &parts); err == nil {
			var texts []string
			for _, part := range parts {
				if part.Text != "" {
					texts = append(texts, part.Text)
				}
			}
			if len(texts) > 0 {
				return strings.Join(texts, "\n"), rawCopy, nil
			}
		}
	}
	if trimmed[0] == '{' {
		var obj struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal(trimmed, &obj); err == nil && obj.Content != "" {
			return obj.Content, rawCopy, nil
		}
	}
	return string(trimmed), rawCopy, nil
}

func parseRolloutCompacted(raw json.RawMessage, wrapped bool) (RolloutCompacted, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return RolloutCompacted{WrappedInResponse: wrapped}, nil
	}
	if !bytes.HasPrefix(trimmed, []byte("{")) {
		return RolloutCompacted{}, fmt.Errorf("%w: compacted payload must be a JSON object", ErrRolloutMalformedLine)
	}
	var wire struct {
		Message            string            `json:"message"`
		ReplacementHistory []json.RawMessage `json:"replacement_history"`
		FirstWindowID      string            `json:"first_window_id"`
		PreviousWindowID   string            `json:"previous_window_id"`
		WindowID           string            `json:"window_id"`
		WindowNumber       int               `json:"window_number"`
	}
	if err := json.Unmarshal(trimmed, &wire); err != nil {
		return RolloutCompacted{}, fmt.Errorf("%w: compacted payload: %v", ErrRolloutMalformedLine, err)
	}
	comp := RolloutCompacted{
		Message:           wire.Message,
		FirstWindowID:     wire.FirstWindowID,
		PreviousWindowID:  wire.PreviousWindowID,
		WindowID:          wire.WindowID,
		WindowNumber:      wire.WindowNumber,
		WrappedInResponse: wrapped,
	}
	if len(wire.ReplacementHistory) > 0 {
		comp.ReplacementHistoryRaw = make([]json.RawMessage, len(wire.ReplacementHistory))
		for i, itemRaw := range wire.ReplacementHistory {
			trimmedItem := bytes.TrimSpace(itemRaw)
			comp.ReplacementHistoryRaw[i] = append([]byte(nil), trimmedItem...)
			rec, unknownKey, err := parseRolloutResponseItem(trimmedItem, 0, time.Time{})
			if err != nil {
				return RolloutCompacted{}, err
			}
			if unknownKey == "" {
				comp.ReplacementHistory = append(comp.ReplacementHistory, rec)
			}
		}
	}
	return comp, nil
}

// EncodeCodexRollout serializes a CodexRollout back into Codex-style JSONL bytes.
func EncodeCodexRollout(rollout *CodexRollout) ([]byte, error) {
	if rollout == nil {
		return nil, fmt.Errorf("%w: nil rollout", ErrRolloutMissingSessionMeta)
	}
	var buf bytes.Buffer
	writeLine := func(ts time.Time, typ string, payload any) error {
		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		env := rolloutLineEnvelope{
			Type:    typ,
			Payload: payloadBytes,
		}
		if !ts.IsZero() {
			env.Timestamp = ts.UTC().Format(time.RFC3339Nano)
		}
		lineBytes, err := json.Marshal(env)
		if err != nil {
			return err
		}
		buf.Write(lineBytes)
		buf.WriteByte('\n')
		return nil
	}

	metaPayload := map[string]any{}
	if rollout.Meta.ID != "" {
		metaPayload["id"] = rollout.Meta.ID
	}
	if !rollout.Meta.Timestamp.IsZero() {
		metaPayload["timestamp"] = rollout.Meta.Timestamp.UTC().Format(time.RFC3339Nano)
	}
	if rollout.Meta.CWD != "" {
		metaPayload["cwd"] = rollout.Meta.CWD
	}
	if rollout.Meta.Originator != "" {
		metaPayload["originator"] = rollout.Meta.Originator
	}
	if rollout.Meta.CLIVersion != "" {
		metaPayload["cli_version"] = rollout.Meta.CLIVersion
	}
	if rollout.Meta.ModelProvider != "" {
		metaPayload["model_provider"] = rollout.Meta.ModelProvider
	}
	if len(rollout.Meta.Source) > 0 {
		metaPayload["source"] = rollout.Meta.Source
	}
	if rollout.Meta.Instructions != "" {
		metaPayload["instructions"] = rollout.Meta.Instructions
	}
	if err := writeLine(rollout.MetaTimestamp, "session_meta", metaPayload); err != nil {
		return nil, err
	}

	for i, rec := range rollout.Records {
		typ, payload, err := marshalRolloutRecordPayload(rec)
		if err != nil {
			return nil, fmt.Errorf("record %d: %w", i, err)
		}
		if err := writeLine(rec.Timestamp, typ, payload); err != nil {
			return nil, fmt.Errorf("record %d: %w", i, err)
		}
	}
	return buf.Bytes(), nil
}

func marshalRolloutRecordPayload(rec RolloutRecord) (string, any, error) {
	switch rec.Type {
	case RolloutItemMessage:
		if rec.Message == nil {
			return "", nil, errors.New("message record missing payload")
		}
		return "response_item", marshalRolloutMessageMap(*rec.Message), nil
	case RolloutItemFunctionCall:
		if rec.FunctionCall == nil {
			return "", nil, errors.New("function_call record missing payload")
		}
		m := map[string]any{
			"type":      string(RolloutItemFunctionCall),
			"name":      rec.FunctionCall.Name,
			"arguments": rec.FunctionCall.Arguments,
			"call_id":   rec.FunctionCall.CallID,
		}
		if rec.FunctionCall.ID != "" {
			m["id"] = rec.FunctionCall.ID
		}
		if rec.FunctionCall.Status != "" {
			m["status"] = rec.FunctionCall.Status
		}
		return "response_item", m, nil
	case RolloutItemFunctionCallOutput:
		if rec.FunctionCallOutput == nil {
			return "", nil, errors.New("function_call_output record missing payload")
		}
		m := map[string]any{
			"type":    string(RolloutItemFunctionCallOutput),
			"call_id": rec.FunctionCallOutput.CallID,
		}
		if rec.FunctionCallOutput.ID != "" {
			m["id"] = rec.FunctionCallOutput.ID
		}
		if len(rec.FunctionCallOutput.OutputRaw) > 0 {
			m["output"] = rec.FunctionCallOutput.OutputRaw
		} else {
			m["output"] = rec.FunctionCallOutput.Output
		}
		return "response_item", m, nil
	case RolloutItemCompacted:
		if rec.Compacted == nil {
			return "", nil, errors.New("compacted record missing payload")
		}
		m := map[string]any{}
		if rec.Compacted.Message != "" {
			m["message"] = rec.Compacted.Message
		}
		if rec.Compacted.FirstWindowID != "" {
			m["first_window_id"] = rec.Compacted.FirstWindowID
		}
		if rec.Compacted.PreviousWindowID != "" {
			m["previous_window_id"] = rec.Compacted.PreviousWindowID
		}
		if rec.Compacted.WindowID != "" {
			m["window_id"] = rec.Compacted.WindowID
		}
		if rec.Compacted.WindowNumber != 0 {
			m["window_number"] = rec.Compacted.WindowNumber
		}
		if len(rec.Compacted.ReplacementHistoryRaw) > 0 {
			m["replacement_history"] = rec.Compacted.ReplacementHistoryRaw
		} else if len(rec.Compacted.ReplacementHistory) > 0 {
			hist := make([]any, 0, len(rec.Compacted.ReplacementHistory))
			for _, sub := range rec.Compacted.ReplacementHistory {
				_, subPayload, err := marshalRolloutRecordPayload(sub)
				if err != nil {
					return "", nil, err
				}
				hist = append(hist, subPayload)
			}
			m["replacement_history"] = hist
		}
		if rec.Compacted.WrappedInResponse {
			m["type"] = string(RolloutItemCompacted)
			return "response_item", m, nil
		}
		return "compacted", m, nil
	default:
		return "", nil, fmt.Errorf("unsupported rollout item type %q", rec.Type)
	}
}

func marshalRolloutMessageMap(msg RolloutMessage) map[string]any {
	m := map[string]any{
		"type": string(RolloutItemMessage),
		"role": msg.Role,
	}
	if msg.ID != "" {
		m["id"] = msg.ID
	}
	if msg.Phase != "" {
		m["phase"] = msg.Phase
	}
	if msg.ContentIsString {
		m["content"] = msg.Text
	} else if len(msg.Content) > 0 {
		m["content"] = msg.Content
	} else if msg.Text != "" {
		m["content"] = msg.Text
	}
	return m
}

// ConvertRolloutToTrajectory converts a parsed CodexRollout into a fak SessionTrajectory.
// Converting an empty record set returns an empty SessionTrajectory with no error.
func ConvertRolloutToTrajectory(rollout *CodexRollout) (*SessionTrajectory, error) {
	if rollout == nil {
		return &SessionTrajectory{
			Records: make([]SessionTrajectoryRecord, 0),
		}, nil
	}
	createdAt := rollout.Meta.Timestamp
	if createdAt.IsZero() {
		createdAt = rollout.MetaTimestamp
	}
	var sourceCopy json.RawMessage
	if len(rollout.Meta.Source) > 0 {
		sourceCopy = append([]byte(nil), rollout.Meta.Source...)
	}
	traj := &SessionTrajectory{
		SessionID:     rollout.Meta.ID,
		CWD:           rollout.Meta.CWD,
		Provider:      rollout.Meta.ModelProvider,
		Originator:    rollout.Meta.Originator,
		CLIVersion:    rollout.Meta.CLIVersion,
		CreatedAt:     createdAt,
		MetaTimestamp: rollout.MetaTimestamp,
		MetaSource:    sourceCopy,
		Instructions:  rollout.Meta.Instructions,
		Records:       make([]SessionTrajectoryRecord, 0, len(rollout.Records)),
	}
	for i, rec := range rollout.Records {
		converted, err := rolloutRecordToTrajectory(i+1, rec)
		if err != nil {
			return nil, fmt.Errorf("record %d: %w", i, err)
		}
		traj.Records = append(traj.Records, converted)
	}
	return traj, nil
}

func rolloutRecordToTrajectory(seq int, rec RolloutRecord) (SessionTrajectoryRecord, error) {
	out := SessionTrajectoryRecord{
		Seq:       seq,
		Timestamp: rec.Timestamp,
	}
	switch rec.Type {
	case RolloutItemMessage:
		if rec.Message == nil {
			return SessionTrajectoryRecord{}, errors.New("message record missing payload")
		}
		out.Kind = SessionTrajectoryMessage
		out.ItemID = rec.Message.ID
		out.Role = rec.Message.Role
		out.Content = rec.Message.Text
		out.ContentIsString = rec.Message.ContentIsString
		out.Phase = rec.Message.Phase
		if len(rec.Message.Content) > 0 {
			out.ContentParts = append([]RolloutContentPart(nil), rec.Message.Content...)
		}
		return out, nil
	case RolloutItemFunctionCall:
		if rec.FunctionCall == nil {
			return SessionTrajectoryRecord{}, errors.New("function_call record missing payload")
		}
		out.Kind = SessionTrajectoryToolAction
		out.ItemID = rec.FunctionCall.ID
		out.ToolName = rec.FunctionCall.Name
		out.Arguments = rec.FunctionCall.Arguments
		out.CallID = rec.FunctionCall.CallID
		out.Status = rec.FunctionCall.Status
		return out, nil
	case RolloutItemFunctionCallOutput:
		if rec.FunctionCallOutput == nil {
			return SessionTrajectoryRecord{}, errors.New("function_call_output record missing payload")
		}
		out.Kind = SessionTrajectoryToolObservation
		out.ItemID = rec.FunctionCallOutput.ID
		out.CallID = rec.FunctionCallOutput.CallID
		out.Output = rec.FunctionCallOutput.Output
		if len(rec.FunctionCallOutput.OutputRaw) > 0 {
			out.OutputRaw = append([]byte(nil), rec.FunctionCallOutput.OutputRaw...)
		}
		return out, nil
	case RolloutItemCompacted:
		if rec.Compacted == nil {
			return SessionTrajectoryRecord{}, errors.New("compacted record missing payload")
		}
		out.Kind = SessionTrajectoryMarker
		out.MarkerKind = string(RolloutItemCompacted)
		out.Summary = rec.Compacted.Message
		out.FirstWindowID = rec.Compacted.FirstWindowID
		out.PreviousWindowID = rec.Compacted.PreviousWindowID
		out.WindowID = rec.Compacted.WindowID
		out.WindowNumber = rec.Compacted.WindowNumber
		out.WrappedInResponse = rec.Compacted.WrappedInResponse
		if len(rec.Compacted.ReplacementHistory) > 0 {
			out.ReplacementHistory = append([]RolloutRecord(nil), rec.Compacted.ReplacementHistory...)
		}
		if len(rec.Compacted.ReplacementHistoryRaw) > 0 {
			out.RawReplacement = make([]json.RawMessage, len(rec.Compacted.ReplacementHistoryRaw))
			for i, raw := range rec.Compacted.ReplacementHistoryRaw {
				out.RawReplacement[i] = append([]byte(nil), raw...)
			}
			out.ReplacementCount = len(rec.Compacted.ReplacementHistoryRaw)
		} else {
			out.ReplacementCount = len(rec.Compacted.ReplacementHistory)
		}
		return out, nil
	default:
		return SessionTrajectoryRecord{}, fmt.Errorf("unsupported rollout item type %q", rec.Type)
	}
}

// ConvertTrajectoryToRollout converts a SessionTrajectory back into a CodexRollout
// so imported trajectories can round-trip back through Codex rollout records and JSONL.
func ConvertTrajectoryToRollout(traj *SessionTrajectory) (*CodexRollout, error) {
	if traj == nil {
		return &CodexRollout{
			Records:      make([]RolloutRecord, 0),
			UnknownTypes: make(map[string]int),
		}, nil
	}
	var sourceCopy json.RawMessage
	if len(traj.MetaSource) > 0 {
		sourceCopy = append([]byte(nil), traj.MetaSource...)
	}
	out := &CodexRollout{
		MetaTimestamp: traj.MetaTimestamp,
		Meta: RolloutSessionMeta{
			ID:            traj.SessionID,
			Timestamp:     traj.CreatedAt,
			CWD:           traj.CWD,
			Originator:    traj.Originator,
			CLIVersion:    traj.CLIVersion,
			ModelProvider: traj.Provider,
			Source:        sourceCopy,
			Instructions:  traj.Instructions,
		},
		Records:      make([]RolloutRecord, 0, len(traj.Records)),
		UnknownTypes: make(map[string]int),
	}
	for i, rec := range traj.Records {
		rolloutRec, err := trajectoryRecordToRollout(rec)
		if err != nil {
			return nil, fmt.Errorf("record %d: %w", i, err)
		}
		out.Records = append(out.Records, rolloutRec)
	}
	return out, nil
}

func trajectoryRecordToRollout(rec SessionTrajectoryRecord) (RolloutRecord, error) {
	out := RolloutRecord{
		Timestamp: rec.Timestamp,
	}
	switch rec.Kind {
	case SessionTrajectoryMessage:
		msg := RolloutMessage{
			ID:              rec.ItemID,
			Role:            rec.Role,
			Text:            rec.Content,
			ContentIsString: rec.ContentIsString,
			Phase:           rec.Phase,
		}
		if len(rec.ContentParts) > 0 {
			msg.Content = append([]RolloutContentPart(nil), rec.ContentParts...)
		} else if rec.Content != "" {
			msg.Content = []RolloutContentPart{{Type: "text", Text: rec.Content}}
			msg.ContentIsString = true
		}
		out.Type = RolloutItemMessage
		out.Message = &msg
		return out, nil
	case SessionTrajectoryToolAction:
		out.Type = RolloutItemFunctionCall
		out.FunctionCall = &RolloutFunctionCall{
			ID:        rec.ItemID,
			Name:      rec.ToolName,
			Arguments: rec.Arguments,
			CallID:    rec.CallID,
			Status:    rec.Status,
		}
		return out, nil
	case SessionTrajectoryToolObservation:
		fco := RolloutFunctionCallOutput{
			ID:     rec.ItemID,
			CallID: rec.CallID,
			Output: rec.Output,
		}
		if len(rec.OutputRaw) > 0 {
			fco.OutputRaw = append([]byte(nil), rec.OutputRaw...)
		}
		out.Type = RolloutItemFunctionCallOutput
		out.FunctionCallOutput = &fco
		return out, nil
	case SessionTrajectoryMarker:
		comp := RolloutCompacted{
			Message:           rec.Summary,
			FirstWindowID:     rec.FirstWindowID,
			PreviousWindowID:  rec.PreviousWindowID,
			WindowID:          rec.WindowID,
			WindowNumber:      rec.WindowNumber,
			WrappedInResponse: rec.WrappedInResponse,
		}
		if len(rec.ReplacementHistory) > 0 {
			comp.ReplacementHistory = append([]RolloutRecord(nil), rec.ReplacementHistory...)
		}
		if len(rec.RawReplacement) > 0 {
			comp.ReplacementHistoryRaw = make([]json.RawMessage, len(rec.RawReplacement))
			for i, raw := range rec.RawReplacement {
				comp.ReplacementHistoryRaw[i] = append([]byte(nil), raw...)
			}
		}
		out.Type = RolloutItemCompacted
		out.Compacted = &comp
		return out, nil
	default:
		return RolloutRecord{}, fmt.Errorf("unsupported trajectory record kind %q", rec.Kind)
	}
}

// ToTrajectoryEvents projects the imported SessionTrajectory into canonical
// trajectory.Event values so existing trajectory analysis tools can inspect it.
func (t *SessionTrajectory) ToTrajectoryEvents() ([]trajectory.Event, error) {
	if t == nil || len(t.Records) == 0 {
		return []trajectory.Event{}, nil
	}
	sessionID := strings.TrimSpace(t.SessionID)
	if sessionID == "" {
		sessionID = "codex-rollout-import"
	}
	events := make([]trajectory.Event, 0, len(t.Records))
	for i, rec := range t.Records {
		seq := rec.Seq
		if seq <= 0 {
			seq = i + 1
		}
		ts := rec.Timestamp
		if ts.IsZero() {
			ts = time.Unix(0, int64(seq)).UTC()
		}
		var (
			kind          trajectory.EventKind
			action        string
			sourceEventID string
			parents       []string
			body          any
		)
		switch rec.Kind {
		case SessionTrajectoryMessage:
			kind = trajectory.EventMessage
			action = "completed"
			sourceEventID = rec.ItemID
			body = map[string]any{
				"role": rec.Role,
				"text": rec.Content,
			}
		case SessionTrajectoryToolAction:
			kind = trajectory.EventTool
			action = "proposed"
			sourceEventID = rec.CallID
			if sourceEventID == "" {
				sourceEventID = rec.ItemID
			}
			body = map[string]any{
				"call_id":   rec.CallID,
				"name":      rec.ToolName,
				"arguments": rec.Arguments,
			}
		case SessionTrajectoryToolObservation:
			kind = trajectory.EventTool
			action = "completed"
			sourceEventID = rec.CallID
			if sourceEventID == "" {
				sourceEventID = rec.ItemID
			}
			if rec.CallID != "" {
				parents = []string{rec.CallID}
			}
			body = map[string]any{
				"call_id": rec.CallID,
				"output":  rec.Output,
			}
		case SessionTrajectoryMarker:
			kind = trajectory.EventCheckpoint
			action = "compacted"
			sourceEventID = rec.WindowID
			body = map[string]any{
				"marker_kind":       rec.MarkerKind,
				"summary":           rec.Summary,
				"replacement_count": rec.ReplacementCount,
			}
		default:
			return nil, fmt.Errorf("record %d: unsupported trajectory record kind %q", i, rec.Kind)
		}

		payload, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("record %d: marshal payload: %w", i, err)
		}
		eventID := fmt.Sprintf("codex-rollout-%d", seq)
		if sourceEventID != "" {
			eventID = fmt.Sprintf("codex-rollout-%d-%s", seq, sourceEventID)
		}
		ev := trajectory.Event{
			Schema:         trajectory.EventSchema,
			ID:             eventID,
			ConversationID: sessionID,
			Kind:           kind,
			Action:         action,
			Timestamp:      ts,
			Sequence:       uint64(seq),
			ParentIDs:      parents,
			Visibility:     trajectory.VisibilityDeveloper,
			Source: trajectory.EventSource{
				Type:           "codex-rollout-jsonl",
				SessionID:      sessionID,
				EventID:        sourceEventID,
				OrderingKey:    strconv.Itoa(seq),
				Adapter:        "codex-rollout-import",
				AdapterVersion: "1",
			},
			Payload: payload,
		}
		if err := ev.Validate(); err != nil {
			return nil, fmt.Errorf("record %d: %w", i, err)
		}
		events = append(events, ev)
	}
	return events, nil
}

// ToSessionCheckpoint projects an imported SessionTrajectory into an in-memory,
// read-only SessionCheckpoint compatible with SessionCheckpoint consumers.
// It never writes to disk and marks Status as "imported".
func (t *SessionTrajectory) ToSessionCheckpoint() SessionCheckpoint {
	if t == nil {
		return SessionCheckpoint{
			Version: SessionCheckpointVersion,
			Status:  "imported",
		}
	}
	cp := SessionCheckpoint{
		Version:   SessionCheckpointVersion,
		SessionID: t.SessionID,
		CWD:       t.CWD,
		Provider:  t.Provider,
		CreatedAt: t.CreatedAt,
		UpdatedAt: t.CreatedAt,
		Status:    "imported",
	}
	callNames := make(map[string]string)
	for _, rec := range t.Records {
		if !rec.Timestamp.IsZero() && (cp.UpdatedAt.IsZero() || rec.Timestamp.After(cp.UpdatedAt)) {
			cp.UpdatedAt = rec.Timestamp
		}
		switch rec.Kind {
		case SessionTrajectoryMessage:
			role := rec.Role
			if role == "developer" {
				role = RoleSystem
			}
			if role == RoleUser && cp.Task == "" {
				cp.Task = rec.Content
			}
			cp.Messages = append(cp.Messages, Message{
				Role:    role,
				Content: rec.Content,
			})
		case SessionTrajectoryToolAction:
			if rec.CallID != "" && rec.ToolName != "" {
				callNames[rec.CallID] = rec.ToolName
			}
			cp.Turn++
			call := ToolCall{
				ID:   rec.CallID,
				Type: "function",
				Function: Func{
					Name:      rec.ToolName,
					Arguments: rec.Arguments,
				},
			}
			if n := len(cp.Messages); n > 0 && cp.Messages[n-1].Role == RoleAssistant {
				cp.Messages[n-1].ToolCalls = append(cp.Messages[n-1].ToolCalls, call)
			} else {
				cp.Messages = append(cp.Messages, Message{
					Role:      RoleAssistant,
					ToolCalls: []ToolCall{call},
				})
			}
		case SessionTrajectoryToolObservation:
			cp.Messages = append(cp.Messages, Message{
				Role:       RoleTool,
				ToolCallID: rec.CallID,
				Name:       callNames[rec.CallID],
				Content:    rec.Output,
			})
		}
	}
	return cp
}
