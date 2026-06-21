package agent

// anthropic_server.go is the SERVER half of the Anthropic Messages wire. The
// adapter in adapters.go encodes a canonical transcript INTO an Anthropic request
// and parses an Anthropic response back (the CLIENT direction, for `fak agent`).
// This file is the inverse: it DECODES an inbound /v1/messages request (from a
// Claude-Code-shaped client) into the canonical Message/ToolDef vocabulary, and
// builds the Anthropic response content blocks back out of a Completion.
//
// It exists so `fak serve` can expose a native POST /v1/messages front door:
// point Claude Code's ANTHROPIC_BASE_URL at the gateway and every tool call the
// local model proposes is adjudicated by the kernel before Claude Code sees it.
// The gateway owns the HTTP + SSE framing; this file owns the wire SHAPE only
// (no net/http), keeping all Anthropic-format knowledge in one package.

import (
	"encoding/json"
	"strings"
)

// AnthropicMessagesRequest is an inbound /v1/messages body decoded into the
// canonical transcript vocabulary. System is folded to a single string (a leading
// RoleSystem message is already prepended to Messages); the separate field is kept
// for token estimation. Stream mirrors the request's "stream":true.
type AnthropicMessagesRequest struct {
	Model         string
	System        string
	Messages      []Message
	Tools         []ToolDef
	MaxTokens     int
	Temperature   float64
	TopP          *float64
	TopK          *int
	StopSequences []string
	Stream        bool
	// Raw is the inbound request body, byte-for-byte. The anthropic→anthropic
	// passthrough path forwards these bytes verbatim to the real Anthropic API so
	// the client's prompt-cache prefix survives intact (a real cache hit). Set by
	// DecodeAnthropicMessagesRequest; otherwise unused.
	Raw []byte
}

// anthropicInbound mirrors the subset of the Messages API request Claude Code
// sends. System and message Content are json.RawMessage because each may be a bare
// string OR an array of typed blocks; tool input_schema is passed through verbatim.
type anthropicInbound struct {
	Model         string                    `json:"model"`
	MaxTokens     int                       `json:"max_tokens"`
	Temperature   *float64                  `json:"temperature"`
	TopP          *float64                  `json:"top_p"`
	TopK          *int                      `json:"top_k"`
	StopSequences []string                  `json:"stop_sequences"`
	System        json.RawMessage           `json:"system"`
	Messages      []anthropicInboundMessage `json:"messages"`
	Tools         []anthropicInboundTool    `json:"tools"`
	Stream        bool                      `json:"stream"`
}

type anthropicInboundMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicInboundBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// tool_use (assistant)
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
	// tool_result (user)
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

type anthropicInboundTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// DecodeAnthropicMessagesRequest parses an inbound Anthropic /v1/messages body into
// the canonical transcript the gateway planner consumes. It is the structural
// inverse of anthropicAdapter.MarshalRequest: a `system` (string or block array)
// becomes a leading RoleSystem message; assistant `tool_use` blocks become
// ToolCalls (id preserved); user `tool_result` blocks become RoleTool messages
// keyed by tool_use_id so the kernel's per-trace ledger correlates them.
func DecodeAnthropicMessagesRequest(raw []byte) (*AnthropicMessagesRequest, error) {
	var in anthropicInbound
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, err
	}
	out := &AnthropicMessagesRequest{
		Model:         in.Model,
		MaxTokens:     in.MaxTokens,
		TopP:          in.TopP,
		TopK:          in.TopK,
		StopSequences: in.StopSequences,
		Stream:        in.Stream,
		System:        parseAnthropicText(in.System),
		Raw:           raw,
	}
	if in.Temperature != nil {
		out.Temperature = *in.Temperature
	}
	if out.System != "" {
		out.Messages = append(out.Messages, Message{Role: RoleSystem, Content: out.System})
	}
	for _, m := range in.Messages {
		out.Messages = append(out.Messages, decodeAnthropicMessage(m)...)
	}
	for _, t := range in.Tools {
		out.Tools = append(out.Tools, ToolDef{
			Type: "function",
			Function: ToolDefFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	return out, nil
}

// decodeAnthropicMessage converts one inbound message into zero or more canonical
// messages. A user turn carrying tool_result blocks fans them out into RoleTool
// messages (one per result, in order), with any free text emitted as a trailing
// RoleUser message — the shape the OpenAI upstream adapter expects.
func decodeAnthropicMessage(m anthropicInboundMessage) []Message {
	// content may be a bare string (the common simple-prompt case).
	if s, ok := asJSONString(m.Content); ok {
		if strings.TrimSpace(s) == "" {
			return nil
		}
		return []Message{{Role: canonRole(m.Role), Content: s}}
	}
	var blocks []anthropicInboundBlock
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		return nil
	}

	switch m.Role {
	case "assistant":
		var text strings.Builder
		var calls []ToolCall
		for _, b := range blocks {
			switch b.Type {
			case "text":
				appendText(&text, b.Text)
			case "tool_use":
				calls = append(calls, ToolCall{
					ID:       b.ID,
					Type:     "function",
					Function: Func{Name: b.Name, Arguments: inputToArgs(b.Input)},
				})
			}
		}
		msg := Message{Role: RoleAssistant, Content: text.String(), ToolCalls: calls}
		if msg.Content == "" && len(msg.ToolCalls) == 0 {
			return nil
		}
		return []Message{msg}
	default: // user (and any other role): text + tool_result fan-out
		var msgs []Message
		var text strings.Builder
		for _, b := range blocks {
			switch b.Type {
			case "tool_result":
				msgs = append(msgs, Message{
					Role:       RoleTool,
					ToolCallID: b.ToolUseID,
					Content:    parseAnthropicText(b.Content),
				})
			case "text":
				appendText(&text, b.Text)
			}
		}
		if text.Len() > 0 {
			msgs = append(msgs, Message{Role: RoleUser, Content: text.String()})
		}
		return msgs
	}
}

// --- outbound: a Completion's assistant turn -> Anthropic content blocks --------

// AnthropicBlockOut is one rendered response content block (text or tool_use). The
// gateway serializes these either into a buffered message object or as the
// content_block_* SSE events.
type AnthropicBlockOut struct {
	Type  string          `json:"type"`            // "text" | "tool_use"
	Text  string          `json:"text,omitempty"`  // type=text
	ID    string          `json:"id,omitempty"`    // type=tool_use
	Name  string          `json:"name,omitempty"`  // type=tool_use
	Input json.RawMessage `json:"input,omitempty"` // type=tool_use (always a JSON object)
}

// AnthropicResponseBlocks renders the (post-adjudication) assistant message as
// ordered Anthropic content blocks: a leading text block when there is prose, then
// one tool_use block per surviving tool call (id preserved for the result round-trip).
func AnthropicResponseBlocks(m Message) []AnthropicBlockOut {
	blocks := make([]AnthropicBlockOut, 0, 1+len(m.ToolCalls))
	if strings.TrimSpace(m.Content) != "" {
		blocks = append(blocks, AnthropicBlockOut{Type: "text", Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		blocks = append(blocks, AnthropicBlockOut{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(inputObject(tc.Function.Arguments)),
		})
	}
	return blocks
}

// AnthropicStopReason maps the canonical finish reason onto the Messages API
// vocabulary. hasToolUse is authoritative: "tool_use" is returned ONLY when a tool
// call actually SURVIVED adjudication (Claude Code branches on it to run the tool).
// A model that asked for tools the kernel then denied has no surviving tool_use
// block, so it must collapse to a turn-ending reason — not "tool_use", which would
// send the client hunting for a block that isn't there. A length cap maps to
// "max_tokens"; everything else is "end_turn".
func AnthropicStopReason(finishReason string, hasToolUse bool) string {
	if hasToolUse {
		return "tool_use"
	}
	switch strings.ToLower(finishReason) {
	case "length", "max_tokens":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

// EstimateAnthropicTokens is a cheap, tokenizer-free input-token estimate (~4 chars
// per token) over the decoded system+messages+tool surface — enough for the
// optional count_tokens endpoint, never billed against a real model.
func EstimateAnthropicTokens(req *AnthropicMessagesRequest) int {
	chars := len(req.System)
	for _, m := range req.Messages {
		chars += len(m.Content)
		for _, tc := range m.ToolCalls {
			chars += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	for _, t := range req.Tools {
		chars += len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)
	}
	return chars / 4
}

// --- small wire helpers ---------------------------------------------------------

func canonRole(role string) string {
	switch role {
	case "assistant":
		return RoleAssistant
	default:
		return RoleUser
	}
}

func appendText(b *strings.Builder, s string) {
	if s == "" {
		return
	}
	if b.Len() > 0 {
		b.WriteString("\n")
	}
	b.WriteString(s)
}

// asJSONString reports whether raw is a JSON string literal, returning its value.
func asJSONString(raw json.RawMessage) (string, bool) {
	t := skipSpace(raw)
	if len(t) == 0 || t[0] != '"' {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

// parseAnthropicText folds a `system`/`content` field that may be a bare string OR
// an array of {type:"text",text:...} (or {type:"tool_result"...}) blocks into a
// single string. Non-text blocks contribute their text payload only.
func parseAnthropicText(raw json.RawMessage) string {
	if len(skipSpace(raw)) == 0 {
		return ""
	}
	if s, ok := asJSONString(raw); ok {
		return s
	}
	var blocks []anthropicInboundBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Text != "" {
			appendText(&b, blk.Text)
			continue
		}
		// A nested tool_result.content array of text blocks.
		if len(blk.Content) > 0 {
			appendText(&b, parseAnthropicText(blk.Content))
		}
	}
	return b.String()
}

func skipSpace(b []byte) []byte {
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return b[i:]
}

// inputToArgs keeps a tool_use `input` object as the RAW argument string the kernel
// adjudicates (verbatim, like the OpenAI path) so an alias/malformed object reaches
// the grammar rung unchanged. Empty input normalizes to "{}".
func inputToArgs(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" {
		return "{}"
	}
	return s
}

// inputObject renders a raw argument string as a JSON OBJECT for the response
// `input` field. A non-object (empty / malformed) becomes "{}" so the block is
// always well-formed for Claude Code's parser.
func inputObject(args string) string {
	s := strings.TrimSpace(args)
	if s == "" {
		return "{}"
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return "{}"
	}
	if _, ok := v.(map[string]any); !ok {
		return "{}"
	}
	return s
}
