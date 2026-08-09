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
	"bytes"
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
	// ContentBlocks preserves each message content value byte-for-byte for ledger replay.
	ContentBlocks []json.RawMessage
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
	// thinking / redacted_thinking (assistant, extended thinking): ThinkingText is the
	// reasoning prose (JSON key "thinking", distinct from a text block's "text"); Signature
	// signs it for the Anthropic round-trip; Data is the opaque redacted payload.
	ThinkingText string `json:"thinking"`
	Signature    string `json:"signature"`
	Data         string `json:"data"`
	// image (user / tool_result): Source is the {type,media_type,data|url} object. It is
	// kept RAW because the canonical Message has no image field — the decoder only needs to
	// know a block IS an image (so it emits a non-empty placeholder rather than collapsing).
	Source json.RawMessage `json:"source"`
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
		out.ContentBlocks = append(out.ContentBlocks, bytes.Clone(m.Content))
		out.Messages = append(out.Messages, decodeAnthropicMessage(m)...)
	}
	bindAnthropicToolResultNames(out.Messages)
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

// bindAnthropicToolResultNames restores the tool name that Anthropic tool_result
// blocks do not carry. The matching assistant tool_use appears earlier in the same
// stateless transcript; preserving the name keeps the downstream OpenAI/in-kernel
// prompt in the canonical named <tool_response> shape instead of a bare user blob.
func bindAnthropicToolResultNames(messages []Message) {
	toolByID := make(map[string]string)
	for i := range messages {
		m := &messages[i]
		switch m.Role {
		case RoleAssistant:
			for _, tc := range m.ToolCalls {
				id := strings.TrimSpace(tc.ID)
				name := strings.TrimSpace(tc.Function.Name)
				if id != "" && name != "" {
					toolByID[id] = name
				}
			}
		case RoleTool:
			if strings.TrimSpace(m.Name) != "" {
				continue
			}
			if name := toolByID[strings.TrimSpace(m.ToolCallID)]; name != "" {
				m.Name = name
			}
		}
	}
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
		var thinking string
		var signature string
		var redacted []string
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
			case "thinking":
				// Extended-thinking replay (#3120): preserve the reasoning text and its
				// signature so a later re-emit keeps the block wire-valid. Dropping it and
				// reordering the turn is a documented Anthropic 400 on a thinking-enabled wire.
				if b.ThinkingText != "" {
					thinking = joinNonEmpty(thinking, b.ThinkingText, "\n")
				}
				if b.Signature != "" {
					signature = b.Signature
				}
			case "redacted_thinking":
				if b.Data != "" {
					redacted = append(redacted, b.Data)
				}
			default:
				// Parent-class guard (#3118): any other block type (image on an assistant
				// turn, or a future client-internal type) becomes a non-empty text marker
				// instead of silently vanishing and collapsing the turn to empty content.
				appendText(&text, unknownBlockPlaceholder(b))
			}
		}
		return assistantMessagesWithThinking(text.String(), calls, thinking, signature, redacted)
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
			default:
				// #3118/#3119: an image (or any unrecognized) block on a user turn becomes a
				// non-empty text marker so an image-only user turn never decodes to empty.
				appendText(&text, unknownBlockPlaceholder(b))
			}
		}
		return appendUserText(msgs, &text)
	}
}

// unknownBlockPlaceholder renders the non-empty in-band text that stands in for a content
// block the decoder does not fold into a first-class field (#3118). It names the block type
// (and, for an image, marks it) so a message made ENTIRELY of such blocks never collapses to
// empty content — the shape a strict non-passthrough downstream 400s. Never returns "".
func unknownBlockPlaceholder(b anthropicInboundBlock) string {
	switch b.Type {
	case "image":
		return "[image]"
	case "":
		return "[content block]"
	default:
		return "[" + b.Type + " block]"
	}
}

// joinNonEmpty joins a and b with sep, skipping the separator when either side is empty.
func joinNonEmpty(a, b, sep string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	default:
		return a + sep + b
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
//
// Images are the one block the ~4-chars/token rule cannot see: the decoder folds an image
// block down to the literal "[image]" placeholder (unknownBlockPlaceholder), so the decoded
// m.Content carries ~7 chars for a picture the provider bills at ~imageTokenCost tokens. Left
// unadjusted, an image-heavy request reports near-zero input tokens, and any client trusting
// this endpoint to decide when to compact/summarize is told the window is empty right up to a
// real overflow. So each image block preserved in req.ContentBlocks (the verbatim per-message
// content the decoder keeps for ledger replay) is charged its real cost on top of the text
// estimate — the SAME per-image currency the byte-level compaction path uses
// (estimateElementTokens), so the two estimators finally agree on what a picture costs. That cost
// is geometry-derived where the dimensions are cheaply recoverable and the flat imageTokenCost
// ceiling otherwise (#5165).
func EstimateAnthropicTokens(req *AnthropicMessagesRequest) int {
	proseBytes := len(req.System)
	for _, m := range req.Messages {
		proseBytes += messageBytes(m)
	}
	schemaBytes := 0
	for _, t := range req.Tools {
		schemaBytes += toolBytes(t)
	}
	// Count the preserved image blocks once. The decoder folded each image down to the
	// "[image]" placeholder already summed into proseBytes above, so subtract that
	// placeholder before charging the preserved image's real cost (#5166).
	numImages, imageTokens := 0, 0
	for _, cb := range req.ContentBlocks {
		n, tk := countContentBlockImages(cb)
		numImages += n
		imageTokens += tk
	}
	if drop := numImages * len("[image]"); drop < proseBytes {
		proseBytes -= drop
	} else {
		proseBytes = 0
	}
	return estimateTokens(proseBytes, proseTokenDivisor) +
		estimateTokens(schemaBytes, jsonSchemaTokenDivisor) + imageTokens
}

// countContentBlockImages counts image blocks in one preserved message content value and sums their
// ~token cost (a bare string has none; a block array is walked, recursing one level into a
// tool_result's own content so a screenshot returned by a tool is counted). It shares
// contentImageWeight's traversal but needs the count and the cost, not the byte weight — the count
// only to net out the "[image]" placeholder chars the decoded text walk already summed.
func countContentBlockImages(content json.RawMessage) (imgs, imgTokens int) {
	i, _, tk := contentImageWeight(content)
	return i, tk
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

// assistantMessages wraps a decoded assistant turn (its accumulated text + tool calls)
// into the canonical 0-or-1-element Message slice the inbound decoders return: nil when
// the turn is empty (no content AND no tool calls), else a single assistant Message.
// Shared by the Anthropic and Gemini content decoders, which assemble identical turns.
func assistantMessages(text string, calls []ToolCall) []Message {
	return assistantMessagesWithThinking(text, calls, "", "", nil)
}

// assistantMessagesWithThinking is assistantMessages plus preserved extended-thinking
// (#3120): the reasoning text, its signature, and any redacted_thinking payloads ride the
// canonical Message so the outbound Anthropic re-encode (textAndToolUseBlocks) can replay
// them in order. A turn that carries ONLY thinking (no visible text, no tool call) is still
// emitted — dropping it would desync the assistant/user alternation and lose the signature
// a later turn needs.
func assistantMessagesWithThinking(text string, calls []ToolCall, thinking, signature string, redacted []string) []Message {
	msg := Message{
		Role:              RoleAssistant,
		Content:           text,
		ToolCalls:         calls,
		Thinking:          thinking,
		ThinkingSignature: signature,
		RedactedThinking:  redacted,
	}
	if msg.Content == "" && len(msg.ToolCalls) == 0 && msg.Thinking == "" && len(msg.RedactedThinking) == 0 {
		return nil
	}
	return []Message{msg}
}

// appendUserText flushes any accumulated plain text as one trailing user-role Message,
// appending it to msgs only when the builder is non-empty and returning the (possibly
// extended) slice. It is the shared tail of the Anthropic and Gemini inbound user-turn
// decoders: each fans tool results into msgs, then emits leftover text as a final user turn.
func appendUserText(msgs []Message, text *strings.Builder) []Message {
	if text.Len() > 0 {
		msgs = append(msgs, Message{Role: RoleUser, Content: text.String()})
	}
	return msgs
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
// single string. A text-less block (image, or any type the decoder does not fold into a
// first-class field) contributes a non-empty placeholder naming it, so a tool_result whose
// content is ENTIRELY such blocks never collapses to empty content (#3118/#3119) — the
// shape a strict non-passthrough downstream 400s.
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
			continue
		}
		// Text-less, contentless block (image / unknown): keep a non-empty marker so an
		// all-image or all-unknown content array never folds to the empty string.
		appendText(&b, unknownBlockPlaceholder(blk))
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
