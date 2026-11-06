// Package anthropicbridge translates between the Anthropic Messages wire and the
// OpenAI Chat Completions wire — the translation core of the /v1/messages routing
// mode (an Anthropic-speaking client, e.g. Claude Code with ANTHROPIC_BASE_URL on
// the router, served by an OpenAI-compatible routed backend).
//
// It is the public promotion of the translator fak's own /v1/messages front door
// already runs (internal/gateway + internal/agent; vllm_messages_dropin_test.go
// witnesses it end to end). Every function here delegates to that same core —
// nothing is re-implemented — so a routed turn and an in-fak turn see the same
// wire shapes:
//
//	request   DecodeMessagesRequest / TranslateMessagesRequest  (Anthropic -> OpenAI)
//	response  TranslateChatResponse / DecodeChatResponse       (OpenAI -> Anthropic)
//	stream    StreamTranslator (OpenAI SSE chunks -> Anthropic SSE events)
//
// Mappings (the in-fak shapes, with the two known routed-vs-in-fak deltas
// named rather than hidden):
//
//   - model passes through (a router rewrites spec.Model when it re-marshals);
//   - the Anthropic `system` (string or block array) folds into a leading
//     system message, and late system messages fold back together on marshal;
//   - tools become OpenAI function tools (input_schema passed through verbatim);
//     tool_choice {"type":"auto"} -> "auto", {"type":"any"} -> "required",
//     {"type":"tool",...} -> {"type":"function",...}; absent keeps the encoder
//     default ("auto" when tools are offered);
//   - assistant tool_use blocks decode to tool calls and tool_result blocks
//     decode to role=tool messages keyed by tool_use_id (name re-bound from the
//     matching assistant call), and the response direction renders tool_use
//     blocks with the id preserved for the result round-trip;
//   - max_tokens, temperature (omitted when 0 — the in-fak /v1/messages
//     planner path sends an explicit 0 when the client omits it; omitting
//     matches the WithClientTemperature doctrine and lets the upstream
//     default apply), top_p, and stop_sequences pass through; stream:true
//     adds stream_options.include_usage so a streamed turn still reports
//     usage. tool_choice is mapped and forwarded here, where the in-fak
//     planner path drops the client's Anthropic tool_choice entirely — this
//     package is the more conformant of the two. top_k has no OpenAI
//     chat-wire carrier on this path (the in-fak buffered path omits it
//     too) and is not forwarded;
//   - thinking round-trips as the in-fak path does: inbound assistant thinking /
//     redacted_thinking blocks (with signature) decode into the canonical
//     Message fields, ride the OpenAI body through the canonical tags, and a
//     streamed upstream's reasoning_content is accumulated on the returned
//     Completion. The response renderer emits text and tool_use blocks — the
//     same blocks the in-fak /v1/messages wire emits today.
//
// The package owns no HTTP, no credential, and no policy: a router decodes,
// routes, marshals, dispatches, and translates with its own upstream plumbing.
package anthropicbridge

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Canonical vocabulary aliases. The translator speaks the same canonical
// Message/ToolDef/Completion vocabulary as the in-fak planner path; the aliases
// let an external consumer name every type through this package alone.
type (
	// MessagesRequest is a decoded Anthropic /v1/messages body: system folded to
	// a leading system message, tool_use blocks decoded to tool calls, and
	// tool_result blocks decoded to role=tool messages keyed by tool_use_id.
	MessagesRequest = agent.AnthropicMessagesRequest
	// Message is one canonical transcript message (request or response side).
	Message = agent.Message
	// ToolDef is one canonical tool definition (OpenAI function shape).
	ToolDef = agent.ToolDef
	// ToolCall is one tool call the model emitted (raw JSON arguments kept verbatim).
	ToolCall = agent.ToolCall
	// Completion is the canonical first-choice turn a chat response decodes to.
	Completion = agent.Completion
	// Block is one rendered Anthropic response content block (text or tool_use).
	Block = agent.AnthropicBlockOut
	// UpstreamStreamError is an in-band SSE error frame (unwraps to ErrStreamError).
	UpstreamStreamError = agent.UpstreamStreamError
)

// Sentinel errors shared with the in-fak planner path.
var (
	// ErrIncompleteStream reports a stream that ended before a terminal marker
	// ([DONE]) or choice completion. Content already emitted stays partial.
	ErrIncompleteStream = agent.ErrUpstreamIncompleteStream
	// ErrStreamError reports an in-band SSE `data:` frame carrying an `error`
	// object; the turn did not finish and must not be certified clean.
	ErrStreamError = agent.ErrUpstreamStreamError
)

// MessagesUsage is the Anthropic Messages usage object. The cache counters are
// load-bearing for cache-aware clients: they are reported independently of
// input_tokens (Anthropic's input_tokens is already the uncached remainder) and
// omitted when zero, exactly as the in-fak /v1/messages wire renders them.
type MessagesUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// MessagesUsageFromCanonical maps the canonical (OpenAI-shaped) usage a chat
// response decodes to onto the Messages usage object — prompt->input,
// completion->output, cache counters carried under their Anthropic names.
func MessagesUsageFromCanonical(u agent.Usage) MessagesUsage {
	return MessagesUsage{
		InputTokens:              u.PromptTokens,
		OutputTokens:             u.CompletionTokens,
		CacheReadInputTokens:     u.CacheReadInputTokens,
		CacheCreationInputTokens: u.CacheCreationInputTokens,
	}
}

// MessageResponse is the buffered (non-streaming) Anthropic Messages body the
// translator renders — the shape the in-fak /v1/messages wire emits, without the
// kernel `fak` extension (a routed turn carries no adjudication record).
type MessageResponse struct {
	ID           string        `json:"id"`
	Type         string        `json:"type"` // always "message"
	Role         string        `json:"role"` // always "assistant"
	Model        string        `json:"model"`
	Content      []Block       `json:"content"`
	StopReason   string        `json:"stop_reason"`
	StopSequence *string       `json:"stop_sequence"`
	Usage        MessagesUsage `json:"usage"`
}

// ResponseOptions carries the echo fields a Messages response envelope needs.
// The gateway echoes the requested model and a msg_* id; empty values are
// rendered as empty strings, never omitted.
type ResponseOptions struct {
	ID    string
	Model string
}

// DecodeMessagesRequest parses an inbound Anthropic /v1/messages body into the
// canonical transcript. The structural inverse of the Anthropic request encoder:
// `system` becomes a leading system message, assistant `tool_use` blocks become
// tool calls (id preserved), and user `tool_result` blocks become role=tool
// messages keyed by tool_use_id. Raw keeps the request bytes verbatim.
func DecodeMessagesRequest(raw []byte) (*MessagesRequest, error) {
	return agent.DecodeAnthropicMessagesRequest(raw)
}

// EstimateInputTokens is the tokenizer-free input estimate behind the
// /v1/messages/count_tokens endpoint (~4 chars/token, images charged their real
// block cost) — enough for a client's context heuristics, never a billing figure.
func EstimateInputTokens(req *MessagesRequest) int {
	return agent.EstimateAnthropicTokens(req)
}

// ChatRequestSpec is the planner-free input to MarshalChatRequest. It aliases the
// canonical OpenAI chat request the planner's own encoder consumes, so a
// translated body is byte-identical to the one the planner would send.
type ChatRequestSpec = agent.OpenAIChatRequestSpec

// MarshalChatRequest encodes the canonical request as an OpenAI Chat
// Completions body: model passthrough, system/messages/tools, tool_choice,
// max_tokens, temperature/top_p/stop (unset optionals omitted), and — when
// Stream is set — stream_options.include_usage.
func MarshalChatRequest(spec ChatRequestSpec) ([]byte, error) {
	return agent.MarshalOpenAIChatRequest(spec)
}

// TranslateMessagesRequest is the one-shot request translation: decode an
// Anthropic /v1/messages body and encode the equivalent OpenAI Chat Completions
// body with the model passed through. Routers that pick a backend model first
// decode (DecodeMessagesRequest), route on req.Model, and marshal with the
// backend model in the spec instead.
func TranslateMessagesRequest(raw []byte) (chatBody []byte, req *MessagesRequest, err error) {
	req, err = DecodeMessagesRequest(raw)
	if err != nil {
		return nil, nil, err
	}
	chatBody, err = MarshalChatRequest(requestSpec(req))
	if err != nil {
		return nil, req, err
	}
	return chatBody, req, nil
}

// requestSpec projects a decoded Messages request onto the chat request spec —
// the same fields the in-fak planner path forwards on this wire: max_tokens,
// temperature (omitted when 0), top_p, stop_sequences, and the mapped
// tool_choice. top_k is not forwarded (no OpenAI chat-wire carrier on this path).
func requestSpec(req *MessagesRequest) ChatRequestSpec {
	var temp *float64
	if req.Temperature != 0 {
		temp = &req.Temperature
	}
	return ChatRequestSpec{
		Model:       req.Model,
		Messages:    req.Messages,
		Tools:       req.Tools,
		ToolChoice:  ToolChoiceForChat(req.ToolChoice),
		MaxTokens:   req.MaxTokens,
		Temperature: temp,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
		Stream:      req.Stream,
	}
}

// ToolChoiceForChat maps an Anthropic tool_choice (kept raw by the decode) onto
// the OpenAI Chat Completions vocabulary: {"type":"auto"} -> "auto",
// {"type":"any"} -> "required", {"type":"tool","name":N} ->
// {"type":"function","function":{"name":N}}, with the bare-string forms mapped
// likewise. Absent/empty returns nil so the encoder default applies ("auto"
// when tools are offered) — the shape the in-fak planner path sends.
func ToolChoiceForChat(raw json.RawMessage) json.RawMessage {
	trimmed := trimJSON(raw)
	if len(trimmed) == 0 {
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if json.Unmarshal(trimmed, &s) == nil {
			return toolChoiceString(s)
		}
		return nil
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(trimmed, &obj) != nil {
		return nil
	}
	switch obj.Type {
	case "auto":
		return json.RawMessage(`"auto"`)
	case "any":
		return json.RawMessage(`"required"`)
	case "tool":
		if obj.Name == "" {
			return nil
		}
		// Ordered struct, not a map: the wire value keeps the conventional
		// {"type":...,"function":...} field order.
		encoded, err := json.Marshal(struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}{Type: "function", Function: struct {
			Name string `json:"name"`
		}{Name: obj.Name}})
		if err != nil {
			return nil
		}
		return encoded
	default:
		return nil
	}
}

// toolChoiceString maps the tolerant bare-string tool_choice forms some clients
// send where the object form is documented.
func toolChoiceString(s string) json.RawMessage {
	switch s {
	case "auto":
		return json.RawMessage(`"auto"`)
	case "any":
		return json.RawMessage(`"required"`)
	case "none":
		return json.RawMessage(`"none"`)
	default:
		return nil
	}
}

// DecodeChatResponse parses an OpenAI Chat Completions body into the canonical
// completion: first choice's message (legacy function_call upgraded to a tool
// call), finish reason, usage, and the upstream-reported model.
func DecodeChatResponse(chatBody []byte) (*Completion, error) {
	ad, err := agent.NewTranscriptAdapter(agent.ProviderOpenAI)
	if err != nil {
		return nil, err
	}
	return ad.ParseResponse(chatBody)
}

// ResponseBlocks renders a canonical assistant message as ordered Anthropic
// content blocks: a leading text block when there is prose, then one tool_use
// block per tool call (id preserved for the result round-trip).
func ResponseBlocks(m Message) []Block {
	return agent.AnthropicResponseBlocks(m)
}

// StopReason maps the canonical finish reason onto the Messages stop_reason
// vocabulary. hasToolUse is authoritative: "tool_use" is returned only when a
// tool call actually survived (a client branches on it to run the tool); a
// length cap maps to "max_tokens"; everything else is "end_turn".
func StopReason(finishReason string, hasToolUse bool) string {
	return agent.AnthropicStopReason(finishReason, hasToolUse)
}

// TranslateChatResponse is the one-shot response translation: decode an OpenAI
// Chat Completions body and encode the equivalent Anthropic Messages body —
// content blocks (text + tool_use), stop_reason, usage (input/output + cache
// counters), and the echo fields from opt.
func TranslateChatResponse(chatBody []byte, opt ResponseOptions) (messagesBody []byte, err error) {
	comp, err := DecodeChatResponse(chatBody)
	if err != nil {
		return nil, err
	}
	resp := RenderChatResponse(comp, opt)
	encoded, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

// RenderChatResponse builds the buffered Messages response object for a decoded
// chat completion — the structured half of TranslateChatResponse, for routers
// that post-process the envelope (metrics, header echoes) before serializing.
func RenderChatResponse(comp *Completion, opt ResponseOptions) *MessageResponse {
	blocks := ResponseBlocks(comp.Message)
	stop := StopReason(comp.FinishReason, len(comp.Message.ToolCalls) > 0)
	return &MessageResponse{
		ID:         opt.ID,
		Type:       "message",
		Role:       "assistant",
		Model:      opt.Model,
		Content:    blocks,
		StopReason: stop,
		Usage:      MessagesUsageFromCanonical(comp.Usage),
	}
}

// trimJSON returns raw with surrounding whitespace removed; nil-safe.
func trimJSON(raw json.RawMessage) json.RawMessage {
	for len(raw) > 0 && (raw[0] == ' ' || raw[0] == '\t' || raw[0] == '\n' || raw[0] == '\r') {
		raw = raw[1:]
	}
	for len(raw) > 0 && (raw[len(raw)-1] == ' ' || raw[len(raw)-1] == '\t' || raw[len(raw)-1] == '\n' || raw[len(raw)-1] == '\r') {
		raw = raw[:len(raw)-1]
	}
	if len(raw) == 1 && raw[0] == 'n' { // bare "null" literal tail guard
		return nil
	}
	if string(raw) == "null" {
		return nil
	}
	return raw
}
