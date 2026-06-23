package agent

// gemini_server.go is the SERVER half of the Gemini generateContent wire. The
// adapter in adapters.go encodes a canonical transcript INTO a Gemini request and
// parses a Gemini response back (the CLIENT direction, for `fak agent` talking to a
// Gemini upstream). This file is the inverse: it DECODES an inbound generateContent
// request (from a Gemini-CLI / google-genai-shaped client) into the canonical
// Message/ToolDef vocabulary, and renders a Completion's assistant turn back out as
// the Gemini candidate/parts shape.
//
// It exists so `fak serve` can expose a native POST /v1beta/models/{model}:
// generateContent front door: point a Gemini-native client's base URL at the
// gateway and every function call the served model proposes is adjudicated by the
// kernel before the client sees it. The gateway owns the HTTP + SSE framing; this
// file owns the wire SHAPE only (no net/http), keeping all Gemini-format knowledge
// in one package alongside the outbound adapter.

import (
	"encoding/json"
	"strings"
)

// GeminiGenerateContentRequest is an inbound generateContent body decoded into the
// canonical transcript vocabulary. SystemInstruction is folded to a single string
// (a leading RoleSystem message is already prepended to Messages). Stream mirrors
// whether the client hit :streamGenerateContent (the gateway may also force it on
// from the route method).
type GeminiGenerateContentRequest struct {
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
}

// DecodeGeminiGenerateContentRequest parses an inbound Gemini generateContent body
// into the canonical transcript the gateway planner consumes. It is the structural
// inverse of geminiAdapter.MarshalRequest: a systemInstruction becomes a leading
// RoleSystem message; a model content's functionCall parts become ToolCalls (id
// preserved); a user content's functionResponse parts become RoleTool messages
// keyed by the call id so the kernel's per-trace result-side ledger correlates
// them. The model id is supplied by the gateway from the request path
// (/v1beta/models/{model}:generateContent) since Gemini carries it there, not in
// the body.
func DecodeGeminiGenerateContentRequest(raw []byte, model string) (*GeminiGenerateContentRequest, error) {
	var in geminiRequest
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, err
	}
	out := &GeminiGenerateContentRequest{Model: model}
	if in.GenerationConfig != nil {
		out.MaxTokens = in.GenerationConfig.MaxTokens
		out.TopP = in.GenerationConfig.TopP
		out.TopK = in.GenerationConfig.TopK
		out.StopSequences = in.GenerationConfig.StopSequences
		// Temperature is a bare float64 on this wire (no pointer), so 0 is
		// indistinguishable from unset — the same limitation the outbound adapter
		// has. The gateway forwards it only when non-zero.
		out.Temperature = in.GenerationConfig.Temperature
	}
	if in.SystemInstruction != nil {
		out.System = geminiPartsText(in.SystemInstruction.Parts)
	}
	if out.System != "" {
		out.Messages = append(out.Messages, Message{Role: RoleSystem, Content: out.System})
	}
	for _, c := range in.Contents {
		out.Messages = append(out.Messages, decodeGeminiContent(c)...)
	}
	for _, t := range in.Tools {
		for _, d := range t.FunctionDeclarations {
			out.Tools = append(out.Tools, ToolDef{
				Type: "function",
				Function: ToolDefFunction{
					Name:        d.Name,
					Description: d.Description,
					Parameters:  geminiParamsToSchema(d.Parameters),
				},
			})
		}
	}
	return out, nil
}

// decodeGeminiContent converts one inbound content into zero or more canonical
// messages. A "model" turn carrying functionCall parts becomes one RoleAssistant
// message (text + ToolCalls); a "user" turn carrying functionResponse parts fans
// them out into RoleTool messages (one per response, in order), with any free text
// emitted as a trailing RoleUser message — the shape every upstream adapter
// expects.
func decodeGeminiContent(c geminiContent) []Message {
	if strings.EqualFold(c.Role, "model") {
		var text strings.Builder
		var calls []ToolCall
		for _, p := range c.Parts {
			if p.Text != "" {
				appendText(&text, p.Text)
			}
			if p.FunctionCall != nil {
				calls = append(calls, ToolCall{
					ID:   p.FunctionCall.ID,
					Type: "function",
					Function: Func{
						Name:      p.FunctionCall.Name,
						Arguments: geminiArgsToString(p.FunctionCall.Args),
					},
				})
			}
		}
		msg := Message{Role: RoleAssistant, Content: text.String(), ToolCalls: calls}
		if msg.Content == "" && len(msg.ToolCalls) == 0 {
			return nil
		}
		return []Message{msg}
	}
	// user (and any other role): functionResponse fan-out + trailing text.
	var msgs []Message
	var text strings.Builder
	for _, p := range c.Parts {
		if p.FunctionResponse != nil {
			msgs = append(msgs, Message{
				Role:       RoleTool,
				ToolCallID: p.FunctionResponse.ID,
				Name:       p.FunctionResponse.Name,
				Content:    geminiResponseToString(p.FunctionResponse.Response),
			})
		}
		if p.Text != "" {
			appendText(&text, p.Text)
		}
	}
	if text.Len() > 0 {
		msgs = append(msgs, Message{Role: RoleUser, Content: text.String()})
	}
	return msgs
}

// --- outbound: a Completion's assistant turn -> Gemini candidate parts ----------

// GeminiPartOut is one rendered response part (text or functionCall). The gateway
// serializes these into a candidate's content.parts, either as a buffered
// generateContent response or as the synthesized streamGenerateContent SSE frames.
type GeminiPartOut struct {
	Text         string              `json:"text,omitempty"`
	FunctionCall *GeminiFunctionCall `json:"functionCall,omitempty"`
}

// GeminiFunctionCall is the model-side function call a Gemini client round-trips
// results against. Args is normalized to a JSON OBJECT so a client's parser always
// sees a well-formed argument object; the call id is preserved for the
// functionResponse the client sends back next turn.
type GeminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
	ID   string          `json:"id,omitempty"`
}

// GeminiResponseParts renders the (post-adjudication) assistant message as ordered
// Gemini parts: a leading text part when there is prose, then one functionCall
// part per surviving tool call (id preserved for the result round-trip). It is the
// inverse of the inbound functionCall decode.
func GeminiResponseParts(m Message) []GeminiPartOut {
	parts := make([]GeminiPartOut, 0, 1+len(m.ToolCalls))
	if strings.TrimSpace(m.Content) != "" {
		parts = append(parts, GeminiPartOut{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, GeminiPartOut{
			FunctionCall: &GeminiFunctionCall{
				Name: tc.Function.Name,
				Args: geminiArgsObject(tc.Function.Arguments),
				ID:   tc.ID,
			},
		})
	}
	return parts
}

// GeminiFinishReason maps the canonical finish reason onto the Gemini vocabulary.
// Gemini returns "STOP" for a normal turn AND for a turn that produced function
// calls (the functionCall parts themselves signal the tool use — unlike Anthropic,
// Gemini has no distinct tool-use finish reason), so the caller need not branch on
// whether any call survived. A length cap maps to "MAX_TOKENS"; everything else is
// "STOP".
func GeminiFinishReason(finishReason string) string {
	switch strings.ToLower(finishReason) {
	case "length", "max_tokens":
		return "MAX_TOKENS"
	default:
		return "STOP"
	}
}

// --- small wire helpers ---------------------------------------------------------

// geminiPartsText folds a systemInstruction/content parts slice into a single
// string from its text parts.
func geminiPartsText(parts []geminiPart) string {
	var b strings.Builder
	for _, p := range parts {
		if p.Text != "" {
			appendText(&b, p.Text)
		}
	}
	return b.String()
}

// geminiArgsToString renders a decoded functionCall `args` value as the RAW
// argument string the kernel adjudicates (verbatim, like the OpenAI/Anthropic
// paths) so an alias/malformed object reaches the grammar rung unchanged. nil /
// null / empty normalizes to "{}".
func geminiArgsToString(args any) string {
	if args == nil {
		return "{}"
	}
	raw, err := json.Marshal(args)
	if err != nil {
		return "{}"
	}
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return "{}"
	}
	return s
}

// geminiResponseToString renders a decoded functionResponse `response` value as
// the content string the result-side floor sees (and forwards to the model). It
// mirrors responseObject's round-trip: an object passes through verbatim, anything
// else is wrapped so the planner still sees valid JSON.
func geminiResponseToString(resp any) string {
	if resp == nil {
		return "{}"
	}
	raw, err := json.Marshal(resp)
	if err != nil {
		return "{}"
	}
	return string(raw)
}

// geminiArgsObject renders a raw argument string as a JSON OBJECT for the
// functionCall args field. A non-object (empty / malformed) becomes "{}" so the
// part is always well-formed for a Gemini client's parser.
func geminiArgsObject(args string) json.RawMessage {
	s := strings.TrimSpace(args)
	if s == "" {
		return json.RawMessage("{}")
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return json.RawMessage("{}")
	}
	if _, ok := v.(map[string]any); !ok {
		return json.RawMessage("{}")
	}
	return json.RawMessage(s)
}

// geminiParamsToSchema converts a decoded functionDeclaration `parameters` value
// into the JSON Schema bytes the canonical ToolDef carries. A Gemini client sends
// UPPERCASE type names (STRING, OBJECT, ...) per the Gemini schema convention;
// every other inbound path produces lowercase OpenAI-style JSON Schema, so this
// lowercases the type fields to normalize to that canonical form regardless of
// which upstream ultimately serves the turn. A nil/empty parameters yields nil
// (omitted on the planner wire).
func geminiParamsToSchema(params any) json.RawMessage {
	if params == nil {
		return nil
	}
	raw, err := json.Marshal(params)
	if err != nil {
		return nil
	}
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return raw
	}
	lowercaseSchemaTypes(v)
	out, err := json.Marshal(v)
	if err != nil {
		return raw
	}
	return out
}

// lowercaseSchemaTypes is the inverse of the outbound adapter's
// uppercaseSchemaTypes: it rewrites every JSON Schema "type" field value to
// lowercase in place, so a Gemini-inbound schema (UPPERCASE) normalizes to the
// lowercase OpenAI-style JSON Schema every other inbound path produces.
func lowercaseSchemaTypes(v any) {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			if k == "type" {
				if s, ok := val.(string); ok {
					x[k] = strings.ToLower(s)
					continue
				}
			}
			lowercaseSchemaTypes(val)
		}
	case []any:
		for _, val := range x {
			lowercaseSchemaTypes(val)
		}
	}
}
