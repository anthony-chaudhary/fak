package agent

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Anthropic Claude Messages API.
// ---------------------------------------------------------------------------

// anthropicAdapter speaks the Claude Messages API wire. Its one field declares HOW a
// credential is presented; the zero value keeps the historical shape-sniffing, so
// `anthropicAdapter{}` is byte-for-byte the pre-field adapter.
type anthropicAdapter struct{ auth AnthropicAuthScheme }

// Provider reports ProviderAnthropic (the Claude Messages API wire).
func (anthropicAdapter) Provider() Provider { return ProviderAnthropic }

func (anthropicAdapter) Endpoint(baseURL, model string) string {
	return joinEndpoint(baseURL, "/v1/messages")
}

// AnthropicOAuthBeta is the anthropic-beta flag that gates the OAuth (Claude
// Pro/Max SUBSCRIPTION) code path on api.anthropic.com. The official Claude Code
// client sends it alongside an "Authorization: Bearer <oauth-token>"; the gateway
// mirrors that so a subscription token is accepted upstream.
const AnthropicOAuthBeta = "oauth-2025-04-20"

// IsAnthropicOAuthToken reports whether tok is an Anthropic OAuth access token (a
// Claude Code SUBSCRIPTION credential), which carry the "sk-ant-oat" prefix.
// Anthropic rejects these as an x-api-key ("invalid x-api-key") and accepts them
// ONLY as a bearer token; a plain API key ("sk-ant-api…") is the inverse. The
// prefix is the provider's own stable discriminator, so the gateway can pick the
// right auth scheme with no extra configuration — which is what lets a forwarded
// or server-held subscription token work through the same passthrough path as a
// raw API key.
func IsAnthropicOAuthToken(tok string) bool {
	return strings.HasPrefix(tok, "sk-ant-oat")
}

// AnthropicAuthScheme declares HOW an Anthropic-wire credential is presented upstream.
//
// WHY IT EXISTS. The credential's SHAPE is a reliable discriminator only for
// FIRST-PARTY Anthropic: api.anthropic.com wants a plain key as x-api-key and a
// subscription token (sk-ant-oat…) as a Bearer, and IsAnthropicOAuthToken tells them
// apart with no configuration. A THIRD-PARTY Anthropic-COMPATIBLE endpoint — a cloud
// vendor's serving endpoint, a corporate proxy, an aggregating gateway — authenticates
// its OWN tenant credential, whose prefix fak cannot know and must not guess. Those
// endpoints generally accept the token ONLY as `Authorization: Bearer`, so sniffing the
// prefix sends x-api-key and the call 401s ("credential ... of an unsupported type")
// even though the base URL, model, and body were all correct. That failure is
// indistinguishable from a bad token, which is what made it worth making declarable.
//
// The zero value keeps the sniff, so every existing caller is unchanged.
type AnthropicAuthScheme string

const (
	// AnthropicAuthAuto sniffs the credential shape — sk-ant-oat => Bearer + the oauth
	// beta, anything else => x-api-key. The default, and the right answer for
	// first-party Anthropic.
	AnthropicAuthAuto AnthropicAuthScheme = ""
	// AnthropicAuthAPIKey always presents the credential as x-api-key.
	AnthropicAuthAPIKey AnthropicAuthScheme = "x-api-key"
	// AnthropicAuthBearer always presents it as `Authorization: Bearer` and NEVER as
	// x-api-key: a third-party gateway's tenant token must not also be copied into a
	// header that endpoint does not expect and may log. The oauth beta still rides
	// along for a genuine sk-ant-oat token, since that flag is a property of the
	// SUBSCRIPTION credential rather than of the scheme.
	AnthropicAuthBearer AnthropicAuthScheme = "bearer"
)

// ParseAnthropicAuthScheme maps an operator-supplied string to a scheme. It accepts
// "auto"/"" for the sniff and tolerates the common spellings of the two explicit
// schemes ("bearer"/"authorization", "x-api-key"/"apikey"/"api-key"), so a config
// value does not fail on a hyphen. An unrecognized value is a MISS, never a silent
// fallback to the sniff — a typo'd scheme must fail loud at its call site rather than
// re-introduce the 401 it was set to avoid.
func ParseAnthropicAuthScheme(s string) (AnthropicAuthScheme, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "auto":
		return AnthropicAuthAuto, true
	case "bearer", "authorization":
		return AnthropicAuthBearer, true
	case "x-api-key", "xapikey", "apikey", "api-key":
		return AnthropicAuthAPIKey, true
	default:
		return "", false
	}
}

// NewAnthropicTranscriptAdapter returns the Claude Messages API adapter with an
// explicit auth scheme. NewTranscriptAdapter(ProviderAnthropic) is the AnthropicAuthAuto
// case of this constructor; callers reaching a third-party Anthropic-compatible endpoint
// pass AnthropicAuthBearer.
func NewAnthropicTranscriptAdapter(scheme AnthropicAuthScheme) TranscriptAdapter {
	return anthropicAdapter{auth: scheme}
}

// Headers sets Content-Type and anthropic-version, then presents the credential under
// the adapter's declared AnthropicAuthScheme. Under the default (auto) it picks by
// credential shape: an OAuth (sk-ant-oat) subscription token rides as a Bearer with the
// oauth beta flag, a plain API key as x-api-key, and an empty key sends neither.
func (a anthropicAdapter) Headers(apiKey string) map[string]string {
	h := map[string]string{
		"Content-Type":      "application/json",
		"anthropic-version": "2023-06-01",
	}
	if apiKey == "" {
		// No credential (loopback dogfood / mock) — send neither auth scheme.
		return h
	}
	// The oauth beta is a property of the SUBSCRIPTION credential, not of the scheme,
	// so it rides with an sk-ant-oat token however that token was asked to present.
	if IsAnthropicOAuthToken(apiKey) {
		h["anthropic-beta"] = AnthropicOAuthBeta
	}
	switch a.auth {
	case AnthropicAuthBearer:
		h["Authorization"] = "Bearer " + apiKey
	case AnthropicAuthAPIKey:
		h["x-api-key"] = apiKey
	default:
		// AnthropicAuthAuto: sniff the shape. Anthropic rejects a subscription token as
		// an x-api-key ("invalid x-api-key") and accepts it ONLY as a bearer — which is
		// exactly what the official Claude Code client sends, and what makes a Claude
		// Pro/Max subscription usable through the gateway.
		if IsAnthropicOAuthToken(apiKey) {
			h["Authorization"] = "Bearer " + apiKey
		} else {
			h["x-api-key"] = apiKey
		}
	}
	return h
}

type anthropicRequest struct {
	ServiceTier   string             `json:"service_tier,omitempty"`
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   float64            `json:"temperature"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	System        string             `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string           `json:"role"`
	Content []anthropicBlock `json:"content"`
}

type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     any             `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`  // type=thinking reasoning text
	Signature string          `json:"signature,omitempty"` // signs a thinking block for round-trip
	Data      string          `json:"data,omitempty"`      // type=redacted_thinking opaque payload
	RawInput  json.RawMessage `json:"-"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicResponse struct {
	Model   string `json:"model"` // the model the upstream reports it served (#82 echo)
	Content []struct {
		Type      string          `json:"type"`
		Text      string          `json:"text,omitempty"`
		ID        string          `json:"id,omitempty"`
		Name      string          `json:"name,omitempty"`
		Input     json.RawMessage `json:"input,omitempty"`
		Thinking  string          `json:"thinking,omitempty"`
		Signature string          `json:"signature,omitempty"`
		Data      string          `json:"data,omitempty"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		ServiceTier              string `json:"service_tier,omitempty"`
		InputTokens              int    `json:"input_tokens"`
		OutputTokens             int    `json:"output_tokens"`
		CacheReadInputTokens     int    `json:"cache_read_input_tokens,omitempty"`
		CacheCreationInputTokens int    `json:"cache_creation_input_tokens,omitempty"`
	} `json:"usage"`
	Error *apiError `json:"error"`
}

// MarshalRequest encodes the canonical request as an Anthropic Messages body:
// system messages are concatenated into the top-level `system` field, assistant
// turns are lowered to thinking/text/tool_use blocks, tool results become tool_result
// user blocks, and max_tokens defaults to 1024 when unset (the API requires it).
func (anthropicAdapter) MarshalRequest(r adapterRequest) ([]byte, error) {
	maxTokens := r.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}
	req := anthropicRequest{
		ServiceTier:   serviceTierWire(ProviderAnthropic, r.ServiceTier),
		Model:         r.Model,
		MaxTokens:     maxTokens,
		Temperature:   r.Temperature,
		TopP:          r.TopP,
		TopK:          positiveTopK(r.TopK),
		StopSequences: r.Stop,
		Messages:      make([]anthropicMessage, 0, len(r.Messages)),
		Tools:         anthropicTools(r.Tools),
	}
	for _, m := range r.Messages {
		switch m.Role {
		case RoleSystem:
			if req.System != "" && m.Content != "" {
				req.System += "\n\n" + m.Content
			} else {
				req.System = m.Content
			}
		case RoleAssistant:
			blocks := textAndToolUseBlocks(m)
			if len(blocks) > 0 {
				req.Messages = append(req.Messages, anthropicMessage{Role: "assistant", Content: blocks})
			}
		case RoleTool:
			req.Messages = append(req.Messages, anthropicMessage{Role: "user", Content: []anthropicBlock{{
				Type: "tool_result", ToolUseID: m.ToolCallID, Content: m.Content,
			}}})
		default:
			if m.Content != "" {
				req.Messages = append(req.Messages, anthropicMessage{Role: "user", Content: []anthropicBlock{{Type: "text", Text: m.Content}}})
			}
		}
	}
	return json.Marshal(req)
}

func anthropicTools(tools []ToolDef) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropicTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}
	return out
}

func textAndToolUseBlocks(m Message) []anthropicBlock {
	blocks := make([]anthropicBlock, 0, 1+len(m.ToolCalls))
	// Extended-thinking blocks must precede text/tool_use on an assistant turn and
	// carry their signature so the Anthropic API accepts the round-trip.
	if m.Thinking != "" {
		blocks = append(blocks, anthropicBlock{Type: "thinking", Thinking: m.Thinking, Signature: m.ThinkingSignature})
	}
	for _, d := range m.RedactedThinking {
		blocks = append(blocks, anthropicBlock{Type: "redacted_thinking", Data: d})
	}
	if m.Content != "" {
		blocks = append(blocks, anthropicBlock{Type: "text", Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		blocks = append(blocks, anthropicBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: jsonObjectOrRaw(tc.Function.Arguments),
		})
	}
	return blocks
}

// ParseResponse decodes an Anthropic Messages response into a Completion: text blocks
// become content, tool_use blocks become tool calls, and thinking/redacted_thinking
// blocks (with their signature) are preserved so extended reasoning round-trips. It
// maps the input/output and cache_read/cache_creation token counts into Usage.
func (anthropicAdapter) ParseResponse(raw []byte) (*Completion, error) {
	var ar anthropicResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return nil, fmt.Errorf("decode: %w (body: %s)", err, truncate(raw, 200))
	}
	if ar.Error != nil {
		return nil, fmt.Errorf("api error: %s", ar.Error.Message)
	}
	var content []string
	var calls []ToolCall
	var thinking []string
	var signature string
	var redacted []string
	for _, b := range ar.Content {
		switch b.Type {
		case "text":
			if b.Text != "" {
				cText, rText := extractThinking(b.Text)
				if cText != "" {
					content = append(content, cText)
				}
				if rText != "" {
					thinking = append(thinking, rText)
				}
			}
		case "tool_use":
			args := string(b.Input)
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{ID: b.ID, Type: "function", Function: Func{Name: b.Name, Arguments: args}})
		case "thinking":
			// Preserve Claude extended-thinking instead of dropping it; keep the
			// signature so the block can round-trip back upstream on a later turn.
			if b.Thinking != "" {
				thinking = append(thinking, b.Thinking)
			}
			if b.Signature != "" {
				signature = b.Signature
			}
		case "redacted_thinking":
			// Opaque (encrypted) reasoning — carry it verbatim so it survives the proxy.
			if b.Data != "" {
				redacted = append(redacted, b.Data)
			}
		}
	}
	finish := ar.StopReason
	if len(calls) > 0 {
		finish = "tool_calls"
	}
	msg := Message{
		Role:              RoleAssistant,
		Content:           strings.Join(content, "\n"),
		ToolCalls:         calls,
		Thinking:          strings.Join(thinking, "\n"),
		ThinkingSignature: signature,
		RedactedThinking:  redacted,
	}
	separateMessageReasoning(&msg)
	return normalizeCompletionToolCalls(&Completion{
		Message:      msg,
		FinishReason: finish,
		Model:        ar.Model,
		ServiceTier:  parseServiceTier(ProviderAnthropic, ar.Usage.ServiceTier),
		Usage: Usage{
			PromptTokens:             ar.Usage.InputTokens,
			CompletionTokens:         ar.Usage.OutputTokens,
			TotalTokens:              ar.Usage.InputTokens + ar.Usage.OutputTokens,
			CacheReadInputTokens:     ar.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: ar.Usage.CacheCreationInputTokens,
		},
	}), nil
}
