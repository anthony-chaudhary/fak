package agent

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Provider names the remote transcript wire to use at the model boundary.
type Provider string

const (
	ProviderOpenAI          Provider = "openai"           // GPT / OpenAI-compatible chat completions
	ProviderOpenAIResponses Provider = "openai-responses" // GPT Responses API item wire
	ProviderAnthropic       Provider = "anthropic"        // Claude Messages API
	ProviderGemini          Provider = "gemini"           // Gemini generateContent API
	ProviderXAI             Provider = "xai"              // Grok / xAI chat completions
)

// ParseProvider accepts the public names and common model-family aliases.
func ParseProvider(s string) (Provider, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "openai", "gpt", "chat-completions", "openai-compatible":
		return ProviderOpenAI, true
	case "responses", "responses-api", "openai-responses":
		return ProviderOpenAIResponses, true
	case "anthropic", "claude":
		return ProviderAnthropic, true
	case "gemini", "google":
		return ProviderGemini, true
	case "xai", "grok":
		return ProviderXAI, true
	default:
		return "", false
	}
}

// TranscriptAdapter converts the canonical agent transcript into one provider's
// request/response wire shape. Adapters do not decide policy; HTTPPlanner applies
// pre-send quarantine before invoking them.
type TranscriptAdapter interface {
	Provider() Provider
	Endpoint(baseURL, model string) string
	Headers(apiKey string) map[string]string
	MarshalRequest(adapterRequest) ([]byte, error)
	ParseResponse(raw []byte) (*Completion, error)
}

type adapterRequest struct {
	Model       string
	Messages    []Message
	Tools       []ToolDef
	Temperature float64
	MaxTokens   int
	TopP        *float64 // nil => omit from the wire (planner/provider default)
	TopK        *int     // nil => omit; only the providers with a native top-k field carry it
	Stop        []string // empty => omit from the wire
	// ResponseFormat / LogitBias are the OpenAI structured/guided-decode carriers
	// (#560). They ride on the wire ONLY where the provider has a native field
	// (OpenAI/xAI chat-completions); other providers omit them (their path is
	// ExtraBody). Empty => omit, so an unset structured-decode request is byte-for-byte
	// the pre-seam body.
	ResponseFormat json.RawMessage
	LogitBias      map[int]float64
	ExtraBody      json.RawMessage
}

// NewTranscriptAdapter returns the adapter for a provider.
func NewTranscriptAdapter(provider Provider) (TranscriptAdapter, error) {
	if provider == "" {
		provider = ProviderOpenAI
	}
	switch provider {
	case ProviderOpenAI:
		return openAIAdapter{provider: ProviderOpenAI}, nil
	case ProviderOpenAIResponses:
		return openAIResponsesAdapter{}, nil
	case ProviderXAI:
		return openAIAdapter{provider: ProviderXAI}, nil
	case ProviderAnthropic:
		return anthropicAdapter{}, nil
	case ProviderGemini:
		return geminiAdapter{}, nil
	default:
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
}

func joinEndpoint(baseURL, suffix string) string {
	return strings.TrimRight(baseURL, "/") + suffix
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    any    `json:"code,omitempty"`
}

func providerError(raw []byte) string {
	var doc struct {
		Error *apiError `json:"error"`
	}
	if err := json.Unmarshal(raw, &doc); err == nil && doc.Error != nil && doc.Error.Message != "" {
		return doc.Error.Message
	}
	return ""
}

// ---------------------------------------------------------------------------
// OpenAI-compatible chat completions (OpenAI GPT and xAI Grok).
// ---------------------------------------------------------------------------

type openAIAdapter struct{ provider Provider }

func (a openAIAdapter) Provider() Provider { return a.provider }

func (a openAIAdapter) Endpoint(baseURL, model string) string {
	return joinEndpoint(baseURL, "/chat/completions")
}

func (a openAIAdapter) Headers(apiKey string) map[string]string {
	h := map[string]string{"Content-Type": "application/json"}
	if apiKey != "" {
		h["Authorization"] = "Bearer " + apiKey
	}
	return h
}

type openAIRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []ToolDef       `json:"tools,omitempty"`
	ToolChoice     string          `json:"tool_choice,omitempty"`
	Temperature    float64         `json:"temperature"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"` // #560 structured/guided decode (OpenAI/xAI native)
	LogitBias      map[int]float64 `json:"logit_bias,omitempty"`      // #560 per-token logit mask (OpenAI/xAI native)
}

type openAIResponse struct {
	Model   string `json:"model"` // the model the upstream reports it served (#82 echo)
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage     `json:"usage"`
	Error *apiError `json:"error"`
}

func (a openAIAdapter) MarshalRequest(r adapterRequest) ([]byte, error) {
	toolChoice := ""
	if len(r.Tools) > 0 {
		toolChoice = "auto"
	}
	return marshalWithExtraBody(openAIRequest{
		Model:          r.Model,
		Messages:       r.Messages,
		Tools:          openAICompatibleTools(r.Tools),
		ToolChoice:     toolChoice,
		Temperature:    r.Temperature,
		MaxTokens:      r.MaxTokens,
		TopP:           r.TopP,
		Stop:           r.Stop,
		ResponseFormat: r.ResponseFormat,
		LogitBias:      r.LogitBias,
	}, r.ExtraBody)
}

func openAICompatibleTools(tools []ToolDef) []ToolDef {
	if len(tools) == 0 {
		return tools
	}
	out := make([]ToolDef, len(tools))
	copy(out, tools)
	for i := range out {
		out[i].Function.Parameters = openAICompatibleSchema(out[i].Function.Parameters, true)
	}
	return out
}

func openAICompatibleSchema(raw json.RawMessage, root bool) json.RawMessage {
	var v any
	if len(raw) == 0 || json.Unmarshal(raw, &v) != nil {
		return rawSchema(`{"type":"object","properties":{}}`)
	}
	normalized := normalizeSchemaValue(v, root)
	b, err := json.Marshal(normalized)
	if err != nil {
		return rawSchema(`{"type":"object","properties":{}}`)
	}
	return b
}

func normalizeSchemaValue(v any, root bool) any {
	obj, ok := v.(map[string]any)
	if !ok {
		if root {
			return map[string]any{"type": "object", "properties": map[string]any{}}
		}
		return map[string]any{"type": "string"}
	}
	if props, ok := obj["properties"].(map[string]any); ok {
		for k, child := range props {
			props[k] = normalizeSchemaValue(child, false)
		}
	}
	if items, ok := obj["items"]; ok {
		obj["items"] = normalizeSchemaValue(items, false)
	}
	if addl, ok := obj["additionalProperties"]; ok {
		switch addl.(type) {
		case map[string]any, []any:
			obj["additionalProperties"] = normalizeSchemaValue(addl, false)
		}
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if alts, ok := obj[key].([]any); ok {
			for i, alt := range alts {
				alts[i] = normalizeSchemaValue(alt, false)
			}
		}
	}
	if _, ok := obj["type"]; !ok && !hasSchemaComposition(obj) {
		switch {
		case root || obj["properties"] != nil || obj["required"] != nil || obj["additionalProperties"] != nil:
			obj["type"] = "object"
			if root && obj["properties"] == nil {
				obj["properties"] = map[string]any{}
			}
		case obj["items"] != nil:
			obj["type"] = "array"
		default:
			obj["type"] = "string"
		}
	}
	return obj
}

func hasSchemaComposition(obj map[string]any) bool {
	for _, key := range []string{"anyOf", "oneOf", "allOf", "$ref", "enum", "const"} {
		if _, ok := obj[key]; ok {
			return true
		}
	}
	return false
}

func marshalWithExtraBody(base any, extra json.RawMessage) ([]byte, error) {
	raw, err := json.Marshal(base)
	if err != nil {
		return nil, err
	}
	if len(extra) == 0 {
		return raw, nil
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	var add map[string]json.RawMessage
	if err := json.Unmarshal(extra, &add); err != nil {
		return nil, fmt.Errorf("provider extra body: %w", err)
	}
	for k, v := range add {
		if _, exists := doc[k]; exists {
			return nil, fmt.Errorf("provider extra body must not override %q", k)
		}
		doc[k] = v
	}
	return json.Marshal(doc)
}

func (a openAIAdapter) ParseResponse(raw []byte) (*Completion, error) {
	var cr openAIResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return nil, fmt.Errorf("decode: %w (body: %s)", err, truncate(raw, 200))
	}
	if cr.Error != nil {
		return nil, fmt.Errorf("api error: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("no choices (body: %s)", truncate(raw, 200))
	}
	msg := cr.Choices[0].Message
	finish := cr.Choices[0].FinishReason
	normalizeLegacyOpenAIFunctionCall(&msg, &finish)
	return normalizeCompletionToolCalls(&Completion{
		Message:      msg,
		FinishReason: finish,
		Usage:        cr.Usage,
		Model:        cr.Model,
	}), nil
}

func normalizeLegacyOpenAIFunctionCall(msg *Message, finish *string) {
	if msg == nil || msg.FunctionCall == nil {
		return
	}
	if len(msg.ToolCalls) == 0 && msg.FunctionCall.Name != "" {
		msg.ToolCalls = []ToolCall{{
			ID:       "legacy_function_call",
			Type:     "function",
			Function: *msg.FunctionCall,
		}}
		if finish != nil && *finish == "function_call" {
			*finish = "tool_calls"
		}
	}
	msg.FunctionCall = nil
}

// ---------------------------------------------------------------------------
// OpenAI Responses API.
// ---------------------------------------------------------------------------

type openAIResponsesAdapter struct{}

func (openAIResponsesAdapter) Provider() Provider { return ProviderOpenAIResponses }

func (openAIResponsesAdapter) Endpoint(baseURL, model string) string {
	return joinEndpoint(baseURL, "/responses")
}

func (openAIResponsesAdapter) Headers(apiKey string) map[string]string {
	h := map[string]string{"Content-Type": "application/json"}
	if apiKey != "" {
		h["Authorization"] = "Bearer " + apiKey
	}
	return h
}

type openAIResponsesRequest struct {
	Model           string                `json:"model"`
	Input           []openAIResponsesItem `json:"input"`
	Tools           []openAIResponsesTool `json:"tools,omitempty"`
	ToolChoice      string                `json:"tool_choice,omitempty"`
	Temperature     float64               `json:"temperature"`
	MaxOutputTokens int                   `json:"max_output_tokens,omitempty"`
	TopP            *float64              `json:"top_p,omitempty"` // Responses API has no `stop`
	// Text carries the Responses-API structured-output control. The chat wire's
	// `response_format` carrier (adapterRequest.ResponseFormat) maps here as
	// `text.format`: a `json_schema` body is flattened (its inner `json_schema`
	// wrapper hoisted into `format`), `json_object`/`text` pass through. The
	// Responses API deprecated the flat `response_format` key in favor of this
	// nesting, so this is how the SAME structured-output request reaches /responses.
	Text  *openAIResponsesText `json:"text,omitempty"`
	Store bool                 `json:"store"`
}

// openAIResponsesText is the `text` envelope on the Responses API; only its
// `format` member carries the structured-output spec we forward.
type openAIResponsesText struct {
	Format json.RawMessage `json:"format,omitempty"`
}

type openAIResponsesItem struct {
	Type      string `json:"type,omitempty"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	ID        string `json:"id,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

type openAIResponsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict"`
}

type openAIResponsesResponse struct {
	Status string `json:"status"`
	Model  string `json:"model"` // the model the upstream reports it served (#82 echo)
	Output []struct {
		ID        string `json:"id"`
		Type      string `json:"type"`
		Role      string `json:"role,omitempty"`
		Status    string `json:"status,omitempty"`
		CallID    string `json:"call_id,omitempty"`
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
		Content   []struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		} `json:"content,omitempty"`
	} `json:"output"`
	OutputText string `json:"output_text,omitempty"`
	Usage      struct {
		InputTokens         int                `json:"input_tokens"`
		OutputTokens        int                `json:"output_tokens"`
		TotalTokens         int                `json:"total_tokens"`
		InputTokensDetails  *UsageTokenDetails `json:"input_tokens_details,omitempty"`
		PromptTokensDetails *UsageTokenDetails `json:"prompt_tokens_details,omitempty"`
	} `json:"usage"`
	Error *apiError `json:"error"`
}

func (openAIResponsesAdapter) MarshalRequest(r adapterRequest) ([]byte, error) {
	toolChoice := ""
	if len(r.Tools) > 0 {
		toolChoice = "auto"
	}
	return json.Marshal(openAIResponsesRequest{
		Model:           r.Model,
		Input:           openAIResponsesInput(r.Messages),
		Tools:           openAIResponsesTools(r.Tools),
		ToolChoice:      toolChoice,
		Temperature:     r.Temperature,
		MaxOutputTokens: r.MaxTokens,
		TopP:            r.TopP,
		Text:            responsesText(r.ResponseFormat),
		Store:           false,
	})
}

func openAIResponsesInput(messages []Message) []openAIResponsesItem {
	out := make([]openAIResponsesItem, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case RoleAssistant:
			if m.Content != "" {
				out = append(out, openAIResponsesItem{Type: "message", Role: "assistant", Content: []map[string]string{{
					"type": "output_text", "text": m.Content,
				}}})
			}
			for _, tc := range m.ToolCalls {
				callID := tc.ID
				out = append(out, openAIResponsesItem{
					Type:      "function_call",
					ID:        callID,
					CallID:    callID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				})
			}
		case RoleTool:
			out = append(out, openAIResponsesItem{
				Type:   "function_call_output",
				CallID: m.ToolCallID,
				Output: m.Content,
			})
		default:
			if m.Content != "" {
				out = append(out, openAIResponsesItem{Role: m.Role, Content: m.Content})
			}
		}
	}
	return out
}

func openAIResponsesTools(tools []ToolDef) []openAIResponsesTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]openAIResponsesTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, openAIResponsesTool{
			Type:        "function",
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
			Strict:      false,
		})
	}
	return out
}

func (openAIResponsesAdapter) ParseResponse(raw []byte) (*Completion, error) {
	var rr openAIResponsesResponse
	if err := json.Unmarshal(raw, &rr); err != nil {
		return nil, fmt.Errorf("decode: %w (body: %s)", err, truncate(raw, 200))
	}
	if rr.Error != nil {
		return nil, fmt.Errorf("api error: %s", rr.Error.Message)
	}
	var content []string
	var calls []ToolCall
	for _, item := range rr.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					content = append(content, part.Text)
				}
			}
		case "function_call":
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			args := item.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{ID: id, Type: "function", Function: Func{Name: item.Name, Arguments: args}})
		}
	}
	if len(content) == 0 && rr.OutputText != "" {
		content = append(content, rr.OutputText)
	}
	if len(content) == 0 && len(calls) == 0 {
		return nil, fmt.Errorf("no output items (body: %s)", truncate(raw, 200))
	}
	finish := "stop"
	if len(calls) > 0 {
		finish = "tool_calls"
	} else if rr.Status != "" && rr.Status != "completed" {
		finish = rr.Status
	}
	details := rr.Usage.InputTokensDetails
	if details == nil {
		details = rr.Usage.PromptTokensDetails
	}
	return normalizeCompletionToolCalls(&Completion{
		Message:      Message{Role: RoleAssistant, Content: strings.Join(content, "\n"), ToolCalls: calls},
		FinishReason: finish,
		Model:        rr.Model,
		Usage: Usage{
			PromptTokens:        rr.Usage.InputTokens,
			CompletionTokens:    rr.Usage.OutputTokens,
			TotalTokens:         rr.Usage.TotalTokens,
			PromptTokensDetails: details,
		},
	}), nil
}

// ---------------------------------------------------------------------------
// Anthropic Claude Messages API.
// ---------------------------------------------------------------------------

type anthropicAdapter struct{}

func (anthropicAdapter) Provider() Provider { return ProviderAnthropic }

func (anthropicAdapter) Endpoint(baseURL, model string) string {
	return joinEndpoint(baseURL, "/v1/messages")
}

func (anthropicAdapter) Headers(apiKey string) map[string]string {
	h := map[string]string{
		"Content-Type":      "application/json",
		"anthropic-version": "2023-06-01",
	}
	if apiKey != "" {
		h["x-api-key"] = apiKey
	}
	return h
}

type anthropicRequest struct {
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
		Type  string          `json:"type"`
		Text  string          `json:"text,omitempty"`
		ID    string          `json:"id,omitempty"`
		Name  string          `json:"name,omitempty"`
		Input json.RawMessage `json:"input,omitempty"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	} `json:"usage"`
	Error *apiError `json:"error"`
}

func (anthropicAdapter) MarshalRequest(r adapterRequest) ([]byte, error) {
	maxTokens := r.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}
	req := anthropicRequest{
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
	for _, b := range ar.Content {
		switch b.Type {
		case "text":
			if b.Text != "" {
				content = append(content, b.Text)
			}
		case "tool_use":
			args := string(b.Input)
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{ID: b.ID, Type: "function", Function: Func{Name: b.Name, Arguments: args}})
		}
	}
	finish := ar.StopReason
	if len(calls) > 0 {
		finish = "tool_calls"
	}
	return normalizeCompletionToolCalls(&Completion{
		Message:      Message{Role: RoleAssistant, Content: strings.Join(content, "\n"), ToolCalls: calls},
		FinishReason: finish,
		Model:        ar.Model,
		Usage: Usage{
			PromptTokens:             ar.Usage.InputTokens,
			CompletionTokens:         ar.Usage.OutputTokens,
			TotalTokens:              ar.Usage.InputTokens + ar.Usage.OutputTokens,
			CacheReadInputTokens:     ar.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: ar.Usage.CacheCreationInputTokens,
		},
	}), nil
}

// ---------------------------------------------------------------------------
// Gemini native generateContent API.
// ---------------------------------------------------------------------------

type geminiAdapter struct{}

func (geminiAdapter) Provider() Provider { return ProviderGemini }

func (geminiAdapter) Endpoint(baseURL, model string) string {
	modelPath := strings.TrimLeft(model, "/")
	if !strings.HasPrefix(modelPath, "models/") {
		modelPath = "models/" + modelPath
	}
	return joinEndpoint(baseURL, "/"+modelPath+":generateContent")
}

func (geminiAdapter) Headers(apiKey string) map[string]string {
	h := map[string]string{"Content-Type": "application/json"}
	if apiKey != "" {
		h["x-goog-api-key"] = apiKey
	}
	return h
}

type geminiRequest struct {
	Contents          []geminiContent `json:"contents"`
	SystemInstruction *geminiContent  `json:"systemInstruction,omitempty"`
	Tools             []geminiTool    `json:"tools,omitempty"`
	GenerationConfig  *geminiConfig   `json:"generationConfig,omitempty"`
}

type geminiConfig struct {
	Temperature   float64  `json:"temperature"`
	MaxTokens     int      `json:"maxOutputTokens,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	TopK          *int     `json:"topK,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args,omitempty"`
	ID   string `json:"id,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string `json:"name"`
	ID       string `json:"id,omitempty"`
	Response any    `json:"response"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content      geminiResponseContent `json:"content"`
		FinishReason string                `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		TotalTokenCount         int `json:"totalTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	} `json:"usageMetadata"`
	ModelVersion string    `json:"modelVersion,omitempty"` // the model the upstream reports it served (#82 echo)
	Error        *apiError `json:"error"`
}

type geminiResponseContent struct {
	Role  string               `json:"role,omitempty"`
	Parts []geminiResponsePart `json:"parts"`
}

type geminiResponsePart struct {
	Text         string `json:"text,omitempty"`
	FunctionCall *struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args,omitempty"`
		ID   string          `json:"id,omitempty"`
	} `json:"functionCall,omitempty"`
}

func (geminiAdapter) MarshalRequest(r adapterRequest) ([]byte, error) {
	req := geminiRequest{
		Contents: make([]geminiContent, 0, len(r.Messages)),
		Tools:    geminiTools(r.Tools),
		GenerationConfig: &geminiConfig{
			Temperature:   r.Temperature,
			MaxTokens:     r.MaxTokens,
			TopP:          r.TopP,
			TopK:          positiveTopK(r.TopK),
			StopSequences: r.Stop,
		},
	}
	for _, m := range r.Messages {
		switch m.Role {
		case RoleSystem:
			if m.Content != "" {
				if req.SystemInstruction == nil {
					req.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: m.Content}}}
				} else {
					req.SystemInstruction.Parts = append(req.SystemInstruction.Parts, geminiPart{Text: m.Content})
				}
			}
		case RoleAssistant:
			parts := geminiAssistantParts(m)
			if len(parts) > 0 {
				req.Contents = append(req.Contents, geminiContent{Role: "model", Parts: parts})
			}
		case RoleTool:
			req.Contents = append(req.Contents, geminiContent{Role: "user", Parts: []geminiPart{{
				FunctionResponse: &geminiFunctionResponse{
					Name:     m.Name,
					ID:       m.ToolCallID,
					Response: responseObject(m.Content),
				},
			}}})
		default:
			if m.Content != "" {
				req.Contents = append(req.Contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: m.Content}}})
			}
		}
	}
	return json.Marshal(req)
}

func geminiAssistantParts(m Message) []geminiPart {
	parts := make([]geminiPart, 0, 1+len(m.ToolCalls))
	if m.Content != "" {
		parts = append(parts, geminiPart{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{
			Name: tc.Function.Name,
			Args: jsonObjectOrRaw(tc.Function.Arguments),
			ID:   tc.ID,
		}})
	}
	return parts
}

func geminiTools(tools []ToolDef) []geminiTool {
	if len(tools) == 0 {
		return nil
	}
	decls := make([]geminiFunctionDeclaration, 0, len(tools))
	for _, t := range tools {
		decls = append(decls, geminiFunctionDeclaration{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  geminiSchema(t.Function.Parameters),
		})
	}
	return []geminiTool{{FunctionDeclarations: decls}}
}

func geminiSchema(raw json.RawMessage) any {
	var v any
	if len(raw) == 0 || json.Unmarshal(raw, &v) != nil {
		return raw
	}
	return uppercaseSchemaTypes(v)
}

func uppercaseSchemaTypes(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			if k == "type" {
				if s, ok := val.(string); ok {
					x[k] = strings.ToUpper(s)
					continue
				}
			}
			x[k] = uppercaseSchemaTypes(val)
		}
	case []any:
		for i, val := range x {
			x[i] = uppercaseSchemaTypes(val)
		}
	}
	return v
}

func (geminiAdapter) ParseResponse(raw []byte) (*Completion, error) {
	var gr geminiResponse
	if err := json.Unmarshal(raw, &gr); err != nil {
		return nil, fmt.Errorf("decode: %w (body: %s)", err, truncate(raw, 200))
	}
	if gr.Error != nil {
		return nil, fmt.Errorf("api error: %s", gr.Error.Message)
	}
	if len(gr.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates (body: %s)", truncate(raw, 200))
	}
	c := gr.Candidates[0]
	var content []string
	var calls []ToolCall
	for _, p := range c.Content.Parts {
		if p.Text != "" {
			content = append(content, p.Text)
		}
		if p.FunctionCall != nil {
			args := string(p.FunctionCall.Args)
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{
				ID:   p.FunctionCall.ID,
				Type: "function",
				Function: Func{
					Name:      p.FunctionCall.Name,
					Arguments: args,
				},
			})
		}
	}
	finish := strings.ToLower(c.FinishReason)
	if len(calls) > 0 {
		finish = "tool_calls"
	}
	var details *UsageTokenDetails
	if gr.UsageMetadata.CachedContentTokenCount > 0 {
		details = &UsageTokenDetails{CachedTokens: gr.UsageMetadata.CachedContentTokenCount}
	}
	return normalizeCompletionToolCalls(&Completion{
		Message:      Message{Role: RoleAssistant, Content: strings.Join(content, "\n"), ToolCalls: calls},
		FinishReason: finish,
		Model:        gr.ModelVersion,
		Usage: Usage{
			PromptTokens:        gr.UsageMetadata.PromptTokenCount,
			CompletionTokens:    gr.UsageMetadata.CandidatesTokenCount,
			TotalTokens:         gr.UsageMetadata.TotalTokenCount,
			PromptTokensDetails: details,
		},
	}), nil
}

// ---------------------------------------------------------------------------
// shared adapter helpers
// ---------------------------------------------------------------------------

// positiveTopK normalizes a per-request top-k for the wire: nil (unset) and any
// non-positive value (the planner's "0 => no truncation" convention) both omit the
// field, because the native top_k providers (Anthropic, Gemini) require a positive
// integer and reject 0/negative. A positive k is forwarded as-is. Returning a
// *int keeps the request struct's `omitempty` working — a nil result drops the key.
func positiveTopK(k *int) *int {
	if k == nil || *k <= 0 {
		return nil
	}
	v := *k
	return &v
}

// responsesText maps the chat-style `response_format` carrier onto the Responses
// API's `text.format` shape. The two OpenAI surfaces spell structured output
// differently: chat nests the schema under `response_format.json_schema.{name,
// strict,schema}`, while Responses flattens it to `text.format.{type:"json_schema",
// name,strict,schema}`. So a `json_schema` carrier is rewritten with its inner
// `json_schema` object's members hoisted up alongside `type`; a `json_object` /
// `text` carrier (no inner wrapper) passes through verbatim. An unset or
// unparseable carrier returns nil so the `omitempty` drops `text` entirely —
// the body is then byte-for-byte the pre-seam request.
func responsesText(rf json.RawMessage) *openAIResponsesText {
	if len(rf) == 0 {
		return nil
	}
	var carrier map[string]json.RawMessage
	if err := json.Unmarshal(rf, &carrier); err != nil {
		return nil
	}
	typ := carrier["type"]
	if len(typ) == 0 {
		return nil
	}
	// json_schema: hoist the inner json_schema members up beside `type`.
	if string(typ) == `"json_schema"` {
		var inner map[string]json.RawMessage
		if err := json.Unmarshal(carrier["json_schema"], &inner); err != nil || inner == nil {
			// Malformed/absent inner wrapper — forward the carrier as-is rather
			// than drop it, so an odd shape still reaches the provider's validator.
			return &openAIResponsesText{Format: rf}
		}
		flat := make(map[string]json.RawMessage, len(inner)+1)
		flat["type"] = typ
		for k, v := range inner {
			flat[k] = v
		}
		format, err := json.Marshal(flat)
		if err != nil {
			return &openAIResponsesText{Format: rf}
		}
		return &openAIResponsesText{Format: format}
	}
	// json_object / text and any other typed shape: pass the carrier through verbatim.
	return &openAIResponsesText{Format: rf}
}

func jsonObjectOrRaw(raw string) any {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err == nil {
		if _, ok := v.(map[string]any); ok {
			return v
		}
	}
	return map[string]any{"_raw": raw}
}

func responseObject(content string) any {
	var v any
	if err := json.Unmarshal([]byte(content), &v); err == nil {
		if _, ok := v.(map[string]any); ok {
			return v
		}
	}
	return map[string]any{"content": content}
}
