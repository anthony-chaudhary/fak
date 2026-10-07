package agent

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Gemini native generateContent API.
// ---------------------------------------------------------------------------

type geminiAdapter struct{}

// Provider reports ProviderGemini (the native generateContent API wire).
func (geminiAdapter) Provider() Provider { return ProviderGemini }

func (geminiAdapter) Endpoint(baseURL, model string) string {
	modelPath := strings.TrimLeft(model, "/")
	if !strings.HasPrefix(modelPath, "models/") {
		modelPath = "models/" + modelPath
	}
	return joinEndpoint(baseURL, "/"+modelPath+":generateContent")
}

// Headers sets Content-Type and, when apiKey is non-empty, the Gemini
// x-goog-api-key header.
func (geminiAdapter) Headers(apiKey string) map[string]string {
	return jsonAuthHeaders(apiKey, "x-goog-api-key", apiKey)
}

type geminiRequest struct {
	Contents          []geminiContent `json:"contents"`
	SystemInstruction *geminiContent  `json:"systemInstruction,omitempty"`
	Tools             []geminiTool    `json:"tools,omitempty"`
	GenerationConfig  *geminiConfig   `json:"generationConfig,omitempty"`
}

type geminiThinkingConfig struct {
	ThinkingBudget *int   `json:"thinkingBudget,omitempty"`
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
}

type geminiConfig struct {
	Temperature    float64               `json:"temperature"`
	MaxTokens      int                   `json:"maxOutputTokens,omitempty"`
	TopP           *float64              `json:"topP,omitempty"`
	TopK           *int                  `json:"topK,omitempty"`
	StopSequences  []string              `json:"stopSequences,omitempty"`
	ThinkingConfig *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"`
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
	Text             string `json:"text,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	FunctionCall     *struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args,omitempty"`
		ID   string          `json:"id,omitempty"`
	} `json:"functionCall,omitempty"`
}

// MarshalRequest encodes the canonical request as a Gemini generateContent body:
// system messages become systemInstruction, assistant turns map to "model" contents
// with functionCall parts, tool results map to functionResponse parts, and the
// sampling controls (with uppercased tool-schema types) go in generationConfig.
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
	if r.ThinkingBudget != nil {
		if *r.ThinkingBudget > 0 {
			req.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
				ThinkingBudget: r.ThinkingBudget,
			}
		}
	} else if r.ReasoningEffort != "" {
		req.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
			ThinkingLevel: strings.ToLower(r.ReasoningEffort),
		}
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
			// A run of adjacent tool results is one logical turn (the several results of
			// one assistant turn's parallel tool calls). Fold them into ONE user content
			// with N functionResponse parts rather than N stacked user turns, so Gemini
			// does not read the results as serialized turns that suppress parallelism
			// (#5797). Run order is preserved.
			part := geminiPart{FunctionResponse: &geminiFunctionResponse{
				Name:     m.Name,
				ID:       m.ToolCallID,
				Response: responseObject(m.Content),
			}}
			if n := len(req.Contents); n > 0 && geminiFunctionResponseOnly(req.Contents[n-1]) {
				req.Contents[n-1].Parts = append(req.Contents[n-1].Parts, part)
			} else {
				req.Contents = append(req.Contents, geminiContent{Role: "user", Parts: []geminiPart{part}})
			}
		default:
			if m.Content != "" {
				req.Contents = append(req.Contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: m.Content}}})
			}
		}
	}
	// Google Gemini GenerateContent API strictly requires alternating turns and
	// rejects any request ending with a model turn with HTTP 400:
	// "Requests ending with a model turn are not supported" (anomalyco/opencode #47034,
	// decolua/9router #3816, fak #11464).
	// If the projected transcript ends on a model turn without function calls
	// (e.g. trailing reasoning-only or assistant text awaiting continuation), inject
	// a user continuation prompt so the wire call satisfies turn alternation.
	if len(req.Contents) > 0 && req.Contents[len(req.Contents)-1].Role == "model" {
		lastContent := req.Contents[len(req.Contents)-1]
		hasFunctionCall := false
		for _, p := range lastContent.Parts {
			if p.FunctionCall != nil {
				hasFunctionCall = true
				break
			}
		}
		if !hasFunctionCall {
			req.Contents = append(req.Contents, geminiContent{
				Role:  "user",
				Parts: []geminiPart{{Text: "Continue"}},
			})
		}
	}
	return json.Marshal(req)
}

// geminiFunctionResponseOnly reports whether every part is a functionResponse — i.e.
// the content is a pure tool-result turn a subsequent result may join without mixing
// an assistant turn or a genuine user text turn into it.
func geminiFunctionResponseOnly(c geminiContent) bool {
	if len(c.Parts) == 0 {
		return false
	}
	for _, p := range c.Parts {
		if p.FunctionResponse == nil {
			return false
		}
	}
	return true
}

func geminiAssistantParts(m Message) []geminiPart {
	parts := make([]geminiPart, 0, 1+len(m.ToolCalls))
	if m.Content != "" {
		parts = append(parts, geminiPart{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, geminiPart{
			ThoughtSignature: tc.ThoughtSignature,
			FunctionCall: &geminiFunctionCall{
				Name: tc.Function.Name,
				Args: jsonObjectOrRaw(tc.Function.Arguments),
				ID:   tc.ID,
			},
		})
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
	// Same per-turn re-normalization waste as the OpenAI adapter (#796): memoize the
	// (provider, raw-bytes) -> normalized-bytes mapping, content-addressed and self-
	// invalidating. We cache the marshaled bytes (returned as json.RawMessage, which the
	// request marshaler emits verbatim — byte-identical to marshaling the uppercased tree
	// because Go's encoder sorts map keys deterministically) rather than the uppercased
	// map tree, so a hit never aliases a map a caller could mutate. Provider is in the key
	// because OpenAI lowercases/fills type while Gemini uppercases it — the same raw schema
	// normalizes differently per adapter.
	if cached, ok := loadNormalizedSchema(schemaCacheKeyGemini, false, raw); ok {
		return cached
	}
	out := geminiSchemaCompute(raw)
	if b, ok := out.(json.RawMessage); ok {
		storeNormalizedSchema(schemaCacheKeyGemini, false, raw, b)
	}
	return out
}

func geminiSchemaCompute(raw json.RawMessage) any {
	var v any
	if len(raw) == 0 || json.Unmarshal(raw, &v) != nil {
		return raw
	}
	v = sanitizeGeminiSchema(v)
	b, err := json.Marshal(uppercaseSchemaTypes(v))
	if err != nil {
		return raw
	}
	return json.RawMessage(b)
}

// sanitizeGeminiSchema strips or repairs schema constructs that Gemini's
// function_declarations validation rejects. Specifically, Gemini requires that:
//  1. A schema with "required" must have type OBJECT.
//  2. Every field in "required" must be defined in "properties" of that schema.
//
// Constructs like "anyOf": [{"required": ["a"]}, {"required": ["b"]}] (a draft-07
// idiom for "either a or b is required") violate both rules and produce
// GenerateContentRequest 400s on the Gemini API wire.
func sanitizeGeminiSchema(v any) any {
	m, ok := v.(map[string]any)
	if !ok {
		if arr, ok := v.([]any); ok {
			for i, elem := range arr {
				arr[i] = sanitizeGeminiSchema(elem)
			}
		}
		return v
	}

	delete(m, "additionalProperties")
	delete(m, "$schema")

	for _, key := range []string{"anyOf", "any_of", "oneOf", "one_of"} {
		if alts, ok := m[key].([]any); ok {
			validAlts := make([]any, 0, len(alts))
			for _, alt := range alts {
				sanitized := sanitizeGeminiSchema(alt)
				altMap, ok := sanitized.(map[string]any)
				if !ok {
					validAlts = append(validAlts, sanitized)
					continue
				}
				if len(altMap) == 0 {
					continue
				}
				t, _ := altMap["type"].(string)
				isObject := strings.EqualFold(t, "object")
				props, _ := altMap["properties"].(map[string]any)
				if isObject && len(props) == 0 && altMap["anyOf"] == nil && altMap["any_of"] == nil && altMap["oneOf"] == nil && altMap["one_of"] == nil && altMap["allOf"] == nil && altMap["$ref"] == nil {
					// An object branch with no properties or composition (e.g. leftover
					// from a stripped draft-07 required constraint) is invalid for Gemini.
					continue
				}
				validAlts = append(validAlts, altMap)
			}
			if len(validAlts) == 0 {
				delete(m, key)
			} else {
				m[key] = validAlts
			}
		}
	}

	if allOf, ok := m["allOf"].([]any); ok {
		var validAllOf []any
		for _, item := range allOf {
			sanitized := sanitizeGeminiSchema(item)
			if sMap, ok := sanitized.(map[string]any); ok && len(sMap) == 0 {
				continue
			}
			validAllOf = append(validAllOf, sanitized)
		}
		if len(validAllOf) == 0 {
			delete(m, "allOf")
		} else {
			m["allOf"] = validAllOf
		}
	}

	if props, ok := m["properties"].(map[string]any); ok {
		for k, child := range props {
			props[k] = sanitizeGeminiSchema(child)
		}
	}
	if items, ok := m["items"]; ok {
		m["items"] = sanitizeGeminiSchema(items)
	}

	t, _ := m["type"].(string)
	if strings.EqualFold(t, "array") {
		// Gemini requires that every ARRAY schema specifies an items definition.
		// If missing or an empty object (e.g. from MadAppGang/claudish #232), populate default type.
		if itemsVal, hasItems := m["items"]; !hasItems || itemsVal == nil {
			m["items"] = map[string]any{"type": "STRING"}
		} else if itemsMap, isMap := itemsVal.(map[string]any); isMap {
			if len(itemsMap) == 0 {
				m["items"] = map[string]any{"type": "STRING"}
			} else if itemsMap["type"] == nil && itemsMap["$ref"] == nil && itemsMap["properties"] == nil && itemsMap["anyOf"] == nil && itemsMap["oneOf"] == nil && itemsMap["allOf"] == nil {
				itemsMap["type"] = "STRING"
			}
		}
	}

	sanitizeSchemaRequired(m)
	return m
}

func sanitizeSchemaRequired(m map[string]any) {
	reqVal, hasReq := m["required"]
	if !hasReq {
		return
	}

	var reqList []string
	switch r := reqVal.(type) {
	case []any:
		for _, item := range r {
			if s, ok := item.(string); ok {
				reqList = append(reqList, s)
			}
		}
	case []string:
		reqList = r
	default:
		delete(m, "required")
		return
	}

	t, _ := m["type"].(string)
	isObject := strings.EqualFold(t, "object")
	props, _ := m["properties"].(map[string]any)

	// If type is omitted but properties are defined, infer object type
	if !isObject && props != nil && t == "" {
		m["type"] = "object"
		isObject = true
	}

	if !isObject || props == nil {
		delete(m, "required")
		return
	}

	var validReq []any
	for _, rStr := range reqList {
		if props[rStr] != nil {
			validReq = append(validReq, rStr)
		}
	}
	if len(validReq) == 0 {
		delete(m, "required")
	} else {
		m["required"] = validReq
	}
}

// ParseResponse decodes a Gemini generateContent response into a Completion: the
// first candidate's text parts become content and its functionCall parts become tool
// calls, the finishReason is lowercased (or "tool_calls" when calls are present), and
// the usageMetadata token counts (including cached content) map into Usage.
func (a geminiAdapter) ParseResponse(raw []byte) (*Completion, error) {
	comp, err := a.parseResponseFields(raw)
	return normalizeCompletionToolCalls(comp), err
}

func (a geminiAdapter) parseResponseFields(raw []byte) (*Completion, error) {
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
	var reasoning []string
	var calls []ToolCall
	for _, p := range c.Content.Parts {
		if p.Text != "" {
			cText, rText := extractThinking(p.Text)
			if cText != "" {
				content = append(content, cText)
			}
			if rText != "" {
				reasoning = append(reasoning, rText)
			}
		}
		if p.FunctionCall != nil {
			args := string(p.FunctionCall.Args)
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{
				ID:               p.FunctionCall.ID,
				Type:             "function",
				ThoughtSignature: p.ThoughtSignature,
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
	msg := Message{
		Role:             RoleAssistant,
		Content:          strings.Join(content, "\n"),
		ReasoningContent: strings.Join(reasoning, "\n"),
		ToolCalls:        calls,
	}
	separateMessageReasoning(&msg)
	return &Completion{
		Message:      msg,
		FinishReason: finish,
		Model:        gr.ModelVersion,
		Usage: Usage{
			PromptTokens:        gr.UsageMetadata.PromptTokenCount,
			CompletionTokens:    gr.UsageMetadata.CandidatesTokenCount,
			TotalTokens:         gr.UsageMetadata.TotalTokenCount,
			PromptTokensDetails: details,
		},
	}, nil
}
