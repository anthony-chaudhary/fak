package gateway

import (
	"errors"
	"fmt"
	"strings"
)

// MCPToolAnnotation defines standard MCP tool annotations (e.g. MCP 2024-11-05+)
// along with per-tool granular capability annotations for reasoning models (#11527).
type MCPToolAnnotation struct {
	ReadOnly            *bool `json:"readOnly,omitempty"`
	Idempotent          *bool `json:"idempotent,omitempty"`
	Consequential       *bool `json:"consequential,omitempty"`
	ReadOnlyHint        bool  `json:"readOnlyHint,omitempty"`
	ReadOnlyHintSnake   bool  `json:"read_only_hint,omitempty"`
	IdempotentHint      bool  `json:"idempotentHint,omitempty"`
	IdempotentHintSnake bool  `json:"idempotent_hint,omitempty"`

	// Granular capability hints for model deliberation (#11527).
	// Distinguishes safe local mutations from high-risk open-world actions.
	DestructiveHint bool `json:"destructive_hint,omitempty"`
	OpenWorldHint   bool `json:"open_world_hint,omitempty"`
}

// SchemaAdaptationProfile identifies a target model or provider's JSON schema dialect.
type SchemaAdaptationProfile string

const (
	// StrictOpenAIV2 enforces OpenAI Structured Outputs: strict additionalProperties=false,
	// all properties required, optional fields made nullable unions, draft keywords stripped.
	StrictOpenAIV2 SchemaAdaptationProfile = "StrictOpenAIV2"

	// PermissiveClaudeThinking retains rich descriptions, markdown documentation, rich constraints
	// (minLength, maxLength, pattern, default), nullable types, and permissive additionalProperties.
	PermissiveClaudeThinking SchemaAdaptationProfile = "PermissiveClaudeThinking"

	// GeminiToolsDialect converts schemas to OpenAPI 3.0: single string type, nullable: true boolean,
	// unwrapped anyOf nullable unions, and strips unsupported draft/2020-12 keywords and additionalProperties.
	GeminiToolsDialect SchemaAdaptationProfile = "GeminiToolsDialect"

	// StandardDraft7 canonicalizes and passes through standard JSON Schema Draft 7 structures.
	StandardDraft7 SchemaAdaptationProfile = "StandardDraft7"
)

// SchemaTransformer defines the interface for adapting raw MCP tool input schemas into a provider dialect.
type SchemaTransformer interface {
	Transform(rawSchema map[string]any) (map[string]any, error)
}

// GetSchemaTransformer returns the SchemaTransformer implementation for the specified profile.
func GetSchemaTransformer(profile SchemaAdaptationProfile) (SchemaTransformer, error) {
	switch profile {
	case StrictOpenAIV2:
		return strictOpenAIV2Transformer{}, nil
	case PermissiveClaudeThinking:
		return permissiveClaudeThinkingTransformer{}, nil
	case GeminiToolsDialect:
		return geminiToolsDialectTransformer{}, nil
	case StandardDraft7:
		return standardDraft7Transformer{}, nil
	default:
		return nil, fmt.Errorf("unsupported schema adaptation profile: %q", profile)
	}
}

// AdaptMCPSchema adapts a canonical MCP tool schema into the dialect demanded by the specified profile.
// The input rawSchema map is deeply copied and never modified in place.
func AdaptMCPSchema(rawSchema map[string]any, profile SchemaAdaptationProfile) (map[string]any, error) {
	if rawSchema == nil {
		return nil, errors.New("schema cannot be nil")
	}
	transformer, err := GetSchemaTransformer(profile)
	if err != nil {
		return nil, err
	}
	return transformer.Transform(rawSchema)
}

// ResolveSchemaProfile selects the appropriate schema adaptation profile based on the model identifier.
func ResolveSchemaProfile(model string) SchemaAdaptationProfile {
	norm := strings.ToLower(strings.TrimSpace(model))
	switch norm {
	case "strictopenaiv2", "strictopenai", "strict", "openai", "strict_openai_v2":
		return StrictOpenAIV2
	case "permissiveclaudethinking", "claude", "anthropic", "permissive_claude_thinking":
		return PermissiveClaudeThinking
	case "geminitoolsdialect", "gemini", "vertex", "gemini_tools_dialect":
		return GeminiToolsDialect
	case "standarddraft7", "draft7", "standard_draft7", "openapi":
		return StandardDraft7
	}

	switch {
	case strings.Contains(norm, "claude") || strings.Contains(norm, "anthropic"):
		return PermissiveClaudeThinking
	case strings.Contains(norm, "gemini") || strings.Contains(norm, "vertex"):
		return GeminiToolsDialect
	case strings.Contains(norm, "gpt-") || strings.Contains(norm, "o1") || strings.Contains(norm, "o3") ||
		strings.Contains(norm, "o4") || strings.Contains(norm, "astra") || strings.Contains(norm, "chatgpt") ||
		strings.Contains(norm, "openai"):
		return StrictOpenAIV2
	default:
		return StandardDraft7
	}
}

func cloneMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	cp := make(map[string]any, len(m))
	for k, v := range m {
		cp[k] = cloneValue(v)
	}
	return cp
}

func cloneValue(v any) any {
	switch val := v.(type) {
	case map[string]any:
		return cloneMap(val)
	case []any:
		cp := make([]any, len(val))
		for i, item := range val {
			cp[i] = cloneValue(item)
		}
		return cp
	case []string:
		cp := make([]string, len(val))
		copy(cp, val)
		return cp
	default:
		return val
	}
}

// strictOpenAIV2Transformer implements OpenAI Structured Outputs strict mode.
type strictOpenAIV2Transformer struct{}

func (t strictOpenAIV2Transformer) Transform(rawSchema map[string]any) (map[string]any, error) {
	cloned := cloneMap(rawSchema)
	return strictTransformObject(cloned), nil
}

// permissiveClaudeThinkingTransformer implements Claude / Thinking models permissive mode.
type permissiveClaudeThinkingTransformer struct{}

func (t permissiveClaudeThinkingTransformer) Transform(rawSchema map[string]any) (map[string]any, error) {
	cloned := cloneMap(rawSchema)
	return transformClaudeSchema(cloned, true), nil
}

func transformClaudeSchema(s map[string]any, isRoot bool) map[string]any {
	if s == nil {
		return nil
	}

	delete(s, "$schema")
	delete(s, "$id")
	delete(s, "id")

	if isRoot {
		if s["type"] == nil || s["type"] == "" {
			s["type"] = "object"
		}
		if s["type"] == "object" && s["properties"] == nil {
			s["properties"] = map[string]any{}
		}
	}

	if props, ok := s["properties"].(map[string]any); ok {
		for k, v := range props {
			if pMap, isMap := v.(map[string]any); isMap {
				props[k] = transformClaudeSchema(pMap, false)
			}
		}
	}

	if itemsMap, ok := s["items"].(map[string]any); ok {
		s["items"] = transformClaudeSchema(itemsMap, false)
	} else if itemsList, ok := s["items"].([]any); ok {
		for i, item := range itemsList {
			if itemMap, isMap := item.(map[string]any); isMap {
				itemsList[i] = transformClaudeSchema(itemMap, false)
			}
		}
	}

	for _, k := range []string{"anyOf", "allOf", "oneOf"} {
		if list, ok := s[k].([]any); ok {
			for i, item := range list {
				if itemMap, isMap := item.(map[string]any); isMap {
					list[i] = transformClaudeSchema(itemMap, false)
				}
			}
		}
	}

	for _, k := range []string{"$defs", "definitions"} {
		if defs, ok := s[k].(map[string]any); ok {
			for dk, dv := range defs {
				if dMap, isMap := dv.(map[string]any); isMap {
					defs[dk] = transformClaudeSchema(dMap, false)
				}
			}
		}
	}

	return s
}

// geminiToolsDialectTransformer implements OpenAPI 3.0 / Gemini tools schema dialect.
type geminiToolsDialectTransformer struct{}

func (t geminiToolsDialectTransformer) Transform(rawSchema map[string]any) (map[string]any, error) {
	cloned := cloneMap(rawSchema)
	return transformGeminiSchema(cloned, true), nil
}

func transformGeminiSchema(s map[string]any, isRoot bool) map[string]any {
	if s == nil {
		return nil
	}

	// Strip unsupported OpenAPI 3.0 / Gemini FunctionDeclaration keywords.
	delete(s, "$schema")
	delete(s, "$id")
	delete(s, "id")
	delete(s, "$defs")
	delete(s, "definitions")
	delete(s, "dependentRequired")
	delete(s, "dependentSchemas")
	delete(s, "dependencies")
	delete(s, "patternProperties")
	delete(s, "unevaluatedProperties")
	delete(s, "unevaluatedItems")
	delete(s, "prefixItems")
	delete(s, "propertyNames")
	delete(s, "contains")
	delete(s, "minContains")
	delete(s, "maxContains")
	delete(s, "if")
	delete(s, "then")
	delete(s, "else")
	delete(s, "not")
	delete(s, "additionalProperties") // Gemini rejects additionalProperties in tool declarations

	// Convert const: val to enum: [val]
	if constVal, hasConst := s["const"]; hasConst {
		delete(s, "const")
		s["enum"] = []any{constVal}
	}

	// Handle array types: type: ["string", "null"] -> type: "string", nullable: true
	if tSlice, ok := s["type"].([]any); ok {
		var primaryType string
		isNullable := false
		for _, it := range tSlice {
			str, ok := it.(string)
			if !ok {
				continue
			}
			if str == "null" {
				isNullable = true
			} else if primaryType == "" {
				primaryType = str
			}
		}
		if primaryType != "" {
			s["type"] = primaryType
		} else {
			s["type"] = "string"
		}
		if isNullable {
			s["nullable"] = true
		}
	} else if tStrSlice, ok := s["type"].([]string); ok {
		var primaryType string
		isNullable := false
		for _, str := range tStrSlice {
			if str == "null" {
				isNullable = true
			} else if primaryType == "" {
				primaryType = str
			}
		}
		if primaryType != "" {
			s["type"] = primaryType
		} else {
			s["type"] = "string"
		}
		if isNullable {
			s["nullable"] = true
		}
	} else if tStr, ok := s["type"].(string); ok && tStr == "null" {
		s["type"] = "string"
		s["nullable"] = true
	}

	// Handle anyOf nullable unions: anyOf: [{"type": "string", ...}, {"type": "null"}]
	if anyOfVal, ok := s["anyOf"].([]any); ok {
		var nonNullBranches []map[string]any
		hasNullBranch := false
		for _, it := range anyOfVal {
			itMap, isMap := it.(map[string]any)
			if !isMap {
				continue
			}
			if isNullSchemaNode(itMap) {
				hasNullBranch = true
			} else {
				nonNullBranches = append(nonNullBranches, itMap)
			}
		}
		if hasNullBranch {
			s["nullable"] = true
			if len(nonNullBranches) == 1 {
				// Unwrap single non-null branch directly into s
				delete(s, "anyOf")
				for k, v := range nonNullBranches[0] {
					if _, exists := s[k]; !exists {
						s[k] = v
					}
				}
			} else if len(nonNullBranches) > 1 {
				newAnyOf := make([]any, len(nonNullBranches))
				for i, b := range nonNullBranches {
					newAnyOf[i] = b
				}
				s["anyOf"] = newAnyOf
			}
		}
	}

	if isRoot {
		if s["type"] == nil || s["type"] == "" {
			s["type"] = "object"
		}
		if s["type"] == "object" && s["properties"] == nil {
			s["properties"] = map[string]any{}
		}
	}

	if props, ok := s["properties"].(map[string]any); ok {
		for k, v := range props {
			if pMap, isMap := v.(map[string]any); isMap {
				props[k] = transformGeminiSchema(pMap, false)
			}
		}
	}

	if itemsMap, ok := s["items"].(map[string]any); ok {
		s["items"] = transformGeminiSchema(itemsMap, false)
	} else if itemsList, ok := s["items"].([]any); ok {
		if len(itemsList) > 0 {
			if firstMap, ok := itemsList[0].(map[string]any); ok {
				s["items"] = transformGeminiSchema(firstMap, false)
			}
		}
	}

	for _, k := range []string{"anyOf", "allOf", "oneOf"} {
		if list, ok := s[k].([]any); ok {
			for i, item := range list {
				if itemMap, isMap := item.(map[string]any); isMap {
					list[i] = transformGeminiSchema(itemMap, false)
				}
			}
		}
	}

	return s
}

func isNullSchemaNode(m map[string]any) bool {
	if t, ok := m["type"].(string); ok && t == "null" {
		return true
	}
	if tList, ok := m["type"].([]any); ok && len(tList) == 1 {
		if tStr, ok := tList[0].(string); ok && tStr == "null" {
			return true
		}
	}
	if tList, ok := m["type"].([]string); ok && len(tList) == 1 {
		if tList[0] == "null" {
			return true
		}
	}
	return false
}

// standardDraft7Transformer canonicalizes standard Draft 7 JSON schemas.
type standardDraft7Transformer struct{}

func (t standardDraft7Transformer) Transform(rawSchema map[string]any) (map[string]any, error) {
	cloned := cloneMap(rawSchema)
	return transformDraft7Schema(cloned, true), nil
}

func transformDraft7Schema(s map[string]any, isRoot bool) map[string]any {
	if s == nil {
		return nil
	}

	if isRoot {
		if s["$schema"] == nil || s["$schema"] == "" {
			s["$schema"] = "http://json-schema.org/draft-07/schema#"
		}
		if s["type"] == nil || s["type"] == "" {
			s["type"] = "object"
		}
		if s["type"] == "object" && s["properties"] == nil {
			s["properties"] = map[string]any{}
		}
	}

	if props, ok := s["properties"].(map[string]any); ok {
		for k, v := range props {
			if pMap, isMap := v.(map[string]any); isMap {
				props[k] = transformDraft7Schema(pMap, false)
			}
		}
	}

	if itemsMap, ok := s["items"].(map[string]any); ok {
		s["items"] = transformDraft7Schema(itemsMap, false)
	} else if itemsList, ok := s["items"].([]any); ok {
		for i, item := range itemsList {
			if itemMap, isMap := item.(map[string]any); isMap {
				itemsList[i] = transformDraft7Schema(itemMap, false)
			}
		}
	}

	for _, k := range []string{"anyOf", "allOf", "oneOf"} {
		if list, ok := s[k].([]any); ok {
			for i, item := range list {
				if itemMap, isMap := item.(map[string]any); isMap {
					list[i] = transformDraft7Schema(itemMap, false)
				}
			}
		}
	}

	for _, k := range []string{"$defs", "definitions"} {
		if defs, ok := s[k].(map[string]any); ok {
			for dk, dv := range defs {
				if dMap, isMap := dv.(map[string]any); isMap {
					defs[dk] = transformDraft7Schema(dMap, false)
				}
			}
		}
	}

	return s
}
