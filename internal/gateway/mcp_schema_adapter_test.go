package gateway

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// sampleCanonicalMCPSchema returns a rich canonical JSON Schema exercising all
// schema facets: required & optional fields, nested objects, arrays, union types,
// rich constraints, markdown descriptions, and formats.
func sampleCanonicalMCPSchema() map[string]any {
	return map[string]any{
		"$schema":     "http://json-schema.org/draft-07/schema#",
		"$id":         "https://example.com/user_profile.json",
		"title":       "user_profile_tool",
		"description": "## User Profile Tool\nRetrieves and updates user profile data.",
		"type":        "object",
		"properties": map[string]any{
			"userId": map[string]any{
				"type":        "string",
				"description": "Unique user identifier",
				"minLength":   3,
				"maxLength":   64,
				"pattern":     "^usr_[a-zA-Z0-9]+$",
			},
			"email": map[string]any{
				"type":        "string",
				"format":      "email",
				"description": "User email address",
			},
			"role": map[string]any{
				"type":        "string",
				"enum":        []any{"admin", "member", "guest"},
				"default":     "member",
				"description": "Access control role",
			},
			"metadata": map[string]any{
				"type":                 "object",
				"description":          "Custom user metadata",
				"additionalProperties": true,
				"properties": map[string]any{
					"tier": map[string]any{
						"type": "string",
					},
				},
			},
			"tags": map[string]any{
				"type":        "array",
				"description": "User classification tags",
				"minItems":    1,
				"items": map[string]any{
					"type": "string",
				},
			},
			"nickname": map[string]any{
				"description": "Optional user nickname",
				"anyOf": []any{
					map[string]any{"type": "string", "maxLength": 30},
					map[string]any{"type": "null"},
				},
			},
			"score": map[string]any{
				"type":        []any{"number", "null"},
				"description": "Reputation score",
			},
		},
		"required": []any{"userId"},
	}
}

// TestSchemaAdapter_ResolveProfile verifies model string resolution to profiles.
func TestSchemaAdapter_ResolveProfile(t *testing.T) {
	tests := []struct {
		model string
		want  SchemaAdaptationProfile
	}{
		// OpenAI / Astra / Structured Outputs
		{"gpt-4o", StrictOpenAIV2},
		{"gpt-4o-2024-08-06", StrictOpenAIV2},
		{"gpt-4o-mini", StrictOpenAIV2},
		{"gpt-4-turbo", StrictOpenAIV2},
		{"o1", StrictOpenAIV2},
		{"o1-preview", StrictOpenAIV2},
		{"o1-mini", StrictOpenAIV2},
		{"o3", StrictOpenAIV2},
		{"o3-mini", StrictOpenAIV2},
		{"o4-preview", StrictOpenAIV2},
		{"astra", StrictOpenAIV2},
		{"astra-pro-v1", StrictOpenAIV2},
		{"openai/gpt-4o", StrictOpenAIV2},
		{"chatgpt-4o-latest", StrictOpenAIV2},
		{"StrictOpenAIV2", StrictOpenAIV2},
		{"strict", StrictOpenAIV2},

		// Claude / Anthropic Thinking
		{"claude-3-5-sonnet-20241022", PermissiveClaudeThinking},
		{"claude-3-7-sonnet", PermissiveClaudeThinking},
		{"claude-3-opus-20240229", PermissiveClaudeThinking},
		{"claude-3-haiku-20240307", PermissiveClaudeThinking},
		{"anthropic.claude-v2", PermissiveClaudeThinking},
		{"PermissiveClaudeThinking", PermissiveClaudeThinking},
		{"claude", PermissiveClaudeThinking},

		// Gemini / Google Vertex
		{"gemini-1.5-pro", GeminiToolsDialect},
		{"gemini-1.5-flash", GeminiToolsDialect},
		{"gemini-2.0-flash", GeminiToolsDialect},
		{"gemini-2.5-flash", GeminiToolsDialect},
		{"vertex/gemini-1.5-pro", GeminiToolsDialect},
		{"GeminiToolsDialect", GeminiToolsDialect},
		{"gemini", GeminiToolsDialect},

		// Standard Draft 7 / Fallback
		{"meta-llama/Llama-3-70b-Instruct", StandardDraft7},
		{"deepseek-ai/DeepSeek-V3", StandardDraft7},
		{"mistralai/Mistral-Large-Instruct-2407", StandardDraft7},
		{"qwen-2.5-72b-instruct", StandardDraft7},
		{"unknown-custom-model", StandardDraft7},
		{"StandardDraft7", StandardDraft7},
		{"", StandardDraft7},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			got := ResolveSchemaProfile(tc.model)
			if got != tc.want {
				t.Errorf("ResolveSchemaProfile(%q) = %q, want %q", tc.model, got, tc.want)
			}
		})
	}
}

// TestSchemaAdapter_StrictOpenAIV2 verifies OpenAI strict mode transformations.
func TestSchemaAdapter_StrictOpenAIV2(t *testing.T) {
	orig := sampleCanonicalMCPSchema()
	adapted, err := AdaptMCPSchema(orig, StrictOpenAIV2)
	if err != nil {
		t.Fatalf("AdaptMCPSchema failed: %v", err)
	}

	// 1. Must serialize cleanly and pass ValidateOpenAIStrictMode
	rawJSON, err := json.Marshal(adapted)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	validationErrs := ValidateOpenAIStrictMode(rawJSON)
	if len(validationErrs) > 0 {
		t.Errorf("ValidateOpenAIStrictMode returned errors: %v", validationErrs)
	}

	// 2. Root and nested objects must have additionalProperties: false
	if ap, ok := adapted["additionalProperties"].(bool); !ok || ap {
		t.Errorf("expected root additionalProperties: false, got %v", adapted["additionalProperties"])
	}
	props := adapted["properties"].(map[string]any)
	metadataProp := props["metadata"].(map[string]any)
	// Metadata might be wrapped in nullable anyOf because it was optional
	if anyOfList, ok := metadataProp["anyOf"].([]any); ok {
		branch0 := anyOfList[0].(map[string]any)
		if ap, ok := branch0["additionalProperties"].(bool); !ok || ap {
			t.Errorf("expected nested metadata additionalProperties: false, got %v", branch0["additionalProperties"])
		}
	} else if ap, ok := metadataProp["additionalProperties"].(bool); !ok || ap {
		t.Errorf("expected nested metadata additionalProperties: false, got %v", metadataProp["additionalProperties"])
	}

	// 3. Every property must be in required
	reqList, ok := adapted["required"].([]any)
	if !ok {
		t.Fatalf("expected required to be []any, got %T", adapted["required"])
	}
	reqSet := make(map[string]bool)
	for _, r := range reqList {
		reqSet[r.(string)] = true
	}
	for propName := range props {
		if !reqSet[propName] {
			t.Errorf("property %q missing from required in StrictOpenAIV2", propName)
		}
	}

	// 4. Originally optional fields (email, role, tags) must be made nullable via anyOf
	emailProp := props["email"].(map[string]any)
	emailAnyOf, ok := emailProp["anyOf"].([]any)
	if !ok || len(emailAnyOf) != 2 {
		t.Fatalf("expected optional email to have anyOf with 2 branches, got %v", emailProp["anyOf"])
	}
	hasNull := false
	for _, branch := range emailAnyOf {
		bMap := branch.(map[string]any)
		if bMap["type"] == "null" {
			hasNull = true
		}
	}
	if !hasNull {
		t.Errorf("expected nullable union for optional email, got %v", emailAnyOf)
	}

	// 5. Stripped draft keywords: $schema, $id, minLength, maxLength, pattern, minItems, default
	if _, hasSchema := adapted["$schema"]; hasSchema {
		t.Errorf("expected $schema to be stripped in StrictOpenAIV2")
	}
	if _, hasID := adapted["$id"]; hasID {
		t.Errorf("expected $id to be stripped in StrictOpenAIV2")
	}
	userIdProp := props["userId"].(map[string]any)
	if _, hasML := userIdProp["minLength"]; hasML {
		t.Errorf("expected minLength stripped in StrictOpenAIV2")
	}
	if _, hasPattern := userIdProp["pattern"]; hasPattern {
		t.Errorf("expected pattern stripped in StrictOpenAIV2")
	}
}

// TestSchemaAdapter_PermissiveClaudeThinking verifies Claude permissive transformations.
func TestSchemaAdapter_PermissiveClaudeThinking(t *testing.T) {
	orig := sampleCanonicalMCPSchema()
	adapted, err := AdaptMCPSchema(orig, PermissiveClaudeThinking)
	if err != nil {
		t.Fatalf("AdaptMCPSchema failed: %v", err)
	}

	// 1. Preserves markdown description
	desc, ok := adapted["description"].(string)
	if !ok || !strings.Contains(desc, "## User Profile Tool") {
		t.Errorf("expected markdown description preserved, got %v", adapted["description"])
	}

	// 2. Permissive additionalProperties: metadata had additionalProperties: true, must remain true!
	props := adapted["properties"].(map[string]any)
	metadataProp := props["metadata"].(map[string]any)
	if ap, ok := metadataProp["additionalProperties"].(bool); !ok || !ap {
		t.Errorf("expected metadata additionalProperties: true preserved, got %v", metadataProp["additionalProperties"])
	}
	// Root additionalProperties was not set in orig, so must NOT be forced to false!
	if ap, hasAP := adapted["additionalProperties"]; hasAP && ap == false {
		t.Errorf("expected permissive root additionalProperties, got false")
	}

	// 3. Required should NOT force all optional properties!
	reqList, ok := adapted["required"].([]any)
	if !ok {
		// might be []string or []any
		if reqStrList, ok2 := adapted["required"].([]string); ok2 {
			reqList = make([]any, len(reqStrList))
			for i, s := range reqStrList {
				reqList[i] = s
			}
		} else {
			t.Fatalf("expected required list, got %T", adapted["required"])
		}
	}
	if len(reqList) != 1 || reqList[0] != "userId" {
		t.Errorf("expected required to contain only ['userId'], got %v", reqList)
	}

	// 4. Preserves rich constraints
	userIdProp := props["userId"].(map[string]any)
	if userIdProp["minLength"] != 3 || userIdProp["maxLength"] != 64 {
		t.Errorf("expected minLength and maxLength preserved on userId, got %v", userIdProp)
	}
	if userIdProp["pattern"] != "^usr_[a-zA-Z0-9]+$" {
		t.Errorf("expected pattern preserved on userId, got %v", userIdProp["pattern"])
	}

	roleProp := props["role"].(map[string]any)
	if roleProp["default"] != "member" {
		t.Errorf("expected default preserved on role, got %v", roleProp["default"])
	}

	tagsProp := props["tags"].(map[string]any)
	if tagsProp["minItems"] != 1 {
		t.Errorf("expected minItems preserved on tags, got %v", tagsProp["minItems"])
	}
}

// TestSchemaAdapter_GeminiToolsDialect verifies OpenAPI 3.0 / Gemini transformations.
func TestSchemaAdapter_GeminiToolsDialect(t *testing.T) {
	orig := sampleCanonicalMCPSchema()
	adapted, err := AdaptMCPSchema(orig, GeminiToolsDialect)
	if err != nil {
		t.Fatalf("AdaptMCPSchema failed: %v", err)
	}

	// 1. additionalProperties MUST BE STRIPPED everywhere (Gemini rejects additionalProperties)
	if _, hasAP := adapted["additionalProperties"]; hasAP {
		t.Errorf("Gemini schema must NOT contain root additionalProperties")
	}
	props := adapted["properties"].(map[string]any)
	metadataProp := props["metadata"].(map[string]any)
	if _, hasAP := metadataProp["additionalProperties"]; hasAP {
		t.Errorf("Gemini schema must NOT contain metadata additionalProperties")
	}

	// 2. $schema and $id must be stripped
	if _, hasSchema := adapted["$schema"]; hasSchema {
		t.Errorf("Gemini schema must NOT contain $schema")
	}
	if _, hasID := adapted["$id"]; hasID {
		t.Errorf("Gemini schema must NOT contain $id")
	}

	// 3. Union type: score had type: ["number", "null"].
	// Gemini OpenAPI 3.0 requires single string type and nullable: true boolean.
	scoreProp := props["score"].(map[string]any)
	if scoreType, ok := scoreProp["type"].(string); !ok || scoreType != "number" {
		t.Errorf("expected score type to be 'number', got %v", scoreProp["type"])
	}
	if nullable, ok := scoreProp["nullable"].(bool); !ok || !nullable {
		t.Errorf("expected score nullable: true, got %v", scoreProp["nullable"])
	}

	// 4. anyOf nullable: nickname had anyOf: [{"type": "string", "maxLength": 30}, {"type": "null"}].
	// Gemini OpenAPI 3.0 unwraps this to type: "string", maxLength: 30, nullable: true.
	nicknameProp := props["nickname"].(map[string]any)
	if _, hasAnyOf := nicknameProp["anyOf"]; hasAnyOf {
		t.Errorf("expected nickname anyOf nullable union unwrapped in GeminiToolsDialect, got %v", nicknameProp["anyOf"])
	}
	if nickType, ok := nicknameProp["type"].(string); !ok || nickType != "string" {
		t.Errorf("expected nickname type to be 'string', got %v", nicknameProp["type"])
	}
	if nullable, ok := nicknameProp["nullable"].(bool); !ok || !nullable {
		t.Errorf("expected nickname nullable: true, got %v", nicknameProp["nullable"])
	}
	if nicknameProp["maxLength"] != 30 {
		t.Errorf("expected nickname maxLength preserved, got %v", nicknameProp["maxLength"])
	}

	// 5. Preserves valid OpenAPI 3.0 keywords
	userIdProp := props["userId"].(map[string]any)
	if userIdProp["type"] != "string" || userIdProp["minLength"] != 3 {
		t.Errorf("userId constraints corrupted: %v", userIdProp)
	}

	// 6. Required only has userId
	reqList, _ := adapted["required"].([]any)
	if reqStrList, ok := adapted["required"].([]string); ok {
		reqList = make([]any, len(reqStrList))
		for i, s := range reqStrList {
			reqList[i] = s
		}
	}
	if len(reqList) != 1 || reqList[0] != "userId" {
		t.Errorf("expected required ['userId'], got %v", reqList)
	}
}

// TestSchemaAdapter_StandardDraft7 verifies standard Draft 7 passthrough.
func TestSchemaAdapter_StandardDraft7(t *testing.T) {
	orig := sampleCanonicalMCPSchema()
	adapted, err := AdaptMCPSchema(orig, StandardDraft7)
	if err != nil {
		t.Fatalf("AdaptMCPSchema failed: %v", err)
	}

	// 1. Root $schema is set to draft-07
	if schemaVal, ok := adapted["$schema"].(string); !ok || !strings.Contains(schemaVal, "draft-07") {
		t.Errorf("expected draft-07 $schema, got %v", adapted["$schema"])
	}

	// 2. Type is object
	if adapted["type"] != "object" {
		t.Errorf("expected type object, got %v", adapted["type"])
	}

	// 3. Properties and constraints preserved intact
	props := adapted["properties"].(map[string]any)
	metadataProp := props["metadata"].(map[string]any)
	if ap, ok := metadataProp["additionalProperties"].(bool); !ok || !ap {
		t.Errorf("expected additionalProperties: true preserved, got %v", metadataProp["additionalProperties"])
	}

	userIdProp := props["userId"].(map[string]any)
	if userIdProp["minLength"] != 3 || userIdProp["pattern"] != "^usr_[a-zA-Z0-9]+$" {
		t.Errorf("userId constraints corrupted: %v", userIdProp)
	}
}

// TestSchemaAdapter_DeepCloneImmutability verifies caller's map is never mutated.
func TestSchemaAdapter_DeepCloneImmutability(t *testing.T) {
	orig := sampleCanonicalMCPSchema()
	origJSONBefore, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	profiles := []SchemaAdaptationProfile{
		StrictOpenAIV2,
		PermissiveClaudeThinking,
		GeminiToolsDialect,
		StandardDraft7,
	}

	for _, p := range profiles {
		_, err := AdaptMCPSchema(orig, p)
		if err != nil {
			t.Fatalf("AdaptMCPSchema with profile %q failed: %v", p, err)
		}
		origJSONAfter, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		if string(origJSONBefore) != string(origJSONAfter) {
			t.Errorf("profile %q mutated original input schema map!", p)
		}
	}
}

// TestSchemaAdapter_NilAndInvalidInputs verifies fail-closed error handling.
func TestSchemaAdapter_NilAndInvalidInputs(t *testing.T) {
	_, err := AdaptMCPSchema(nil, StrictOpenAIV2)
	if err == nil {
		t.Errorf("expected error for nil schema, got nil")
	}

	validSchema := map[string]any{"type": "object"}
	_, err = AdaptMCPSchema(validSchema, "NonExistentProfile")
	if err == nil {
		t.Errorf("expected error for unknown profile, got nil")
	}
}

// TestSchemaAdapter_ConstToEnumInGemini verifies const: "val" converts to enum: ["val"].
func TestSchemaAdapter_ConstToEnumInGemini(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"version": map[string]any{
				"const": "v1.0",
			},
		},
	}
	adapted, err := AdaptMCPSchema(schema, GeminiToolsDialect)
	if err != nil {
		t.Fatalf("AdaptMCPSchema failed: %v", err)
	}
	props := adapted["properties"].(map[string]any)
	versionProp := props["version"].(map[string]any)
	if _, hasConst := versionProp["const"]; hasConst {
		t.Errorf("expected const stripped in GeminiToolsDialect")
	}
	enumVal, ok := versionProp["enum"].([]any)
	if !ok || len(enumVal) != 1 || enumVal[0] != "v1.0" {
		t.Errorf("expected enum ['v1.0'], got %v", versionProp["enum"])
	}
}

// TestSchemaAdapter_EmptySchemaEdgeCases verifies empty maps adapt safely without panics.
func TestSchemaAdapter_EmptySchemaEdgeCases(t *testing.T) {
	emptySchema := map[string]any{}
	profiles := []SchemaAdaptationProfile{
		StrictOpenAIV2,
		PermissiveClaudeThinking,
		GeminiToolsDialect,
		StandardDraft7,
	}

	for _, p := range profiles {
		res, err := AdaptMCPSchema(emptySchema, p)
		if err != nil {
			t.Errorf("profile %q failed on empty schema: %v", p, err)
		}
		if res == nil {
			t.Errorf("profile %q returned nil for empty schema", p)
		}
		if res["type"] != "object" {
			t.Errorf("profile %q did not set type: object on empty schema: %v", p, res)
		}
	}
}

// TestSchemaAdapter_TransformerInterface verifies that GetSchemaTransformer returns
// the transformer interface and executes properly.
func TestSchemaAdapter_TransformerInterface(t *testing.T) {
	transformer, err := GetSchemaTransformer(StrictOpenAIV2)
	if err != nil {
		t.Fatalf("GetSchemaTransformer failed: %v", err)
	}
	if transformer == nil {
		t.Fatalf("expected non-nil transformer")
	}

	input := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"param": map[string]any{"type": "string"},
		},
	}
	out, err := transformer.Transform(input)
	if err != nil {
		t.Fatalf("Transform error: %v", err)
	}
	if out["additionalProperties"] != false {
		t.Errorf("expected additionalProperties: false, got %v", out["additionalProperties"])
	}
}

// TestSchemaAdapter_DeepNestedObjectValidation tests deeply nested structures across profiles.
func TestSchemaAdapter_DeepNestedObjectValidation(t *testing.T) {
	nested := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"level1": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"level2": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"leaf": map[string]any{
								"type": "string",
							},
						},
					},
				},
			},
		},
	}

	for _, p := range []SchemaAdaptationProfile{StrictOpenAIV2, PermissiveClaudeThinking, GeminiToolsDialect, StandardDraft7} {
		adapted, err := AdaptMCPSchema(nested, p)
		if err != nil {
			t.Fatalf("profile %q failed on nested schema: %v", p, err)
		}
		if !reflect.DeepEqual(adapted["type"], "object") {
			t.Errorf("profile %q root type != object", p)
		}
	}
}
