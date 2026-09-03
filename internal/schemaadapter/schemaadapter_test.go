package schemaadapter

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestSpine drives the leaf surface.
func TestSpine(t *testing.T) {
	if !Ready() {
		t.Fatal("leaf spine did not reach Ready")
	}
}

func TestToGemini_UppercaseTypes(t *testing.T) {
	input := `{
		"type": "object",
		"properties": {
			"s": {"type": "string"},
			"n": {"type": "number"},
			"i": {"type": "integer"},
			"b": {"type": "boolean"},
			"a": {"type": "array", "items": {"type": "string"}},
			"sub": {
				"type": "object",
				"properties": {
					"child": {"type": "string"}
				}
			}
		}
	}`

	out, err := ToGemini(json.RawMessage(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if m["type"] != "OBJECT" {
		t.Errorf("expected root type OBJECT, got %v", m["type"])
	}

	props := m["properties"].(map[string]any)
	expectedTypes := map[string]string{
		"s": "STRING",
		"n": "NUMBER",
		"i": "INTEGER",
		"b": "BOOLEAN",
		"a": "ARRAY",
	}

	for k, want := range expectedTypes {
		propMap := props[k].(map[string]any)
		if propMap["type"] != want {
			t.Errorf("prop %q: expected type %s, got %v", k, want, propMap["type"])
		}
	}

	aMap := props["a"].(map[string]any)
	itemsMap := aMap["items"].(map[string]any)
	if itemsMap["type"] != "STRING" {
		t.Errorf("array items: expected type STRING, got %v", itemsMap["type"])
	}

	subMap := props["sub"].(map[string]any)
	if subMap["type"] != "OBJECT" {
		t.Errorf("sub object: expected type OBJECT, got %v", subMap["type"])
	}
	subProps := subMap["properties"].(map[string]any)
	childMap := subProps["child"].(map[string]any)
	if childMap["type"] != "STRING" {
		t.Errorf("child: expected type STRING, got %v", childMap["type"])
	}
}

func TestToGemini_Defect10769_BareRequiredAnyOf(t *testing.T) {
	// The #10769 defect: reintroducing "anyOf": [{"required": ["tool"]}, {"required": ["items"]}]
	// to an object without defining them in branch properties must be stripped for Gemini.
	input := `{
		"type": "object",
		"properties": {
			"tool": {"type": "string"},
			"items": {"type": "array", "items": {"type": "string"}}
		},
		"anyOf": [
			{"required": ["tool"]},
			{"required": ["items"]}
		]
	}`

	out, err := ToGemini(json.RawMessage(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if _, hasAnyOf := m["anyOf"]; hasAnyOf {
		t.Fatalf("expected anyOf with bare required branches to be stripped, but found: %v", m["anyOf"])
	}

	if m["type"] != "OBJECT" {
		t.Errorf("expected type OBJECT, got %v", m["type"])
	}

	props := m["properties"].(map[string]any)
	toolProp := props["tool"].(map[string]any)
	if toolProp["type"] != "STRING" {
		t.Errorf("expected tool type STRING, got %v", toolProp["type"])
	}

	// Mixed anyOf: one bare required branch and one valid branch
	mixedInput := `{
		"type": "object",
		"properties": {
			"tool": {"type": "string"}
		},
		"anyOf": [
			{"required": ["phantom"]},
			{
				"type": "object",
				"properties": {
					"extra": {"type": "string"}
				},
				"required": ["extra"]
			}
		]
	}`

	mixedOut, err := ToGemini(json.RawMessage(mixedInput))
	if err != nil {
		t.Fatalf("mixed unexpected error: %v", err)
	}

	var mixedMap map[string]any
	if err := json.Unmarshal(mixedOut, &mixedMap); err != nil {
		t.Fatalf("mixed unmarshal error: %v", err)
	}

	branches, ok := mixedMap["anyOf"].([]any)
	if !ok || len(branches) != 1 {
		t.Fatalf("expected anyOf with 1 valid branch remaining, got %v", mixedMap["anyOf"])
	}
	vBranch := branches[0].(map[string]any)
	if vBranch["type"] != "OBJECT" {
		t.Errorf("expected valid branch type OBJECT, got %v", vBranch["type"])
	}
	vProps := vBranch["properties"].(map[string]any)
	extraProp := vProps["extra"].(map[string]any)
	if extraProp["type"] != "STRING" {
		t.Errorf("expected extra type STRING, got %v", extraProp["type"])
	}
}

func TestToGemini_FilteringRequired(t *testing.T) {
	// 1. Partial required properties: only properties present in properties are kept
	input1 := `{
		"type": "object",
		"properties": {
			"present": {"type": "string"}
		},
		"required": ["present", "missing_field"]
	}`

	out1, err := ToGemini(json.RawMessage(input1))
	if err != nil {
		t.Fatalf("input1 error: %v", err)
	}
	var m1 map[string]any
	if err := json.Unmarshal(out1, &m1); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	req1, ok := m1["required"].([]any)
	if !ok || len(req1) != 1 || req1[0] != "present" {
		t.Fatalf("expected required to contain only ['present'], got %v", m1["required"])
	}

	// 2. All required properties are missing from properties -> required key is deleted
	input2 := `{
		"type": "object",
		"properties": {
			"present": {"type": "string"}
		},
		"required": ["missing_field_1", "missing_field_2"]
	}`

	out2, err := ToGemini(json.RawMessage(input2))
	if err != nil {
		t.Fatalf("input2 error: %v", err)
	}
	var m2 map[string]any
	if err := json.Unmarshal(out2, &m2); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if _, hasReq := m2["required"]; hasReq {
		t.Fatalf("expected required key to be deleted when empty, got %v", m2["required"])
	}

	// 3. Schema type is not OBJECT -> required is deleted
	input3 := `{
		"type": "string",
		"required": ["something"]
	}`

	out3, err := ToGemini(json.RawMessage(input3))
	if err != nil {
		t.Fatalf("input3 error: %v", err)
	}
	var m3 map[string]any
	if err := json.Unmarshal(out3, &m3); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if _, hasReq := m3["required"]; hasReq {
		t.Fatalf("expected required to be deleted for non-OBJECT type, got %v", m3["required"])
	}
}

func TestToGemini_PreservingNestedRequired(t *testing.T) {
	input := `{
		"type": "object",
		"properties": {
			"user": {
				"type": "object",
				"properties": {
					"id": {"type": "string"},
					"name": {"type": "string"}
				},
				"required": ["id"]
			}
		},
		"required": ["user"]
	}`

	out, err := ToGemini(json.RawMessage(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	rootReq, ok := m["required"].([]any)
	if !ok || len(rootReq) != 1 || rootReq[0] != "user" {
		t.Fatalf("expected root required ['user'], got %v", m["required"])
	}

	props := m["properties"].(map[string]any)
	userProp := props["user"].(map[string]any)
	if userProp["type"] != "OBJECT" {
		t.Errorf("expected user type OBJECT, got %v", userProp["type"])
	}

	userReq, ok := userProp["required"].([]any)
	if !ok || len(userReq) != 1 || userReq[0] != "id" {
		t.Fatalf("expected user required ['id'], got %v", userProp["required"])
	}
}

func TestToOpenAI_NonStrict(t *testing.T) {
	// Non-strict lowercases types and infers root object type if properties exist
	input := `{
		"properties": {
			"query": {"type": "STRING", "pattern": "^[a-z]+$", "minLength": 3}
		},
		"required": ["query"]
	}`

	out, err := ToOpenAI(json.RawMessage(input), false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if m["type"] != "object" {
		t.Errorf("expected root type inferred as 'object', got %v", m["type"])
	}

	props := m["properties"].(map[string]any)
	queryProp := props["query"].(map[string]any)
	if queryProp["type"] != "string" {
		t.Errorf("expected query type 'string', got %v", queryProp["type"])
	}

	// Constraints are preserved in non-strict mode
	if queryProp["pattern"] != "^[a-z]+$" {
		t.Errorf("expected pattern preserved in non-strict, got %v", queryProp["pattern"])
	}
	if queryProp["minLength"] != float64(3) {
		t.Errorf("expected minLength preserved in non-strict, got %v", queryProp["minLength"])
	}

	// additionalProperties: false is NOT injected in non-strict
	if _, hasAP := m["additionalProperties"]; hasAP {
		t.Errorf("did not expect additionalProperties injected in non-strict")
	}
}

func TestToOpenAI_Strict(t *testing.T) {
	input := `{
		"type": "object",
		"properties": {
			"req_field": {
				"type": "string",
				"description": "a required field",
				"pattern": "^[0-9]+$",
				"format": "email"
			},
			"opt_field": {
				"type": "integer",
				"description": "an optional field",
				"minimum": 10,
				"format": "uri"
			},
			"nested": {
				"type": "object",
				"properties": {
					"inner_req": {"type": "string"},
					"inner_opt": {"type": "boolean"}
				},
				"required": ["inner_req"]
			}
		},
		"required": ["req_field"]
	}`

	out, err := ToOpenAI(json.RawMessage(input), true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	// 1. Root additionalProperties: false
	if ap, ok := m["additionalProperties"].(bool); !ok || ap {
		t.Errorf("expected root additionalProperties: false, got %v", m["additionalProperties"])
	}

	// 2. All properties present in required
	rootReqList, ok := m["required"].([]any)
	if !ok {
		t.Fatalf("expected required list, got %v", m["required"])
	}
	reqSet := make(map[string]bool)
	for _, r := range rootReqList {
		reqSet[r.(string)] = true
	}
	for _, propName := range []string{"req_field", "opt_field", "nested"} {
		if !reqSet[propName] {
			t.Errorf("expected %q in root required list, got %v", propName, rootReqList)
		}
	}

	props := m["properties"].(map[string]any)

	// 3. req_field: already required, so NOT nullable, pattern stripped, email format retained
	reqField := props["req_field"].(map[string]any)
	if reqField["type"] != "string" {
		t.Errorf("expected req_field type string, got %v", reqField["type"])
	}
	if _, hasAnyOf := reqField["anyOf"]; hasAnyOf {
		t.Errorf("expected required field to not be wrapped in anyOf")
	}
	if _, hasPattern := reqField["pattern"]; hasPattern {
		t.Errorf("expected pattern removed in strict mode")
	}
	if reqField["format"] != "email" {
		t.Errorf("expected email format retained in strict mode, got %v", reqField["format"])
	}

	// 4. opt_field: was optional, so made nullable with anyOf, minimum removed, uri format removed
	optField := props["opt_field"].(map[string]any)
	anyOfBranches, ok := optField["anyOf"].([]any)
	if !ok || len(anyOfBranches) != 2 {
		t.Fatalf("expected opt_field anyOf with 2 branches, got %v", optField["anyOf"])
	}
	b0 := anyOfBranches[0].(map[string]any)
	b1 := anyOfBranches[1].(map[string]any)
	if b0["type"] != "integer" || b1["type"] != "null" {
		t.Errorf("expected integer and null branches, got %v and %v", b0, b1)
	}
	if _, hasMin := b0["minimum"]; hasMin {
		t.Errorf("expected minimum removed from opt_field in strict mode")
	}
	if _, hasFormat := b0["format"]; hasFormat {
		t.Errorf("expected uri format removed from opt_field in strict mode")
	}

	// 5. nested: was optional, so wrapped in anyOf null, and inner object has additionalProperties: false
	nestedProp := props["nested"].(map[string]any)
	nestedBranches, ok := nestedProp["anyOf"].([]any)
	if !ok || len(nestedBranches) != 2 {
		t.Fatalf("expected nested anyOf with 2 branches, got %v", nestedProp["anyOf"])
	}
	nestedObj := nestedBranches[0].(map[string]any)
	if ap, ok := nestedObj["additionalProperties"].(bool); !ok || ap {
		t.Errorf("expected nested object additionalProperties: false, got %v", nestedObj["additionalProperties"])
	}
	nestedReqList := nestedObj["required"].([]any)
	nestedReqSet := make(map[string]bool)
	for _, r := range nestedReqList {
		nestedReqSet[r.(string)] = true
	}
	if !nestedReqSet["inner_req"] || !nestedReqSet["inner_opt"] {
		t.Errorf("expected both inner_req and inner_opt in nested required, got %v", nestedReqList)
	}
}

func TestToAnthropic(t *testing.T) {
	input := `{
		"type": "OBJECT",
		"properties": {
			"query": {
				"type": "STRING",
				"pattern": "^[a-z]+$",
				"minLength": 2
			}
		},
		"required": ["query"]
	}`

	out, err := ToAnthropic(json.RawMessage(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if m["type"] != "object" {
		t.Errorf("expected root type 'object', got %v", m["type"])
	}

	props := m["properties"].(map[string]any)
	queryProp := props["query"].(map[string]any)
	if queryProp["type"] != "string" {
		t.Errorf("expected query type 'string', got %v", queryProp["type"])
	}

	// Standard draft-07 constraints are preserved
	if queryProp["pattern"] != "^[a-z]+$" {
		t.Errorf("expected pattern preserved, got %v", queryProp["pattern"])
	}
	if queryProp["minLength"] != float64(2) {
		t.Errorf("expected minLength preserved, got %v", queryProp["minLength"])
	}

	reqList := m["required"].([]any)
	if len(reqList) != 1 || reqList[0] != "query" {
		t.Errorf("expected required ['query'], got %v", m["required"])
	}
}

func TestNormalize(t *testing.T) {
	input := json.RawMessage(`{
		"type": "object",
		"properties": {
			"path": {"type": "string"}
		},
		"required": ["path"]
	}`)

	// DialectGemini
	geminiOut, err := Normalize(input, DialectGemini)
	if err != nil {
		t.Fatalf("gemini normalize error: %v", err)
	}
	var gMap map[string]any
	if err := json.Unmarshal(geminiOut, &gMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if gMap["type"] != "OBJECT" {
		t.Errorf("expected gemini type OBJECT, got %v", gMap["type"])
	}

	// DialectOpenAI
	openaiOut, err := Normalize(input, DialectOpenAI)
	if err != nil {
		t.Fatalf("openai normalize error: %v", err)
	}
	var oMap map[string]any
	if err := json.Unmarshal(openaiOut, &oMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if oMap["type"] != "object" {
		t.Errorf("expected openai type object, got %v", oMap["type"])
	}

	// DialectOpenAIStrict
	strictOut, err := Normalize(input, DialectOpenAIStrict)
	if err != nil {
		t.Fatalf("strict normalize error: %v", err)
	}
	var sMap map[string]any
	if err := json.Unmarshal(strictOut, &sMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if ap, ok := sMap["additionalProperties"].(bool); !ok || ap {
		t.Errorf("expected strict additionalProperties: false, got %v", sMap["additionalProperties"])
	}

	// DialectAnthropic
	anthropicOut, err := Normalize(input, DialectAnthropic)
	if err != nil {
		t.Fatalf("anthropic normalize error: %v", err)
	}
	var aMap map[string]any
	if err := json.Unmarshal(anthropicOut, &aMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if aMap["type"] != "object" {
		t.Errorf("expected anthropic type object, got %v", aMap["type"])
	}

	// Unknown dialect
	_, err = Normalize(input, Dialect("unknown_dialect"))
	if err == nil {
		t.Fatal("expected error for unknown dialect, got nil")
	}
	if !strings.Contains(err.Error(), "unknown dialect") {
		t.Errorf("expected error message to mention unknown dialect, got %v", err)
	}
}

func TestComplexRepresentativeSchemas(t *testing.T) {
	// 1. Batch read schema (fak_read style)
	fakReadSchema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"file_path": {"type": "string", "description": "path of the file to read"},
			"file_paths": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}, "description": "file paths to read"},
			"trace_id": {"type": "string", "description": "trace id"},
			"witness": {"type": "string", "description": "witness token"}
		}
	}`)

	// ToGemini on fak_read
	gRead, err := ToGemini(fakReadSchema)
	if err != nil {
		t.Fatalf("ToGemini(fakReadSchema) error: %v", err)
	}
	var gReadMap map[string]any
	if err := json.Unmarshal(gRead, &gReadMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if gReadMap["type"] != "OBJECT" {
		t.Errorf("fak_read gemini: expected type OBJECT, got %v", gReadMap["type"])
	}
	gReadProps := gReadMap["properties"].(map[string]any)
	gFilePaths := gReadProps["file_paths"].(map[string]any)
	if gFilePaths["type"] != "ARRAY" {
		t.Errorf("fak_read gemini file_paths: expected type ARRAY, got %v", gFilePaths["type"])
	}
	gItems := gFilePaths["items"].(map[string]any)
	if gItems["type"] != "STRING" {
		t.Errorf("fak_read gemini items: expected type STRING, got %v", gItems["type"])
	}

	// ToOpenAI strict on fak_read
	sRead, err := ToOpenAI(fakReadSchema, true)
	if err != nil {
		t.Fatalf("ToOpenAI(fakReadSchema, true) error: %v", err)
	}
	var sReadMap map[string]any
	if err := json.Unmarshal(sRead, &sReadMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if ap, ok := sReadMap["additionalProperties"].(bool); !ok || ap {
		t.Errorf("fak_read strict: expected additionalProperties: false, got %v", sReadMap["additionalProperties"])
	}
	sReadReq := sReadMap["required"].([]any)
	if len(sReadReq) != 4 {
		t.Errorf("fak_read strict: expected 4 required fields, got %d", len(sReadReq))
	}
	sReadProps := sReadMap["properties"].(map[string]any)
	sFilePaths := sReadProps["file_paths"].(map[string]any)
	// minItems stripped in strict mode
	if _, hasMinItems := sFilePaths["minItems"]; hasMinItems {
		t.Errorf("fak_read strict: expected minItems stripped")
	}

	// 2. Batch admit schema (fak_admit style: array of objects with nested required)
	fakAdmitSchema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"tool": {"type": "string"},
			"result": {"description": "result content"},
			"items": {
				"type": "array",
				"minItems": 1,
				"items": {
					"type": "object",
					"properties": {
						"tool": {"type": "string"},
						"result": {},
						"trace_id": {"type": "string"},
						"witness": {"type": "string"}
					},
					"required": ["tool"]
				}
			},
			"trace_id": {"type": "string"},
			"witness": {"type": "string"}
		}
	}`)

	// ToGemini on fak_admit
	gAdmit, err := ToGemini(fakAdmitSchema)
	if err != nil {
		t.Fatalf("ToGemini(fakAdmitSchema) error: %v", err)
	}
	var gAdmitMap map[string]any
	if err := json.Unmarshal(gAdmit, &gAdmitMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	gAdmitProps := gAdmitMap["properties"].(map[string]any)
	gItemsObj := gAdmitProps["items"].(map[string]any)["items"].(map[string]any)
	if gItemsObj["type"] != "OBJECT" {
		t.Errorf("fak_admit gemini items.items: expected type OBJECT, got %v", gItemsObj["type"])
	}
	gItemsReq := gItemsObj["required"].([]any)
	if len(gItemsReq) != 1 || gItemsReq[0] != "tool" {
		t.Errorf("fak_admit gemini items.items: expected required ['tool'], got %v", gItemsReq)
	}

	// ToOpenAI strict on fak_admit
	sAdmit, err := ToOpenAI(fakAdmitSchema, true)
	if err != nil {
		t.Fatalf("ToOpenAI(fakAdmitSchema, true) error: %v", err)
	}
	var sAdmitMap map[string]any
	if err := json.Unmarshal(sAdmit, &sAdmitMap); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	sAdmitProps := sAdmitMap["properties"].(map[string]any)
	sItemsProp := sAdmitProps["items"].(map[string]any)
	sItemsAnyOf := sItemsProp["anyOf"].([]any)
	sItemsArr := sItemsAnyOf[0].(map[string]any)
	sItemSchema := sItemsArr["items"].(map[string]any)
	if ap, ok := sItemSchema["additionalProperties"].(bool); !ok || ap {
		t.Errorf("fak_admit strict items.items: expected additionalProperties: false, got %v", sItemSchema["additionalProperties"])
	}
	sItemReq := sItemSchema["required"].([]any)
	if len(sItemReq) != 4 {
		t.Errorf("fak_admit strict items.items: expected 4 required properties, got %d", len(sItemReq))
	}
}

func TestEdgeCasesAndErrors(t *testing.T) {
	// Empty input
	for _, fn := range []func(json.RawMessage) (json.RawMessage, error){
		ToGemini,
		func(raw json.RawMessage) (json.RawMessage, error) { return ToOpenAI(raw, false) },
		func(raw json.RawMessage) (json.RawMessage, error) { return ToOpenAI(raw, true) },
		ToAnthropic,
	} {
		if _, err := fn(nil); err == nil {
			t.Errorf("expected error on nil input, got nil")
		}
		if _, err := fn(json.RawMessage("   ")); err == nil {
			t.Errorf("expected error on whitespace input, got nil")
		}
		if _, err := fn(json.RawMessage("{invalid-json")); err == nil {
			t.Errorf("expected error on invalid json, got nil")
		}
		if _, err := fn(json.RawMessage("12345")); err == nil {
			t.Errorf("expected error on non-object json, got nil")
		}
	}
}
