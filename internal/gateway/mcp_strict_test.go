package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestOpenAIStrictSchema(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		validate func(t *testing.T, res json.RawMessage)
		wantErrs int
	}{
		{
			name: "objects with required and optional fields",
			input: `{
				"type": "object",
				"properties": {
					"id": {"type": "string", "description": "unique identifier"},
					"count": {"type": "integer"}
				},
				"required": ["id"]
			}`,
			validate: func(t *testing.T, res json.RawMessage) {
				var m map[string]any
				if err := json.Unmarshal(res, &m); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				if ap, ok := m["additionalProperties"].(bool); !ok || ap {
					t.Errorf("expected additionalProperties: false, got %v", m["additionalProperties"])
				}
				reqList, ok := m["required"].([]any)
				if !ok {
					t.Fatalf("expected required slice, got %T", m["required"])
				}
				reqSet := make(map[string]bool)
				for _, r := range reqList {
					reqSet[r.(string)] = true
				}
				if !reqSet["id"] || !reqSet["count"] {
					t.Errorf("expected both id and count in required, got %v", reqList)
				}

				props := m["properties"].(map[string]any)
				idProp := props["id"].(map[string]any)
				if idProp["type"] != "string" {
					t.Errorf("required id should retain type string, got %v", idProp["type"])
				}
				if _, hasAnyOf := idProp["anyOf"]; hasAnyOf {
					t.Errorf("required id should not have anyOf")
				}

				countProp := props["count"].(map[string]any)
				anyOfList, ok := countProp["anyOf"].([]any)
				if !ok || len(anyOfList) != 2 {
					t.Fatalf("optional count should have anyOf with 2 branches, got %v", countProp["anyOf"])
				}
				b0 := anyOfList[0].(map[string]any)
				b1 := anyOfList[1].(map[string]any)
				if b0["type"] != "integer" || b1["type"] != "null" {
					t.Errorf("expected integer and null branches in anyOf, got %v and %v", b0, b1)
				}
			},
		},
		{
			name: "optional fields converted to required with nullable anyOf",
			input: `{
				"type": "object",
				"properties": {
					"note": {"type": "string", "description": "some note"},
					"multi": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
				}
			}`,
			validate: func(t *testing.T, res json.RawMessage) {
				var m map[string]any
				if err := json.Unmarshal(res, &m); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				props := m["properties"].(map[string]any)

				noteProp := props["note"].(map[string]any)
				if noteProp["description"] != "some note" {
					t.Errorf("expected description preserved on outer property, got %v", noteProp["description"])
				}
				noteAnyOf, ok := noteProp["anyOf"].([]any)
				if !ok || len(noteAnyOf) != 2 {
					t.Fatalf("expected note anyOf with 2 branches, got %v", noteProp["anyOf"])
				}
				if noteAnyOf[0].(map[string]any)["type"] != "string" || noteAnyOf[1].(map[string]any)["type"] != "null" {
					t.Errorf("unexpected branches for note: %v", noteAnyOf)
				}

				multiProp := props["multi"].(map[string]any)
				multiAnyOf, ok := multiProp["anyOf"].([]any)
				if !ok || len(multiAnyOf) != 3 {
					t.Fatalf("expected multi anyOf with 3 branches, got %v", multiProp["anyOf"])
				}
				if multiAnyOf[2].(map[string]any)["type"] != "null" {
					t.Errorf("expected last branch of multi to be null, got %v", multiAnyOf[2])
				}
			},
		},
		{
			name: "nested objects and arrays of objects getting additionalProperties: false",
			input: `{
				"type": "object",
				"properties": {
					"nested": {
						"type": "object",
						"properties": {
							"inner_req": {"type": "string"},
							"inner_opt": {"type": "boolean"}
						},
						"required": ["inner_req"]
					},
					"items_list": {
						"type": "array",
						"items": {
							"type": "object",
							"properties": {
								"elem": {"type": "string"}
							}
						}
					}
				}
			}`,
			validate: func(t *testing.T, res json.RawMessage) {
				var m map[string]any
				if err := json.Unmarshal(res, &m); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				props := m["properties"].(map[string]any)

				nestedProp := props["nested"].(map[string]any)
				nestedAnyOf := nestedProp["anyOf"].([]any)
				nestedInner := nestedAnyOf[0].(map[string]any)
				if ap, ok := nestedInner["additionalProperties"].(bool); !ok || ap {
					t.Errorf("nested object missing additionalProperties: false, got %v", nestedInner["additionalProperties"])
				}

				nestedInnerProps := nestedInner["properties"].(map[string]any)
				if len(nestedInnerProps) != 2 {
					t.Errorf("expected 2 inner properties, got %d", len(nestedInnerProps))
				}

				itemsListProp := props["items_list"].(map[string]any)
				itemsAnyOf := itemsListProp["anyOf"].([]any)
				arrObj := itemsAnyOf[0].(map[string]any)
				itemSchema := arrObj["items"].(map[string]any)
				if ap, ok := itemSchema["additionalProperties"].(bool); !ok || ap {
					t.Errorf("array item object missing additionalProperties: false, got %v", itemSchema["additionalProperties"])
				}
			},
		},
		{
			name:  "empty object schema bare",
			input: `{}`,
			validate: func(t *testing.T, res json.RawMessage) {
				var m map[string]any
				if err := json.Unmarshal(res, &m); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				if m["type"] != "object" {
					t.Errorf("expected type object, got %v", m["type"])
				}
				if ap, ok := m["additionalProperties"].(bool); !ok || ap {
					t.Errorf("expected additionalProperties: false, got %v", m["additionalProperties"])
				}
				props, ok := m["properties"].(map[string]any)
				if !ok || len(props) != 0 {
					t.Errorf("expected empty properties map, got %v", m["properties"])
				}
			},
		},
		{
			name:  "empty object schema with type",
			input: `{"type": "object"}`,
			validate: func(t *testing.T, res json.RawMessage) {
				var m map[string]any
				if err := json.Unmarshal(res, &m); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				if ap, ok := m["additionalProperties"].(bool); !ok || ap {
					t.Errorf("expected additionalProperties: false, got %v", m["additionalProperties"])
				}
				props, ok := m["properties"].(map[string]any)
				if !ok || len(props) != 0 {
					t.Errorf("expected empty properties map, got %v", m["properties"])
				}
			},
		},
		{
			name:  "empty object schema with properties map",
			input: `{"type": "object", "properties": {}}`,
			validate: func(t *testing.T, res json.RawMessage) {
				var m map[string]any
				if err := json.Unmarshal(res, &m); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				if ap, ok := m["additionalProperties"].(bool); !ok || ap {
					t.Errorf("expected additionalProperties: false, got %v", m["additionalProperties"])
				}
			},
		},
		{
			name: "deeply nested $defs with patternProperties, minLength, and default",
			input: `{
				"type": "object",
				"properties": {
					"req": {"$ref": "#/$defs/Sub"}
				},
				"required": ["req"],
				"$defs": {
					"Sub": {
						"type": "object",
						"patternProperties": {"^x-": {"type": "string"}},
						"properties": {
							"val": {"type": "string", "minLength": 2, "default": "hi"}
						}
					}
				}
			}`,
			wantErrs: 0,
			validate: func(t *testing.T, res json.RawMessage) {
				var parsed any
				if err := json.Unmarshal(res, &parsed); err != nil {
					t.Fatalf("unmarshal error: %v", err)
				}
				assertNoForbiddenDraftKeywords(t, parsed, "$")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := ToOpenAIStrictSchema(json.RawMessage(tt.input))
			if err != nil {
				t.Fatalf("unexpected transformation error: %v", err)
			}
			errs := ValidateOpenAIStrictMode(res)
			if len(errs) != tt.wantErrs {
				t.Errorf("ValidateOpenAIStrictMode returned %d errors (want %d): %v\nschema: %s", len(errs), tt.wantErrs, errs, string(res))
			}
			if tt.validate != nil {
				tt.validate(t, res)
			}
		})
	}
}

func TestOpenAIStrictModeValidation(t *testing.T) {
	tests := []struct {
		name        string
		schema      string
		expectErrs  int
		errContains string
	}{
		{
			name:       "valid compliant schema",
			schema:     `{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}`,
			expectErrs: 0,
		},
		{
			name:        "root missing additionalProperties: false",
			schema:      `{"type": "object", "properties": {}}`,
			expectErrs:  1,
			errContains: "missing additionalProperties: false",
		},
		{
			name:        "root additionalProperties true",
			schema:      `{"type": "object", "properties": {}, "additionalProperties": true}`,
			expectErrs:  1,
			errContains: "additionalProperties must be false",
		},
		{
			name:        "property missing from required",
			schema:      `{"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": false}`,
			expectErrs:  1,
			errContains: `property "x" in properties is not listed in required`,
		},
		{
			name: "nested object missing additionalProperties",
			schema: `{
				"type": "object",
				"properties": {
					"child": {
						"type": "object",
						"properties": {}
					}
				},
				"required": ["child"],
				"additionalProperties": false
			}`,
			expectErrs:  1,
			errContains: "$.properties.child: missing additionalProperties: false",
		},
		{
			name:        "invalid JSON",
			schema:      `{invalid`,
			expectErrs:  1,
			errContains: "invalid json",
		},
		{
			name:        "empty JSON",
			schema:      ``,
			expectErrs:  1,
			errContains: "empty schema",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := ValidateOpenAIStrictMode(json.RawMessage(tt.schema))
			if len(errs) != tt.expectErrs {
				t.Fatalf("expected %d errors, got %d: %v", tt.expectErrs, len(errs), errs)
			}
			if tt.errContains != "" && len(errs) > 0 {
				found := false
				for _, e := range errs {
					if strings.Contains(e, tt.errContains) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected error containing %q, got %v", tt.errContains, errs)
				}
			}
		})
	}
}

func TestAllMCPToolsPassOpenAIStrictMode(t *testing.T) {
	tools := StrictToolDescriptors()
	if len(tools) == 0 {
		t.Fatal("expected non-empty tool descriptors from StrictToolDescriptors()")
	}

	expectedTools := []string{
		"fak_adjudicate",
		"fak_syscall",
		"fak_read",
		"fak_admit",
		"fak_changes",
		"fak_revoke",
		"fak_session_reset",
		"fak_context_change",
	}
	foundTools := make(map[string]bool)

	for _, tool := range tools {
		name, _ := tool["name"].(string)
		if name == "" {
			t.Errorf("tool descriptor missing name: %v", tool)
			continue
		}
		foundTools[name] = true

		strictVal, ok := tool["strict"].(bool)
		if !ok || !strictVal {
			t.Errorf("tool %q missing strict: true", name)
		}

		schemaRaw, ok := tool["inputSchema"].(json.RawMessage)
		if !ok {
			t.Fatalf("tool %q inputSchema is not json.RawMessage (got %T)", name, tool["inputSchema"])
		}

		errs := ValidateOpenAIStrictMode(schemaRaw)
		if len(errs) > 0 {
			t.Errorf("tool %q failed strict mode validation with %d error(s):\n%s\nschema:\n%s",
				name, len(errs), strings.Join(errs, "\n"), string(schemaRaw))
		}
	}

	for _, exp := range expectedTools {
		if !foundTools[exp] {
			t.Errorf("expected tool %q was not found in StrictToolDescriptors() output", exp)
		}
	}
}

func TestStrictSanitizerNestedDefs(t *testing.T) {
	input := `{
		"type": "object",
		"properties": {
			"filter": {
				"$ref": "#/$defs/FilterConfig"
			},
			"options": {
				"type": "object",
				"patternProperties": {
					"^opt_": {"type": "string"}
				},
				"properties": {
					"verbose": {
						"type": "boolean",
						"default": false
					}
				}
			}
		},
		"required": ["filter"],
		"$defs": {
			"FilterConfig": {
				"type": "object",
				"patternProperties": {
					"^[a-z]+$": {"type": "string"}
				},
				"properties": {
					"name": {
						"type": "string",
						"minLength": 1,
						"maxLength": 100,
						"default": "unnamed"
					},
					"rules": {
						"type": "array",
						"minItems": 1,
						"maxItems": 10,
						"uniqueItems": true,
						"items": {
							"type": "object",
							"patternProperties": {
								"^rule_": {"type": "string"}
							},
							"properties": {
								"pattern": {
									"type": "string",
									"minLength": 3,
									"default": ".*"
								},
								"action": {
									"type": "string",
									"default": "allow"
								}
							},
							"required": ["pattern"]
						}
					},
					"nestedDef": {
						"type": "object",
						"properties": {
							"inner": {
								"type": "string",
								"minLength": 5,
								"default": "val"
							}
						},
						"patternProperties": {
							"^meta_": {"type": "string"}
						},
						"dependentRequired": {
							"inner": ["name"]
						}
					}
				},
				"dependentRequired": {
					"rules": ["name"]
				},
				"required": ["name"]
			}
		}
	}`

	res, err := ToOpenAIStrictSchema(json.RawMessage(input))
	if err != nil {
		t.Fatalf("ToOpenAIStrictSchema failed: %v", err)
	}

	errs := ValidateOpenAIStrictMode(res)
	if len(errs) != 0 {
		t.Fatalf("ValidateOpenAIStrictMode returned errors (%d): %v\nschema: %s", len(errs), errs, string(res))
	}

	var parsed any
	if err := json.Unmarshal(res, &parsed); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	assertNoForbiddenDraftKeywords(t, parsed, "$")
}

func TestStrictSanitizerAllContainers(t *testing.T) {
	input := `{
		"type": "object",
		"properties": {
			"choice": {
				"anyOf": [
					{
						"type": "string",
						"minLength": 1,
						"maxLength": 10,
						"default": "abc"
					},
					{
						"type": "object",
						"patternProperties": {
							"^val_": {"type": "string"}
						},
						"properties": {
							"count": {
								"type": "integer",
								"default": 10
							}
						}
					}
				]
			},
			"combo": {
				"allOf": [
					{
						"type": "object",
						"patternProperties": {"^k_": {"type": "string"}},
						"properties": {
							"id": {"type": "string", "minLength": 2, "default": "id"}
						}
					}
				]
			},
			"single": {
				"oneOf": [
					{
						"type": "string",
						"minLength": 1,
						"default": "x"
					}
				]
			}
		},
		"required": ["choice"],
		"definitions": {
			"OldDef": {
				"type": "object",
				"patternProperties": {
					"^p_": {"type": "string"}
				},
				"properties": {
					"title": {
						"type": "string",
						"minLength": 4,
						"default": "defTitle"
					}
				},
				"dependentRequired": {
					"title": ["other"]
				}
			}
		}
	}`

	res, err := ToOpenAIStrictSchema(json.RawMessage(input))
	if err != nil {
		t.Fatalf("ToOpenAIStrictSchema failed: %v", err)
	}

	errs := ValidateOpenAIStrictMode(res)
	if len(errs) != 0 {
		t.Fatalf("ValidateOpenAIStrictMode returned errors (%d): %v\nschema: %s", len(errs), errs, string(res))
	}

	var parsed any
	if err := json.Unmarshal(res, &parsed); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	assertNoForbiddenDraftKeywords(t, parsed, "$")
}

func assertNoForbiddenDraftKeywords(t *testing.T, node any, path string) {
	forbidden := []string{
		"patternProperties",
		"minLength",
		"maxLength",
		"minItems",
		"maxItems",
		"uniqueItems",
		"default",
		"dependentRequired",
	}

	switch val := node.(type) {
	case map[string]any:
		for _, f := range forbidden {
			if _, exists := val[f]; exists {
				t.Errorf("found forbidden keyword %q at %s", f, path)
			}
		}
		for k, v := range val {
			assertNoForbiddenDraftKeywords(t, v, path+"."+k)
		}
	case []any:
		for i, item := range val {
			assertNoForbiddenDraftKeywords(t, item, fmt.Sprintf("%s[%d]", path, i))
		}
	}
}
