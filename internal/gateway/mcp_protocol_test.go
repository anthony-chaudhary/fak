package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestMCPProtocolListIsSingleSourceOfTruth pins the invariant the centralization
// refactor establishes: defaultProtocol and supportedProtocols are BOTH derived
// from mcpProtocolVersions, so the list is the only thing an editor touches.
func TestMCPProtocolListIsSingleSourceOfTruth(t *testing.T) {
	if len(mcpProtocolVersions) == 0 {
		t.Fatal("mcpProtocolVersions must declare at least one revision")
	}
	// The default is the first declared revision.
	if defaultProtocol != mcpProtocolVersions[0] {
		t.Errorf("defaultProtocol = %q, want first list entry %q", defaultProtocol, mcpProtocolVersions[0])
	}
	// supportedProtocols is exactly the set of declared revisions — no more, no less.
	if len(supportedProtocols) != len(mcpProtocolVersions) {
		t.Errorf("supportedProtocols has %d entries, list declares %d", len(supportedProtocols), len(mcpProtocolVersions))
	}
	for _, v := range mcpProtocolVersions {
		if !supportedProtocols[v] {
			t.Errorf("declared revision %q missing from supportedProtocols", v)
		}
	}
	// The default must itself be supported (we never answer with a revision we'd reject).
	if !supportedProtocols[defaultProtocol] {
		t.Errorf("defaultProtocol %q is not in supportedProtocols", defaultProtocol)
	}
}

// TestMCPNegotiatorUsesDerivedList confirms the negotiator answers from the
// derived set: every declared revision is echoed back, an undeclared one falls
// back to the default.
func TestMCPNegotiatorUsesDerivedList(t *testing.T) {
	srv := newTestServer(t)
	for _, v := range mcpProtocolVersions {
		got := srv.initializeResult(jsonProto(v))["protocolVersion"]
		if got != v {
			t.Errorf("declared revision %q must be echoed, got %v", v, got)
		}
	}
	if got := srv.initializeResult(jsonProto("0000-00-00"))["protocolVersion"]; got != defaultProtocol {
		t.Errorf("undeclared revision must fall back to %q, got %v", defaultProtocol, got)
	}
}

func jsonProto(v string) []byte {
	return []byte(`{"protocolVersion":"` + v + `"}`)
}

// TestMCPToolSchemasConformToOpenAPIAndGemini witnesses that all registered MCP tools
// provide parameter schemas that satisfy OpenAPI 3.0 and Gemini API validation rules:
// - Parameters schema must have type "object".
// - Any "required" field must only appear on type "object".
// - Every entry in "required" must be defined in "properties" of that object.
// - No partial draft-07 "anyOf": [{"required": [...]}] constructs that Gemini rejects with:
//   "parameters.any_of[i].required: only allowed for OBJECT type" / "property is not defined".
func TestMCPToolSchemasConformToOpenAPIAndGemini(t *testing.T) {
	tools := toolDescriptors()
	if len(tools) == 0 {
		t.Fatal("no tools registered")
	}

	var validateSchema func(path string, v any) []string
	validateSchema = func(path string, v any) []string {
		var errs []string
		m, ok := v.(map[string]any)
		if !ok {
			return nil
		}
		stype, _ := m["type"].(string)

		if req, ok := m["required"].([]any); ok {
			if !strings.EqualFold(stype, "object") {
				errs = append(errs, path+".required: only allowed for OBJECT type (type is "+stype+")")
			}
			props, _ := m["properties"].(map[string]any)
			for _, r := range req {
				rStr, _ := r.(string)
				if props == nil || props[rStr] == nil {
					errs = append(errs, path+".required: property "+rStr+" is not defined in properties")
				}
			}
		}

		for _, key := range []string{"anyOf", "oneOf", "allOf"} {
			if alts, ok := m[key].([]any); ok {
				for i, alt := range alts {
					errs = append(errs, validateSchema(path+"."+key+"["+string(rune('0'+i))+"]", alt)...)
				}
			}
		}

		if props, ok := m["properties"].(map[string]any); ok {
			for k, child := range props {
				errs = append(errs, validateSchema(path+".properties."+k, child)...)
			}
		}
		if items, ok := m["items"]; ok {
			errs = append(errs, validateSchema(path+".items", items)...)
		}
		return errs
	}

	for _, tool := range tools {
		name, _ := tool["name"].(string)
		schemaRaw, ok := tool["inputSchema"].(json.RawMessage)
		if !ok {
			t.Errorf("tool %s inputSchema is not json.RawMessage", name)
			continue
		}
		var schemaObj map[string]any
		if err := json.Unmarshal(schemaRaw, &schemaObj); err != nil {
			t.Errorf("tool %s inputSchema is not valid JSON: %v", name, err)
			continue
		}
		errs := validateSchema(name, schemaObj)
		for _, e := range errs {
			t.Errorf("tool %s schema violation: %s", name, e)
		}
	}
}

func TestValidateToolDescriptorsPassesAtBoot(t *testing.T) {
	if err := validateToolDescriptors(); err != nil {
		t.Fatalf("validateToolDescriptors failed for registered tools: %v", err)
	}
}

func TestValidateOpenAPISchemaNodeCatchesBareRequiredAnyOf(t *testing.T) {
	broken := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tool": map[string]any{"type": "string"},
		},
		"anyOf": []any{
			map[string]any{"required": []any{"tool"}},
		},
	}
	err := validateOpenAPISchemaNode("broken", broken)
	if err == nil {
		t.Fatal("expected validateOpenAPISchemaNode to reject bare required anyOf, got nil")
	}
	if !strings.Contains(err.Error(), "only allowed for type object") {
		t.Fatalf("unexpected error message: %v", err)
	}
}
