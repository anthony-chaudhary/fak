package gateway

import (
	"context"
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
//   - Parameters schema must have type "object".
//   - Any "required" field must only appear on type "object".
//   - Every entry in "required" must be defined in "properties" of that object.
//   - No partial draft-07 "anyOf": [{"required": [...]}] constructs that Gemini rejects with:
//     "parameters.any_of[i].required: only allowed for OBJECT type" / "property is not defined".
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

func TestMCPToolsListBootstrapCeilingAndSchemaSize(t *testing.T) {
	// With --defer-tools=true (bootstrap active / disableMCPDefer=false)
	srv := newTestServer(t)
	res, rerr := srv.handleMethod(context.Background(), "tools/list", nil)
	if rerr != nil {
		t.Fatalf("tools/list error: %v", rerr)
	}
	respMap, ok := res.(map[string]any)
	if !ok {
		t.Fatalf("tools/list response not a map: %T", res)
	}
	rawTools, ok := respMap["tools"].([]map[string]any)
	if !ok {
		t.Fatalf("tools not []map[string]any: %T", respMap["tools"])
	}

	// Assert exactly 4 tools: fak_adjudicate, fak_syscall, fak_read, fak_tools_search
	if len(rawTools) != 4 {
		t.Fatalf("expected 4 bootstrap tools, got %d", len(rawTools))
	}
	expectedTools := []string{"fak_adjudicate", "fak_syscall", "fak_read", "fak_tools_search"}
	for i, exp := range expectedTools {
		name, _ := rawTools[i]["name"].(string)
		if name != exp {
			t.Errorf("tool[%d] = %q, want %q", i, name, exp)
		}
	}

	// Assert total serialized descriptor schema size <= 4500 bytes (4.5 KB)
	serialized, err := json.Marshal(rawTools)
	if err != nil {
		t.Fatalf("json.Marshal(tools): %v", err)
	}
	if len(serialized) > 4500 {
		t.Fatalf("serialized schema size = %d bytes, want <= 4500 bytes", len(serialized))
	}

	// Also test that with mcpToolCeiling = 10 and disableMCPDefer = true, it clamps to at most 10 tools
	srvCeiling := newTestServerWithConfig(t, Config{
		EngineID:        "test",
		DisableMCPDefer: true,
		MCPToolCeiling:  10,
	})
	resCeiling, rerrCeiling := srvCeiling.handleMethod(context.Background(), "tools/list", nil)
	if rerrCeiling != nil {
		t.Fatalf("tools/list error with ceiling: %v", rerrCeiling)
	}
	respMapCeiling := resCeiling.(map[string]any)
	ceilingTools := respMapCeiling["tools"].([]map[string]any)
	if len(ceilingTools) > 10 {
		t.Fatalf("expected at most 10 tools with mcpToolCeiling=10, got %d", len(ceilingTools))
	}
	if len(ceilingTools) != 10 {
		t.Fatalf("expected exactly 10 tools clamped by ceiling, got %d", len(ceilingTools))
	}
	meta := respMapCeiling["_meta"].(map[string]any)
	filterStatus := meta["fak/tool_filter"].(MCPToolFilterStatus)
	if filterStatus.Mode != "ceiling" || filterStatus.Reason != "advertisement_ceiling" {
		t.Fatalf("filterStatus=%+v, want mode=ceiling, reason=advertisement_ceiling", filterStatus)
	}

	// Test that curatedCeilingToolDescriptors clamps a list of 45 tools to MaxMCPToolAdvertisementCeiling (40)
	var syntheticTools []map[string]any
	for i := 0; i < 45; i++ {
		syntheticTools = append(syntheticTools, map[string]any{
			"name":        "tool_" + string(rune('a'+i%26)) + "_" + string(rune('0'+i)),
			"description": "synthetic tool",
		})
	}
	capped := srvCeiling.curatedCeilingToolDescriptors(syntheticTools, MaxMCPToolAdvertisementCeiling)
	if len(capped) != MaxMCPToolAdvertisementCeiling {
		t.Fatalf("expected exactly %d capped tools, got %d", MaxMCPToolAdvertisementCeiling, len(capped))
	}
}

func TestMCPToolDescriptorsReadOnlyAnnotations(t *testing.T) {
	tools := toolDescriptors()
	byName := make(map[string]map[string]any)
	for _, td := range tools {
		if name, ok := td["name"].(string); ok {
			byName[name] = td
		}
	}

	readOnlyTools := []string{
		"fak_tools_search",
		"fak_read",
		"fak_adjudicate",
		"fak_changes",
		"fak_memory_drivers",
		"fak_memory_explain",
		"fak_feature_query",
		"fak_capabilities",
		"view_image",
		"fak_arch_check",
		"fak_trajquery",
		"fak_context_value",
		"fak_context_restore",
		"fak_context_spans",
		"fak_resume_history",
	}
	for _, name := range readOnlyTools {
		td, ok := byName[name]
		if !ok {
			t.Fatalf("tool %q not found in descriptors", name)
		}
		ann, ok := td["annotations"].(map[string]any)
		if !ok {
			t.Fatalf("tool %q missing annotations map", name)
		}
		if ann["readOnly"] != true {
			t.Errorf("tool %q annotations[readOnly] = %v, want true", name, ann["readOnly"])
		}
		if ann["idempotent"] != true {
			t.Errorf("tool %q annotations[idempotent] = %v, want true", name, ann["idempotent"])
		}
		if ann["consequential"] != false {
			t.Errorf("tool %q annotations[consequential] = %v, want false", name, ann["consequential"])
		}
		if ann["readOnlyHint"] != true {
			t.Errorf("tool %q annotations[readOnlyHint] = %v, want true", name, ann["readOnlyHint"])
		}
		if ann["read_only_hint"] != true {
			t.Errorf("tool %q annotations[read_only_hint] = %v, want true", name, ann["read_only_hint"])
		}
	}

	mutatingTools := []string{
		"fak_syscall",
		"fak_admit",
		"fak_revoke",
		"fak_session_reset",
		"fak_context_change",
		"fak_memory_run",
	}
	for _, name := range mutatingTools {
		td, ok := byName[name]
		if !ok {
			t.Fatalf("mutating tool %q not found in descriptors", name)
		}
		ann, ok := td["annotations"].(map[string]any)
		if !ok {
			t.Fatalf("mutating tool %q missing annotations map", name)
		}
		if ann["readOnly"] != false {
			t.Errorf("mutating tool %q annotations[readOnly] = %v, want false", name, ann["readOnly"])
		}
		if ann["consequential"] != true {
			t.Errorf("mutating tool %q annotations[consequential] = %v, want true", name, ann["consequential"])
		}
		if ann["readOnlyHint"] == true || ann["read_only_hint"] == true {
			t.Errorf("mutating tool %q must not have readOnlyHint or read_only_hint: %v", name, ann)
		}
	}
}

func TestMCPToolsListReturnsAnnotationsAndStrictCompliance(t *testing.T) {
	srv := newTestServer(t)
	ctx := context.Background()

	// 1. Standard tools/list: check annotations on returned tools
	res, rerr := srv.handleMethod(ctx, "tools/list", nil)
	if rerr != nil {
		t.Fatalf("tools/list error: %v", rerr)
	}
	respMap, ok := res.(map[string]any)
	if !ok {
		t.Fatalf("tools/list response not map: %T", res)
	}
	tools, ok := respMap["tools"].([]map[string]any)
	if !ok {
		t.Fatalf("tools not []map[string]any: %T", respMap["tools"])
	}

	for _, tool := range tools {
		name, _ := tool["name"].(string)
		ann, hasAnn := tool["annotations"].(map[string]any)
		if !hasAnn {
			t.Fatalf("tool %q missing annotations map", name)
		}
		switch name {
		case "fak_adjudicate", "fak_read", "fak_tools_search":
			if ann["readOnly"] != true {
				t.Errorf("tool %q readOnly = %v, want true", name, ann["readOnly"])
			}
			if ann["consequential"] != false {
				t.Errorf("tool %q consequential = %v, want false", name, ann["consequential"])
			}
			if ann["idempotent"] != true {
				t.Errorf("tool %q idempotent = %v, want true", name, ann["idempotent"])
			}
		case "fak_syscall":
			if ann["readOnly"] != false {
				t.Errorf("tool %q readOnly = %v, want false", name, ann["readOnly"])
			}
			if ann["consequential"] != true {
				t.Errorf("tool %q consequential = %v, want true", name, ann["consequential"])
			}
		}
	}

	// 2. Strict tools/list: request strict schemas via params {"strict": true}
	strictParams := json.RawMessage(`{"strict": true}`)
	resStrict, rerrStrict := srv.handleMethod(ctx, "tools/list", strictParams)
	if rerrStrict != nil {
		t.Fatalf("tools/list strict error: %v", rerrStrict)
	}
	respMapStrict := resStrict.(map[string]any)
	strictTools := respMapStrict["tools"].([]map[string]any)
	for _, tool := range strictTools {
		name, _ := tool["name"].(string)
		if strictVal, _ := tool["strict"].(bool); !strictVal {
			t.Errorf("tool %q missing strict: true in strict mode", name)
		}
		schemaRaw, ok := tool["inputSchema"].(json.RawMessage)
		if !ok {
			t.Fatalf("tool %q missing json.RawMessage inputSchema", name)
		}
		errs := ValidateOpenAIStrictMode(schemaRaw)
		if len(errs) > 0 {
			t.Errorf("tool %q strict mode validation failed: %v", name, errs)
		}
	}

	// 3. All registered descriptors pass strict mode via StrictToolDescriptors()
	allStrict := StrictToolDescriptors()
	if len(allStrict) == 0 {
		t.Fatal("StrictToolDescriptors returned 0 tools")
	}
	for _, tool := range allStrict {
		name, _ := tool["name"].(string)
		schemaRaw, ok := tool["inputSchema"].(json.RawMessage)
		if !ok {
			t.Fatalf("tool %q inputSchema is not json.RawMessage", name)
		}
		errs := ValidateOpenAIStrictMode(schemaRaw)
		if len(errs) > 0 {
			t.Errorf("StrictToolDescriptors tool %q failed ValidateOpenAIStrictMode: %v", name, errs)
		}
	}
}

// TestMCPProtocolGranularCapabilityAnnotations verifies that MCPToolDescriptor,
// ToolRegistration, and registered tools properly serialize and expose destructive_hint
// and open_world_hint.
func TestMCPProtocolGranularCapabilityAnnotations(t *testing.T) {
	// 1. Verify serialization of destructive and open-world hints
	desc := MCPToolDescriptor{
		Name:            "test_dangerous_tool",
		Description:     "A test tool that mutates disk and makes network calls",
		DestructiveHint: true,
		OpenWorldHint:   true,
	}
	data, err := json.Marshal(desc)
	if err != nil {
		t.Fatalf("json.Marshal(desc) failed: %v", err)
	}
	var unmarshaled map[string]any
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("json.Unmarshal(data) failed: %v", err)
	}
	if dest, ok := unmarshaled["destructive_hint"].(bool); !ok || !dest {
		t.Errorf("destructive_hint not serialized as true: %v", unmarshaled["destructive_hint"])
	}
	if ow, ok := unmarshaled["open_world_hint"].(bool); !ok || !ow {
		t.Errorf("open_world_hint not serialized as true: %v", unmarshaled["open_world_hint"])
	}

	// 2. Verify omitempty behavior when hints are false
	safeDesc := MCPToolDescriptor{
		Name:            "test_safe_tool",
		Description:     "A read-only tool",
		DestructiveHint: false,
		OpenWorldHint:   false,
	}
	safeData, err := json.Marshal(safeDesc)
	if err != nil {
		t.Fatalf("json.Marshal(safeDesc) failed: %v", err)
	}
	var safeUnmarshaled map[string]any
	if err := json.Unmarshal(safeData, &safeUnmarshaled); err != nil {
		t.Fatalf("json.Unmarshal(safeData) failed: %v", err)
	}
	if _, has := safeUnmarshaled["destructive_hint"]; has {
		t.Errorf("destructive_hint should be omitted when false: %v", safeUnmarshaled)
	}
	if _, has := safeUnmarshaled["open_world_hint"]; has {
		t.Errorf("open_world_hint should be omitted when false: %v", safeUnmarshaled)
	}

	// 3. Verify ToolRegistration schema descriptor
	reg := ToolRegistration{
		Name:            "test_reg_tool",
		DestructiveHint: true,
		OpenWorldHint:   false,
	}
	regData, err := json.Marshal(reg)
	if err != nil {
		t.Fatalf("json.Marshal(reg) failed: %v", err)
	}
	var regMap map[string]any
	if err := json.Unmarshal(regData, &regMap); err != nil {
		t.Fatalf("json.Unmarshal(regData) failed: %v", err)
	}
	if dest, ok := regMap["destructive_hint"].(bool); !ok || !dest {
		t.Errorf("ToolRegistration destructive_hint not true: %v", regMap)
	}
	if _, has := regMap["open_world_hint"]; has {
		t.Errorf("ToolRegistration open_world_hint should be omitted when false: %v", regMap)
	}
}

// TestCoretoolPaletteCapabilityAnnotations verifies that the core tools palette
// has mutating tools marked with DestructiveHint: true and network/arbitrary shell
// tools marked with OpenWorldHint: true.
func TestCoretoolPaletteCapabilityAnnotations(t *testing.T) {
	if len(CoreToolsPalette) == 0 {
		t.Fatal("CoreToolsPalette must not be empty")
	}

	paletteMap := make(map[string]MCPToolDescriptor)
	for _, tool := range CoreToolsPalette {
		paletteMap[tool.Name] = tool
	}

	// Mutating tools must have DestructiveHint: true
	mutatingTools := []string{
		"write",
		"edit",
		"bash",
		"exec",
		"fak_syscall",
		"fak_admit",
		"fak_revoke",
		"fak_context_change",
		"fak_session_reset",
		"fak_memory_run",
	}
	for _, name := range mutatingTools {
		tool, ok := paletteMap[name]
		if !ok {
			t.Errorf("expected tool %q in CoreToolsPalette", name)
			continue
		}
		if !tool.DestructiveHint {
			t.Errorf("mutating tool %q in CoreToolsPalette must have DestructiveHint: true", name)
		}
	}

	// Network / arbitrary shell execution tools must have OpenWorldHint: true
	openWorldTools := []string{
		"bash",
		"exec",
		"web_fetch",
		"fetch",
		"fak_syscall",
	}
	for _, name := range openWorldTools {
		tool, ok := paletteMap[name]
		if !ok {
			t.Errorf("expected tool %q in CoreToolsPalette", name)
			continue
		}
		if !tool.OpenWorldHint {
			t.Errorf("open-world tool %q in CoreToolsPalette must have OpenWorldHint: true", name)
		}
	}

	// Pure read-only tools must have both hints false
	readOnlyTools := []string{
		"read",
		"fak_read",
		"fak_adjudicate",
		"fak_tools_search",
	}
	for _, name := range readOnlyTools {
		tool, ok := paletteMap[name]
		if !ok {
			t.Errorf("expected tool %q in CoreToolsPalette", name)
			continue
		}
		if tool.DestructiveHint {
			t.Errorf("read-only tool %q must have DestructiveHint: false", name)
		}
		if tool.OpenWorldHint {
			t.Errorf("read-only tool %q must have OpenWorldHint: false", name)
		}
	}
}

// TestMCPRegisteredToolsExposeCapabilityHintsInToolList verifies that registered tools
// expose destructive_hint and open_world_hint in their schema descriptors and tool list representations.
func TestMCPRegisteredToolsExposeCapabilityHintsInToolList(t *testing.T) {
	toolList := RegisteredMCPToolListRepresentations()
	if len(toolList) == 0 {
		t.Fatal("RegisteredMCPToolListRepresentations returned empty list")
	}

	byName := make(map[string]map[string]any)
	for _, tool := range toolList {
		if name, ok := tool["name"].(string); ok {
			byName[name] = tool
		}
	}

	// bash: destructive and open-world
	bashTool, ok := byName["bash"]
	if !ok {
		t.Fatal("bash tool not found in registered tool list")
	}
	if dest, _ := bashTool["destructive_hint"].(bool); !dest {
		t.Errorf("bash tool list representation missing destructive_hint: true, got %v", bashTool["destructive_hint"])
	}
	if ow, _ := bashTool["open_world_hint"].(bool); !ow {
		t.Errorf("bash tool list representation missing open_world_hint: true, got %v", bashTool["open_world_hint"])
	}

	// web_fetch: open-world, not destructive
	fetchTool, ok := byName["web_fetch"]
	if !ok {
		t.Fatal("web_fetch tool not found in registered tool list")
	}
	if dest, _ := fetchTool["destructive_hint"].(bool); dest {
		t.Errorf("web_fetch tool list representation should not have destructive_hint: true")
	}
	if ow, _ := fetchTool["open_world_hint"].(bool); !ow {
		t.Errorf("web_fetch tool list representation missing open_world_hint: true, got %v", fetchTool["open_world_hint"])
	}

	// write: destructive, not open-world
	writeTool, ok := byName["write"]
	if !ok {
		t.Fatal("write tool not found in registered tool list")
	}
	if dest, _ := writeTool["destructive_hint"].(bool); !dest {
		t.Errorf("write tool list representation missing destructive_hint: true, got %v", writeTool["destructive_hint"])
	}
	if ow, _ := writeTool["open_world_hint"].(bool); ow {
		t.Errorf("write tool list representation should not have open_world_hint: true")
	}

	// read: neither destructive nor open-world
	readTool, ok := byName["read"]
	if !ok {
		t.Fatal("read tool not found in registered tool list")
	}
	if dest, _ := readTool["destructive_hint"].(bool); dest {
		t.Errorf("read tool list representation should not have destructive_hint: true")
	}
	if ow, _ := readTool["open_world_hint"].(bool); ow {
		t.Errorf("read tool list representation should not have open_world_hint: true")
	}
}
