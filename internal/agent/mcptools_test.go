package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

func TestMCPArmDisarmCatalog(t *testing.T) {
	DisarmMCPTools()
	if cat := MCPToolCatalog(); cat != nil {
		t.Fatalf("MCPToolCatalog() before arming = %v, want nil", cat)
	}
	if allowed := mcpToolAllow(); allowed != nil {
		t.Fatalf("mcpToolAllow() before arming = %v, want nil", allowed)
	}

	cat, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools() error = %v", err)
	}
	defer DisarmMCPTools()

	if len(cat) != 5 {
		t.Fatalf("ArmMCPTools() returned %d tools, want 5", len(cat))
	}

	expected := map[string]bool{
		"fak_read":         false,
		"fak_tools_search": false,
		"fak_adjudicate":   false,
		"fak_syscall":      false,
		"fak_capabilities": false,
	}
	for _, tool := range cat {
		if _, ok := expected[tool.Function.Name]; ok {
			expected[tool.Function.Name] = true
		}
	}
	for name, found := range expected {
		if !found {
			t.Errorf("tool %q not found in returned catalog", name)
		}
	}

	armedCat := MCPToolCatalog()
	if len(armedCat) != 5 {
		t.Fatalf("MCPToolCatalog() when armed returned %d tools, want 5", len(armedCat))
	}

	allowed := mcpToolAllow()
	if len(allowed) == 0 {
		t.Fatalf("mcpToolAllow() when armed returned empty list")
	}

	DisarmMCPTools()
	if catAfter := MCPToolCatalog(); catAfter != nil {
		t.Fatalf("MCPToolCatalog() after disarm = %v, want nil", catAfter)
	}
	if allowedAfter := mcpToolAllow(); allowedAfter != nil {
		t.Fatalf("mcpToolAllow() after disarm = %v, want nil", allowedAfter)
	}
}

func TestMCPRead(t *testing.T) {
	tmpDir := t.TempDir()
	testFileName := "test_sample.txt"
	testContent := "hello from mcp test file"
	testFilePath := filepath.Join(tmpDir, testFileName)
	if err := os.WriteFile(testFilePath, []byte(testContent), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()
	RegisterReadEngine(tmpDir)
	t.Cleanup(func() { RegisterReadEngine("") })

	ctx := context.Background()

	for _, toolName := range []string{"fak_read", "mcp__fak__fak_read", "mcp__fak_guard__fak_read"} {
		argsJSON, _ := json.Marshal(map[string]any{"file_path": testFileName})
		c := &abi.ToolCall{
			Tool: toolName,
			Args: putBytes(ctx, argsJSON),
		}

		v := (mcpToolGate{}).Adjudicate(ctx, c)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("%s Adjudicate: got kind %v, want VerdictAllow", toolName, v.Kind)
		}
		if v.By != "mcp" {
			t.Fatalf("%s Adjudicate: got By %q, want 'mcp'", toolName, v.By)
		}
		if c.Engine != FakReadEngineID {
			t.Fatalf("%s Adjudicate: got Engine %q, want %q", toolName, c.Engine, FakReadEngineID)
		}

		eng := abi.Engine(c.Engine)
		if eng == nil {
			t.Fatalf("abi.Engine(%q) is nil", c.Engine)
		}

		res, err := eng.Complete(ctx, c)
		if err != nil {
			t.Fatalf("%s Complete error = %v", toolName, err)
		}
		if res == nil || res.Status != abi.StatusOK {
			t.Fatalf("%s Complete returned non-OK status: %+v", toolName, res)
		}

		payloadBytes := refutil.Bytes(ctx, res.Payload)
		var body map[string]any
		if err := json.Unmarshal(payloadBytes, &body); err != nil {
			t.Fatalf("%s unmarshal payload error: %v, raw: %s", toolName, err, string(payloadBytes))
		}
		if body["content"] != testContent {
			t.Fatalf("%s payload content = %v, want %q", toolName, body["content"], testContent)
		}
	}
}

func TestMCPToolsSearch(t *testing.T) {
	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	ctx := context.Background()

	// 1. Search for "read" with default detail_level
	argsJSON, _ := json.Marshal(map[string]any{
		"query": "read",
	})
	c := &abi.ToolCall{
		Tool: "fak_tools_search",
		Args: putBytes(ctx, argsJSON),
	}

	v := (mcpToolGate{}).Adjudicate(ctx, c)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("fak_tools_search Adjudicate: got kind %v, want VerdictAllow", v.Kind)
	}
	if c.Engine != "inprocess_mcp" {
		t.Fatalf("fak_tools_search Adjudicate: got Engine %q, want 'inprocess_mcp'", c.Engine)
	}

	eng := abi.Engine(c.Engine)
	if eng == nil {
		t.Fatalf("abi.Engine(%q) is nil", c.Engine)
	}

	res, err := eng.Complete(ctx, c)
	if err != nil {
		t.Fatalf("Complete error = %v", err)
	}
	payloadBytes := refutil.Bytes(ctx, res.Payload)
	var resp struct {
		Tools []map[string]any `json:"tools"`
	}
	if err := json.Unmarshal(payloadBytes, &resp); err != nil {
		t.Fatalf("Unmarshal error = %v, payload = %s", err, string(payloadBytes))
	}

	foundFakRead := false
	for _, tool := range resp.Tools {
		if tool["name"] == "fak_read" {
			foundFakRead = true
			if tool["description"] == nil || tool["description"] == "" {
				t.Fatalf("fak_read missing description in search results: %+v", tool)
			}
		}
	}
	if !foundFakRead {
		t.Fatalf("expected fak_read to be found in query 'read', got: %+v", resp.Tools)
	}

	// 2. Search with detail_level = "name"
	argsNameJSON, _ := json.Marshal(map[string]any{
		"query":        "capabilities",
		"detail_level": "name",
	})
	cName := &abi.ToolCall{
		Tool: "fak_tools_search",
		Args: putBytes(ctx, argsNameJSON),
	}
	resName, err := eng.Complete(ctx, cName)
	if err != nil {
		t.Fatalf("Complete error = %v", err)
	}
	var respName struct {
		Tools []map[string]any `json:"tools"`
	}
	_ = json.Unmarshal(refutil.Bytes(ctx, resName.Payload), &respName)
	if len(respName.Tools) == 0 {
		t.Fatalf("expected tools for query 'capabilities'")
	}
	for _, tool := range respName.Tools {
		if tool["name"] == "fak_capabilities" {
			if _, hasDesc := tool["description"]; hasDesc {
				t.Fatalf("detail_level 'name' should not include description: %+v", tool)
			}
		}
	}

	// 3. Search with detail_level = "full"
	argsFullJSON, _ := json.Marshal(map[string]any{
		"query":        "adjudicate",
		"detail_level": "full",
	})
	cFull := &abi.ToolCall{
		Tool: "fak_tools_search",
		Args: putBytes(ctx, argsFullJSON),
	}
	resFull, err := eng.Complete(ctx, cFull)
	if err != nil {
		t.Fatalf("Complete error = %v", err)
	}
	var respFull struct {
		Tools []map[string]any `json:"tools"`
	}
	_ = json.Unmarshal(refutil.Bytes(ctx, resFull.Payload), &respFull)
	if len(respFull.Tools) == 0 {
		t.Fatalf("expected tools for query 'adjudicate'")
	}
	for _, tool := range respFull.Tools {
		if tool["name"] == "fak_adjudicate" {
			if tool["inputSchema"] == nil && tool["parameters"] == nil {
				t.Fatalf("detail_level 'full' missing schema: %+v", tool)
			}
		}
	}
}

func TestMCPAdjudicate(t *testing.T) {
	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	callAdjudicate := func(toolName string, args any) (verdict string, allowed bool, reason string) {
		argsJSON, _ := json.Marshal(map[string]any{
			"tool":      toolName,
			"arguments": args,
		})
		c := &abi.ToolCall{
			Tool: "fak_adjudicate",
			Args: putBytes(ctx, argsJSON),
		}
		res, err := eng.Complete(ctx, c)
		if err != nil {
			t.Fatalf("Complete error for %s: %v", toolName, err)
		}
		var resp struct {
			Verdict string `json:"verdict"`
			Allowed bool   `json:"allowed"`
			Reason  string `json:"reason"`
			By      string `json:"by"`
		}
		if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
			t.Fatalf("Unmarshal error for %s: %v", toolName, err)
		}
		return resp.Verdict, resp.Allowed, resp.Reason
	}

	// 1. Benign tool: get_user_details -> ALLOW
	verdict, allowed, _ := callAdjudicate("get_user_details", map[string]any{"user_id": "u123"})
	if !allowed || !strings.EqualFold(verdict, "allow") {
		t.Fatalf("get_user_details: got allowed=%v, verdict=%q, want allow", allowed, verdict)
	}

	// 2. Dangerous tool: delete_account -> DENY
	verdict, allowed, reason := callAdjudicate("delete_account", map[string]any{"user_id": "mia"})
	if allowed || !strings.EqualFold(verdict, "deny") {
		t.Fatalf("delete_account: got allowed=%v, verdict=%q, want deny", allowed, verdict)
	}
	if reason != "POLICY_BLOCK" {
		t.Fatalf("delete_account: got reason=%q, want POLICY_BLOCK", reason)
	}

	// 3. Dangerous tool: rm -rf / -> DENY
	verdict, allowed, reason = callAdjudicate("rm -rf /", nil)
	if allowed || !strings.EqualFold(verdict, "deny") {
		t.Fatalf("rm -rf /: got allowed=%v, verdict=%q, want deny", allowed, verdict)
	}
	if reason != "POLICY_BLOCK" {
		t.Fatalf("rm -rf /: got reason=%q, want POLICY_BLOCK", reason)
	}

	// 4. Dangerous tool: Bash with command "rm -rf /" -> DENY
	verdict, allowed, reason = callAdjudicate("Bash", map[string]any{"command": "rm -rf /"})
	if allowed || !strings.EqualFold(verdict, "deny") {
		t.Fatalf("Bash rm -rf: got allowed=%v, verdict=%q, want deny", allowed, verdict)
	}
	if reason != "POLICY_BLOCK" {
		t.Fatalf("Bash rm -rf: got reason=%q, want POLICY_BLOCK", reason)
	}
}

func TestMCPCapabilities(t *testing.T) {
	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	c := &abi.ToolCall{
		Tool: "fak_capabilities",
		Args: putBytes(ctx, []byte("{}")),
	}

	res, err := eng.Complete(ctx, c)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var resp struct {
		Status       string   `json:"status"`
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}

	if resp.Status != "ok" {
		t.Fatalf("Status = %q, want 'ok'", resp.Status)
	}

	expectedCaps := []string{
		"vdso_dedup",
		"context_mmu",
		"posture_default_open",
		"default_permissive_allow",
		"mcp_features",
	}
	capSet := make(map[string]bool)
	for _, cap := range resp.Capabilities {
		capSet[cap] = true
	}
	for _, want := range expectedCaps {
		if !capSet[want] {
			t.Errorf("capability %q not found in response: %v", want, resp.Capabilities)
		}
	}
}

func TestMCPSyscall(t *testing.T) {
	Configure()
	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	// 1. Benign tool: calculate
	argsCalc, _ := json.Marshal(map[string]any{
		"tool": "calculate",
		"arguments": map[string]any{
			"a": 15,
			"b": 25,
		},
	})
	cCalc := &abi.ToolCall{
		Tool: "fak_syscall",
		Args: putBytes(ctx, argsCalc),
	}
	resCalc, err := eng.Complete(ctx, cCalc)
	if err != nil {
		t.Fatalf("calculate syscall Complete error: %v", err)
	}
	var respCalc struct {
		Verdict string         `json:"verdict"`
		Result  map[string]any `json:"result"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, resCalc.Payload), &respCalc); err != nil {
		t.Fatalf("calculate Unmarshal error: %v", err)
	}
	if respCalc.Verdict != "allow" {
		t.Fatalf("calculate verdict = %q, want 'allow'", respCalc.Verdict)
	}
	sumVal, ok := respCalc.Result["sum"].(float64)
	if !ok || sumVal != 40 {
		t.Fatalf("calculate sum = %v, want 40", respCalc.Result["sum"])
	}

	// 2. Denied tool: delete_account
	argsDel, _ := json.Marshal(map[string]any{
		"tool": "delete_account",
		"arguments": map[string]any{
			"user_id": "mia",
		},
	})
	cDel := &abi.ToolCall{
		Tool: "fak_syscall",
		Args: putBytes(ctx, argsDel),
	}
	resDel, err := eng.Complete(ctx, cDel)
	if err != nil {
		t.Fatalf("delete_account syscall Complete error: %v", err)
	}
	var respDel struct {
		Verdict string         `json:"verdict"`
		Result  map[string]any `json:"result"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, resDel.Payload), &respDel); err != nil {
		t.Fatalf("delete_account Unmarshal error: %v", err)
	}
	if respDel.Verdict != "deny" {
		t.Fatalf("delete_account verdict = %q, want 'deny'", respDel.Verdict)
	}
}

func TestMCPGatePrefixes(t *testing.T) {
	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	ctx := context.Background()

	cases := []struct {
		tool       string
		wantEngine string
		wantKind   abi.VerdictKind
	}{
		{"fak_read", FakReadEngineID, abi.VerdictAllow},
		{"mcp__fak__fak_read", FakReadEngineID, abi.VerdictAllow},
		{"mcp__fak_guard__fak_read", FakReadEngineID, abi.VerdictAllow},
		{"fak_tools_search", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__fak_tools_search", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__fak_tools_search", "inprocess_mcp", abi.VerdictAllow},
		{"fak_adjudicate", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__fak_adjudicate", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__fak_adjudicate", "inprocess_mcp", abi.VerdictAllow},
		{"fak_syscall", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__fak_syscall", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__fak_syscall", "inprocess_mcp", abi.VerdictAllow},
		{"fak_capabilities", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__fak_capabilities", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__fak_capabilities", "inprocess_mcp", abi.VerdictAllow},
		{"unrelated_tool", "", abi.VerdictDefer},
	}

	gate := mcpToolGate{}
	for _, tc := range cases {
		c := &abi.ToolCall{Tool: tc.tool}
		v := gate.Adjudicate(ctx, c)
		if v.Kind != tc.wantKind {
			t.Errorf("Adjudicate(%q): got Kind=%v, want %v", tc.tool, v.Kind, tc.wantKind)
		}
		if tc.wantEngine != "" && c.Engine != tc.wantEngine {
			t.Errorf("Adjudicate(%q): got Engine=%q, want %q", tc.tool, c.Engine, tc.wantEngine)
		}
	}
}
