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

	if len(cat) != 9 {
		t.Fatalf("ArmMCPTools() returned %d tools, want 9", len(cat))
	}

	expected := map[string]bool{
		"fak_read":         false,
		"fak_tools_search": false,
		"fak_adjudicate":   false,
		"fak_syscall":      false,
		"fak_capabilities": false,
		"sandbox_exec":     false,
		"sandbox_read":     false,
		"sandbox_write":    false,
		"sandbox_reset":    false,
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
	if len(armedCat) != 9 {
		t.Fatalf("MCPToolCatalog() when armed returned %d tools, want 9", len(armedCat))
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
		{"sandbox_exec", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__sandbox_exec", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__sandbox_exec", "inprocess_mcp", abi.VerdictAllow},
		{"sandbox_read", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__sandbox_read", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__sandbox_read", "inprocess_mcp", abi.VerdictAllow},
		{"sandbox_write", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__sandbox_write", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__sandbox_write", "inprocess_mcp", abi.VerdictAllow},
		{"sandbox_reset", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak__sandbox_reset", "inprocess_mcp", abi.VerdictAllow},
		{"mcp__fak_guard__sandbox_reset", "inprocess_mcp", abi.VerdictAllow},
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

func TestMCPSandboxTools(t *testing.T) {
	tmpDir := t.TempDir()
	SetSandboxWorkspace(tmpDir)
	defer SetSandboxWorkspace("")

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

	// 1. sandbox_write
	writeArgs, _ := json.Marshal(map[string]any{
		"path":    "notes/sample.txt",
		"content": "line 1\nline 2\nline 3\nline 4\nline 5",
	})
	cWrite := &abi.ToolCall{
		Tool: "sandbox_write",
		Args: putBytes(ctx, writeArgs),
	}
	resWrite, err := eng.Complete(ctx, cWrite)
	if err != nil {
		t.Fatalf("sandbox_write Complete error: %v", err)
	}
	if resWrite == nil || resWrite.Status != abi.StatusOK {
		t.Fatalf("sandbox_write failed with non-OK status: %+v", resWrite)
	}

	// Verify file was written to disk inside workspace
	diskPath := filepath.Join(tmpDir, "notes", "sample.txt")
	diskData, err := os.ReadFile(diskPath)
	if err != nil {
		t.Fatalf("ReadFile from disk failed: %v", err)
	}
	if string(diskData) != "line 1\nline 2\nline 3\nline 4\nline 5" {
		t.Fatalf("file content on disk mismatch: %q", string(diskData))
	}

	// 2. sandbox_read full
	readArgs, _ := json.Marshal(map[string]any{
		"path": "notes/sample.txt",
	})
	cRead := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(ctx, readArgs),
	}
	resRead, err := eng.Complete(ctx, cRead)
	if err != nil {
		t.Fatalf("sandbox_read Complete error: %v", err)
	}
	var readBody map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, resRead.Payload), &readBody); err != nil {
		t.Fatalf("sandbox_read unmarshal payload error: %v", err)
	}
	if readBody["content"] != "line 1\nline 2\nline 3\nline 4\nline 5" {
		t.Fatalf("sandbox_read content = %v, want full text", readBody["content"])
	}

	// 3. sandbox_read with offset and limit (lines 2 to 3)
	readSliceArgs, _ := json.Marshal(map[string]any{
		"path":   "notes/sample.txt",
		"offset": 2,
		"limit":  2,
	})
	cReadSlice := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(ctx, readSliceArgs),
	}
	resReadSlice, err := eng.Complete(ctx, cReadSlice)
	if err != nil {
		t.Fatalf("sandbox_read slice Complete error: %v", err)
	}
	var readSliceBody map[string]any
	_ = json.Unmarshal(refutil.Bytes(ctx, resReadSlice.Payload), &readSliceBody)
	if readSliceBody["content"] != "line 2\nline 3" {
		t.Fatalf("sandbox_read slice content = %v, want 'line 2\\nline 3'", readSliceBody["content"])
	}

	// 4. sandbox_read traversal escape refusal
	escapeReadArgs, _ := json.Marshal(map[string]any{
		"path": "../../secret.txt",
	})
	cEscapeRead := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(ctx, escapeReadArgs),
	}
	resEscapeRead, _ := eng.Complete(ctx, cEscapeRead)
	if resEscapeRead == nil || resEscapeRead.Status != abi.StatusError {
		t.Fatalf("sandbox_read path escape should return StatusError: %+v", resEscapeRead)
	}

	// 5. sandbox_write traversal escape refusal
	escapeWriteArgs, _ := json.Marshal(map[string]any{
		"path":    "../escape.txt",
		"content": "illegal",
	})
	cEscapeWrite := &abi.ToolCall{
		Tool: "sandbox_write",
		Args: putBytes(ctx, escapeWriteArgs),
	}
	resEscapeWrite, _ := eng.Complete(ctx, cEscapeWrite)
	if resEscapeWrite == nil || resEscapeWrite.Status != abi.StatusError {
		t.Fatalf("sandbox_write path escape should return StatusError: %+v", resEscapeWrite)
	}

	// 6. sandbox_reset
	cReset := &abi.ToolCall{
		Tool: "sandbox_reset",
		Args: putBytes(ctx, []byte("{}")),
	}
	resReset, err := eng.Complete(ctx, cReset)
	if err != nil {
		t.Fatalf("sandbox_reset Complete error: %v", err)
	}
	if resReset == nil || resReset.Status != abi.StatusOK {
		t.Fatalf("sandbox_reset non-OK status: %+v", resReset)
	}

	// 7. sandbox_exec
	execArgs, _ := json.Marshal(map[string]any{
		"command": "echo sandbox_ok",
	})
	cExec := &abi.ToolCall{
		Tool: "sandbox_exec",
		Args: putBytes(ctx, execArgs),
	}
	resExec, err := eng.Complete(ctx, cExec)
	if err != nil {
		t.Fatalf("sandbox_exec Complete error: %v", err)
	}
	if resExec == nil || resExec.Status != abi.StatusOK {
		t.Fatalf("sandbox_exec non-OK status: %+v", resExec)
	}
	var execBody map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, resExec.Payload), &execBody); err != nil {
		t.Fatalf("sandbox_exec unmarshal error: %v", err)
	}
	stdoutStr, _ := execBody["stdout"].(string)
	if !strings.Contains(stdoutStr, "sandbox_ok") {
		t.Fatalf("sandbox_exec stdout = %q, want it to contain 'sandbox_ok'", stdoutStr)
	}
}

func TestSandboxSymlinkConfinement(t *testing.T) {
	temp := t.TempDir()
	wsDir := filepath.Join(temp, "ws")
	outsideDir := filepath.Join(temp, "outside")
	if err := os.Mkdir(wsDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outsideDir, 0755); err != nil {
		t.Fatal(err)
	}

	secretFile := filepath.Join(outsideDir, "secret.txt")
	if err := os.WriteFile(secretFile, []byte("super-secret-content"), 0644); err != nil {
		t.Fatal(err)
	}

	outsideFileSymlink := filepath.Join(wsDir, "link_outside_file.txt")
	if err := os.Symlink(secretFile, outsideFileSymlink); err != nil {
		t.Skipf("skipping symlink test: symlink creation not permitted: %v", err)
	}

	outsideDirSymlink := filepath.Join(wsDir, "link_outside_dir")
	if err := os.Symlink(outsideDir, outsideDirSymlink); err != nil {
		t.Skipf("skipping symlink test: symlink creation not permitted: %v", err)
	}

	SetSandboxWorkspace(wsDir)
	defer SetSandboxWorkspace("")

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

	// 1. sandbox_read through a symlink file pointing outside must fail
	readArgs, _ := json.Marshal(map[string]any{
		"path": "link_outside_file.txt",
	})
	cRead := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(ctx, readArgs),
	}
	resRead, _ := eng.Complete(ctx, cRead)
	if resRead == nil || resRead.Status != abi.StatusError {
		t.Fatalf("sandbox_read through symlink should return StatusError: %+v", resRead)
	}
	readPayload := string(refutil.Bytes(ctx, resRead.Payload))
	if !strings.Contains(readPayload, "escapes sandbox workspace via symlink") {
		t.Fatalf("sandbox_read error payload = %q, want confinement failure mentioning escapes via symlink", readPayload)
	}

	// Direct check on handleSandboxRead
	_, directReadErr := handleSandboxRead(ctx, map[string]any{"path": "link_outside_file.txt"})
	if directReadErr == nil || !strings.Contains(directReadErr.Error(), "escapes sandbox workspace via symlink") {
		t.Fatalf("handleSandboxRead direct error = %v, want 'escapes sandbox workspace via symlink'", directReadErr)
	}

	// 2. sandbox_read through a symlinked directory pointing outside must fail
	readDirArgs, _ := json.Marshal(map[string]any{
		"path": "link_outside_dir/secret.txt",
	})
	cReadDir := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(ctx, readDirArgs),
	}
	resReadDir, _ := eng.Complete(ctx, cReadDir)
	if resReadDir == nil || resReadDir.Status != abi.StatusError {
		t.Fatalf("sandbox_read through dir symlink should return StatusError: %+v", resReadDir)
	}
	readDirPayload := string(refutil.Bytes(ctx, resReadDir.Payload))
	if !strings.Contains(readDirPayload, "escapes sandbox workspace via symlink") {
		t.Fatalf("sandbox_read dir error payload = %q, want confinement failure", readDirPayload)
	}

	// 3. sandbox_write through a symlink file pointing outside must fail
	writeLinkArgs, _ := json.Marshal(map[string]any{
		"path":    "link_outside_file.txt",
		"content": "overwritten-secret",
	})
	cWriteLink := &abi.ToolCall{
		Tool: "sandbox_write",
		Args: putBytes(ctx, writeLinkArgs),
	}
	resWriteLink, _ := eng.Complete(ctx, cWriteLink)
	if resWriteLink == nil || resWriteLink.Status != abi.StatusError {
		t.Fatalf("sandbox_write through symlink file should return StatusError: %+v", resWriteLink)
	}
	writeLinkPayload := string(refutil.Bytes(ctx, resWriteLink.Payload))
	if !strings.Contains(writeLinkPayload, "escapes sandbox workspace via symlink") {
		t.Fatalf("sandbox_write error payload = %q, want confinement failure", writeLinkPayload)
	}

	// Verify outside file was NOT overwritten
	contentAfter, err := os.ReadFile(secretFile)
	if err != nil {
		t.Fatal(err)
	}
	if string(contentAfter) != "super-secret-content" {
		t.Fatalf("outside file content was modified: %q", string(contentAfter))
	}

	// 4. sandbox_write to a new file through a symlinked directory pointing outside must fail
	writeDirArgs, _ := json.Marshal(map[string]any{
		"path":    "link_outside_dir/escaped_write.txt",
		"content": "escaped-payload",
	})
	cWriteDir := &abi.ToolCall{
		Tool: "sandbox_write",
		Args: putBytes(ctx, writeDirArgs),
	}
	resWriteDir, _ := eng.Complete(ctx, cWriteDir)
	if resWriteDir == nil || resWriteDir.Status != abi.StatusError {
		t.Fatalf("sandbox_write through dir symlink should return StatusError: %+v", resWriteDir)
	}
	writeDirPayload := string(refutil.Bytes(ctx, resWriteDir.Payload))
	if !strings.Contains(writeDirPayload, "escapes sandbox workspace via symlink") {
		t.Fatalf("sandbox_write dir error payload = %q, want confinement failure", writeDirPayload)
	}

	// Direct check on handleSandboxWrite
	_, directWriteErr := handleSandboxWrite(ctx, map[string]any{
		"path":    "link_outside_dir/escaped_write.txt",
		"content": "escaped-payload",
	})
	if directWriteErr == nil || !strings.Contains(directWriteErr.Error(), "escapes sandbox workspace via symlink") {
		t.Fatalf("handleSandboxWrite direct error = %v, want 'escapes sandbox workspace via symlink'", directWriteErr)
	}

	// Verify the file was NOT created outside
	if _, err := os.Stat(filepath.Join(outsideDir, "escaped_write.txt")); err == nil {
		t.Fatalf("escaped_write.txt was created in outside directory!")
	}

	// 5. Legitimate symlink inside workspace should succeed
	insideFile := filepath.Join(wsDir, "inside.txt")
	if err := os.WriteFile(insideFile, []byte("valid-inside"), 0644); err != nil {
		t.Fatal(err)
	}
	insideLink := filepath.Join(wsDir, "link_inside.txt")
	if err := os.Symlink(insideFile, insideLink); err == nil {
		readInsideArgs, _ := json.Marshal(map[string]any{
			"path": "link_inside.txt",
		})
		cReadInside := &abi.ToolCall{
			Tool: "sandbox_read",
			Args: putBytes(ctx, readInsideArgs),
		}
		resReadInside, err := eng.Complete(ctx, cReadInside)
		if err != nil || resReadInside.Status != abi.StatusOK {
			t.Fatalf("sandbox_read through valid inside symlink failed: %+v", resReadInside)
		}
	}
}

func TestMCPTransform_AdjudicateReadNormalization(t *testing.T) {
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

	argsJSON, _ := json.Marshal(map[string]any{
		"tool": "Read",
		"arguments": map[string]any{
			"filePath": "internal/agent/mcptools.go",
		},
	})
	c := &abi.ToolCall{
		Tool: "fak_adjudicate",
		Args: putBytes(ctx, argsJSON),
	}
	res, err := eng.Complete(ctx, c)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var resp map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}

	if resp["verdict"] != "transform" {
		t.Errorf("verdict = %v, want 'transform'", resp["verdict"])
	}
	if resp["allowed"] != true {
		t.Errorf("allowed = %v, want true", resp["allowed"])
	}
	if resp["repaired_tool"] != "fak_read" {
		t.Errorf("repaired_tool = %v, want 'fak_read'", resp["repaired_tool"])
	}
	repairedArgs, ok := resp["repaired_arguments"].(map[string]any)
	if !ok {
		t.Fatalf("repaired_arguments type = %T, want map[string]any", resp["repaired_arguments"])
	}
	if repairedArgs["file_path"] != "internal/agent/mcptools.go" {
		t.Errorf("file_path in repaired_arguments = %v, want 'internal/agent/mcptools.go'", repairedArgs["file_path"])
	}
	if _, hasOld := repairedArgs["filePath"]; hasOld {
		t.Errorf("legacy filePath was not stripped: %v", repairedArgs)
	}
}

func TestMCPTransform_SyscallReadNormalization(t *testing.T) {
	tmpDir := t.TempDir()
	testFileName := "test_sample_transform.txt"
	testContent := "hello from mcp syscall transform"
	testFilePath := filepath.Join(tmpDir, testFileName)
	if err := os.WriteFile(testFilePath, []byte(testContent), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	RegisterReadEngine(tmpDir)
	t.Cleanup(func() { RegisterReadEngine("") })

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	argsJSON, _ := json.Marshal(map[string]any{
		"tool": "Read",
		"arguments": map[string]any{
			"filePath": testFileName,
		},
	})
	c := &abi.ToolCall{
		Tool: "fak_syscall",
		Args: putBytes(ctx, argsJSON),
	}
	res, err := eng.Complete(ctx, c)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var resp map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}

	if resp["verdict"] != "transform" {
		t.Errorf("verdict = %v, want 'transform'", resp["verdict"])
	}
	if resp["repaired_tool"] != "fak_read" {
		t.Errorf("repaired_tool = %v, want 'fak_read'", resp["repaired_tool"])
	}

	resultMap, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatalf("result type = %T, want map[string]any; resp = %+v", resp["result"], resp)
	}
	if resultMap["content"] != testContent {
		t.Errorf("result content = %v, want %q", resultMap["content"], testContent)
	}

	repairedArgs, ok := resp["repaired_arguments"].(map[string]any)
	if !ok {
		t.Fatalf("repaired_arguments type = %T, want map[string]any", resp["repaired_arguments"])
	}
	if repairedArgs["file_path"] != testFileName {
		t.Errorf("repaired_arguments file_path = %v, want %q", repairedArgs["file_path"], testFileName)
	}
}

type mcpRecordLocalEngine struct {
	lastArgs map[string]any
	rawArgs  []byte
}

func (r *mcpRecordLocalEngine) Caps() []abi.Capability { return nil }
func (r *mcpRecordLocalEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, m := decodeCallArgs(ctx, c.Args)
	r.rawArgs = body
	r.lastArgs = m
	out, isErr := execTool(c.Tool, m)
	return engineResult(ctx, c, body, out, isErr, "localtools"), nil
}

func TestMCPTransform_SecretRedaction(t *testing.T) {
	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	snap := adjudicator.Default.PolicySnapshot()
	defer adjudicator.Default.SetPolicy(snap)

	pol := snap
	pol.RedactFields = []string{"api_key"}
	adjudicator.Default.SetPolicy(pol)

	rec := &mcpRecordLocalEngine{}
	abi.RegisterEngine("localtools", rec)
	defer abi.RegisterEngine("localtools", localEngine{})

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	secretVal := "secret-api-key-12345"
	callArgs := map[string]any{
		"a":       12,
		"b":       18,
		"api_key": secretVal,
	}

	// 1. fak_adjudicate with secret redaction
	adjJSON, _ := json.Marshal(map[string]any{
		"tool":      "calculate",
		"arguments": callArgs,
	})
	cAdj := &abi.ToolCall{
		Tool: "fak_adjudicate",
		Args: putBytes(ctx, adjJSON),
	}
	resAdj, err := eng.Complete(ctx, cAdj)
	if err != nil {
		t.Fatalf("fak_adjudicate Complete error: %v", err)
	}
	var respAdj map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, resAdj.Payload), &respAdj); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if respAdj["verdict"] != "transform" {
		t.Errorf("adjudicate verdict = %v, want 'transform'", respAdj["verdict"])
	}
	if respAdj["allowed"] != true {
		t.Errorf("adjudicate allowed = %v, want true", respAdj["allowed"])
	}
	repairedAdjArgs, ok := respAdj["repaired_arguments"].(map[string]any)
	if !ok {
		t.Fatalf("repaired_arguments type = %T, want map[string]any", respAdj["repaired_arguments"])
	}
	if repairedAdjArgs["api_key"] != "[REDACTED]" {
		t.Errorf("repaired_arguments api_key = %v, want '[REDACTED]'", repairedAdjArgs["api_key"])
	}
	if _, hasTool := respAdj["repaired_tool"]; hasTool {
		t.Errorf("repaired_tool should not be present on arg-only transform: %v", respAdj["repaired_tool"])
	}

	// 2. fak_syscall with secret redaction
	sysJSON, _ := json.Marshal(map[string]any{
		"tool":      "calculate",
		"arguments": callArgs,
	})
	cSys := &abi.ToolCall{
		Tool: "fak_syscall",
		Args: putBytes(ctx, sysJSON),
	}
	resSys, err := eng.Complete(ctx, cSys)
	if err != nil {
		t.Fatalf("fak_syscall Complete error: %v", err)
	}
	var respSys map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, resSys.Payload), &respSys); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if respSys["verdict"] != "transform" {
		t.Errorf("syscall verdict = %v, want 'transform'", respSys["verdict"])
	}
	repairedSysArgs, ok := respSys["repaired_arguments"].(map[string]any)
	if !ok {
		t.Fatalf("syscall repaired_arguments type = %T, want map[string]any", respSys["repaired_arguments"])
	}
	if repairedSysArgs["api_key"] != "[REDACTED]" {
		t.Errorf("syscall repaired_arguments api_key = %v, want '[REDACTED]'", repairedSysArgs["api_key"])
	}
	resultMap, ok := respSys["result"].(map[string]any)
	if !ok {
		t.Fatalf("syscall result type = %T, want map[string]any", respSys["result"])
	}
	if sumVal, ok := resultMap["sum"].(float64); !ok || sumVal != 30 {
		t.Errorf("syscall sum = %v, want 30", resultMap["sum"])
	}

	// Verify executed arguments on the dispatched engine were redacted
	if rec.lastArgs == nil {
		t.Fatal("engine was not dispatched")
	}
	if rec.lastArgs["api_key"] != "[REDACTED]" {
		t.Errorf("engine received unredacted api_key: %v", rec.lastArgs["api_key"])
	}
	if strings.Contains(string(rec.rawArgs), secretVal) {
		t.Errorf("engine rawArgs contains secret value: %s", string(rec.rawArgs))
	}
	if !strings.Contains(string(rec.rawArgs), "[REDACTED]") {
		t.Errorf("engine rawArgs missing [REDACTED]: %s", string(rec.rawArgs))
	}
}

type mcpSpyDispatchEngine struct {
	dispatched bool
}

func (s *mcpSpyDispatchEngine) Caps() []abi.Capability { return nil }
func (s *mcpSpyDispatchEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	s.dispatched = true
	return &abi.Result{Status: abi.StatusOK}, nil
}

func TestMCPTransform_NonExecutableOutcomeRefusal(t *testing.T) {
	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	_, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer DisarmMCPTools()

	spy := &mcpSpyDispatchEngine{}
	abi.RegisterEngine("localtools", spy)
	defer abi.RegisterEngine("localtools", localEngine{})

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	t.Run("VerdictDeny_PolicyBlock", func(t *testing.T) {
		spy.dispatched = false
		argsJSON, _ := json.Marshal(map[string]any{
			"tool": "delete_account",
			"arguments": map[string]any{
				"user_id": "target_user",
			},
		})
		c := &abi.ToolCall{
			Tool: "fak_syscall",
			Args: putBytes(ctx, argsJSON),
		}
		res, err := eng.Complete(ctx, c)
		if err != nil {
			t.Fatalf("Complete error: %v", err)
		}
		var resp map[string]any
		if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		if resp["verdict"] != "deny" {
			t.Errorf("verdict = %v, want 'deny'", resp["verdict"])
		}
		resultMap, ok := resp["result"].(map[string]any)
		if !ok {
			t.Fatalf("result type = %T, want map[string]any", resp["result"])
		}
		if resultMap["error"] != "tool call denied by policy" {
			t.Errorf("error = %v, want 'tool call denied by policy'", resultMap["error"])
		}
		if resultMap["reason"] != "POLICY_BLOCK" {
			t.Errorf("reason = %v, want 'POLICY_BLOCK'", resultMap["reason"])
		}
		if spy.dispatched {
			t.Error("engine was dispatched on VerdictDeny, want no dispatch")
		}
	})

	t.Run("VerdictRequireWitness_NonExecutableOutcome", func(t *testing.T) {
		spy.dispatched = false
		argsJSON, _ := json.Marshal(map[string]any{
			"tool": "Bash",
			"arguments": map[string]any{
				"command": "rm -f x",
			},
		})
		c := &abi.ToolCall{
			Tool: "fak_syscall",
			Args: putBytes(ctx, argsJSON),
		}
		res, err := eng.Complete(ctx, c)
		if err != nil {
			t.Fatalf("Complete error: %v", err)
		}
		var resp map[string]any
		if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		if resp["verdict"] != "deny" {
			t.Errorf("verdict = %v, want 'deny'", resp["verdict"])
		}
		resultMap, ok := resp["result"].(map[string]any)
		if !ok {
			t.Fatalf("result type = %T, want map[string]any", resp["result"])
		}
		if resultMap["error"] != "tool call not executable" {
			t.Errorf("error = %v, want 'tool call not executable'", resultMap["error"])
		}
		if resultMap["reason"] == "" || resultMap["reason"] == nil {
			t.Errorf("reason should not be empty: %v", resultMap["reason"])
		}
		if spy.dispatched {
			t.Error("engine was dispatched on VerdictRequireWitness, want no dispatch")
		}
	})

	t.Run("Adjudicate_NonExecutableOutcome", func(t *testing.T) {
		argsJSON, _ := json.Marshal(map[string]any{
			"tool": "Bash",
			"arguments": map[string]any{
				"command": "rm -f x",
			},
		})
		c := &abi.ToolCall{
			Tool: "fak_adjudicate",
			Args: putBytes(ctx, argsJSON),
		}
		res, err := eng.Complete(ctx, c)
		if err != nil {
			t.Fatalf("Complete error: %v", err)
		}
		var resp map[string]any
		if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &resp); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		if resp["verdict"] != "deny" {
			t.Errorf("verdict = %v, want 'deny'", resp["verdict"])
		}
		if resp["allowed"] != false {
			t.Errorf("allowed = %v, want false", resp["allowed"])
		}
		if resp["reason"] == "" || resp["reason"] == nil {
			t.Errorf("reason should not be empty: %v", resp["reason"])
		}
	})
}
