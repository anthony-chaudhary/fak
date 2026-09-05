package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

func TestNativeHarnessDefaultPermissiveAllow(t *testing.T) {
	t.Cleanup(func() {
		SetConfiguredPosture(adjudicator.PostureDefaultOpen)
		Configure()
	})

	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	ctx := context.Background()

	// 1. Unlisted benign tools yield VerdictAllow with meta["posture"] == "default_open".
	unlistedBenign := []string{"custom_calculation", "unlisted_mcp_query"}
	for _, tool := range unlistedBenign {
		v := adjudicator.Default.Adjudicate(ctx, guardedCall(tool, `{"action":"test_query"}`))
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("unlisted tool %q: got Kind=%v (%s), want VerdictAllow", tool, v.Kind, abi.ReasonName(v.Reason))
		}
		if v.Meta["posture"] != "default_open" {
			t.Fatalf("unlisted tool %q: got Meta[posture]=%q, want %q", tool, v.Meta["posture"], "default_open")
		}
	}

	// 2. Explicit deny (delete_account) is strictly DENIED with POLICY_BLOCK.
	v := adjudicator.Default.Adjudicate(ctx, guardedCall("delete_account", `{"user_id":"u_12345"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("delete_account: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}

	// 3. Self-modification (internal/abi/kernel.go, dos.toml) is strictly DENIED with SELF_MODIFY.
	targets := []string{"internal/abi/kernel.go", "dos.toml"}
	for _, target := range targets {
		// Direct file write
		vDirect := adjudicator.Default.Adjudicate(ctx, guardedCall("write_file", fmt.Sprintf(`{"path":%q}`, target)))
		if vDirect.Kind != abi.VerdictDeny || vDirect.Reason != abi.ReasonSelfModify {
			t.Fatalf("direct self-modify %s: got %v/%s, want Deny/SELF_MODIFY",
				target, vDirect.Kind, abi.ReasonName(vDirect.Reason))
		}

		// Shell-mediated modification
		cmd := fmt.Sprintf("echo malicious >> %s", target)
		vShell := adjudicator.Default.Adjudicate(ctx, guardedCall("Bash", fmt.Sprintf(`{"command":%q}`, cmd)))
		if vShell.Kind != abi.VerdictDeny || vShell.Reason != abi.ReasonSelfModify {
			t.Fatalf("shell self-modify %s: got %v/%s, want Deny/SELF_MODIFY",
				target, vShell.Kind, abi.ReasonName(vShell.Reason))
		}
	}

	// 4. Dangerous gotchas (rm -rf / or kill -9 1) are strictly DENIED with POLICY_BLOCK.
	dangerousCmds := []string{"rm -rf /", "kill -9 1"}
	for _, cmd := range dangerousCmds {
		vGotcha := adjudicator.Default.Adjudicate(ctx, guardedCall("Bash", fmt.Sprintf(`{"command":%q}`, cmd)))
		if vGotcha.Kind != abi.VerdictDeny || vGotcha.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("dangerous gotcha %q: got %v/%s, want Deny/POLICY_BLOCK",
				cmd, vGotcha.Kind, abi.ReasonName(vGotcha.Reason))
		}
	}
}

func TestNativeHarnessMCPFeaturesDefaultExecution(t *testing.T) {
	t.Cleanup(func() {
		SetConfiguredPosture(adjudicator.PostureDefaultOpen)
		Configure()
		DisarmMCPTools()
	})

	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	cat, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	if len(cat) == 0 {
		t.Fatal("ArmMCPTools returned empty catalog")
	}

	ctx := context.Background()
	mcpEng := abi.Engine("inprocess_mcp")
	if mcpEng == nil {
		t.Fatal("inprocess_mcp engine is not registered")
	}

	// 1. Test fak_read reads a real file in a temporary workspace.
	tmpDir := t.TempDir()
	testFileName := "workspace_witness.txt"
	testFileContent := "fak native mcp read verification payload"
	if err := os.WriteFile(filepath.Join(tmpDir, testFileName), []byte(testFileContent), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	RegisterReadEngine(tmpDir)
	t.Cleanup(func() { RegisterReadEngine("") })

	readArgsJSON, _ := json.Marshal(map[string]any{"file_path": testFileName})
	readCall := &abi.ToolCall{
		Tool: "fak_read",
		Args: putBytes(ctx, readArgsJSON),
	}
	vRead := (mcpToolGate{}).Adjudicate(ctx, readCall)
	if vRead.Kind != abi.VerdictAllow {
		t.Fatalf("mcpToolGate fak_read verdict: got %v, want VerdictAllow", vRead.Kind)
	}
	if readCall.Engine != FakReadEngineID {
		t.Fatalf("fak_read engine: got %q, want %q", readCall.Engine, FakReadEngineID)
	}
	readEng := abi.Engine(readCall.Engine)
	if readEng == nil {
		t.Fatalf("Engine(%q) is nil", readCall.Engine)
	}
	readRes, err := readEng.Complete(ctx, readCall)
	if err != nil {
		t.Fatalf("fak_read Complete error: %v", err)
	}
	var readData map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, readRes.Payload), &readData); err != nil {
		t.Fatalf("Unmarshal fak_read result: %v", err)
	}
	if readData["content"] != testFileContent {
		t.Fatalf("fak_read content = %v, want %q", readData["content"], testFileContent)
	}

	// 2. Test fak_tools_search searches and returns tool definitions.
	searchArgsJSON, _ := json.Marshal(map[string]any{
		"query":        "read",
		"detail_level": "full",
	})
	searchCall := &abi.ToolCall{
		Tool: "fak_tools_search",
		Args: putBytes(ctx, searchArgsJSON),
	}
	searchRes, err := mcpEng.Complete(ctx, searchCall)
	if err != nil {
		t.Fatalf("fak_tools_search Complete error: %v", err)
	}
	var searchOut struct {
		Tools []map[string]any `json:"tools"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, searchRes.Payload), &searchOut); err != nil {
		t.Fatalf("Unmarshal fak_tools_search: %v", err)
	}
	if len(searchOut.Tools) == 0 {
		t.Fatal("fak_tools_search returned 0 tools for query 'read'")
	}
	foundRead := false
	for _, tool := range searchOut.Tools {
		if tool["name"] == "fak_read" {
			foundRead = true
			if desc, _ := tool["description"].(string); desc == "" {
				t.Errorf("fak_read missing description in full search")
			}
			if tool["parameters"] == nil && tool["inputSchema"] == nil {
				t.Errorf("fak_read missing schema in full search")
			}
		}
	}
	if !foundRead {
		t.Fatalf("expected fak_read to be found in search results: %+v", searchOut.Tools)
	}

	// 3. Test fak_adjudicate returns allowed for benign tools and denied for dangerous tools.
	runAdjudicate := func(toolName string, args any) (verdict string, allowed bool, reason string) {
		payloadJSON, _ := json.Marshal(map[string]any{
			"tool":      toolName,
			"arguments": args,
		})
		c := &abi.ToolCall{
			Tool: "fak_adjudicate",
			Args: putBytes(ctx, payloadJSON),
		}
		res, err := mcpEng.Complete(ctx, c)
		if err != nil {
			t.Fatalf("fak_adjudicate Complete for %s error: %v", toolName, err)
		}
		var out struct {
			Verdict string `json:"verdict"`
			Allowed bool   `json:"allowed"`
			Reason  string `json:"reason"`
		}
		if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &out); err != nil {
			t.Fatalf("Unmarshal fak_adjudicate output for %s: %v", toolName, err)
		}
		return out.Verdict, out.Allowed, out.Reason
	}

	// Benign tool checks
	for _, benign := range []string{"custom_calculation", "unlisted_mcp_query", "get_user_details"} {
		v, allowed, _ := runAdjudicate(benign, map[string]any{"action": "test"})
		if !allowed || v != "allow" {
			t.Errorf("fak_adjudicate %s: got allowed=%v verdict=%q, want allow", benign, allowed, v)
		}
	}

	// Dangerous tool checks
	v, allowed, reason := runAdjudicate("delete_account", map[string]any{"user_id": "u1"})
	if allowed || v != "deny" || reason != "POLICY_BLOCK" {
		t.Errorf("fak_adjudicate delete_account: got allowed=%v verdict=%q reason=%q, want deny/POLICY_BLOCK", allowed, v, reason)
	}

	v, allowed, reason = runAdjudicate("Bash", map[string]any{"command": "rm -rf /"})
	if allowed || v != "deny" || reason != "POLICY_BLOCK" {
		t.Errorf("fak_adjudicate rm -rf /: got allowed=%v verdict=%q reason=%q, want deny/POLICY_BLOCK", allowed, v, reason)
	}

	v, allowed, reason = runAdjudicate("Bash", map[string]any{"command": "kill -9 1"})
	if allowed || v != "deny" || reason != "POLICY_BLOCK" {
		t.Errorf("fak_adjudicate kill -9 1: got allowed=%v verdict=%q reason=%q, want deny/POLICY_BLOCK", allowed, v, reason)
	}

	v, allowed, reason = runAdjudicate("write_file", map[string]any{"path": "internal/abi/kernel.go"})
	if allowed || v != "deny" || reason != "SELF_MODIFY" {
		t.Errorf("fak_adjudicate write internal/abi/kernel.go: got allowed=%v verdict=%q reason=%q, want deny/SELF_MODIFY", allowed, v, reason)
	}

	// 4. Test fak_capabilities returns kernel capabilities.
	capCall := &abi.ToolCall{
		Tool: "fak_capabilities",
		Args: putBytes(ctx, []byte("{}")),
	}
	capRes, err := mcpEng.Complete(ctx, capCall)
	if err != nil {
		t.Fatalf("fak_capabilities Complete error: %v", err)
	}
	var capOut struct {
		Status       string   `json:"status"`
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, capRes.Payload), &capOut); err != nil {
		t.Fatalf("Unmarshal fak_capabilities: %v", err)
	}
	if capOut.Status != "ok" {
		t.Errorf("fak_capabilities status = %q, want 'ok'", capOut.Status)
	}
	capLookup := make(map[string]bool)
	for _, c := range capOut.Capabilities {
		capLookup[c] = true
	}
	for _, want := range []string{"vdso_dedup", "context_mmu", "posture_default_open", "default_permissive_allow", "mcp_features"} {
		if !capLookup[want] {
			t.Errorf("fak_capabilities missing expected capability: %q", want)
		}
	}

	// 5. Test fak_syscall executes and returns results.
	// Benign tool execution: calculate
	calcArgs, _ := json.Marshal(map[string]any{
		"tool": "calculate",
		"arguments": map[string]any{
			"a": 19,
			"b": 23,
		},
	})
	resSysCalc, err := mcpEng.Complete(ctx, &abi.ToolCall{Tool: "fak_syscall", Args: putBytes(ctx, calcArgs)})
	if err != nil {
		t.Fatalf("fak_syscall calculate Complete error: %v", err)
	}
	var sysCalcOut struct {
		Verdict string         `json:"verdict"`
		Result  map[string]any `json:"result"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, resSysCalc.Payload), &sysCalcOut); err != nil {
		t.Fatalf("Unmarshal fak_syscall calculate: %v", err)
	}
	if sysCalcOut.Verdict != "allow" {
		t.Fatalf("fak_syscall calculate verdict = %q, want 'allow'", sysCalcOut.Verdict)
	}
	sumVal, ok := sysCalcOut.Result["sum"].(float64)
	if !ok || sumVal != 42 {
		t.Fatalf("fak_syscall calculate sum = %v, want 42", sysCalcOut.Result["sum"])
	}

	// Execution of fak_read through fak_syscall
	readSysArgs, _ := json.Marshal(map[string]any{
		"tool": "fak_read",
		"arguments": map[string]any{
			"file_path": testFileName,
		},
	})
	resSysRead, err := mcpEng.Complete(ctx, &abi.ToolCall{Tool: "fak_syscall", Args: putBytes(ctx, readSysArgs)})
	if err != nil {
		t.Fatalf("fak_syscall fak_read Complete error: %v", err)
	}
	var sysReadOut struct {
		Verdict string         `json:"verdict"`
		Result  map[string]any `json:"result"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, resSysRead.Payload), &sysReadOut); err != nil {
		t.Fatalf("Unmarshal fak_syscall fak_read: %v", err)
	}
	if sysReadOut.Verdict != "allow" || sysReadOut.Result["content"] != testFileContent {
		t.Fatalf("fak_syscall fak_read: verdict=%q, content=%v", sysReadOut.Verdict, sysReadOut.Result["content"])
	}

	// Denied tool execution: delete_account through fak_syscall
	delSysArgs, _ := json.Marshal(map[string]any{
		"tool": "delete_account",
		"arguments": map[string]any{
			"user_id": "u1",
		},
	})
	resSysDel, err := mcpEng.Complete(ctx, &abi.ToolCall{Tool: "fak_syscall", Args: putBytes(ctx, delSysArgs)})
	if err != nil {
		t.Fatalf("fak_syscall delete_account Complete error: %v", err)
	}
	var sysDelOut struct {
		Verdict string         `json:"verdict"`
		Result  map[string]any `json:"result"`
	}
	if err := json.Unmarshal(refutil.Bytes(ctx, resSysDel.Payload), &sysDelOut); err != nil {
		t.Fatalf("Unmarshal fak_syscall delete_account: %v", err)
	}
	if sysDelOut.Verdict != "deny" {
		t.Fatalf("fak_syscall delete_account verdict = %q, want 'deny'", sysDelOut.Verdict)
	}
}

func TestNativeHarnessRunArmWithMCPTools(t *testing.T) {
	t.Cleanup(func() {
		SetConfiguredPosture(adjudicator.PostureDefaultOpen)
		Configure()
		DisarmMCPTools()
	})

	SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	Configure()

	catalog, err := ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools: %v", err)
	}

	ctx := context.Background()

	// Sub-test 1: Full RunArm turn proposing fak_tools_search
	t.Run("fak_tools_search", func(t *testing.T) {
		planner := &scriptedPlanner{turns: []*Completion{
			toolCallTurn("fak_tools_search", `{"query":"read"}`),
			{Message: Message{Content: "search done"}},
		}}
		var log []traceEvent
		m, err := RunArm(ctx, planner, "search tools", true, 5, &log, WithToolCatalog(catalog))
		if err != nil {
			t.Fatalf("RunArm with fak_tools_search failed: %v", err)
		}
		if m.Turns == 0 {
			t.Fatal("RunArm did not execute any turns")
		}
		found := false
		for _, ev := range log {
			if ev.Tool == "fak_tools_search" {
				found = true
				if ev.Verdict != "ALLOW" {
					t.Errorf("fak_tools_search verdict = %q, want ALLOW", ev.Verdict)
				}
				if ev.Reason == "DEFAULT_DENY" {
					t.Errorf("fak_tools_search was denied with DEFAULT_DENY")
				}
			}
		}
		if !found {
			t.Fatalf("fak_tools_search call not observed in trace log: %+v", log)
		}
	})

	// Sub-test 2: Full RunArm turn proposing fak_read in a temporary workspace
	t.Run("fak_read", func(t *testing.T) {
		tmpDir := t.TempDir()
		testFile := "task_spec.md"
		content := "# Mission Spec"
		if err := os.WriteFile(filepath.Join(tmpDir, testFile), []byte(content), 0644); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}
		RegisterReadEngine(tmpDir)
		t.Cleanup(func() { RegisterReadEngine("") })

		planner := &scriptedPlanner{turns: []*Completion{
			toolCallTurn("fak_read", fmt.Sprintf(`{"file_path":%q}`, testFile)),
			{Message: Message{Content: "read finished"}},
		}}
		var log []traceEvent
		m, err := RunArm(ctx, planner, "read task spec", true, 5, &log, WithToolCatalog(catalog))
		if err != nil {
			t.Fatalf("RunArm with fak_read failed: %v", err)
		}
		if m.Turns == 0 {
			t.Fatal("RunArm did not execute any turns")
		}
		found := false
		for _, ev := range log {
			if ev.Tool == "fak_read" {
				found = true
				if ev.Verdict != "ALLOW" {
					t.Errorf("fak_read verdict = %q, want ALLOW", ev.Verdict)
				}
				if ev.Reason == "DEFAULT_DENY" {
					t.Errorf("fak_read was denied with DEFAULT_DENY")
				}
			}
		}
		if !found {
			t.Fatalf("fak_read call not observed in trace log: %+v", log)
		}
	})

	// Sub-test 3: Full RunArm turn proposing an unlisted tool
	t.Run("unlisted_tool", func(t *testing.T) {
		planner := &scriptedPlanner{turns: []*Completion{
			toolCallTurn("custom_calculation", `{"expr":"1+1"}`),
			{Message: Message{Content: "unlisted calculation complete"}},
		}}
		var log []traceEvent
		m, err := RunArm(ctx, planner, "calculate unlisted", true, 5, &log)
		if err != nil {
			t.Fatalf("RunArm with custom_calculation failed: %v", err)
		}
		if m.Turns == 0 {
			t.Fatal("RunArm did not execute any turns")
		}
		found := false
		for _, ev := range log {
			if ev.Tool == "custom_calculation" {
				found = true
				if ev.Verdict != "ALLOW" {
					t.Errorf("custom_calculation verdict = %q, want ALLOW", ev.Verdict)
				}
				if ev.Reason == "DEFAULT_DENY" {
					t.Errorf("custom_calculation was denied with DEFAULT_DENY")
				}
			}
		}
		if !found {
			t.Fatalf("custom_calculation call not observed in trace log: %+v", log)
		}
	})
}
