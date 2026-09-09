package projectassets

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// canonicalSampleTools represents the standard suite of 20+ MCP tools advertised by fak serve and dos.
var canonicalSampleTools = []ToolDescriptor{
	{Name: "read", Description: "Read a file from the workspace", Category: "read"},
	{Name: "fak_read", Description: "Read files with verified-fresh cache reuse or cold read", Category: "read"},
	{Name: "grep", Description: "Fast content search across workspace files", Category: "search"},
	{Name: "glob", Description: "Fast file pattern matching across workspace", Category: "search"},
	{Name: "fak_tools_search", Description: "Search and retrieve tool schemas with progressive disclosure", Category: "discovery"},
	{Name: "edit", Description: "Performs exact string replacements in workspace files", Category: "mutation"},
	{Name: "write", Description: "Writes a file to the workspace filesystem", Category: "mutation"},
	{Name: "apply_patch", Description: "Applies a unified patch to workspace files", Category: "mutation"},
	{Name: "bash", Description: "Executes a given shell command in persistent session", Category: "execution"},
	{Name: "task", Description: "Launch a new agent to handle complex multistep tasks", Category: "orchestration"},
	{Name: "skill", Description: "Load a specialized skill when task matches available skills", Category: "discovery"},
	{Name: "todowrite", Description: "Create and maintain structured task list for session", Category: "tracking"},
	{Name: "webfetch", Description: "Fetches content from a specified URL", Category: "network"},
	{Name: "fak_adjudicate", Description: "Adjudicate a proposed tool call through the kernel", Category: "governance"},
	{Name: "fak_syscall", Description: "Adjudicate AND execute a tool call through the kernel", Category: "governance"},
	{Name: "fak_context_change", Description: "Switch context branch or reset context window", Category: "context"},
	{Name: "fak_context_restore", Description: "Restore prior session context checkpoint", Category: "context"},
	{Name: "dos_verify", Description: "Did (plan, phase) actually ship? — the truth syscall", Category: "verification"},
	{Name: "dos_arbitrate", Description: "May a worker take this lane right now? admission kernel", Category: "admission"},
	{Name: "dos_refuse_reasons", Description: "The closed structured-refusal vocabulary for workspace", Category: "vocabulary"},
	{Name: "dos_check_reason", Description: "Is reason_class a member of closed refusal vocabulary?", Category: "vocabulary"},
	{Name: "dos_commit_audit", Description: "Does a commit claim match what its diff actually did?", Category: "audit"},
	{Name: "dos_review", Description: "Review the residual, not the diff — attention bands", Category: "audit"},
	{Name: "dos_status", Description: "One folded status fact for a run — liveness, progress", Category: "status"},
}

// TestOpenCodeSubagentToolFilter is the canonical witness test for Issue #12306.
// It verifies:
// 1. explore & researcher expose only read-only search tools (fak_read, grep, glob, read) and strictly forbid mutation tools (edit, write, bash).
// 2. worker exposes implementation tools (edit, write, bash).
// 3. tester exposes test execution tools (bash, read, etc.) without file modification access.
// 4. Request identity extraction (headers and payload).
// 5. Prompt-prefix token cost reduction >60% on Metal / Apple Silicon inference turns.
func TestOpenCodeSubagentToolFilter(t *testing.T) {
	// 1. Test explore role: read-only search tools only
	t.Run("ExploreRoleReadOnly", func(t *testing.T) {
		filtered := FilterToolsForSubagent(SubagentRoleExplore, canonicalSampleTools)
		names := toolNames(filtered)

		// Must include read-only search tools
		for _, required := range []string{"fak_read", "grep", "glob"} {
			if !containsString(names, required) {
				t.Errorf("explore role missing required tool %q; got: %v", required, names)
			}
		}

		// Must strictly forbid modification tools (edit, write, bash)
		for _, forbidden := range []string{"edit", "write", "apply_patch", "bash"} {
			if containsString(names, forbidden) {
				t.Errorf("explore role received forbidden mutation tool %q (violates least privilege); got: %v", forbidden, names)
			}
		}
	})

	// 2. Test researcher role: read-only search tools only
	t.Run("ResearcherRoleReadOnly", func(t *testing.T) {
		filtered := FilterToolsForSubagent(SubagentRoleResearcher, canonicalSampleTools)
		names := toolNames(filtered)

		for _, required := range []string{"fak_read", "grep", "glob"} {
			if !containsString(names, required) {
				t.Errorf("researcher role missing required tool %q; got: %v", required, names)
			}
		}

		for _, forbidden := range []string{"edit", "write", "apply_patch", "bash"} {
			if containsString(names, forbidden) {
				t.Errorf("researcher role received forbidden mutation tool %q; got: %v", forbidden, names)
			}
		}
	})

	// 3. Test worker role: implementation tools
	t.Run("WorkerRoleImplementation", func(t *testing.T) {
		filtered := FilterToolsForSubagent(SubagentRoleWorker, canonicalSampleTools)
		names := toolNames(filtered)

		for _, required := range []string{"edit", "write", "bash"} {
			if !containsString(names, required) {
				t.Errorf("worker role missing implementation tool %q; got: %v", required, names)
			}
		}
	})

	// 4. Test tester role: test execution without file modification
	t.Run("TesterRoleExecutionWithoutModification", func(t *testing.T) {
		filtered := FilterToolsForSubagent(SubagentRoleTester, canonicalSampleTools)
		names := toolNames(filtered)

		// Tester must have execution capability (bash for test commands)
		if !containsString(names, "bash") {
			t.Errorf("tester role missing execution tool 'bash'; got: %v", names)
		}

		// Tester must NOT have file modification access
		for _, forbidden := range []string{"edit", "write", "apply_patch"} {
			if containsString(names, forbidden) {
				t.Errorf("tester role received forbidden modification tool %q; got: %v", forbidden, names)
			}
		}
	})

	// 5. Test token cost reduction >60% on local Apple Silicon inference
	t.Run("TokenCostReductionAbove60Percent", func(t *testing.T) {
		roles := []string{SubagentRoleExplore, SubagentRoleResearcher, SubagentRoleWorker, SubagentRoleTester}

		for _, role := range roles {
			receipt := EvaluateSubagentTokenSavings(role, canonicalSampleTools)
			if receipt.SavingsPercent < 60.0 {
				t.Errorf("role %q token savings %.1f%% does not meet >60%% threshold (before: %d tokens, after: %d tokens)",
					role, receipt.SavingsPercent, receipt.EstimatedTokensBefore, receipt.EstimatedTokensAfter)
			}
			if !receipt.ReductionTargetMet {
				t.Errorf("role %q failed ReductionTargetMet flag", role)
			}
		}
	})

	// 6. Test subagent identity extraction from HTTP request headers and payload
	t.Run("ExtractSubagentIdentity", func(t *testing.T) {
		req, _ := http.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"fak-local","messages":[{"role":"user","content":"search code"}]}`))
		req.Header.Set("X-OpenCode-Subagent", "explore")

		role := ExtractSubagentFromRequest(req)
		if role != SubagentRoleExplore {
			t.Errorf("expected role 'explore', got %q", role)
		}

		// Header variation: X-Subagent-Type
		req2, _ := http.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"fak-local"}`))
		req2.Header.Set("X-Subagent-Type", "tester")
		if got := ExtractSubagentFromRequest(req2); got != SubagentRoleTester {
			t.Errorf("expected role 'tester', got %q", got)
		}

		// In-payload subagent declaration
		req3, _ := http.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"fak-local","subagent":"worker"}`))
		if got := ExtractSubagentFromRequest(req3); got != SubagentRoleWorker {
			t.Errorf("expected role 'worker', got %q", got)
		}
	})
}

func TestSubagentCapabilityFloorEnforcement(t *testing.T) {
	// Verify that IsToolAllowedEnforced respects least privilege
	if IsToolAllowed(SubagentRoleExplore, "edit") {
		t.Errorf("IsToolAllowed(explore, edit) should be false")
	}
	if IsToolAllowed(SubagentRoleExplore, "bash") {
		t.Errorf("IsToolAllowed(explore, bash) should be false")
	}
	if !IsToolAllowed(SubagentRoleExplore, "fak_read") {
		t.Errorf("IsToolAllowed(explore, fak_read) should be true")
	}
	if !IsToolAllowed(SubagentRoleExplore, "grep") {
		t.Errorf("IsToolAllowed(explore, grep) should be true")
	}

	if !IsToolAllowed(SubagentRoleWorker, "edit") {
		t.Errorf("IsToolAllowed(worker, edit) should be true")
	}
	if !IsToolAllowed(SubagentRoleWorker, "write") {
		t.Errorf("IsToolAllowed(worker, write) should be true")
	}

	if !IsToolAllowed(SubagentRoleTester, "bash") {
		t.Errorf("IsToolAllowed(tester, bash) should be true")
	}
	if IsToolAllowed(SubagentRoleTester, "write") {
		t.Errorf("IsToolAllowed(tester, write) should be false")
	}
}

func TestEnsureOpenCodeSubagentConfig(t *testing.T) {
	tmp := t.TempDir()
	initialConfig := `{
  "$schema": "https://opencode.ai/config.json",
  "snapshot": false,
  "agent": {
    "explore": { "variant": "default" },
    "worker": { "variant": "medium" },
    "tester": { "variant": "medium" }
  }
}`
	configPath := filepath.Join(tmp, "opencode.json")
	if err := os.WriteFile(configPath, []byte(initialConfig), 0644); err != nil {
		t.Fatal(err)
	}

	modified, err := EnsureOpenCodeSubagentPermissions(tmp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true on config injection")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	agentMap, ok := parsed["agent"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing agent map")
	}

	exploreMap, ok := agentMap["explore"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing explore agent entry")
	}
	toolsMap, ok := exploreMap["tools"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing explore tools map")
	}
	if toolsMap["edit"] != false || toolsMap["write"] != false || toolsMap["bash"] != false {
		t.Errorf("explore agent tools should disable edit/write/bash: %v", toolsMap)
	}
	if toolsMap["grep"] != true || toolsMap["glob"] != true {
		t.Errorf("explore agent tools should enable grep/glob: %v", toolsMap)
	}
}

func toolNames(tools []ToolDescriptor) []string {
	res := make([]string, len(tools))
	for i, t := range tools {
		res[i] = t.Name
	}
	return res
}

func containsString(list []string, item string) bool {
	for _, s := range list {
		if s == item {
			return true
		}
	}
	return false
}
