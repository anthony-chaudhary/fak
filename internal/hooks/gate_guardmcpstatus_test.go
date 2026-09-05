package hooks

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func writeTestJSON(t *testing.T, root, rel string, data any) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTestText(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestGuardMCPStatus_LiveTreeClean(t *testing.T) {
	root := repoRoot(t)
	checks := runGuardMCPStatusAudit(root)
	if len(checks) != 13 {
		t.Fatalf("expected 13 checks, got %d", len(checks))
	}
	for _, c := range checks {
		if !c.passed {
			t.Errorf("check failed on live tree: %s (%s): %s", c.name, c.file, c.detail)
		}
	}

	tree, err := ReadTrackedTree(root)
	if err != nil {
		t.Fatal(err)
	}
	findings, err := gateGuardMCPStatusTree(tree)
	if err != nil {
		t.Fatalf("gateGuardMCPStatusTree error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings on live tree, got: %v", findings)
	}
}

func TestGuardMCPStatus_StatusPacketMissingSubstring(t *testing.T) {
	root := t.TempDir()
	// Write empty status packet
	writeTestText(t, root, guardMCPStatusPacketPath, "# Incomplete packet\n")

	var checks []guardCheckResult
	checkGuardStatusPacket(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected status packet check to fail, got: %+v", checks)
	}
	if !strings.Contains(checks[0].detail, "missing") {
		t.Errorf("expected detail to mention missing items: %s", checks[0].detail)
	}
}

func TestGuardMCPStatus_GuardTestsMissing(t *testing.T) {
	root := t.TempDir()
	writeTestText(t, root, guardMCPGuardTestPath, "package main\nfunc TestSomethingElse() {}\n")

	var checks []guardCheckResult
	checkGuardDefaultTests(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected guard tests check to fail, got: %+v", checks)
	}
	if !strings.Contains(checks[0].detail, "TestGuardDefaultPolicyDeniesDangerAllowsBenign") {
		t.Errorf("expected detail to name missing test: %s", checks[0].detail)
	}
}

func TestGuardMCPStatus_MCPStdioFails(t *testing.T) {
	root := t.TempDir()
	badDogfood := map[string]any{
		"checks": map[string]any{
			"mcp_stdio_adjudication": map[string]any{
				"status": "PASS",
				"denies_publish": map[string]any{
					"kind":   "ALLOW",
					"reason": "NONE",
				},
				"allows_status": map[string]any{
					"kind": "ALLOW",
				},
				"missing_tools": []any{},
			},
		},
	}
	writeTestJSON(t, root, guardMCPDogfoodPath, badDogfood)

	var checks []guardCheckResult
	checkGuardMCPStdio(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected mcp stdio check to fail, got: %+v", checks)
	}
}

func TestGuardMCPStatus_GitGateReportsFails(t *testing.T) {
	root := t.TempDir()
	badReport := map[string]any{
		"tool":        "git_push",
		"status":      "EXECUTED",
		"expect_deny": false,
		"executed":    true,
		"preflight": map[string]any{
			"verdict": "ALLOW",
			"reason":  "",
		},
	}
	writeTestJSON(t, root, "experiments/agent-live/codex-fak-gate-git-push.json", badReport)

	var checks []guardCheckResult
	checkGuardGitGateReports(root, &checks)
	failedPush := false
	for _, c := range checks {
		if strings.Contains(c.name, "git_push") && !c.passed {
			failedPush = true
		}
	}
	if !failedPush {
		t.Fatalf("expected git_push gate report check to fail, got: %+v", checks)
	}
}

func TestGuardMCPStatus_HistoricalAuditFailsOnGitWrite(t *testing.T) {
	root := t.TempDir()
	badAudit := map[string]any{
		"status": "PASS",
		"actionability": map[string]any{
			"status":                         "PASS",
			"residual":                       []any{"HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE"},
			"reasons":                        []any{},
			"post_repair_shell_shape_counts": map[string]any{"shell_no_write_target_detected": 10},
		},
		"git_gate_evidence": map[string]any{
			"status": "PASS",
			"post_gate_command_shapes": map[string]any{
				"shell_family_counts": map[string]any{
					"git_write": 5, // leaky git_write in post-gate
				},
			},
		},
		"summary": map[string]any{
			"workspace_stop_failures_total":                   10,
			"workspace_stop_failure_active_consecutive_total": 0,
		},
	}
	writeTestJSON(t, root, guardMCPDosAuditPath, badAudit)

	var checks []guardCheckResult
	checkGuardHistoricalAudit(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected historical audit to fail on leaky git_write, got: %+v", checks)
	}
}

func TestGuardMCPStatus_ClaudeHistoricalFailsOnLeakedPayload(t *testing.T) {
	root := t.TempDir()
	data := map[string]any{
		"schema":                     "fak-claude-historical-guard-audit/1",
		"status":                     "PASS",
		"all_accounts":               true,
		"root_labels":                []any{"a", "b"},
		"sessions_discovered":        5,
		"sessions_audited":           5,
		"tool_calls_seen":            10,
		"unique_tool_calls_replayed": 10,
		"verdict_counts":             map[string]any{"ALLOW": 5, "DENY": 5},
		"reason_counts":              map[string]any{"POLICY_BLOCK": 5},
		"truncated":                  false,
		"top_friction_sessions":      []any{map[string]any{"session_digest": "abc"}},
		"transcript_shape": map[string]any{
			"summarized_sessions": 5,
			"evidence_tag_counts": map[string]any{
				"HOOK_OR_API_WALL_FEEDBACK": 2,
				"HOST_PERMISSION_INTERRUPT": 2,
			},
			"remediation_session_counts": map[string]any{
				"clear_hook_or_api_wall_feedback": 2,
			},
		},
	}
	writeTestJSON(t, root, guardMCPClaudeHistJSONPath, data)
	// Leaked payload: "rm -rf"
	md := "# Audit\nstatus: **`PASS`**\nTranscript Friction Signals\nIt never writes prompts, tool arguments, tool results, or raw transcript text.\nLeaked rm -rf here\n"
	writeTestText(t, root, guardMCPClaudeHistMDPath, md)

	var checks []guardCheckResult
	checkGuardClaudeHistorical(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected claude historical check to fail on leaked payload, got: %+v", checks)
	}
	if !strings.Contains(checks[0].detail, "leaked_payload=true") {
		t.Errorf("expected detail to state leaked_payload=true: %s", checks[0].detail)
	}
}

func TestGuardMCPStatus_ClaudeLiveFails(t *testing.T) {
	root := t.TempDir()
	data := map[string]any{
		"status": "PASS",
		"session": map[string]any{
			"claude_session_id": "sess-1",
		},
		"dangerous_attempt": map[string]any{
			"verdict": map[string]any{"kind": "DENY", "reason": "POLICY_BLOCK"},
		},
		"useful_continuation": map[string]any{
			"same_claude_session_id": "sess-1",
			"verdict":                map[string]any{"kind": "ALLOW"},
			"final_message": map[string]any{
				"useful_completed": false, // failed
			},
		},
	}
	writeTestJSON(t, root, guardMCPClaudeLivePath, data)

	var checks []guardCheckResult
	checkGuardClaudeLive(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected claude live check to fail when useful_completed=false, got: %+v", checks)
	}
}

func TestGuardMCPStatus_CodexMCPLiveFails(t *testing.T) {
	root := t.TempDir()
	data := map[string]any{
		"status": "PASS",
		"dangerous_attempt": map[string]any{
			"verdict": map[string]any{"kind": "DENY", "reason": "POLICY_BLOCK"},
		},
		"useful_continuation": map[string]any{
			"verdict": map[string]any{"kind": "ALLOW"},
		},
		"final_message": map[string]any{
			"denied_attempt":   true,
			"useful_continued": false, // failed
		},
	}
	writeTestJSON(t, root, guardMCPCodexMCPLivePath, data)

	var checks []guardCheckResult
	checkGuardCodexMCPLive(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected codex mcp live check to fail when useful_continued=false, got: %+v", checks)
	}
}

func TestGuardMCPStatus_OpenAIAgentsAdapterFails(t *testing.T) {
	root := t.TempDir()
	// Demo file missing
	writeTestText(t, root, guardMCPOpenAIAgentsOutput, "incomplete output\n")

	var checks []guardCheckResult
	checkGuardOpenAIAgentsAdapter(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected openai agents adapter check to fail, got: %+v", checks)
	}
}

func TestGuardMCPStatus_OpenAILivePrereqFailsOnSecretLeak(t *testing.T) {
	root := t.TempDir()
	data := map[string]any{
		"schema":              "fak-openai-live-prereq-audit/1",
		"status":              "PARTIAL",
		"codex_login_ready":   true,
		"hosted_openai_ready": true,
		"auth_sources": map[string]any{
			"codex_login": true,
		},
		"blockers": []any{},
	}
	writeTestJSON(t, root, guardMCPOpenAIPrereqJSON, data)
	// Leak an API key prefix
	md := "# Prereq\nstatus: **`PARTIAL`**\nIt never writes API key values\nCodex token values\nsk-1234567890\n"
	writeTestText(t, root, guardMCPOpenAIPrereqMD, md)

	var checks []guardCheckResult
	checkGuardOpenAILivePrereq(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected openai live prereqs to fail on secret leak, got: %+v", checks)
	}
}

func TestGuardMCPStatus_OpenAIHostedLivePilotFails(t *testing.T) {
	root := t.TempDir()
	data := map[string]any{
		"schema": "fak-openai-hosted-live-pilot/1",
		"status": "PASS",
		"guard": map[string]any{
			"status": "PASS",
			"dangerous_attempt": map[string]any{
				"verdict":  map[string]any{"kind": "ALLOW"}, // should be DENY
				"executed": true,
			},
			"useful_continuation": map[string]any{
				"verdict": map[string]any{"kind": "ALLOW"},
			},
		},
		"hosted_openai": map[string]any{
			"status":                   "PASS",
			"auth_source":              "codex_login",
			"codex_exec_exit_code":     0,
			"contains_expected_marker": true,
			"output_text_sha256":       "abc123",
		},
	}
	writeTestJSON(t, root, guardMCPOpenAIHostedJSON, data)
	writeTestText(t, root, guardMCPOpenAIHostedMD, "# Pilot\nstatus: **`PASS`**\n")

	var checks []guardCheckResult
	checkGuardOpenAIHostedLivePilot(root, &checks)
	if len(checks) != 1 || checks[0].passed {
		t.Fatalf("expected openai hosted live pilot check to fail, got: %+v", checks)
	}
}

func TestGuardMCPStatus_DifferentialAgainstPython(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	root := repoRoot(t)
	script := filepath.Join(root, "tools", "guard_mcp_status_audit.py")
	if _, err := os.Stat(script); err != nil {
		t.Skip("tools/guard_mcp_status_audit.py not found")
	}

	cmd := exec.Command("python3", script)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("python tools/guard_mcp_status_audit.py failed: %v\n%s", err, string(out))
	}

	checks := runGuardMCPStatusAudit(root)
	for _, c := range checks {
		if !c.passed {
			t.Errorf("Go check %s failed on clean repo: %s", c.name, c.detail)
		}
	}
}
