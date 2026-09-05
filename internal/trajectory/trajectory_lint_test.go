package trajectory

import (
	"testing"
)

func TestFileReadToolSelection(t *testing.T) {
	t.Run("unsteered_trajectory_fails_ratio_check", func(t *testing.T) {
		// 1 fak_read, 49 exec_command cat calls (approx 2% fak_read)
		var calls []ToolCallRecord
		calls = append(calls, ToolCallRecord{Tool: "fak_read", Command: ""})
		for i := 0; i < 49; i++ {
			calls = append(calls, ToolCallRecord{
				Tool:    "exec_command",
				Command: "cat internal/policy/harness_profiles.go",
			})
		}

		ratio := LintFileReadToolSelection(calls, 0.80)
		if ratio.Passed {
			t.Errorf("unsteered trajectory with 2%% fak_read should FAIL 0.80 minRatio; got passed: %+v", ratio)
		}
		if ratio.FakReadCalls != 1 {
			t.Errorf("expected 1 FakReadCalls, got %d", ratio.FakReadCalls)
		}
		if ratio.ExecReadCalls != 49 {
			t.Errorf("expected 49 ExecReadCalls, got %d", ratio.ExecReadCalls)
		}
		if ratio.EffectiveRatio > 0.05 {
			t.Errorf("expected effective ratio ~0.02, got %f", ratio.EffectiveRatio)
		}
		if ratio.Defect == "" {
			t.Errorf("expected non-empty Defect explanation on failure")
		}
	})

	t.Run("steered_trajectory_passes_ratio_check", func(t *testing.T) {
		// 85 fak_read, 15 exec_command reads (85% fak_read)
		var calls []ToolCallRecord
		for i := 0; i < 85; i++ {
			calls = append(calls, ToolCallRecord{Tool: "fak_read", Command: ""})
		}
		for i := 0; i < 15; i++ {
			calls = append(calls, ToolCallRecord{
				Tool:    "exec_command",
				Command: "head -n 20 internal/policy/harness_profiles.go",
			})
		}

		ratio := LintFileReadToolSelection(calls, 0.80)
		if !ratio.Passed {
			t.Errorf("steered trajectory with 85%% fak_read should PASS 0.80 minRatio; got failed: %+v", ratio)
		}
		if ratio.FakReadCalls != 85 {
			t.Errorf("expected 85 FakReadCalls, got %d", ratio.FakReadCalls)
		}
		if ratio.ExecReadCalls != 15 {
			t.Errorf("expected 15 ExecReadCalls, got %d", ratio.ExecReadCalls)
		}
		if ratio.EffectiveRatio < 0.80 {
			t.Errorf("expected effective ratio >= 0.80, got %f", ratio.EffectiveRatio)
		}
		if ratio.Defect != "" {
			t.Errorf("expected empty Defect on success, got %q", ratio.Defect)
		}
		AssertFileReadToolSelectionRatio(t, ratio)
	})

	t.Run("empty_or_non_read_commands_pass", func(t *testing.T) {
		calls := []ToolCallRecord{
			{Tool: "exec_command", Command: "git status"},
			{Tool: "exec_command", Command: "go test ./..."},
		}
		ratio := LintFileReadToolSelection(calls, 0.80)
		if !ratio.Passed {
			t.Errorf("no read operations should yield 1.0 ratio and pass; got: %+v", ratio)
		}
		if ratio.EffectiveRatio != 1.0 {
			t.Errorf("expected 1.0 EffectiveRatio, got %f", ratio.EffectiveRatio)
		}
	})

	t.Run("zero_calls_passes_with_perfect_ratio", func(t *testing.T) {
		var calls []ToolCallRecord
		ratio := LintFileReadToolSelection(calls, 0.80)
		if !ratio.Passed {
			t.Errorf("expected 0 calls to pass, got: %+v", ratio)
		}
		if ratio.EffectiveRatio != 1.0 {
			t.Errorf("expected 1.0 EffectiveRatio, got %f", ratio.EffectiveRatio)
		}
	})

	t.Run("all_fak_read_passes_with_perfect_ratio", func(t *testing.T) {
		calls := []ToolCallRecord{
			{Tool: "fak_read", Command: ""},
			{Tool: "fak_read", Command: ""},
			{Tool: "read", Command: ""},
			{Tool: "mcp__fak__fak_read", Command: ""},
		}
		ratio := LintFileReadToolSelection(calls, 0.80)
		if !ratio.Passed {
			t.Errorf("expected all fak_read calls to pass, got: %+v", ratio)
		}
		if ratio.EffectiveRatio != 1.0 {
			t.Errorf("expected 1.0 EffectiveRatio, got %f", ratio.EffectiveRatio)
		}
		if ratio.FakReadCalls != 4 {
			t.Errorf("expected 4 FakReadCalls, got %d", ratio.FakReadCalls)
		}
	})

	t.Run("all_exec_read_fails", func(t *testing.T) {
		calls := []ToolCallRecord{
			{Tool: "exec_command", Command: "cat file1.txt"},
			{Tool: "bash", Command: "head -n 10 file2.txt"},
		}
		ratio := LintFileReadToolSelection(calls, 0.80)
		if ratio.Passed {
			t.Errorf("expected all exec_read calls to fail, got: %+v", ratio)
		}
		if ratio.EffectiveRatio != 0.0 {
			t.Errorf("expected 0.0 EffectiveRatio, got %f", ratio.EffectiveRatio)
		}
	})

	t.Run("heredoc_write_not_treated_as_read", func(t *testing.T) {
		calls := []ToolCallRecord{
			{Tool: "exec_command", Command: "cat << 'EOF' > file.txt\ncontent\nEOF"},
			{Tool: "bash", Command: "cat input.txt > output.txt"},
			{Tool: "fak_read", Command: ""},
		}
		ratio := LintFileReadToolSelection(calls, 0.80)
		if !ratio.Passed {
			t.Errorf("expected heredoc/redirect to not count as read, got: %+v", ratio)
		}
		if ratio.ExecReadCalls != 0 {
			t.Errorf("expected 0 ExecReadCalls for heredoc/redirect, got %d", ratio.ExecReadCalls)
		}
	})

	t.Run("case_insensitivity_and_aliases", func(t *testing.T) {
		calls := []ToolCallRecord{
			{Tool: "FAK_READ", Command: ""},
			{Tool: "READ", Command: ""},
			{Tool: "EXEC_COMMAND", Command: "CAT file.txt"},
			{Tool: "Bash", Command: "HEAD -n 5 file.txt"},
			{Tool: "powershell", Command: "Get-Content file.txt"},
			{Tool: "pwsh", Command: "gc file.txt"},
			{Tool: "cmd", Command: "type file.txt"},
			{Tool: "sh", Command: "dir"},
		}
		ratio := LintFileReadToolSelection(calls, 0.80)
		if ratio.FakReadCalls != 2 {
			t.Errorf("expected 2 FakReadCalls from case-insensitive tools, got %d", ratio.FakReadCalls)
		}
		if ratio.ExecReadCalls != 6 {
			t.Errorf("expected 6 ExecReadCalls from case-insensitive verbs, got %d", ratio.ExecReadCalls)
		}
	})

	t.Run("complex_commands_and_arguments", func(t *testing.T) {
		calls := []ToolCallRecord{
			// Non-read commands with argument mentioning read verbs
			{Tool: "exec_command", Command: "git commit -m 'cat is cute'"},
			{Tool: "exec_command", Command: "echo 'cat head tail'"},
			{Tool: "exec_command", Command: "pytest tests/test_category.py"},
			// Read commands with wrappers, pipes, chained commands
			{Tool: "exec_command", Command: "sudo cat /etc/hosts"},
			{Tool: "exec_command", Command: "echo foo | cat"},
			{Tool: "exec_command", Command: "cd /tmp; head log.txt"},
			{Tool: "exec_command", Command: "VAR=1 env /usr/bin/tail -f app.log"},
		}
		ratio := LintFileReadToolSelection(calls, 0.50)
		if ratio.ExecReadCalls != 4 {
			t.Errorf("expected exactly 4 ExecReadCalls, got %d", ratio.ExecReadCalls)
		}
	})
}

func TestSubagentTimeoutRecovery(t *testing.T) {
	t.Run("timeout_followed_by_close_and_spawn_passes", func(t *testing.T) {
		events := []string{
			"spawn_agent",
			"wait_agent",
			"timeout",
			"close_agent",
			"spawn_agent",
			"wait_agent",
		}
		audit := LintSubagentTimeoutRecovery(events)
		if !audit.Passed {
			t.Errorf("expected timeout with close_agent and spawn_agent to pass; got audit: %+v", audit)
		}
		if audit.StalledWithoutCancel != 0 {
			t.Errorf("expected StalledWithoutCancel == 0, got %d", audit.StalledWithoutCancel)
		}
		if audit.FallbackToMonolithicShell != 0 {
			t.Errorf("expected FallbackToMonolithicShell == 0, got %d", audit.FallbackToMonolithicShell)
		}
		if audit.Timeouts != 1 {
			t.Errorf("expected 1 timeout, got %d", audit.Timeouts)
		}
		if audit.Defect != "" {
			t.Errorf("expected empty defect, got %s", audit.Defect)
		}
	})

	t.Run("timeout_followed_by_5_sequential_exec_commands_fails", func(t *testing.T) {
		events := []string{
			"spawn_agent",
			"wait_agent",
			"timeout",
			"exec_command",
			"exec_command",
			"exec_command",
			"exec_command",
			"exec_command",
		}
		audit := LintSubagentTimeoutRecovery(events)
		if audit.Passed {
			t.Errorf("expected timeout followed by 5 exec_command calls to FAIL; got passed: %+v", audit)
		}
		if audit.FallbackToMonolithicShell == 0 {
			t.Errorf("expected FallbackToMonolithicShell > 0, got %d", audit.FallbackToMonolithicShell)
		}
		if audit.StalledWithoutCancel == 0 {
			t.Errorf("expected StalledWithoutCancel > 0, got %d", audit.StalledWithoutCancel)
		}
		if audit.Defect == "" {
			t.Errorf("expected non-empty Defect on failure")
		}
	})

	t.Run("zero_events_passes", func(t *testing.T) {
		var events []string
		audit := LintSubagentTimeoutRecovery(events)
		if !audit.Passed {
			t.Errorf("expected 0 events to pass, got: %+v", audit)
		}
		if audit.Timeouts != 0 || audit.Defect != "" {
			t.Errorf("unexpected audit values for empty events: %+v", audit)
		}
	})

	t.Run("case_insensitivity", func(t *testing.T) {
		events := []string{
			"SPAWN_AGENT",
			"WAIT_AGENT",
			"TIMEOUT: subagent 123",
			"CLOSE_AGENT",
			"SPAWNAGENT",
		}
		audit := LintSubagentTimeoutRecovery(events)
		if !audit.Passed {
			t.Errorf("expected case-insensitive events to pass, got: %+v", audit)
		}
		if audit.Timeouts != 1 {
			t.Errorf("expected 1 timeout detected, got %d", audit.Timeouts)
		}
		if audit.StalledWithoutCancel != 0 {
			t.Errorf("expected 0 StalledWithoutCancel, got %d", audit.StalledWithoutCancel)
		}
	})

	t.Run("timeout_unclosed_without_shell_fails_stalled_without_cancel", func(t *testing.T) {
		events := []string{
			"spawn_agent",
			"wait_agent",
			"timeout",
		}
		audit := LintSubagentTimeoutRecovery(events)
		if audit.Passed {
			t.Errorf("expected timeout without close to fail, got passed: %+v", audit)
		}
		if audit.StalledWithoutCancel != 1 {
			t.Errorf("expected 1 StalledWithoutCancel, got %d", audit.StalledWithoutCancel)
		}
		if audit.FallbackToMonolithicShell != 0 {
			t.Errorf("expected 0 FallbackToMonolithicShell, got %d", audit.FallbackToMonolithicShell)
		}
	})
}
