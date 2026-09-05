package auditreason

import (
	"strings"
	"testing"
)

func TestToolFailuresClosedMetadata(t *testing.T) {
	rows := ToolFailures()
	want := []ToolFailure{
		ToolFailureHang,
		ToolFailureHangShellMismatch,
		ToolFailurePartialApply,
		ToolFailureShellMismatch,
		ToolFailureTimeout,
	}
	if len(rows) != len(want) {
		t.Fatalf("ToolFailures length = %d, want %d: %+v", len(rows), len(want), rows)
	}
	for i, row := range rows {
		if row.Token != want[i] {
			t.Fatalf("row %d token = %q, want %q (closed sorted vocabulary)", i, row.Token, want[i])
		}
		if strings.TrimSpace(row.Summary) == "" || strings.TrimSpace(row.Fix) == "" {
			t.Fatalf("row %q missing summary/fix metadata: %+v", row.Token, row)
		}
	}
}

func TestLookupToolFailure(t *testing.T) {
	spec, ok := LookupToolFailure("tool-hang-shell-mismatch")
	if !ok {
		t.Fatal("hyphenated lookup did not resolve TOOL_HANG_SHELL_MISMATCH")
	}
	if spec.Token != ToolFailureHangShellMismatch || !spec.Retryable {
		t.Fatalf("lookup = %+v, want retryable TOOL_HANG_SHELL_MISMATCH", spec)
	}
	if _, ok := LookupToolFailure("FILE_ADMISSION"); ok {
		t.Fatal("guard refusal token must not resolve as a non-guard tool failure")
	}
}

func TestToolFailureFromMessage(t *testing.T) {
	cases := []struct {
		msg  string
		want ToolFailure
	}{
		{"Bash exited with exit status 143 while running gh issue list", ToolFailureHangShellMismatch},
		{"context deadline exceeded while waiting for tool output", ToolFailureTimeout},
		{"shell mismatch: syntax error near unexpected token `then'", ToolFailureShellMismatch},
		{"partial apply: edit wrote two files before the third hunk failed", ToolFailurePartialApply},
		{"tool hung: no output for 120s", ToolFailureHang},
	}
	for _, c := range cases {
		t.Run(string(c.want), func(t *testing.T) {
			got, ok := ToolFailureFromMessage(c.msg)
			if !ok {
				t.Fatalf("ToolFailureFromMessage(%q) did not match", c.msg)
			}
			if got.Token != c.want {
				t.Fatalf("ToolFailureFromMessage(%q) = %q, want %q", c.msg, got.Token, c.want)
			}
		})
	}
	if _, ok := ToolFailureFromMessage("everything succeeded"); ok {
		t.Fatal("unrelated success text must not map into the closed failure vocabulary")
	}
	for _, text := range []string{
		"stopped making progress on ticket #123",
		"we hung the painting on the wall",
		"timed out after 5pm",
	} {
		if spec, ok := ToolFailureFromMessage(text); ok {
			t.Fatalf("benign phrase %q should not match tool failure, got: %+v", text, spec)
		}
	}
}

func TestToolFailureRetryContract(t *testing.T) {
	for _, row := range ToolFailures() {
		switch row.Token {
		case ToolFailurePartialApply:
			if row.Retryable {
				t.Fatal("TOOL_PARTIAL_APPLY must not be marked directly retryable; it needs read-back first")
			}
		default:
			if !row.Retryable {
				t.Fatalf("%s should be retryable after the recovery action", row.Token)
			}
		}
	}
}

// Every vocabulary row must carry a pre-filled recovery command so an agent can
// branch on the code and run NextCommand instead of parsing prose.
func TestToolFailureNextCommandPresent(t *testing.T) {
	for _, row := range ToolFailures() {
		if strings.TrimSpace(row.NextCommand) == "" {
			t.Fatalf("%s missing next_command recovery", row.Token)
		}
	}
}

// A SIMULATED tool hang returns a structured payload with a recognized code and a
// runnable next_command.
func TestToolFailurePayloadHang(t *testing.T) {
	payload, ok := ToolFailurePayloadFromMessage("tool hung: no output for 120s")
	if !ok {
		t.Fatal("simulated hang did not classify into the closed vocabulary")
	}
	if payload.Code != ToolFailureHang {
		t.Fatalf("code = %q, want %q", payload.Code, ToolFailureHang)
	}
	if strings.TrimSpace(payload.NextCommand) == "" {
		t.Fatalf("hang payload has no runnable next_command: %+v", payload)
	}
	if strings.TrimSpace(payload.Evidence) == "" || strings.TrimSpace(payload.Cause) == "" || strings.TrimSpace(payload.Fix) == "" {
		t.Fatalf("hang payload missing cause/evidence/fix: %+v", payload)
	}
}

// A SIMULATED non-guard tool failure (partial mutation) returns a structured
// payload with a recognized code and a runnable next_command.
func TestToolFailurePayloadGenericFailure(t *testing.T) {
	payload, ok := ToolFailurePayloadFromMessage("partial apply: edit wrote two files before the third hunk failed")
	if !ok {
		t.Fatal("simulated partial-apply failure did not classify")
	}
	if payload.Code != ToolFailurePartialApply {
		t.Fatalf("code = %q, want %q", payload.Code, ToolFailurePartialApply)
	}
	if payload.Retryable {
		t.Fatal("partial-apply payload must stay non-retryable until read-back")
	}
	if strings.TrimSpace(payload.NextCommand) == "" {
		t.Fatalf("generic-failure payload has no runnable next_command: %+v", payload)
	}
}

// The canonical case: a Bash git/gh command killed with exit 143 classifies to
// TOOL_HANG_SHELL_MISMATCH and its next_command names the PowerShell recovery.
func TestToolFailurePayloadExit143NamesPowerShell(t *testing.T) {
	payload, ok := ToolFailurePayloadFromMessage("Bash exited with exit status 143 while running gh issue list")
	if !ok {
		t.Fatal("exit-143 git/gh hang did not classify")
	}
	if payload.Code != ToolFailureHangShellMismatch {
		t.Fatalf("code = %q, want %q", payload.Code, ToolFailureHangShellMismatch)
	}
	if !strings.Contains(strings.ToLower(payload.NextCommand), "powershell") {
		t.Fatalf("exit-143 next_command must name the PowerShell recovery, got %q", payload.NextCommand)
	}
}

// With the exact failing command, the shell-mismatch recovery is pre-filled into
// a literally-runnable PowerShell next_command rather than the placeholder.
func TestToolFailurePayloadForCommandPreFillsPowerShell(t *testing.T) {
	payload, ok := ToolFailurePayloadForCommand(
		"Bash exited with exit status 143 while running the command",
		"gh issue view 2072 --json title",
	)
	if !ok {
		t.Fatal("exit-143 with command did not classify")
	}
	want := "powershell -NoProfile -Command 'gh issue view 2072 --json title'"
	if payload.NextCommand != want {
		t.Fatalf("next_command = %q, want %q", payload.NextCommand, want)
	}
	if strings.Contains(payload.NextCommand, "<") {
		t.Fatalf("placeholder left unfilled in exact recovery: %q", payload.NextCommand)
	}
}

func TestExit127CommandNotFoundClassifiesShellMismatchWithPowerShellRecovery(t *testing.T) {
	payload, ok := ToolFailurePayloadForCommand(
		"Bash exited with exit status 127: Get-ChildItem: command not found",
		"Get-ChildItem | Select-Object -First 5",
	)
	if !ok {
		t.Fatal("exit-127 command-not-found did not classify")
	}
	if payload.Code != ToolFailureShellMismatch {
		t.Fatalf("code=%s, want %s", payload.Code, ToolFailureShellMismatch)
	}
	if !strings.Contains(strings.ToLower(payload.NextCommand), "powershell") || !strings.Contains(payload.NextCommand, "Get-ChildItem") {
		t.Fatalf("next_command=%q, want runnable PowerShell recovery", payload.NextCommand)
	}
}

func TestToolFailurePayloadUnrecognized(t *testing.T) {
	if _, ok := ToolFailurePayloadFromMessage("everything succeeded"); ok {
		t.Fatal("success text must not fabricate a failure payload")
	}
}
