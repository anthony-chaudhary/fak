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
