package vdso

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestParseShellRead(t *testing.T) {
	tests := []struct {
		cmd        string
		wantOp     string
		wantPath   string
		wantLines  int
		wantNumber bool
		wantOK     bool
	}{
		// cat tests
		{"cat file.txt", "cat", "file.txt", 0, false, true},
		{"cat -n file.txt", "cat", "file.txt", 0, true, true},
		{"cat --number file.txt", "cat", "file.txt", 0, true, true},
		{"cat \"my file.txt\"", "cat", "my file.txt", 0, false, true},
		{"cat 'quoted/path.go'", "cat", "quoted/path.go", 0, false, true},
		{"cat -v file.txt", "", "", 0, false, false},
		{"cat a.txt b.txt", "", "", 0, false, false},
		{"cat", "", "", 0, false, false},

		// head tests
		{"head file.txt", "head", "file.txt", 10, false, true},
		{"head -n 25 file.txt", "head", "file.txt", 25, false, true},
		{"head -n25 file.txt", "head", "file.txt", 25, false, true},
		{"head -25 file.txt", "head", "file.txt", 25, false, true},
		{"head -n +5 file.txt", "head", "file.txt", 5, false, true},
		{"head \"file with space.txt\"", "head", "file with space.txt", 10, false, true},
		{"head -x file.txt", "", "", 0, false, false},
		{"head", "", "", 0, false, false},

		// tail tests
		{"tail file.txt", "tail", "file.txt", 10, false, true},
		{"tail -n 50 file.txt", "tail", "file.txt", 50, false, true},
		{"tail -n50 file.txt", "tail", "file.txt", 50, false, true},
		{"tail -50 file.txt", "tail", "file.txt", 50, false, true},
		{"tail \"deep/path.md\"", "tail", "deep/path.md", 10, false, true},

		// Disallowed / chaining / redirection tests
		{"cat file.txt > out.txt", "", "", 0, false, false},
		{"cat file.txt | grep foo", "", "", 0, false, false},
		{"cat file.txt; rm file.txt", "", "", 0, false, false},
		{"cat file.txt && echo done", "", "", 0, false, false},
		{"head `which cat`", "", "", 0, false, false},
		{"tail file.txt $(whoami)", "", "", 0, false, false},
		{"echo hello", "", "", 0, false, false},
		{"", "", "", 0, false, false},
	}

	for _, tc := range tests {
		t.Run(tc.cmd, func(t *testing.T) {
			spec, ok := ParseShellRead(tc.cmd)
			if ok != tc.wantOK {
				t.Fatalf("ParseShellRead(%q) ok = %v, want %v", tc.cmd, ok, tc.wantOK)
			}
			if !ok {
				return
			}
			if spec.Op != tc.wantOp {
				t.Errorf("Op = %q, want %q", spec.Op, tc.wantOp)
			}
			if spec.FilePath != tc.wantPath {
				t.Errorf("FilePath = %q, want %q", spec.FilePath, tc.wantPath)
			}
			if spec.Lines != tc.wantLines {
				t.Errorf("Lines = %d, want %d", spec.Lines, tc.wantLines)
			}
			if spec.LineNumbers != tc.wantNumber {
				t.Errorf("LineNumbers = %v, want %v", spec.LineNumbers, tc.wantNumber)
			}
		})
	}
}

func TestExecuteInProcessRead(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "sample.txt")

	lines := []string{
		"line 1: alpha",
		"line 2: beta",
		"line 3: gamma",
		"line 4: delta",
		"line 5: epsilon",
	}
	content := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	t.Run("cat full file", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "cat", FilePath: filePath}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d, stderr: %s", res.ExitCode, res.Stderr)
		}
		if res.Stdout != content {
			t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, content)
		}
	})

	t.Run("cat with line numbers", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "cat", FilePath: filePath, LineNumbers: true}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
		}
		if !strings.Contains(res.Stdout, "     1  line 1: alpha") {
			t.Fatalf("expected line numbers in stdout, got:\n%s", res.Stdout)
		}
		if !strings.Contains(res.Stdout, "     5  line 5: epsilon") {
			t.Fatalf("expected line 5 in stdout, got:\n%s", res.Stdout)
		}
	})

	t.Run("head 2 lines", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "head", FilePath: filePath, Lines: 2}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
		}
		want := "line 1: alpha\nline 2: beta\n"
		if res.Stdout != want {
			t.Fatalf("head stdout mismatch: got %q, want %q", res.Stdout, want)
		}
	})

	t.Run("tail 2 lines", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "tail", FilePath: filePath, Lines: 2}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
		}
		want := "line 4: delta\nline 5: epsilon\n"
		if res.Stdout != want {
			t.Fatalf("tail stdout mismatch: got %q, want %q", res.Stdout, want)
		}
	})

	t.Run("relative path with workDir", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "cat", FilePath: "sample.txt"}, dir)
		if res.ExitCode != 0 || res.Stdout != content {
			t.Fatalf("relative path failed: code=%d, stdout=%q, stderr=%q", res.ExitCode, res.Stdout, res.Stderr)
		}
	})

	t.Run("non-existent file", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "cat", FilePath: "missing.txt"}, dir)
		if res.ExitCode != 1 {
			t.Fatalf("expected ExitCode 1, got %d", res.ExitCode)
		}
		if !strings.Contains(res.Stderr, "No such file or directory") {
			t.Fatalf("expected No such file or directory, got: %s", res.Stderr)
		}
	})

	t.Run("directory path", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "cat", FilePath: dir}, "")
		if res.ExitCode != 1 {
			t.Fatalf("expected ExitCode 1, got %d", res.ExitCode)
		}
		if !strings.Contains(res.Stderr, "Is a directory") {
			t.Fatalf("expected Is a directory error, got: %s", res.Stderr)
		}
	})
}

func TestPromoteInProcessRead(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "greet.txt")
	_ = os.WriteFile(p, []byte("hello in-process world\n"), 0644)

	call := &abi.ToolCall{
		Tool: "Bash",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"command":"cat %s"}`, filepath.ToSlash(p))),
		},
	}

	res, ok := PromoteInProcessRead(call, "")
	if !ok || res == nil {
		t.Fatalf("expected PromoteInProcessRead to return true and non-nil result")
	}

	if res.Status != abi.StatusOK {
		t.Fatalf("expected StatusOK, got %v", res.Status)
	}
	if res.Meta["served_by"] != "in_process_read" {
		t.Fatalf("meta served_by = %q, want in_process_read", res.Meta["served_by"])
	}
	if res.Meta["promoted"] != "true" {
		t.Fatalf("meta promoted = %q, want true", res.Meta["promoted"])
	}

	var bashRes ShellReadResult
	if err := json.Unmarshal(res.Payload.Inline, &bashRes); err != nil {
		t.Fatalf("failed to unmarshal ShellReadResult: %v", err)
	}
	if bashRes.ExitCode != 0 || bashRes.Stdout != "hello in-process world\n" {
		t.Fatalf("unexpected bash result: %+v", bashRes)
	}

	// Non-promotable call returns false
	nonCall := &abi.ToolCall{
		Tool: "Bash",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
	}
	if _, ok := PromoteInProcessRead(nonCall, ""); ok {
		t.Fatalf("mutating command rm must not be promotable")
	}

	// Non-bash tool returns false
	otherCall := &abi.ToolCall{
		Tool: "Edit",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"cat greet.txt"}`)},
	}
	if _, ok := PromoteInProcessRead(otherCall, ""); ok {
		t.Fatalf("non-bash tool must not be promotable")
	}
}
