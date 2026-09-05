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
		wantTail   bool
		wantName   bool
		wantOK     bool
	}{
		// cat tests
		{"cat file.txt", "cat", "file.txt", 0, false, false, false, true},
		{"cat -n file.txt", "cat", "file.txt", 0, true, false, false, true},
		{"cat --number file.txt", "cat", "file.txt", 0, true, false, false, true},
		{"cat \"my file.txt\"", "cat", "my file.txt", 0, false, false, false, true},
		{"cat 'quoted/path.go'", "cat", "quoted/path.go", 0, false, false, false, true},
		{"cat -v file.txt", "", "", 0, false, false, false, false},
		{"cat a.txt b.txt", "", "", 0, false, false, false, false},
		{"cat", "", "", 0, false, false, false, false},

		// head tests
		{"head file.txt", "head", "file.txt", 10, false, false, false, true},
		{"head -n 25 file.txt", "head", "file.txt", 25, false, false, false, true},
		{"head -n25 file.txt", "head", "file.txt", 25, false, false, false, true},
		{"head -25 file.txt", "head", "file.txt", 25, false, false, false, true},
		{"head -n +5 file.txt", "head", "file.txt", 5, false, false, false, true},
		{"head \"file with space.txt\"", "head", "file with space.txt", 10, false, false, false, true},
		{"head -x file.txt", "", "", 0, false, false, false, false},
		{"head", "", "", 0, false, false, false, false},

		// tail tests
		{"tail file.txt", "tail", "file.txt", 10, false, false, false, true},
		{"tail -n 50 file.txt", "tail", "file.txt", 50, false, false, false, true},
		{"tail -n50 file.txt", "tail", "file.txt", 50, false, false, false, true},
		{"tail -50 file.txt", "tail", "file.txt", 50, false, false, false, true},
		{"tail \"deep/path.md\"", "tail", "deep/path.md", 10, false, false, false, true},

		// PowerShell Get-Content & gc tests
		{"Get-Content file.txt", "get-content", "file.txt", 0, false, false, false, true},
		{"Get-Content \"file with space.txt\"", "get-content", "file with space.txt", 0, false, false, false, true},
		{"Get-Content -Path file.txt", "get-content", "file.txt", 0, false, false, false, true},
		{"Get-Content -LiteralPath 'C:\\work\\fak\\file.txt'", "get-content", "C:\\work\\fak\\file.txt", 0, false, false, false, true},
		{"Get-Content -LiteralPath 'C:\\work\\fak\\my file.txt'", "get-content", "C:\\work\\fak\\my file.txt", 0, false, false, false, true},
		{"Get-Content -Path:file.txt -TotalCount 20", "get-content", "file.txt", 20, false, false, false, true},
		{"Get-Content -Path:\"my file.txt\"", "get-content", "my file.txt", 0, false, false, false, true},
		{"Get-Content -Head 15 file.txt", "get-content", "file.txt", 15, false, false, false, true},
		{"Get-Content -First 12 file.txt", "get-content", "file.txt", 12, false, false, false, true},
		{"Get-Content -Tail 10 file.txt", "get-content", "file.txt", 10, false, true, false, true},
		{"Get-Content -Last 5 file.txt", "get-content", "file.txt", 5, false, true, false, true},
		{"Get-Content -Tail:7 file.txt", "get-content", "file.txt", 7, false, true, false, true},
		{"Get-Content -Raw file.txt", "get-content", "file.txt", 0, false, false, false, true},
		{"gc file.txt", "get-content", "file.txt", 0, false, false, false, true},
		{"gc -TotalCount 5 file.txt", "get-content", "file.txt", 5, false, false, false, true},
		{"gc -Tail 3 file.txt", "get-content", "file.txt", 3, false, true, false, true},
		{"Get-Content -TotalCount -1 file.txt", "", "", 0, false, false, false, false},
		{"Get-Content -TotalCount:abc file.txt", "", "", 0, false, false, false, false},
		{"Get-Content -Tail -5 file.txt", "", "", 0, false, false, false, false},

		// type tests
		{"type file.txt", "type", "file.txt", 0, false, false, false, true},
		{"type -TotalCount 8 file.txt", "type", "file.txt", 8, false, false, false, true},
		{"type -Tail 4 file.txt", "type", "file.txt", 4, false, true, false, true},
		{"type a.txt b.txt", "", "", 0, false, false, false, false},

		// Get-ChildItem & gci & dir tests
		{"Get-ChildItem", "get-childitem", ".", 0, false, false, false, true},
		{"Get-ChildItem \"dir with space\"", "get-childitem", "dir with space", 0, false, false, false, true},
		{"Get-ChildItem -Path C:\\work\\fak", "get-childitem", "C:\\work\\fak", 0, false, false, false, true},
		{"Get-ChildItem -Path:C:\\work\\fak", "get-childitem", "C:\\work\\fak", 0, false, false, false, true},
		{"Get-ChildItem -LiteralPath C:\\work\\fak", "get-childitem", "C:\\work\\fak", 0, false, false, false, true},
		{"Get-ChildItem -Name", "get-childitem", ".", 0, false, false, true, true},
		{"Get-ChildItem -Path C:\\work\\fak -Name", "get-childitem", "C:\\work\\fak", 0, false, false, true, true},
		{"gci", "get-childitem", ".", 0, false, false, false, true},
		{"gci -Path C:\\test -Name", "get-childitem", "C:\\test", 0, false, false, true, true},
		{"dir", "get-childitem", ".", 0, false, false, false, true},
		{"dir -Name", "get-childitem", ".", 0, false, false, true, true},
		{"dir C:\\test", "get-childitem", "C:\\test", 0, false, false, false, true},

		// Disallowed / chaining / redirection tests
		{"cat file.txt > out.txt", "", "", 0, false, false, false, false},
		{"cat file.txt | grep foo", "", "", 0, false, false, false, false},
		{"cat file.txt; rm file.txt", "", "", 0, false, false, false, false},
		{"cat file.txt && echo done", "", "", 0, false, false, false, false},
		{"head `which cat`", "", "", 0, false, false, false, false},
		{"tail file.txt $(whoami)", "", "", 0, false, false, false, false},
		{"echo hello", "", "", 0, false, false, false, false},
		{"", "", "", 0, false, false, false, false},
		{"Get-Content -Wait file.txt", "", "", 0, false, false, false, false},
		{"Get-Content -Stream file.txt", "", "", 0, false, false, false, false},
		{"Get-Content -Filter *.go file.txt", "", "", 0, false, false, false, false},
		{"Get-Content *.txt", "", "", 0, false, false, false, false},
		{"Get-Content", "", "", 0, false, false, false, false},
		{"Get-ChildItem -Recurse", "", "", 0, false, false, false, false},
		{"Get-ChildItem -r", "", "", 0, false, false, false, false},
		{"Get-ChildItem -s", "", "", 0, false, false, false, false},
		{"Get-ChildItem *.go", "", "", 0, false, false, false, false},
		{"dir -Recurse", "", "", 0, false, false, false, false},
		{"gci -r", "", "", 0, false, false, false, false},
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
			if spec.Tail != tc.wantTail {
				t.Errorf("Tail = %v, want %v", spec.Tail, tc.wantTail)
			}
			if spec.NameOnly != tc.wantName {
				t.Errorf("NameOnly = %v, want %v", spec.NameOnly, tc.wantName)
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

	t.Run("get-content full file", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-content", FilePath: filePath}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d, stderr: %s", res.ExitCode, res.Stderr)
		}
		if res.Stdout != content {
			t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, content)
		}
	})

	t.Run("get-content -TotalCount 2", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-content", FilePath: filePath, Lines: 2, HasLines: true}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
		}
		want := "line 1: alpha\nline 2: beta\n"
		if res.Stdout != want {
			t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, want)
		}
	})

	t.Run("get-content -TotalCount 0", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-content", FilePath: filePath, Lines: 0, HasLines: true}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
		}
		if res.Stdout != "" {
			t.Fatalf("expected empty stdout for count 0, got %q", res.Stdout)
		}
	})

	t.Run("get-content -Tail 2", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-content", FilePath: filePath, Lines: 2, Tail: true}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
		}
		want := "line 4: delta\nline 5: epsilon\n"
		if res.Stdout != want {
			t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, want)
		}
	})

	t.Run("type full file", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "type", FilePath: filePath}, "")
		if res.ExitCode != 0 || res.Stdout != content {
			t.Fatalf("type failed: code=%d, stdout=%q", res.ExitCode, res.Stdout)
		}
	})

	t.Run("type with lines", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "type", FilePath: filePath, Lines: 2}, "")
		if res.ExitCode != 0 {
			t.Fatalf("type failed: code=%d", res.ExitCode)
		}
		want := "line 1: alpha\nline 2: beta\n"
		if res.Stdout != want {
			t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, want)
		}
	})

	t.Run("get-childitem default directory", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-childitem", FilePath: dir}, "")
		if res.ExitCode != 0 {
			t.Fatalf("get-childitem failed: code=%d, stderr=%s", res.ExitCode, res.Stderr)
		}
		if !strings.Contains(res.Stdout, "Mode") || !strings.Contains(res.Stdout, "LastWriteTime") ||
			!strings.Contains(res.Stdout, "Length") || !strings.Contains(res.Stdout, "Name") {
			t.Fatalf("missing table header in get-childitem output: %s", res.Stdout)
		}
		if !strings.Contains(res.Stdout, "sample.txt") {
			t.Fatalf("expected sample.txt in directory output: %s", res.Stdout)
		}
	})

	t.Run("get-childitem -Path file", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-childitem", FilePath: filePath}, "")
		if res.ExitCode != 0 {
			t.Fatalf("get-childitem file failed: code=%d", res.ExitCode)
		}
		if !strings.Contains(res.Stdout, "sample.txt") || !strings.Contains(res.Stdout, "Mode") {
			t.Fatalf("unexpected file format: %s", res.Stdout)
		}
	})

	t.Run("get-childitem -Name", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-childitem", FilePath: dir, NameOnly: true}, "")
		if res.ExitCode != 0 {
			t.Fatalf("get-childitem -Name failed: code=%d", res.ExitCode)
		}
		if res.Stdout != "sample.txt\n" {
			t.Fatalf("expected exact filename with newline: got %q, want %q", res.Stdout, "sample.txt\n")
		}
	})

	t.Run("get-childitem missing path", func(t *testing.T) {
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-childitem", FilePath: filepath.Join(dir, "not_exist")}, "")
		if res.ExitCode != 1 {
			t.Fatalf("expected ExitCode 1 for missing path, got %d", res.ExitCode)
		}
		if !strings.Contains(res.Stderr, "Cannot find path") {
			t.Fatalf("expected Cannot find path error, got: %s", res.Stderr)
		}
	})

	t.Run("get-childitem empty directory", func(t *testing.T) {
		emptyDir := t.TempDir()
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-childitem", FilePath: emptyDir}, "")
		if res.ExitCode != 0 {
			t.Fatalf("get-childitem failed on empty dir: %v", res.Stderr)
		}
		if !strings.Contains(res.Stdout, "Mode") {
			t.Fatalf("expected header in empty dir: %s", res.Stdout)
		}
	})

	t.Run("get-childitem empty directory name only", func(t *testing.T) {
		emptyDir := t.TempDir()
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "get-childitem", FilePath: emptyDir, NameOnly: true}, "")
		if res.ExitCode != 0 {
			t.Fatalf("get-childitem failed on empty dir name only: %v", res.Stderr)
		}
		if res.Stdout != "" {
			t.Fatalf("expected empty stdout for empty dir name only, got: %q", res.Stdout)
		}
	})

	t.Run("compound parallel execution returning combined outputs in order", func(t *testing.T) {
		f1 := filepath.Join(dir, "f1.txt")
		f2 := filepath.Join(dir, "f2.txt")
		f3 := filepath.Join(dir, "f3.txt")
		_ = os.WriteFile(f1, []byte("alpha\n"), 0644)
		_ = os.WriteFile(f2, []byte("beta\n"), 0644)
		_ = os.WriteFile(f3, []byte("gamma\n"), 0644)

		spec := &ShellReadSpec{
			Op: "compound",
			SubSpecs: []*ShellReadSpec{
				{Op: "cat", FilePath: f1},
				{Op: "cat", FilePath: f2},
				{Op: "cat", FilePath: f3},
			},
			Connectors: []string{";", ";"},
		}

		res := ExecuteInProcessRead(spec, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d, stderr: %s", res.ExitCode, res.Stderr)
		}
		want := "alpha\nbeta\ngamma\n"
		if res.Stdout != want {
			t.Fatalf("stdout mismatch: got %q, want %q", res.Stdout, want)
		}
	})

	t.Run("compound && short-circuit on non-existent file in first segment", func(t *testing.T) {
		fGood := filepath.Join(dir, "good.txt")
		_ = os.WriteFile(fGood, []byte("should not be read on short-circuit\n"), 0644)

		spec := &ShellReadSpec{
			Op: "compound",
			SubSpecs: []*ShellReadSpec{
				{Op: "cat", FilePath: filepath.Join(dir, "nonexistent.txt")},
				{Op: "cat", FilePath: fGood},
			},
			Connectors: []string{"&&"},
		}

		res := ExecuteInProcessRead(spec, "")
		if res.ExitCode == 0 {
			t.Fatalf("expected non-zero ExitCode, got 0")
		}
		if !strings.Contains(res.Stderr, "No such file or directory") {
			t.Fatalf("expected Stderr to contain 'No such file or directory', got: %s", res.Stderr)
		}
		if res.Stdout != "" {
			t.Fatalf("expected empty Stdout due to short-circuit, got %q", res.Stdout)
		}
	})

	t.Run("compound ; does not short-circuit on failure in first segment", func(t *testing.T) {
		fGood := filepath.Join(dir, "good_semi.txt")
		_ = os.WriteFile(fGood, []byte("read after failure\n"), 0644)

		spec := &ShellReadSpec{
			Op: "compound",
			SubSpecs: []*ShellReadSpec{
				{Op: "cat", FilePath: filepath.Join(dir, "nonexistent2.txt")},
				{Op: "cat", FilePath: fGood},
			},
			Connectors: []string{";"},
		}

		res := ExecuteInProcessRead(spec, "")
		if res.ExitCode == 0 {
			t.Fatalf("expected non-zero ExitCode due to first command failure, got 0")
		}
		if !strings.Contains(res.Stderr, "No such file or directory") {
			t.Fatalf("expected Stderr to contain 'No such file or directory', got: %s", res.Stderr)
		}
		if res.Stdout != "read after failure\n" {
			t.Fatalf("expected Stdout from second command, got %q", res.Stdout)
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

	// Test exec_command promotion with workdir extraction
	execCall := &abi.ToolCall{
		Tool: "exec_command",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"cmd":"Get-Content greet.txt","workdir":"%s"}`, filepath.ToSlash(dir))),
		},
	}
	resExec, okExec := PromoteInProcessRead(execCall, "")
	if !okExec || resExec == nil {
		t.Fatalf("expected exec_command with workdir to be promoted")
	}
	if resExec.Meta["in_process_op"] != "get-content" {
		t.Fatalf("expected in_process_op get-content, got %s", resExec.Meta["in_process_op"])
	}

	// Test functions.exec_command tool
	fnCall := &abi.ToolCall{
		Tool: "functions.exec_command",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"command":"cat %s"}`, filepath.ToSlash(p))),
		},
	}
	if _, okFn := PromoteInProcessRead(fnCall, ""); !okFn {
		t.Fatalf("expected functions.exec_command to be promotable")
	}

	// Test powershell tool
	psCall := &abi.ToolCall{
		Tool: "powershell",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"command":"Get-Content %s"}`, filepath.ToSlash(p))),
		},
	}
	if _, okPs := PromoteInProcessRead(psCall, ""); !okPs {
		t.Fatalf("expected powershell to be promotable")
	}

	// Test pwsh tool
	pwshCall := &abi.ToolCall{
		Tool: "pwsh",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"command":"Get-Content %s"}`, filepath.ToSlash(p))),
		},
	}
	if _, okPwsh := PromoteInProcessRead(pwshCall, ""); !okPwsh {
		t.Fatalf("expected pwsh to be promotable")
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

func BenchmarkInProcessRead_GetContent(b *testing.B) {
	dir := b.TempDir()
	filePath := filepath.Join(dir, "bench_sample.txt")
	lines := make([]string, 100)
	for i := range lines {
		lines[i] = fmt.Sprintf("bench line %d content payload", i)
	}
	if err := os.WriteFile(filePath, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		b.Fatalf("WriteFile failed: %v", err)
	}

	spec := &ShellReadSpec{
		Op:       "get-content",
		FilePath: filePath,
		Lines:    10,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := ExecuteInProcessRead(spec, "")
		if res.ExitCode != 0 {
			b.Fatalf("ExecuteInProcessRead failed: %s", res.Stderr)
		}
	}
}

func TestSplitCompoundCommand(t *testing.T) {
	t.Run("splitting on semicolon, double-ampersand, and newline", func(t *testing.T) {
		tests := []struct {
			name           string
			cmd            string
			wantSegments   []string
			wantConnectors []string
		}{
			{
				name:           "semicolon separation",
				cmd:            "cat file1.txt; cat file2.txt",
				wantSegments:   []string{"cat file1.txt", "cat file2.txt"},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "double-ampersand separation",
				cmd:            "Get-Content A.txt && Get-Content B.txt",
				wantSegments:   []string{"Get-Content A.txt", "Get-Content B.txt"},
				wantConnectors: []string{"&&", ""},
			},
			{
				name:           "newline separation",
				cmd:            "cat file1.txt\ncat file2.txt",
				wantSegments:   []string{"cat file1.txt", "cat file2.txt"},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "semicolon followed by newline",
				cmd:            "cat file1.txt;\ncat file2.txt",
				wantSegments:   []string{"cat file1.txt", "cat file2.txt"},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "trailing semicolon",
				cmd:            "cat file1.txt; cat file2.txt;",
				wantSegments:   []string{"cat file1.txt", "cat file2.txt"},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "trailing newline",
				cmd:            "cat file1.txt\ncat file2.txt\n",
				wantSegments:   []string{"cat file1.txt", "cat file2.txt"},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "three commands with mixed connectors",
				cmd:            "cat A.txt && head -n 5 B.txt; tail C.txt",
				wantSegments:   []string{"cat A.txt", "head -n 5 B.txt", "tail C.txt"},
				wantConnectors: []string{"&&", ";", ""},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				segs, ok := SplitCompoundCommand(tc.cmd)
				if !ok {
					t.Fatalf("SplitCompoundCommand(%q) returned ok=false", tc.cmd)
				}
				if len(segs) != len(tc.wantSegments) {
					t.Fatalf("got %d segments, want %d", len(segs), len(tc.wantSegments))
				}
				for i, seg := range segs {
					if seg.RawCmd != tc.wantSegments[i] {
						t.Errorf("seg[%d].RawCmd = %q, want %q", i, seg.RawCmd, tc.wantSegments[i])
					}
					if seg.Connector != tc.wantConnectors[i] {
						t.Errorf("seg[%d].Connector = %q, want %q", i, seg.Connector, tc.wantConnectors[i])
					}
				}
			})
		}
	})

	t.Run("preserving semicolons and ampersands inside quotes", func(t *testing.T) {
		tests := []struct {
			name           string
			cmd            string
			wantSegments   []string
			wantConnectors []string
		}{
			{
				name:           "semicolon in double quotes",
				cmd:            `cat "file;1.txt"`,
				wantSegments:   []string{`cat "file;1.txt"`},
				wantConnectors: []string{""},
			},
			{
				name:           "semicolon in single quotes",
				cmd:            `cat 'file;1.txt'`,
				wantSegments:   []string{`cat 'file;1.txt'`},
				wantConnectors: []string{""},
			},
			{
				name:           "double-ampersand in double quotes",
				cmd:            `cat "file&&1.txt"`,
				wantSegments:   []string{`cat "file&&1.txt"`},
				wantConnectors: []string{""},
			},
			{
				name:           "double-ampersand in single quotes",
				cmd:            `cat 'file&&1.txt'`,
				wantSegments:   []string{`cat 'file&&1.txt'`},
				wantConnectors: []string{""},
			},
			{
				name:           "semicolon in quotes chained with another command",
				cmd:            `cat "file;1.txt"; cat 'file&&2.txt'`,
				wantSegments:   []string{`cat "file;1.txt"`, `cat 'file&&2.txt'`},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "escaped double quote inside double quotes",
				cmd:            `cat "quoted\"word.txt"; cat other.txt`,
				wantSegments:   []string{`cat "quoted\"word.txt"`, `cat other.txt`},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "doubled single quote inside single quotes",
				cmd:            `cat 'it''s.txt'; cat other.txt`,
				wantSegments:   []string{`cat 'it''s.txt'`, `cat other.txt`},
				wantConnectors: []string{";", ""},
			},
			{
				name:           "doubled double quote inside double quotes",
				cmd:            `cat "it""s.txt"; cat other.txt`,
				wantSegments:   []string{`cat "it""s.txt"`, `cat other.txt`},
				wantConnectors: []string{";", ""},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				segs, ok := SplitCompoundCommand(tc.cmd)
				if !ok {
					t.Fatalf("SplitCompoundCommand(%q) returned ok=false", tc.cmd)
				}
				if len(segs) != len(tc.wantSegments) {
					t.Fatalf("got %d segments, want %d", len(segs), len(tc.wantSegments))
				}
				for i, seg := range segs {
					if seg.RawCmd != tc.wantSegments[i] {
						t.Errorf("seg[%d].RawCmd = %q, want %q", i, seg.RawCmd, tc.wantSegments[i])
					}
					if seg.Connector != tc.wantConnectors[i] {
						t.Errorf("seg[%d].Connector = %q, want %q", i, seg.Connector, tc.wantConnectors[i])
					}
				}
			})
		}
	})

	t.Run("rejecting pipes, redirects, command substitution and invalid syntax", func(t *testing.T) {
		disallowed := []string{
			"cat a.txt | grep foo",
			"cat a.txt || cat b.txt",
			"cat a.txt > out.txt",
			"cat a.txt >> out.txt",
			"cat a.txt < in.txt",
			"cat $(whoami)",
			`cat "$(whoami)"`,
			"cat 'unterminated",
			`cat "unterminated`,
			"cat a.txt &",
			"cat a.txt &&& cat b.txt",
			"cat a.txt ;; cat b.txt",
		}

		for _, cmd := range disallowed {
			t.Run(cmd, func(t *testing.T) {
				segs, ok := SplitCompoundCommand(cmd)
				if ok {
					t.Fatalf("SplitCompoundCommand(%q) returned ok=true, segments=%+v", cmd, segs)
				}
			})
		}
	})
}

func TestParseShellRead_CompoundCommands(t *testing.T) {
	t.Run("all-read commands succeed", func(t *testing.T) {
		tests := []struct {
			cmd            string
			wantSubOps     []string
			wantConnectors []string
		}{
			{
				cmd:            "cat file1.txt; cat file2.txt",
				wantSubOps:     []string{"cat", "cat"},
				wantConnectors: []string{";"},
			},
			{
				cmd:            "Get-Content A.txt && Get-Content B.txt",
				wantSubOps:     []string{"get-content", "get-content"},
				wantConnectors: []string{"&&"},
			},
			{
				cmd:            "Get-Content A; Get-ChildItem B",
				wantSubOps:     []string{"get-content", "get-childitem"},
				wantConnectors: []string{";"},
			},
			{
				cmd:            "head -n 5 file1.txt && tail -n 10 file2.txt; gc file3.txt",
				wantSubOps:     []string{"head", "tail", "get-content"},
				wantConnectors: []string{"&&", ";"},
			},
		}

		for _, tc := range tests {
			t.Run(tc.cmd, func(t *testing.T) {
				spec, ok := ParseShellRead(tc.cmd)
				if !ok || spec == nil {
					t.Fatalf("ParseShellRead(%q) ok=%v, spec=%v", tc.cmd, ok, spec)
				}
				if spec.Op != "compound" {
					t.Fatalf("Op = %q, want compound", spec.Op)
				}
				if len(spec.SubSpecs) != len(tc.wantSubOps) {
					t.Fatalf("got %d SubSpecs, want %d", len(spec.SubSpecs), len(tc.wantSubOps))
				}
				for i, wantOp := range tc.wantSubOps {
					if spec.SubSpecs[i].Op != wantOp {
						t.Errorf("SubSpecs[%d].Op = %q, want %q", i, spec.SubSpecs[i].Op, wantOp)
					}
				}
				if len(spec.Connectors) != len(tc.wantConnectors) {
					t.Fatalf("got %d Connectors, want %d", len(spec.Connectors), len(tc.wantConnectors))
				}
				for i, wantConn := range tc.wantConnectors {
					if spec.Connectors[i] != wantConn {
						t.Errorf("Connectors[%d] = %q, want %q", i, spec.Connectors[i], wantConn)
					}
				}
			})
		}
	})

	t.Run("mixed read and write commands return ok=false", func(t *testing.T) {
		tests := []string{
			"cat file1.txt; rm file2.txt",
			"Get-Content A.txt && Remove-Item B.txt",
			"cat file1.txt && echo hi",
			"cat file1.txt; touch file2.txt",
			"rm file.txt; cat file2.txt",
			"Get-Content A.txt && Set-Content B.txt hello",
		}

		for _, cmd := range tests {
			t.Run(cmd, func(t *testing.T) {
				spec, ok := ParseShellRead(cmd)
				if ok || spec != nil {
					t.Fatalf("ParseShellRead(%q) expected ok=false, got ok=%v, spec=%+v", cmd, ok, spec)
				}
			})
		}
	})
}

func BenchmarkInProcessRead_CompoundParallel(b *testing.B) {
	dir := b.TempDir()
	files := make([]string, 4)
	for i := 0; i < 4; i++ {
		p := filepath.Join(dir, fmt.Sprintf("bench_part_%d.txt", i))
		lines := make([]string, 100)
		for j := range lines {
			lines[j] = fmt.Sprintf("part %d line %d benchmark payload", i, j)
		}
		if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
			b.Fatalf("WriteFile failed: %v", err)
		}
		files[i] = p
	}

	spec := &ShellReadSpec{
		Op: "compound",
		SubSpecs: []*ShellReadSpec{
			{Op: "get-content", FilePath: files[0], Lines: 10},
			{Op: "get-content", FilePath: files[1], Lines: 10},
			{Op: "get-content", FilePath: files[2], Lines: 10},
			{Op: "get-content", FilePath: files[3], Lines: 10},
		},
		Connectors: []string{";", ";", ";"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := ExecuteInProcessRead(spec, "")
		if res.ExitCode != 0 {
			b.Fatalf("ExecuteInProcessRead failed: %s", res.Stderr)
		}
	}
}
