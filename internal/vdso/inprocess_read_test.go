package vdso

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

		// PowerShell / Windows inspection tests
		{`Get-Content -Path foo\bar.txt`, "cat", "foo/bar.txt", 0, false, true},
		{`Get-Content foo.txt -TotalCount 20`, "head", "foo.txt", 20, false, true},
		{`Get-Content foo.txt -Tail 50`, "tail", "foo.txt", 50, false, true},
		{`type foo.txt`, "cat", "foo.txt", 0, false, true},
		{`Get-ChildItem -Path dir`, "dir", "dir", 0, false, true},

		{`Get-Content foo\bar.txt`, "cat", "foo/bar.txt", 0, false, true},
		{`gc foo\bar.txt`, "cat", "foo/bar.txt", 0, false, true},
		{`gc foo.txt -TotalCount 10`, "head", "foo.txt", 10, false, true},
		{`gc foo.txt -Tail 5`, "tail", "foo.txt", 5, false, true},
		{`cat foo\bar.txt`, "cat", "foo/bar.txt", 0, false, true},
		{`cat -Path foo\bar.txt`, "cat", "foo/bar.txt", 0, false, true},
		{`cat foo.txt -TotalCount 20`, "head", "foo.txt", 20, false, true},
		{`cat foo.txt -Tail 50`, "tail", "foo.txt", 50, false, true},
		{`type foo\bar.txt`, "cat", "foo/bar.txt", 0, false, true},
		{`Get-ChildItem dir`, "dir", "dir", 0, false, true},
		{`Get-ChildItem -Path foo\bar`, "dir", "foo/bar", 0, false, true},
		{`gci -Path dir`, "dir", "dir", 0, false, true},
		{`dir foo\bar`, "dir", "foo/bar", 0, false, true},
		{`dir`, "dir", ".", 0, false, true},
		{`Get-ChildItem -Path dir -Recurse`, "", "", 0, false, false},
		{`Get-ChildItem -Recurse`, "", "", 0, false, false},
		{`Get-Content foo.txt -TotalCount 10 -Tail 5`, "", "", 0, false, false},

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

	t.Run("dir directory listing", func(t *testing.T) {
		subDir := filepath.Join(dir, "sublisting")
		_ = os.Mkdir(subDir, 0755)
		_ = os.WriteFile(filepath.Join(subDir, "file_a.txt"), []byte("a"), 0644)
		_ = os.WriteFile(filepath.Join(subDir, "file_b.txt"), []byte("b"), 0644)
		res := ExecuteInProcessRead(&ShellReadSpec{Op: "dir", FilePath: subDir}, "")
		if res.ExitCode != 0 {
			t.Fatalf("expected ExitCode 0, got %d, stderr: %s", res.ExitCode, res.Stderr)
		}
		if !strings.Contains(res.Stdout, "file_a.txt") || !strings.Contains(res.Stdout, "file_b.txt") {
			t.Fatalf("expected file_a.txt and file_b.txt in stdout, got:\n%s", res.Stdout)
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

	// exec_command tool returns true for Get-Content
	execCall := &abi.ToolCall{
		Tool: "exec_command",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"cmd":"Get-Content -Path %s"}`, filepath.ToSlash(p))),
		},
	}
	execRes, ok := PromoteInProcessRead(execCall, "")
	if !ok || execRes == nil {
		t.Fatalf("expected PromoteInProcessRead to return true for exec_command")
	}
	if execRes.Status != abi.StatusOK {
		t.Fatalf("expected StatusOK, got %v", execRes.Status)
	}
	if execRes.Meta["in_process_op"] != "cat" {
		t.Fatalf("meta in_process_op = %q, want cat", execRes.Meta["in_process_op"])
	}

	// PowerShell tool returns true for Get-Content with -TotalCount
	psCall := &abi.ToolCall{
		Tool: "PowerShell",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(fmt.Sprintf(`{"command":"Get-Content %s -TotalCount 20"}`, filepath.ToSlash(p))),
		},
	}
	psRes, ok := PromoteInProcessRead(psCall, "")
	if !ok || psRes == nil {
		t.Fatalf("expected PromoteInProcessRead to return true for PowerShell")
	}
	if psRes.Status != abi.StatusOK {
		t.Fatalf("expected StatusOK, got %v", psRes.Status)
	}
	if psRes.Meta["in_process_op"] != "head" {
		t.Fatalf("meta in_process_op = %q, want head", psRes.Meta["in_process_op"])
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

func TestPowerShellExecutionLatencyUnder10ms(t *testing.T) {
	dir := t.TempDir()
	fooPath := filepath.Join(dir, "foo.txt")
	_ = os.WriteFile(fooPath, []byte("line 1: alpha\nline 2: beta\nline 3: gamma\n"), 0644)
	subFoo := filepath.Join(dir, "foo")
	_ = os.Mkdir(subFoo, 0755)
	_ = os.WriteFile(filepath.Join(subFoo, "bar.txt"), []byte("nested content\n"), 0644)
	dirPath := filepath.Join(dir, "dir")
	_ = os.Mkdir(dirPath, 0755)
	_ = os.WriteFile(filepath.Join(dirPath, "child.txt"), []byte("child item\n"), 0644)

	commands := []struct {
		cmd      string
		wantOp   string
		wantExit int
	}{
		{`Get-Content -Path foo\bar.txt`, "cat", 0},
		{`Get-Content foo.txt -TotalCount 20`, "head", 0},
		{`Get-Content foo.txt -Tail 50`, "tail", 0},
		{`type foo.txt`, "cat", 0},
		{`Get-ChildItem -Path dir`, "dir", 0},
	}

	for _, tc := range commands {
		t.Run(tc.cmd, func(t *testing.T) {
			start := time.Now()
			spec, ok := ParseShellRead(tc.cmd)
			if !ok || spec == nil {
				t.Fatalf("ParseShellRead(%q) failed", tc.cmd)
			}
			if spec.Op != tc.wantOp {
				t.Errorf("Op = %q, want %q", spec.Op, tc.wantOp)
			}
			res := ExecuteInProcessRead(spec, dir)
			elapsed := time.Since(start)
			if res.ExitCode != tc.wantExit {
				t.Errorf("exit code = %d, want %d (stderr: %s)", res.ExitCode, tc.wantExit, res.Stderr)
			}
			if elapsed >= 10*time.Millisecond {
				t.Errorf("execution of %q took %v, want <10ms", tc.cmd, elapsed)
			}
		})
	}
}

func BenchmarkParseShellRead_PowerShell(b *testing.B) {
	commands := []string{
		`Get-Content -Path foo\bar.txt`,
		`Get-Content foo.txt -TotalCount 20`,
		`Get-Content foo.txt -Tail 50`,
		`type foo.txt`,
		`Get-ChildItem -Path dir`,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd := commands[i%len(commands)]
		spec, ok := ParseShellRead(cmd)
		if !ok || spec == nil {
			b.Fatalf("failed to parse %s", cmd)
		}
	}
}

func BenchmarkExecuteInProcessRead_PowerShell(b *testing.B) {
	dir := b.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "foo.txt"), []byte("line 1\nline 2\nline 3\n"), 0644)
	subFoo := filepath.Join(dir, "foo")
	_ = os.Mkdir(subFoo, 0755)
	_ = os.WriteFile(filepath.Join(subFoo, "bar.txt"), []byte("bar\n"), 0644)
	subDir := filepath.Join(dir, "dir")
	_ = os.Mkdir(subDir, 0755)
	_ = os.WriteFile(filepath.Join(subDir, "item.txt"), []byte("content\n"), 0644)

	commands := []string{
		`Get-Content -Path foo\bar.txt`,
		`Get-Content foo.txt -TotalCount 20`,
		`Get-Content foo.txt -Tail 50`,
		`type foo.txt`,
		`Get-ChildItem -Path dir`,
	}
	specs := make([]*ShellReadSpec, len(commands))
	for i, c := range commands {
		s, ok := ParseShellRead(c)
		if !ok {
			b.Fatalf("failed to parse: %s", c)
		}
		specs[i] = s
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spec := specs[i%len(specs)]
		res := ExecuteInProcessRead(spec, dir)
		if res.ExitCode != 0 {
			b.Fatalf("failed execution: %s", res.Stderr)
		}
	}
}
