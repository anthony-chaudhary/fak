package codetools

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestGoTestDoesNotLeaveBinariesInPackageTree(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not available on path")
	}

	root := t.TempDir()
	gitignore := "/_scratch/\n/_*\n"
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte(gitignore), 0o644); err != nil {
		t.Fatal(err)
	}
	gomod := "module testcontainment\n\ngo 1.26\n"
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte(gomod), 0o644); err != nil {
		t.Fatal(err)
	}
	src := "package testcontainment\n\nfunc Answer() int { return 42 }\n"
	if err := os.WriteFile(filepath.Join(root, "pkg.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	testSrc := "package testcontainment\n\nimport \"testing\"\n\nfunc TestAnswer(t *testing.T) { if Answer() != 42 { t.Fatal() } }\n"
	if err := os.WriteFile(filepath.Join(root, "pkg_test.go"), []byte(testSrc), 0o644); err != nil {
		t.Fatal(err)
	}

	hasGit := false
	if _, err := exec.LookPath("git"); err == nil {
		initCmd := exec.Command("git", "init", "-q", root)
		if err := initCmd.Run(); err == nil {
			addCmd := exec.Command("git", "-C", root, "add", ".gitignore", "go.mod", "pkg.go", "pkg_test.go")
			if err := addCmd.Run(); err == nil {
				commitCmd := exec.Command("git", "-C", root, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init", "-q")
				if err := commitCmd.Run(); err == nil {
					hasGit = true
				}
			}
		}
	}

	ts, err := New(Config{Root: root, FocusedCommands: false})
	if err != nil {
		t.Fatal(err)
	}

	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "go test -c", Cwd: "."}))
	if bad {
		t.Fatalf("bash returned refusal: %s", out)
	}
	res := decodeResult(t, out)
	if res["exit_code"].(float64) != 0 {
		t.Fatalf("go test -c failed: stderr=%s stdout=%s", res["stderr"], res["stdout"])
	}

	// 1. Assert package tree (outside _scratch and .git) contains no test binaries
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		name := e.Name()
		if name == "_scratch" || name == ".git" {
			continue
		}
		lower := strings.ToLower(name)
		if strings.HasSuffix(lower, ".test") || strings.HasSuffix(lower, ".test.exe") {
			t.Fatalf("compiled binary %q leaked into package directory %s", name, root)
		}
	}

	// 2. Assert compiled binary is isolated in GOTMPDIR / scratch
	gotmpDir := filepath.Join(root, "_scratch", "gotmp")
	gotmpEntries, err := os.ReadDir(gotmpDir)
	if err != nil {
		t.Fatalf("cannot read gotmp dir %s: %v", gotmpDir, err)
	}
	foundBinary := false
	for _, e := range gotmpEntries {
		lower := strings.ToLower(e.Name())
		if strings.HasSuffix(lower, ".test") || strings.HasSuffix(lower, ".test.exe") {
			foundBinary = true
			break
		}
	}
	if !foundBinary {
		t.Fatalf("expected compiled test binary in gotmp dir %s, entries=%v", gotmpDir, gotmpEntries)
	}

	// 3. Assert git status --porcelain remains clean of untracked binary artifacts
	if hasGit {
		statusCmd := exec.Command("git", "-C", root, "status", "--porcelain")
		statusOut, err := statusCmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git status failed: %v: %s", err, statusOut)
		}
		for _, line := range strings.Split(string(statusOut), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			lower := strings.ToLower(line)
			if strings.Contains(lower, ".test") || strings.Contains(lower, ".test.exe") {
				t.Fatalf("untracked binary artifact in git status: %q", line)
			}
		}
	}
}

func TestGoTestCommandRewriterContainment(t *testing.T) {
	ts, root := newTestToolset(t)
	if err := os.WriteFile(filepath.Join(root, "pkg.go"), []byte("package codetools\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gotmpDir := ts.ensureGoTmpDir()

	tests := []struct {
		name     string
		input    string
		contains []string
	}{
		{
			name:     "go test -c without -o injects -o into gotmp",
			input:    "go test -c",
			contains: []string{"-o", filepath.Join(gotmpDir, "codetools.test"), "-c"},
		},
		{
			name:     "go test -c with bare -o redirects to gotmp",
			input:    "go test -c -o custom.test",
			contains: []string{"-c", "-o", filepath.Join(gotmpDir, "custom.test")},
		},
		{
			name:     "go test -c with bare -o= redirects to gotmp",
			input:    "go test -c -o=custom.test",
			contains: []string{"-c", "-o=" + filepath.Join(gotmpDir, "custom.test")},
		},
		{
			name:  "go test profiling flags bare paths redirect to gotmp",
			input: "go test -cpuprofile cpu.pprof -memprofile=mem.pprof -coverprofile cover.out",
			contains: []string{
				filepath.Join(gotmpDir, "cpu.pprof"),
				filepath.Join(gotmpDir, "mem.pprof"),
				filepath.Join(gotmpDir, "cover.out"),
			},
		},
		{
			name:     "chained command rewrites go test segment",
			input:    "echo start && go test -c",
			contains: []string{"echo start && go test -o", filepath.Join(gotmpDir, "codetools.test"), "-c"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rewritten := ts.rewriteCommandForContainment(tc.input, root, gotmpDir)
			for _, expected := range tc.contains {
				if !strings.Contains(rewritten, expected) {
					t.Fatalf("rewrite %q: got %q, missing %q", tc.input, rewritten, expected)
				}
			}
		})
	}
}

func TestGoTmpDirEnvEnforcement(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not available on path")
	}

	root := t.TempDir()
	ts, err := New(Config{Root: root, FocusedCommands: false})
	if err != nil {
		t.Fatal(err)
	}

	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "go env GOTMPDIR", Cwd: "."}))
	if bad {
		t.Fatalf("bash returned refusal: %s", out)
	}
	res := decodeResult(t, out)
	expectedGoTmp := filepath.Join(root, "_scratch", "gotmp")
	got := strings.TrimSpace(res["stdout"].(string))
	if !strings.EqualFold(got, expectedGoTmp) {
		t.Fatalf("go env GOTMPDIR = %q, want %q", got, expectedGoTmp)
	}
}

func TestPostExecutionContainmentCleanup(t *testing.T) {
	root := t.TempDir()
	ts, err := New(Config{Root: root, FocusedCommands: false})
	if err != nil {
		t.Fatal(err)
	}

	// Drop dummy test binary directly in root
	leakedBinary := filepath.Join(root, "leaked.test")
	if err := os.WriteFile(leakedBinary, []byte("fake-test-binary"), 0o755); err != nil {
		t.Fatal(err)
	}

	// Run any command through bash toolset
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashEcho("done"), Cwd: "."}))
	if bad {
		t.Fatalf("bash failed: %s", out)
	}

	// Leaked binary should be cleaned from root
	if _, err := os.Stat(leakedBinary); !os.IsNotExist(err) {
		t.Fatalf("leaked test binary still exists at %s", leakedBinary)
	}

	// Binary should be moved to GOTMPDIR
	movedBinary := filepath.Join(root, "_scratch", "gotmp", "leaked.test")
	if _, err := os.Stat(movedBinary); err != nil {
		t.Fatalf("leaked binary was not moved to %s: %v", movedBinary, err)
	}
}
