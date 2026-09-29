package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// binPath is the compiled cmd/codesearch binary, built once in TestMain. The tests
// below drive it as a REAL process, so the delegation contract main() implements
// (os.Exit(codesearch.Run(os.Args[1:], os.Stdout, os.Stderr))) is witnessed
// end-to-end rather than inferred from the source.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "codesearch-bin-")
	if err != nil {
		panic("mkdtemp: " + err.Error())
	}
	name := "codesearch"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binPath = filepath.Join(dir, name)
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		os.RemoveAll(dir)
		panic("build cmd/codesearch: " + err.Error())
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// runBin executes the built binary and returns (exitCode, stdout, stderr).
func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	cmd := exec.Command(binPath, args...)
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if err != nil && cmd.ProcessState == nil {
		t.Fatalf("run %v: %v", args, err)
	}
	return cmd.ProcessState.ExitCode(), out.String(), errb.String()
}

// TestNoArgsPrintsUsageToStderr witnesses that main forwards os.Args[1:] (empty)
// to the engine, which emits the usage banner on stderr and exits 2.
func TestNoArgsPrintsUsageToStderr(t *testing.T) {
	code, out, errb := runBin(t)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (stderr=%q)", code, errb)
	}
	if !strings.Contains(errb, "usage: fak dev codesearch") {
		t.Errorf("stderr missing usage banner; got:\n%s", errb)
	}
	if out != "" {
		t.Errorf("usage must go to stderr, stdout got:\n%s", out)
	}
}

// TestUnknownSubcommandPrintsUsage witnesses the default branch: an unrecognized
// subcommand falls through to the usage banner and exit 2.
func TestUnknownSubcommandPrintsUsage(t *testing.T) {
	code, _, errb := runBin(t, "bogus")
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errb, "usage: fak dev codesearch") {
		t.Errorf("stderr missing usage banner; got:\n%s", errb)
	}
}

// TestLitDelegatesAndSearches witnesses the delegation contract positively: a real
// `lit` search over this very package reaches the filesystem and prints the file
// that defines main. This is the runtime proof that the binary runs.
func TestLitDelegatesAndSearches(t *testing.T) {
	code, out, errb := runBin(t, "lit", "--root", ".", "func main")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr=%q)", code, errb)
	}
	if !strings.Contains(out, "main.go") {
		t.Errorf("lit 'func main' over . missed main.go; got:\n%s", out)
	}
}

// TestBadRegexpExitsTwo witnesses that a usage-class error from the engine maps
// through main() to exit code 2.
func TestBadRegexpExitsTwo(t *testing.T) {
	code, _, errb := runBin(t, "grep", "--root", ".", "[unterminated")
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (stderr=%q)", code, errb)
	}
	if !strings.Contains(errb, "bad regexp") {
		t.Errorf("stderr missing 'bad regexp'; got:\n%s", errb)
	}
}
