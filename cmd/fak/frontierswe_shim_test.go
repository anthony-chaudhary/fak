package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TestFrontierSWEHarborShim binds the FrontierSWE fak-routing shim's Python gate
// (examples/frontierswe-harbor-shim/fak_routed_test.py) into `make ci` wherever a
// python3 interpreter exists. The shim (epic #1706, C6) is inherently Python — it is a
// harbor_ext `import_path` FrontierSWE loads by string — so its behavioral witness lives
// in a stdlib-only Python test; this driver runs that test and fails the Go build if the
// routing invariant regresses. Where python3 is absent the driver skips (honest gate,
// never a false green), exactly as the shim's own README documents.
func TestFrontierSWEHarborShim(t *testing.T) {
	py := lookPython(t)

	// go test runs with CWD = the package dir (cmd/fak); the shim lives two levels up.
	dir := filepath.Join("..", "..", "examples", "frontierswe-harbor-shim")
	script := filepath.Join(dir, "fak_routed_test.py")
	if _, err := os.Stat(script); err != nil {
		t.Skipf("shim test not present (%v)", err)
	}

	cmd := exec.Command(py, "fak_routed_test.py")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	t.Logf("python shim test output:\n%s", out)
	if err != nil {
		t.Fatalf("frontierswe harbor shim test failed: %v", err)
	}
}

// lookPython finds a Python 3 interpreter, skipping the test when none is on PATH.
func lookPython(t *testing.T) string {
	names := []string{"python3", "python"}
	if runtime.GOOS == "windows" {
		names = []string{"python", "python3", "py"}
	}
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	t.Skip("no python3 interpreter on PATH; skipping the FrontierSWE shim gate")
	return ""
}
