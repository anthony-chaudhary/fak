package witness

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// --- structural rung (fast, no real repo, no go test) -----------------------------------

// A fix commit that touched NO _test.go is REFUTED regardless of the execution flag — a fix
// with no symptom witness is exactly the gap the rung exists to catch.
func TestSymptomNoTestTouchedRefutes(t *testing.T) {
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "") // structural rung is flag-independent
	changed := "internal/x/x.go\ndocs/readme.md\n"
	if got := NewWithRunner((&fakeGit{out: changed, code: 0}).run, "").Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("symptom with no test touch = %v, want refuted", got)
	}
}

// A fix that DID touch a test abstains when the execution rung is opted OUT (default): the
// structural half passes, but we never run the heavy red-then-green without the flag, and an
// abstain (not a CONFIRM) is the fail-closed answer.
func TestSymptomTestTouchedFlagOffAbstains(t *testing.T) {
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "")
	changed := "internal/x/x.go\ninternal/x/x_test.go\n"
	if got := NewWithRunner((&fakeGit{out: changed, code: 0}).run, "").Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessAbstain {
		t.Fatalf("symptom, test touched, flag off = %v, want abstain", got)
	}
}

// testdata fixtures named *_test.go are NOT gating tests — a fix touching only those is REFUTED.
func TestSymptomTestdataIsNotAWitness(t *testing.T) {
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")
	changed := "internal/x/x.go\ninternal/x/testdata/case_test.go\n"
	if got := NewWithRunner((&fakeGit{out: changed, code: 0}).run, "").Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("symptom with only testdata test = %v, want refuted", got)
	}
}

// A fix that touched a Python test (e.g. tools/worktree_doctor_test.py) passes the structural
// rung and abstains when the execution rung is off (default).
func TestSymptomPythonTestTouchedFlagOffAbstains(t *testing.T) {
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "")
	changed := "tools/worktree_doctor.py\ntools/worktree_doctor_test.py\n"
	if got := NewWithRunner((&fakeGit{out: changed, code: 0}).run, "").Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessAbstain {
		t.Fatalf("symptom, python test touched, flag off = %v, want abstain", got)
	}
}

// Python testdata fixtures (e.g. tools/testdata/case_test.py) are NOT gating tests — a fix touching only those is REFUTED.
func TestSymptomPythonTestdataIsNotAWitness(t *testing.T) {
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")
	changed := "tools/worktree_doctor.py\ntools/testdata/case_test.py\n"
	if got := NewWithRunner((&fakeGit{out: changed, code: 0}).run, "").Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("symptom with only python testdata = %v, want refuted", got)
	}
}

// A bad ref / git failure abstains — never a false CONFIRM, never a REFUTE on uncertainty.
func TestSymptomBadRefAbstains(t *testing.T) {
	ctx := context.Background()
	if got := NewWithRunner((&fakeGit{code: 128}).run, "").Resolve(ctx, nil, "symptom:bogus"); got != abi.WitnessAbstain {
		t.Fatalf("symptom bad ref = %v, want abstain", got)
	}
}

// An empty arg abstains.
func TestSymptomEmptyArgAbstains(t *testing.T) {
	ctx := context.Background()
	if got := NewWithRunner((&fakeGit{code: 0}).run, "").Resolve(ctx, nil, "symptom: "); got != abi.WitnessAbstain {
		t.Fatalf("symptom empty arg = %v, want abstain", got)
	}
}

// --- pure helpers -----------------------------------------------------------------------

func TestChangedTestFilesAndPackages(t *testing.T) {
	out := "internal/x/x.go\ninternal/x/x_test.go\ncmd/y/y_test.go\ninternal/z/testdata/q_test.go\n\n"
	tests := changedTestFiles(out)
	if len(tests) != 2 || tests[0] != "internal/x/x_test.go" || tests[1] != "cmd/y/y_test.go" {
		t.Fatalf("changedTestFiles = %v, want the two non-testdata tests", tests)
	}
	pkgs := testPackages(tests)
	if len(pkgs) != 2 || pkgs[0] != "./internal/x" || pkgs[1] != "./cmd/y" {
		t.Fatalf("testPackages = %v, want the two package dirs", pkgs)
	}
}

func TestChangedTestFilesRecognizesPythonTests(t *testing.T) {
	out := strings.Join([]string{
		"tools/worktree_doctor.py",
		"tools/worktree_doctor_test.py",
		"tools/test_worktree_doctor.py",
		"tools/testdata/fixture_test.py",
		"tools/other_test.go",
		"cmd/y/y_test.go",
		"",
	}, "\n")
	tests := changedTestFiles(out)
	wantTests := []string{
		"tools/worktree_doctor_test.py",
		"tools/test_worktree_doctor.py",
		"tools/other_test.go",
		"cmd/y/y_test.go",
	}
	if !reflect.DeepEqual(tests, wantTests) {
		t.Fatalf("changedTestFiles = %v, want %v", tests, wantTests)
	}
	pkgs := testPackages(tests)
	wantPkgs := []string{"./tools", "./cmd/y"}
	if !reflect.DeepEqual(pkgs, wantPkgs) {
		t.Fatalf("testPackages = %v, want %v", pkgs, wantPkgs)
	}
	pyTests := pythonTestFiles(tests)
	wantPy := []string{"tools/worktree_doctor_test.py", "tools/test_worktree_doctor.py"}
	if !reflect.DeepEqual(pyTests, wantPy) {
		t.Fatalf("pythonTestFiles = %v, want %v", pyTests, wantPy)
	}
}

// --- execution rung (real repo, runs go test; opt-in) -----------------------------------

// TestSymptomRedThenGreenConfirms builds a real two-commit history of a tiny Go module: the
// parent has a buggy Sign(); the fix corrects it AND adds a test that exercises the bug. With
// FAK_WITNESS_SYMPTOM on, the rung overlays the new test onto the parent (red) and runs it at
// the fix (green) → CONFIRMED.
func TestSymptomRedThenGreenConfirms(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	// Parent: a buggy Sign that returns 0 for negatives (the symptom).
	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\tif n > 0 {\n\t\treturn 1\n\t}\n\treturn 0\n}\n")
	gitIn(t, dir, "add", "sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: buggy Sign")

	// Fix: correct Sign AND add a test that fails on the parent's source, passes here.
	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\tif n > 0 {\n\t\treturn 1\n\t}\n\tif n < 0 {\n\t\treturn -1\n\t}\n\treturn 0\n}\n")
	writeRepoFile(t, dir, "sign_test.go", "package m\n\nimport \"testing\"\n\nfunc TestSignNegative(t *testing.T) {\n\tif Sign(-3) != -1 {\n\t\tt.Fatalf(\"Sign(-3)=%d, want -1\", Sign(-3))\n\t}\n}\n")
	gitIn(t, dir, "add", "sign.go", "sign_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): Sign returns -1 for negatives")

	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("red-then-green symptom = %v, want confirmed", got)
	}
	assertRepoClean(t, dir)
}

// TestSymptomTautologicalTestRefutes: the fix adds a test, but the test passes against the
// parent's source too (it constrains nothing about the bug) → REFUTED.
func TestSymptomTautologicalTestRefutes(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\tif n > 0 {\n\t\treturn 1\n\t}\n\treturn 0\n}\n")
	gitIn(t, dir, "add", "sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent")

	// The fix touches source, but the added test only checks the POSITIVE case, which already
	// passed at the parent — a tautology that does not witness the negative-number bug.
	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\tif n > 0 {\n\t\treturn 1\n\t}\n\tif n < 0 {\n\t\treturn -1\n\t}\n\treturn 0\n}\n")
	writeRepoFile(t, dir, "sign_test.go", "package m\n\nimport \"testing\"\n\nfunc TestSignPositive(t *testing.T) {\n\tif Sign(3) != 1 {\n\t\tt.Fatalf(\"Sign(3)=%d, want 1\", Sign(3))\n\t}\n}\n")
	gitIn(t, dir, "add", "sign.go", "sign_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): Sign with tautological test")

	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("tautological-test symptom = %v, want refuted", got)
	}
	assertRepoClean(t, dir)
}

func requireGoAndGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
}

func requirePythonAndGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	py := pythonBin()
	if _, err := exec.LookPath(py); err != nil {
		t.Skip("python not on PATH")
	}
}

func writeRepoPath(t *testing.T, dir, relPath, body string) {
	t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(relPath))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestSymptomPythonRedThenGreenConfirms builds a real two-commit history of a Python tool:
// the parent has a buggy compute(); the fix corrects it AND adds tools/worktree_doctor_test.py
// that exercises the bug. With FAK_WITNESS_SYMPTOM on, the rung overlays the new test onto the
// parent (red) and runs it at the fix (green) → CONFIRMED.
func TestSymptomPythonRedThenGreenConfirms(t *testing.T) {
	requirePythonAndGit(t)
	dir := newExecutionRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	// Parent: a buggy compute function that returns 0 for negatives (the symptom).
	writeRepoPath(t, dir, "tools/calc.py", "def compute(n):\n    if n > 0:\n        return 1\n    return 0\n")
	gitIn(t, dir, "add", "tools/calc.py")
	gitIn(t, dir, "commit", "-q", "-m", "parent: buggy compute")

	// Fix: correct compute AND add tools/worktree_doctor_test.py that fails on the parent's source, passes here.
	writeRepoPath(t, dir, "tools/calc.py", "def compute(n):\n    if n > 0:\n        return 1\n    if n < 0:\n        return -1\n    return 0\n")
	writeRepoPath(t, dir, "tools/worktree_doctor_test.py", `import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import calc

class TestCalc(unittest.TestCase):
    def test_negative(self):
        self.assertEqual(calc.compute(-3), -1)

if __name__ == '__main__':
    unittest.main()
`)
	gitIn(t, dir, "add", "tools/calc.py", "tools/worktree_doctor_test.py")
	gitIn(t, dir, "commit", "-q", "-m", "fix(tools): compute returns -1 for negatives")

	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("red-then-green python symptom = %v, want confirmed", got)
	}
	assertRepoClean(t, dir)
}

// TestSymptomPythonTautologicalTestRefutes: the fix adds a Python test, but the test passes
// against the parent's source too (it constrains nothing about the bug) → REFUTED.
func TestSymptomPythonTautologicalTestRefutes(t *testing.T) {
	requirePythonAndGit(t)
	dir := newExecutionRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	writeRepoPath(t, dir, "tools/calc.py", "def compute(n):\n    if n > 0:\n        return 1\n    return 0\n")
	gitIn(t, dir, "add", "tools/calc.py")
	gitIn(t, dir, "commit", "-q", "-m", "parent")

	// The fix touches source, but the added test only checks the POSITIVE case, which already
	// passed at the parent — a tautology that does not witness the negative-number bug.
	writeRepoPath(t, dir, "tools/calc.py", "def compute(n):\n    if n > 0:\n        return 1\n    if n < 0:\n        return -1\n    return 0\n")
	writeRepoPath(t, dir, "tools/worktree_doctor_test.py", `import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import calc

class TestCalc(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(calc.compute(3), 1)

if __name__ == '__main__':
    unittest.main()
`)
	gitIn(t, dir, "add", "tools/calc.py", "tools/worktree_doctor_test.py")
	gitIn(t, dir, "commit", "-q", "-m", "fix(tools): compute with tautological test")

	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("tautological python symptom = %v, want refuted", got)
	}
	assertRepoClean(t, dir)
}

// newGoModuleRepo is newExecutionRepo plus a go.mod so `go test ./...` builds.
func newGoModuleRepo(t *testing.T) string {
	t.Helper()
	dir := newExecutionRepo(t)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module m\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitIn(t, dir, "add", "go.mod")
	gitIn(t, dir, "commit", "-q", "-m", "init module")
	return dir
}
