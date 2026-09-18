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

// TestSymptomRejectsParentBuildFailure: when the overlaid test on parent fails to compile
// (e.g. references a new helper/API introduced by the fix), that failure must NOT be
// counted as red symptom evidence — it must ABSTAIN as unproven, not CONFIRM (#12058).
func TestSymptomRejectsParentBuildFailure(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	// Parent: has Sign, but does NOT have NewHelper.
	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\treturn 0\n}\n")
	gitIn(t, dir, "add", "sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: Sign only")

	// Fix: introduces NewHelper AND corrects Sign, and sign_test.go calls NewHelper.
	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc NewHelper() bool {\n\treturn true\n}\n\nfunc Sign(n int) int {\n\tif n < 0 {\n\t\treturn -1\n\t}\n\treturn 0\n}\n")
	writeRepoFile(t, dir, "sign_test.go", "package m\n\nimport \"testing\"\n\nfunc TestSignNegative(t *testing.T) {\n\tif !NewHelper() {\n\t\tt.Fatal(\"helper failed\")\n\t}\n\tif Sign(-3) != -1 {\n\t\tt.Fatalf(\"Sign(-3)=%d, want -1\", Sign(-3))\n\t}\n}\n")
	gitIn(t, dir, "add", "sign.go", "sign_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): introduce NewHelper and fix Sign")

	// On fix commit, tests pass (green).
	// On parent, sign_test.go fails to compile because NewHelper is undefined.
	// This compilation failure must be classified as ABSTAIN, never CONFIRMED.
	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessAbstain {
		t.Fatalf("parent build failure symptom = %v, want abstain", got)
	}
	assertRepoClean(t, dir)
}

func TestIsBuildFailureDetection(t *testing.T) {
	buildFailures := []string{
		"# m [m.test]\n./sign_test.go:5:2: undefined: NewAPI\nFAIL\tm [build failed]\nFAIL",
		"FAIL\tm [setup failed]\nFAIL",
		"compile error: syntax error",
		"compiler error: internal failure",
		"build error: failed to resolve dependencies",
		"syntax error: unexpected token",
		"cannot find package \"foo\" in any of:",
		"no Go files in /some/path",
		"# m\nsign_test.go:5:2: undefined: SomeFunc\nFAIL",
	}
	for _, out := range buildFailures {
		if !isGoBuildFailure(out) {
			t.Errorf("isGoBuildFailure(%q) = false, want true", out)
		}
	}

	testFailures := []string{
		"=== RUN   TestSignNegative\n--- FAIL: TestSignNegative (0.00s)\n    sign_test.go:6: Sign(-3)=0, want -1\nFAIL\nFAIL\tm\t0.010s\nFAIL",
		"=== RUN   TestPanic\n--- FAIL: TestPanic (0.00s)\npanic: boom\nFAIL\tm\t0.010s\nFAIL",
	}
	for _, out := range testFailures {
		if isGoBuildFailure(out) {
			t.Errorf("isGoBuildFailure(%q) = true, want false", out)
		}
	}

	pyBuildFailures := []string{
		"Traceback (most recent call last):\n  File \"test.py\", line 1\n    def foo(\nSyntaxError: invalid syntax",
		"Traceback (most recent call last):\n  File \"test.py\", line 2\nImportError: cannot import name 'NewAPI'",
		"ModuleNotFoundError: No module named 'foo'",
	}
	for _, out := range pyBuildFailures {
		if !isPythonBuildFailure(out) {
			t.Errorf("isPythonBuildFailure(%q) = false, want true", out)
		}
	}

	pyTestFailures := []string{
		"FAIL: test_negative (__main__.TestCalc)\nAssertionError: 0 != -1\nFAILED (failures=1)",
	}
	for _, out := range pyTestFailures {
		if isPythonBuildFailure(out) {
			t.Errorf("isPythonBuildFailure(%q) = true, want false", out)
		}
	}
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

// --- build-tag threading (#13243) -------------------------------------------------------

// taggedTestFiles derives the build constraints a changed test file declares, so the
// execution rung can pass the matching -tags to `go test`. A test gated behind
// `//go:build vulkan` is invisible to a bare `go test`, which let the gate pass
// trivially at BOTH refs and falsely REFUTE a genuine device witness (#13243).
func TestSymptomBuildTagsDerivedFromChangedTests(t *testing.T) {
	cases := []struct {
		name string
		body string
		want []string
	}{
		{"none", "package m\n\nimport \"testing\"\n", nil},
		{"single", "//go:build vulkan\n\npackage m\n", []string{"vulkan"}},
		{"or", "//go:build vulkan || metal\n\npackage m\n", []string{"metal", "vulkan"}},
		{"and", "//go:build linux && cuda\n\npackage m\n", []string{"cuda", "linux"}},
		{"not", "//go:build !windows\n\npackage m\n", []string{"!windows"}},
		{"parens", "//go:build (vulkan || metal) && !windows\n\npackage m\n", []string{"!windows", "metal", "vulkan"}},
		{"legacy", "// +build vulkan\n\npackage m\n", []string{"vulkan"}},
		{"blank-before-comment", "\n\n//go:build gpu\n\npackage m\n", []string{"gpu"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildTagsFromTestSources([]string{tc.body})
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("buildTagsFromTestSources(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}

// The execution rung must run the changed package with the tags its test files declare.
// This is the fail-to-pass core of #13243: before the fix the argv is a bare
// `go test -count=1 <pkg>` (so a `//go:build vulkan` test is silently excluded and a
// genuine device witness is refuted); after the fix the derived tag is threaded in.
func TestSymptomGoTestArgvThreadsDerivedTags(t *testing.T) {
	ctx := context.Background()
	var seen [][]string
	runner := func(_ context.Context, _ string, argv ...string) (string, int, error) {
		seen = append(seen, append([]string(nil), argv...))
		return "", 0, nil
	}

	// Tagged change: the derived constraint must land in the argv.
	goTestPasses(ctx, runner, "/tmp", []string{"./internal/compute"}, []string{"vulkan"})
	if len(seen) != 1 {
		t.Fatalf("runner invoked %d times, want 1", len(seen))
	}
	if !containsArg(seen[0], "-tags") || !containsArg(seen[0], "vulkan") {
		t.Fatalf("tagged argv = %v, want a -tags vulkan byte in it (bare `go test` = the #13243 defect)", seen[0])
	}

	// Untagged change: byte-identical to today — no -tags flag smuggled in.
	seen = nil
	goTestPasses(ctx, runner, "/tmp", []string{"./internal/compute"}, nil)
	if len(seen) != 1 {
		t.Fatalf("runner invoked %d times, want 1", len(seen))
	}
	if containsArg(seen[0], "-tags") {
		t.Fatalf("untagged argv = %v, want NO -tags flag", seen[0])
	}
	want := []string{"go", "test", "-count=1", "./internal/compute"}
	if !reflect.DeepEqual(seen[0], want) {
		t.Fatalf("untagged argv = %v, want %v", seen[0], want)
	}
}

// Tag derivation reads the REAL changed file contents (from git), not the request text:
// a source-only change thread yields no tags and an untagged argv (#13243 scope guard).
func TestSymptomDerivesTagsFromGitBlobs(t *testing.T) {
	requireGoAndGit(t)
	dir := newExecutionRepo(t)
	ctx := context.Background()

	writeRepoFile(t, dir, "a_test.go", "//go:build vulkan\n\npackage m\n")
	writeRepoFile(t, dir, "b_test.go", "package m\n\nimport \"testing\"\n")
	gitIn(t, dir, "add", "a_test.go", "b_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "add tagged and untagged tests")

	// Both constraints come from the blobs at HEAD; the untagged file contributes nothing.
	got := resolveGoBuildTags(ctx, gitRunner, dir, "HEAD", []string{"a_test.go", "b_test.go"})
	if !reflect.DeepEqual(got, []string{"vulkan"}) {
		t.Fatalf("tag derivation = %v, want [vulkan]", got)
	}
}

// A tagged run that cannot build must ABSTAIN, never CONFIRM — fail-closed preserved (#13243).
func TestSymptomTaggedRedThenGreenConfirms(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\tif n > 0 {\n\t\treturn 1\n\t}\n\treturn 0\n}\n")
	gitIn(t, dir, "add", "sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: buggy Sign")

	// The ONLY test witnessing the fix is build-tag gated: a bare `go test` excludes it
	// at both refs and the rung would falsely REFUTE (the #13243 symptom).
	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int {\n\tif n > 0 {\n\t\treturn 1\n\t}\n\tif n < 0 {\n\t\treturn -1\n\t}\n\treturn 0\n}\n")
	writeRepoFile(t, dir, "sign_test.go", "//go:build vulkan\n\npackage m\n\nimport \"testing\"\n\nfunc TestSignNegative(t *testing.T) {\n\tif Sign(-3) != -1 {\n\t\tt.Fatalf(\"Sign(-3)=%d, want -1\", Sign(-3))\n\t}\n}\n")
	gitIn(t, dir, "add", "sign.go", "sign_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): Sign returns -1 for negatives (tagged witness)")

	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("tagged red-then-green symptom = %v, want confirmed (bare untagged `go test` = the #13243 false refutation)", got)
	}
	assertRepoClean(t, dir)
}

func containsArg(argv []string, want string) bool {
	for _, a := range argv {
		if a == want {
			return true
		}
	}
	return false
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
