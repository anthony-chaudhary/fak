package validate

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestOwnedTestRunExpressionSelectsOnlyOwnedTests(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "p", "owned_test.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `package p

import "testing"

func helper() {}
func TestZulu(t *testing.T) {}
func BenchmarkOwned(b *testing.B) {}
func FuzzOwned(f *testing.F) {}
func TestAlpha(t *testing.T) {}
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ownedTestRunExpression(root, []string{"p/owned_test.go"})
	if err != nil {
		t.Fatal(err)
	}
	want := "^(FuzzOwned|TestAlpha|TestZulu)$"
	if got != want {
		t.Fatalf("test run expression=%q, want %q", got, want)
	}

	gotMixed, err := ownedTestRunExpression(root, []string{"p/owned_test.go", "p/production.go"})
	if err != nil {
		t.Fatal(err)
	}
	if gotMixed != "" {
		t.Fatalf("test run expression with mixed paths=%q, want %q", gotMixed, "")
	}
}

func TestValidateTestRunnerSelectsWSLOnWindows(t *testing.T) {
	if !defaultValidateWSLTests("windows") {
		t.Fatal("Windows must default to WSL tests")
	}
	if defaultValidateWSLTests("linux") {
		t.Fatal("non-Windows hosts must retain native tests")
	}
	if got := validateTestRunner("windows", true); got != "wsl.exe bash -lc go test" {
		t.Fatalf("Windows runner = %q, want WSL", got)
	}
	if got := validateTestRunner("linux", true); got != "go test" {
		t.Fatalf("Linux runner = %q, want native Go", got)
	}
	if got := validateTestRunner("windows", false); got != "go test" {
		t.Fatalf("opt-out runner = %q, want native Go", got)
	}
}

func TestValidateBuildAndVetArgsArePathPortable(t *testing.T) {
	tests := []struct {
		mode string
		want []string
	}{
		{"build", []string{"build", "-trimpath", "-buildvcs=false", "./internal/owned"}},
		{"vet", []string{"vet", "-trimpath", "./internal/owned"}},
	}
	for _, tc := range tests {
		got := validateGoCheckArgs(tc.mode, []string{"./internal/owned"})
		if strings.Join(got, "|") != strings.Join(tc.want, "|") {
			t.Fatalf("%s args = %v, want exact %v", tc.mode, got, tc.want)
		}
	}
}

func TestRenderValidateReportsRunnerOnFailure(t *testing.T) {
	var out bytes.Buffer
	renderValidate(&out, validateResult{
		Tip:    "0123456789abcdef",
		Runner: "wsl.exe bash -lc go test",
		Failures: []ciPreflightFailure{{
			Step:   "test",
			Detail: "deliberate fixture failure",
		}},
	})
	got := out.String()
	for _, want := range []string{"runner: wsl.exe bash -lc go test", "test: deliberate fixture failure"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render = %q; want %q", got, want)
		}
	}
}

func TestValidatePhaseOrderIncludesSmoke(t *testing.T) {
	withoutSmoke := validatePhaseOrder(false, false, false)
	if slices.Contains(withoutSmoke, "smoke") {
		t.Fatalf("expected validatePhaseOrder without smoke to omit 'smoke', got %v", withoutSmoke)
	}

	withSmoke := validatePhaseOrder(false, false, true)
	if !slices.Contains(withSmoke, "smoke") {
		t.Fatalf("expected validatePhaseOrder with smoke to contain 'smoke', got %v", withSmoke)
	}
	if withSmoke[len(withSmoke)-1] != "smoke" {
		t.Fatalf("expected 'smoke' phase to be at the end, got %v", withSmoke)
	}
}

func TestValidateSmokePolicyPathUsesForwardSlashes(t *testing.T) {
	policyPath := validateSmokePolicyPath()
	if strings.Contains(policyPath, `\`) {
		t.Fatalf("expected validateSmokePolicyPath to contain no backslashes, got %q", policyPath)
	}
	if !strings.Contains(policyPath, "/") {
		t.Fatalf("expected validateSmokePolicyPath to contain forward slashes, got %q", policyPath)
	}
	expected := "examples/customer-support-readonly-policy.json"
	if policyPath != expected {
		t.Fatalf("expected validateSmokePolicyPath == %q, got %q", expected, policyPath)
	}

	repoRel := filepath.Join("..", "..", filepath.FromSlash(policyPath))
	if info, err := os.Stat(repoRel); err != nil || info.IsDir() {
		t.Fatalf("expected policy path %q to resolve to an existing file in repo, err: %v", repoRel, err)
	}
}

func TestValidateSmokeWSLPathNormalization(t *testing.T) {
	// On Windows, filepath.Join with a WSL Linux path like /tmp/... generates backslashes.
	// Verify that filepath.ToSlash correctly normalizes it for WSL execution.
	wslDir := "/tmp/fak-validate-12345"
	reportPath := filepath.Join(wslDir, "agent-smoke-report.json")
	reportArg := filepath.ToSlash(reportPath)

	if strings.Contains(reportArg, `\`) {
		t.Fatalf("expected WSL smoke report arg to contain no backslashes, got %q", reportArg)
	}
	expected := "/tmp/fak-validate-12345/agent-smoke-report.json"
	if reportArg != expected {
		t.Fatalf("expected WSL smoke report arg == %q, got %q", expected, reportArg)
	}
}
