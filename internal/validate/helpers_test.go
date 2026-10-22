package validate

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
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
	if got := validateTestRunner("windows", true); got != "wsl.exe -e bash -lc go test" {
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

// fak-test:runtime fast est=1s
func TestValidateGoVetArgsWithinUsesSelectedCgoMetadata(t *testing.T) {
	t.Setenv("CGO_ENABLED", "1")
	t.Setenv("GOWORK", "off")
	writeModule := func(root, source string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Join(root, "p"), 0o755); err != nil {
			t.Fatal(err)
		}
		for rel, body := range map[string]string{
			"go.mod": "module vetargs.test\n\ngo 1.21\n",
			"p/p.go": source,
		} {
			if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(rel)), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}

	pure := filepath.Join(t.TempDir(), "pure")
	writeModule(pure, "package p\n\nfunc Value() int { return 1 }\n")
	args, err := validateGoVetArgsWithin(context.Background(), pure, false, []string{"./p"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(args, "|"), "vet|-trimpath|./p"; got != want {
		t.Fatalf("pure-Go vet args = %q, want %q", got, want)
	}

	cgo := filepath.Join(t.TempDir(), "cgo")
	writeModule(cgo, "package p\n\n/* int value(void); */\nimport \"C\"\n\nfunc Value() int { return int(C.value()) }\n")
	args, err = validateGoVetArgsWithin(context.Background(), cgo, false, []string{"./p"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(args, "|"), "vet|./p"; got != want {
		t.Fatalf("cgo vet args = %q, want %q", got, want)
	}
	if _, err := validateGoVetArgsWithin(context.Background(), pure, false, []string{"./missing"}); err == nil {
		t.Fatal("missing target metadata probe succeeded; want fail-closed error")
	}
}

// fak-test:runtime slow est=60s
func TestValidateGoVetArgsWithinAvoidsRemovedCgoWorkspaceCachePath(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("WSL cache-path regression applies to Windows validation")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	if verdict := preflightValidateWSLCapabilitiesWithin(ctx); verdict.Status != "ready" {
		t.Skipf("WSL validation unavailable: %s", verdict.Detail)
	}
	defer holdValidateWSLVM(ctx)()

	nonce := fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	first := "/tmp/fak-validate-vetargs-" + nonce + "-one"
	second := "/tmp/fak-validate-vetargs-" + nonce + "-two"
	cache := "/tmp/fak-validate-vetargs-cache-" + nonce
	for _, dir := range []string{first, second, cache} {
		dir := dir
		defer cleanupValidateWSLDir(dir)
	}
	if out, err := runValidateWSLCommandWithin(ctx, "/", "mkdir", "-p", first, second, cache); err != nil {
		t.Fatalf("create WSL fixtures: %v: %s", err, out)
	}

	source := t.TempDir()
	files := map[string]string{
		"go.mod":          "module buildcheck.test\n\ngo 1.21\n",
		"p/p.go":          "package p\n\n/*\nint native_value(void);\n*/\nimport \"C\"\n\nfunc Value() int { return int(C.native_value()) }\n",
		"p/native.c":      "int native_value(void) { return 2; }\n",
		"cmd/fak/main.go": "package main\n\nimport \"buildcheck.test/p\"\n\nfunc main() { _ = p.Value() }\n",
	}
	var paths []string
	for rel, body := range files {
		paths = append(paths, rel)
		full := filepath.Join(source, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	slices.Sort(paths)
	for _, root := range []string{first, second} {
		if err := overlayMinePathsWSLWithin(ctx, source, root, paths, nil); err != nil {
			t.Fatal(err)
		}
	}
	targets := []string{"buildcheck.test/cmd/fak", "buildcheck.test/p"}
	env := []string{"GOWORK=off", "GOCACHE=" + cache}
	// Seed through the build phase, as validate does before vet: a vet-seeded cache
	// replays the whole vet action and never re-reads the removed cgo source.
	if out, ok := validateRunGoEnvWithin(ctx, first, true, env, validateGoCheckArgs("build", targets)...); !ok {
		t.Fatalf("seed trimmed cgo build cache: %s", out)
	}
	cleanupValidateWSLDir(first)
	if out, err := runValidateWSLCommandWithin(ctx, "/", "test", "!", "-e", first); err != nil {
		t.Fatalf("first workspace still exists: %v: %s", err, out)
	}
	if out, ok := validateRunGoEnvWithin(ctx, second, true, env, validateGoCheckArgs("vet", targets)...); ok {
		t.Fatalf("trimmed warm-cache control unexpectedly passed after its source workspace was removed: %q", out)
	}
	args, err := validateGoVetArgsWithin(ctx, second, true, targets)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(args, "-trimpath") {
		t.Fatalf("selected cgo vet args retain unsafe trimpath: %v", args)
	}
	if out, ok := validateRunGoEnvWithin(ctx, second, true, env, args...); !ok {
		t.Fatalf("cgo-safe warm-cache vet failed: %s", out)
	}
}

func TestRenderValidateReportsRunnerOnFailure(t *testing.T) {
	var out bytes.Buffer
	renderValidate(&out, validateResult{
		Tip:    "0123456789abcdef",
		Runner: "wsl.exe -e bash -lc go test",
		Failures: []ciPreflightFailure{{
			Step:   "test",
			Detail: "deliberate fixture failure",
		}},
	})
	got := out.String()
	for _, want := range []string{"runner: wsl.exe -e bash -lc go test", "test: deliberate fixture failure"} {
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
