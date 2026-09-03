package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/interspersedflags"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestValidateTimeoutReturnsStructuredPartialResultAndProgress(t *testing.T) {
	oldHook := validatePhaseHook
	validatePhaseHook = func(ctx context.Context, phase string) {
		if phase == "resolve_ref" {
			<-ctx.Done()
		}
	}
	t.Cleanup(func() { validatePhaseHook = oldHook })

	res, code, stderr := runValidateJSON(t, []string{
		"--root", t.TempDir(),
		"--mine", "p/p.go",
		"--timeout", "5ms",
		"--progress",
		"--json",
	})
	if code == 0 || res.OK || !res.Partial || !res.TimedOut || res.Reason != "TIMEOUT" {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if res.TimeoutMS != (5*time.Millisecond).Milliseconds() || res.ElapsedMS < 0 {
		t.Fatalf("timeout_ms=%d elapsed_ms=%d", res.TimeoutMS, res.ElapsedMS)
	}
	if len(res.Overlays.Checked) != 0 || len(res.Overlays.Skipped) != 1 || res.Overlays.Skipped[0] != "p/p.go" {
		t.Fatalf("overlays=%+v; want p/p.go reported skipped", res.Overlays)
	}
	if len(res.Phases) < 2 || res.Phases[len(res.Phases)-1].Name != "resolve_ref" || res.Phases[len(res.Phases)-1].Status != "timeout" {
		t.Fatalf("phases=%+v; want timed-out resolve_ref phase", res.Phases)
	}
	if len(res.SkippedPhases) == 0 || !validateContains(res.SkippedPhases, "overlay") || !validateContains(res.SkippedPhases, "test") {
		t.Fatalf("skipped_phases=%v; want unrun overlay and test phases", res.SkippedPhases)
	}
	for _, want := range []string{"phase=resolve_root status=start", "phase=resolve_ref status=start", "phase=resolve_ref status=timeout"} {
		if !strings.Contains(stderr, want) {
			t.Fatalf("progress stderr=%q; want %q", stderr, want)
		}
	}
}

func TestNormalizeMinePathsPrunesScratchDirsUnlessExplicit(t *testing.T) {
	root := t.TempDir()
	write := func(rel string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(rel), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, rel := range []string{
		"owned/visible.go",
		"owned/nested/visible_test.go",
		"owned/.dispatch-runs/run.log",
		"owned/.hidden/peer.go",
		"owned/_scratch/generated.go",
		"owned/nested/_generated/peer.go",
	} {
		write(rel)
	}

	got, err := normalizeMinePaths(root, []string{"owned"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"owned/nested/visible_test.go", "owned/visible.go"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("broad directory expansion=%v, want %v", got, want)
	}
	explicit, err := normalizeMinePaths(root, []string{"owned/.hidden"})
	if err != nil {
		t.Fatal(err)
	}
	if len(explicit) != 1 || explicit[0] != "owned/.hidden/peer.go" {
		t.Fatalf("explicit hidden directory=%v; want its owned file", explicit)
	}
}

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
	got, err := ownedTestRunExpression(root, []string{"p/owned_test.go", "p/production.go"})
	if err != nil {
		t.Fatal(err)
	}
	want := "^(FuzzOwned|TestAlpha|TestZulu)$"
	if got != want {
		t.Fatalf("test run expression=%q, want %q", got, want)
	}
}

func runValidateJSON(t *testing.T, argv []string) (validateResult, int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runValidate(&stdout, &stderr, argv)
	var res validateResult
	if stdout.Len() > 0 {
		if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
			t.Fatalf("invalid JSON: %v\n%s", err, stdout.String())
		}
	}
	return res, code, stderr.String()
}

func TestValidateTestOnlyIgnoresBrokenPeerWIPAndReportsMode(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": cleanGoFile,
		"p/p_test.go": `package p

import "testing"

func TestAdd(t *testing.T) {
	if Add(1, 2) != 3 { t.Fatal("bad") }
}
`,
		"peer/peer.go": "package peer\n\nfunc OK() {}\n",
	})
	if err := os.WriteFile(filepath.Join(repo, "p", "p_test.go"), []byte(`package p

import "testing"

func TestAdd(t *testing.T) {
	if Add(1, 2) != 3 { t.Fatal("bad") }
}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer", "peer.go"), []byte("package peer\n\nfunc Broken( {\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, code, stderr := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p_test.go", "--test-only", "--json"})
	if code != 0 || !res.OK || res.Schema != "fak-validate/1" || res.Mode != "test-only" {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if len(res.Tested) == 0 {
		t.Fatalf("expected affected package tests: %+v", res)
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

func TestValidateWSLCapabilitySurfaceCoversEveryExternalCommand(t *testing.T) {
	got := strings.Join(validateWSLRequiredCommandNames(), ",")
	want := "bash,cp,git,go,gofmt,ls,mkdir,mv,pwd,rm,tail,tar,wsl.exe,xargs"
	if got != want {
		t.Fatalf("WSL capability surface = %q, want %q", got, want)
	}
	phaseCommands := make(map[string]bool)
	for _, surface := range validateWSLCommandSurface {
		if surface.phase == "" || len(surface.commands) == 0 {
			t.Fatalf("invalid WSL command surface: %+v", surface)
		}
		for _, command := range surface.commands {
			phaseCommands[surface.phase+"/"+command] = true
		}
	}
	for _, want := range []string{
		"test_materialize/tar", "extract_tip/git", "extract_tip/xargs",
		"cleanup/rm", "overlay/cp", "overlay/pwd", "go_checks/gofmt",
	} {
		if !phaseCommands[want] {
			t.Fatalf("WSL command surface omits %s", want)
		}
	}
}

func TestValidateWSLCapabilityPreflightReportsMissingAndCachesByIdentity(t *testing.T) {
	checkRuns := stubValidateWSLCapabilityCommands(t, "Ubuntu-24.04\n/usr/local/bin:/usr/bin", "rm\ngit\n")
	first := preflightValidateWSLCapabilitiesWithin(context.Background())
	if first.Status != "missing" || !strings.HasPrefix(first.Identity, "Ubuntu-24.04@path:") || strings.Contains(first.Identity, "/usr") || strings.Join(first.Missing, ",") != "git,rm" || first.Cached {
		t.Fatalf("first verdict = %+v", first)
	}
	if !strings.Contains(first.Detail, "no fallback selected") {
		t.Fatalf("detail = %q; want explicit no-fallback verdict", first.Detail)
	}
	encoded, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "/usr/local/bin") {
		t.Fatalf("verdict leaks raw PATH: %s", encoded)
	}
	second := preflightValidateWSLCapabilitiesWithin(context.Background())
	if !second.Cached || strings.Join(second.Missing, ",") != "git,rm" {
		t.Fatalf("cached verdict = %+v", second)
	}
	if *checkRuns != 1 {
		t.Fatalf("capability check runs = %d, want 1 for stable identity", *checkRuns)
	}
}

func TestValidateWSLCapabilityScriptReportsGofmtMissingFromPATH(t *testing.T) {
	script := validateWSLCapabilityScript([]string{"wsl.exe", "go", "gofmt"})
	if !strings.Contains(script, "for cmd in 'go' 'gofmt'") || !strings.Contains(script, "command -v \"$cmd\"") {
		t.Fatalf("capability script does not check bare go and gofmt through PATH: %s", script)
	}
	if strings.Contains(script, "GOROOT") || strings.Contains(script, "go env") {
		t.Fatalf("capability script accepts a GOROOT-only gofmt that the real command cannot invoke: %s", script)
	}

	stubValidateWSLCapabilityCommands(t, "Ubuntu-24.04\n/usr/bin", "gofmt\n")
	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "missing" || strings.Join(verdict.Missing, ",") != "gofmt" {
		t.Fatalf("verdict = %+v; want go present and bare gofmt missing", verdict)
	}
}

func TestValidateWSLCapabilityPreflightTypesMissingHostLauncher(t *testing.T) {
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	validateWSLLookPath = func(string) (string, error) { return "", os.ErrNotExist }
	validateWSLCommand = func(context.Context, ...string) ([]byte, error) {
		t.Fatal("WSL command must not run without the host launcher")
		return nil, nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})
	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "missing" || strings.Join(verdict.Missing, ",") != "wsl.exe" {
		t.Fatalf("verdict = %+v", verdict)
	}
}

func TestValidateWSLCapabilityPreflightBoundsProbe(t *testing.T) {
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	validateWSLLookPath = func(string) (string, error) { return `C:\Windows\System32\wsl.exe`, nil }
	var remaining time.Duration
	validateWSLCommand = func(ctx context.Context, _ ...string) ([]byte, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Fatal("WSL capability probe has no deadline")
		}
		remaining = time.Until(deadline)
		return []byte(validateWSLDistroPrefix + "Ubuntu-24.04\n" + validateWSLPathPrefix + "/usr/bin\n"), nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})

	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "ready" {
		t.Fatalf("verdict = %+v", verdict)
	}
	if remaining <= 0 || remaining > validateWSLPreflightTimeout {
		t.Fatalf("probe deadline remaining = %s, want (0, %s]", remaining, validateWSLPreflightTimeout)
	}
}

func TestValidateMissingWSLCapabilityStopsBeforeWorkspaceAllocation(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows selects the WSL workspace")
	}
	stubValidateWSLCapabilityCommands(t, "Ubuntu-24.04\n/usr/bin", "rm\n")
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "fixture", map[string]string{"go.mod": cleanGoMod, "p/p.go": cleanGoFile})

	res, code, stderr := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--wsl-tests", "--json"})
	if code != 2 || res.OK || !res.Partial || res.Reason != "WSL_CAPABILITY_MISSING" {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if res.WSLPreflight == nil || strings.Join(res.WSLPreflight.Missing, ",") != "rm" {
		t.Fatalf("WSL preflight = %+v", res.WSLPreflight)
	}
	if !validateContains(res.SkippedPhases, "normalize_mine") || !validateContains(res.SkippedPhases, "extract_tip") {
		t.Fatalf("skipped phases = %v; workspace phases must not start", res.SkippedPhases)
	}
	for _, phase := range res.Phases {
		if phase.Name == "extract_tip" {
			t.Fatalf("workspace allocation phase unexpectedly started: %+v", res.Phases)
		}
	}
	if !strings.Contains(stderr, "no fallback selected") {
		t.Fatalf("stderr = %q; want prompt no-fallback verdict", stderr)
	}
}

func stubValidateWSLCapabilityCommands(t *testing.T, identity, missing string) *int {
	t.Helper()
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	checkRuns := 0
	validateWSLLookPath = func(string) (string, error) { return `C:\Windows\System32\wsl.exe`, nil }
	validateWSLCommand = func(_ context.Context, args ...string) ([]byte, error) {
		if len(args) < 3 || args[0] != "bash" || args[1] != "-lc" {
			return nil, fmt.Errorf("unexpected WSL capability command: %v", args)
		}
		if !strings.Contains(args[2], validateWSLDistroPrefix) || !strings.Contains(args[2], validateWSLPathPrefix) {
			return nil, fmt.Errorf("capability script omits identity protocol: %s", args[2])
		}
		checkRuns++
		parts := strings.SplitN(identity, "\n", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid stub identity %q", identity)
		}
		return []byte(validateWSLDistroPrefix + parts[0] + "\n" + validateWSLPathPrefix + parts[1] + "\n" + missing), nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})
	return &checkRuns
}

func resetValidateWSLCapabilityCacheForTest() {
	validateWSLCapabilities.Lock()
	validateWSLCapabilities.byIdentity = make(map[string]validateWSLCapabilityVerdict)
	validateWSLCapabilities.identityByLauncher = make(map[string]string)
	validateWSLCapabilities.Unlock()
}

func TestValidateTestArgsDisableCacheAndKeepRunBeforeTargets(t *testing.T) {
	got := validateTestArgs("^TestOwned$", []string{"./internal/owned"})
	want := []string{"test", "-count=1", "-run", "^TestOwned$", "./internal/owned"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("validate test args = %v, want %v", got, want)
	}
}

func TestValidateBuildAndVetArgsArePathPortable(t *testing.T) {
	tests := []struct {
		mode string
		want []string
	}{
		{"build", []string{"build", "-trimpath", "./internal/owned"}},
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

func TestValidateRequiresExplicitMine(t *testing.T) {
	_, code, stderr := runValidateJSON(t, []string{"--json"})
	if code != 2 || !bytes.Contains([]byte(stderr), []byte("at least one --mine")) {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
}

func TestValidateMultipleAndInterspersedMinePaths(t *testing.T) {
	var fs flag.FlagSet
	var mine pathList
	fs.Var(&mine, "mine", "")
	asJSON := fs.Bool("json", false, "")

	argv := []string{"--mine", "p1.go", "p2.go", "--json", "--mine", "p3.go", "p4.go"}
	positional, err := interspersedflags.Parse(&fs, argv)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range positional {
		if p = strings.TrimSpace(p); p != "" {
			mine = append(mine, p)
		}
	}
	if !*asJSON {
		t.Fatal("expected --json to be parsed after positional arguments")
	}
	want := []string{"p1.go", "p3.go", "p2.go", "p4.go"}
	if len(mine) != len(want) {
		t.Fatalf("mine count = %d, want %d: %v", len(mine), len(want), mine)
	}
}

func TestValidateCommittedTipPlusOnlyMine(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": cleanGoFile,
		"p/p_test.go": `package p

import "testing"

func TestAdd(t *testing.T) {
	if Add(1, 2) != 3 {
		t.Fatal("bad")
	}
}
`,
		"peer/peer.go": "package peer\n\nfunc OK() {}\n",
	})
	// Caller-owned change is valid; unrelated tracked peer WIP is intentionally uncompilable.
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte("package p\n\n// Add returns a + b.\nfunc Add(a, b int) int {\n\treturn a + b + 0\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer", "peer.go"), []byte("package peer\n\nfunc Broken( {\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, code, stderr := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if len(res.Mine) != 1 || res.Mine[0] != "p/p.go" {
		t.Fatalf("mine=%v", res.Mine)
	}
	if len(res.Tested) == 0 {
		t.Fatalf("expected affected package test selection")
	}
	if res.SelectionAudit != nil {
		t.Fatalf("default validation unexpectedly emitted selection audit: %+v", res.SelectionAudit)
	}
}

func TestValidateAuditSelectionReportsFullOnlyFailure(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "fixture", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": cleanGoFile,
		"p/p_test.go": `package p

import "testing"

func TestAdd(t *testing.T) {
	if Add(1, 2) != 3 { t.Fatal("bad") }
}
`,
		"q/q.go": "package q\n\nfunc Value() int { return 1 }\n",
		"q/q_test.go": `package q

import "testing"

func TestFullOnlyFailure(t *testing.T) { t.Fatal("truth failure") }
`,
	})
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte("package p\n\n// Add returns a + b.\nfunc Add(a, b int) int { return a + b + 0 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, code, stderr := runValidateJSON(t, []string{
		"--root", repo, "--mine", "p/p.go", "--audit-selection", "--wsl-tests=false", "--json",
	})
	if code != 1 || res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if res.SelectionAudit == nil {
		t.Fatalf("missing selection audit: %+v", res)
	}
	audit := res.SelectionAudit
	if audit.Base != res.Tip || !strings.HasPrefix(audit.Head, res.Tip+"+mine:") {
		t.Fatalf("base=%q head=%q tip=%q", audit.Base, audit.Head, res.Tip)
	}
	if got, want := audit.SelectedPackages, []string{"gitfixture.test/p"}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("selected_packages=%v, want %v", got, want)
	}
	if audit.Sound || !audit.Complete {
		t.Fatalf("audit=%+v; want complete unsound result", audit.SelectionAudit)
	}
	if got, want := audit.FullFailures, []string{"gitfixture.test/q"}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("full_failures=%v, want %v", got, want)
	}
	if got, want := audit.SelectorMisses, []string{"gitfixture.test/q"}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("selector_misses=%v, want %v", got, want)
	}
}

func TestValidateSelectsTrackedObjectiveCAndHeaderOverlays(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if runtime.GOOS != "darwin" {
		t.Skip("Objective-C cgo integration witness requires Darwin")
	}
	cgoEnabled, err := exec.Command("go", "env", "CGO_ENABLED").Output()
	if err != nil || strings.TrimSpace(string(cgoEnabled)) != "1" {
		t.Skip("Objective-C cgo integration witness requires CGO_ENABLED=1")
	}

	repo, git := seedGitFixtureRepo(t)
	const baseGo = `package p

/*
#include "p.h"
int base_value(void);
*/
import "C"

func Value() int { return int(C.base_value()) }
`
	const baseHeader = "#define P_VALUE 1\n"
	const baseObjectiveC = "#include \"p.h\"\nint base_value(void) { return P_VALUE; }\n"
	commitFiles(t, repo, git, "clean native package", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": baseGo,
		"p/p.h":  baseHeader,
		"p/p.m":  baseObjectiveC,
	})
	t.Setenv("GOCACHE", t.TempDir())

	write := func(rel, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(repo, filepath.FromSlash(rel)), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	validateGreen := func(paths ...string) {
		t.Helper()
		argv := []string{"--root", repo, "--json", "--test-run", "^$"}
		for _, path := range paths {
			argv = append(argv, "--mine", path)
		}
		res, code, stderr := runValidateJSON(t, argv)
		if code != 0 || !res.OK {
			t.Fatalf("paths=%v code=%d stderr=%q result=%+v", paths, code, stderr, res)
		}
		if !validateContains(res.Tested, "gitfixture.test/p") {
			t.Fatalf("paths=%v tested=%v; want owning package selected", paths, res.Tested)
		}
		for _, path := range paths {
			if !validateContains(res.Overlays.Checked, path) {
				t.Fatalf("paths=%v overlays=%+v; want %s checked", paths, res.Overlays, path)
			}
		}
	}

	write("p/p.m", "#include \"p.h\"\nint base_value(void) { return P_VALUE + 1; }\n")
	validateGreen("p/p.m")
	write("p/p.m", baseObjectiveC)

	write("p/p.h", "#define P_VALUE 2\n")
	validateGreen("p/p.h")
	write("p/p.h", baseHeader)

	write("p/p.go", strings.Replace(baseGo,
		"int base_value(void);", "int base_value(void);\nint overlay_value(void);", 1)+
		"\nfunc OverlayValue() int { return int(C.overlay_value()) }\n")
	write("p/p.m", baseObjectiveC+"int overlay_value(void) { return P_VALUE + 2; }\n")
	validateGreen("p/p.go", "p/p.m")
}

func TestValidateWSLIsolatedCheckoutPreservesRequestedGitIdentity(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	files := map[string]string{
		"go.mod":       "module validate.test\n\ngo 1.26\n",
		"common/id.go": "package common\n\nimport \"errors\"\n\nfunc Check(string) error { return errors.New(\"owned overlay missing\") }\n",
		"tracked.txt":  "requested-ref\n",
	}
	for i := 1; i <= 6; i++ {
		pkg := fmt.Sprintf("p%d", i)
		files[pkg+"/identity_test.go"] = fmt.Sprintf(`package %s

import (
	"testing"

	"validate.test/common"
)

func TestRequestedGitIdentity(t *testing.T) {
	if err := common.Check(".."); err != nil {
		t.Fatal(err)
	}
}
`, pkg)
	}
	commitFiles(t, repo, git, "requested ref", files)
	wantHEAD, err := git("rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	commitFiles(t, repo, git, "later ref", map[string]string{"later.txt": "must stay outside requested ref\n"})

	overlay := strings.ReplaceAll(`package common

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const expectedHEAD = "__EXPECTED_HEAD__"

func Check(root string) error {
	head, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").CombinedOutput()
	if err != nil {
		return fmt.Errorf("git rev-parse HEAD: %w: %s", err, head)
	}
	if got := strings.TrimSpace(string(head)); got != expectedHEAD {
		return fmt.Errorf("HEAD = %s, want requested ref %s", got, expectedHEAD)
	}
	tracked, err := exec.Command("git", "-C", root, "ls-files").CombinedOutput()
	if err != nil {
		return fmt.Errorf("git ls-files: %w: %s", err, tracked)
	}
	trackedSet := "\n" + strings.TrimSpace(string(tracked)) + "\n"
	for _, path := range []string{"go.mod", "common/id.go", "tracked.txt"} {
		if !strings.Contains(trackedSet, "\n"+path+"\n") {
			return fmt.Errorf("tracked files omit %s: %s", path, tracked)
		}
	}
	for _, path := range []string{"later.txt", "peer-only.txt"} {
		if strings.Contains(trackedSet, "\n"+path+"\n") {
			return fmt.Errorf("tracked files include out-of-ref %s: %s", path, tracked)
		}
		if _, err := os.Stat(filepath.Join(root, path)); !os.IsNotExist(err) {
			return fmt.Errorf("isolated checkout leaked %s: %v", path, err)
		}
	}
	body, err := os.ReadFile(filepath.Join(root, "tracked.txt"))
	if err != nil {
		return err
	}
	if string(body) != "requested-ref\n" {
		return fmt.Errorf("tracked.txt = %q, want requested ref content", body)
	}
	return nil
}
`, "__EXPECTED_HEAD__", strings.TrimSpace(wantHEAD))
	if err := os.WriteFile(filepath.Join(repo, "common", "id.go"), []byte(overlay), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "tracked.txt"), []byte("peer-wip\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer-only.txt"), []byte("peer-wip\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, code, stderr := runValidateJSON(t, []string{
		"--root", repo,
		"--ref", strings.TrimSpace(wantHEAD),
		"--mine", "common/id.go",
		"--mine", "p1/identity_test.go",
		"--test-only",
		"--wsl-tests",
		"--json",
	})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	for _, want := range []string{"validate.test/common", "validate.test/p1"} {
		if !validateContains(res.Tested, want) {
			t.Fatalf("tested=%v; want changed package %s", res.Tested, want)
		}
	}
	if validateContains(res.Tested, "validate.test/p2") {
		t.Fatalf("tested=%v; unchanged importer must remain build/vet-only", res.Tested)
	}
}

func TestValidateIgnoresUnformattedPeerWIP(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{"go.mod": cleanGoMod, "p/p.go": cleanGoFile, "peer/peer.go": "package peer\n\nfunc OK() {}\n"})
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte(cleanGoFile), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer", "peer.go"), []byte("package peer\nfunc Peer( ){ }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code, stderr := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
}

func TestValidateReportsMineFailure(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{"go.mod": cleanGoMod, "p/p.go": cleanGoFile})
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte("package p\n\nfunc Broken( {\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code, _ := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--json"})
	if code != 1 || res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	found := false
	for _, failure := range res.Failures {
		if failure.Step == "build" {
			found = true
		}
	}
	if !found {
		t.Fatalf("failures=%+v; want build failure", res.Failures)
	}
}

func TestValidateBuildsButDoesNotTestReverseDependencies(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedGitFixtureRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{
		"go.mod":                    "module validate.test\n\ngo 1.26\n",
		"lib/lib.go":                "package lib\n\nfunc Value() int { return 1 }\n",
		"consumer/consumer.go":      "package consumer\n\nimport \"validate.test/lib\"\n\nfunc Value() int { return lib.Value() }\n",
		"consumer/consumer_test.go": "package consumer\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() != 1 { t.Fatal(\"contract changed\") }\n}\n",
	})
	if err := os.WriteFile(filepath.Join(repo, "lib", "lib.go"), []byte("package lib\n\nfunc Value() int { return 2 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code, _ := runValidateJSON(t, []string{"--root", repo, "--mine", "lib/lib.go", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	if !validateContains(res.Tested, "validate.test/lib") || validateContains(res.Tested, "validate.test/consumer") {
		t.Fatalf("tested=%v; want changed package only", res.Tested)
	}
	for _, phase := range res.Phases {
		if (phase.Name == "build" || phase.Name == "vet") && phase.Status != "ok" {
			t.Fatalf("phase=%+v; importer closure must remain build/vet clean", phase)
		}
	}
}

func TestNormalizeMinePathsRejectsEscape(t *testing.T) {
	root := t.TempDir()
	if _, err := normalizeMinePaths(root, []string{"../peer.go"}); err == nil {
		t.Fatal("expected repo escape refusal")
	}
	if _, err := normalizeMinePaths(root, []string{"."}); err == nil {
		t.Fatal("expected repo-root refusal")
	}
}

// TestOverlayMinePathsContainmentUsesResolvedRoot pins overlayMinePaths' containment
// check to a canonicalized root. EvalSymlinks(src) hands back a fully resolved path, so
// comparing it against a raw srcRoot refused every owned path on any host whose repo root
// is merely reachable through a symlink — macOS puts TMPDIR under /var, a symlink to
// /private/var, and `fak validate` there died with "resolves outside repo root" for files
// plainly inside the repo (#5364). The escape case runs on every platform and pins that
// resolving both sides did not widen what the check admits.
func TestOverlayMinePathsContainmentUsesResolvedRoot(t *testing.T) {
	t.Run("refuses an owned path that resolves outside the root", func(t *testing.T) {
		parent := t.TempDir()
		root := filepath.Join(parent, "repo")
		writeOverlayFixtureFile(t, filepath.Join(root, "p", "p.go"), "package p\n")
		writeOverlayFixtureFile(t, filepath.Join(parent, "peer", "peer.go"), "package peer\n")
		err := overlayMinePaths(root, t.TempDir(), []string{"../peer/peer.go"})
		if err == nil || !strings.Contains(err.Error(), "resolves outside repo root") {
			t.Fatalf("err=%v; want a containment refusal", err)
		}
	})

	t.Run("accepts an owned path under an aliased root spelling", func(t *testing.T) {
		body := "package p\n\nfunc Add(a, b int) int { return a + b }\n"
		root := filepath.Join(t.TempDir(), "validate_overlay_root_with_a_long_name")
		writeOverlayFixtureFile(t, filepath.Join(root, "p", "p.go"), body)
		alias := aliasedOverlayRootSpelling(t, root)

		// Non-vacuity: the alias must actually reproduce #5364, i.e. the resolved source
		// must read as an escape when measured against the raw alias spelling. Otherwise
		// the case below would pass with or without the fix.
		resolved, err := filepath.EvalSymlinks(filepath.Join(alias, "p", "p.go"))
		if err != nil {
			t.Fatalf("resolve owned path through alias %q: %v", alias, err)
		}
		if raw, relErr := filepath.Rel(alias, resolved); relErr == nil && !strings.HasPrefix(raw, "..") {
			t.Fatalf("alias %q does not exercise the raw-root asymmetry (rel=%q)", alias, raw)
		}

		dst := t.TempDir()
		if err := overlayMinePaths(alias, dst, []string{"p/p.go"}); err != nil {
			t.Fatalf("overlay through aliased root %q: %v", alias, err)
		}
		got, err := os.ReadFile(filepath.Join(dst, "p", "p.go"))
		if err != nil {
			t.Fatalf("read overlaid file: %v", err)
		}
		if string(got) != body {
			t.Fatalf("overlaid body=%q want %q", got, body)
		}
	})
}

func writeOverlayFixtureFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// aliasedOverlayRootSpelling returns a second spelling of root that resolves to the same
// directory: a symlink where the OS grants one, otherwise the Windows 8.3 short name,
// which needs no privilege and which EvalSymlinks normalizes back to the long form. Both
// stand in for the darwin /var -> /private/var TMPDIR the issue was reported against.
func aliasedOverlayRootSpelling(t *testing.T, root string) string {
	t.Helper()
	link := filepath.Join(t.TempDir(), "alias")
	symlinkErr := os.Symlink(root, link)
	if symlinkErr == nil {
		return link
	}
	if runtime.GOOS != "windows" {
		t.Fatalf("symlink an aliased root: %v", symlinkErr)
	}
	// Unprivileged Windows cannot create a symlink at all, so fall back to the 8.3 alias
	// rather than skipping: a containment check must not be witnessed vacuously.
	short := windowsShortPathSpelling(root)
	if short == "" || short == root {
		t.Skipf("no aliased root spelling available on this host: os.Symlink refused (%v) and 8.3 short names are off for %q", symlinkErr, root)
	}
	return short
}

// windowsShortPathSpelling reports dir's 8.3 name. The directory travels as the child's
// working directory rather than as an argument: cmd.exe re-parses its own command line
// and does not understand the backslash-escaped quotes Go emits, so a spelled-out path
// with a space or a quote in it comes back mangled.
func windowsShortPathSpelling(dir string) string {
	cmd := exec.Command("cmd", "/c", "for", "%I", "in", "(.)", "do", "@echo", "%~sfI")
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func validateContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
