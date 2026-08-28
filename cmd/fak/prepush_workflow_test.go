package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workflowlint"
)

func withPrepushWorkflowSeams(t *testing.T, paths []string, tree string) {
	t.Helper()
	oldChanged := prepushWorkflowChangedFiles
	oldTipPaths := prepushWorkflowTipPaths
	oldExtract := prepushWorkflowExtractTip
	oldCheck := prepushWorkflowCheckTree
	t.Cleanup(func() {
		prepushWorkflowChangedFiles = oldChanged
		prepushWorkflowTipPaths = oldTipPaths
		prepushWorkflowExtractTip = oldExtract
		prepushWorkflowCheckTree = oldCheck
	})
	prepushWorkflowChangedFiles = func(string, string, string) ([]string, error) {
		return paths, nil
	}
	prepushWorkflowTipPaths = func(string, string) ([]string, error) { return paths, nil }
	prepushWorkflowExtractTip = func(string, string) (string, error) { return tree, nil }
	prepushWorkflowCheckTree = workflowlint.CheckTree
}

func writeWorkflowFixture(t *testing.T, root, body string) {
	t.Helper()
	dir := filepath.Join(root, ".github", "workflows")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ci.yml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestPrepushWorkflowMalformedStructureBlocks(t *testing.T) {
	tree := t.TempDir()
	writeWorkflowFixture(t, tree, "jobs:\n  build:\n    runs-on: [ubuntu-latest\n")
	withPrepushWorkflowSeams(t, []string{".github/workflows/ci.yml"}, tree)

	res, code := evaluatePrePushWorkflow("repo", "base", "tip")
	if code != 1 || res.Verdict != "WORKFLOW_STRUCTURE" || res.OK || len(res.Findings) == 0 {
		t.Fatalf("malformed pushed workflow must block: code=%d result=%+v", code, res)
	}
}

func TestPrepushWorkflowCleanPasses(t *testing.T) {
	tree := t.TempDir()
	writeWorkflowFixture(t, tree, "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n")
	withPrepushWorkflowSeams(t, []string{".github/workflows/ci.yml"}, tree)

	res, code := evaluatePrePushWorkflow("repo", "base", "tip")
	if code != 0 || res.Verdict != "OK" || !res.OK || len(res.Findings) != 0 {
		t.Fatalf("clean pushed workflow must pass: code=%d result=%+v", code, res)
	}
}

func TestPrepushWorkflowUnrelatedRangeSkips(t *testing.T) {
	withPrepushWorkflowSeams(t, []string{"cmd/fak/main.go"}, t.TempDir())
	prepushWorkflowExtractTip = func(string, string) (string, error) {
		t.Fatal("unrelated range must not archive or scan the tip")
		return "", nil
	}

	res, code := evaluatePrePushWorkflow("repo", "base", "tip")
	if code != 0 || res.Verdict != "NOOP" || !res.OK || res.Touched {
		t.Fatalf("unrelated range must be a clean skip: code=%d result=%+v", code, res)
	}
}

func TestPrepushWorkflowDirtyCheckoutCannotChangePushedTipVerdict(t *testing.T) {
	repo := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("config", "user.email", "workflow-test@example.invalid")
	git("config", "user.name", "Workflow Test")
	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "README.md")
	git("commit", "-q", "-m", "base")
	base := git("rev-parse", "HEAD")
	writeWorkflowFixture(t, repo, "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n")
	git("add", ".github/workflows/ci.yml")
	git("commit", "-q", "-m", "clean workflow")
	tip := git("rev-parse", "HEAD")

	// The checkout is now malformed, while the object actually being pushed remains clean.
	writeWorkflowFixture(t, repo, "jobs:\n  build:\n    runs-on: [ubuntu-latest\n")
	res, code := evaluatePrePushWorkflow(repo, base, tip)
	if code != 0 || res.Verdict != "OK" || !res.OK {
		t.Fatalf("dirty checkout changed immutable-tip verdict: code=%d result=%+v", code, res)
	}
}

func TestWorkflowCheckWarnsAboutGitHubWorkflowScopeEarly(t *testing.T) {
	tree := t.TempDir()
	writeWorkflowFixture(t, tree, "jobs:\n  build:\n    runs-on: [ubuntu-latest\n")
	withPrepushWorkflowSeams(t, []string{".github/workflows/ci.yml"}, tree)
	var stdout, stderr bytes.Buffer
	if code := workflowCheck(&stdout, &stderr, []string{"--root", tree, "--base", "base", "--tip", "tip"}); code != 1 {
		t.Fatalf("workflow check code=%d stderr=%s", code, stderr.String())
	}
	warning := stderr.String()
	if !strings.Contains(warning, "GitHub OAuth/PAT") || !strings.Contains(warning, "`workflow` scope") || !strings.Contains(warning, "before the server-side refusal") {
		t.Fatalf("scope warning is not early/actionable: %q", warning)
	}
	if scopeAt, findingAt := strings.Index(warning, "WORKFLOW_SCOPE warning:"), strings.Index(warning, "WORKFLOW_STRUCTURE "); scopeAt < 0 || findingAt < 0 || scopeAt > findingAt {
		t.Fatalf("scope warning must precede the structural refusal: %q", warning)
	}
}

func TestPrepushHookWorkflowRungIsIndependentOfBuildGuard(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test path")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	b, err := os.ReadFile(filepath.Join(root, "tools", "githooks", "pre-push"))
	if err != nil {
		t.Fatal(err)
	}
	hook := string(b)
	workflowAt := strings.Index(hook, "workflow_mode=\"")
	buildAt := strings.Index(hook, "build_mode=\"")
	if workflowAt < 0 || buildAt < 0 || workflowAt > buildAt {
		t.Fatalf("workflow rung must run independently before the build rung: workflow=%d build=%d", workflowAt, buildAt)
	}
	workflowRung := hook[workflowAt : strings.Index(hook[workflowAt:], "# Bulk remote-ref deletion gate")+workflowAt]
	if strings.Contains(workflowRung, "FLEET_BUILD_GUARD") {
		t.Fatal("workflow rung is incorrectly conditional on FLEET_BUILD_GUARD")
	}
}

func TestPrepushHookWorkflowPathGateBehavior(t *testing.T) {
	shell := gitHookShell(t)
	cases := []struct {
		name        string
		workflow    string
		unrelated   bool
		dirty       string
		newRef      bool
		wantBlock   bool
		wantWarning bool
	}{
		{name: "malformed block", workflow: "jobs:\n  build:\n    runs-on: [ubuntu-latest\n", wantBlock: true, wantWarning: true},
		{name: "clean pass", workflow: "jobs:\n  build:\n    runs-on: ubuntu-latest\n", wantWarning: true},
		{name: "unrelated skip", unrelated: true},
		{name: "dirty checkout isolation", workflow: "jobs:\n  build:\n    runs-on: ubuntu-latest\n", dirty: "jobs:\n  build:\n    runs-on: [ubuntu-latest\n", wantWarning: true},
		{name: "new ref scans full tip", workflow: "jobs:\n  build:\n    runs-on: [ubuntu-latest\n", newRef: true, wantBlock: true, wantWarning: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root, base, tip := makeHookWorkflowRepo(t, tc.workflow, tc.unrelated, tc.dirty)
			old := base
			if tc.newRef {
				old = strings.Repeat("0", 40)
			}
			out, code := runWorkflowHook(t, shell, root, tip, old)
			if blocked := code != 0; blocked != tc.wantBlock {
				t.Fatalf("blocked=%v want=%v code=%d\n%s", blocked, tc.wantBlock, code, out)
			}
			if warned := strings.Contains(out, "WORKFLOW_SCOPE warning:"); warned != tc.wantWarning {
				t.Fatalf("warning=%v want=%v\n%s", warned, tc.wantWarning, out)
			}
			if tc.wantBlock {
				warningAt := strings.Index(out, "WORKFLOW_SCOPE warning:")
				refusalAt := strings.Index(out, "WORKFLOW_STRUCTURE (blocked):")
				if warningAt < 0 || refusalAt < 0 || warningAt > refusalAt {
					t.Fatalf("operator warning must precede refusal\n%s", out)
				}
			}
		})
	}
}

// TestPrepushWorkflowHookHelper turns the current test binary into the fak executable used by
// the real shell hook above. All non-workflow commands are clean no-ops, keeping the witness
// focused on the workflow rung while the hook itself, stdin parsing, and operator output are real.
func TestPrepushWorkflowHookHelper(t *testing.T) {
	if os.Getenv("FAK_WORKFLOW_HOOK_HELPER") != "1" {
		return
	}
	sep := -1
	for i, arg := range os.Args {
		if arg == "--" {
			sep = i
			break
		}
	}
	if sep < 0 || sep+1 >= len(os.Args) {
		os.Exit(2)
	}
	args := os.Args[sep+1:]
	if len(args) >= 2 && args[0] == "workflow" && args[1] == "check" {
		os.Exit(workflowCheck(os.Stdout, os.Stderr, args[2:]))
	}
	os.Exit(0)
}

func gitHookShell(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		for _, path := range []string{`C:\Program Files\Git\usr\bin\sh.exe`, `C:\Program Files\Git\bin\bash.exe`} {
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
	}
	path, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("POSIX shell unavailable")
	}
	return path
}

func makeHookWorkflowRepo(t *testing.T, workflow string, unrelated bool, dirty string) (root, base, tip string) {
	t.Helper()
	root = t.TempDir()
	git := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("config", "user.email", "hook-test@example.invalid")
	git("config", "user.name", "Hook Test")
	git("symbolic-ref", "HEAD", "refs/heads/main")
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "README.md")
	git("commit", "-q", "-m", "base")
	base = git("rev-parse", "HEAD")
	if unrelated {
		if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("unrelated\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	} else {
		writeWorkflowFixture(t, root, workflow)
	}
	git("add", ".")
	git("commit", "-q", "-m", "tip")
	tip = git("rev-parse", "HEAD")
	if dirty != "" {
		writeWorkflowFixture(t, root, dirty)
	}
	return root, base, tip
}

func runWorkflowHook(t *testing.T, shell, repo, tip, old string) (string, int) {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	sourceRoot := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	hook := filepath.Join(sourceRoot, "tools", "githooks", "pre-push")
	bin := t.TempDir()
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	launcher := "#!/bin/sh\nexec '" + filepath.ToSlash(testBinary) + "' -test.run '^TestPrepushWorkflowHookHelper$' -- \"$@\"\n"
	if err := os.WriteFile(filepath.Join(bin, "fak"), []byte(launcher), 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(shell, "-x", filepath.ToSlash(hook))
	cmd.Dir = repo
	cmd.Stdin = strings.NewReader("refs/heads/main " + tip + " refs/heads/main " + old + "\n")
	cmd.Env = append(os.Environ(),
		"FAK_WORKFLOW_HOOK_HELPER=1",
		"FLEET_BUILD_GUARD=off",
		"FLEET_TIER_GUARD=off",
		"FLEET_POPUP_GUARD=off",
		"FLEET_REVIEW_GUARD=off",
		"PATH="+filepath.ToSlash(bin)+":/usr/bin:/bin:"+os.Getenv("PATH"),
	)
	out, runErr := cmd.CombinedOutput()
	if runErr == nil {
		return string(out), 0
	}
	if exitErr, ok := runErr.(*exec.ExitError); ok {
		return string(out), exitErr.ExitCode()
	}
	t.Fatalf("run hook: %v\n%s", runErr, out)
	return "", -1
}

func TestWorkflowStructureCurrentTreeZeroFindingsWitness(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test path")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	tipCmd := exec.Command("git", "rev-parse", "HEAD")
	tipCmd.Dir = root
	tipOut, err := tipCmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	tip := strings.TrimSpace(string(tipOut))
	findings, err := workflowlint.CheckTree(root)
	if err != nil {
		t.Fatalf("current tree %s: %v", tip, err)
	}
	if len(findings) != 0 {
		t.Fatalf("current tree %s has %d workflow structural findings; first=%+v", tip, len(findings), findings[0])
	}
	t.Logf("current-tree zero-findings witness: tip=%s workflows=.github/workflows/** findings=0", tip)
}
