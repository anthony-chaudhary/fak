package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

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
		"--test-only",
		"--wsl-tests",
		"--json",
	})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if len(res.Tested) < 6 {
		t.Fatalf("tested=%v; want at least six Git-aware affected packages", res.Tested)
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

func TestValidateIncludesReverseDependencyTests(t *testing.T) {
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
	if code != 1 || res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	if !validateContains(res.Tested, "validate.test/consumer") {
		t.Fatalf("tested=%v; reverse dependency absent", res.Tested)
	}
	found := false
	for _, failure := range res.Failures {
		if failure.Step == "test" {
			found = true
		}
	}
	if !found {
		t.Fatalf("failures=%+v; want affected test failure", res.Failures)
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
