package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// The pre-commit hook is driven through a real `git commit` in a throwaway
// repository. go, python and fak-run are shell stubs on PATH so the hook's
// control flow is observed without building the real gates.

const hookStubGo = `#!/bin/sh
case " $* " in
*" ./cmd/fak-dev boundary"*)
	if [ -f .hooktest-worktree-marker ]; then echo worktree; else echo other; fi > "$HOOKTEST_LOG/boundary.cwd"
	exit "${HOOKTEST_BOUNDARY_RC:-0}"
	;;
*" ./cmd/fak-boundary "*)
	echo "${GIT_INDEX_FILE-unset}" > "$HOOKTEST_LOG/companion-boundary.index"
	exit 0
	;;
esac
exit 0
`

const hookStubPython = `#!/bin/sh
git diff --cached --name-only > "$HOOKTEST_LOG/audit.staged"
if grep -q violation "$HOOKTEST_LOG/audit.staged"; then
	echo "stub audit: violation staged" >&2
	exit 1
fi
exit 0
`

const hookStubFakRun = "#!/bin/sh\nexit 0\n"

type hookFixture struct {
	t     *testing.T
	main  string
	logs  string
	env   []string
	hooks string
}

func newHookFixture(t *testing.T) *hookFixture {
	t.Helper()
	if runtime.GOOS == "plan9" {
		t.Skip("no sh hooks on plan9")
	}
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not on PATH")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	hookSrc, err := os.ReadFile(filepath.Join("..", "..", ".githooks", "pre-commit"))
	if err != nil {
		t.Fatalf("read hook: %v", err)
	}

	base := t.TempDir()
	f := &hookFixture{
		t:    t,
		main: filepath.Join(base, "main"),
		logs: filepath.Join(base, "logs"),
	}
	stubs := filepath.Join(base, "stubs")
	for _, d := range []string{f.main, f.logs, stubs} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, body := range map[string]string{"go": hookStubGo, "python": hookStubPython, "python3": hookStubPython, "fak-run": hookStubFakRun} {
		if err := os.WriteFile(filepath.Join(stubs, name), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	f.hooks = filepath.Join(f.main, ".githooks")
	f.write(f.main, ".githooks/pre-commit", string(hookSrc), 0o755)
	f.write(f.main, "cmd/fak-dev/main.go", "package main\n\nfunc main() {}\n", 0o644)

	f.env = append(hookTestCleanEnv(),
		"PATH="+stubs+string(os.PathListSeparator)+os.Getenv("PATH"),
		"HOOKTEST_LOG="+f.logs,
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL="+filepath.Join(base, "gitconfig"),
		"GIT_AUTHOR_NAME=hooktest", "GIT_AUTHOR_EMAIL=hooktest@example.invalid",
		"GIT_COMMITTER_NAME=hooktest", "GIT_COMMITTER_EMAIL=hooktest@example.invalid",
	)
	f.git(f.main, nil, "init", "-q")
	f.git(f.main, nil, "add", "-A")
	f.git(f.main, nil, "commit", "-q", "--no-verify", "-m", "base")
	f.git(f.main, nil, "config", "core.hooksPath", filepath.ToSlash(f.hooks))
	return f
}

// hookTestCleanEnv drops the outer git and gate configuration so the hook sees
// only what each test sets.
func hookTestCleanEnv() []string {
	var out []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(strings.ToUpper(k), "GIT_") || strings.EqualFold(k, "FAK_PRIVATE_ROOT") || strings.EqualFold(k, "PATH") {
			continue
		}
		out = append(out, kv)
	}
	return out
}

func (f *hookFixture) write(dir, rel, body string, mode os.FileMode) {
	f.t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		f.t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), mode); err != nil {
		f.t.Fatal(err)
	}
}

func (f *hookFixture) run(dir string, extra []string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(append([]string{}, f.env...), extra...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func (f *hookFixture) git(dir string, extra []string, args ...string) string {
	f.t.Helper()
	out, err := f.run(dir, extra, args...)
	if err != nil {
		f.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return out
}

func (f *hookFixture) readLog(name string) string {
	f.t.Helper()
	b, err := os.ReadFile(filepath.Join(f.logs, name))
	if err != nil {
		f.t.Fatalf("hook never wrote %s: %v", name, err)
	}
	return strings.TrimSpace(string(b))
}

func TestPreCommitHookStagedAuditReadsAlternateIndex(t *testing.T) {
	f := newHookFixture(t)
	companion := filepath.Join(filepath.Dir(f.main), "companion")
	f.write(companion, "tools/scrub_public_copy.py", "# stubbed via PATH\n", 0o644)
	f.write(f.main, "violation.txt", "private\n", 0o644)

	altIndex := filepath.Join(filepath.Dir(f.main), "alt.index")
	alt := []string{"GIT_INDEX_FILE=" + altIndex, "FAK_PRIVATE_ROOT=" + companion}
	f.git(f.main, alt, "read-tree", "HEAD")
	f.git(f.main, alt, "add", "violation.txt")

	out, err := f.run(f.main, alt, "commit", "-q", "-m", "violating partial commit")
	if err == nil {
		t.Fatalf("hook admitted a violation staged only in the alternate index; audit saw %q\n%s", f.readLog("audit.staged"), out)
	}
	if got := f.readLog("audit.staged"); !strings.Contains(got, "violation.txt") {
		t.Fatalf("staged audit read the wrong index: saw %q", got)
	}
	if !strings.Contains(out, "Secret leak") {
		t.Fatalf("refusal should come from the staged audit:\n%s", out)
	}
}

func TestPreCommitHookCompanionChildrenGetIsolatedGitEnv(t *testing.T) {
	f := newHookFixture(t)
	companion := filepath.Join(filepath.Dir(f.main), "companion")
	f.write(companion, "tools/scrub_public_copy.py", "# stubbed via PATH\n", 0o644)
	f.write(f.main, "ok.txt", "fine\n", 0o644)

	altIndex := filepath.Join(filepath.Dir(f.main), "alt.index")
	alt := []string{"GIT_INDEX_FILE=" + altIndex, "FAK_PRIVATE_ROOT=" + companion}
	f.git(f.main, alt, "read-tree", "HEAD")
	f.git(f.main, alt, "add", "ok.txt")
	if out, err := f.run(f.main, alt, "commit", "-q", "-m", "clean"); err != nil {
		t.Fatalf("clean alternate-index commit refused: %v\n%s", err, out)
	}
	if got := f.readLog("audit.staged"); !strings.Contains(got, "ok.txt") {
		t.Fatalf("staged audit read the wrong index: saw %q", got)
	}
	if got := f.readLog("companion-boundary.index"); got != "unset" {
		t.Fatalf("companion go child must not inherit the public index, got GIT_INDEX_FILE=%q", got)
	}
}

func TestPreCommitHookStaysInLinkedWorktree(t *testing.T) {
	f := newHookFixture(t)
	wt := filepath.Join(filepath.Dir(f.main), "linked")
	f.git(f.main, nil, "worktree", "add", "-q", "-b", "lane", wt)
	f.write(wt, ".hooktest-worktree-marker", "", 0o644)
	f.write(wt, "change.txt", "lane\n", 0o644)
	standalone := []string{"FAK_PRIVATE_ROOT=none"}
	f.git(wt, standalone, "add", "change.txt")

	if out, err := f.run(wt, standalone, "commit", "-q", "-m", "lane change"); err != nil {
		t.Fatalf("worktree commit refused: %v\n%s", err, out)
	}
	if got := f.readLog("boundary.cwd"); got != "worktree" {
		t.Fatalf("hook ran its checks outside the active worktree (cwd=%s)", got)
	}
}

func TestPreCommitHookLinkedWorktreeFindsMainCheckoutCompanion(t *testing.T) {
	f := newHookFixture(t)
	companion := filepath.Join(filepath.Dir(f.main), "fak-private")
	f.write(companion, "tools/scrub_public_copy.py", "# stubbed via PATH\n", 0o644)
	wt := filepath.Join(f.main, ".worktrees", "ticket-relci-hook-companion")
	if err := os.MkdirAll(filepath.Dir(wt), 0o755); err != nil {
		t.Fatal(err)
	}
	f.git(f.main, nil, "worktree", "add", "-q", "-b", "lane", wt)
	f.write(wt, "change.txt", "lane\n", 0o644)
	f.git(wt, nil, "add", "change.txt")

	// The hook's shebang names bash; plain sh is dash on Debian-family hosts.
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not on PATH")
	}
	cmd := exec.Command("bash", filepath.Join(".githooks", "pre-commit"))
	cmd.Dir = wt
	cmd.Env = f.env
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("linked-worktree hook refused a clean change: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "Companion fak-private detected") {
		t.Fatalf("linked worktree did not resolve the main checkout's sibling companion:\n%s", out)
	}
	if got := f.readLog("audit.staged"); !strings.Contains(got, "change.txt") {
		t.Fatalf("companion staged audit did not run on the lane change: saw %q", got)
	}
}

func TestPreCommitHookBoundaryFailureBlocks(t *testing.T) {
	f := newHookFixture(t)
	f.write(f.main, "change.txt", "x\n", 0o644)
	standalone := []string{"FAK_PRIVATE_ROOT=none"}
	f.git(f.main, standalone, "add", "change.txt")

	out, err := f.run(f.main, append(standalone, "HOOKTEST_BOUNDARY_RC=3"), "commit", "-q", "-m", "red boundary")
	if err == nil {
		t.Fatalf("hook admitted a commit while the boundary check failed:\n%s", out)
	}
	if !strings.Contains(out, "COMMIT BLOCKED") {
		t.Fatalf("refusal should name the boundary block:\n%s", out)
	}
	if out, err := f.run(f.main, standalone, "commit", "-q", "-m", "green boundary"); err != nil {
		t.Fatalf("green boundary check must admit: %v\n%s", err, out)
	}
}

func TestPreCommitHookPowerShellParity(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", ".githooks", "pre-commit.ps1"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(src)
	audit := strings.Index(s, "scrub_public_copy.py")
	if audit < 0 {
		t.Fatal("staged audit missing from pre-commit.ps1")
	}
	if i := strings.Index(s, "Env:\\GIT_INDEX_FILE"); i >= 0 && i < audit {
		t.Fatal("pre-commit.ps1 clears GIT_INDEX_FILE before the staged audit")
	}
	if !strings.Contains(s, "rev-parse --show-toplevel") {
		t.Fatal("pre-commit.ps1 must resolve the active worktree with git rev-parse --show-toplevel")
	}
	if !strings.Contains(s, "rev-parse --git-common-dir") {
		t.Fatal("pre-commit.ps1 must resolve the main checkout with git rev-parse --git-common-dir for linked worktrees")
	}
	boundary := strings.Index(s, "& go run ./cmd/fak-dev boundary")
	if boundary < 0 || !strings.Contains(s[boundary:min(len(s), boundary+200)], "$LASTEXITCODE -ne 0") {
		t.Fatal("pre-commit.ps1 must fail closed on the public boundary exit code")
	}
}
