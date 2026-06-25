//go:build linux && (amd64 || arm64)

package guard

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestLandlockHookFloorDeniesHookWrites is the LOAD-BEARING acceptance test: it proves the
// union-correct ruleset actually makes .git/hooks read-only while the rest of the tree stays
// writable. It can only run on a real kernel >= 5.13 with Landlock enabled; it SKIPS (never
// fails) when Landlock is unavailable, so it is a CI gate on a capable Linux runner, not a
// hard dep of every build.
//
// Mechanism: it builds a temp git-like tree, computes a RulesetSpec, then re-execs THIS test
// binary as a tiny "probe" helper (via the FAK_LANDLOCK_PROBE env trip) under the trampoline.
// The probe applies the hook-floor to itself and writes to three paths, printing which
// succeeded. The parent asserts: hook write DENIED, .git/refs write + worktree write ALLOWED.
func TestLandlockHookFloorDeniesHookWrites(t *testing.T) {
	if !landlockAvailable() {
		t.Skip("Landlock unavailable on this kernel; acceptance test is a CI gate on a capable runner")
	}

	root := t.TempDir()
	mustMkdir(t, filepath.Join(root, ".git", "hooks"))
	mustMkdir(t, filepath.Join(root, ".git", "refs", "heads"))
	mustMkdir(t, filepath.Join(root, "src"))
	// Pre-create the .git loose files + a repo-root file so the probe overwrites EXISTING files
	// (the realistic case: safecommit writes .git/index and .git/HEAD that already exist).
	mustWrite(t, filepath.Join(root, ".git", "index"))
	mustWrite(t, filepath.Join(root, ".git", "HEAD"))
	mustWrite(t, filepath.Join(root, "README.md"))
	mustWrite(t, filepath.Join(root, ".git", "hooks", "pre-commit"))

	spec := RulesetSpec{
		RepoRoot:     root,
		GitDir:       filepath.Join(root, ".git"),
		ReadOnlyDirs: []string{filepath.Join(root, ".git", "hooks")},
	}

	probeBin, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}

	cmd := exec.Command(probeBin, "-test.run=TestLandlockProbeHelper")
	cmd.Env = append(os.Environ(),
		"FAK_LANDLOCK_PROBE=1",
		"FAK_PROBE_SPEC="+spec.Encode(),
		"FAK_PROBE_HOOK="+filepath.Join(root, ".git", "hooks", "pre-commit"),
		"FAK_PROBE_HOOKNEW="+filepath.Join(root, ".git", "hooks", "post-commit"), // a NEW hook file
		"FAK_PROBE_REF="+filepath.Join(root, ".git", "refs", "heads", "probe"),
		"FAK_PROBE_INDEX="+filepath.Join(root, ".git", "index"),
		"FAK_PROBE_HEAD="+filepath.Join(root, ".git", "HEAD"),
		"FAK_PROBE_README="+filepath.Join(root, "README.md"),
		"FAK_PROBE_TREE="+filepath.Join(root, "src", "probe.txt"),
	)
	out, _ := cmd.CombinedOutput()
	got := string(out)

	// The four union-semantics assertions: hooks read-only, everything else writable.
	mustContain(t, got, "HOOK=denied", "an EXISTING .git/hooks file write must be DENIED")
	mustContain(t, got, "HOOKNEW=denied", "creating a NEW file in .git/hooks must be DENIED")
	mustContain(t, got, "REF=ok", "a .git/refs subdir write must SUCCEED")
	mustContain(t, got, "INDEX=ok", "overwriting .git/index (loose file) must SUCCEED — safecommit needs it")
	mustContain(t, got, "HEAD=ok", "overwriting .git/HEAD (loose file) must SUCCEED")
	mustContain(t, got, "README=ok", "a repo-root regular-file write must SUCCEED")
	mustContain(t, got, "TREE=ok", "a worktree subdir write must SUCCEED")
}

// TestLandlockInTreeHooksBlockJailedGitCommit documents the FUNDAMENTAL limitation (not a
// bug): when hooks are the default .git/hooks (UNDER .git), a jailed process cannot create a
// new file directly in .git/ — including git's own .git/index.lock — so a jailed `git commit`
// is blocked. This is by design: fak guard channels commits through the UNjailed parent
// (safecommit), not raw git in the sandbox. The test asserts the block so the boundary is a
// witnessed contract, not an accident.
func TestLandlockInTreeHooksBlockJailedGitCommit(t *testing.T) {
	if !landlockAvailable() {
		t.Skip("Landlock unavailable on this kernel")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := initGitRepo(t)
	gitDir := filepath.Join(root, ".git")
	spec := RulesetSpec{RepoRoot: root, GitDir: gitDir, ReadOnlyDirs: []string{filepath.Join(gitDir, "hooks")}}

	got := runCommitProbe(t, spec, root)
	mustContain(t, got, "COMMIT=denied", "a jailed git commit must be BLOCKED when hooks are in-tree (.git/index.lock not creatable) — the documented limitation")
}

// TestLandlockOutOfTreeHooksAllowJailedGitCommit proves the RELAXATION: when hooks live OUT
// of .git (core.hooksPath points to a sibling), .git is not on the deny chain, so it gets a
// full grant and a jailed `git commit` ALSO lands — while the out-of-tree hooks dir stays
// read-only. This is the one configuration where in-jail commit and hook protection coexist.
func TestLandlockOutOfTreeHooksAllowJailedGitCommit(t *testing.T) {
	if !landlockAvailable() {
		t.Skip("Landlock unavailable on this kernel")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := initGitRepo(t)
	gitDir := filepath.Join(root, ".git")
	// Hooks live at <repo>/.githooks — a SIBLING of .git, not under it.
	outHooks := filepath.Join(root, ".githooks")
	mustMkdir(t, outHooks)
	spec := RulesetSpec{RepoRoot: root, GitDir: gitDir, ReadOnlyDirs: []string{outHooks}}

	got := runCommitProbe(t, spec, root)
	mustContain(t, got, "COMMIT=ok", "with out-of-tree hooks, .git is fully writable so a jailed git commit must LAND")

	logOut, _ := exec.Command("git", "-C", root, "log", "--oneline", "-1").CombinedOutput()
	if len(strings.TrimSpace(string(logOut))) == 0 {
		t.Fatalf("expected a commit after the jailed git commit (out-of-tree hooks); git log empty.\nprobe:\n%s", got)
	}
}

// initGitRepo creates a temp git repo with a staged-able file and returns its root.
func initGitRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
		{"config", "commit.gpgsign", "false"},
	} {
		c := exec.Command("git", args...)
		c.Dir = root
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	mustWrite(t, filepath.Join(root, "file.txt"))
	return root
}

// runCommitProbe re-execs the probe in gitcommit mode under the given spec and returns its
// output (COMMIT=ok or COMMIT=denied ...).
func runCommitProbe(t *testing.T, spec RulesetSpec, repo string) string {
	t.Helper()
	probeBin, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	cmd := exec.Command(probeBin, "-test.run=TestLandlockProbeHelper")
	cmd.Env = append(os.Environ(),
		"FAK_LANDLOCK_PROBE=1",
		"FAK_PROBE_MODE=gitcommit",
		"FAK_PROBE_SPEC="+spec.Encode(),
		"FAK_PROBE_REPO="+repo,
	)
	out, _ := cmd.CombinedOutput()
	return string(out)
}

// TestLandlockProbeHelper is the re-exec'd child. It is a normal test that does nothing unless
// FAK_LANDLOCK_PROBE is set, in which case it applies the hook-floor to itself and probes the
// requested mode. It prints results and exits without returning to the test framework.
func TestLandlockProbeHelper(t *testing.T) {
	if os.Getenv("FAK_LANDLOCK_PROBE") != "1" {
		return // ordinary run: this helper is inert
	}
	if os.Getenv("FAK_PROBE_MODE") == "gitcommit" {
		probeGitCommit()
		return
	}
	spec, err := DecodeSpec(os.Getenv("FAK_PROBE_SPEC"))
	if err != nil {
		os.Stdout.WriteString("SPEC=bad\n")
		os.Exit(3)
	}
	applyHookFloor(spec) // restricts this thread; logs+fails-open internally

	// report opens path for write (creating if absent) and prints ok/denied. Used for both
	// "overwrite an existing file" and "create a new file" cases via O_CREATE.
	report := func(label, path string) {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			os.Stdout.WriteString(label + "=denied\n")
			return
		}
		_, werr := f.WriteString("x")
		f.Close()
		if werr != nil {
			os.Stdout.WriteString(label + "=denied\n")
			return
		}
		os.Stdout.WriteString(label + "=ok\n")
	}
	report("HOOK", os.Getenv("FAK_PROBE_HOOK"))       // overwrite existing hook → must deny
	report("HOOKNEW", os.Getenv("FAK_PROBE_HOOKNEW")) // create new hook → must deny
	report("REF", os.Getenv("FAK_PROBE_REF"))         // .git/refs subdir → ok
	report("INDEX", os.Getenv("FAK_PROBE_INDEX"))     // .git/index loose file → ok
	report("HEAD", os.Getenv("FAK_PROBE_HEAD"))       // .git/HEAD loose file → ok
	report("README", os.Getenv("FAK_PROBE_README"))   // repo-root regular file → ok
	report("TREE", os.Getenv("FAK_PROBE_TREE"))       // worktree subdir → ok
	os.Stdout.Sync()
	os.Exit(0)
}

// TestLandlockFailOpenWhenUnsupported documents the fail-open posture: applyHookFloor never
// panics and never aborts even when given a spec, regardless of kernel support — on an
// unsupported kernel it logs and returns, leaving the caller to exec unrestricted. We can
// only positively assert "does not panic / returns" here; the deny behavior is the test above.
func TestLandlockApplyDoesNotPanicOnEmptySpec(t *testing.T) {
	applyHookFloor(RulesetSpec{}) // empty spec → logs "no hook dir" and returns
}

// probeGitCommit applies the hook-floor, then runs `git add` + `git commit` in the repo. It
// reports COMMIT=ok only if the commit lands — proving .git/index, .git/objects, .git/refs,
// and .git/HEAD are writable under the floor. Runs as the re-exec'd probe subprocess.
func probeGitCommit() {
	spec, err := DecodeSpec(os.Getenv("FAK_PROBE_SPEC"))
	if err != nil {
		os.Stdout.WriteString("SPEC=bad\n")
		os.Exit(3)
	}
	repo := os.Getenv("FAK_PROBE_REPO")
	applyHookFloor(spec)

	run := func(args ...string) bool {
		c := exec.Command("git", args...)
		c.Dir = repo
		out, err := c.CombinedOutput()
		if err != nil {
			os.Stdout.WriteString("COMMIT=denied (" + strings.TrimSpace(string(out)) + ")\n")
			return false
		}
		return true
	}
	if !run("add", "file.txt") {
		os.Exit(0)
	}
	if !run("commit", "-m", "jailed commit", "--no-gpg-sign") {
		os.Exit(0)
	}
	os.Stdout.WriteString("COMMIT=ok\n")
	os.Stdout.Sync()
	os.Exit(0)
}

func landlockAvailable() bool {
	v, errno := probeABI()
	return v >= 1 && errno == 0
}

func mustMkdir(t *testing.T, p string) {
	t.Helper()
	if err := os.MkdirAll(p, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", p, err)
	}
}

func mustWrite(t *testing.T, p string) {
	t.Helper()
	if err := os.WriteFile(p, []byte("seed"), 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
}

func mustContain(t *testing.T, out, want, why string) {
	t.Helper()
	if !strings.Contains(out, want) {
		t.Fatalf("%s (want %q); probe output:\n%s", why, want, out)
	}
}
