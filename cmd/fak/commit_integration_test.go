package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// TestRunCommit_integration_realGit drives runCommit end to end against a real temporary
// git repo: it commits one file by path and asserts, via real git, that exactly that path
// landed and the Result verified. Hooks are disabled (the temp repo is not the fak clone)
// so this proves the executor's own assertion, not the fak commit-gate hooks.
func TestRunCommit_integration_realGit(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	emptyHooks := t.TempDir() // a hooks dir with no hooks => the gate hooks don't fire
	git := func(args ...string) (string, error) {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t",
		)
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if _, err := git("init", "-q", "-b", "main"); err != nil {
		// Older git without -b: init then rename.
		if _, e2 := git("init", "-q"); e2 != nil {
			t.Skipf("git init failed: %v", e2)
		}
		_, _ = git("symbolic-ref", "HEAD", "refs/heads/main")
	}
	if _, err := git("config", "core.hooksPath", emptyHooks); err != nil {
		t.Skipf("git config failed: %v", err)
	}
	// Persist an author identity in the repo config: runCommit shells out to the
	// real git binary through safecommit.Commit, which inherits ambient identity
	// only. A CI runner has none, so without this the commit-under-test fails with
	// "empty ident name" -> HOOK_REFUSED. (The seed commit below uses the helper's
	// GIT_*_NAME env, but safecommit.Commit does not see those.)
	if _, err := git("config", "user.email", "t@t"); err != nil {
		t.Skipf("git config failed: %v", err)
	}
	if _, err := git("config", "user.name", "t"); err != nil {
		t.Skipf("git config failed: %v", err)
	}
	// Seed an initial commit so HEAD exists.
	if err := os.WriteFile(filepath.Join(repo, "seed.txt"), []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := git("add", "seed.txt"); err != nil {
		t.Fatal(err)
	}
	if out, err := git("commit", "-qm", "seed"); err != nil {
		t.Skipf("seed commit failed (likely no user identity): %s", out)
	}

	// Now write the file under test and commit it by path through runCommit.
	target := "internal/foo/bar.txt"
	if err := os.MkdirAll(filepath.Join(repo, "internal", "foo"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, target), []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--json", "--dir", repo, "--no-signoff",
		"--path", target, "-m", "feat(foo): add bar (fak safecommit)",
	})
	if code != 0 {
		t.Fatalf("runCommit exit %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if !res.Committed || !res.Verified || res.Reason != "" {
		t.Fatalf("expected a clean verified commit, got %+v", res)
	}

	// Cross-check with real git: exactly the one path landed in HEAD.
	landed, err := git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
	if err != nil {
		t.Fatalf("diff-tree: %v", err)
	}
	names := strings.Fields(strings.TrimSpace(landed))
	if len(names) != 1 || names[0] != target {
		t.Fatalf("HEAD should contain exactly %q, got %v", target, names)
	}
}

// TestRunCommit_integration_nothingStaged proves the lock-free fast fail against real git.
func TestRunCommit_integration_nothingStaged(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	git := func(args ...string) (string, error) {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t",
		)
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if _, err := git("init", "-q", "-b", "main"); err != nil {
		if _, e2 := git("init", "-q"); e2 != nil {
			t.Skipf("git init failed: %v", e2)
		}
		_, _ = git("symbolic-ref", "HEAD", "refs/heads/main")
	}
	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := git("add", "a.txt"); err != nil {
		t.Fatal(err)
	}
	if out, err := git("commit", "-qm", "seed"); err != nil {
		t.Skipf("seed commit failed: %s", out)
	}

	// a.txt is unchanged since the seed commit => NOTHING_STAGED.
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--json", "--dir", repo, "--no-signoff", "--path", "a.txt", "-m", "noop",
	})
	if code != 3 {
		t.Fatalf("nothing-staged should exit 3, got %d (stdout=%q)", code, out.String())
	}
	var res safecommit.Result
	_ = json.Unmarshal(out.Bytes(), &res)
	if res.Reason != safecommit.ReasonNothingStaged {
		t.Fatalf("want NOTHING_STAGED, got %q", res.Reason)
	}
}
