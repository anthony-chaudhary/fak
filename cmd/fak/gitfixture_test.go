package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// gitfixture_test.go — the temp-git-repo fixture the runtime verb tests share.
//
// These helpers used to live beside `fak ci-preflight`, which moved to `fak-dev`
// (#6022). The verbs that still need a real git workspace — `fak session checkpoint`
// and `fak validate` — are runtime-owned and stay here, so the fixture they depend on
// is re-homed here rather than reached across into the dev artifact's test files.

// seedGitFixtureRepo returns a temp repo with an initial commit and a git() helper.
// Hooks are pointed at an empty dir so the commit-gate hooks don't fire in-fixture.
func seedGitFixtureRepo(t *testing.T) (repo string, git func(args ...string) (string, error)) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo = t.TempDir()
	emptyHooks := t.TempDir()
	git = func(args ...string) (string, error) {
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
	if _, err := git("config", "core.hooksPath", emptyHooks); err != nil {
		t.Skipf("git config failed: %v", err)
	}
	_, _ = git("config", "user.name", "t")
	_, _ = git("config", "user.email", "t@t")
	return repo, git
}

// commitFiles writes files (path->content, relative to repo) and commits them by
// add-all. The add-all is scoped to the throwaway fixture repo, never the checkout.
func commitFiles(t *testing.T, repo string, git func(args ...string) (string, error), msg string, files map[string]string) {
	t.Helper()
	for rel, content := range files {
		full := filepath.Join(repo, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if out, err := git("add", "-A"); err != nil {
		t.Fatalf("git add: %s", out)
	}
	if out, err := git("commit", "-qm", msg); err != nil {
		t.Skipf("commit failed (likely no git identity): %s", out)
	}
}

// A clean, gofmt-formatted, self-contained module — its committed tip must verify OK.
const cleanGoMod = "module gitfixture.test\n\ngo 1.26\n"
const cleanGoFile = "package p\n\n// Add returns a + b.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
