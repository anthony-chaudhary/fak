package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=3s lane=default
func TestCandidateSessionMovesOneCheckoutAcrossCASAttempts(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "trunk@test")
	mustGit(t, root, "config", "user.name", "trunk")
	mustGit(t, root, "config", "commit.gpgsign", "false")
	commit := func(content string) string {
		if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		mustGit(t, root, "add", "a.txt")
		mustGit(t, root, "commit", "-q", "-m", content)
		return strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
	}
	first, second := commit("one"), commit("two")

	type call struct {
		dir  string
		args []string
	}
	var calls []call
	git := func(dir string, args []string) (int, string) {
		calls = append(calls, call{dir: dir, args: append([]string(nil), args...)})
		return defaultGit(dir, args)
	}

	var dirs, contents []string
	var droppingSurvived bool
	hook := func(dir string) (bool, string) {
		dirs = append(dirs, dir)
		b, err := os.ReadFile(filepath.Join(dir, "a.txt"))
		if err != nil {
			return false, err.Error()
		}
		contents = append(contents, strings.TrimSpace(string(b)))
		if _, err := os.Stat(filepath.Join(dir, "dropping.out")); err == nil {
			droppingSurvived = true
		}
		if err := os.WriteFile(filepath.Join(dir, "dropping.out"), []byte("x"), 0o644); err != nil {
			return false, err.Error()
		}
		return true, ""
	}

	cand := newTopologyCandidateSession(root, git)
	if ok, detail := cand.verify(first, "", hook); !ok {
		t.Fatalf("first verify: %s", detail)
	}
	if ok, detail := cand.verify(second, "", hook); !ok {
		t.Fatalf("second verify: %s", detail)
	}

	if len(dirs) != 2 || dirs[0] != dirs[1] {
		t.Fatalf("verify dirs = %q, want one reused checkout", dirs)
	}
	if !slices.Equal(contents, []string{"one", "two"}) {
		t.Fatalf("verified contents = %q, want each attempt's exact commit", contents)
	}
	if droppingSurvived {
		t.Fatal("untracked output from the first attempt reached the second attempt's tree")
	}
	adds, moved := 0, false
	for _, c := range calls {
		if slices.Contains(c.args, "worktree") && slices.Contains(c.args, "add") {
			adds++
		}
		if c.dir == dirs[0] && slices.Equal(c.args, []string{"-c", "core.longpaths=true", "checkout", "--detach", "--force", second}) {
			moved = true
		}
	}
	if adds != 1 {
		t.Fatalf("worktree add ran %d times, want 1", adds)
	}
	if !moved {
		t.Fatal("second attempt did not check the reused candidate out at the new commit")
	}

	cand.close()
	if _, err := os.Stat(dirs[0]); !os.IsNotExist(err) {
		t.Fatalf("candidate %q remains after close: %v", dirs[0], err)
	}
	if got := registeredCandidates(t, root); len(got) != 0 {
		t.Fatalf("candidate registrations remain after close: %v", got)
	}
}
