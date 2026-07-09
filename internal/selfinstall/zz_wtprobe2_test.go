package selfinstall

// THROWAWAY empirical probe #2 — detached worktree (.git=file). DELETE before committing.

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func wtRun(t *testing.T, dir, name string, args ...string) (string, bool) {
	t.Helper()
	cmd := exec.CommandContext(context.Background(), name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

func TestZZWorktreeProbe(t *testing.T) {
	rootOut, _ := wtRun(t, ".", "git", "rev-parse", "--show-toplevel")
	repoRoot := strings.TrimSpace(rootOut)

	wt := filepath.Join(t.TempDir(), "wt")
	addOut, addOK := wtRun(t, repoRoot, "git", "worktree", "add", "--detach", wt, "HEAD")
	t.Logf("worktree add ok=%v out=%q", addOK, strings.TrimSpace(addOut))
	if !addOK {
		t.Logf("WORKTREE-ADD-BLOCKED — cannot test .git=file path")
		return
	}
	// confirm .git is a FILE (pointer), the worktree signature
	fi, _ := os.Stat(filepath.Join(wt, ".git"))
	t.Logf("worktree .git is-dir=%v", fi != nil && fi.IsDir())
	defer wtRun(t, repoRoot, "git", "worktree", "remove", "--force", wt)

	for _, tc := range []struct {
		label string
		args  []string
	}{
		{"AUTO(default)", []string{"build", "-o", "", "./cmd/fak"}},
		{"BUILDVCS=true", []string{"build", "-buildvcs=true", "-o", "", "./cmd/fak"}},
	} {
		bin := filepath.Join(t.TempDir(), "b.exe")
		args := append([]string{}, tc.args...)
		for i := range args {
			if args[i] == "" {
				args[i] = bin
			}
		}
		out, ok := wtRun(t, wt, "go", args...)
		stamped := "n/a"
		if ok {
			s, _ := wtRun(t, "", bin, "version", "--json")
			stamped = "stamped=" + boolStr(strings.Contains(s, "\"stamped\": true"))
		}
		t.Logf("[%s] build_ok=%v %s out=%q", tc.label, ok, stamped, strings.TrimSpace(truncate(out, 200)))
	}
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
