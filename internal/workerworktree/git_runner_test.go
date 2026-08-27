package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultGitKeepsSuccessfulStderrOutOfPatchStdout(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	repo := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	git("init", "-q")
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	path := filepath.Join(repo, "file.txt")
	if err := os.WriteFile(path, []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "file.txt")
	git("commit", "-qm", "base")
	if err := os.WriteFile(path, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	code, patch := defaultGit(repo, []string{"-c", "core.autocrlf=true", "diff"})
	if code != 0 || !strings.HasPrefix(patch, "diff --git ") {
		t.Fatalf("defaultGit diff code=%d output=%q", code, patch)
	}
	if strings.Contains(patch, "warning:") {
		t.Fatalf("successful stderr contaminated patch stdout: %q", patch)
	}
	patchPath := filepath.Join(t.TempDir(), "captured.patch")
	if err := os.WriteFile(patchPath, []byte(patch), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if code, detail := defaultGit(repo, []string{"apply", "--check", patchPath}); code != 0 {
		t.Fatalf("captured patch is not parseable: %s", detail)
	}
}
