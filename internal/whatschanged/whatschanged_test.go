package whatschanged

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWhatsChangedPreviewListsIntersectingCommitsAndFiles(t *testing.T) {
	repo := whatsChangedRepo(t)
	base := whatsChangedGitOut(t, repo, "rev-parse", "HEAD")

	whatsChangedWrite(t, filepath.Join(repo, "docs", "readme.md"), "docs\n")
	whatsChangedGit(t, repo, "add", "docs/readme.md")
	whatsChangedGit(t, repo, "commit", "-m", "docs only", "-q")
	whatsChangedWrite(t, filepath.Join(repo, "cmd", "fak", "main.go"), "package main\nfunc changed() {}\n")
	whatsChangedGit(t, repo, "add", "cmd/fak/main.go")
	whatsChangedGit(t, repo, "commit", "-m", "touch target", "-q")

	rep, err := Preview(context.Background(), repo, Options{Since: base, Paths: []string{"cmd/fak/*.go"}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Empty {
		t.Fatalf("report is empty, want target commit")
	}
	if len(rep.Commits) != 1 || rep.Commits[0].Subject != "touch target" {
		t.Fatalf("commits = %+v, want only target commit", rep.Commits)
	}
	if got := strings.Join(rep.Commits[0].Files, ","); got != "cmd/fak/main.go" {
		t.Fatalf("commit files = %q", got)
	}
	if got := strings.Join(rep.ChangedFiles, ","); got != "cmd/fak/main.go" {
		t.Fatalf("changed files = %q", got)
	}
}

func TestWhatsChangedPreviewEmptyWhenPathsUntouched(t *testing.T) {
	repo := whatsChangedRepo(t)
	base := whatsChangedGitOut(t, repo, "rev-parse", "HEAD")
	whatsChangedWrite(t, filepath.Join(repo, "docs", "readme.md"), "docs\n")
	whatsChangedGit(t, repo, "add", "docs/readme.md")
	whatsChangedGit(t, repo, "commit", "-m", "docs only", "-q")

	rep, err := Preview(context.Background(), repo, Options{Since: base, Paths: []string{"cmd/fak/*.go"}})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Empty || len(rep.Commits) != 0 || len(rep.ChangedFiles) != 0 {
		t.Fatalf("report = %+v, want empty", rep)
	}
}

func TestWhatsChangedPreviewRequiresPaths(t *testing.T) {
	_, err := Preview(context.Background(), t.TempDir(), Options{Since: "HEAD"})
	if err == nil || !strings.Contains(err.Error(), "at least one --paths") {
		t.Fatalf("err = %v, want paths error", err)
	}
}

func whatsChangedRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	whatsChangedGit(t, repo, "init", "-q", "-b", "main")
	whatsChangedGit(t, repo, "config", "core.autocrlf", "false")
	whatsChangedGit(t, repo, "config", "user.name", "test")
	whatsChangedGit(t, repo, "config", "user.email", "test@example.com")
	whatsChangedWrite(t, filepath.Join(repo, "cmd", "fak", "main.go"), "package main\nfunc main() {}\n")
	whatsChangedWrite(t, filepath.Join(repo, "docs", "readme.md"), "base\n")
	whatsChangedGit(t, repo, "add", ".")
	whatsChangedGit(t, repo, "commit", "-m", "base", "-q")
	return repo
}

func whatsChangedGit(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func whatsChangedGitOut(t *testing.T, cwd string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git %v in %s: %v", args, cwd, err)
	}
	return strings.TrimSpace(string(out))
}

func whatsChangedWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
