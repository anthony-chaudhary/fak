package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/whatschanged"
)

func TestRunWhatsChangedReportsIntersectingCommit(t *testing.T) {
	repo := whatsChangedCLIRepo(t)
	base := whatsChangedCLIGitOut(t, repo, "rev-parse", "HEAD")
	whatsChangedCLIWrite(t, filepath.Join(repo, "docs", "readme.md"), "docs\n")
	whatsChangedCLIGit(t, repo, "add", "docs/readme.md")
	whatsChangedCLIGit(t, repo, "commit", "-m", "docs only", "-q")
	whatsChangedCLIWrite(t, filepath.Join(repo, "cmd", "fak", "main.go"), "package main\nfunc changed() {}\n")
	whatsChangedCLIGit(t, repo, "add", "cmd/fak/main.go")
	whatsChangedCLIGit(t, repo, "commit", "-m", "touch target", "-q")

	var out, errb bytes.Buffer
	code := runWhatsChanged(&out, &errb, []string{"--dir", repo, "--since", base, "--paths", "cmd/fak/*.go"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "touch target") || !strings.Contains(got, "cmd/fak/main.go") {
		t.Fatalf("output missed target commit/file:\n%s", got)
	}
	if strings.Contains(got, "docs/readme.md") {
		t.Fatalf("output included non-intersecting file:\n%s", got)
	}
}

func TestRunWhatsChangedJSONEmpty(t *testing.T) {
	repo := whatsChangedCLIRepo(t)
	base := whatsChangedCLIGitOut(t, repo, "rev-parse", "HEAD")
	whatsChangedCLIWrite(t, filepath.Join(repo, "docs", "readme.md"), "docs\n")
	whatsChangedCLIGit(t, repo, "add", "docs/readme.md")
	whatsChangedCLIGit(t, repo, "commit", "-m", "docs only", "-q")

	var out, errb bytes.Buffer
	code := runWhatsChanged(&out, &errb, []string{"--dir", repo, "--since", base, "--paths", "cmd/fak/*.go", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var rep whatschanged.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json did not decode: %v\n%s", err, out.String())
	}
	if !rep.Empty || len(rep.Commits) != 0 {
		t.Fatalf("report = %+v, want empty", rep)
	}
}

func TestRunWhatsChangedRequiresPaths(t *testing.T) {
	repo := whatsChangedCLIRepo(t)
	var out, errb bytes.Buffer
	code := runWhatsChanged(&out, &errb, []string{"--dir", repo, "--since", "HEAD"})
	if code != 2 {
		t.Fatalf("exit = %d, want usage 2; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "--paths is required") {
		t.Fatalf("stderr = %q", errb.String())
	}
}

func whatsChangedCLIRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	whatsChangedCLIGit(t, repo, "init", "-q", "-b", "main")
	whatsChangedCLIGit(t, repo, "config", "core.autocrlf", "false")
	whatsChangedCLIGit(t, repo, "config", "user.name", "test")
	whatsChangedCLIGit(t, repo, "config", "user.email", "test@example.com")
	whatsChangedCLIWrite(t, filepath.Join(repo, "cmd", "fak", "main.go"), "package main\nfunc main() {}\n")
	whatsChangedCLIWrite(t, filepath.Join(repo, "docs", "readme.md"), "base\n")
	whatsChangedCLIGit(t, repo, "add", ".")
	whatsChangedCLIGit(t, repo, "commit", "-m", "base", "-q")
	return repo
}

func whatsChangedCLIGit(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func whatsChangedCLIGitOut(t *testing.T, cwd string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git %v in %s: %v", args, cwd, err)
	}
	return strings.TrimSpace(string(out))
}

func whatsChangedCLIWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
