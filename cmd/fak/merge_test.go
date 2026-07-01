package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mergepreview"
)

func TestRunMergeDryRunReportsEmptyNetDiff(t *testing.T) {
	repo := mergeCLISameTreeFixture(t)
	var out, errb bytes.Buffer
	code := runMerge(&out, &errb, []string{"--dry-run", "--dir", repo, "--target", "right"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "empty_net_diff") || !strings.Contains(out.String(), "cached diff will be empty") {
		t.Fatalf("human output did not report empty net diff:\n%s", out.String())
	}
	if status := mergeCLIStatus(t, repo); status != "" {
		t.Fatalf("dry-run left dirty status: %q", status)
	}
}

func TestRunMergeDryRunConflictJSON(t *testing.T) {
	repo := mergeCLIConflictFixture(t)
	var out, errb bytes.Buffer
	code := runMerge(&out, &errb, []string{"--dry-run", "--dir", repo, "--target", "right", "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want conflict refusal 3; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got mergepreview.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("json did not decode: %v\n%s", err, out.String())
	}
	if got.Outcome != mergepreview.OutcomeConflicts || strings.Join(got.Conflicts, ",") != "f.txt" {
		t.Fatalf("preview = %+v, want conflict on f.txt", got)
	}
}

func TestRunMergeRequiresDryRun(t *testing.T) {
	var out, errb bytes.Buffer
	code := runMerge(&out, &errb, []string{"--target", "origin/main"})
	if code != 2 {
		t.Fatalf("exit = %d, want usage 2; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "only --dry-run is supported") {
		t.Fatalf("stderr = %q", errb.String())
	}
}

func mergeCLISameTreeFixture(t *testing.T) string {
	repo := mergeCLIRepo(t)
	mergeCLIGit(t, repo, "checkout", "-q", "-b", "left")
	mergeCLIWrite(t, filepath.Join(repo, "f.txt"), "same\n")
	mergeCLIGit(t, repo, "commit", "-am", "same-left", "-q")
	mergeCLIGit(t, repo, "checkout", "-q", "main")
	mergeCLIGit(t, repo, "checkout", "-q", "-b", "right")
	mergeCLIWrite(t, filepath.Join(repo, "f.txt"), "same\n")
	mergeCLIGit(t, repo, "commit", "-am", "same-right", "-q")
	mergeCLIGit(t, repo, "checkout", "-q", "left")
	return repo
}

func mergeCLIConflictFixture(t *testing.T) string {
	repo := mergeCLIRepo(t)
	mergeCLIGit(t, repo, "checkout", "-q", "-b", "left")
	mergeCLIWrite(t, filepath.Join(repo, "f.txt"), "left\n")
	mergeCLIGit(t, repo, "commit", "-am", "left", "-q")
	mergeCLIGit(t, repo, "checkout", "-q", "main")
	mergeCLIGit(t, repo, "checkout", "-q", "-b", "right")
	mergeCLIWrite(t, filepath.Join(repo, "f.txt"), "right\n")
	mergeCLIGit(t, repo, "commit", "-am", "right", "-q")
	mergeCLIGit(t, repo, "checkout", "-q", "left")
	return repo
}

func mergeCLIRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	mergeCLIGit(t, repo, "init", "-q", "-b", "main")
	mergeCLIGit(t, repo, "config", "core.autocrlf", "false")
	mergeCLIGit(t, repo, "config", "user.name", "test")
	mergeCLIGit(t, repo, "config", "user.email", "test@example.com")
	mergeCLIWrite(t, filepath.Join(repo, "f.txt"), "base\n")
	mergeCLIGit(t, repo, "add", ".")
	mergeCLIGit(t, repo, "commit", "-m", "base", "-q")
	return repo
}

func mergeCLIGit(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func mergeCLIWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mergeCLIStatus(t *testing.T, repo string) string {
	t.Helper()
	cmd := exec.Command("git", "status", "--short")
	cmd.Dir = repo
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git status: %v", err)
	}
	return string(out)
}
