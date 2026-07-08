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

func TestRunMergeRequiresDryRunOrApply(t *testing.T) {
	var out, errb bytes.Buffer
	code := runMerge(&out, &errb, []string{"--target", "origin/main"})
	if code != 2 {
		t.Fatalf("exit = %d, want usage 2; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "refusing a bare mutating merge") {
		t.Fatalf("stderr = %q", errb.String())
	}
}

func TestRunMergeApplyResolvesSupersetTextlessly(t *testing.T) {
	repo := mergeCLISameTreeFixture(t)
	var out, errb bytes.Buffer
	code := runMerge(&out, &errb, []string{"--apply", "--dir", repo, "--target", "right"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "resolved_superset") || !strings.Contains(out.String(), "git diff --cached empty") {
		t.Fatalf("human output did not report a textless superset resolve:\n%s", out.String())
	}
	// The merge must have committed with `right` as a second parent while keeping HEAD's tree.
	if parents := mergeCLIRevParse(t, repo, "HEAD^2"); parents == "" {
		t.Fatalf("HEAD has no second parent — no merge commit was made")
	}
	if got, want := mergeCLIRevParse(t, repo, "HEAD^{tree}"), mergeCLIRevParse(t, repo, "HEAD^1^{tree}"); got != want {
		t.Fatalf("merged tree %s != prior HEAD tree %s — resolve was not textless", got, want)
	}
	if status := mergeCLIStatus(t, repo); status != "" {
		t.Fatalf("apply left dirty status: %q", status)
	}
}

func TestRunMergeApplyDefersGenuineConflict(t *testing.T) {
	repo := mergeCLIConflictFixture(t)
	headBefore := mergeCLIRevParse(t, repo, "HEAD")
	var out, errb bytes.Buffer
	code := runMerge(&out, &errb, []string{"--apply", "--dir", repo, "--target", "right"})
	if code != 3 {
		t.Fatalf("exit = %d, want conflict deferral 3; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "deferred_conflict") || !strings.Contains(out.String(), "hand-merge") {
		t.Fatalf("human output did not defer the conflict:\n%s", out.String())
	}
	if headAfter := mergeCLIRevParse(t, repo, "HEAD"); headAfter != headBefore {
		t.Fatalf("apply mutated HEAD on a genuine conflict: before %s after %s", headBefore, headAfter)
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

func mergeCLIRevParse(t *testing.T, repo, ref string) string {
	t.Helper()
	cmd := exec.Command("git", "rev-parse", "--verify", "--quiet", ref)
	cmd.Dir = repo
	out, _ := cmd.Output() // empty output (and non-zero exit) means the ref does not resolve
	return strings.TrimSpace(string(out))
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
