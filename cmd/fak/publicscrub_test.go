package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublicScrubAuditRangeAndTree(t *testing.T) {
	repo := t.TempDir()
	publicScrubGitRun(t, repo, "init", "-q", "-b", "main")
	publicScrubGitRun(t, repo, "config", "user.email", "t@t")
	publicScrubGitRun(t, repo, "config", "user.name", "t")
	publicScrubWriteFile(t, repo, ".gitignore", "tools/_registry/\n")
	publicScrubWriteFile(t, repo, "docs/notes.md", "hello\n")
	publicScrubGitRun(t, repo, "add", ".gitignore", "docs/notes.md")
	publicScrubGitRun(t, repo, "commit", "-qm", "seed")
	base := strings.TrimSpace(publicScrubGitRunOutput(t, repo, "rev-parse", "HEAD"))

	needle := "10.11.12.13"
	publicScrubWriteFile(t, repo, "tools/_registry/scrub_needles.private.json", `{"export_audit_needles":["`+needle+`"]}`+"\n")
	publicScrubWriteFile(t, repo, "node.json", `{"ip":"`+needle+`"}`+"\n")
	publicScrubGitRun(t, repo, "add", "node.json")
	publicScrubGitRun(t, repo, "commit", "-qm", "leak")

	var stdout, stderr bytes.Buffer
	rc := runPublicScrub(&stdout, &stderr, []string{"audit-range", "--root", repo, base + "..HEAD"})
	if rc != 1 {
		t.Fatalf("audit-range rc=%d stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "node.json") || !strings.Contains(stdout.String(), needle) {
		t.Fatalf("audit-range did not name leak: %s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	rc = runPublicScrub(&stdout, &stderr, []string{"audit-tree", "--root", repo, "--json"})
	if rc != 1 {
		t.Fatalf("audit-tree rc=%d stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	var report struct {
		OK     bool                    `json:"ok"`
		Mode   string                  `json:"mode"`
		Misses []struct{ File string } `json:"misses"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, stdout.String())
	}
	if report.OK || report.Mode != "full" || len(report.Misses) == 0 || report.Misses[0].File != "node.json" {
		t.Fatalf("unexpected report: %+v", report)
	}
}

func TestPublicScrubAuditMessage(t *testing.T) {
	repo := t.TempDir()
	publicScrubGitRun(t, repo, "init", "-q", "-b", "main")
	needle := "100" + ".64.0.10"
	msg := filepath.Join(repo, "MSG")
	if err := os.WriteFile(msg, []byte("fix: ok\n\nbody "+needle+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	rc := runPublicScrub(&stdout, &stderr, []string{"audit-message", "--root", repo, msg})
	if rc != 1 {
		t.Fatalf("audit-message rc=%d stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "message:3") || !strings.Contains(stdout.String(), needle) {
		t.Fatalf("audit-message did not name leak: %s", stdout.String())
	}
}

func publicScrubWriteFile(t *testing.T, repo, rel, content string) {
	t.Helper()
	full := filepath.Join(repo, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func publicScrubGitRun(t *testing.T, repo string, args ...string) {
	t.Helper()
	_ = publicScrubGitRunOutput(t, repo, args...)
}

func publicScrubGitRunOutput(t *testing.T, repo string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git %v failed: %v", args, err)
	}
	return string(out)
}
