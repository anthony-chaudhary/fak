package pagespublish

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAuditFreshnessUsesCommittedFileHistory(t *testing.T) {
	d := t.TempDir()
	runGit(t, d, "init")
	runGit(t, d, "config", "user.email", "test@example.com")
	runGit(t, d, "config", "user.name", "Test")
	if err := os.MkdirAll(filepath.Join(d, "docs", "marketing"), 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(d, "docs", "marketing", "campaign.md")
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitEnv(t, d, []string{"GIT_AUTHOR_DATE=2026-07-01T00:00:00Z", "GIT_COMMITTER_DATE=2026-07-01T00:00:00Z"}, "add", ".")
	runGitEnv(t, d, []string{"GIT_AUTHOR_DATE=2026-07-01T00:00:00Z", "GIT_COMMITTER_DATE=2026-07-01T00:00:00Z"}, "commit", "-m", "old")

	report, err := AuditFreshness(d, []string{"docs/marketing"}, 21*24*time.Hour, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if err == nil || !strings.Contains(err.Error(), "delete or substantively refresh") {
		t.Fatalf("error = %v", err)
	}
	if len(report.Stale) != 1 || report.Stale[0].Path != "docs/marketing/campaign.md" {
		t.Fatalf("report = %+v", report)
	}

	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	report, err = AuditFreshness(d, []string{"docs/marketing"}, 21*24*time.Hour, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if err != nil || report.Checked != 0 {
		t.Fatalf("deleted report = %+v, error = %v", report, err)
	}

	if err := os.WriteFile(path, []byte("fresh"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, d, "add", ".")
	runGitEnv(t, d, []string{"GIT_AUTHOR_DATE=2026-08-10T00:00:00Z", "GIT_COMMITTER_DATE=2026-08-10T00:00:00Z"}, "commit", "-m", "fresh")
	report, err = AuditFreshness(d, []string{"docs/marketing"}, 21*24*time.Hour, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if err != nil || len(report.Stale) != 0 {
		t.Fatalf("report = %+v, error = %v", report, err)
	}
}

func runGit(t *testing.T, dir string, args ...string) { t.Helper(); runGitEnv(t, dir, nil, args...) }
func runGitEnv(t *testing.T, dir string, env []string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), env...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
}
