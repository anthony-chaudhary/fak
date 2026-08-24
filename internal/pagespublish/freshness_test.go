package pagespublish

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAuditFreshnessReviewsVolatileAndPreservesDurable(t *testing.T) {
	d := initFreshnessRepo(t)
	writeCommit(t, d, "2026-07-01T00:00:00Z", map[string]string{
		"docs/marketing/hub.md":      "stable",
		"docs/marketing/campaign.md": "old",
	})
	targets := FreshnessTargets{Schema: FreshnessTargetsSchema, Roots: []string{"docs/marketing"}, Targets: []FreshnessTarget{
		{Path: "docs/marketing/hub.md", Class: "durable", Check: "Canonical hub remains useful."},
		{Path: "docs/marketing/campaign.md", Class: "review", ReviewAfterDays: 21, Check: "Campaign claims remain current."},
	}}
	report, err := AuditFreshness(d, targets, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if err == nil || !strings.Contains(err.Error(), "update, archive, or retain") {
		t.Fatalf("error = %v", err)
	}
	if report.Durable != 1 || len(report.Due) != 1 || report.Due[0].Path != "docs/marketing/campaign.md" {
		t.Fatalf("report = %+v", report)
	}
}

func TestAuditFreshnessAcceptsWitnessedReviewCommit(t *testing.T) {
	d := initFreshnessRepo(t)
	writeCommit(t, d, "2026-07-01T00:00:00Z", map[string]string{"docs/marketing/campaign.md": "old"})
	writeCommit(t, d, "2026-08-10T00:00:00Z", map[string]string{"docs/marketing/campaign.md": "reviewed and current"})
	targets := FreshnessTargets{Schema: FreshnessTargetsSchema, Roots: []string{"docs/marketing"}, Targets: []FreshnessTarget{{Path: "docs/marketing/campaign.md", Class: "review", ReviewAfterDays: 21, Check: "Claims remain current."}}}
	report, err := AuditFreshness(d, targets, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if err != nil || len(report.Due) != 0 {
		t.Fatalf("report = %+v, error = %v", report, err)
	}
}

func TestAuditFreshnessRequiresEveryRootAssetClassified(t *testing.T) {
	d := initFreshnessRepo(t)
	writeCommit(t, d, "2026-08-10T00:00:00Z", map[string]string{"docs/marketing/forgotten.md": "content"})
	targets := FreshnessTargets{Schema: FreshnessTargetsSchema, Roots: []string{"docs/marketing"}, Targets: []FreshnessTarget{}}
	_, err := AuditFreshness(d, targets, time.Now())
	if err == nil || !strings.Contains(err.Error(), "unclassified") {
		t.Fatalf("error = %v", err)
	}
}

func TestAuditFreshnessRejectsShallowHistory(t *testing.T) {
	d := initFreshnessRepo(t)
	writeCommit(t, d, "2026-07-01T00:00:00Z", map[string]string{"docs/marketing/campaign.md": "old"})
	writeCommit(t, d, "2026-08-10T00:00:00Z", map[string]string{"README.md": "unrelated"})
	head, err := gitOutput(d, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(d, ".git", "shallow"), []byte(strings.TrimSpace(head)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	targets := FreshnessTargets{Schema: FreshnessTargetsSchema, Roots: []string{"docs/marketing"}, Targets: []FreshnessTarget{{Path: "docs/marketing/campaign.md", Class: "review", ReviewAfterDays: 21, Check: "Claims remain current."}}}
	_, err = AuditFreshness(d, targets, time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC))
	if err == nil || !strings.Contains(err.Error(), "full git history") {
		t.Fatalf("error = %v, want full-history refusal", err)
	}
}

func initFreshnessRepo(t *testing.T) string {
	t.Helper()
	d := t.TempDir()
	runGit(t, d, "init")
	runGit(t, d, "config", "user.email", "test@example.com")
	runGit(t, d, "config", "user.name", "Test")
	return d
}

func writeCommit(t *testing.T, dir, date string, files map[string]string) {
	t.Helper()
	for name, content := range files {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGit(t, dir, "add", ".")
	runGitEnv(t, dir, []string{"GIT_AUTHOR_DATE=" + date, "GIT_COMMITTER_DATE=" + date}, "commit", "-m", "update")
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
