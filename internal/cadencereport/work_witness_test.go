package cadencereport

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestWorkFromGitCreditsOnlyPublishedWitnessedShips is #4224's authoritative
// current-state fixture: one remote+witnessed ship, one remote+unwitnessed ship,
// and one local-only ship-shaped commit. Exactly one receives delivery credit.
func TestWorkFromGitCreditsOnlyPublishedWitnessedShips(t *testing.T) {
	repo := t.TempDir()
	remote := filepath.Join(t.TempDir(), "origin.git")
	gitRun(t, "", "init", "--bare", remote)
	gitRun(t, "", "init", "-b", "main", repo)
	gitRun(t, repo, "config", "user.email", "cadence@example.test")
	gitRun(t, repo, "config", "user.name", "Cadence Test")
	gitRun(t, repo, "remote", "add", "origin", remote)

	witnessedSHA := fixtureCommit(t, repo, "published-witnessed.txt",
		"feat(gateway): add published witnessed fixture (fak gateway)")
	unwitnessedSHA := fixtureCommit(t, repo, "published-unwitnessed.txt",
		"feat(policy): add published unwitnessed fixture (fak policy)")
	gitRun(t, repo, "push", "-u", "origin", "main")
	localSHA := fixtureCommit(t, repo, "local-only.txt",
		"feat(model): add local-only fixture (fak model)")

	audit := func(_ string, _ string, commits []shipCommit) (map[string]shipAuditResult, string) {
		out := map[string]shipAuditResult{}
		for _, commit := range commits {
			switch commit.SHA {
			case witnessedSHA:
				out[commit.SHA] = shipAuditResult{Witnessed: true, Detail: "OK diff-witnessed"}
			case unwitnessedSHA:
				out[commit.SHA] = shipAuditResult{Detail: "CLAIM_UNWITNESSED subject-only"}
			}
		}
		return out, ""
	}
	work := workFromGit(repo, 7, audit)
	if work.Err != "" {
		t.Fatalf("workFromGit: %s", work.Err)
	}
	if work.Commits != 3 || work.Ships != 1 {
		t.Fatalf("commits/ships=%d/%d, want 3/1: %+v", work.Commits, work.Ships, work)
	}
	if len(work.ByLane) != 1 || work.ByLane["gateway"] != 1 {
		t.Fatalf("ByLane=%v, want gateway:1 only", work.ByLane)
	}
	if len(work.Held) != 2 {
		t.Fatalf("Held=%+v, want two typed holds", work.Held)
	}
	holds := map[string]ShipHold{}
	for _, hold := range work.Held {
		holds[hold.Reason] = hold
	}
	if hold := holds[shipHoldLocalOnly]; hold.SHA != shortCommit(localSHA) || hold.Leaf != "model" {
		t.Fatalf("local hold=%+v, want %s/model", hold, shortCommit(localSHA))
	}
	if hold := holds[shipHoldPublishedUnwitnessed]; hold.SHA != shortCommit(unwitnessedSHA) || hold.Leaf != "policy" || !strings.Contains(hold.Detail, "CLAIM_UNWITNESSED") {
		t.Fatalf("unwitnessed hold=%+v, want %s/policy with audit detail", hold, shortCommit(unwitnessedSHA))
	}

	// Human and JSON-facing Work carry the same authoritative hold reasons. Render
	// must not hide the distinction behind the aggregate one-ship count.
	report := Fold(okScores(), work, okReleases(), foldOpts())
	rendered := Render(report)
	for _, want := range []string{"1 ship(s)", "held: local_only", "held: published_unwitnessed"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("Render missing %q:\n%s", want, rendered)
		}
	}
	payload, err := json.Marshal(work)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"ships":1`, `"reason":"local_only"`, `"reason":"published_unwitnessed"`} {
		if !strings.Contains(string(payload), want) {
			t.Fatalf("Work JSON missing %s: %s", want, payload)
		}
	}
}

func TestFoldDOSAuditRowsRequiresDiffWitnessedOK(t *testing.T) {
	commits := []shipCommit{
		{SHA: "aaaaaaaaaaaaaaaa", Leaf: "gateway"},
		{SHA: "bbbbbbbbbbbbbbbb", Leaf: "policy"},
		{SHA: "cccccccccccccccc", Leaf: "model"},
	}
	got := foldDOSAuditRows(commits, []dosAuditRow{
		{SHA: "aaaaaaaaa", Verdict: "OK", Witness: "diff-witnessed", Reason: "source touched"},
		{SHA: "bbbbbbbbb", Verdict: "CLAIM_UNWITNESSED", Witness: "subject-only", Reason: "docs-only"},
		{SHA: "ccccccccc", Verdict: "OK", Witness: "subject-only", Reason: "not independent"},
	})
	if !got[commits[0].SHA].Witnessed {
		t.Fatal("diff-witnessed OK did not qualify")
	}
	if got[commits[1].SHA].Witnessed || got[commits[2].SHA].Witnessed {
		t.Fatalf("subject-only audit qualified: %+v", got)
	}
}

func TestWorkFromGitAuditFailureNeverCreditsPublishedSubjects(t *testing.T) {
	repo := t.TempDir()
	remote := filepath.Join(t.TempDir(), "origin.git")
	gitRun(t, "", "init", "--bare", remote)
	gitRun(t, "", "init", "-b", "main", repo)
	gitRun(t, repo, "config", "user.email", "cadence@example.test")
	gitRun(t, repo, "config", "user.name", "Cadence Test")
	gitRun(t, repo, "remote", "add", "origin", remote)
	fixtureCommit(t, repo, "ship.txt", "feat(gateway): add audit-failure fixture (fak gateway)")
	gitRun(t, repo, "push", "-u", "origin", "main")

	work := workFromGit(repo, 7, func(string, string, []shipCommit) (map[string]shipAuditResult, string) {
		return nil, "commit audit unavailable"
	})
	if work.Ships != 0 || work.Err != "commit audit unavailable" || len(work.Held) != 1 || work.Held[0].Reason != shipHoldPublishedUnwitnessed {
		t.Fatalf("audit failure must fail closed, got %+v", work)
	}
}

// TestWorkFromGitRealDOSIntegration exercises the production referee when the DOS
// binary is installed. cadence.yml installs dos-kernel before this package test, so the
// scheduled consumer cannot silently drift from the fake-audit unit fixture. Ordinary
// minimal Go environments may skip; WorkFromGit itself still fails closed if DOS is
// absent at runtime.
func TestWorkFromGitRealDOSIntegration(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("dos commit-audit not installed")
	}
	repo := t.TempDir()
	remote := filepath.Join(t.TempDir(), "origin.git")
	gitRun(t, "", "init", "--bare", remote)
	gitRun(t, "", "init", "-b", "main", repo)
	gitRun(t, repo, "config", "user.email", "cadence@example.test")
	gitRun(t, repo, "config", "user.name", "Cadence Test")
	gitRun(t, repo, "remote", "add", "origin", remote)

	if err := os.MkdirAll(filepath.Join(repo, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	fixtureCommit(t, repo, filepath.Join("internal", "gateway", "witness.go"),
		"feat(gateway): add published witnessed fixture (fak gateway)")
	fixtureCommit(t, repo, "README.md",
		"feat(policy): add published unwitnessed fixture (fak policy)")
	gitRun(t, repo, "push", "-u", "origin", "main")
	fixtureCommit(t, repo, filepath.Join("internal", "gateway", "local.go"),
		"feat(gateway): add local-only fixture (fak gateway)")

	work := WorkFromGit(repo, 7)
	if work.Err != "" {
		t.Fatalf("real DOS WorkFromGit: %s", work.Err)
	}
	if work.Ships != 1 || len(work.Held) != 2 {
		t.Fatalf("real DOS commits/ships/held=%d/%d/%d, want 3/1/2: %+v", work.Commits, work.Ships, len(work.Held), work)
	}
	reasons := map[string]bool{}
	for _, hold := range work.Held {
		reasons[hold.Reason] = true
	}
	if !reasons[shipHoldLocalOnly] || !reasons[shipHoldPublishedUnwitnessed] {
		t.Fatalf("real DOS held reasons=%v, want local_only + published_unwitnessed", reasons)
	}
}

func fixtureCommit(t *testing.T, repo, name, subject string) string {
	t.Helper()
	path := filepath.Join(repo, name)
	if err := os.WriteFile(path, []byte(subject+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitRun(t, repo, "add", "--", name)
	gitRun(t, repo, "commit", "-m", subject)
	return strings.TrimSpace(gitRun(t, repo, "rev-parse", "HEAD"))
}

func gitRun(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}
