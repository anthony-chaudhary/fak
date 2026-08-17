package fleetmetrics

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestMeasureCommitThroughputUsesAdjacentTenMinuteWindows(t *testing.T) {
	repo := initThroughputRepo(t)
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	commitAt(t, repo, now.Add(-19*time.Minute), "previous")
	commitAt(t, repo, now.Add(-10*time.Minute), "boundary-current")
	commitAt(t, repo, now.Add(-2*time.Minute), "current")

	got := MeasureCommitThroughput(repo, now)
	if !got.Measured || got.Error != "" {
		t.Fatalf("measurement = %+v", got)
	}
	if got.Current != 2 || got.Previous != 1 {
		t.Fatalf("current=%d previous=%d, want 2 and 1", got.Current, got.Previous)
	}
	if !got.LatestCommitAt.Equal(now.Add(-2 * time.Minute)) {
		t.Fatalf("latest=%v", got.LatestCommitAt)
	}
}

func TestCommitThroughputHealthClassifiesRecoveryUrgency(t *testing.T) {
	tests := []struct {
		name    string
		metric  CommitThroughput
		active  int
		state   string
		healthy bool
	}{
		{name: "idle is neutral", metric: CommitThroughput{Measured: true}, state: "idle", healthy: true},
		{name: "unreadable", metric: CommitThroughput{}, active: 2, state: "unknown"},
		{name: "positive current", metric: CommitThroughput{Measured: true, Current: 1}, active: 2, state: "healthy", healthy: true},
		{name: "first zero window", metric: CommitThroughput{Measured: true, Previous: 2}, active: 2, state: "stalled"},
		{name: "two zero windows", metric: CommitThroughput{Measured: true}, active: 2, state: "blocked"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.metric.Health(tt.active)
			if got.State != tt.state || got.Healthy != tt.healthy {
				t.Fatalf("health=%+v", got)
			}
			if tt.active > 0 && !got.Healthy && got.NextAction == "" {
				t.Fatalf("unhealthy verdict has no recovery action: %+v", got)
			}
		})
	}
}

func TestMeasureCommitThroughputIgnoresEmptyCommit(t *testing.T) {
	repo := initThroughputRepo(t)
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	commitAt(t, repo, now.Add(-2*time.Minute), "real")
	env := []string{"GIT_AUTHOR_DATE=" + now.Add(-time.Minute).Format(time.RFC3339), "GIT_COMMITTER_DATE=" + now.Add(-time.Minute).Format(time.RFC3339)}
	runGit(t, repo, env, "commit", "--allow-empty", "-q", "-m", "empty")

	got := MeasureCommitThroughput(repo, now)
	if got.Current != 1 {
		t.Fatalf("current=%d, want only the file-changing commit", got.Current)
	}
}

func TestMeasureCommitThroughputFailsClosedOutsideRepository(t *testing.T) {
	dir, err := os.MkdirTemp("", "fleetmetrics-nonrepo-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	got := MeasureCommitThroughput(dir, time.Now())
	if got.Measured || got.Error == "" || got.Healthy(1) {
		t.Fatalf("measurement = %+v", got)
	}
}

func initThroughputRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	runGit(t, dir, nil, "init", "-q")
	runGit(t, dir, nil, "config", "user.email", "fleet@example.test")
	runGit(t, dir, nil, "config", "user.name", "Fleet Test")
	return dir
}

func commitAt(t *testing.T, repo string, at time.Time, message string) {
	t.Helper()
	path := filepath.Join(repo, "progress.txt")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(message + "\n"); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, nil, "add", "progress.txt")
	env := []string{"GIT_AUTHOR_DATE=" + at.Format(time.RFC3339), "GIT_COMMITTER_DATE=" + at.Format(time.RFC3339)}
	runGit(t, repo, env, "commit", "-q", "-m", message)
}

func runGit(t *testing.T, dir string, extraEnv []string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), extraEnv...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
}
