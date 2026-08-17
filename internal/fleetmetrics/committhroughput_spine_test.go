package fleetmetrics

import (
	"testing"
	"time"
)

// TestCommitThroughputSpine captures the operator-visible contract: active work
// with no real commits is red, and one real commit restores a positive rate.
func TestCommitThroughputSpine(t *testing.T) {
	repo := initThroughputRepo(t)
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)

	commitAt(t, repo, now.Add(-30*time.Minute), "baseline")
	before := MeasureCommitThroughput(repo, now).Health(4)
	if before.State != "blocked" || before.Healthy {
		t.Fatalf("before commit = %+v, want blocked", before)
	}

	commitAt(t, repo, now.Add(-time.Minute), "ship real work")
	metric := MeasureCommitThroughput(repo, now)
	after := metric.Health(4)
	if metric.Current != 1 || after.State != "healthy" || !after.Healthy {
		t.Fatalf("after commit metric=%+v health=%+v", metric, after)
	}
}
