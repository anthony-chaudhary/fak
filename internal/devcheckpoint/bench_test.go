package devcheckpoint

import (
	"testing"
	"time"
)

func BenchmarkDevCheckpoint(b *testing.B) {
	in := Input{
		Actor:        "bench-worker",
		Scope:        "issue-bench",
		State:        StateProgress,
		StageCurrent: 1,
		StageTotal:   100,
		StageName:    "benchmarking",
		Summary:      "Measuring progress calculation throughput",
		Evidence:     []string{"benchmarks/run.log"},
		Next:         "Complete benchmark iteration",
	}
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		in.StageCurrent = (i % 100) + 1
		rec, err := New(in, now)
		if err != nil {
			b.Fatalf("New failed: %v", err)
		}
		if rec.Stage == nil || rec.Stage.Percent < 1 || rec.Stage.Percent > 100 {
			b.Fatalf("invalid stage calculation: %#v", rec.Stage)
		}
	}
}

func TestBenchmarkDevCheckpointSanity(t *testing.T) {
	in := Input{
		Actor:        "test-worker",
		Scope:        "issue-test",
		State:        StateProgress,
		StageCurrent: 50,
		StageTotal:   100,
		StageName:    "testing",
		Summary:      "Testing progress calculation sanity",
		Next:         "Next step",
	}
	rec, err := New(in, time.Now())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rec.Stage == nil || rec.Stage.Percent != 50 {
		t.Fatalf("expected 50%% progress, got %#v", rec.Stage)
	}
}
