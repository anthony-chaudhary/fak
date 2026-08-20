package sessionjournal

import (
	"testing"
	"time"
)

func TestRunBenchmarkEquivalentPathsAndMetadata(t *testing.T) {
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	report, err := RunBenchmark([]int{10, 100}, 3, now)
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != BenchmarkSchema || len(report.Cases) != 2 || report.Repetitions != 3 {
		t.Fatalf("report=%+v", report)
	}
	for _, c := range report.Cases {
		if len(c.Paths) != 3 {
			t.Fatalf("case=%+v", c)
		}
		digest := c.Paths[0].ResultDigest
		for _, p := range c.Paths {
			if p.ResultDigest != digest || p.MedianNS < 0 || p.MedianAllocatedB == 0 {
				t.Fatalf("path=%+v", p)
			}
		}
	}
	if report.Decision.ReopenAtEvents != 200_000 || report.Decision.ReopenAtMedianMS != 1000 {
		t.Fatalf("decision=%+v", report.Decision)
	}
}

func TestRunBenchmarkRejectsUnsafeBounds(t *testing.T) {
	for _, tc := range []struct {
		counts []int
		reps   int
	}{{[]int{1}, 1}, {[]int{3}, 1}, {[]int{1_000_002}, 1}, {[]int{10}, 0}, {[]int{10}, 21}} {
		if _, err := RunBenchmark(tc.counts, tc.reps, time.Now()); err == nil {
			t.Fatalf("accepted %+v", tc)
		}
	}
}
