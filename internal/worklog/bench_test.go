package worklog

import (
	"fmt"
	"testing"
)

// BenchmarkWorklog exercises log appending and cursor-drained reads in a loop.
func BenchmarkWorklog(b *testing.B) {
	f := NewFeed(1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		change := WorkChange{
			Kind:    KindCommit,
			SHA:     fmt.Sprintf("bench-sha-%08d", i),
			Claim:   "benchmark payload",
			Verdict: "OK",
			Witness: "diff-witnessed",
		}
		f.Append(change)
		if i%128 == 0 {
			_, _ = f.Drain("", 0)
		}
	}
}
