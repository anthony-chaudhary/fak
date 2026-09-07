package orgdebt

import (
	"testing"
)

// BenchmarkEvaluate measures the throughput of Evaluate over a clean shift-left input.
func BenchmarkEvaluate(b *testing.B) {
	in := Input{
		Issues: []Issue{
			{
				Number: 101,
				Title:  "feat(cache): add bounded kv cache lease #101",
				Body: `## Current state
The cache is unbounded.

## Scope
Bound KV cache storage to 500MB.

## Done condition
Cache evicts LRU items above 500MB.

## Witness
go test ./internal/cache -run TestEviction

## Likely files
internal/cache/cache.go`,
				Labels: []string{"class:dev", "priority/P1"},
			},
		},
		Commits: []Commit{
			{
				SHA:          "abcdef123456",
				Subject:      "feat(cache): add bounded kv cache lease #101 (fak cache)",
				FilesTouched: []string{"internal/cache/cache.go", "internal/cache/cache_test.go"},
				LinesAdded:   120,
			},
		},
		InternalPackages: []string{"cache"},
		DeclaredLanes:    []string{"cache"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Evaluate(in)
	}
}
