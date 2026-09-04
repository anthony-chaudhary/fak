package mergepreview

import (
	"context"
	"strings"
	"testing"
)

// BenchmarkMergePreview measures three-way merge preview evaluation throughput
// for clean merges that introduce tree changes.
func BenchmarkMergePreview(b *testing.B) {
	ctx := context.Background()
	const (
		headSHA   = "1111111111111111111111111111111111111111"
		targetSHA = "2222222222222222222222222222222222222222"
		mergeTree = "3333333333333333333333333333333333333333"
	)

	mergeOut := []byte(mergeTree + "\x00internal/mergepreview/a.go\x00internal/mergepreview/b.go\x00")
	diffOut := []byte("internal/mergepreview/a.go\x00internal/mergepreview/b.go\x00")

	runner := func(ctx context.Context, dir string, args ...string) (RunResult, error) {
		if len(args) == 0 {
			return RunResult{}, nil
		}
		switch args[0] {
		case "rev-parse":
			ref := args[len(args)-1]
			if strings.HasPrefix(ref, "HEAD") {
				return RunResult{Stdout: []byte(headSHA + "\n"), Code: 0}, nil
			}
			return RunResult{Stdout: []byte(targetSHA + "\n"), Code: 0}, nil
		case "merge-tree":
			return RunResult{Stdout: mergeOut, Code: 0}, nil
		case "diff":
			return RunResult{Stdout: diffOut, Code: 0}, nil
		default:
			return RunResult{Code: 0}, nil
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Preview(ctx, "/virtual/repo", "origin/main", runner)
		if err != nil {
			b.Fatal(err)
		}
		if res.Outcome != OutcomeCleanMerge || len(res.ChangedFiles) != 2 {
			b.Fatalf("unexpected preview result: %+v", res)
		}
	}
}

// BenchmarkPreviewEmptyNetDiff measures preview throughput when the merge tree matches HEAD.
func BenchmarkPreviewEmptyNetDiff(b *testing.B) {
	ctx := context.Background()
	const (
		headSHA   = "1111111111111111111111111111111111111111"
		targetSHA = "2222222222222222222222222222222222222222"
		mergeTree = "3333333333333333333333333333333333333333"
	)

	mergeOut := []byte(mergeTree + "\x00")

	runner := func(ctx context.Context, dir string, args ...string) (RunResult, error) {
		if len(args) == 0 {
			return RunResult{}, nil
		}
		switch args[0] {
		case "rev-parse":
			ref := args[len(args)-1]
			if strings.HasPrefix(ref, "HEAD") {
				return RunResult{Stdout: []byte(headSHA + "\n"), Code: 0}, nil
			}
			return RunResult{Stdout: []byte(targetSHA + "\n"), Code: 0}, nil
		case "merge-tree":
			return RunResult{Stdout: mergeOut, Code: 0}, nil
		case "diff":
			return RunResult{Stdout: nil, Code: 0}, nil
		default:
			return RunResult{Code: 0}, nil
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Preview(ctx, "/virtual/repo", "origin/main", runner)
		if err != nil {
			b.Fatal(err)
		}
		if res.Outcome != OutcomeEmptyNetDiff || !res.CachedDiffEmpty {
			b.Fatalf("unexpected preview result: %+v", res)
		}
	}
}

// BenchmarkPreviewConflicts measures preview throughput when merge conflicts are detected.
func BenchmarkPreviewConflicts(b *testing.B) {
	ctx := context.Background()
	const (
		headSHA   = "1111111111111111111111111111111111111111"
		targetSHA = "2222222222222222222222222222222222222222"
		mergeTree = "3333333333333333333333333333333333333333"
	)

	conflictOut := []byte(mergeTree + "\x00conflict_file_a.go\x00conflict_file_b.go\x00")

	runner := func(ctx context.Context, dir string, args ...string) (RunResult, error) {
		if len(args) == 0 {
			return RunResult{}, nil
		}
		switch args[0] {
		case "rev-parse":
			ref := args[len(args)-1]
			if strings.HasPrefix(ref, "HEAD") {
				return RunResult{Stdout: []byte(headSHA + "\n"), Code: 0}, nil
			}
			return RunResult{Stdout: []byte(targetSHA + "\n"), Code: 0}, nil
		case "merge-tree":
			return RunResult{Stdout: conflictOut, Code: 1}, nil
		default:
			return RunResult{Code: 0}, nil
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Preview(ctx, "/virtual/repo", "origin/main", runner)
		if err != nil {
			b.Fatal(err)
		}
		if res.Outcome != OutcomeConflicts || len(res.Conflicts) != 2 {
			b.Fatalf("unexpected preview result: %+v", res)
		}
	}
}

// BenchmarkApply measures textless superset merge resolution execution throughput.
func BenchmarkApply(b *testing.B) {
	ctx := context.Background()
	const (
		headSHA    = "1111111111111111111111111111111111111111"
		targetSHA  = "2222222222222222222222222222222222222222"
		mergeTree  = "3333333333333333333333333333333333333333"
		mergedHead = "4444444444444444444444444444444444444444"
	)

	mergeOut := []byte(mergeTree + "\x00")
	treeOID := []byte(mergeTree + "\n")
	headOID := []byte(headSHA + "\n")
	nextHeadOID := []byte(mergedHead + "\n")

	applied := false
	runner := func(ctx context.Context, dir string, args ...string) (RunResult, error) {
		if len(args) == 0 {
			return RunResult{}, nil
		}
		switch args[0] {
		case "rev-parse":
			ref := args[len(args)-1]
			if strings.Contains(ref, "^{tree}") {
				return RunResult{Stdout: treeOID, Code: 0}, nil
			}
			if strings.HasPrefix(ref, "HEAD") {
				if applied {
					return RunResult{Stdout: nextHeadOID, Code: 0}, nil
				}
				return RunResult{Stdout: headOID, Code: 0}, nil
			}
			return RunResult{Stdout: []byte(targetSHA + "\n"), Code: 0}, nil
		case "merge-tree":
			return RunResult{Stdout: mergeOut, Code: 0}, nil
		case "diff":
			return RunResult{Stdout: nil, Code: 0}, nil
		case "merge":
			applied = true
			return RunResult{Code: 0}, nil
		default:
			return RunResult{Code: 0}, nil
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applied = false
		res, err := Apply(ctx, "/virtual/repo", "origin/main", "merge commit", runner)
		if err != nil {
			b.Fatal(err)
		}
		if res.ApplyOutcome != ApplyResolvedSuperset || res.MergeCommit != mergedHead {
			b.Fatalf("unexpected apply result: %+v", res)
		}
	}
}

// BenchmarkSplitNUL measures NUL-delimited byte splitting throughput.
func BenchmarkSplitNUL(b *testing.B) {
	raw := []byte("cmd/fak/main.go\x00internal/mergepreview/preview.go\x00internal/safesync/sync.go\x00")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parts := splitNUL(raw)
		if len(parts) != 3 {
			b.Fatalf("expected 3 paths, got %d", len(parts))
		}
	}
}

// BenchmarkUniqueSorted measures deduplication and lexicographic sorting throughput.
func BenchmarkUniqueSorted(b *testing.B) {
	input := []string{"z.go", "a.go", "m.go", "a.go", "b.go", "m.go", "c.go"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		unique := uniqueSorted(input)
		if len(unique) != 5 {
			b.Fatalf("expected 5 unique paths, got %d", len(unique))
		}
	}
}

// TestBenchmarkMergePreviewSanity verifies the benchmark executes iterations cleanly.
func TestBenchmarkMergePreviewSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkMergePreview)
	if res.N <= 0 {
		t.Fatalf("expected positive benchmark iterations, got %d", res.N)
	}
}
