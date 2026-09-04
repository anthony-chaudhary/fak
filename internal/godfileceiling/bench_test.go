package godfileceiling

import (
	"fmt"
	"strings"
	"testing"
)

// Invariant: benchmarks must measure realistic god-file workloads across small, ceiling-boundary, and large source payloads.
// Contract: benchmark setups must allocate test fixtures outside b.ResetTimer to ensure pure evaluation metrics.

func BenchmarkLineCount(b *testing.B) {
	small := []byte(strings.Repeat("package foo\nfunc Bar() {}\n", 50)) // 100 lines
	medium := []byte(strings.Repeat("// line of code\n", 1500))         // 1500 lines (HardCeiling)
	large := []byte(strings.Repeat("const x = 1\n", 10000))             // 10000 lines (extreme godfile)
	noNewline := []byte("func NoNewline() {}")

	b.Run("small_100_lines", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(small)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = LineCount(small)
		}
	})

	b.Run("medium_1500_ceiling", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(medium)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = LineCount(medium)
		}
	})

	b.Run("large_10000_lines", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(large)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = LineCount(large)
		}
	})

	b.Run("no_trailing_newline", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(noNewline)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = LineCount(noNewline)
		}
	})
}

func BenchmarkExcluded(b *testing.B) {
	firstParty := "internal/godfileceiling/godfileceiling.go"
	excludedRoot := "vendor/github.com/foo/bar/baz.go"
	nestedWorktree := ".claude/worktrees/agent-1/internal/foo.go"
	deepPath := "cmd/fak/sub/deeply/nested/component/feature.go"

	b.Run("first_party", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Excluded(firstParty)
		}
	})

	b.Run("excluded_vendor", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Excluded(excludedRoot)
		}
	})

	b.Run("excluded_nested_worktree", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Excluded(nestedWorktree)
		}
	})

	b.Run("deep_first_party", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Excluded(deepPath)
		}
	})
}

func BenchmarkEvaluate(b *testing.B) {
	cleanMeasured := make(map[string]int, 500)
	caps := make(map[string]int, 15)

	for i := 0; i < 500; i++ {
		path := fmt.Sprintf("internal/pkg%d/file%d.go", i/10, i)
		lines := 200 + (i % 800)
		cleanMeasured[path] = lines
	}

	for i := 0; i < 15; i++ {
		path := fmt.Sprintf("internal/godfiles/pinned%d.go", i)
		capVal := 1600 + i*100
		caps[path] = capVal
		cleanMeasured[path] = capVal
	}

	b.Run("clean_tree_500_files", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := Evaluate(cleanMeasured, caps)
			if !v.OK {
				b.Fatalf("expected clean tree to pass")
			}
		}
	})

	dirtyMeasured := make(map[string]int, len(cleanMeasured)+5)
	for k, v := range cleanMeasured {
		dirtyMeasured[k] = v
	}
	dirtyMeasured["internal/bad/new_monolith.go"] = 2500
	dirtyMeasured["internal/godfiles/pinned0.go"] = caps["internal/godfiles/pinned0.go"] + 100
	dirtyMeasured["internal/godfiles/pinned1.go"] = caps["internal/godfiles/pinned1.go"] - 200

	b.Run("violations_tree_500_files", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := Evaluate(dirtyMeasured, caps)
			if v.OK {
				b.Fatalf("expected dirty tree to fail")
			}
		}
	})

	b.Run("baseline_evaluation", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Evaluate(cleanMeasured, Baseline)
		}
	})
}

func BenchmarkProposeBaseline(b *testing.B) {
	measured := make(map[string]int, 500)
	for i := 0; i < 500; i++ {
		path := fmt.Sprintf("internal/pkg%d/file%d.go", i/10, i)
		lines := 100 + (i * 5)
		measured[path] = lines
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ProposeBaseline(measured)
	}
}

func BenchmarkRepin(b *testing.B) {
	oldCaps := map[string]int{
		"cmd/fak/large1.go": 2000,
		"cmd/fak/large2.go": 1800,
		"internal/gate.go":  2200,
	}

	validShrunk := map[string]int{
		"cmd/fak/large1.go": 1900,
		"cmd/fak/large2.go": 1700,
		"internal/gate.go":  2100,
		"internal/small.go": 500,
	}

	refusedGrowth := map[string]int{
		"cmd/fak/large1.go": 2100,
		"cmd/fak/large2.go": 1800,
		"internal/gate.go":  2200,
	}

	refusedNew := map[string]int{
		"cmd/fak/large1.go":     2000,
		"cmd/fak/large2.go":     1800,
		"internal/gate.go":      2200,
		"internal/brand_new.go": 1600,
	}

	b.Run("valid_monotonic_shrink", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			accepted, refusals := Repin(validShrunk, oldCaps)
			if accepted == nil || len(refusals) != 0 {
				b.Fatalf("repin rejected valid shrink")
			}
		}
	})

	b.Run("refused_growth", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			accepted, refusals := Repin(refusedGrowth, oldCaps)
			if accepted != nil || len(refusals) == 0 {
				b.Fatalf("repin should refuse growth")
			}
		}
	})

	b.Run("refused_new_offender", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			accepted, refusals := Repin(refusedNew, oldCaps)
			if accepted != nil || len(refusals) == 0 {
				b.Fatalf("repin should refuse new offender")
			}
		}
	})
}

func BenchmarkFormatBaseline(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FormatBaseline(Baseline)
	}
}
