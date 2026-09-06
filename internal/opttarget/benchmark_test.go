package opttarget

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

type benchMeasurer struct{}

func (benchMeasurer) Baseline(OptTarget) (float64, string, error) {
	return 0.5, "main", nil
}

func (benchMeasurer) Measure(_ OptTarget, _ rsiloop.Candidate) (rsiloop.Measurement, error) {
	return rsiloop.Measurement{Metric: 0.85, SuiteGreen: true, TruthClean: true}, nil
}

// BenchmarkResolve measures resolving an OptTarget through the closed registry,
// verifying target sites and instantiating the measurer seam.
func BenchmarkResolve(b *testing.B) {
	tgt := CacheSizeTarget([]int{4, 6, 8, 12, 16})
	repoRoot := "/fak/repo"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := Resolve(tgt, repoRoot)
		if err != nil || m == nil {
			b.Fatalf("resolve failed: %v", err)
		}
	}
}

// BenchmarkKnownMeasurers measures enumerating and sorting registered measurer keys.
func BenchmarkKnownMeasurers(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km := KnownMeasurers()
		if len(km) == 0 {
			b.Fatal("empty known measurers")
		}
	}
}

// BenchmarkValidate measures schema and constraint validation for a declared OptTarget.
func BenchmarkValidate(b *testing.B) {
	tgt := CacheSizeTarget([]int{4, 6, 8, 12, 16})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := tgt.Validate(); err != nil {
			b.Fatalf("validate failed: %v", err)
		}
	}
}

// BenchmarkCandidates measures lowering an OptTarget's bounded candidate grammar
// into rsiloop Candidate structs.
func BenchmarkCandidates(b *testing.B) {
	tgt := CacheSizeTarget([]int{2, 4, 6, 8, 12, 16, 24, 32})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cs := tgt.candidates()
		if len(cs) != 8 {
			b.Fatalf("candidates len = %d, want 8", len(cs))
		}
	}
}

// BenchmarkCompile measures lowering a declared OptTarget into an executable rsiloop.Harness.
func BenchmarkCompile(b *testing.B) {
	tgt := CacheSizeTarget([]int{4, 6, 8, 12, 16})
	m := benchMeasurer{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h, err := Compile(tgt, m)
		if err != nil || h.MetricName == "" {
			b.Fatalf("compile failed: %v", err)
		}
	}
}

// BenchmarkSavingsVsBudget measures evaluating optimization objectives across multiple
// candidate budgets, computing hit-rate scores, savings proxies, and knee detection.
func BenchmarkSavingsVsBudget(b *testing.B) {
	trace := loopTrace(30, 15) // 450 accesses
	budgets := []int{5, 10, 15, 20, 25, 30, 40, 50}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		curve, err := SavingsVsBudget(trace, budgets)
		if err != nil || curve.Knee == 0 {
			b.Fatalf("SavingsVsBudget failed: %v", err)
		}
	}
}

// BenchmarkSavingsVsBudget_Scaling measures optimization score evaluation across
// scaling trace lengths.
func BenchmarkSavingsVsBudget_Scaling(b *testing.B) {
	for _, n := range []int{100, 500, 2000} {
		b.Run(fmt.Sprintf("trace_len_%d", n), func(b *testing.B) {
			trace := loopTrace(20, n/20)
			budgets := []int{5, 10, 15, 20, 25, 30}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				curve, err := SavingsVsBudget(trace, budgets)
				if err != nil || len(curve.Points) != len(budgets) {
					b.Fatalf("unexpected curve: %v", err)
				}
			}
		})
	}
}

// BenchmarkKneeBudget measures calculating the diminishing-returns knee over
// candidate curve points.
func BenchmarkKneeBudget(b *testing.B) {
	points := make([]SavingsPoint, 20)
	for i := range points {
		budget := (i + 1) * 5
		savings := 100.0 * (1.0 - 1.0/float64(i+1))
		points[i] = SavingsPoint{
			Budget:  budget,
			HitRate: savings / 100.0,
			Savings: savings,
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := kneeBudget(points)
		if k <= 0 {
			b.Fatalf("invalid knee: %d", k)
		}
	}
}

// BenchmarkLRUHits measures the deterministic LRU cache hit evaluation on access traces.
func BenchmarkLRUHits(b *testing.B) {
	trace := loopTrace(30, 20) // 600 accesses
	budget := 30
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hits := lruHits(trace, budget)
		if hits <= 0 {
			b.Fatalf("unexpected hits: %d", hits)
		}
	}
}

// BenchmarkParseAnnotation measures parsing source code annotation directives into OptTargets.
func BenchmarkParseAnnotation(b *testing.B) {
	tail := "metric=lru_hit_rate dir=higher sweep=4,8,16,32 measurer=worktree-int name=cache-opt baseline=main"
	constName := "DefaultCacheSize"
	relPath := "internal/rsiloop/tunable.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t, err := parseAnnotation(tail, constName, relPath)
		if err != nil || t.Name == "" {
			b.Fatalf("parseAnnotation failed: %v", err)
		}
	}
}

// BenchmarkCheckRatchet measures validating coverage keep-bits against required target names.
func BenchmarkCheckRatchet(b *testing.B) {
	disc := []OptTarget{
		{Name: "cache-size"},
		{Name: "worker-pool"},
		{Name: "batch-window"},
		{Name: "queue-depth"},
		{Name: "timeout-ms"},
	}
	required := []string{"cache-size", "worker-pool", "batch-window"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Check(disc, required); err != nil {
			b.Fatalf("check failed: %v", err)
		}
	}
}

// BenchmarkMarshalInventory measures serializing discovered target inventory to JSON.
func BenchmarkMarshalInventory(b *testing.B) {
	targets := []OptTarget{
		CacheSizeTarget([]int{4, 8, 16}),
		{
			Name:        "worker-pool",
			Metric:      "throughput_rps",
			Direction:   HigherBetter,
			BaselineRef: "main",
			Site:        Site{Path: "internal/pool/pool.go", Const: "DefaultWorkers"},
			Grammar:     Grammar{Kind: GrammarIntSweep, Ints: []int{2, 4, 8, 16}},
			Measurer:    "worktree-int",
			Guards:      Guards{ChangedPaths: []string{"internal/pool/pool.go"}},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := MarshalInventory(targets)
		if err != nil || len(data) == 0 {
			b.Fatalf("marshal failed: %v", err)
		}
	}
}

// BenchmarkDiscoverDir measures directory walking, source parsing, and target discovery.
func BenchmarkDiscoverDir(b *testing.B) {
	dir := filepath.Join("testdata", "discover")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		targets, err := DiscoverDir(dir)
		if err != nil || len(targets) != 2 {
			b.Fatalf("discover failed: %v (targets: %d)", err, len(targets))
		}
	}
}
