package fleettrend

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

func makeBenchmarkRows(count int) []map[string]any {
	rows := make([]map[string]any, count)
	base := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < count; i++ {
		t := base.Add(time.Duration(i) * time.Hour).Format(time.RFC3339)
		rows[i] = map[string]any{
			"ts":              t,
			"usable":          float64(10 + (i % 5)),
			"live":            float64(4 + (i % 3)),
			"sessions":        float64(20 + i*2),
			"escalate":        float64(i % 2),
			"lands":           float64(5 + i*3),
			"resumes":         float64(20 + i*4),
			"deaths":          float64(1 + i),
			"lands_witnessed": 1,
		}
	}
	return rows
}

func makeBenchmarkSnapshot(i int) map[string]any {
	return map[string]any{
		"sessions": map[string]any{
			"total":       25 + i,
			"by_category": map[string]any{"LIVE": 6, "AGENT": 19},
		},
		"accounts": map[string]any{
			"usable": 8,
			"total":  12,
		},
		"system": map[string]any{
			"verdict":      "OPERATIONAL",
			"escalate":     0,
			"self_healing": 1,
		},
		"throughput": map[string]any{
			"lands":         42 + i*2,
			"resumes":       120 + i*3,
			"deaths":        8 + i,
			"lands_witness": "git",
		},
	}
}

// BenchmarkFleetTrend measures the core fleet-trend evaluation pipeline across a 24-tick observation window.
func BenchmarkFleetTrend(b *testing.B) {
	snap := makeBenchmarkSnapshot(0)
	rows := makeBenchmarkRows(24)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metrics := MetricsOf(snap)
		_ = metrics

		trendLine := RenderLine(rows)
		if trendLine == "" {
			b.Fatal("unexpected empty trend line")
		}

		rates, ok := WindowRates(rows)
		if !ok || !rates.GoodputPresent {
			b.Fatal("expected valid window rates")
		}

		stalls := HeadStallTicks(rows)
		_ = stalls

		throughputLine := RenderThroughput(rows)
		if throughputLine == "" {
			b.Fatal("unexpected empty throughput line")
		}
	}
}

// BenchmarkRenderLine measures formatting of gauge metric trends and sparklines across window sizes.
func BenchmarkRenderLine(b *testing.B) {
	windows := []int{12, 24, 48}
	for _, size := range windows {
		b.Run(fmt.Sprintf("Ticks%d", size), func(b *testing.B) {
			rows := makeBenchmarkRows(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				line := RenderLine(rows)
				if line == "" {
					b.Fatal("unexpected empty render")
				}
			}
		})
	}
}

// BenchmarkWindowRates measures rate derivation, window duration arithmetic, and goodput calculation.
func BenchmarkWindowRates(b *testing.B) {
	windows := []int{12, 24, 48}
	for _, size := range windows {
		b.Run(fmt.Sprintf("Ticks%d", size), func(b *testing.B) {
			rows := makeBenchmarkRows(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rates, ok := WindowRates(rows)
				if !ok {
					b.Fatal("expected rates present")
				}
				_ = rates.Goodput
			}
		})
	}
}

// BenchmarkRenderThroughput measures formatting of throughput SLO rates and provenance annotations.
func BenchmarkRenderThroughput(b *testing.B) {
	rows := makeBenchmarkRows(24)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tp := RenderThroughput(rows)
		if tp == "" {
			b.Fatal("unexpected empty throughput")
		}
	}
}

// BenchmarkHeadStallTicks measures detection of trailing stalled ticks where HEAD does not advance.
func BenchmarkHeadStallTicks(b *testing.B) {
	stalled := makeBenchmarkRows(24)
	for i := 20; i < 24; i++ {
		stalled[i]["lands"] = stalled[19]["lands"]
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stall := HeadStallTicks(stalled)
		if stall != 4 {
			b.Fatalf("expected 4 stalled ticks, got %d", stall)
		}
	}
}

// BenchmarkSpark measures sparkline rune generation across various series lengths.
func BenchmarkSpark(b *testing.B) {
	lengths := []int{8, 24, 72, 168}
	for _, length := range lengths {
		b.Run(fmt.Sprintf("Len%d", length), func(b *testing.B) {
			values := make([]float64, length)
			for i := range values {
				values[i] = float64((i*7 + 3) % 20)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s := Spark(values)
				if len(s) == 0 {
					b.Fatal("unexpected empty spark")
				}
			}
		})
	}
}

// BenchmarkMetricsOf measures metric extraction and counter mapping from raw snapshot structures.
func BenchmarkMetricsOf(b *testing.B) {
	snap := makeBenchmarkSnapshot(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := MetricsOf(snap)
		if len(m) == 0 {
			b.Fatal("expected metrics")
		}
	}
}

// BenchmarkTailFold measures incremental checkpoint folding performance over an appending ledger.
func BenchmarkTailFold(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "history.jsonl")

	for i := 0; i < 20; i++ {
		metrics := map[string]float64{
			"usable":   float64(5 + i%3),
			"live":     float64(2 + i%2),
			"sessions": float64(10 + i),
			"escalate": 0,
			"lands":    float64(i * 2),
			"resumes":  float64(i * 5),
			"deaths":   float64(i),
		}
		if _, err := Append(path, metrics, fmt.Sprintf("2026-09-01T%02d:00:00Z", i), DefaultCap); err != nil {
			b.Fatal(err)
		}
	}

	Tail(path, 24)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows := Tail(path, 24)
		if len(rows) != 20 {
			b.Fatalf("expected 20 rows, got %d", len(rows))
		}
	}
}
