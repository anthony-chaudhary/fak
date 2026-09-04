package fleetcompare

import "testing"

func benchmarkCols() map[string][]float64 {
	return map[string][]float64{
		"agents":            {50, 50, 50, 20, 20, 10, 10, 50, 30, 40},
		"turns":             {30, 10, 20, 30, 10, 10, 20, 40, 50, 60},
		"shared_saved_mean": {90, 40, 65, 30, 20, 15, 25, 95, 45, 55},
		"cross_uplift_mean": {25, 10, 18, 8, 5, 3, 6, 28, 12, 16},
	}
}

func BenchmarkCompare(b *testing.B) {
	cols := benchmarkCols()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SliceFixed(cols, "agents", 50)
	}
}

func BenchmarkCompareParallel(b *testing.B) {
	cols := benchmarkCols()
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = SliceFixed(cols, "agents", 50)
		}
	})
}
