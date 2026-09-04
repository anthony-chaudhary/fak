package horizonrecovery

import (
	"fmt"
	"testing"
)

var (
	sinkBand  RecoveryBand
	sinkBands []RecoveryBand
	sinkRatio float64
	sinkKeys  map[string]bool
	sinkErr   error
)

func generateBenchmarkReport(n int) Report {
	sessions := make([]Session, n)
	var totalLinear, totalCompact, totalPlanned, totalFaultTax int64
	var totalRefs, totalFaults, totalServed, totalRefused, totalLoss, totalRecovered int

	for i := 0; i < n; i++ {
		s := Session{
			Source:              fmt.Sprintf("bench-session-%d", i),
			Turns:               40 + (i % 20),
			Budget:              8000,
			LinearCumTok:        400000 + int64(i*5000),
			CompactCumTok:       80000 + int64(i*1000),
			PlannedCumTok:       80000 + int64(i*1000),
			FaultTaxCum:         1000 + int64(i*50),
			References:          30 + (i % 15),
			Faults:              2 + (i % 3),
			Served:              2 + (i % 3),
			Refused:             0,
			FaultRate:           0.05,
			CompactionLossTurns: 3,
			FactsRecovered:      2,
		}
		sessions[i] = s
		totalLinear += s.LinearCumTok
		totalCompact += s.CompactCumTok
		totalPlanned += s.PlannedCumTok
		totalFaultTax += s.FaultTaxCum
		totalRefs += s.References
		totalFaults += s.Faults
		totalServed += s.Served
		totalRefused += s.Refused
		totalLoss += s.CompactionLossTurns
		totalRecovered += s.FactsRecovered
	}

	var faultRate float64
	if totalRefs > 0 {
		faultRate = float64(totalFaults) / float64(totalRefs)
	}

	return Report{
		Budget:   8000,
		Window:   6,
		Sessions: sessions,
		Total: Total{
			Sessions:            n,
			Turns:               50 * n,
			LinearCumTok:        totalLinear,
			CompactCumTok:       totalCompact,
			PlannedCumTok:       totalPlanned,
			FaultTaxCum:         totalFaultTax,
			References:          totalRefs,
			Faults:              totalFaults,
			Served:              totalServed,
			Refused:             totalRefused,
			FaultRate:           faultRate,
			CompactionLossTurns: totalLoss,
			FactsRecovered:      totalRecovered,
		},
	}
}

func BenchmarkSessionBand(b *testing.B) {
	s := Session{
		Source:              "session-prod-001",
		Turns:               50,
		Budget:              8000,
		LinearCumTok:        500000,
		CompactCumTok:       100000,
		PlannedCumTok:       100000,
		FaultTaxCum:         1200,
		References:          40,
		Faults:              2,
		Served:              2,
		Refused:             0,
		FaultRate:           0.05,
		CompactionLossTurns: 3,
		FactsRecovered:      2,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBand = sessionBand(s)
	}
}

func BenchmarkBandsFromReport(b *testing.B) {
	cohortSizes := []int{10, 20, 50, 100, 500}
	for _, size := range cohortSizes {
		b.Run(fmt.Sprintf("sessions_%d", size), func(b *testing.B) {
			rep := generateBenchmarkReport(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBands = BandsFromReport(rep)
			}
		})
	}
}

func BenchmarkAggregateBand(b *testing.B) {
	b.Run("Valid_20_Floor", func(b *testing.B) {
		rep := generateBenchmarkReport(MinAggregateSessions)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBand, sinkErr = AggregateBand(rep)
		}
	})

	b.Run("Valid_100_Cohort", func(b *testing.B) {
		rep := generateBenchmarkReport(100)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBand, sinkErr = AggregateBand(rep)
		}
	})

	b.Run("Valid_500_Large", func(b *testing.B) {
		rep := generateBenchmarkReport(500)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBand, sinkErr = AggregateBand(rep)
		}
	})

	b.Run("Refused_BelowFloor", func(b *testing.B) {
		rep := generateBenchmarkReport(1)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBand, sinkErr = AggregateBand(rep)
		}
	})
}

func BenchmarkRatio(b *testing.B) {
	b.Run("Standard", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkRatio = ratio(500000, 100000)
		}
	})

	b.Run("FloorNoRecovery", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkRatio = ratio(100000, 100000)
		}
	})

	b.Run("ZeroBounded", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkRatio = ratio(100000, 0)
		}
	})
}

func BenchmarkJsonKeys(b *testing.B) {
	band := sessionBand(Session{
		Source:        "bench-session",
		Turns:         50,
		LinearCumTok:  500000,
		CompactCumTok: 100000,
		FaultRate:     0.05,
		Served:        2,
		Refused:       0,
	})

	b.Run("RecoveryBand", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkKeys, sinkErr = jsonKeys(band)
		}
	})
}

func BenchmarkSelfcheck(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = Selfcheck()
	}
}

func BenchmarkParallelAggregateBand(b *testing.B) {
	rep := generateBenchmarkReport(100)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = AggregateBand(rep)
		}
	})
}

func BenchmarkParallelBandsFromReport(b *testing.B) {
	rep := generateBenchmarkReport(50)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = BandsFromReport(rep)
		}
	})
}

func TestBenchmarkExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark execution in short mode")
	}

	tests := []struct {
		name string
		f    func(b *testing.B)
	}{
		{"BenchmarkSessionBand", BenchmarkSessionBand},
		{"BenchmarkSelfcheck", BenchmarkSelfcheck},
		{"BenchmarkParallelAggregateBand", BenchmarkParallelAggregateBand},
		{"BenchmarkParallelBandsFromReport", BenchmarkParallelBandsFromReport},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := testing.Benchmark(tt.f)
			if res.N <= 0 {
				t.Fatalf("%s failed to iterate: N=%d", tt.name, res.N)
			}
		})
	}
}
