package callavoid

import (
	"fmt"
	"testing"
)

var (
	benchMemoProofSink       MemoProof
	benchTurnReportSink      TurnReport
	benchWitnessSink         WitnessedRedirect
	benchWitnessesSink       []WitnessedRedirect
	benchGateDecisionSink    ClassGateDecision
	benchGateDecisionsSink   []ClassGateDecision
	benchAdmittedClassesSink []string
	benchObservedGateSink    ObservedClassGate
	benchObservedGatesSink   []ObservedClassGate
	benchSessionReportSink   SessionReport
	benchTallySink           Tally
)

func BenchmarkProveMemo(b *testing.B) {
	cases := []struct {
		name  string
		input MemoInput
	}{
		{
			name:  "ReusePays",
			input: MemoInput{Accesses: 20, ValidateCost: 0.01, MutationRate: 0.05, CaptureCost: 0.02},
		},
		{
			name:  "SingleUseLoss",
			input: MemoInput{Accesses: 1, ValidateCost: 0.02, MutationRate: 0.10, CaptureCost: 0.02},
		},
		{
			name:  "VolatileRefute",
			input: MemoInput{Accesses: 50, ValidateCost: 0.02, MutationRate: 0.95, CaptureCost: 0.02},
		},
		{
			name:  "ExpensiveValidation",
			input: MemoInput{Accesses: 100, ValidateCost: 1.0, MutationRate: 0.01, CaptureCost: 0.02},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchMemoProofSink = ProveMemo(tc.input)
			}
		})
	}
}

func BenchmarkAccount(b *testing.B) {
	b.Run("RealizedOnly", func(b *testing.B) {
		tally := Tally{
			Execute:      25,
			MemoHit:      120,
			Repair:       15,
			StaleMiss:    5,
			HardDeny:     10,
			ValidateCost: 0.01,
			CaptureCost:  0.02,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTurnReportSink = Account(tally)
		}
	})

	b.Run("SpeculativeRedirects", func(b *testing.B) {
		redirects := make([]int, 64)
		for i := range redirects {
			redirects[i] = (i * 37) % 2048
		}
		tally := Tally{
			Execute:      25,
			MemoHit:      120,
			Repair:       15,
			Redirects:    redirects,
			ValidateCost: 0.01,
			CaptureCost:  0.02,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTurnReportSink = Account(tally)
		}
	})

	b.Run("WitnessedRedirects", func(b *testing.B) {
		witnesses := make([]WitnessedRedirect, 16)
		for i := range witnesses {
			witnesses[i] = WitnessedRedirect{
				Rule: fmt.Sprintf("policy.rule.%d", i),
				Variants: []string{
					"variant_alpha", "variant_beta", "variant_gamma", "variant_delta",
					"variant_alpha", "", "variant_epsilon", "variant_zeta",
				},
			}
		}
		tally := Tally{
			Execute:            20,
			MemoHit:            80,
			Repair:             10,
			HardDeny:           5,
			WitnessedRedirects: witnesses,
			ValidateCost:       0.01,
			CaptureCost:        0.02,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTurnReportSink = Account(tally)
		}
	})
}

func BenchmarkWitnessFromCoverage(b *testing.B) {
	covered := make([]string, 60)
	for i := range covered {
		covered[i] = fmt.Sprintf("read:/repo/subpath/%d", i)
	}
	cov := DenyRuleCoverage{
		Rule:      "policy.read_scope",
		Covered:   covered,
		MaxFanout: 32,
	}

	pruned := make([]string, 40)
	for i := range pruned {
		switch {
		case i%6 == 0:
			pruned[i] = ""
		case i%4 == 0:
			pruned[i] = fmt.Sprintf("read:/repo/subpath/%d", i%12)
		case i%5 == 0:
			pruned[i] = fmt.Sprintf("write:/external/%d", i)
		default:
			pruned[i] = fmt.Sprintf("read:/repo/subpath/%d", i)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchWitnessSink = WitnessFromCoverage(cov, pruned)
	}
}

func BenchmarkWitnessesFromCoverageBatch(b *testing.B) {
	fired := make([]FiredDeny, 10)
	for i := range fired {
		covered := make([]string, 20)
		for j := range covered {
			covered[j] = fmt.Sprintf("scope:%d:target:%d", i, j)
		}
		pruned := make([]string, 12)
		for j := range pruned {
			pruned[j] = fmt.Sprintf("scope:%d:target:%d", i, j%16)
		}
		fired[i] = FiredDeny{
			Coverage: DenyRuleCoverage{
				Rule:      fmt.Sprintf("policy.rule.%d", i),
				Covered:   covered,
				MaxFanout: 10,
			},
			Pruned: pruned,
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchWitnessesSink = WitnessesFromCoverage(fired)
	}
}

func BenchmarkGateClasses(b *testing.B) {
	classes := []ClassMemoInput{
		{Class: "Read", Accesses: 25, ValidateCost: 0.01, MutationRate: 0.02, CaptureCost: 0.01},
		{Class: "Grep", Accesses: 15, ValidateCost: 0.02, MutationRate: 0.05, CaptureCost: 0.02},
		{Class: "Glob", Accesses: 10, ValidateCost: 0.02, MutationRate: 0.08, CaptureCost: 0.02},
		{Class: "Bash", Accesses: 5, ValidateCost: 0.10, MutationRate: 0.80, CaptureCost: 0.05},
		{Class: "Edit", Accesses: 2, ValidateCost: 0.05, MutationRate: 0.95, CaptureCost: 0.05},
		{Class: "Fetch", Accesses: 8, ValidateCost: 0.03, MutationRate: 0.10, CaptureCost: 0.03},
		{Class: "Diff", Accesses: 12, ValidateCost: 0.01, MutationRate: 0.04, CaptureCost: 0.01},
		{Class: "Search", Accesses: 50, ValidateCost: 0.85, MutationRate: 0.10, CaptureCost: 0.20},
	}

	b.Run("GateBatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchGateDecisionsSink = GateClasses(classes)
		}
	})

	b.Run("AdmittedProjection", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmittedClassesSink = AdmittedClasses(classes)
		}
	})
}

func BenchmarkFoldClassObservations(b *testing.B) {
	observations := []ClassObservation{
		{Class: "Read", ReuseAttempts: 60, Invalidations: 2, ValidateCostSamples: 1.2, ValidateCostCount: 60, CaptureCostSamples: 0.6, CaptureCostCount: 10},
		{Class: "Grep", ReuseAttempts: 35, Invalidations: 3, ValidateCostSamples: 0.7, ValidateCostCount: 35, CaptureCostSamples: 0.35, CaptureCostCount: 7},
		{Class: "Glob", ReuseAttempts: 20, Invalidations: 2, ValidateCostSamples: 0.4, ValidateCostCount: 20, CaptureCostSamples: 0.2, CaptureCostCount: 5},
		{Class: "Bash", ReuseAttempts: 12, Invalidations: 10, ValidateCostSamples: 0.6, ValidateCostCount: 6, CaptureCostSamples: 0.6, CaptureCostCount: 6},
		{Class: "Write", ReuseAttempts: 0, Invalidations: 0},
		{Class: "Diff", ReuseAttempts: 18, Invalidations: 1, ValidateCostSamples: 0.18, ValidateCostCount: 18, CaptureCostSamples: 0.18, CaptureCostCount: 4},
		{Class: "Patch", ReuseAttempts: 6, Invalidations: 5, ValidateCostSamples: 0.3, ValidateCostCount: 3, CaptureCostSamples: 0.3, CaptureCostCount: 3},
		{Class: "Status", ReuseAttempts: 40, Invalidations: 3},
	}

	b.Run("FoldBatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchObservedGatesSink = FoldClassObservations(observations)
		}
	})

	b.Run("AdmittedFromObservations", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAdmittedClassesSink = AdmittedFromObservations(observations)
		}
	})
}

func BenchmarkTallyFromCountersWitnessed(b *testing.B) {
	counters := Counters{
		EngineCalls: 60,
		VDSOHits:    240,
		Transforms:  20,
		Denies:      25,
	}
	witnessed := make([]WitnessedRedirect, 10)
	for i := range witnessed {
		witnessed[i] = WitnessedRedirect{
			Rule:     fmt.Sprintf("policy.gate.%d", i),
			Variants: []string{"opt_a", "opt_b", "opt_c"},
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchTallySink = TallyFromCountersWitnessed(counters, witnessed)
	}
}

func BenchmarkAccountFromObservations(b *testing.B) {
	counters := Counters{
		EngineCalls: 45,
		VDSOHits:    180,
		Transforms:  15,
		Denies:      20,
	}

	witnessed := make([]WitnessedRedirect, 8)
	for i := range witnessed {
		witnessed[i] = WitnessedRedirect{
			Rule:     fmt.Sprintf("policy.rule.%d", i),
			Variants: []string{"variant_1", "variant_2", "variant_3"},
		}
	}

	observations := []ClassObservation{
		{Class: "Read", ReuseAttempts: 50, Invalidations: 2, ValidateCostSamples: 1.0, ValidateCostCount: 50},
		{Class: "Grep", ReuseAttempts: 25, Invalidations: 2, ValidateCostSamples: 0.5, ValidateCostCount: 25},
		{Class: "Glob", ReuseAttempts: 15, Invalidations: 1, ValidateCostSamples: 0.3, ValidateCostCount: 15},
		{Class: "Bash", ReuseAttempts: 10, Invalidations: 9},
	}

	speculative := []int{12, 18, 24, 30}

	b.Run("ComposedSession", func(b *testing.B) {
		session := SessionInput{
			Counters:             counters,
			WitnessedDenies:      witnessed,
			ClassObservations:    observations,
			SpeculativeRedirects: speculative,
			ValidateCost:         0.01,
			CaptureCost:          0.02,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSessionReportSink = AccountFromObservations(session)
		}
	})

	b.Run("PureNoToolCall", func(b *testing.B) {
		session := SessionInput{
			Counters: Counters{
				EngineCalls: 0,
				VDSOHits:    150,
				Transforms:  20,
				Denies:      10,
			},
			WitnessedDenies:   witnessed,
			ClassObservations: observations,
			ValidateCost:      0.01,
			CaptureCost:       0.02,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSessionReportSink = AccountFromObservations(session)
		}
	})

	b.Run("EmptySession", func(b *testing.B) {
		session := SessionInput{}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSessionReportSink = AccountFromObservations(session)
		}
	})
}

// TestBenchmarkSanity ensures benchmark routines execute without panic and perform iterations.
func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	res := testing.Benchmark(BenchmarkAccountFromObservations)
	if res.N <= 0 {
		t.Fatalf("expected positive iterations, got %d", res.N)
	}
}
