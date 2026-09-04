package balance

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

var (
	sinkBudget ResumeBudget
	sinkBool   bool
	sinkStr    string
	sinkRows   []string
)

func generateResumeStates(n int) []resume.ResumeState {
	states := make([]resume.ResumeState, n)
	palette := []resume.ResumeState{
		resume.ResumeTook,
		resume.ResumeReStranded,
		resume.ResumeGaveUp,
		resume.ResumeLaunched,
		resume.ResumePending,
		resume.ResumeSettled,
	}
	for i := 0; i < n; i++ {
		states[i] = palette[i%len(palette)]
	}
	return states
}

func BenchmarkFoldResumeBudget(b *testing.B) {
	for _, size := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("fleet_%d", size), func(b *testing.B) {
			states := generateResumeStates(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBudget = FoldResumeBudget(states)
			}
		})
	}
}

func BenchmarkReStrandingOutpacesCompletion(b *testing.B) {
	cases := []struct {
		name   string
		budget ResumeBudget
	}{
		{"balanced", ResumeBudget{Took: 10, ReStranded: 2, Measured: true}},
		{"red", ResumeBudget{Took: 2, ReStranded: 8, Measured: true}},
		{"unmeasured", ResumeBudget{}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			budget := tc.budget
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBool = budget.ReStrandingOutpacesCompletion()
			}
		})
	}
}

func BenchmarkEvidenceStatus(b *testing.B) {
	cases := []struct {
		name string
		ev   Evidence
	}{
		{
			name: "no_data",
			ev:   Evidence{},
		},
		{
			name: "red",
			ev: Evidence{
				Resume: ResumeBudget{Took: 1, ReStranded: 4, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 2, Throughput: 2, TargetThroughputPct: 50},
			},
		},
		{
			name: "leaning",
			ev: Evidence{
				Resume: ResumeBudget{Took: 5, ReStranded: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 4, Throughput: 1, TargetThroughputPct: 50, Favor: superloop.WorkThroughput},
			},
		},
		{
			name: "ok",
			ev: Evidence{
				Resume: ResumeBudget{Took: 5, ReStranded: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 2, Throughput: 2, TargetThroughputPct: 50},
			},
		},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			ev := tc.ev
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkStr = ev.Status()
			}
		})
	}
}

func BenchmarkRender(b *testing.B) {
	cases := []struct {
		name string
		ev   Evidence
	}{
		{
			name: "fully_measured_balanced",
			ev: Evidence{
				Resume: ResumeBudget{Took: 5, ReStranded: 2, GaveUp: 1, Launched: 3, Settled: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 2, Throughput: 2, Neutral: 1, TargetThroughputPct: 50},
			},
		},
		{
			name: "fully_measured_red",
			ev: Evidence{
				Resume: ResumeBudget{Took: 1, ReStranded: 5, GaveUp: 2, Pending: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 3, Throughput: 1, TargetThroughputPct: 50, Favor: superloop.WorkThroughput},
			},
		},
		{
			name: "fully_measured_leaning",
			ev: Evidence{
				Resume: ResumeBudget{Took: 4, ReStranded: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 1, Throughput: 3, TargetThroughputPct: 50, Favor: superloop.WorkGardening},
			},
		},
		{
			name: "degraded_resume_only",
			ev: Evidence{
				Resume: ResumeBudget{Took: 4, Launched: 1, Settled: 2, Measured: true},
			},
		},
		{
			name: "degraded_mix_only",
			ev: Evidence{
				Mix: &superloop.WorkMix{Gardening: 2, Throughput: 2, TargetThroughputPct: 50},
			},
		},
		{
			name: "degraded_no_data",
			ev:   Evidence{},
		},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			ev := tc.ev
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkRows = Render(ev)
			}
		})
	}
}

func BenchmarkSharePct(b *testing.B) {
	b.Run("standard", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkStr = sharePct(2, 3)
		}
	})
	b.Run("zero_denom", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkStr = sharePct(0, 0)
		}
	})
}

func TestBenchmarkExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark execution in short mode")
	}
	res := testing.Benchmark(func(b *testing.B) {
		states := generateResumeStates(10)
		ev := Evidence{
			Resume: ResumeBudget{Took: 5, ReStranded: 2, Measured: true},
			Mix:    &superloop.WorkMix{Gardening: 2, Throughput: 2, TargetThroughputPct: 50},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBudget = FoldResumeBudget(states)
			sinkBool = sinkBudget.ReStrandingOutpacesCompletion()
			sinkStr = ev.Status()
			sinkRows = Render(ev)
			sinkStr = sharePct(2, 2)
		}
	})
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
