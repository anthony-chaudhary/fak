package dormancysim

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// BenchmarkDormancySim exercises dormancy simulation in a loop across horizon transitions.
func BenchmarkDormancySim(b *testing.B) {
	ctx := context.Background()
	gate := rehydrate.NewGate(
		rehydrate.NewRung(rehydrate.ColdCache, func(context.Context) rehydrate.Verdict { return rehydrate.Clear() }),
		rehydrate.NewRung(rehydrate.StaleCred, func(context.Context) rehydrate.Verdict { return rehydrate.Clear() }),
		rehydrate.NewRung(rehydrate.StaleRecall, func(context.Context) rehydrate.Verdict { return rehydrate.Clear() }),
		rehydrate.NewRung(rehydrate.StaleLease, func(context.Context) rehydrate.Verdict { return rehydrate.Clear() }),
		rehydrate.NewRung(rehydrate.StalePlan, func(context.Context) rehydrate.Verdict { return rehydrate.Clear() }),
	)
	stamp := dormancy.At(epoch)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sim := New(epoch, stamp, gate)
		adm := sim.Advance(ctx, 90*24*time.Hour)
		if !adm.Admitted {
			b.Fatalf("advance not admitted: %+v", adm)
		}
	}
}

// TestBenchmarkDormancySim verifies the benchmark function executes without errors.
func TestBenchmarkDormancySim(t *testing.T) {
	res := testing.Benchmark(BenchmarkDormancySim)
	if res.N <= 0 {
		t.Fatalf("benchmark did not run: %+v", res)
	}
}
