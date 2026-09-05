package rehydrate

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

func newBenchmarkGate() *Gate {
	return NewGate(
		NewRung(ColdCache, func(context.Context) Verdict { return Clear() }),
		NewRung(StaleCred, func(context.Context) Verdict { return Clear() }),
		NewRung(StaleRecall, func(context.Context) Verdict { return Clear() }),
		NewRung(StaleLease, func(context.Context) Verdict { return Clear() }),
		NewRung(StalePlan, func(context.Context) Verdict { return Clear() }),
	)
}

func BenchmarkNewGate(b *testing.B) {
	r1 := NewRung(ColdCache, func(context.Context) Verdict { return Clear() })
	r2 := NewRung(StaleCred, func(context.Context) Verdict { return Clear() })
	r3 := NewRung(StaleRecall, func(context.Context) Verdict { return Clear() })
	r4 := NewRung(StaleLease, func(context.Context) Verdict { return Clear() })
	r5 := NewRung(StalePlan, func(context.Context) Verdict { return Clear() })
	bogus := NewRung(Reason("UNKNOWN_BOGUS"), func(context.Context) Verdict { return Clear() })

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := NewGate(r5, r4, r3, r2, r1, bogus)
		if len(g.rungs) != 5 {
			b.Fatalf("expected 5 rungs, got %d", len(g.rungs))
		}
	}
}

func BenchmarkGateAdmitWarm(b *testing.B) {
	ctx := context.Background()
	gate := newBenchmarkGate()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := gate.Admit(ctx, dormancy.Warm)
		if !adm.Admitted {
			b.Fatalf("warm restore refused: %+v", adm)
		}
	}
}

func BenchmarkGateAdmitCool(b *testing.B) {
	ctx := context.Background()
	gate := newBenchmarkGate()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := gate.Admit(ctx, dormancy.Cool)
		if !adm.Admitted || len(adm.Ran) != 1 {
			b.Fatalf("cool restore unexpected result: %+v", adm)
		}
	}
}

func BenchmarkGateAdmitCold(b *testing.B) {
	ctx := context.Background()
	gate := newBenchmarkGate()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := gate.Admit(ctx, dormancy.Cold)
		if !adm.Admitted || len(adm.Ran) != 3 {
			b.Fatalf("cold restore unexpected result: %+v", adm)
		}
	}
}

func BenchmarkGateAdmitFrozen(b *testing.B) {
	ctx := context.Background()
	gate := newBenchmarkGate()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := gate.Admit(ctx, dormancy.Frozen)
		if !adm.Admitted || len(adm.Ran) != 5 {
			b.Fatalf("frozen restore unexpected result: %+v", adm)
		}
	}
}

func BenchmarkGateAdmitRefusalShortCircuit(b *testing.B) {
	ctx := context.Background()
	gate := NewGate(
		NewRung(ColdCache, func(context.Context) Verdict { return Clear() }),
		NewRung(StaleCred, func(context.Context) Verdict { return Refuse(StaleCred, "oauth token expired") }),
		NewRung(StaleRecall, func(context.Context) Verdict { return Clear() }),
		NewRung(StaleLease, func(context.Context) Verdict { return Clear() }),
		NewRung(StalePlan, func(context.Context) Verdict { return Clear() }),
	)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := gate.Admit(ctx, dormancy.Frozen)
		if adm.Admitted || adm.RefusedBy != StaleCred || len(adm.Ran) != 2 {
			b.Fatalf("expected short-circuit at StaleCred: %+v", adm)
		}
	}
}

func BenchmarkGateAdmitParallel(b *testing.B) {
	ctx := context.Background()
	gate := newBenchmarkGate()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			adm := gate.Admit(ctx, dormancy.Frozen)
			if !adm.Admitted {
				b.Fatalf("parallel frozen restore refused: %+v", adm)
			}
		}
	})
}

func BenchmarkCacheProjectionAdmit(b *testing.B) {
	ctx := context.Background()
	input := resume.Input{
		IdleSeconds:           301,
		TTL:                   resume.TTL5m,
		EffectiveReuseSeconds: 300,
		ResidentTokens:        256,
		ShedBudgetTokens:      128,
		HorizonTurns:          4,
		Pricing:               resume.Pricing{InputPerMTokUSD: 1, OutputPerMTokUSD: 1},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		projection := NewCacheProjection(input)
		gate := NewGate(projection.Rung())
		adm := gate.Admit(ctx, dormancy.Cool)
		if adm.Admitted || adm.RefusedBy != ColdCache {
			b.Fatalf("expected ColdCache refusal: %+v", adm)
		}
	}
}

func BenchmarkBGLoopGateAdmit(b *testing.B) {
	ctx := context.Background()
	bg := BGLoopGate{Gate: newBenchmarkGate()}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adm := bg.Admit(ctx, dormancy.Frozen)
		if !adm.Admitted {
			b.Fatalf("bgloop admit refused: %+v", adm)
		}
	}
}

func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	res := testing.Benchmark(BenchmarkGateAdmitFrozen)
	if res.N <= 0 {
		t.Fatalf("benchmark did not run: %+v", res)
	}
}
