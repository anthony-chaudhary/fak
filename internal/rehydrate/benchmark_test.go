package rehydrate

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

var (
	benchAdmissionSink Admission
	benchGateSink      *Gate
	benchProjSink      CacheProjection
)

func makeBenchGate() *Gate {
	clearRung := func(r Reason) Rung {
		return NewRung(r, func(context.Context) Verdict {
			return Clear()
		})
	}
	return NewGate(
		clearRung(ColdCache),
		clearRung(StaleCred),
		clearRung(StaleRecall),
		clearRung(StaleLease),
		clearRung(StalePlan),
	)
}

func BenchmarkGateAdmitWarm(b *testing.B) {
	g := makeBenchGate()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAdmissionSink = g.Admit(ctx, dormancy.Warm)
	}
}

func BenchmarkGateAdmitCool(b *testing.B) {
	g := makeBenchGate()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAdmissionSink = g.Admit(ctx, dormancy.Cool)
	}
}

func BenchmarkGateAdmitFullLadder(b *testing.B) {
	g := makeBenchGate()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAdmissionSink = g.Admit(ctx, dormancy.Frozen)
	}
}

func BenchmarkGateAdmitShortCircuit(b *testing.B) {
	refuseRung := func(r Reason, shouldRefuse bool) Rung {
		return NewRung(r, func(context.Context) Verdict {
			if shouldRefuse {
				return Refuse(r, "lease expired")
			}
			return Clear()
		})
	}
	g := NewGate(
		refuseRung(ColdCache, false),
		refuseRung(StaleCred, true),
		refuseRung(StaleRecall, false),
		refuseRung(StaleLease, false),
		refuseRung(StalePlan, false),
	)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAdmissionSink = g.Admit(ctx, dormancy.Frozen)
	}
}

func BenchmarkNewGate(b *testing.B) {
	rungs := []Rung{
		NewRung(ColdCache, nil),
		NewRung(StaleCred, nil),
		NewRung(StaleRecall, nil),
		NewRung(StaleLease, nil),
		NewRung(StalePlan, nil),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchGateSink = NewGate(rungs...)
	}
}

func BenchmarkCacheProjection(b *testing.B) {
	in := resume.Input{
		IdleSeconds:           120,
		TTL:                   resume.TTL5m,
		EffectiveReuseSeconds: 300,
		ResidentTokens:        1024,
		ShedBudgetTokens:      256,
		HorizonTurns:          4,
		Pricing:               resume.Pricing{InputPerMTokUSD: 1, OutputPerMTokUSD: 1},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchProjSink = NewCacheProjection(in)
	}
}

func BenchmarkBGLoopGateAdmit(b *testing.B) {
	bg := BGLoopGate{Gate: makeBenchGate()}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bg.Admit(ctx, dormancy.Frozen)
	}
}
