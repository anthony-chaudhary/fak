package breathgate

import (
	"context"
	"testing"
	"time"
)

// BenchmarkBreathGateCheck measures gate evaluation latency in a b.N loop.
func BenchmarkBreathGateCheck(b *testing.B) {
	g := New(Config{MinInterval: 0})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !g.Check() {
			b.Fatal("expected gate check to admit immediately")
		}
	}
}

// BenchmarkBreathGateCheckActiveCooldown measures gate evaluation when cooldown is active.
func BenchmarkBreathGateCheckActiveCooldown(b *testing.B) {
	g := New(Config{
		MinInterval: 10 * time.Millisecond,
		Cooldown:    time.Hour,
	})
	g.TriggerCooldown()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.Check()
	}
}

// BenchmarkBreathGateRemaining measures remaining duration calculation in a b.N loop.
func BenchmarkBreathGateRemaining(b *testing.B) {
	g := New(DefaultConfig())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.Remaining()
	}
}

// BenchmarkBreathGateCheckAndRecord measures checking and recording turn completion in a b.N loop.
func BenchmarkBreathGateCheckAndRecord(b *testing.B) {
	g := New(Config{MinInterval: 0})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if g.Check() {
			g.RecordTurn()
		}
	}
}

// BenchmarkBreathGateWaitZeroInterval measures zero-interval wait throughput in a b.N loop.
func BenchmarkBreathGateWaitZeroInterval(b *testing.B) {
	g := New(Config{MinInterval: 0})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.Wait(ctx)
	}
}

// BenchmarkBreathGateParallelCheck measures concurrent gate checks across goroutines.
func BenchmarkBreathGateParallelCheck(b *testing.B) {
	g := New(Config{MinInterval: 0})
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = g.Check()
		}
	})
}

// BenchmarkBreathGate exercises gate checking, pacing reservation, and burst tracking in sub-benchmarks.
func BenchmarkBreathGate(b *testing.B) {
	b.Run("Check", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = g.Check()
		}
	})

	b.Run("CheckAndRecord", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if g.Check() {
				g.RecordTurn()
			}
		}
	})

	b.Run("Remaining", func(b *testing.B) {
		g := New(DefaultConfig())
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = g.Remaining()
		}
	})

	b.Run("WaitZeroInterval", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		ctx := context.Background()
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = g.Wait(ctx)
		}
	})
}

// TestBenchmarkGateSanity validates gate behavior under benchmark configurations.
func TestBenchmarkGateSanity(t *testing.T) {
	g := New(Config{MinInterval: 0})
	if !g.Check() {
		t.Fatal("expected Check to return true for zero MinInterval")
	}
	g.RecordTurn()
	if !g.Check() {
		t.Fatal("expected Check to remain true for zero MinInterval")
	}
}

// TestBenchmarkBreathGateCheckExecution verifies that BenchmarkBreathGateCheck executes cleanly.
func TestBenchmarkBreathGateCheckExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkBreathGateCheck)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
