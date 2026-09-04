package breathgate

import (
	"context"
	"testing"
)

// BenchmarkBreathGate exercises gate checking, pacing reservation, and burst tracking in a loop.
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

// TestBenchmarkGateSanity validates that gate checking operates correctly during benchmarks.
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
