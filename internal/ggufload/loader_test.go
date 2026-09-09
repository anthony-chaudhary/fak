package ggufload

import (
	"math"
	"runtime/debug"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestGGUFLoadMemoryLimitPreserved(t *testing.T) {
	origLimit := debug.SetMemoryLimit(-1)
	defer debug.SetMemoryLimit(origLimit)

	total, _, known := compute.HostSystemMemoryInfo()
	if !known || total <= 0 {
		t.Skip("Host physical memory unknown on this platform")
	}

	t.Run("explicit memory limit preserved", func(t *testing.T) {
		const explicitLimit = int64(4 << 30) // 4 GiB
		debug.SetMemoryLimit(explicitLimit)
		defer debug.SetMemoryLimit(origLimit)

		applied, changed := ConfigureHostMemoryLimit(DefaultMemoryLimitFraction)
		if changed {
			t.Fatalf("expected changed=false when explicit limit was configured, got changed=true (applied=%d)", applied)
		}
		if applied != explicitLimit {
			t.Fatalf("applied limit = %d, want explicit limit %d", applied, explicitLimit)
		}
		if current := debug.SetMemoryLimit(-1); current != explicitLimit {
			t.Fatalf("current memory limit = %d, want preserved explicit limit %d", current, explicitLimit)
		}
	})

	t.Run("GOMEMLIMIT env var preserved", func(t *testing.T) {
		debug.SetMemoryLimit(math.MaxInt64)
		defer debug.SetMemoryLimit(origLimit)

		t.Setenv("GOMEMLIMIT", "8GiB")
		applied, changed := ConfigureHostMemoryLimit(DefaultMemoryLimitFraction)
		if changed {
			t.Fatalf("expected changed=false when GOMEMLIMIT is set in env, got changed=true (applied=%d)", applied)
		}
		if applied != math.MaxInt64 {
			t.Fatalf("expected applied=%d, got %d", math.MaxInt64, applied)
		}
	})

	t.Run("unset limit applies default fraction of host capacity", func(t *testing.T) {
		debug.SetMemoryLimit(math.MaxInt64)
		defer debug.SetMemoryLimit(origLimit)

		applied, changed := ConfigureHostMemoryLimit(DefaultMemoryLimitFraction)
		expectedLimit := int64(float64(total) * DefaultMemoryLimitFraction)
		if !changed {
			t.Fatalf("expected changed=true when memory limit was unset, got changed=false")
		}
		if applied != expectedLimit {
			t.Fatalf("applied limit = %d, want %d (85%% of total %d)", applied, expectedLimit, total)
		}
		if current := debug.SetMemoryLimit(-1); current != expectedLimit {
			t.Fatalf("current memory limit = %d, want %d", current, expectedLimit)
		}
	})

	t.Run("load memory governor restores limits and gc percent", func(t *testing.T) {
		const baseLimit = int64(16 << 30)
		debug.SetMemoryLimit(baseLimit)
		defer debug.SetMemoryLimit(origLimit)

		origGC := debug.SetGCPercent(100)
		defer debug.SetGCPercent(origGC)

		gov := StartLoadMemoryGovernor(0.80, 50)
		if currentGC := debug.SetGCPercent(50); currentGC != 50 {
			t.Fatalf("governor did not set GOGC to 50, got %d", currentGC)
		}

		gov.Close()

		if current := debug.SetMemoryLimit(-1); current != baseLimit {
			t.Fatalf("memory limit after Close = %d, want restored base limit %d", current, baseLimit)
		}
		if currentGC := debug.SetGCPercent(100); currentGC != 100 {
			t.Fatalf("GOGC after Close = %d, want restored 100", currentGC)
		}
	})

	t.Run("gc pacer triggers on large tensors and intervals", func(t *testing.T) {
		pacer := NewGCPacer(WithGCPacingEnabled(true), WithLargeTensorThreshold(200*1024*1024), WithPacingInterval(4))

		// Tensor smaller than threshold: no GC
		if pacer.Pace("small.weight", 50*1024*1024) {
			t.Fatal("pacer triggered GC on tensor smaller than threshold")
		}
		if pacer.GCCount() != 0 {
			t.Fatalf("gc count = %d, want 0", pacer.GCCount())
		}

		// Large tensor >= 200MB: triggers GC
		if !pacer.Pace("blk.0.ffn_gate.weight", 250*1024*1024) {
			t.Fatal("pacer did not trigger GC on large tensor > 200MB")
		}
		if pacer.GCCount() != 1 {
			t.Fatalf("gc count = %d, want 1", pacer.GCCount())
		}

		// Interval pacing: after 4 tensors
		pacer.Reset()
		for i := 1; i <= 3; i++ {
			if pacer.Pace("tensor.weight", 10*1024*1024) {
				t.Fatalf("pacer prematurely triggered GC at tensor %d", i)
			}
		}
		if !pacer.Pace("tensor.weight", 10*1024*1024) {
			t.Fatal("pacer did not trigger GC on 4th tensor (interval=4)")
		}
		if pacer.GCCount() != 1 {
			t.Fatalf("gc count = %d, want 1", pacer.GCCount())
		}
	})
}
