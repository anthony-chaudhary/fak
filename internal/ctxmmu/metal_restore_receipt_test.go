package ctxmmu

import (
	"testing"
	"time"
)

// TestMetalRestorePerformanceReceiptRequiresMeasuredInputs is the controlled-input
// witness for issue #12601. It replaces the timer-dependent TenXTTFTSpeedup
// assertion: performance attribution is produced only from explicit, run-bound
// inputs, never from a historical constant or an unstable elapsed clock.
func TestMetalRestorePerformanceReceiptRequiresMeasuredInputs(t *testing.T) {
	t.Run("missing baseline remains unmeasured", func(t *testing.T) {
		got := deriveMetalRestorePerformance(4096, 500*time.Microsecond, 0)
		if got.Measured || got.EstimatedColdPrefill != 0 || got.SpeedupRatio != 0 {
			t.Fatalf("receipt = %+v, want explicitly unmeasured zero values", got)
		}
	})

	t.Run("zero duration remains unmeasured", func(t *testing.T) {
		got := deriveMetalRestorePerformance(4096, 0, 2*time.Second)
		if got.Measured || got.SpeedupRatio != 0 {
			t.Fatalf("receipt = %+v, want explicitly unmeasured zero ratio", got)
		}
	})

	t.Run("negative baseline remains unmeasured", func(t *testing.T) {
		got := deriveMetalRestorePerformance(4096, 20*time.Millisecond, -2*time.Second)
		if got.Measured || got.SpeedupRatio != 0 {
			t.Fatalf("receipt = %+v, want explicitly unmeasured zero ratio", got)
		}
	})

	t.Run("measured inputs produce the ratio", func(t *testing.T) {
		got := deriveMetalRestorePerformance(4096, 20*time.Millisecond, 2*time.Second)
		if !got.Measured || got.SpeedupRatio != 100 {
			t.Fatalf("receipt = %+v, want measured 100x ratio", got)
		}
		if got.EstimatedColdPrefill != 2*time.Second {
			t.Fatalf("EstimatedColdPrefill = %v, want 2s", got.EstimatedColdPrefill)
		}
	})
}
