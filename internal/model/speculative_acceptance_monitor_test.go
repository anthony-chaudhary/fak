package model

import "testing"

// TestSpeculativeRollingAcceptanceFallback is the software witness for the native
// Qwen3.8 MTP rolling acceptance-rate monitor: a 32-token rolling window tracks
// draft acceptance, and the engine falls back to unassisted serial decode once
// the rolling rate drops below the 50% floor, without corrupting session state.
func TestSpeculativeRollingAcceptanceFallback(t *testing.T) {
	t.Run("DefaultsMatchProductionContract", func(t *testing.T) {
		cfg := DefaultSpeculativeEngineConfig()
		if cfg.MinAcceptanceRate != 0.50 {
			t.Errorf("MinAcceptanceRate = %v, want 0.50", cfg.MinAcceptanceRate)
		}
		if cfg.AcceptanceWindow != 32 {
			t.Errorf("AcceptanceWindow = %d, want 32", cfg.AcceptanceWindow)
		}
		if !cfg.FallbackToSerial {
			t.Error("FallbackToSerial = false, want true")
		}
	})

	t.Run("HighAcceptanceStaysSpeculative", func(t *testing.T) {
		eng := NewSpeculativeEngine(nil, nil, DefaultSpeculativeEngineConfig())
		// Eight rounds of K=4 drafts, three accepted each (75%): above the floor.
		for i := 0; i < 8; i++ {
			eng.RecordVerificationOutcome(4, 3, 1, true)
		}
		stats := eng.AcceptanceStats()
		if stats.InFallback {
			t.Fatalf("InFallback = true with rolling rate %v > floor", stats.RollingRate)
		}
		if stats.TripwireTripped {
			t.Fatal("TripwireTripped = true with healthy acceptance")
		}
		if stats.RollingRate < 0.50 {
			t.Errorf("RollingRate = %v, want >= 0.50", stats.RollingRate)
		}
		if got := len(eng.windowOutcomes); got != 32 {
			t.Errorf("rolling window length = %d, want 32", got)
		}
	})

	t.Run("SustainedLowAcceptanceFallsBackToSerial", func(t *testing.T) {
		eng := NewSpeculativeEngine(nil, nil, DefaultSpeculativeEngineConfig())
		// Eight rounds of K=4 drafts, zero accepted (0%): well below the floor.
		for i := 0; i < 8; i++ {
			eng.RecordVerificationOutcome(4, 0, 4, false)
		}
		stats := eng.AcceptanceStats()
		if !stats.InFallback {
			t.Fatalf("InFallback = false with rolling rate %v < floor", stats.RollingRate)
		}
		if !stats.TripwireTripped {
			t.Fatal("TripwireTripped = false after sustained sub-floor acceptance")
		}
		if stats.FallbackReason == "" {
			t.Error("FallbackReason is empty on an active fallback")
		}
	})

	t.Run("PartialWindowBelowFloorDoesNotTripEarly", func(t *testing.T) {
		eng := NewSpeculativeEngine(nil, nil, DefaultSpeculativeEngineConfig())
		// One bad round of four tokens is a partial window; it must warn, not abort.
		eng.RecordVerificationOutcome(4, 0, 4, false)
		stats := eng.AcceptanceStats()
		if stats.InFallback {
			t.Fatalf("InFallback = true after a single bad round (window %d/32, rate %v)", stats.WindowTokens, stats.RollingRate)
		}
	})

	t.Run("RecoveryAboveFloorDoesNotResurrectSeriallyFallenBackEngine", func(t *testing.T) {
		eng := NewSpeculativeEngine(nil, nil, DefaultSpeculativeEngineConfig())
		for i := 0; i < 8; i++ {
			eng.RecordVerificationOutcome(4, 0, 4, false)
		}
		if !eng.AcceptanceStats().InFallback {
			t.Fatal("precondition: engine did not enter fallback")
		}
		// A later healthy round must not silently re-enable speculation mid-request;
		// the fallback is sticky until the engine is explicitly reset.
		eng.RecordVerificationOutcome(4, 4, 0, true)
		if !eng.AcceptanceStats().InFallback {
			t.Error("InFallback cleared without an explicit reset")
		}
		eng.Reset()
		if eng.AcceptanceStats().InFallback {
			t.Error("InFallback = true after Reset")
		}
	})

	t.Run("UnsetFloorNeverTrips", func(t *testing.T) {
		cfg := DefaultSpeculativeEngineConfig()
		cfg.MinAcceptanceRate = 0
		cfg.FallbackToSerial = false
		eng := NewSpeculativeEngine(nil, nil, cfg)
		for i := 0; i < 16; i++ {
			eng.RecordVerificationOutcome(4, 0, 4, false)
		}
		if eng.AcceptanceStats().InFallback {
			t.Error("InFallback = true with fallback disabled")
		}
	})
}
