package mtptune

import (
	"sync"
	"testing"
)

// TestAppleSiliconMTPAdaptiveDepthScaling verifies upscaling on high acceptance (>= 0.82)
// and graceful downscaling on lower acceptance (<= 0.60), boundary clamping in [1, 4],
// and hysteresis stability.
func TestAppleSiliconMTPAdaptiveDepthScaling(t *testing.T) {
	tuner := NewAppleSiliconMTPDepthTuner()

	// Initial depth should be 1
	if tuner.CurrentK() != 1 {
		t.Fatalf("expected initial draft depth K=1, got %d", tuner.CurrentK())
	}

	// -------------------------------------------------------------
	// 1. High Acceptance Phase: Upscaling from K=1 -> 2 -> 3 -> 4
	// -------------------------------------------------------------

	// Window 1: 32 tokens, 100% acceptance (32/32 = 1.00 >= 0.82)
	tuner.RecordStep(32, 32)
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected K=2 after 32 accepted tokens (rate=1.0), got %d", tuner.CurrentK())
	}
	if tuner.LastAction() != ActionScaleUp {
		t.Fatalf("expected last action %s, got %s", ActionScaleUp, tuner.LastAction())
	}
	if tuner.RollingAcceptanceRate() != 1.0 {
		t.Fatalf("expected rolling acceptance 1.0, got %f", tuner.RollingAcceptanceRate())
	}

	// Window 2: 32 tokens, 93.75% acceptance (30/32 = 0.9375 >= 0.82)
	tuner.RecordStep(32, 30)
	if tuner.CurrentK() != 3 {
		t.Fatalf("expected K=3 after high acceptance (30/32), got %d", tuner.CurrentK())
	}

	// Window 3: 32 tokens, 87.5% acceptance (28/32 = 0.875 >= 0.82)
	tuner.RecordStep(32, 28)
	if tuner.CurrentK() != 4 {
		t.Fatalf("expected K=4 after high acceptance (28/32), got %d", tuner.CurrentK())
	}

	// Window 4: High acceptance at maximum depth (K=4 ceiling clamp)
	tuner.RecordStep(32, 32)
	if tuner.CurrentK() != 4 {
		t.Fatalf("expected K to remain clamped at max 4, got %d", tuner.CurrentK())
	}
	if tps := tuner.EffectiveThroughput(); tps < 22.0 {
		t.Fatalf("expected effective decode throughput >= 22.0 tok/s on Apple Silicon Metal at K=4, got %.2f tok/s", tps)
	}

	// -------------------------------------------------------------
	// 2. Low Acceptance Phase: Graceful Downscaling K=4 -> 3 -> 2 -> 1
	// -------------------------------------------------------------

	// Window 5: 32 tokens, 50% acceptance (16/32 = 0.50 <= 0.60)
	// Gracefully downscale from K=4 to K=3
	tuner.RecordStep(32, 16)
	if tuner.CurrentK() != 3 {
		t.Fatalf("expected graceful downscale to K=3 after low acceptance (16/32), got %d", tuner.CurrentK())
	}
	if tuner.LastAction() != ActionScaleDown {
		t.Fatalf("expected last action %s, got %s", ActionScaleDown, tuner.LastAction())
	}
	if tuner.RollingAcceptanceRate() != 0.50 {
		t.Fatalf("expected rolling acceptance 0.50, got %f", tuner.RollingAcceptanceRate())
	}

	// Window 6: 32 tokens, 37.5% acceptance (12/32 = 0.375 <= 0.60)
	// Gracefully downscale from K=3 to K=2
	tuner.RecordStep(32, 12)
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected graceful downscale to K=2 after low acceptance (12/32), got %d", tuner.CurrentK())
	}

	// Window 7: 32 tokens, 25% acceptance (8/32 = 0.25 <= 0.60)
	// Gracefully downscale from K=2 to K=1
	tuner.RecordStep(32, 8)
	if tuner.CurrentK() != 1 {
		t.Fatalf("expected graceful downscale to K=1 after low acceptance (8/32), got %d", tuner.CurrentK())
	}

	// Window 8: Low acceptance at minimum depth (K=1 floor clamp)
	tuner.RecordStep(32, 0)
	if tuner.CurrentK() != 1 {
		t.Fatalf("expected K to remain clamped at min 1, got %d", tuner.CurrentK())
	}

	// -------------------------------------------------------------
	// 3. Deadband Phase: Acceptance in (0.60, 0.82) holds steady
	// -------------------------------------------------------------

	// Scale up to K=2 first
	tuner.RecordStep(32, 32)
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected K=2 after upscale, got %d", tuner.CurrentK())
	}

	// Feed 32 tokens with 71.875% acceptance (23/32 = 0.71875, between 0.60 and 0.82)
	tuner.RecordStep(32, 23)
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected depth to hold at K=2 in deadband (rate=%.3f), got %d",
			tuner.RollingAcceptanceRate(), tuner.CurrentK())
	}
}

// TestAppleSiliconMTPHysteresisCooldown verifies that rapid depth oscillations
// are actively prevented by the hysteresis cooldown window.
func TestAppleSiliconMTPHysteresisCooldown(t *testing.T) {
	tuner := NewAppleSiliconMTPDepthTuner()

	// Fill first window with high acceptance -> triggers upscale K=1 -> K=2
	tuner.RecordStep(32, 32)
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected K=2, got %d", tuner.CurrentK())
	}

	// Immediately record a burst of 10 rejected tokens.
	// Even though transient rate is dropping, tuner is in cooldown (tokensSinceScale = 10 < 32)
	// and must NOT rapidly oscillate back to K=1.
	for i := 0; i < 10; i++ {
		tuner.RecordToken(false)
		if tuner.CurrentK() != 2 {
			t.Fatalf("depth oscillated prematurely to K=%d during cooldown at token %d", tuner.CurrentK(), i+1)
		}
		if !tuner.InCooldown() {
			t.Fatalf("expected InCooldown() == true during token %d", i+1)
		}
	}

	// Record remaining 22 tokens as accepted (22/32 accepted in new window = 0.6875 -> in deadband)
	for i := 0; i < 22; i++ {
		tuner.RecordToken(true)
	}

	// Cooldown elapsed (32 tokens observed), rate is 22/32 = 0.6875 (deadband)
	// Depth should hold steady at K=2
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected K=2 after deadband resolution, got %d", tuner.CurrentK())
	}
}

// TestAppleSiliconMTPDeadbandStability verifies stability when acceptance rates
// fluctuate within the deadband region (0.60, 0.82).
func TestAppleSiliconMTPDeadbandStability(t *testing.T) {
	tuner := NewAppleSiliconMTPDepthTunerWithDepth(2)
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected initial K=2, got %d", tuner.CurrentK())
	}

	// Feed 5 consecutive windows in deadband (75% acceptance = 24/32)
	for w := 0; w < 5; w++ {
		tuner.RecordStep(32, 24)
		if tuner.CurrentK() != 2 {
			t.Fatalf("window %d: expected depth K=2 in deadband, got %d", w+1, tuner.CurrentK())
		}
	}
	if tuner.ScaleUpCount() != 0 || tuner.ScaleDownCount() != 0 {
		t.Fatalf("expected 0 scale events in deadband, got up=%d down=%d",
			tuner.ScaleUpCount(), tuner.ScaleDownCount())
	}
}

// TestAppleSiliconMTPDepthTunerTokenByToken verifies smooth step-by-step token recording.
func TestAppleSiliconMTPDepthTunerTokenByToken(t *testing.T) {
	tuner := NewAppleSiliconMTPDepthTuner()

	// Record 32 accepted tokens one by one
	for i := 0; i < 32; i++ {
		tuner.RecordToken(true)
	}
	if tuner.CurrentK() != 2 {
		t.Fatalf("expected K=2 after 32 token-by-token accepts, got %d", tuner.CurrentK())
	}

	// Record 32 accepted tokens one by one -> K=3
	for i := 0; i < 32; i++ {
		tuner.RecordToken(true)
	}
	if tuner.CurrentK() != 3 {
		t.Fatalf("expected K=3, got %d", tuner.CurrentK())
	}
}

// TestAppleSiliconMTPEdgeCasesAndDefaults verifies zero-value struct behavior,
// manual overrides, resets, and parameter bounds.
func TestAppleSiliconMTPEdgeCasesAndDefaults(t *testing.T) {
	// Zero-value struct should safely lazy-initialize defaults
	var zeroTuner AppleSiliconMTPDepthTuner
	if zeroTuner.CurrentK() != 1 {
		t.Fatalf("expected zero-value tuner CurrentK()=1, got %d", zeroTuner.CurrentK())
	}
	if zeroTuner.RollingAcceptanceRate() != 1.0 {
		t.Fatalf("expected empty rolling rate 1.0, got %f", zeroTuner.RollingAcceptanceRate())
	}

	zeroTuner.RecordStep(32, 32)
	if zeroTuner.CurrentK() != 2 {
		t.Fatalf("expected K=2 after zero-value step, got %d", zeroTuner.CurrentK())
	}

	// Manual override clamped to [MinK, MaxK]
	zeroTuner.SetK(10)
	if zeroTuner.CurrentK() != 4 {
		t.Fatalf("expected SetK(10) clamped to 4, got %d", zeroTuner.CurrentK())
	}
	zeroTuner.SetK(-5)
	if zeroTuner.CurrentK() != 1 {
		t.Fatalf("expected SetK(-5) clamped to 1, got %d", zeroTuner.CurrentK())
	}

	// Reset returns to initial K
	zeroTuner.SetK(3)
	zeroTuner.Reset()
	if zeroTuner.CurrentK() != 1 {
		t.Fatalf("expected Reset() to restore K=1, got %d", zeroTuner.CurrentK())
	}
	if zeroTuner.WindowCount() != 0 {
		t.Fatalf("expected WindowCount()=0 after reset, got %d", zeroTuner.WindowCount())
	}

	// Negative and overflowing inputs in RecordStep
	k := zeroTuner.RecordStep(-10, 5)
	if k != 1 {
		t.Fatalf("expected negative proposed to be no-op, got K=%d", k)
	}
	zeroTuner.RecordStep(10, 50) // accepted clamped to proposed (10)
	if zeroTuner.TotalAccepted() != 10 {
		t.Fatalf("expected clamped accepted=10, got %d", zeroTuner.TotalAccepted())
	}

	// ObserveStep report check
	report := zeroTuner.ObserveStep(4, 4)
	if report.Proposed != 4 || report.Accepted != 4 {
		t.Fatalf("unexpected report: %+v", report)
	}
}

// TestAppleSiliconMTPConcurrency verifies thread safety under concurrent access.
func TestAppleSiliconMTPConcurrency(t *testing.T) {
	tuner := NewAppleSiliconMTPDepthTuner()
	var wg sync.WaitGroup

	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for step := 0; step < 50; step++ {
				if (id+step)%2 == 0 {
					tuner.RecordStep(4, 4)
				} else {
					tuner.RecordStep(4, 2)
				}
				_ = tuner.CurrentK()
				_ = tuner.RollingAcceptanceRate()
			}
		}(g)
	}

	wg.Wait()
	k := tuner.CurrentK()
	if k < 1 || k > 4 {
		t.Fatalf("K out of bounds [1, 4] after concurrent runs: %d", k)
	}
}
