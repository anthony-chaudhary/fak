package mtptune

import (
	"sync"
	"testing"
)

func TestAdaptiveDepthTunerScaling(t *testing.T) {
	cfg := AppleSiliconMetalAdaptiveConfig()
	cfg.InitialK = 1
	cfg.MinTokensBeforeScale = 8
	tuner := NewAdaptiveDepthTuner(cfg)

	if tuner.CurrentK() != 1 {
		t.Fatalf("expected initial K=1, got %d", tuner.CurrentK())
	}

	// 1. Verify dynamic upscaling on high acceptance (e.g. repetitive code / JSON workloads)
	// We propose at current K and simulate high acceptance (all or almost all accepted).
	// Over multiple steps, as high acceptance is observed in the rolling window (32 tokens),
	// K should scale dynamically: 1 -> 2 -> 3 -> 4.
	for step := 0; step < 25; step++ {
		k := tuner.CurrentK()
		tuner.RecordStep(k, k) // 100% acceptance at current depth
	}

	if tuner.CurrentK() != 4 {
		t.Fatalf("expected tuner to upscale to K=4 on high acceptance, got K=%d (rolling rate: %.2f)",
			tuner.CurrentK(), tuner.RollingAcceptanceRate())
	}

	stats := tuner.Stats()
	if stats.UpscaleCount == 0 {
		t.Fatal("expected at least one upscale event")
	}
	if stats.RollingRate < 0.90 {
		t.Fatalf("expected high rolling acceptance rate >= 0.90, got %.2f", stats.RollingRate)
	}

	// Verify effective throughput on Apple Silicon Metal reaches 22+ tok/s at K=4
	tps := tuner.EffectiveThroughput()
	if tps < 22.0 {
		t.Fatalf("expected effective decode throughput >= 22.0 tok/s on Apple Silicon Metal, got %.2f tok/s", tps)
	}
	t.Logf("Upscaling successful: reached K=%d with %.2f tok/s decode throughput (acceptance: %.1f%%)",
		tuner.CurrentK(), tps, stats.RollingRate*100)

	// 2. Verify graceful downscaling on entropy shifts / low acceptance (e.g. natural text, math)
	// Acceptance drops sharply (e.g. 0 accepted out of proposed).
	// K should downscale gracefully step-by-step: 4 -> 3 -> 2 -> 1, preventing wasted base forward passes.
	observedDepths := []int{tuner.CurrentK()}
	for step := 0; step < 40; step++ {
		k := tuner.CurrentK()
		tuner.RecordStep(0, k) // 0% acceptance
		if len(observedDepths) == 0 || observedDepths[len(observedDepths)-1] != tuner.CurrentK() {
			observedDepths = append(observedDepths, tuner.CurrentK())
		}
	}

	if tuner.CurrentK() != 1 {
		t.Fatalf("expected tuner to downscale to K=1 on low acceptance, got K=%d (rolling rate: %.2f)",
			tuner.CurrentK(), tuner.RollingAcceptanceRate())
	}

	downStats := tuner.Stats()
	if downStats.DownscaleCount == 0 {
		t.Fatal("expected downscale events on low acceptance")
	}
	if downStats.TotalRollbacks == 0 {
		t.Fatal("expected rollbacks recorded during low acceptance")
	}

	// Verify graceful progression (visited intermediate depths without abrupt jumping)
	t.Logf("Downscale progression through depths: %v", observedDepths)
	hasIntermediate := false
	for _, d := range observedDepths {
		if d == 2 || d == 3 {
			hasIntermediate = true
			break
		}
	}
	if !hasIntermediate {
		t.Fatalf("expected graceful downscaling through intermediate depths (2 or 3), got sequence: %v", observedDepths)
	}

	// Verify that at low acceptance (e.g. alpha=0.20), K=1 avoids wasted draft passes
	// and yields higher net throughput than remaining stuck at K=4.
	tpsLowK1 := EstimateThroughput(1, 0.20, cfg.Hardware)
	tpsLowK4 := EstimateThroughput(4, 0.20, cfg.Hardware)
	if tpsLowK1 <= tpsLowK4 {
		t.Fatalf("expected K=1 throughput (%.2f) > K=4 throughput (%.2f) during low-acceptance entropy shifts",
			tpsLowK1, tpsLowK4)
	}
	t.Logf("Wasted forward pass prevention verified: K=1 yields %.2f tok/s vs K=4 %.2f tok/s at 20%% acceptance",
		tpsLowK1, tpsLowK4)

	// 3. Verify re-upscaling recovery when high acceptance returns
	for step := 0; step < 30; step++ {
		k := tuner.CurrentK()
		tuner.RecordStep(k, k)
	}
	if tuner.CurrentK() <= 1 {
		t.Fatalf("expected tuner to re-upscale after recovery, got K=%d", tuner.CurrentK())
	}

	// 4. Verify hysteresis prevents thrashing/flapping in the deadband
	tuner.Reset()
	// Prime window with ~70% acceptance
	for i := 0; i < 32; i++ {
		tuner.RecordOutcome(i%10 < 7) // 70% acceptance
	}
	kStable := tuner.CurrentK()
	if kStable < 2 || kStable > 3 {
		t.Fatalf("expected stable depth in deadband [2, 3], got K=%d", kStable)
	}

	for i := 0; i < 30; i++ {
		tuner.RecordStep(7, 10) // continue 70% acceptance
		if tuner.CurrentK() != kStable {
			t.Fatalf("hysteresis flapping detected: depth changed from %d to %d under stable 70%% acceptance",
				kStable, tuner.CurrentK())
		}
	}
	t.Logf("Hysteresis verified: draft depth stayed stable at K=%d in deadband (acceptance: %.1f%%)",
		tuner.CurrentK(), tuner.RollingAcceptanceRate()*100)
}

func TestRollingWindowRingBuffer(t *testing.T) {
	w := newRollingWindow(4)

	if w.count() != 0 || w.acceptedCount() != 0 {
		t.Fatalf("expected empty window, got count=%d, accepted=%d", w.count(), w.acceptedCount())
	}
	if w.rate() != 1.0 {
		t.Fatalf("expected initial rate=1.0 for empty window, got %f", w.rate())
	}

	w.push(true)
	w.push(true)
	w.push(false)
	w.push(true)

	if w.count() != 4 || w.acceptedCount() != 3 {
		t.Fatalf("expected count=4, accepted=3, got count=%d, accepted=%d", w.count(), w.acceptedCount())
	}
	if w.rate() != 0.75 {
		t.Fatalf("expected rate=0.75, got %f", w.rate())
	}

	// Evict first 'true' by pushing 'false'
	w.push(false)
	if w.count() != 4 || w.acceptedCount() != 2 {
		t.Fatalf("expected count=4, accepted=2 after eviction, got count=%d, accepted=%d", w.count(), w.acceptedCount())
	}
	if w.rate() != 0.50 {
		t.Fatalf("expected rate=0.50, got %f", w.rate())
	}

	w.reset()
	if w.count() != 0 || w.acceptedCount() != 0 {
		t.Fatalf("expected reset window, got count=%d, accepted=%d", w.count(), w.acceptedCount())
	}
}

func TestAdaptiveDepthConfigValidation(t *testing.T) {
	valid := DefaultAdaptiveDepthConfig()
	if err := valid.Validate(); err != nil {
		t.Fatalf("expected DefaultAdaptiveDepthConfig to be valid, got: %v", err)
	}

	appleCfg := AppleSiliconMetalAdaptiveConfig()
	if err := appleCfg.Validate(); err != nil {
		t.Fatalf("expected AppleSiliconMetalAdaptiveConfig to be valid, got: %v", err)
	}

	invalidK := valid
	invalidK.MinK = 4
	invalidK.MaxK = 2
	if err := invalidK.Validate(); err == nil {
		t.Fatal("expected error for MinK > MaxK")
	}

	invalidWindow := valid
	invalidWindow.WindowSize = 0
	if err := invalidWindow.Validate(); err == nil {
		t.Fatal("expected error for WindowSize <= 0")
	}

	invalidMinTokens := valid
	invalidMinTokens.MinTokensBeforeScale = 64
	if err := invalidMinTokens.Validate(); err == nil {
		t.Fatal("expected error for MinTokensBeforeScale > WindowSize")
	}

	invalidThresh := valid
	invalidThresh.UpscaleThreshold = 0.50
	invalidThresh.DownscaleThreshold = 0.70
	if err := invalidThresh.Validate(); err == nil {
		t.Fatal("expected error for UpscaleThreshold <= DownscaleThreshold")
	}
}

func TestAdaptiveDepthTunerReset(t *testing.T) {
	tuner := NewAdaptiveDepthTuner()
	tuner.RecordStep(10, 10)
	tuner.RecordStep(0, 5)

	stats := tuner.Stats()
	if stats.TotalProposed == 0 || stats.TotalAccepted == 0 {
		t.Fatal("expected non-zero stats before reset")
	}

	tuner.Reset()
	resetStats := tuner.Stats()
	if resetStats.TotalProposed != 0 || resetStats.TotalAccepted != 0 || resetStats.WindowTokens != 0 {
		t.Fatalf("expected zeroed stats after reset, got %+v", resetStats)
	}
	if tuner.CurrentK() != tuner.Config().InitialK {
		t.Fatalf("expected CurrentK to be InitialK (%d), got %d", tuner.Config().InitialK, tuner.CurrentK())
	}
}

func TestEstimateThroughputMetalScaling(t *testing.T) {
	hw := AppleSiliconMetalSweepConfig()

	// High acceptance (0.90) on Metal unified memory
	tpsK1 := EstimateThroughput(1, 0.90, hw)
	tpsK2 := EstimateThroughput(2, 0.90, hw)
	tpsK3 := EstimateThroughput(3, 0.90, hw)
	tpsK4 := EstimateThroughput(4, 0.90, hw)

	if tpsK4 <= tpsK1 {
		t.Fatalf("expected K=4 throughput (%.2f) > K=1 throughput (%.2f) at 90%% acceptance", tpsK4, tpsK1)
	}
	if tpsK4 < 22.0 {
		t.Fatalf("expected K=4 throughput >= 22.0 tok/s on Apple Silicon Metal at 90%% acceptance, got %.2f", tpsK4)
	}
	if tpsK3 <= tpsK2 || tpsK2 <= tpsK1 {
		t.Fatalf("expected monotonic speedup with depth under high acceptance: K1=%.2f, K2=%.2f, K3=%.2f, K4=%.2f",
			tpsK1, tpsK2, tpsK3, tpsK4)
	}

	// Empty config fallback to Apple Silicon Metal
	fallbackTPS := EstimateThroughput(4, 0.90, SweepConfig{})
	if fallbackTPS <= 0 {
		t.Fatalf("expected positive throughput on empty SweepConfig fallback, got %.2f", fallbackTPS)
	}
}

func TestAdaptiveDepthTunerConcurrency(t *testing.T) {
	tuner := NewAdaptiveDepthTuner()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for step := 0; step < 50; step++ {
				if (workerID+step)%2 == 0 {
					tuner.RecordStep(2, 2)
				} else {
					tuner.RecordStep(0, 2)
				}
				_ = tuner.CurrentK()
				_ = tuner.RollingAcceptanceRate()
				_ = tuner.LifetimeAcceptanceRate()
				_ = tuner.EffectiveThroughput()
				_ = tuner.Stats()
			}
		}(i)
	}

	wg.Wait()

	finalStats := tuner.Stats()
	if finalStats.TotalProposed != 8*50*2 {
		t.Fatalf("expected %d total proposed tokens, got %d", 8*50*2, finalStats.TotalProposed)
	}
	summary := FormatAdaptiveStats(finalStats)
	if len(summary) == 0 {
		t.Fatal("expected non-empty formatted stats summary")
	}
}
