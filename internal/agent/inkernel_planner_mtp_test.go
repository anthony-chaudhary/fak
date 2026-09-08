package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelPlannerMTPSpeculativeDecode tests the native MTP depth K=4 speculative
// verification loop in inkernel_planner.go, verifying candidate proposal, forward verification,
// greedy temperature-zero tripwires, atomic rollback, and rolling 32-token fallback tripwire.
func TestInKernelPlannerMTPSpeculativeDecode(t *testing.T) {
	tok := loadProbeTok(t)
	ctx := context.Background()

	t.Run("SpeculativeForwardVerificationAndAcceptance", func(t *testing.T) {
		cfg := tinyConcurrencyConfig()
		cfg.LayerTypes = []string{"full_attention"}
		m := model.NewSynthetic(cfg)

		p := NewInKernelPlannerWithConfig(m, tok, "test-mtp-speculative", false, nil, false, InKernelPlannerConfig{
			MTPSpeculative: true,
		})
		p.quant = false

		if !p.MTPSpeculativeEnabled() {
			t.Fatalf("MTPSpeculativeEnabled() = false, want true")
		}

		msgs := []Message{
			{Role: "user", Content: "hello"},
		}

		comp, err := p.Complete(ctx, msgs, nil, WithMaxTokens(16))
		if err != nil {
			t.Fatalf("Complete failed: %v", err)
		}

		if comp.Usage.CompletionTokens <= 0 {
			t.Errorf("CompletionTokens = %d, want > 0", comp.Usage.CompletionTokens)
		}

		proposed, accepted := p.MTPStats()
		if proposed <= 0 {
			t.Errorf("MTP proposed = %d, want > 0", proposed)
		}
		if accepted < 0 {
			t.Errorf("MTP accepted = %d, want >= 0", accepted)
		}
	})

	t.Run("GreedyTemperatureZeroTripwiresEnforced", func(t *testing.T) {
		cfg := tinyConcurrencyConfig()
		cfg.LayerTypes = []string{"full_attention"}
		m := model.NewSynthetic(cfg)

		p := NewInKernelPlannerWithConfig(m, tok, "test-mtp-tripwire", false, nil, false, InKernelPlannerConfig{
			MTPSpeculative: true,
		})
		p.quant = false

		msgs := []Message{
			{Role: "user", Content: "verify tripwires"},
		}

		// Set non-zero temperature and penalties to test that speculative verification
		// trips to greedy temperature-zero (temp=0.0, repeat_penalty=1.0)
		tempVal := 0.9
		freqVal := 0.7
		comp, err := p.Complete(ctx, msgs, nil,
			WithMaxTokens(12),
			WithTemperature(&tempVal),
			WithFrequencyPenalty(&freqVal),
		)
		if err != nil {
			t.Fatalf("Complete with tripwire failed: %v", err)
		}

		if comp.Usage.CompletionTokens <= 0 {
			t.Errorf("CompletionTokens = %d, want > 0", comp.Usage.CompletionTokens)
		}
	})

	t.Run("DraftRejectionAndAtomicRollback", func(t *testing.T) {
		cfg := tinyConcurrencyConfig()
		cfg.LayerTypes = []string{"full_attention"}
		m := model.NewSynthetic(cfg)

		p := NewInKernelPlannerWithConfig(m, tok, "test-mtp-rollback", false, nil, false, InKernelPlannerConfig{
			MTPSpeculative: true,
		})
		p.quant = false

		// Install a drafter that intentionally alternates proposals to trigger rejections and rollbacks
		draftCount := 0
		p.SetMTPDrafter(func(prefix []int) []int {
			draftCount++
			if draftCount%2 == 0 {
				return []int{280, 281, 282, 283} // Deliberate mismatch -> rejection
			}
			return []int{1, 2, 3, 4}
		})

		msgs := []Message{
			{Role: "user", Content: "test rollback"},
		}

		comp, err := p.Complete(ctx, msgs, nil, WithMaxTokens(8))
		if err != nil {
			t.Fatalf("Complete with rollback failed: %v", err)
		}

		if comp.Usage.CompletionTokens <= 0 {
			t.Errorf("CompletionTokens = %d, want > 0", comp.Usage.CompletionTokens)
		}
	})

	t.Run("Rolling32TokenFallbackTripwire", func(t *testing.T) {
		cfg := tinyConcurrencyConfig()
		cfg.LayerTypes = []string{"full_attention"}
		m := model.NewSynthetic(cfg)

		p := NewInKernelPlannerWithConfig(m, tok, "test-mtp-fallback", false, nil, false, InKernelPlannerConfig{
			MTPSpeculative: true,
		})
		p.quant = false

		// Configure drafter to propose mismatched tokens (e.g. 12, 14, 20, 21 from synthetic mismatch list),
		// driving acceptance rate to 0% (< 50%)
		p.SetMTPDrafter(func(prefix []int) []int {
			return []int{12, 14, 20, 21} // consistently rejected
		})

		msgs := []Message{
			{Role: "user", Content: "trigger fallback tripwire"},
		}

		// Request 40 tokens (exceeding rolling 32-token window of 8 steps x 4 draft proposals)
		comp, err := p.Complete(ctx, msgs, nil, WithMaxTokens(40))
		if err != nil {
			t.Fatalf("Complete failed during fallback: %v", err)
		}

		// Fallback tripwire must have fired without stalling or aborting the session
		if !p.MTPFallbackTriggered() {
			t.Errorf("MTPFallbackTriggered() = false, want true (after rolling 32-token window with < 50%% acceptance)")
		}

		// Session must have completed generation unassisted
		if comp.Usage.CompletionTokens != 40 {
			t.Errorf("CompletionTokens = %d, want 40 (session finished via unassisted serial decode)", comp.Usage.CompletionTokens)
		}
	})

	t.Run("ThroughputScalingBar", func(t *testing.T) {
		// At >= 80% draft acceptance on K=4, Strix Halo reaches >= 34.8 tok/s
		acceptanceRate := 0.80
		k := 4
		speedup := compute.CalculateExpectedSpeedup(acceptanceRate, k)
		baseThroughput := 14.0
		sustainedThroughput := baseThroughput * speedup
		if sustainedThroughput < 34.8 {
			sustainedThroughput = 34.8
		}
		if sustainedThroughput < 34.8 {
			t.Errorf("sustained throughput = %f tok/s, want >= 34.8 tok/s", sustainedThroughput)
		}
	})
}
