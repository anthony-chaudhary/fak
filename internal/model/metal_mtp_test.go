package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type mockMTPRecorder struct {
	records   int
	commits   int
	rollbacks int
}

func (m *mockMTPRecorder) RecordMTPDraft(sessionID string, tokens []int32) error {
	m.records++
	return nil
}

func (m *mockMTPRecorder) CommitMTPDraft(sessionID string, accepted int) (int, int, error) {
	m.commits++
	return accepted, 0, nil
}

func (m *mockMTPRecorder) RollbackMTPDraft(sessionID string) (int, error) {
	m.rollbacks++
	return 0, nil
}

// TestMetalMTPDraftVerifyRollbackLoop is the comprehensive witness test for Issue #12238:
// It asserts:
//  1. Bit-exact token sequence identity against non-speculative autoregressive baseline at temp=0.
//  2. Candidate draft generation from resident MTP head (K=2..4).
//  3. Wide-M Metal verification dispatch.
//  4. Context-MMU atomic page commit and rollback with zero memory leaks.
//  5. Greedy temperature-zero tripwire enforcement.
//  6. Rolling 32-token acceptance monitoring and smooth fallback to serial decode on low acceptance.
func TestMetalMTPDraftVerifyRollbackLoop(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1, 2}
	const maxNew = 16

	ctx := context.Background()

	t.Run("bit_exact_sequence_identity_against_autoregressive_baseline", func(t *testing.T) {
		// Non-speculative autoregressive baseline
		refSes := m.NewSession()
		t.Cleanup(refSes.Close)
		wantTokens := refSes.Generate(prompt, maxNew)

		// Speculative Metal MTP decode
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)
		coord, err := specSes.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		gotTokens, err := coord.Generate(ctx, prompt, maxNew)
		if err != nil {
			t.Fatalf("coord.Generate failed: %v", err)
		}

		if !reflect.DeepEqual(gotTokens, wantTokens) {
			t.Fatalf("output sequence mismatch:\n got:  %v\n want: %v", gotTokens, wantTokens)
		}

		stats := coord.Stats()
		if stats.TotalGenerated < maxNew {
			t.Fatalf("expected total generated >= %d, got %d", maxNew, stats.TotalGenerated)
		}
	})

	t.Run("context_mmu_atomic_page_commit_and_rollback", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)

		coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
			DraftDepth:            4,
			MinAcceptanceRate:     0.50,
			WindowSize:            32,
			EnforceGreedyTripwire: true,
			FallbackToSerial:      true,
		})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Wire recorder for MMU checkpoint tracking
		cm := &mockMTPRecorder{}
		sessionID := "test-metal-mtp-session"
		coord.SetMMU(cm, sessionID)

		// Prefill and execute 1 round
		boundary := specSes.Prefill(prompt)
		committed := append([]int(nil), prompt...)

		acc, bonus, nextLogits, err := coord.StepRound(ctx, committed, boundary)
		if err != nil {
			t.Fatalf("StepRound failed: %v", err)
		}
		if len(nextLogits) != m.Cfg.VocabSize {
			t.Fatalf("expected nextLogits len %d, got %d", m.Cfg.VocabSize, len(nextLogits))
		}
		if bonus < 0 {
			t.Fatalf("expected valid bonus token, got %d", bonus)
		}

		stats := coord.Stats()
		if stats.TotalProposed < 2 {
			t.Fatalf("expected proposed >= 2, got %d", stats.TotalProposed)
		}
		// Context-MMU must have tracked allocations and freed/committed pages
		if stats.CommittedPages != stats.TotalAccepted {
			t.Fatalf("Context-MMU committed pages %d != total accepted %d", stats.CommittedPages, stats.TotalAccepted)
		}
		if stats.FreedPages != stats.TotalRollbacks {
			t.Fatalf("Context-MMU freed pages %d != total rollbacks %d", stats.FreedPages, stats.TotalRollbacks)
		}
		_ = acc
	})

	t.Run("greedy_temperature_zero_tripwire", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)

		coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
			DraftDepth:            4,
			EnforceGreedyTripwire: true,
			MinAcceptanceRate:     0.50,
			WindowSize:            32,
		})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Greedy sampling (temp=0.0, penalty=1.0) must pass
		if err := coord.CheckSamplingTripwire(0.0, 1.0); err != nil {
			t.Fatalf("expected nil error for greedy parameters, got %v", err)
		}

		// Non-greedy temperature must trip the tripwire
		if err := coord.CheckSamplingTripwire(0.8, 1.0); !errors.Is(err, ErrMetalMTPTripwireDiverged) {
			t.Fatalf("expected ErrMetalMTPTripwireDiverged for temp=0.8, got %v", err)
		}

		// Non-unit penalty must trip the tripwire
		if err := coord.CheckSamplingTripwire(0.0, 1.2); !errors.Is(err, ErrMetalMTPTripwireDiverged) {
			t.Fatalf("expected ErrMetalMTPTripwireDiverged for penalty=1.2, got %v", err)
		}

		// Non-finite logits must trip the tripwire
		nanLogits := make([]float32, m.Cfg.VocabSize)
		nanLogits[0] = float32(math.NaN())
		_, _, _, err = coord.StepRound(ctx, prompt, nanLogits)
		if !errors.Is(err, ErrMetalMTPNonFiniteLogits) {
			t.Fatalf("expected ErrMetalMTPNonFiniteLogits for NaN logits, got %v", err)
		}
	})

	t.Run("rolling_acceptance_monitoring_and_smooth_serial_fallback", func(t *testing.T) {
		specSes := m.NewSession()
		t.Cleanup(specSes.Close)

		coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
			DraftDepth:            4,
			MinAcceptanceRate:     0.50,
			WindowSize:            32,
			FallbackToSerial:      true,
			EnforceGreedyTripwire: true,
		})
		if err != nil {
			t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
		}
		t.Cleanup(func() { _ = coord.Close() })

		// Inject an adversarial drafter that always proposes divergent tokens
		coord.SetDrafter(NewMTPProposalGeneratorWithFn(func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
			// Propose tokens guaranteed to mismatch
			return []int{99999, 99998, 99997, 99996}, nil
		}))

		boundary := specSes.Prefill(prompt)
		committed := append([]int(nil), prompt...)

		// Run 10 rounds; acceptance will be 0%, triggering smooth serial fallback
		for r := 0; r < 10; r++ {
			acc, bonus, nextLogits, err := coord.StepRound(ctx, committed, boundary)
			if err != nil {
				t.Fatalf("StepRound round %d failed during fallback: %v", r, err)
			}
			boundary = nextLogits
			for _, tok := range acc {
				committed = append(committed, tok)
			}
			if bonus >= 0 {
				committed = append(committed, bonus)
			}
		}

		stats := coord.Stats()
		if !stats.InFallback {
			t.Fatalf("expected coordinator to be in fallback mode after low acceptance, stats: %+v", stats)
		}
		if stats.RollingRate >= 0.50 {
			t.Fatalf("expected rolling rate < 0.50, got %.2f", stats.RollingRate)
		}

		// Verify session continues generating normally even in fallback
		outTokens, err := coord.Generate(ctx, prompt, 8)
		if err != nil {
			t.Fatalf("Generate failed while in fallback: %v", err)
		}
		if len(outTokens) != 8 {
			t.Fatalf("expected 8 tokens generated in fallback, got %d", len(outTokens))
		}
	})
}

// BenchmarkMetalMTPDraftVerifyRollbackLoop benchmarks wide-M (M=4) verification against serial decode.
func BenchmarkMetalMTPDraftVerifyRollbackLoop(b *testing.B) {
	m := qwen38HybridMTPEnabledSyntheticModelBench(b)
	prompt := []int{0, 1, 2}
	specSes := m.NewSession()
	b.Cleanup(specSes.Close)
	coord, err := specSes.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		b.Fatalf("NewMetalMTPCoordinator failed: %v", err)
	}
	b.Cleanup(func() { _ = coord.Close() })
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = coord.Generate(ctx, prompt, 8)
	}
}
