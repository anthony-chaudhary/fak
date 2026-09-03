//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestQwen35ResidentMetalDecoderCPUReferenceLogitParity(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)

	probeTokens := []int{27, 41, 73, 83, 98}
	for i, tok := range probeTokens {
		s := m.NewSession()
		s.Q4K = true
		s.MetalQ4K = true

		decoder, err := NewQwen35ResidentMetalDecoder(s)
		if err != nil {
			t.Fatalf("NewQwen35ResidentMetalDecoder failed: %v", err)
		}

		if decoder.IsFallback() {
			t.Fatalf("expected resident decoder active, got fallback: %s", decoder.FallbackReason())
		}

		verdict, err := decoder.VerifyCPUReference(tok, 0)
		decoder.Close()
		s.Close()

		if err != nil {
			t.Fatalf("probe step %d (token %d) failed: %v", i, tok, err)
		}

		t.Logf("probe step %d (token %d): cos=%g maxDiff=%g greedyMatch=%v (want %d, got %d)",
			i, tok, verdict.CosineSimilarity, verdict.MaxAbsDiff, verdict.GreedyMatch, verdict.ExpectedToken, verdict.ActualToken)

		if !verdict.Passed {
			t.Fatalf("probe step %d (token %d) failed parity: cosine=%g (want >=0.9999), maxDiff=%g, greedyMatch=%v (want %d, got %d)",
				i, tok, verdict.CosineSimilarity, verdict.MaxAbsDiff, verdict.GreedyMatch, verdict.ExpectedToken, verdict.ActualToken)
		}
	}
}

func TestQwen35ResidentMetalDecoderGreedyTokenMatchSequentialSteps(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)

	// CPU reference session
	cpu := m.NewSession()
	cpu.Q4K = true
	defer cpu.Close()

	// Metal resident session & decoder
	metalSession := m.NewSession()
	metalSession.Q4K = true
	metalSession.MetalQ4K = true
	defer metalSession.Close()

	decoder, err := NewQwen35ResidentMetalDecoder(metalSession)
	if err != nil {
		t.Fatalf("NewQwen35ResidentMetalDecoder failed: %v", err)
	}
	defer decoder.Close()

	if decoder.IsFallback() {
		t.Fatalf("expected resident decoder active, got fallback: %s", decoder.FallbackReason())
	}

	currToken := 27
	for step := 0; step < 6; step++ {
		pos := metalSession.Cache.Len()
		if pos != cpu.Cache.Len() {
			t.Fatalf("step %d cache length mismatch: metal=%d vs cpu=%d", step, pos, cpu.Cache.Len())
		}

		cpuLogits := cpu.token(currToken, pos)
		metalLogits, profile, stepErr := decoder.DecodeToken(currToken, pos)
		if stepErr != nil {
			t.Fatalf("step %d decode failed: %v", step, stepErr)
		}

		if profile.Fallback {
			t.Fatalf("step %d fell back: %s", step, profile.FallbackReason)
		}

		cos, maxDiff, err := CosineSimilarityAndMaxAbs(cpuLogits, metalLogits)
		if err != nil {
			t.Fatalf("step %d comparison failed: %v", step, err)
		}
		t.Logf("step %d (tok %d, pos %d): cos=%g, maxDiff=%g, cpuNext=%d, metalNext=%d",
			step, currToken, pos, cos, maxDiff, greedyArgmax(cpuLogits), greedyArgmax(metalLogits))

		cpuGreedy := greedyArgmax(cpuLogits)
		metalGreedy := greedyArgmax(metalLogits)
		if cpuGreedy != metalGreedy {
			t.Fatalf("step %d greedy token mismatch: want %d, got %d", step, cpuGreedy, metalGreedy)
		}

		currToken = metalGreedy
	}
}

func TestQwen35ResidentMetalDecoderStageProfileAccountingAndSyncAmortization(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)

	s := m.NewSession()
	s.Q4K = true
	s.MetalQ4K = true
	defer s.Close()

	decoder, err := NewQwen35ResidentMetalDecoder(s)
	if err != nil {
		t.Fatalf("NewQwen35ResidentMetalDecoder failed: %v", err)
	}
	defer decoder.Close()

	logits, profile, err := decoder.DecodeToken(25, 0)
	if err != nil {
		t.Fatalf("DecodeToken failed: %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits length=%d, want %d", len(logits), cfg.VocabSize)
	}

	if profile.Fallback {
		t.Fatalf("unexpected fallback: %s", profile.FallbackReason)
	}

	// Verify all 5 stages are attributed and positive
	if profile.HostCopy <= 0 || profile.HostCopyMS <= 0 {
		t.Fatalf("stage attribution host copy must be positive: %v (%v ms)", profile.HostCopy, profile.HostCopyMS)
	}
	if profile.CommitToComplete <= 0 || profile.CommitToCompleteMS <= 0 {
		t.Fatalf("stage attribution commit-to-complete must be positive: %v (%v ms)", profile.CommitToComplete, profile.CommitToCompleteMS)
	}
	if profile.GPUCompute <= 0 || profile.GPUComputeMS <= 0 {
		t.Fatalf("stage attribution kernel execution must be positive: %v (%v ms)", profile.GPUCompute, profile.GPUComputeMS)
	}
	if profile.ResultReadback <= 0 || profile.ResultReadbackMS <= 0 {
		t.Fatalf("stage attribution result readback must be positive: %v (%v ms)", profile.ResultReadback, profile.ResultReadbackMS)
	}
	if profile.Total <= 0 || profile.TotalMS <= 0 {
		t.Fatalf("stage attribution total must be positive: %v (%v ms)", profile.Total, profile.TotalMS)
	}

	// Verify synchronization amortization: pays synchronization once per token instead of per layer
	if profile.Synchronizations != 1 {
		t.Fatalf("synchronizations paid=%d, want exactly 1 per token", profile.Synchronizations)
	}
	expectedAmortized := 1.0 / float64(cfg.NumLayers)
	if profile.AmortizedSyncPerLayer != expectedAmortized {
		t.Fatalf("amortized sync per layer=%g, want %g", profile.AmortizedSyncPerLayer, expectedAmortized)
	}
	if profile.NumLayers != cfg.NumLayers {
		t.Fatalf("num layers=%d, want %d", profile.NumLayers, cfg.NumLayers)
	}
}

func TestQwen35ResidentMetalDecoderUnsupportedGeometryFallback(t *testing.T) {
	// 1. Non-hybrid model configuration
	t.Run("NonHybridDecline", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.LayerTypes = nil // IsQwen35Hybrid() == false
		m := NewSynthetic(cfg)
		s := m.NewSession()
		defer s.Close()

		decoder, err := NewQwen35ResidentMetalDecoder(s)
		if err != nil {
			t.Fatalf("constructor must not error on unsupported geometry, got: %v", err)
		}
		defer decoder.Close()

		if !decoder.IsFallback() {
			t.Fatal("expected fallback active for non-hybrid model")
		}
		if decoder.FallbackReason() == "" {
			t.Fatal("expected non-empty fallback reason")
		}

		logits, profile, err := decoder.DecodeToken(5, 0)
		if err != nil {
			t.Fatalf("fallback forward failed: %v", err)
		}
		if len(logits) != cfg.VocabSize {
			t.Fatalf("fallback logits length=%d, want %d", len(logits), cfg.VocabSize)
		}
		if !profile.Fallback {
			t.Fatal("profile must indicate fallback")
		}
	})

	// 2. MoE architecture configuration
	t.Run("MoEDecline", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.NumExperts = 8
		cfg.NumExpertsPerTok = 2
		m := NewSynthetic(cfg)
		s := m.NewSession()
		defer s.Close()

		decoder, err := NewQwen35ResidentMetalDecoder(s)
		if err != nil {
			t.Fatalf("constructor must not error on MoE geometry, got: %v", err)
		}
		defer decoder.Close()

		if !decoder.IsFallback() {
			t.Fatal("expected fallback active for MoE model")
		}
	})

	// 3. Unaligned / unsupported HiddenSize
	t.Run("UnalignedHiddenDecline", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.HiddenSize = 250 // not a multiple of 32
		m := NewSynthetic(cfg)
		s := m.NewSession()
		defer s.Close()

		decoder, err := NewQwen35ResidentMetalDecoder(s)
		if err != nil {
			t.Fatalf("constructor must not error, got: %v", err)
		}
		defer decoder.Close()

		if !decoder.IsFallback() {
			t.Fatal("expected fallback active for unaligned HiddenSize")
		}
	})

	// 4. Invalid linear attention dimensions
	t.Run("InvalidLinearDimsDecline", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.LinearKeyHeadDim = 0
		m := NewSynthetic(cfg)
		s := m.NewSession()
		defer s.Close()

		decoder, err := NewQwen35ResidentMetalDecoder(s)
		if err != nil {
			t.Fatalf("constructor must not error, got: %v", err)
		}
		defer decoder.Close()

		if !decoder.IsFallback() {
			t.Fatal("expected fallback active for zero LinearKeyHeadDim")
		}
	})
}
