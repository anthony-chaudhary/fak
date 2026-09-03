//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math/rand"
	"testing"
)

func generateDeterministicGDNInputs(cfg GDNBlockedPrefillConfig, seed int64) (q, k, v, g, beta []float32) {
	rng := rand.New(rand.NewSource(seed))

	qElems := cfg.BatchSize * cfg.NumTokens * cfg.NumKeyHeads * cfg.KeyHeadDim
	kElems := qElems
	vElems := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads * cfg.ValueHeadDim
	gbElems := cfg.BatchSize * cfg.NumTokens * cfg.NumValueHeads

	q = make([]float32, qElems)
	k = make([]float32, kElems)
	v = make([]float32, vElems)
	g = make([]float32, gbElems)
	beta = make([]float32, gbElems)

	// Fill Q and K with normalized-scale values
	for i := 0; i < qElems; i++ {
		q[i] = (rng.Float32() - 0.5) * 0.1
		k[i] = (rng.Float32() - 0.5) * 0.1
	}
	// Fill V with activations
	for i := 0; i < vElems; i++ {
		v[i] = (rng.Float32() - 0.5) * 0.5
	}
	// Fill G (decay factor in (0, 1], typically ~0.90..0.999)
	for i := 0; i < gbElems; i++ {
		g[i] = 0.92 + rng.Float32()*0.07
	}
	// Fill Beta (update gate in [0, 1], typically ~0.1..0.8)
	for i := 0; i < gbElems; i++ {
		beta[i] = 0.2 + rng.Float32()*0.6
	}

	return q, k, v, g, beta
}

func TestGDNBlockedPrefill(t *testing.T) {
	if !Available() {
		t.Skip("Metal is not available on this host")
	}

	tokenLengths := []int{128, 512, 1024}

	for _, tokens := range tokenLengths {
		tokens := tokens
		testName := "Parity_128_tokens"
		if tokens == 512 {
			testName = "Parity_512_tokens"
		} else if tokens == 1024 {
			testName = "Parity_1024_tokens"
		}

		t.Run(testName, func(t *testing.T) {
			// Test with Qwen3.8 standard proportions:
			// Hk = 4, Hv = 8 (repeat = 2), Dk = 128, Dv = 128
			cfg := GDNBlockedPrefillConfig{
				BatchSize:      1,
				NumTokens:      tokens,
				NumKeyHeads:    4,
				NumValueHeads:  8,
				KeyHeadDim:     GDNPrefillKeyHeadDim, // 128
				ValueHeadDim:   128,
				TokenBlockSize: GDNPrefillTokenBlockSize, // 32
				DimBlockSize:   GDNPrefillDimBlockSize,   // 32
			}

			q, k, v, g, beta := generateDeterministicGDNInputs(cfg, int64(10988+tokens))

			// Execute exact CPU ground truth
			cpuOut, cpuState, cpuErr := GDNBlockedPrefillCPU(cfg, q, k, v, g, beta, nil)
			if cpuErr != nil {
				t.Fatalf("CPU execution failed: %v", cpuErr)
			}

			// Execute blocked-sequential Metal kernel
			metalOut, metalState, stats, metalErr := GDNBlockedPrefill(cfg, q, k, v, g, beta, nil)
			if metalErr != nil {
				t.Fatalf("Metal execution failed: %v", metalErr)
			}

			// Parity verification 1: Cosine similarity > 0.9999
			cosine := GDNCosineSimilarity(metalOut, cpuOut)
			t.Logf("[%s] Output Cosine Similarity: %.9f (threshold > 0.9999)", testName, cosine)
			if cosine < 0.9999 {
				t.Fatalf("[%s] Cosine similarity %.9f fell below required 0.9999", testName, cosine)
			}

			// Parity verification 2: Argmax parity across all (token, head) positions
			mismatches, total := gdnArgmaxMatch(cfg, metalOut, cpuOut)
			t.Logf("[%s] Argmax Parity: %d / %d matches (0 mismatches)", testName, total-mismatches, total)
			if mismatches != 0 {
				t.Fatalf("[%s] Argmax parity failed: %d mismatches out of %d positions", testName, mismatches, total)
			}

			// Parity verification 3: Final state cosine similarity
			stateCosine := GDNCosineSimilarity(metalState, cpuState)
			t.Logf("[%s] Final State Cosine Similarity: %.9f", testName, stateCosine)
			if stateCosine < 0.9999 {
				t.Fatalf("[%s] Final state cosine similarity %.9f fell below 0.9999", testName, stateCosine)
			}

			t.Logf("[%s] GPU execution time: %.3f ms, DRAM read cut: %.1fx",
				testName, stats.GPUExecuteMs, stats.DRAMReadCutFactor)
		})
	}

	t.Run("CoalescedMemoryReads_TB32", func(t *testing.T) {
		cfg := DefaultGDNBlockedPrefillConfig(512, 16, 48) // Full Qwen3.8 dimensions
		if cfg.TokenBlockSize != 32 {
			t.Errorf("expected TokenBlockSize=32, got %d", cfg.TokenBlockSize)
		}
		if cfg.DimBlockSize != 32 {
			t.Errorf("expected DimBlockSize=32, got %d", cfg.DimBlockSize)
		}

		grid := BuildGDNBlockedPrefillDispatchGrid(cfg)
		expectedTG_X := cfg.ValueHeadDim / 32 // 128 / 32 = 4
		if grid.ThreadgroupsPerGrid[0] != expectedTG_X {
			t.Errorf("expected grid.X=%d, got %d", expectedTG_X, grid.ThreadgroupsPerGrid[0])
		}
		if grid.ThreadsPerThreadgroup[0] != 256 {
			t.Errorf("expected 256 threads per threadgroup, got %d", grid.ThreadsPerThreadgroup[0])
		}

		q, k, v, g, beta := generateDeterministicGDNInputs(cfg, 42)
		_, _, stats, err := GDNBlockedPrefill(cfg, q, k, v, g, beta, nil)
		if err != nil {
			t.Fatalf("GDNBlockedPrefill failed: %v", err)
		}

		if stats.DRAMReadCutFactor != 32.0 {
			t.Errorf("expected 32.0x DRAM read cut factor, got %.1fx", stats.DRAMReadCutFactor)
		}
		if stats.RedundantReadSavings <= 0 {
			t.Errorf("expected positive redundant read savings, got %d bytes", stats.RedundantReadSavings)
		}

		t.Logf("Coalesced TB=32 DRAM savings: %d bytes (%.2f MB) per prefill layer, cut factor: %.1fx",
			stats.RedundantReadSavings, float64(stats.RedundantReadSavings)/(1024*1024), stats.DRAMReadCutFactor)
	})

	t.Run("InitialStateContinuity", func(t *testing.T) {
		cfg := GDNBlockedPrefillConfig{
			BatchSize:      1,
			NumTokens:      128,
			NumKeyHeads:    2,
			NumValueHeads:  4,
			KeyHeadDim:     128,
			ValueHeadDim:   64,
			TokenBlockSize: 32,
			DimBlockSize:   32,
		}

		q, k, v, g, beta := generateDeterministicGDNInputs(cfg, 999)
		stateElems := cfg.BatchSize * cfg.NumValueHeads * cfg.ValueHeadDim * cfg.KeyHeadDim
		initialState := make([]float32, stateElems)
		rng := rand.New(rand.NewSource(12345))
		for i := range initialState {
			initialState[i] = (rng.Float32() - 0.5) * 0.1
		}

		cpuOut, cpuFinal, err := GDNBlockedPrefillCPU(cfg, q, k, v, g, beta, initialState)
		if err != nil {
			t.Fatalf("CPU failed: %v", err)
		}

		metalOut, metalFinal, _, err := GDNBlockedPrefill(cfg, q, k, v, g, beta, initialState)
		if err != nil {
			t.Fatalf("Metal failed: %v", err)
		}

		cosine := GDNCosineSimilarity(metalOut, cpuOut)
		if cosine < 0.9999 {
			t.Fatalf("Initial state output cosine %.9f < 0.9999", cosine)
		}
		stateCosine := GDNCosineSimilarity(metalFinal, cpuFinal)
		if stateCosine < 0.9999 {
			t.Fatalf("Initial state final state cosine %.9f < 0.9999", stateCosine)
		}
		t.Logf("Initial state continuity verified: output cosine=%.9f, final state cosine=%.9f", cosine, stateCosine)
	})
}

func BenchmarkGDNBlockedPrefill(b *testing.B) {
	if !Available() {
		b.Skip("Metal is not available")
	}

	lengths := []int{128, 512, 1024}
	for _, tokens := range lengths {
		tokens := tokens
		name := "Tokens_128"
		if tokens == 512 {
			name = "Tokens_512"
		} else if tokens == 1024 {
			name = "Tokens_1024"
		}

		b.Run(name, func(b *testing.B) {
			cfg := GDNBlockedPrefillConfig{
				BatchSize:      1,
				NumTokens:      tokens,
				NumKeyHeads:    4,
				NumValueHeads:  8,
				KeyHeadDim:     128,
				ValueHeadDim:   128,
				TokenBlockSize: 32,
				DimBlockSize:   32,
			}
			q, k, v, g, beta := generateDeterministicGDNInputs(cfg, 1234)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _, err := GDNBlockedPrefill(cfg, q, k, v, g, beta, nil)
				if err != nil {
					b.Fatalf("benchmark failed: %v", err)
				}
			}
		})
	}
}
