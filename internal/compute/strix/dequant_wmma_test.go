package strix

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

func TestInRegisterDequantWMMA(t *testing.T) {
	cfg := DefaultDequantWMMAConfig()
	kernel, err := NewDequantWMMAKernel(cfg)
	if err != nil {
		t.Fatalf("failed to initialize DequantWMMAKernel: %v", err)
	}

	// 1. GFX1151 ISA and Zero-LDS Invariant Verification
	t.Run("GFX1151_ISA_ZeroLDS_Verification", func(t *testing.T) {
		inspection := kernel.InspectAssembly()

		if inspection.TargetArch != "gfx1151" {
			t.Errorf("expected TargetArch 'gfx1151', got %q", inspection.TargetArch)
		}
		if inspection.WaveSize != 32 {
			t.Errorf("expected WaveSize 32 (Wave32), got %d", inspection.WaveSize)
		}
		if !inspection.HasGlobalLoadDwordX4 {
			t.Errorf("expected assembly to include 128-bit 'global_load_dwordx4' vector loads")
		}
		if !inspection.HasWMMA {
			t.Errorf("expected assembly to include 'v_wmma_' instructions")
		}
		if !inspection.HasZeroLDS {
			t.Errorf("expected zero LDS allocations in GFX1151 kernel (.amdhsa_group_segment_fixed_size 0, no ds_write/read)")
		}
		if inspection.LDSAllocBytes != 0 {
			t.Errorf("expected 0 LDS allocation bytes, got %d", inspection.LDSAllocBytes)
		}
	})

	// 2. 128-Byte LPDDR5X Burst Coalescing Verification
	t.Run("Coalesced_128Byte_Burst_Alignment", func(t *testing.T) {
		// Test aligned address: offset 0x2000 (aligned to 128 bytes)
		weightSize := 4096 // 4096 bytes = 32 bursts
		alignedOffset := uintptr(0x2000)

		analysis, err := kernel.VerifyCoalescing(weightSize, alignedOffset)
		if err != nil {
			t.Fatalf("expected aligned verification to pass, got error: %v", err)
		}

		if !analysis.Aligned128Byte {
			t.Errorf("expected Aligned128Byte to be true")
		}
		if analysis.BytesPerThread != 16 {
			t.Errorf("expected 16 bytes per thread (dwordx4), got %d", analysis.BytesPerThread)
		}
		if analysis.BytesPerWavefront != 512 {
			t.Errorf("expected 512 bytes per Wave32 wavefront, got %d", analysis.BytesPerWavefront)
		}
		if analysis.BurstsPerWavefront != 4 {
			t.Errorf("expected exactly 4x 128-byte bursts per wavefront, got %d", analysis.BurstsPerWavefront)
		}
		if analysis.EfficiencyPercent != 100.0 {
			t.Errorf("expected 100.0%% efficiency, got %.2f", analysis.EfficiencyPercent)
		}
		if analysis.UncoalescedLoads != 0 {
			t.Errorf("expected 0 uncoalesced loads, got %d", analysis.UncoalescedLoads)
		}
		if !analysis.ZeroIntermediateLDS {
			t.Errorf("expected ZeroIntermediateLDS to be true")
		}

		// Test unaligned address: offset 0x2010 (aligned to 16 bytes, but not 128 bytes)
		unalignedOffset := uintptr(0x2010)
		_, err = kernel.VerifyCoalescing(weightSize, unalignedOffset)
		if err == nil {
			t.Errorf("expected unaligned offset 0x%x to fail coalescing check", unalignedOffset)
		}
	})

	// 3. Numerical Accuracy & Reference Parity (Q4_0)
	t.Run("Numerical_Parity_Q4_0", func(t *testing.T) {
		m := 16
		n := 32
		kDim := 64

		// Generate random inputs
		rng := rand.New(rand.NewSource(42))
		input := make([]float32, m*kDim)
		for i := range input {
			input[i] = rng.Float32()*2.0 - 1.0
		}

		// Packed Q4_0 weights: 2 values per byte
		packedWeights := make([]byte, (kDim*n)/2)
		for i := range packedWeights {
			// random nibbles 0..15
			n0 := byte(rng.Intn(16))
			n1 := byte(rng.Intn(16))
			packedWeights[i] = (n1 << 4) | n0
		}

		// Scales: 1 scale per 32 values
		numBlocks := (kDim * n) / Q4BlockSize
		scales := make([]float32, numBlocks)
		for i := range scales {
			scales[i] = 0.05 + rng.Float32()*0.1
		}

		outWMMA, err := kernel.ExecuteWMMA(input, packedWeights, scales, m, n, kDim)
		if err != nil {
			t.Fatalf("ExecuteWMMA failed: %v", err)
		}

		outRef, err := kernel.ReferenceGEMM(input, packedWeights, scales, m, n, kDim, QuantTypeQ4_0)
		if err != nil {
			t.Fatalf("ReferenceGEMM failed: %v", err)
		}

		if len(outWMMA) != len(outRef) {
			t.Fatalf("output length mismatch: %d vs %d", len(outWMMA), len(outRef))
		}

		var maxDiff float64
		for i := range outWMMA {
			diff := math.Abs(float64(outWMMA[i] - outRef[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > 1e-4 {
			t.Errorf("max numerical difference %.6f exceeds tolerance 1e-4", maxDiff)
		}
	})

	// 4. Numerical Accuracy & Reference Parity (Q8_0)
	t.Run("Numerical_Parity_Q8_0", func(t *testing.T) {
		cfgQ8 := DefaultDequantWMMAConfig()
		cfgQ8.QuantType = QuantTypeQ8_0
		kernelQ8, err := NewDequantWMMAKernel(cfgQ8)
		if err != nil {
			t.Fatalf("failed to create Q8 kernel: %v", err)
		}

		m := 16
		n := 32
		kDim := 64

		rng := rand.New(rand.NewSource(1337))
		input := make([]float32, m*kDim)
		for i := range input {
			input[i] = rng.Float32()*2.0 - 1.0
		}

		packedWeights := make([]byte, kDim*n)
		for i := range packedWeights {
			packedWeights[i] = byte(int8(rng.Intn(256) - 128))
		}

		numBlocks := (kDim * n) / Q8BlockSize
		scales := make([]float32, numBlocks)
		for i := range scales {
			scales[i] = 0.02 + rng.Float32()*0.05
		}

		outWMMA, err := kernelQ8.ExecuteWMMA(input, packedWeights, scales, m, n, kDim)
		if err != nil {
			t.Fatalf("ExecuteWMMA Q8 failed: %v", err)
		}

		outRef, err := kernelQ8.ReferenceGEMM(input, packedWeights, scales, m, n, kDim, QuantTypeQ8_0)
		if err != nil {
			t.Fatalf("ReferenceGEMM Q8 failed: %v", err)
		}

		var maxDiff float64
		for i := range outWMMA {
			diff := math.Abs(float64(outWMMA[i] - outRef[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > 1e-4 {
			t.Errorf("Q8 max numerical difference %.6f exceeds tolerance 1e-4", maxDiff)
		}
	})

	// 5. Numerical Accuracy & Reference Parity (FP4)
	t.Run("Numerical_Parity_FP4", func(t *testing.T) {
		cfgFP4 := DefaultDequantWMMAConfig()
		cfgFP4.QuantType = QuantTypeFP4
		kernelFP4, err := NewDequantWMMAKernel(cfgFP4)
		if err != nil {
			t.Fatalf("failed to create FP4 kernel: %v", err)
		}

		m := 16
		n := 16
		kDim := 32

		rng := rand.New(rand.NewSource(777))
		input := make([]float32, m*kDim)
		for i := range input {
			input[i] = rng.Float32()*1.5 - 0.75
		}

		packedWeights := make([]byte, (kDim*n)/2)
		for i := range packedWeights {
			packedWeights[i] = byte(rng.Intn(256))
		}

		numBlocks := (kDim * n) / Q4BlockSize
		scales := make([]float32, numBlocks)
		for i := range scales {
			scales[i] = 0.1
		}

		outWMMA, err := kernelFP4.ExecuteWMMA(input, packedWeights, scales, m, n, kDim)
		if err != nil {
			t.Fatalf("ExecuteWMMA FP4 failed: %v", err)
		}

		outRef, err := kernelFP4.ReferenceGEMM(input, packedWeights, scales, m, n, kDim, QuantTypeFP4)
		if err != nil {
			t.Fatalf("ReferenceGEMM FP4 failed: %v", err)
		}

		var maxDiff float64
		for i := range outWMMA {
			diff := math.Abs(float64(outWMMA[i] - outRef[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > 1e-4 {
			t.Errorf("FP4 max numerical difference %.6f exceeds tolerance 1e-4", maxDiff)
		}
	})

	// 6. Roofline geometry does not fabricate a hardware measurement.
	t.Run("Roofline_Memory_Bandwidth_Floor", func(t *testing.T) {
		// Single-token autoregressive decode: M=1, N=4096 (hidden dim), K=4096
		m := 1
		n := 4096
		kDim := 4096
		batchSize := 1

		model := kernel.EvaluateRoofline(m, n, kDim, batchSize)

		if model.TheoreticalPeakGBps != StrixHaloMaxLPDDR5XGBps {
			t.Errorf("theoretical peak %.3f GB/s, want %.3f GB/s", model.TheoreticalPeakGBps, StrixHaloMaxLPDDR5XGBps)
		}
		if model.SustainedBandwidthGBps != 0 || model.LatencySeconds != 0 || model.ThroughputTokensPerSec != 0 {
			t.Errorf("shape-only roofline must leave measured metrics at zero: %+v", model)
		}
		if model.MeetsBandwidthFloor {
			t.Errorf("shape-only roofline cannot satisfy the physical bandwidth floor")
		}
		if model.FLOPs <= 0 {
			t.Errorf("invalid FLOPs: %d", model.FLOPs)
		}
		if model.MemoryBytesRead <= 0 {
			t.Errorf("invalid MemoryBytesRead: %d", model.MemoryBytesRead)
		}
	})

	// 7. Concurrent Execution & Race Freedom
	t.Run("Concurrent_Execution_Race", func(t *testing.T) {
		const workers = 8
		var wg sync.WaitGroup
		wg.Add(workers)

		for w := 0; w < workers; w++ {
			go func(workerID int) {
				defer wg.Done()

				// Concurrently inspect assembly
				insp := kernel.InspectAssembly()
				if !insp.HasZeroLDS {
					t.Errorf("worker %d: missing zero LDS", workerID)
				}

				// Concurrently verify coalescing
				analysis, err := kernel.VerifyCoalescing(2048, uintptr(0x1000))
				if err != nil || analysis.EfficiencyPercent != 100.0 {
					t.Errorf("worker %d: coalescing failed", workerID)
				}

				// Concurrently run small matrix multiply
				m, n, kDim := 16, 16, 32
				in := make([]float32, m*kDim)
				wts := make([]byte, (kDim*n)/2)
				scs := make([]float32, (kDim*n)/32)
				for i := range scs {
					scs[i] = 1.0
				}
				_, err = kernel.ExecuteWMMA(in, wts, scs, m, n, kDim)
				if err != nil {
					t.Errorf("worker %d: execute failed: %v", workerID, err)
				}
			}(w)
		}

		wg.Wait()
	})
}
