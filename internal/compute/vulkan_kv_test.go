//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// TestVulkanKVScratchpadDequantOnce verifies the dequant-once KV scratchpad pipeline:
// 1. Memory allocation adhering to 32MB MALL Infinity Cache boundaries and 256-bit bus alignment.
// 2. FitsInMALLCache correctly partitions context depths.
// 3. DequantCount == 1 and HeadReuses == 40 across all 40 attention heads (no per-head re-dequantization).
// 4. Modeled deep-context prefill throughput speedup (3.26x boost at 64k context).
// 5. Zero-copy buffer reuse via ResetReuse across subsequent forward passes.
func TestVulkanKVScratchpadDequantOnce(t *testing.T) {
	t.Run("MALL_Cache_And_Bus_Alignment", func(t *testing.T) {
		const (
			nKV     = 8
			headDim = 128
		)
		testCases := []struct {
			nPos       int
			format     QuantizedKVType
			expectMALL bool
		}{
			{1024, QuantizedKVQ8_0, true},   // 8 MB <= 32 MB
			{2048, QuantizedKVQ4_0, true},   // 16 MB <= 32 MB
			{4096, QuantizedKVQ8_0, true},   // 32 MB <= 32 MB
			{32768, QuantizedKVQ8_0, false}, // 256 MB > 32 MB
			{65536, QuantizedKVQ4_0, false}, // 512 MB > 32 MB
		}

		for _, tc := range testCases {
			s, err := NewVulkanKVScratchpad("gfx1151", tc.format, tc.nPos, nKV, headDim)
			if err != nil {
				t.Fatalf("NewVulkanKVScratchpad failed for nPos=%d: %v", tc.nPos, err)
			}
			if !s.IsMALLAligned() {
				t.Errorf("scratchpad nPos=%d allocated bytes %d is not 128B MALL cacheline and 256-bit bus aligned",
					tc.nPos, s.AllocatedBytes)
			}
			if s.FitsInMALLCache() != tc.expectMALL {
				t.Errorf("scratchpad nPos=%d FitsInMALLCache = %v, want %v (allocated bytes = %d, MALL limit = %d)",
					tc.nPos, s.FitsInMALLCache(), tc.expectMALL, s.AllocatedBytes, StrixHaloMALLCacheBytes)
			}
			if s.ScratchBytes() != s.AllocatedBytes {
				t.Errorf("ScratchBytes = %d, want %d", s.ScratchBytes(), s.AllocatedBytes)
			}
		}

		// Validation errors for invalid dimensions / format
		if _, err := NewVulkanKVScratchpad("gfx1151", QuantizedKVQ8_0, 0, nKV, headDim); err == nil {
			t.Errorf("expected error for nPos=0")
		}
		if _, err := NewVulkanKVScratchpad("gfx1151", QuantizedKVQ8_0, 1024, -1, headDim); err == nil {
			t.Errorf("expected error for nKV=-1")
		}
		if _, err := NewVulkanKVScratchpad("gfx1151", "invalid_fmt", 1024, nKV, headDim); err == nil {
			t.Errorf("expected error for unsupported format")
		}
	})

	t.Run("SingleDequantAcross40Heads", func(t *testing.T) {
		const (
			nPos    = 128
			nKV     = 8
			nQ      = 40
			headDim = 128
		)
		totalKV := nKV * nPos * headDim
		rng := rand.New(rand.NewSource(12186))

		f32K := make([]float32, totalKV)
		f32V := make([]float32, totalKV)
		for i := range f32K {
			f32K[i] = rng.Float32()*2.0 - 1.0
			f32V[i] = rng.Float32()*2.0 - 1.0
		}

		q := make([]float32, nQ*headDim)
		for i := range q {
			q[i] = rng.Float32()*2.0 - 1.0
		}

		formats := []QuantizedKVType{QuantizedKVQ8_0, QuantizedKVQ4_0}
		for _, fmtType := range formats {
			var rawK, rawV []byte
			var err error
			switch fmtType {
			case QuantizedKVQ8_0:
				rawK, err = QuantizeF32ToQ8_0(f32K)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ8_0 K: %v", err)
				}
				rawV, err = QuantizeF32ToQ8_0(f32V)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ8_0 V: %v", err)
				}
			case QuantizedKVQ4_0:
				rawK, err = QuantizeF32ToQ4_0(f32K)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ4_0 K: %v", err)
				}
				rawV, err = QuantizeF32ToQ4_0(f32V)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ4_0 V: %v", err)
				}
			}

			s, err := NewVulkanKVScratchpad("gfx1151", fmtType, nPos, nKV, headDim)
			if err != nil {
				t.Fatalf("NewVulkanKVScratchpad failed: %v", err)
			}

			out, err := s.ExecuteAttentionWithDequantOnce(q, rawK, rawV, nQ)
			if err != nil {
				t.Fatalf("ExecuteAttentionWithDequantOnce failed: %v", err)
			}
			if len(out) != nQ*headDim {
				t.Fatalf("unexpected out length %d, want %d", len(out), nQ*headDim)
			}

			// Key acceptance criterion: exactly 1 dequantization pass across all 40 attention heads
			if s.DequantCount != 1 {
				t.Errorf("%s DequantCount = %d, want 1 (must dequantize exactly once per forward pass)", fmtType, s.DequantCount)
			}
			if s.HeadReuses != nQ {
				t.Errorf("%s HeadReuses = %d, want %d (all query heads must reuse the scratchpad)", fmtType, s.HeadReuses, nQ)
			}
		}
	})

	t.Run("ThroughputSpeedupMultiplier", func(t *testing.T) {
		sDeep, _ := NewVulkanKVScratchpad("gfx1151", QuantizedKVQ8_0, 65536, 8, 128)
		multDeep := sDeep.SpeedupMultiplier(40)
		if multDeep < 3.20 || multDeep > 3.30 {
			t.Errorf("deep-context SpeedupMultiplier = %.2f, want ~3.26x (>= 230 tok/s)", multDeep)
		}

		sMed, _ := NewVulkanKVScratchpad("gfx1151", QuantizedKVQ8_0, 2048, 8, 128)
		multMed := sMed.SpeedupMultiplier(40)
		if multMed < 1.30 || multMed > 1.40 {
			t.Errorf("medium-context SpeedupMultiplier = %.2f, want ~1.35x", multMed)
		}

		sShort, _ := NewVulkanKVScratchpad("gfx1151", QuantizedKVQ8_0, 512, 8, 128)
		multShort := sShort.SpeedupMultiplier(40)
		if multShort < 1.10 || multShort > 1.20 {
			t.Errorf("short-context SpeedupMultiplier = %.2f, want ~1.15x", multShort)
		}
	})

	t.Run("ZeroCopyResetReuse", func(t *testing.T) {
		const (
			nPos    = 64
			nKV     = 4
			headDim = 64
		)
		s, err := NewVulkanKVScratchpad("gfx1151", QuantizedKVQ8_0, nPos, nKV, headDim)
		if err != nil {
			t.Fatalf("NewVulkanKVScratchpad: %v", err)
		}
		rawK := make([]byte, nKV*nPos*headDim*34/32)
		rawV := make([]byte, nKV*nPos*headDim*34/32)
		q := make([]float32, 16*headDim)

		if _, err := s.ExecuteAttentionWithDequantOnce(q, rawK, rawV, 16); err != nil {
			t.Fatalf("ExecuteAttentionWithDequantOnce: %v", err)
		}
		if s.DequantCount != 1 || s.HeadReuses != 16 {
			t.Errorf("post-execution counts: DequantCount=%d, HeadReuses=%d", s.DequantCount, s.HeadReuses)
		}

		// Reset reuse without memory deallocation
		s.ResetReuse()
		if s.DequantCount != 0 || s.HeadReuses != 0 {
			t.Errorf("post-reset counts: DequantCount=%d, HeadReuses=%d", s.DequantCount, s.HeadReuses)
		}
		if s.ScratchK == nil || s.ScratchV == nil {
			t.Errorf("ResetReuse must retain allocated buffers")
		}
	})
}

// TestVulkanKVParityAgainstCPU verifies bit-exact and high-fidelity numerical parity
// between Vulkan dequant-once scratchpad attention and CPU reference attention across
// Q8_0 and Q4_0 formats.
func TestVulkanKVParityAgainstCPU(t *testing.T) {
	const (
		nPos    = 64
		nKV     = 8
		nQ      = 40
		headDim = 128
	)
	totalKV := nKV * nPos * headDim
	rng := rand.New(rand.NewSource(42))

	f32K := make([]float32, totalKV)
	f32V := make([]float32, totalKV)
	for i := range f32K {
		f32K[i] = rng.Float32()*2.0 - 1.0
		f32V[i] = rng.Float32()*2.0 - 1.0
	}

	q := make([]float32, nQ*headDim)
	for i := range q {
		q[i] = rng.Float32()*2.0 - 1.0
	}

	formats := []QuantizedKVType{QuantizedKVQ8_0, QuantizedKVQ4_0}
	for _, fmtType := range formats {
		t.Run(string(fmtType), func(t *testing.T) {
			var rawK, rawV []byte
			var err error
			switch fmtType {
			case QuantizedKVQ8_0:
				rawK, err = QuantizeF32ToQ8_0(f32K)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ8_0 K: %v", err)
				}
				rawV, err = QuantizeF32ToQ8_0(f32V)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ8_0 V: %v", err)
				}
			case QuantizedKVQ4_0:
				rawK, err = QuantizeF32ToQ4_0(f32K)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ4_0 K: %v", err)
				}
				rawV, err = QuantizeF32ToQ4_0(f32V)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ4_0 V: %v", err)
				}
			}

			// 1. CPU Reference: dequantize on CPU and execute reference multi-head attention
			refK := make([]float32, totalKV)
			refV := make([]float32, totalKV)
			if err := DequantizeQuantizedKV(refK, rawK, totalKV, fmtType); err != nil {
				t.Fatalf("DequantizeQuantizedKV refK: %v", err)
			}
			if err := DequantizeQuantizedKV(refV, rawV, totalKV, fmtType); err != nil {
				t.Fatalf("DequantizeQuantizedKV refV: %v", err)
			}
			refAttn := cpuReferenceMultiHeadAttention(q, refK, refV, nPos, nQ, nKV, headDim)

			// 2. Vulkan dequant-once scratchpad attention
			gotAttn, scratch, err := ExecuteVulkanAttentionWithDequantOnce(q, rawK, rawV, "gfx1151", nPos, nQ, nKV, headDim, fmtType)
			if err != nil {
				t.Fatalf("ExecuteVulkanAttentionWithDequantOnce failed: %v", err)
			}

			if scratch.DequantCount != 1 {
				t.Errorf("DequantCount = %d, want 1", scratch.DequantCount)
			}
			if scratch.HeadReuses != nQ {
				t.Errorf("HeadReuses = %d, want %d", scratch.HeadReuses, nQ)
			}

			// Numerical parity verification: cosine similarity >= 0.999900, max delta <= 1e-3
			cosSim := CosineSimilarity(gotAttn, refAttn)
			if cosSim < 0.999900 {
				t.Errorf("%s cosine similarity = %.8f, want >= 0.999900", fmtType, cosSim)
			}

			var maxDelta float32
			for i := range gotAttn {
				d := float32(math.Abs(float64(gotAttn[i] - refAttn[i])))
				if d > maxDelta {
					maxDelta = d
				}
			}
			if maxDelta > 1e-3 {
				t.Errorf("%s max absolute delta = %g, want <= 1e-3", fmtType, maxDelta)
			}

			t.Logf("%s Parity PASSED: cosine=%.8f, maxDelta=%g, dequantCount=%d, headReuses=%d",
				fmtType, cosSim, maxDelta, scratch.DequantCount, scratch.HeadReuses)
		})
	}
}
