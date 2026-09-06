package amdgpu

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestF16KVContiguizeRoundtrip(t *testing.T) {
	numTokens := 128
	numKVHeads := 8
	headDim := 128
	headBytes := headDim * 2
	totalBytes := numTokens * numKVHeads * headBytes

	src := make([]byte, totalBytes)
	rng := rand.New(rand.NewSource(42))
	rng.Read(src)

	contiguizer := NewF16KVContiguizer(numKVHeads, headDim, true)

	contig, err := contiguizer.Contiguize(src, numTokens)
	if err != nil {
		t.Fatalf("Contiguize failed: %v", err)
	}
	if len(contig) != totalBytes {
		t.Fatalf("Contiguize len = %d, want %d", len(contig), totalBytes)
	}

	restored, err := contiguizer.Decontiguize(contig, numTokens)
	if err != nil {
		t.Fatalf("Decontiguize failed: %v", err)
	}

	if !bytes.Equal(src, restored) {
		t.Fatalf("Decontiguize output did not match original src (bit-exact roundtrip failed)")
	}
}

func TestF16KVContiguizeThreshold(t *testing.T) {
	cStrix := NewF16KVContiguizer(8, 128, true)
	cDiscrete := NewF16KVContiguizer(8, 128, false)

	// Single-token decode on Strix should not contiguize (stride penalty negligible for 1 token)
	if cStrix.ShouldContiguize(1) {
		t.Errorf("ShouldContiguize(1) = true on Strix, want false")
	}

	// Prefill batch (64 tokens) on Strix should activate contiguization
	if !cStrix.ShouldContiguize(64) {
		t.Errorf("ShouldContiguize(64) = false on Strix, want true")
	}

	// Deep context (32k / 64k) on Strix must activate contiguization
	if !cStrix.ShouldContiguize(32768) {
		t.Errorf("ShouldContiguize(32768) = false on Strix, want true")
	}
	if !cStrix.ShouldContiguize(65536) {
		t.Errorf("ShouldContiguize(65536) = false on Strix, want true")
	}

	// ShouldContiguizeBatch tests
	if !cStrix.ShouldContiguizeBatch(1, 32768) {
		t.Errorf("ShouldContiguizeBatch(1, 32768) = false on Strix, want true")
	}
	if !cStrix.ShouldContiguizeBatch(64, 512) {
		t.Errorf("ShouldContiguizeBatch(64, 512) = false on Strix, want true")
	}
	if cStrix.ShouldContiguizeBatch(1, 512) {
		t.Errorf("ShouldContiguizeBatch(1, 512) = true on Strix, want false")
	}

	// Discrete GPU should not activate (no LPDDR5X UMA channel camping)
	if cDiscrete.ShouldContiguize(64) {
		t.Errorf("ShouldContiguize(64) = true on discrete GPU, want false")
	}
	if cDiscrete.ShouldContiguize(65536) {
		t.Errorf("ShouldContiguize(65536) = true on discrete GPU, want false")
	}

	// Defensive check: negative offsets do not panic
	eff, active := SimulateChannelDistribution([]int{-64, -128, 0, 64}, 64)
	if active == 0 && eff > 0 {
		t.Errorf("unexpected distribution for empty/negative offsets")
	}
}

func TestF16KVChannelCampingElimination(t *testing.T) {
	// Simulate memory channel distribution for strided vs contiguous f16 KV reading
	// Geometry: 32k tokens, 8 KV heads, headDim 128 (headBytes = 256 bytes = 4 cachelines of 64B)
	numTokens := 32768
	numKVHeads := 8
	headDim := 128
	headBytes := headDim * 2 // 256 bytes

	// 1. Strided access: reading head 0 across tokens in token-interleaved layout
	// In token-interleaved, head 0 of token T is at T * (numKVHeads * headBytes) = T * 2048 bytes
	// Stride is 2048 bytes = 32 cachelines of 64 bytes.
	// Since 32 is a multiple of 16 (the channel count), (T * 32) % 16 == 0!
	// All requests hit channel 0! This is the exact LPDDR5X channel camping defect.
	stridedOffsets := make([]int, numTokens)
	tokenStride := numKVHeads * headBytes
	for i := 0; i < numTokens; i++ {
		stridedOffsets[i] = i * tokenStride
	}

	stridedEff, stridedActive := SimulateChannelDistribution(stridedOffsets, 64)
	if stridedActive > 2 {
		t.Errorf("expected strided access to camp on <= 2 channels, got %d active", stridedActive)
	}

	// 2. Contiguous access: reading head 0 in head-contiguous layout
	// In head-contiguous, head 0 has tokens packed sequentially: T * 256 bytes
	// Stride is 256 bytes = 4 cachelines of 64 bytes.
	// Low-order cachelines advance 0, 4, 8, 12, 0, 4... cycling across channels with bursting!
	// Furthermore, reading all cachelines of head 0 sequentially touches 0, 1, 2, 3, 4, 5...
	contiguousOffsets := make([]int, numTokens*4)
	for i := 0; i < numTokens*4; i++ {
		contiguousOffsets[i] = i * 64 // sequential cachelines
	}

	contigEff, contigActive := SimulateChannelDistribution(contiguousOffsets, 64)
	if contigActive != 16 {
		t.Errorf("expected contiguous access to use all 16 channels, got %d", contigActive)
	}

	// Contiguous efficiency must be significantly higher than strided efficiency (>= 2.5x gain)
	gain := contigEff / (stridedEff + 1e-6)
	if gain < 2.5 {
		t.Errorf("channel distribution efficiency gain = %.2fx, want >= 2.5x (contig=%.4f, strided=%.4f)",
			gain, contigEff, stridedEff)
	}
}
