package strix

import (
	"errors"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// referenceGEMM computes baseline unaccelerated float32 matrix multiplication: C = A * B.
func referenceGEMM(M, N, K int, A, B []float32) []float32 {
	C := make([]float32, M*N)
	for i := 0; i < M; i++ {
		rowA := i * K
		rowC := i * N
		for k := 0; k < K; k++ {
			valA := A[rowA+k]
			rowB := k * N
			for j := 0; j < N; j++ {
				C[rowC+j] += valA * B[rowB+j]
			}
		}
	}
	return C
}

// standardAttention computes unaccelerated reference scaled dot-product attention.
func standardAttention(Q, K, V []float32, seqLen, headDim int) []float32 {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	out := make([]float32, seqLen*headDim)

	for i := 0; i < seqLen; i++ {
		rowQ := i * headDim
		scores := make([]float32, seqLen)
		var maxScore float32 = -float32(math.MaxFloat32)

		for j := 0; j < seqLen; j++ {
			rowK := j * headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += Q[rowQ+d] * K[rowK+d]
			}
			s := dot * scale
			scores[j] = s
			if s > maxScore {
				maxScore = s
			}
		}

		var sumExp float32
		exps := make([]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			e := float32(math.Exp(float64(scores[j] - maxScore)))
			exps[j] = e
			sumExp += e
		}

		invSum := float32(0.0)
		if sumExp > 0 {
			invSum = 1.0 / sumExp
		}

		rowOut := i * headDim
		for j := 0; j < seqLen; j++ {
			weight := exps[j] * invSum
			rowV := j * headDim
			for d := 0; d < headDim; d++ {
				out[rowOut+d] += weight * V[rowV+d]
			}
		}
	}
	return out
}

// TestWave32WMMA_NumericalAccuracy verifies that output logits are bit-identical
// or within floating point epsilon vs standard GEMM and reference implementations.
func TestWave32WMMA_NumericalAccuracy(t *testing.T) {
	t.Run("FP32_16x16x16", func(t *testing.T) {
		M, N, K := 64, 64, 64
		A := make([]float32, M*K)
		B := make([]float32, K*N)
		for i := range A {
			A[i] = float32((i%17)-8) * 0.125
		}
		for i := range B {
			B[i] = float32((i%19)-9) * 0.125
		}

		refC := referenceGEMM(M, N, K, A, B)

		simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())
		gotC, telem, err := simulator.MatMul(M, N, K, A, B)
		if err != nil {
			t.Fatalf("MatMul failed: %v", err)
		}

		if err := telem.Validate(); err != nil {
			t.Fatalf("Telemetry validation failed: %v", err)
		}

		if len(gotC) != len(refC) {
			t.Fatalf("Dimension mismatch: got len %d, want %d", len(gotC), len(refC))
		}

		const epsilon = 1e-5
		for i := range refC {
			diff := math.Abs(float64(gotC[i] - refC[i]))
			if diff > epsilon {
				t.Fatalf("Element %d mismatch: got %f, want %f (diff %e > %e)",
					i, gotC[i], refC[i], diff, epsilon)
			}
		}
	})

	t.Run("FP32_16x16x32", func(t *testing.T) {
		M, N, K := 64, 64, 64
		A := make([]float32, M*K)
		B := make([]float32, K*N)
		for i := range A {
			A[i] = float32((i%13)-6) * 0.25
		}
		for i := range B {
			B[i] = float32((i%11)-5) * 0.25
		}

		refC := referenceGEMM(M, N, K, A, B)

		simulator := NewWave32RetiledMatMul(Wave32WMMAConfig16x16x32())
		gotC, telem, err := simulator.MatMul(M, N, K, A, B)
		if err != nil {
			t.Fatalf("MatMul 16x16x32 failed: %v", err)
		}
		if err := telem.Validate(); err != nil {
			t.Fatalf("Telemetry validation failed: %v", err)
		}

		const epsilon = 1e-5
		for i := range refC {
			diff := math.Abs(float64(gotC[i] - refC[i]))
			if diff > epsilon {
				t.Fatalf("Element %d mismatch: got %f, want %f (diff %e)",
					i, gotC[i], refC[i], diff)
			}
		}
	})

	t.Run("FP32_PartialTiles", func(t *testing.T) {
		// Non-multiples of 16 to test edge tile handling
		M, N, K := 35, 47, 53
		A := make([]float32, M*K)
		B := make([]float32, K*N)
		for i := range A {
			A[i] = float32(i % 7)
		}
		for i := range B {
			B[i] = float32(i % 5)
		}

		refC := referenceGEMM(M, N, K, A, B)

		simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())
		gotC, _, err := simulator.MatMul(M, N, K, A, B)
		if err != nil {
			t.Fatalf("MatMul partial tiles failed: %v", err)
		}

		const epsilon = 1e-5
		for i := range refC {
			diff := math.Abs(float64(gotC[i] - refC[i]))
			if diff > epsilon {
				t.Fatalf("Partial tile mismatch at %d: got %f, want %f", i, gotC[i], refC[i])
			}
		}
	})

	t.Run("BF16_WMMA_Parity", func(t *testing.T) {
		M, N, K := 32, 32, 32
		A_bf16 := make([]uint16, M*K)
		B_bf16 := make([]uint16, K*N)
		for i := range A_bf16 {
			A_bf16[i] = FP32ToBF16(float32((i%9)-4) * 0.5)
		}
		for i := range B_bf16 {
			B_bf16[i] = FP32ToBF16(float32((i%7)-3) * 0.5)
		}

		// Ground truth HAL reference from compute_hal.go
		refC, err := RDNA35_WMMA_GEMM(M, N, K, 1.0, A_bf16, B_bf16, 0.0, nil)
		if err != nil {
			t.Fatalf("RDNA35_WMMA_GEMM failed: %v", err)
		}

		simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())
		gotC, telem, err := simulator.MatMulBF16(M, N, K, A_bf16, B_bf16)
		if err != nil {
			t.Fatalf("MatMulBF16 failed: %v", err)
		}
		if err := telem.Validate(); err != nil {
			t.Fatalf("Telemetry validation failed: %v", err)
		}

		const epsilon = 1e-5
		for i := range refC {
			diff := math.Abs(float64(gotC[i] - refC[i]))
			if diff > epsilon {
				t.Fatalf("BF16 element %d mismatch: got %f, want %f (diff %e)",
					i, gotC[i], refC[i], diff)
			}
		}
	})

	t.Run("FlashAttentionReduction_Parity", func(t *testing.T) {
		seqLen := 64
		headDim := 64
		total := seqLen * headDim
		Q := make([]float32, total)
		K := make([]float32, total)
		V := make([]float32, total)

		for i := range Q {
			Q[i] = float32((i%13)-6) * 0.05
			K[i] = float32((i%11)-5) * 0.05
			V[i] = float32((i%7)-3) * 0.1
		}

		refOut := standardAttention(Q, K, V, seqLen, headDim)

		simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())
		gotOut, telem, err := simulator.FlashAttentionReduction(Q, K, V, seqLen, headDim)
		if err != nil {
			t.Fatalf("FlashAttentionReduction failed: %v", err)
		}

		if err := telem.Validate(); err != nil {
			t.Fatalf("FlashAttention telemetry validation failed: %v", err)
		}

		const epsilon = 1e-4
		for i := range refOut {
			diff := math.Abs(float64(gotOut[i] - refOut[i]))
			if diff > epsilon {
				t.Fatalf("FlashAttention mismatch at %d: got %f, want %f (diff %e)",
					i, gotOut[i], refOut[i], diff)
			}
		}
	})
}

// TestWave32WMMA_BankConflictVerification proves that LDS pad-2 eliminates 8-bank
// conflict stalls and reduces bank conflicts by >= 50%.
func TestWave32WMMA_BankConflictVerification(t *testing.T) {
	tracker := NewLDSBankConflictTracker()

	t.Run("BankIndexing_Formula", func(t *testing.T) {
		// Verify physical bank indexing: bank = (address / 4) % 32
		if bank := tracker.BankIndex(0); bank != 0 {
			t.Errorf("BankIndex(0) = %d, want 0", bank)
		}
		if bank := tracker.BankIndex(4); bank != 1 {
			t.Errorf("BankIndex(4) = %d, want 1", bank)
		}
		if bank := tracker.BankIndex(124); bank != 31 {
			t.Errorf("BankIndex(124) = %d, want 31", bank)
		}
		if bank := tracker.BankIndex(128); bank != 0 {
			t.Errorf("BankIndex(128) = %d, want 0", bank)
		}

		// Word indexing: bank = word % 32
		for w := 0; w < 64; w++ {
			expected := w % 32
			if got := tracker.BankIndexWord(w); got != expected {
				t.Errorf("BankIndexWord(%d) = %d, want %d", w, got, expected)
			}
		}
	})

	t.Run("CompareTileStrides_Eliminates8BankStall", func(t *testing.T) {
		comp := tracker.CompareTileStrides(16, 16, 2)

		// 1. Assert standard unpadded stride hits only 8 of 32 physical banks
		if comp.StandardBanksHit != 8 {
			t.Errorf("StandardBanksHit = %d, want exactly 8 (hitting only 8 of 32 banks)", comp.StandardBanksHit)
		}

		// 2. Assert standard stride causes severe 4-way bank collisions
		if comp.StandardMaxCollision != 4 {
			t.Errorf("StandardMaxCollision = %d, want 4 (4-way collision)", comp.StandardMaxCollision)
		}

		// 3. Assert pad-2 expands hits across at least 16 banks (distributed reads)
		if comp.PaddedBanksHit < 16 {
			t.Errorf("PaddedBanksHit = %d, want >= 16 banks", comp.PaddedBanksHit)
		}

		// 4. Assert pad-2 reduces max collision to <= 2-way distribution
		if comp.PaddedMaxCollision > 2 {
			t.Errorf("PaddedMaxCollision = %d, want <= 2 (conflict-free / 2-way distribution)", comp.PaddedMaxCollision)
		}

		// 5. Assert conflict reduction ratio >= 50%
		if comp.ConflictReductionRatio < 0.50 {
			t.Errorf("ConflictReductionRatio = %.2f, want >= 0.50", comp.ConflictReductionRatio)
		}

		// 6. Assert eliminates 8-bank stall flag is true
		if !comp.Eliminates8BankStall {
			t.Errorf("Eliminates8BankStall = false, expected true")
		}
	})

	t.Run("16Row_ColumnRead_ConflictFree", func(t *testing.T) {
		// 16 lanes in a half-wave reading column 0 across rows 0..15:
		// Standard: addr = r * 16
		// r=0..15: even rows hit bank 0, odd rows hit bank 16.
		// Hits only 2 banks with 8-way collision! (14 stall cycles)
		stdWords := make([]int, 16)
		padWords := make([]int, 16)
		for r := 0; r < 16; r++ {
			stdWords[r] = r * 16
			padWords[r] = r * 18 // pad-2
		}

		stdRes := tracker.AnalyzeWaveAccessWords(stdWords)
		padRes := tracker.AnalyzeWaveAccessWords(padWords)

		if stdRes.BanksHit != 2 {
			t.Errorf("std 16-row column read BanksHit = %d, want 2", stdRes.BanksHit)
		}
		if stdRes.MaxConflictDepth != 8 {
			t.Errorf("std 16-row column read MaxConflictDepth = %d, want 8 (8-way collision)", stdRes.MaxConflictDepth)
		}
		if stdRes.ConflictStallCycles != 14 {
			t.Errorf("std 16-row ConflictStallCycles = %d, want 14", stdRes.ConflictStallCycles)
		}

		// Pad-2: r*18 % 32 for r=0..15 covers all 16 even banks uniquely!
		if padRes.BanksHit != 16 {
			t.Errorf("pad-2 16-row BanksHit = %d, want 16", padRes.BanksHit)
		}
		if padRes.MaxConflictDepth != 1 {
			t.Errorf("pad-2 16-row MaxConflictDepth = %d, want 1 (conflict-free!)", padRes.MaxConflictDepth)
		}
		if padRes.ConflictStallCycles != 0 {
			t.Errorf("pad-2 16-row ConflictStallCycles = %d, want 0 (100%% conflict elimination)", padRes.ConflictStallCycles)
		}
	})

	t.Run("32Row_2ColumnRead_ConflictFree", func(t *testing.T) {
		// 32 lanes reading 2 columns across 16 rows:
		// Pad-2 distributes across all 32 banks!
		padWords := make([]int, 32)
		for c := 0; c < 2; c++ {
			for r := 0; r < 16; r++ {
				idx := c*16 + r
				padWords[idx] = r*18 + c
			}
		}

		padRes := tracker.AnalyzeWaveAccessWords(padWords)
		if padRes.BanksHit != 32 {
			t.Errorf("pad-2 32-lane read BanksHit = %d, want 32 (all banks utilized)", padRes.BanksHit)
		}
		if padRes.MaxConflictDepth != 1 {
			t.Errorf("pad-2 32-lane read MaxConflictDepth = %d, want 1 (100%% conflict-free)", padRes.MaxConflictDepth)
		}
		if padRes.ConflictStallCycles != 0 {
			t.Errorf("pad-2 32-lane read ConflictStallCycles = %d, want 0", padRes.ConflictStallCycles)
		}
	})

	t.Run("Tracker_SnapshotAndReset", func(t *testing.T) {
		tracker.Reset()
		snap0 := tracker.Snapshot()
		if snap0.StandardAccesses != 0 || snap0.StandardConflicts != 0 {
			t.Errorf("Snapshot after Reset not clean: %+v", snap0)
		}

		tracker.Record(24, 2)
		snap1 := tracker.Snapshot()
		if snap1.StandardConflicts != 24 || snap1.PaddedConflicts != 2 {
			t.Errorf("Snapshot after Record incorrect: %+v", snap1)
		}
		if snap1.ConflictReductionRatio < 0.90 {
			t.Errorf("Expected reduction ratio > 0.90, got %f", snap1.ConflictReductionRatio)
		}
	})
}

// TestWave32WMMA_PrefillThroughputGain asserts prefill throughput improvement >= 7.0%.
func TestWave32WMMA_PrefillThroughputGain(t *testing.T) {
	M, N, K := 128, 512, 512
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = float32(i%11) * 0.1
	}
	for i := range B {
		B[i] = float32(i%13) * 0.1
	}

	simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())
	_, telem, err := simulator.MatMul(M, N, K, A, B)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	if err := telem.Validate(); err != nil {
		t.Fatalf("Telemetry validation failed: %v", err)
	}

	// Invariant: throughput gain >= 7.0%
	if telem.ThroughputGainPercent < 7.0 {
		t.Errorf("ThroughputGainPercent = %.2f%%, want >= 7.0%%", telem.ThroughputGainPercent)
	}

	// Invariant: conflict reduction >= 50%
	if telem.ConflictReductionRatio < 0.50 {
		t.Errorf("ConflictReductionRatio = %.2f, want >= 0.50", telem.ConflictReductionRatio)
	}

	// Invariant: native Wave32 SIMD without Wave64 wrapping
	if !telem.NativeWave32 || telem.WaveSize != 32 {
		t.Errorf("Expected NativeWave32=true and WaveSize=32, got NativeWave32=%v, WaveSize=%d",
			telem.NativeWave32, telem.WaveSize)
	}

	// Invariant: arithmetic intensity > 0
	if telem.ArithmeticIntensity <= 0 {
		t.Errorf("ArithmeticIntensity = %f, want > 0", telem.ArithmeticIntensity)
	}

	// Invariant: verified conflict-free / 2-bank distribution
	if !telem.VerifiedConflictFree2Bank {
		t.Errorf("VerifiedConflictFree2Bank = false, want true")
	}

	t.Logf("Prefill Tok/s: Baseline=%.1f, Retiled=%.1f (+%.1f%% gain), ArithmeticIntensity=%.2f FLOPs/B, ConflictReduction=%.1f%%",
		telem.BaselineThroughputTokS, telem.RetiledThroughputTokS, telem.ThroughputGainPercent,
		telem.ArithmeticIntensity, telem.ConflictReductionRatio*100.0)
}

// TestWave32WMMA_ConcurrencyAndRace tests concurrent access to simulator and tracker under -race.
func TestWave32WMMA_ConcurrencyAndRace(t *testing.T) {
	simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())

	const workers = 16
	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		workerID := w
		go func() {
			defer wg.Done()

			r := rand.New(rand.NewSource(int64(workerID)))
			M, N, K := 32, 32, 32
			A := make([]float32, M*K)
			B := make([]float32, K*N)
			for i := range A {
				A[i] = r.Float32()
			}
			for i := range B {
				B[i] = r.Float32()
			}

			// 1. Concurrent MatMul
			_, telem, err := simulator.MatMul(M, N, K, A, B)
			if err != nil {
				t.Errorf("Concurrent MatMul failed: %v", err)
				return
			}
			if err := telem.Validate(); err != nil {
				t.Errorf("Concurrent telemetry validation failed: %v", err)
				return
			}

			// 2. Concurrent FlashAttentionReduction
			seqLen := 32
			headDim := 32
			Q := make([]float32, seqLen*headDim)
			Kv := make([]float32, seqLen*headDim)
			V := make([]float32, seqLen*headDim)
			for i := range Q {
				Q[i] = r.Float32() * 0.1
				Kv[i] = r.Float32() * 0.1
				V[i] = r.Float32() * 0.1
			}
			_, faTelem, err := simulator.FlashAttentionReduction(Q, Kv, V, seqLen, headDim)
			if err != nil {
				t.Errorf("Concurrent FlashAttentionReduction failed: %v", err)
				return
			}
			if err := faTelem.Validate(); err != nil {
				t.Errorf("Concurrent FA telemetry validation failed: %v", err)
				return
			}

			// 3. Concurrent Tracker Access
			_ = simulator.Tracker().CompareTileStrides(16, 16, 2)
			_ = simulator.Tracker().Snapshot()
		}()
	}

	wg.Wait()
}

// TestWave32WMMA_ErrorHandling verifies boundary and dimension validation.
func TestWave32WMMA_ErrorHandling(t *testing.T) {
	simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())

	// Invalid dimensions
	if _, _, err := simulator.MatMul(0, 16, 16, nil, nil); !errors.Is(err, ErrInvalidDimensions) {
		t.Errorf("expected ErrInvalidDimensions, got %v", err)
	}
	if _, _, err := simulator.MatMul(16, -1, 16, nil, nil); !errors.Is(err, ErrInvalidDimensions) {
		t.Errorf("expected ErrInvalidDimensions, got %v", err)
	}

	// Dimension mismatch
	A := make([]float32, 10)
	B := make([]float32, 16*16)
	if _, _, err := simulator.MatMul(16, 16, 16, A, B); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}

	// Config validation
	cfg := DefaultWave32WMMAConfig()
	cfg.WaveSize = 64 // Invalid for native Wave32
	if err := cfg.Validate(); err == nil {
		t.Errorf("expected error for WaveSize 64, got nil")
	}

	cfg2 := DefaultWave32WMMAConfig()
	cfg2.TileK = 48 // Invalid WMMA primitive
	if err := cfg2.Validate(); err == nil {
		t.Errorf("expected error for TileK 48, got nil")
	}
}

// BenchmarkWave32WMMA_PrefillThroughput measures prefill tile throughput and asserts >= 7% gain.
func BenchmarkWave32WMMA_PrefillThroughput(b *testing.B) {
	M, N, K := 64, 256, 256
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = float32(i%17) * 0.1
	}
	for i := range B {
		B[i] = float32(i%19) * 0.1
	}

	simulator := NewWave32RetiledMatMul(DefaultWave32WMMAConfig())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, telem, err := simulator.MatMul(M, N, K, A, B)
		if err != nil {
			b.Fatalf("MatMul failed: %v", err)
		}
		if telem.ThroughputGainPercent < 7.0 {
			b.Fatalf("Throughput gain %.2f%% < 7.0%%", telem.ThroughputGainPercent)
		}
	}
}

// --- Wave32 gfx1151 Q2_K decode GEMV tests --------------------------------------------

// encodeQ2KSuperBlock builds one canonical 84-byte Q2_K super-block for tests, mirroring the
// GGUF layout q2kDequantSuperBlock (and Q2KDequantSuperBlock) reads: 16 scale bytes, 64 quant
// bytes, then f16 d and f16 dmin. It is the fixture the parity test round-trips.
func encodeQ2KSuperBlock(scales []byte, q []byte, d, dmin float32) []byte {
	blk := make([]byte, Q2KSuperBlockBytes)
	copy(blk[0:16], scales)
	copy(blk[16:80], q)
	putF16 := func(off int, f float32) {
		h := f32ToF16Bits(f)
		blk[off] = byte(h)
		blk[off+1] = byte(h >> 8)
	}
	putF16(80, d)
	putF16(82, dmin)
	return blk
}

// f32ToF16Bits converts float32 to IEEE 754 half-precision bits (round-to-nearest-even),
// the inverse of f16BitsToF32 for the fixture values used here.
func f32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	man := bits & 0x7fffff
	switch {
	case (bits & 0x7f800000) == 0x7f800000:
		return sign | 0x7c00 | uint16(man>>13)
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint32(14 - exp)
		half := uint32(1) << (shift - 1)
		man = (man + half) >> shift
		return sign | uint16(man)
	default:
		man += 0x1000
		if man&0x800000 != 0 {
			man = 0
			exp++
			if exp >= 0x1f {
				return sign | 0x7c00
			}
		}
		return sign | uint16(exp)<<10 | uint16(man>>13)
	}
}

// q2kReferenceVectorDot is an independent scalar reference for one super-block dot: it decodes
// the 2-bit codes directly (no shared code with Q2KDequantSuperBlock) and reduces in float64.
func q2kReferenceVectorDot(blk []byte, x []float32) float64 {
	scales := blk[:16]
	q := blk[16:80]
	d := float64(f16BitsToF32(uint16(blk[80]) | uint16(blk[81])<<8))
	dmin := float64(f16BitsToF32(uint16(blk[82]) | uint16(blk[83])<<8))
	var sum float64
	qi, is := 0, 0
	for n := 0; n < 256; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			sc := scales[is]
			is++
			dl, ml := d*float64(sc&0x0f), dmin*float64(sc>>4)
			table := [4]float64{0 - ml, dl - ml, dl*2 - ml, dl*3 - ml}
			for l := 0; l < 16; l++ {
				sum += table[(q[qi+l]>>shift)&3] * float64(x[n+j*32+l])
			}
			sc = scales[is]
			is++
			dl, ml = d*float64(sc&0x0f), dmin*float64(sc>>4)
			table = [4]float64{0 - ml, dl - ml, dl*2 - ml, dl*3 - ml}
			for l := 0; l < 16; l++ {
				sum += table[(q[qi+16+l]>>shift)&3] * float64(x[n+j*32+16+l])
			}
			shift += 2
		}
		qi += 32
	}
	return sum
}

// TestStrixWave32GEMVDecode_Q2KParity verifies the Wave32 decode GEMV kernel body against an
// independent scalar Q2_K reference: exact argmax, rel-L2 <= 1e-4, cosine >= 0.99999.
func TestStrixWave32GEMVDecode_Q2KParity(t *testing.T) {
	const out, in = 17, 512 // 2 super-blocks per row
	rowBlocks := in / Q2KSuperBlockElems
	rng := rand.New(rand.NewSource(0x51ACE))

	raw := make([]byte, out*rowBlocks*Q2KSuperBlockBytes)
	for r := 0; r < out*rowBlocks; r++ {
		scales := make([]byte, 16)
		q := make([]byte, 64)
		for i := range scales {
			scales[i] = byte(rng.Intn(256))
		}
		for i := range q {
			q[i] = byte(rng.Intn(256))
		}
		d := 0.001 + rng.Float32()*0.05
		dmin := 0.0005 + rng.Float32()*0.02
		copy(raw[r*Q2KSuperBlockBytes:], encodeQ2KSuperBlock(scales, q, d, dmin))
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	got, err := Wave32Q2KGEMVStream(raw, x, out, in)
	if err != nil {
		t.Fatalf("Wave32Q2KGEMVStream: %v", err)
	}
	if len(got) != out {
		t.Fatalf("output len %d != out %d", len(got), out)
	}

	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for b := 0; b < rowBlocks; b++ {
			blk := raw[(o*rowBlocks+b)*Q2KSuperBlockBytes : (o*rowBlocks+b+1)*Q2KSuperBlockBytes]
			want[o] += q2kReferenceVectorDot(blk, x[b*Q2KSuperBlockElems:(b+1)*Q2KSuperBlockElems])
		}
	}

	// Exact argmax.
	gotArg, wantArg := 0, 0
	for i := 1; i < out; i++ {
		if got[i] > got[gotArg] {
			gotArg = i
		}
		if want[i] > want[wantArg] {
			wantArg = i
		}
	}
	if gotArg != wantArg {
		t.Fatalf("argmax mismatch: got %d, want %d", gotArg, wantArg)
	}

	// rel-L2 <= 1e-4.
	var l2diff, l2want float64
	for i := 0; i < out; i++ {
		d := float64(got[i]) - want[i]
		l2diff += d * d
		l2want += want[i] * want[i]
	}
	if l2want == 0 {
		t.Fatal("degenerate all-zero reference")
	}
	relL2 := math.Sqrt(l2diff) / math.Sqrt(l2want)
	if relL2 > 1e-4 {
		t.Fatalf("rel-L2 %.3e > 1e-4", relL2)
	}

	// cosine >= 0.99999.
	gf := make([]float32, out)
	wf := make([]float32, out)
	for i := range got {
		gf[i] = got[i]
		wf[i] = float32(want[i])
	}
	cos, err := CosineSimilarity(gf, wf)
	if err != nil {
		t.Fatalf("CosineSimilarity: %v", err)
	}
	if cos < 0.99999 {
		t.Fatalf("cosine %.8f < 0.99999", cos)
	}
	t.Logf("Q2_K decode GEMV parity: argmax=%d rel-L2=%.3e cosine=%.9f", gotArg, relL2, cos)
}

// TestStrixWave32GEMVDecode_FailClosed is the negative test: when the kernel is unavailable the
// toggle MUST refuse (ErrWave32GEMVUnavailable) and MUST NOT silently run a CPU/dequant fallback.
func TestStrixWave32GEMVDecode_FailClosed(t *testing.T) {
	// Unresolved launch path: admission is false by construction.
	k := DefaultWave32GEMVDecodeKernel()
	if k.Available() {
		t.Fatal("default gfx1151 decode GEMV must be unavailable (fail closed)")
	}
	adm := k.Admission()
	if adm.Admitted {
		t.Fatal("admission admitted without a proven launch path")
	}
	if !adm.DeviceVisible {
		t.Fatal("toggle must be device-visible from the decode MatMul path")
	}

	raw := make([]byte, Q2KSuperBlockBytes)
	x := make([]float32, Q2KSuperBlockElems)
	_, err := k.DispatchQ2KGEMV(raw, x, 1, Q2KSuperBlockElems)
	if !errors.Is(err, ErrWave32GEMVUnavailable) {
		t.Fatalf("dispatch must refuse with ErrWave32GEMVUnavailable, got %v", err)
	}

	// The device-visible toggle must also refuse, not degrade.
	toggle := ResolveDecodeGEMVDeviceToggle(k, "")
	if toggle.Admitted {
		t.Fatal("device toggle admitted without a proven launch path")
	}
	if err := RequireDecodeGEMVDevice(toggle); !errors.Is(err, ErrWave32GEMVDecodeUnavailable) {
		t.Fatalf("RequireDecodeGEMVDevice must refuse fail-closed, got %v", err)
	}

	// A wrong-arch kernel is inadmissible even with a launch path asserted.
	badCfg := DefaultWave32GEMVDecodeConfig()
	badCfg.Arch = ArchGFX1150
	bad := NewWave32GEMVDecodeKernel(badCfg, true)
	if bad.Available() {
		t.Fatal("non-gfx1151 arch must never admit the decode GEMV")
	}
}

// TestStrixDecodeGEMVDevice_ToggleEnv verifies the opt-in switch and launch-path proof are
// read fail-closed from the environment.
func TestStrixDecodeGEMVDevice_ToggleEnv(t *testing.T) {
	t.Setenv(EnvStrixWave32GEMVDecode, "")
	t.Setenv(EnvStrixGEMVLaunchPath, "")
	if StrixDecodeGEMVDeviceRequested() {
		t.Fatal("decode GEMV must be opt-in only")
	}
	toggle, _ := ResolveStrixDecodeGEMVDevice()
	if toggle.Admitted {
		t.Fatal("must fail closed when not requested")
	}

	t.Setenv(EnvStrixWave32GEMVDecode, "1")
	t.Setenv(EnvStrixGEMVLaunchPath, "none")
	if !StrixDecodeGEMVDeviceRequested() {
		t.Fatal("explicit opt-in not recognized")
	}
	toggle, _ = ResolveStrixDecodeGEMVDevice()
	if toggle.Admitted {
		t.Fatal("asserted launch path 'none' must not admit")
	}

	t.Setenv(EnvStrixGEMVLaunchPath, "hsaco-aql")
	toggle, kernel := ResolveStrixDecodeGEMVDevice()
	if !toggle.Admitted || !kernel.Available() {
		t.Fatalf("proven launch path must admit, got %+v", toggle)
	}
	if toggle.LaunchPath != "hsaco-aql" {
		t.Fatalf("launch path %q != hsaco-aql", toggle.LaunchPath)
	}
}

// TestStrixDecodeGEMV_UnmeasuredBandwidth cannot fabricate a device number.
func TestStrixDecodeGEMV_UnmeasuredBandwidth(t *testing.T) {
	if _, err := EstimateDecodeBandwidthGBps(1<<30, 0); err == nil {
		t.Fatal("zero elapsed must not yield a bandwidth number")
	}
	if _, err := EstimateDecodeBandwidthGBps(0, time.Second); err == nil {
		t.Fatal("zero bytes must not yield a bandwidth number")
	}
	// 240 GB/s sustained meets the 80% floor; 100 GB/s does not.
	if got, err := EstimateDecodeBandwidthGBps(240<<30, time.Second); err != nil || !MeetsDecodeBandwidthFloor(got) {
		t.Fatalf("240 GB/s should meet the floor: %v", err)
	}
	if MeetsDecodeBandwidthFloor(100) {
		t.Fatal("100 GB/s must not meet the 218.44 GB/s floor")
	}
	if MeetsDecodeBandwidthFloor(0) {
		t.Fatal("unmeasured (0) must not meet the floor")
	}
}

// TestStrixWave32GEMVDecodeBandwidthFloor is the acceptance harness. It runs a timed decode
// GEMV through the kernel body on the host. It is [SW-VERIFIED] only: the byte-rate is an
// in-sim/host number and is deliberately NOT asserted against the physical 218.44 GB/s floor
// here — that criterion is [HW-WITNESSED] and must remain UNCHECKED until run on real gfx1151.
func TestStrixWave32GEMVDecodeBandwidthFloor(t *testing.T) {
	const out, in = 256, 4096
	rowBlocks := in / Q2KSuperBlockElems
	rng := rand.New(rand.NewSource(7))
	raw := make([]byte, out*rowBlocks*Q2KSuperBlockBytes)
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// Sanity: the kernel body executes and returns the right shape.
	if _, err := Wave32Q2KGEMVStream(raw, x, out, in); err != nil {
		t.Fatalf("kernel body: %v", err)
	}

	start := time.Now()
	if _, err := Wave32Q2KGEMVStream(raw, x, out, in); err != nil {
		t.Fatalf("timed kernel body: %v", err)
	}
	elapsed := time.Since(start)
	hostGBps, err := EstimateDecodeBandwidthGBps(int64(len(raw)), elapsed)
	if err != nil {
		t.Fatalf("bandwidth estimate: %v", err)
	}
	t.Logf("[SW-VERIFIED] host decode GEMV: %d bytes in %v = %.2f GB/s (physical floor %.2f GB/s is [HW-WITNESSED], UNCHECKED here)",
		len(raw), elapsed, hostGBps, TargetDecodeBandwidthFloorGBps)

	// The floor constant is the 80% roofline target.
	if TargetDecodeBandwidthFloorGBps < 218.44 || TargetDecodeBandwidthFloorGBps > 218.45 {
		t.Fatalf("TargetDecodeBandwidthFloorGBps %.4f != 218.44", TargetDecodeBandwidthFloorGBps)
	}
}

// TestStrixWave32GEMVDecode_DeviceGuarded is the device-gated witness. It runs only when the
// operator has asserted a proven gfx1151 launch path AND opted in; otherwise it skips and the
// physical floor criterion stays [HW-WITNESSED]-UNCHECKED.
func TestStrixWave32GEMVDecode_DeviceGuarded(t *testing.T) {
	if !StrixDecodeGEMVDeviceRequested() || StrixGEMVLaunchPath() == "" {
		t.Skip("gfx1151 Wave32 decode GEMV device witness requires opt-in + a proven launch path (remains [HW-WITNESSED]-UNCHECKED)")
	}
	toggle, kernel := ResolveStrixDecodeGEMVDevice()
	if !toggle.Admitted || !kernel.Available() {
		t.Fatalf("opted in with launch path %q but not admitted: %s", StrixGEMVLaunchPath(), toggle.Reason)
	}
	t.Log("device GEMV admitted; physical byte-rate witness remains [HW-WITNESSED] pending real strix silicon")
}

// BenchmarkStrixWave32GEMVDecodeGBps measures the host kernel-body throughput the real launch
// must beat. It reports MB/s of streamed weight bytes; it is not a physical device number.
func BenchmarkStrixWave32GEMVDecodeGBps(b *testing.B) {
	const out, in = 256, 4096
	rowBlocks := in / Q2KSuperBlockElems
	raw := make([]byte, out*rowBlocks*Q2KSuperBlockBytes)
	for i := range raw {
		raw[i] = byte(i * 131)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i%7) * 0.1
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Wave32Q2KGEMVStream(raw, x, out, in); err != nil {
			b.Fatal(err)
		}
	}
}
