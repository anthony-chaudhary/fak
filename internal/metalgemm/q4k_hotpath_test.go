package metalgemm

import (
	"encoding/binary"
	"math"
	"math/rand"
	"regexp"
	"testing"
)

// makeTestQ4KSuperBlock creates a 144-byte Q4_K block with random valid weights and scales.
// Exponents for d and dmin are kept in reasonable ranges [0x0c..0x12] to avoid NaN/Inf.
func makeTestQ4KSuperBlock(rng *rand.Rand) []byte {
	blk := make([]byte, Q4KSuperBlockBytes)
	for i := range blk {
		blk[i] = byte(rng.Intn(256))
	}
	putF16 := func(off int) {
		exp := uint16(0x0c + rng.Intn(6)) // ~2^-3 .. 2^2
		frac := uint16(rng.Intn(1024))
		binary.LittleEndian.PutUint16(blk[off:], (exp<<10)|frac)
	}
	putF16(0) // d
	putF16(2) // dmin
	return blk
}

// makeTestQ4KMatrix generates raw Q4_K bytes for an [out, in] matrix.
func makeTestQ4KMatrix(rng *rand.Rand, out, in int) []byte {
	nblk := in / Q4KSuperBlockWeights
	raw := make([]byte, out*nblk*Q4KSuperBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * Q4KSuperBlockBytes
			blk := makeTestQ4KSuperBlock(rng)
			copy(raw[base:base+Q4KSuperBlockBytes], blk)
		}
	}
	return raw
}

// makeTestActivationsRange generates a float32 vector with values in [minVal, maxVal].
func makeTestActivationsRange(rng *rand.Rand, n int, minVal, maxVal float32) []float32 {
	x := make([]float32, n)
	span := maxVal - minVal
	for i := range x {
		x[i] = minVal + rng.Float32()*span
	}
	return x
}

func TestQ4KHotpath(t *testing.T) {
	t.Run("ProvenanceAnchors", func(t *testing.T) {
		shaRegex := regexp.MustCompile(`^[0-9a-f]{40}$`)
		if !shaRegex.MatchString(LlamaCppProvenanceCommit) {
			t.Fatalf("invalid llama.cpp commit anchor: %q", LlamaCppProvenanceCommit)
		}
		if LlamaCppLicense != "MIT" {
			t.Fatalf("expected llama.cpp license MIT, got %q", LlamaCppLicense)
		}
		if !shaRegex.MatchString(MLXLMProvenanceCommit) {
			t.Fatalf("invalid MLX-LM commit anchor: %q", MLXLMProvenanceCommit)
		}
		if MLXMLicense != "MIT" {
			t.Fatalf("expected MLX-LM license MIT, got %q", MLXMLicense)
		}
	})

	t.Run("NumericParity_SingleBlock", func(t *testing.T) {
		rng := rand.New(rand.NewSource(8394))
		for trial := 0; trial < 100; trial++ {
			blk := makeTestQ4KSuperBlock(rng)
			x := makeTestActivationsRange(rng, Q4KSuperBlockWeights, -2.0, 2.0)

			kernelVal := KernelQ4KDotProduct(blk, x)
			refVal := ReferenceQ4KDotProduct(blk, x)

			diff := math.Abs(float64(kernelVal - refVal))
			denom := math.Abs(float64(refVal)) + 1e-6
			relErr := diff / denom
			if relErr > 1e-5 {
				t.Fatalf("trial %d: kernel=%v ref=%v relErr=%v > 1e-5", trial, kernelVal, refVal, relErr)
			}
		}
	})

	t.Run("NumericParity_MultiBlockMatrix", func(t *testing.T) {
		// Test multiple shapes including Qwen3.8 projection dimensions.
		shapes := []struct {
			out, in int
		}{
			{out: 16, in: 512},
			{out: 32, in: 1024},
			{out: 8, in: 5120}, // Qwen3.8 hidden dim = 5120 (20 superblocks)
		}

		for _, shape := range shapes {
			rng := rand.New(rand.NewSource(int64(shape.out*1000 + shape.in)))
			raw := makeTestQ4KMatrix(rng, shape.out, shape.in)
			x := makeTestActivationsRange(rng, shape.in, -1.5, 1.5)

			yKernel := make([]float32, shape.out)
			yRef := make([]float32, shape.out)

			KernelQ4KGEMV(raw, shape.out, shape.in, x, yKernel)
			ReferenceQ4KGEMV(raw, shape.out, shape.in, x, yRef)

			cosSim := CosineSimilarity(yKernel, yRef)
			if cosSim < 0.999999 {
				t.Fatalf("shape [%d, %d]: cosine similarity = %.9f < 0.999999", shape.out, shape.in, cosSim)
			}
			t.Logf("shape [%d, %d]: cosine similarity = %.9f (>= 0.999999 requirement passed)", shape.out, shape.in, cosSim)
		}
	})

	t.Run("EdgeCases_AllZeros", func(t *testing.T) {
		rng := rand.New(rand.NewSource(1234))

		// Case 1: Zero block, random activations.
		zeroBlk := make([]byte, Q4KSuperBlockBytes)
		randX := makeTestActivationsRange(rng, Q4KSuperBlockWeights, -1.0, 1.0)
		kVal := KernelQ4KDotProduct(zeroBlk, randX)
		rVal := ReferenceQ4KDotProduct(zeroBlk, randX)
		if kVal != 0.0 || rVal != 0.0 {
			t.Fatalf("zero block: kernel=%v ref=%v, want 0.0", kVal, rVal)
		}

		// Case 2: Random block, zero activations.
		randBlk := makeTestQ4KSuperBlock(rng)
		zeroX := make([]float32, Q4KSuperBlockWeights)
		kVal2 := KernelQ4KDotProduct(randBlk, zeroX)
		rVal2 := ReferenceQ4KDotProduct(randBlk, zeroX)
		if kVal2 != 0.0 || rVal2 != 0.0 {
			t.Fatalf("zero activations: kernel=%v ref=%v, want 0.0", kVal2, rVal2)
		}

		// Case 3: Both zero.
		kVal3 := KernelQ4KDotProduct(zeroBlk, zeroX)
		rVal3 := ReferenceQ4KDotProduct(zeroBlk, zeroX)
		if kVal3 != 0.0 || rVal3 != 0.0 {
			t.Fatalf("both zero: kernel=%v ref=%v, want 0.0", kVal3, rVal3)
		}
	})

	t.Run("EdgeCases_NegativeActivations", func(t *testing.T) {
		rng := rand.New(rand.NewSource(5678))
		out, in := 16, 1024
		raw := makeTestQ4KMatrix(rng, out, in)
		negX := makeTestActivationsRange(rng, in, -5.0, -0.1)

		yKernel := make([]float32, out)
		yRef := make([]float32, out)

		KernelQ4KGEMV(raw, out, in, negX, yKernel)
		ReferenceQ4KGEMV(raw, out, in, negX, yRef)

		cosSim := CosineSimilarity(yKernel, yRef)
		if cosSim < 0.999999 {
			t.Fatalf("negative activations: cosine similarity = %.9f < 0.999999", cosSim)
		}
		t.Logf("negative activations: cosine similarity = %.9f (>= 0.999999 passed)", cosSim)
	})

	t.Run("EdgeCases_MaxQuantRanges", func(t *testing.T) {
		rng := rand.New(rand.NewSource(9012))

		// Construct block with max 4-bit quants (all 0xFF -> low=15, high=15) and max 6-bit scales (63).
		maxBlk := make([]byte, Q4KSuperBlockBytes)
		// d = 2.0, dmin = 1.0 in f16
		binary.LittleEndian.PutUint16(maxBlk[0:2], 0x4000) // 2.0
		binary.LittleEndian.PutUint16(maxBlk[2:4], 0x3c00) // 1.0
		for i := 4; i < 16; i++ {
			maxBlk[i] = 0xff // all 6-bit scales/mins maxed to 63
		}
		for i := 16; i < 144; i++ {
			maxBlk[i] = 0xff // low nibble = 15, high nibble = 15
		}

		x := makeTestActivationsRange(rng, Q4KSuperBlockWeights, -2.0, 2.0)
		kVal := KernelQ4KDotProduct(maxBlk, x)
		rVal := ReferenceQ4KDotProduct(maxBlk, x)

		diff := math.Abs(float64(kVal - rVal))
		denom := math.Abs(float64(rVal)) + 1e-6
		if diff/denom > 1e-5 {
			t.Fatalf("max quant block: kernel=%v ref=%v relErr=%v", kVal, rVal, diff/denom)
		}

		// Min quants (all 0) with max scales
		minBlk := make([]byte, Q4KSuperBlockBytes)
		binary.LittleEndian.PutUint16(minBlk[0:2], 0x4000)
		binary.LittleEndian.PutUint16(minBlk[2:4], 0x3c00)
		for i := 4; i < 16; i++ {
			minBlk[i] = 0xff
		}
		// quants left at 0x00
		kValMin := KernelQ4KDotProduct(minBlk, x)
		rValMin := ReferenceQ4KDotProduct(minBlk, x)
		diffMin := math.Abs(float64(kValMin - rValMin))
		denomMin := math.Abs(float64(rValMin)) + 1e-6
		if diffMin/denomMin > 1e-5 {
			t.Fatalf("min quant block: kernel=%v ref=%v relErr=%v", kValMin, rValMin, diffMin/denomMin)
		}
	})
}

func BenchmarkKernelQ4KDotProduct(b *testing.B) {
	const in = 5120 // Qwen3.8 hidden dim = 5120 (20 superblocks)
	nblk := in / Q4KSuperBlockWeights
	rng := rand.New(rand.NewSource(42))
	raw := makeTestQ4KMatrix(rng, 1, in)
	x := makeTestActivationsRange(rng, in, -1.0, 1.0)

	b.SetBytes(int64(nblk * Q4KSuperBlockBytes))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = KernelQ4KDotProduct(raw, x)
	}
}

func BenchmarkNaiveQ4KDotProduct(b *testing.B) {
	const in = 5120 // Qwen3.8 hidden dim = 5120 (20 superblocks)
	nblk := in / Q4KSuperBlockWeights
	rng := rand.New(rand.NewSource(42))
	raw := makeTestQ4KMatrix(rng, 1, in)
	x := makeTestActivationsRange(rng, in, -1.0, 1.0)

	b.SetBytes(int64(nblk * Q4KSuperBlockBytes))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NaiveQ4KDotProduct(raw, x)
	}
}
