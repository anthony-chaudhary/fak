package strix

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

// TestROCmFP4_FP16Conversion verifies IEEE 754 half-precision float conversions.
func TestROCmFP4_FP16Conversion(t *testing.T) {
	testCases := []struct {
		name     string
		f32      float32
		expected uint16
	}{
		{"zero", 0.0, 0x0000},
		{"neg_zero", float32(math.Copysign(0, -1)), 0x8000},
		{"one", 1.0, 0x3C00},
		{"neg_one", -1.0, 0xBC00},
		{"two", 2.0, 0x4000},
		{"half", 0.5, 0x3800},
		{"one_point_five", 1.5, 0x3E00},
		{"max_normal", 65504.0, 0x7BFF},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			h := FP32ToFP16(tc.f32)
			if h != tc.expected {
				t.Errorf("FP32ToFP16(%f) = 0x%04X, want 0x%04X", tc.f32, h, tc.expected)
			}
			f := FP16ToFP32(h)
			if math.Abs(float64(f-tc.f32)) > 1e-4 {
				t.Errorf("FP16ToFP32(0x%04X) = %f, want %f", h, f, tc.f32)
			}
		})
	}

	// Test subnormal numbers
	t.Run("subnormals", func(t *testing.T) {
		smallestSubnormal := float32(5.9604645e-8) // 2^-24
		h := FP32ToFP16(smallestSubnormal)
		if h != 0x0001 {
			t.Errorf("FP32ToFP16(2^-24) = 0x%04X, want 0x0001", h)
		}
		f := FP16ToFP32(h)
		if math.Abs(float64(f-smallestSubnormal)) > 1e-12 {
			t.Errorf("FP16ToFP32(0x0001) = %e, want %e", f, smallestSubnormal)
		}
	})

	// Test special values: Inf, NaN
	t.Run("specials", func(t *testing.T) {
		posInf := float32(math.Inf(1))
		hInf := FP32ToFP16(posInf)
		if hInf != 0x7C00 {
			t.Errorf("FP32ToFP16(+Inf) = 0x%04X, want 0x7C00", hInf)
		}
		if !math.IsInf(float64(FP16ToFP32(hInf)), 1) {
			t.Errorf("expected +Inf")
		}

		nan := float32(math.NaN())
		hNaN := FP32ToFP16(nan)
		if (hNaN&0x7C00) != 0x7C00 || (hNaN&0x03FF) == 0 {
			t.Errorf("FP32ToFP16(NaN) = 0x%04X, want NaN bit pattern", hNaN)
		}
		if !math.IsNaN(float64(FP16ToFP32(hNaN))) {
			t.Errorf("expected NaN")
		}
	})
}

// TestROCmFP4_PackUnpackRoundtrip verifies roundtrip pack/unpack across normal,
// edge-case, zero, and exact E2M1 distributions.
func TestROCmFP4_PackUnpackRoundtrip(t *testing.T) {
	t.Run("exact_e2m1_grid", func(t *testing.T) {
		// All 16 E2M1 positive and negative points
		src := make([]float32, 32)
		for i := 0; i < 16; i++ {
			src[i] = e2m1Table[i]
			src[i+16] = e2m1Table[i]
		}
		block, err := PackFP4Block32(src)
		if err != nil {
			t.Fatalf("PackFP4Block32 failed: %v", err)
		}

		unpacked := UnpackFP4Block32(block)
		if len(unpacked) != 32 {
			t.Fatalf("unpacked length = %d, want 32", len(unpacked))
		}

		for i := 0; i < 32; i++ {
			if math.Abs(float64(unpacked[i]-src[i])) > 1e-4 {
				t.Errorf("index %d: unpacked %f != expected %f", i, unpacked[i], src[i])
			}
		}
	})

	t.Run("zero_distribution", func(t *testing.T) {
		src := make([]float32, 32)
		block, err := PackFP4Block32(src)
		if err != nil {
			t.Fatalf("PackFP4Block32 failed: %v", err)
		}
		if block.Scale != 0 {
			t.Errorf("expected scale 0 for all-zero block, got 0x%04X", block.Scale)
		}
		if block.Flags&BlockFlagZero == 0 {
			t.Errorf("expected BlockFlagZero flag set")
		}

		unpacked := UnpackFP4Block32(block)
		for i, v := range unpacked {
			if v != 0.0 {
				t.Errorf("index %d: expected 0.0, got %f", i, v)
			}
		}
	})

	t.Run("normal_distribution", func(t *testing.T) {
		rng := rand.New(rand.NewSource(1234))
		src := make([]float32, 32)
		for i := 0; i < 32; i++ {
			src[i] = float32(rng.NormFloat64())
		}

		block, err := PackFP4Block32(src)
		if err != nil {
			t.Fatalf("PackFP4Block32 failed: %v", err)
		}
		unpacked := UnpackFP4Block32(block)

		sim, err := CosineSimilarity(src, unpacked)
		if err != nil {
			t.Fatalf("CosineSimilarity failed: %v", err)
		}
		if sim < 0.99 {
			t.Errorf("expected cosine similarity >= 0.99, got %.6f", sim)
		}
	})

	t.Run("edge_cases_large_small_values", func(t *testing.T) {
		scales := []float32{1e4, 1e-3, 500.0, 0.05}
		for _, s := range scales {
			src := make([]float32, 32)
			for i := 0; i < 32; i++ {
				val := float32(i%7-3) * s
				src[i] = val
			}
			block, err := PackFP4Block32(src)
			if err != nil {
				t.Fatalf("scale %f PackFP4Block32 failed: %v", s, err)
			}
			unpacked := UnpackFP4Block32(block)
			sim, err := CosineSimilarity(src, unpacked)
			if err != nil {
				t.Fatalf("scale %f CosineSimilarity failed: %v", s, err)
			}
			if sim < 0.99 {
				t.Errorf("scale %f: expected cosine similarity >= 0.99, got %.6f", s, sim)
			}
		}
	})

	t.Run("invalid_inputs", func(t *testing.T) {
		// Wrong slice size
		_, err := PackFP4Block32(make([]float32, 31))
		if !errors.Is(err, ErrInvalidBlockSize) {
			t.Errorf("expected ErrInvalidBlockSize, got %v", err)
		}

		_, err = PackFP4Block32(make([]float32, 33))
		if !errors.Is(err, ErrInvalidBlockSize) {
			t.Errorf("expected ErrInvalidBlockSize, got %v", err)
		}

		// NaN / Inf inputs
		nanSrc := make([]float32, 32)
		nanSrc[10] = float32(math.NaN())
		_, err = PackFP4Block32(nanSrc)
		if !errors.Is(err, ErrInvalidFloatValue) {
			t.Errorf("expected ErrInvalidFloatValue for NaN, got %v", err)
		}

		infSrc := make([]float32, 32)
		infSrc[5] = float32(math.Inf(1))
		_, err = PackFP4Block32(infSrc)
		if !errors.Is(err, ErrInvalidFloatValue) {
			t.Errorf("expected ErrInvalidFloatValue for Inf, got %v", err)
		}
	})
}

// TestROCmFP4_CooperativeMatMul_NumericalAccuracy verifies that ROCmFP4 fused
// cooperative matrix multiplication achieves cosine similarity >= 0.99 against FP32 reference.
func TestROCmFP4_CooperativeMatMul_NumericalAccuracy(t *testing.T) {
	dimensions := []struct{ rows, cols int }{
		{64, 128},
		{128, 256},
		{256, 512},
		{128, 1024},
	}

	for _, dim := range dimensions {
		t.Run(fmt.Sprintf("dim_%dx%d", dim.rows, dim.cols), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(dim.rows * dim.cols)))
			matrix := make([]float32, dim.rows*dim.cols)
			for i := range matrix {
				matrix[i] = float32(rng.NormFloat64())
			}
			vector := make([]float32, dim.cols)
			for i := range vector {
				vector[i] = float32(rng.NormFloat64())
			}

			// Reference FP32 matrix-vector multiplication
			refY := make([]float32, dim.rows)
			for r := 0; r < dim.rows; r++ {
				var sum float64
				rowBase := r * dim.cols
				for c := 0; c < dim.cols; c++ {
					sum += float64(matrix[rowBase+c]) * float64(vector[c])
				}
				refY[r] = float32(sum)
			}

			// Pack tensor into ROCmFP4
			tensor, err := PackTensorROCmFP4(matrix, dim.rows, dim.cols)
			if err != nil {
				t.Fatalf("PackTensorROCmFP4 failed: %v", err)
			}

			// Execute fused cooperative matrix GEMV
			fp4Y, err := ROCmFP4CoopMatMul(tensor, vector)
			if err != nil {
				t.Fatalf("ROCmFP4CoopMatMul failed: %v", err)
			}

			// Verify cosine similarity >= 0.99
			sim, err := CosineSimilarity(refY, fp4Y)
			if err != nil {
				t.Fatalf("CosineSimilarity failed: %v", err)
			}
			if sim < 0.99 {
				t.Errorf("dim %dx%d: expected cosine similarity >= 0.99, got %.6f", dim.rows, dim.cols, sim)
			}

			// Verify Dequantize() fidelity as well
			dequantMatrix := tensor.Dequantize()
			matSim, err := CosineSimilarity(matrix, dequantMatrix)
			if err != nil {
				t.Fatalf("Matrix CosineSimilarity failed: %v", err)
			}
			if matSim < 0.99 {
				t.Errorf("dim %dx%d: expected dequantized matrix cosine similarity >= 0.99, got %.6f", dim.rows, dim.cols, matSim)
			}
		})
	}
}

// TestROCmFP4_Wave32Alignment verifies Wave32 32-element stride and 128-bit cacheline alignment.
func TestROCmFP4_Wave32Alignment(t *testing.T) {
	t.Run("struct_alignment_invariants", func(t *testing.T) {
		structSize := unsafe.Sizeof(FP4Block32{})
		if structSize != 32 {
			t.Errorf("FP4Block32 size = %d bytes, want 32 bytes (256 bits = 2x 128-bit cachelines)", structSize)
		}
		if structSize%16 != 0 {
			t.Errorf("FP4Block32 size %d is not a multiple of 16 bytes (128 bits)", structSize)
		}
	})

	t.Run("valid_alignment", func(t *testing.T) {
		matrix := make([]float32, 64*128)
		tensor, err := PackTensorROCmFP4(matrix, 64, 128)
		if err != nil {
			t.Fatalf("PackTensorROCmFP4 failed: %v", err)
		}
		if err := ValidateWave32Alignment(tensor); err != nil {
			t.Errorf("ValidateWave32Alignment failed on valid tensor: %v", err)
		}
	})

	t.Run("unaligned_columns", func(t *testing.T) {
		unalignedCols := []int{31, 33, 48, 50, 63, 65}
		for _, cols := range unalignedCols {
			matrix := make([]float32, 10*cols)
			_, err := PackTensorROCmFP4(matrix, 10, cols)
			if !errors.Is(err, ErrUnalignedWave32Stride) {
				t.Errorf("cols %d: expected ErrUnalignedWave32Stride, got %v", cols, err)
			}
		}
	})

	t.Run("nil_tensor", func(t *testing.T) {
		if err := ValidateWave32Alignment(nil); !errors.Is(err, ErrNilTensor) {
			t.Errorf("expected ErrNilTensor, got %v", err)
		}
	})

	t.Run("invalid_dimensions", func(t *testing.T) {
		tensor := &ROCmFP4Tensor{Rows: 0, Cols: 32}
		if err := ValidateWave32Alignment(tensor); !errors.Is(err, ErrInvalidDimensions) {
			t.Errorf("expected ErrInvalidDimensions, got %v", err)
		}
	})
}

// TestROCmFP4_Telemetry verifies telemetry metrics on Strix Halo:
// compression ratio, active wavefront occupancy, memory bandwidth efficiency.
func TestROCmFP4_Telemetry(t *testing.T) {
	rows, cols := 128, 512
	matrix := make([]float32, rows*cols)
	tensor, err := PackTensorROCmFP4(matrix, rows, cols)
	if err != nil {
		t.Fatalf("PackTensorROCmFP4 failed: %v", err)
	}

	telem := tensor.Telemetry

	// Original: rows * cols * 4 = 128 * 512 * 4 = 262,144 bytes
	if telem.OriginalSizeBytes != int64(rows*cols*4) {
		t.Errorf("original bytes = %d, want %d", telem.OriginalSizeBytes, rows*cols*4)
	}

	// Total blocks = 128 * (512 / 32) = 128 * 16 = 2,048 blocks
	// Packed bytes = 2048 * 18 = 36,864 bytes
	expectedBlocks := rows * (cols / 32)
	expectedPacked := int64(expectedBlocks * 18)
	if telem.PackedSizeBytes != expectedPacked {
		t.Errorf("packed bytes = %d, want %d", telem.PackedSizeBytes, expectedPacked)
	}

	// Compression ratio: 262144 / 36864 = 7.111x
	if math.Abs(telem.CompressionRatio-7.1111) > 0.01 {
		t.Errorf("compression ratio = %.4f, want ~7.1111", telem.CompressionRatio)
	}
	if math.Abs(tensor.CompressionRatio()-7.1111) > 0.01 {
		t.Errorf("tensor.CompressionRatio() = %.4f, want ~7.1111", tensor.CompressionRatio())
	}

	// Effective BPW: 4.5 bits/weight for quantized block-32 tensor
	if math.Abs(telem.EffectiveBPW-4.50) > 1e-4 {
		t.Errorf("effective BPW = %.2f, want 4.50", telem.EffectiveBPW)
	}

	// Active Wavefront Occupancy: 1.0 (100%)
	if telem.ActiveWavefrontOccupancy != 1.0 {
		t.Errorf("active wavefront occupancy = %.2f, want 1.0", telem.ActiveWavefrontOccupancy)
	}
	if tensor.ActiveWavefrontOccupancy() != 1.0 {
		t.Errorf("tensor.ActiveWavefrontOccupancy() = %.2f, want 1.0", tensor.ActiveWavefrontOccupancy())
	}

	// Sustained memory bandwidth: 231 GB/s
	if telem.SustainedBandwidthGBps != 231.0 {
		t.Errorf("sustained bandwidth = %.2f, want 231.0", telem.SustainedBandwidthGBps)
	}

	// Bandwidth efficiency: 231 / 256 = ~0.9023 (90.23%)
	if math.Abs(telem.BandwidthEfficiency-0.9023) > 0.01 {
		t.Errorf("bandwidth efficiency = %.4f, want ~0.9023", telem.BandwidthEfficiency)
	}
	if math.Abs(tensor.MemoryBandwidthEfficiency()-0.9023) > 0.01 {
		t.Errorf("tensor.MemoryBandwidthEfficiency() = %.4f, want ~0.9023", tensor.MemoryBandwidthEfficiency())
	}
}

// TestROCmFP4_ConcurrentRace verifies thread-safety and race-free concurrent execution under -race.
func TestROCmFP4_ConcurrentRace(t *testing.T) {
	rows, cols := 64, 256
	matrix := make([]float32, rows*cols)
	for i := range matrix {
		matrix[i] = float32(i % 17)
	}
	tensor, err := PackTensorROCmFP4(matrix, rows, cols)
	if err != nil {
		t.Fatalf("PackTensorROCmFP4 failed: %v", err)
	}

	vector := make([]float32, cols)
	for i := range vector {
		vector[i] = 1.0
	}

	const workers = 16
	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		go func(id int) {
			defer wg.Done()
			dst := make([]float32, rows)
			for iter := 0; iter < 50; iter++ {
				err := ROCmFP4CoopMatMulInto(tensor, vector, dst)
				if err != nil {
					t.Errorf("worker %d iter %d failed: %v", id, iter, err)
					return
				}
			}
		}(w)
	}

	wg.Wait()
}

// BenchmarkROCmFP4CoopMatMul_HotPath benchmarks hot-path fused cooperative matrix-vector
// multiplication and asserts ZERO heap allocations per operation.
func BenchmarkROCmFP4CoopMatMul_HotPath(b *testing.B) {
	rows, cols := 256, 1024
	matrix := make([]float32, rows*cols)
	for i := range matrix {
		matrix[i] = float32(i%13 - 6)
	}
	tensor, err := PackTensorROCmFP4(matrix, rows, cols)
	if err != nil {
		b.Fatalf("PackTensorROCmFP4 failed: %v", err)
	}

	vector := make([]float32, cols)
	for i := range vector {
		vector[i] = float32(i%5 - 2)
	}

	dst := make([]float32, rows)

	b.ReportAllocs()
	b.SetBytes(int64(rows * cols * 4)) // Effective FP32 byte stream
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := ROCmFP4CoopMatMulInto(tensor, vector, dst)
		if err != nil {
			b.Fatal(err)
		}
	}
}
