package strix

import (
	"errors"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestComputeHAL_TargetConfig(t *testing.T) {
	// 1. Verify GFX1151 (Ryzen AI Max+ 395) configuration
	cfg1151, err := GetTargetConfig(ArchGFX1151)
	if err != nil {
		t.Fatalf("unexpected error getting gfx1151 config: %v", err)
	}
	if cfg1151.Arch != ArchGFX1151 {
		t.Errorf("expected arch %s, got %s", ArchGFX1151, cfg1151.Arch)
	}
	if cfg1151.ComputeUnits != 40 {
		t.Errorf("expected 40 CUs for gfx1151, got %d", cfg1151.ComputeUnits)
	}
	if !cfg1151.SupportsWMMA {
		t.Errorf("expected gfx1151 to support WMMA")
	}
	if !cfg1151.DualIssueSIMD {
		t.Errorf("expected gfx1151 to support dual-issue SIMD")
	}
	if cfg1151.WaveSize != 32 {
		t.Errorf("expected WaveSize 32, got %d", cfg1151.WaveSize)
	}
	if cfg1151.L3CacheBytes != 32*1024*1024 {
		t.Errorf("expected 32MB L3 cache, got %d bytes", cfg1151.L3CacheBytes)
	}
	if cfg1151.PeakBandwidthGBps != 256.0 {
		t.Errorf("expected 256 GB/s peak bandwidth, got %.1f", cfg1151.PeakBandwidthGBps)
	}
	if cfg1151.PeakBF16TFLOPs < 50.0 {
		t.Errorf("expected >= 50 TFLOPs BF16 peak, got %.2f", cfg1151.PeakBF16TFLOPs)
	}

	// 2. Verify GFX1150 (Strix Point) configuration
	cfg1150, err := GetTargetConfig(ArchGFX1150)
	if err != nil {
		t.Fatalf("unexpected error getting gfx1150 config: %v", err)
	}
	if cfg1150.ComputeUnits != 16 {
		t.Errorf("expected 16 CUs for gfx1150, got %d", cfg1150.ComputeUnits)
	}

	// 3. Verify invalid target
	_, err = GetTargetConfig("gfx9999")
	if err == nil {
		t.Errorf("expected error for invalid arch, got nil")
	}
}

func TestComputeHAL_PageAlignmentAndAlloc(t *testing.T) {
	allocator := NewUnifiedAllocator(64*1024*1024, ArchGFX1151) // 64 MB pool

	testSizes := []int64{64, 128, 4096, 8192, 65536, 1024 * 1024}

	for _, size := range testSizes {
		buf, err := allocator.Allocate(size, StandardUMAPlags)
		if err != nil {
			t.Fatalf("failed to allocate %d bytes: %v", size, err)
		}

		// 1. Verify 4096-byte page alignment
		if !IsPageAligned(buf.HostPtr) {
			t.Fatalf("HostPtr %x is not 4096-byte page-aligned (offset %d)", buf.HostPtr, buf.HostPtr%PageAlignment)
		}
		if !IsPageAligned(buf.DevicePtr) {
			t.Fatalf("DevicePtr %x is not 4096-byte page-aligned", buf.DevicePtr)
		}

		// 2. Verify unified UMA host-device pointer equivalence
		if buf.HostPtr != buf.DevicePtr {
			t.Errorf("HostPtr %x != DevicePtr %x in unified memory", buf.HostPtr, buf.DevicePtr)
		}

		// 3. Verify Zero-Copy & Coherency properties
		if !buf.IsZeroCopy() {
			t.Errorf("expected buffer to be zero-copy mapped")
		}
		if !buf.IsCoherent() {
			t.Errorf("expected buffer to be coherent")
		}

		// 4. Verify buffer slice data integrity
		bytes := buf.AlignedBytes()
		if int64(len(bytes)) != size {
			t.Errorf("expected AlignedBytes() len %d, got %d", size, len(bytes))
		}

		// Write test pattern
		for i := 0; i < len(bytes) && i < 256; i++ {
			bytes[i] = byte(i & 0xFF)
		}
		for i := 0; i < len(bytes) && i < 256; i++ {
			if bytes[i] != byte(i&0xFF) {
				t.Fatalf("byte mismatch at %d: got %d, want %d", i, bytes[i], byte(i&0xFF))
			}
		}

		// Test float interpretation
		if size >= 4096 {
			f32Slice := buf.Float32Slice()
			if f32Slice == nil || len(f32Slice) != int(size/4) {
				t.Errorf("Float32Slice() returned unexpected slice length: %d (want %d)", len(f32Slice), size/4)
			}
			f32Slice[0] = 3.14159
			if math.Abs(float64(f32Slice[0]-3.14159)) > 1e-5 {
				t.Errorf("float value write/read failed")
			}
		}

		// Test BF16 interpretation
		if size >= 4096 {
			u16Slice := buf.Uint16Slice()
			if u16Slice == nil || len(u16Slice) != int(size/2) {
				t.Errorf("Uint16Slice() returned unexpected slice length: %d (want %d)", len(u16Slice), size/2)
			}
		}

		// Clean up
		if err := allocator.Free(buf); err != nil {
			t.Fatalf("failed to free buffer: %v", err)
		}

		// Verify double free protection
		if err := allocator.Free(buf); !errors.Is(err, ErrBufferFreed) {
			t.Errorf("expected ErrBufferFreed on second free, got %v", err)
		}
	}

	// Verify allocator capacity enforcement
	tinyAlloc := NewUnifiedAllocator(8192, ArchGFX1151)
	_, err := tinyAlloc.Allocate(4096, StandardUMAPlags)
	if err != nil {
		t.Fatalf("failed to allocate within capacity: %v", err)
	}
	_, err = tinyAlloc.Allocate(4096, StandardUMAPlags)
	if err != nil {
		t.Fatalf("failed second allocation within capacity: %v", err)
	}
	// Third allocation should fail out-of-memory
	_, err = tinyAlloc.Allocate(1024, StandardUMAPlags)
	if !errors.Is(err, ErrOutOfMemory) {
		t.Errorf("expected ErrOutOfMemory, got %v", err)
	}
}

func TestComputeHAL_BF16ConversionAndParity(t *testing.T) {
	testValues := []float32{
		0.0,
		-0.0,
		1.0,
		-1.0,
		0.5,
		-0.5,
		3.14159265,
		-2.7182818,
		100.25,
		-65504.0,
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
	}

	for _, val := range testValues {
		bf := FP32ToBF16(val)
		roundtrip := BF16ToFP32(bf)

		if math.IsInf(float64(val), 1) {
			if !math.IsInf(float64(roundtrip), 1) {
				t.Errorf("expected +Inf, got %f", roundtrip)
			}
			continue
		}
		if math.IsInf(float64(val), -1) {
			if !math.IsInf(float64(roundtrip), -1) {
				t.Errorf("expected -Inf, got %f", roundtrip)
			}
			continue
		}

		// Normal relative tolerance for 7-bit mantissa BF16 (~1/128 ~= 0.0078)
		relDiff := math.Abs(float64(val-roundtrip)) / (math.Abs(float64(val)) + 1e-6)
		if relDiff > 0.015 {
			t.Errorf("val %f roundtripped to %f (relDiff %.6f > 0.015)", val, roundtrip, relDiff)
		}
	}

	// Test NaN preservation
	nanVal := float32(math.NaN())
	bfNaN := FP32ToBF16(nanVal)
	rtNaN := BF16ToFP32(bfNaN)
	if !math.IsNaN(float64(rtNaN)) {
		t.Errorf("expected NaN preservation, got %f", rtNaN)
	}

	// Test slice conversions
	f32Slice := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	bfSlice := FP32SliceToBF16(f32Slice)
	if len(bfSlice) != len(f32Slice) {
		t.Fatalf("expected len %d, got %d", len(f32Slice), len(bfSlice))
	}
	backF32 := BF16SliceToFP32(bfSlice)
	sim, err := CosineSimilarity(f32Slice, backF32)
	if err != nil {
		t.Fatalf("cosine error: %v", err)
	}
	if sim < 0.9999 {
		t.Errorf("slice roundtrip cosine similarity %.6f < 0.9999", sim)
	}
}

func TestComputeHAL_GEMM_NumericalParity(t *testing.T) {
	// Deterministic seed for reproducible matrix verification
	rng := rand.New(rand.NewSource(42))

	M, N, K := 64, 64, 128
	A := make([]float32, M*K)
	B := make([]float32, K*N)

	for i := range A {
		A[i] = (rng.Float32() - 0.5) * 2.0
	}
	for i := range B {
		B[i] = (rng.Float32() - 0.5) * 2.0
	}

	// Reference FP32 computation
	refOut, err := ReferenceGEMM(M, N, K, 1.0, A, B, 0.0, nil)
	if err != nil {
		t.Fatalf("reference GEMM failed: %v", err)
	}

	// Convert inputs to BF16 as fed to RDNA 3.5 WMMA cores
	A_bf16 := FP32SliceToBF16(A)
	B_bf16 := FP32SliceToBF16(B)

	// Simulated RDNA 3.5 WMMA GEMM
	wmmaOut, err := RDNA35_WMMA_GEMM(M, N, K, 1.0, A_bf16, B_bf16, 0.0, nil)
	if err != nil {
		t.Fatalf("RDNA 3.5 WMMA GEMM failed: %v", err)
	}

	// Verify numerical parity: Cosine similarity MUST be >= 0.9999
	sim, ok, err := VerifyNumericalParity(refOut, wmmaOut, 0.9999)
	if err != nil || !ok {
		t.Fatalf("GEMM numerical parity failed: sim=%.6f, err=%v", sim, err)
	}

	if sim < 0.9999 {
		t.Errorf("GEMM cosine similarity %.6f is strictly below 0.9999 threshold", sim)
	}
}

func TestComputeHAL_GEMV_NumericalParity(t *testing.T) {
	rng := rand.New(rand.NewSource(101))

	M, K := 128, 256
	A := make([]float32, M*K)
	x := make([]float32, K)

	for i := range A {
		A[i] = (rng.Float32() - 0.5) * 1.5
	}
	for i := range x {
		x[i] = (rng.Float32() - 0.5) * 1.5
	}

	refY, err := ReferenceGEMV(M, K, A, x)
	if err != nil {
		t.Fatalf("reference GEMV failed: %v", err)
	}

	A_bf16 := FP32SliceToBF16(A)
	rdnaY, err := RDNA35_GEMV(M, K, A_bf16, x)
	if err != nil {
		t.Fatalf("RDNA 3.5 GEMV failed: %v", err)
	}

	sim, ok, err := VerifyNumericalParity(refY, rdnaY, 0.9999)
	if err != nil || !ok {
		t.Fatalf("GEMV numerical parity failed: sim=%.6f, err=%v", sim, err)
	}

	if sim < 0.9999 {
		t.Errorf("GEMV cosine similarity %.6f is strictly below 0.9999 threshold", sim)
	}
}

func TestComputeHAL_RooflineAndStreamingAnalytics(t *testing.T) {
	target := NewGFX1151Config()

	// 1. Model 70B with 4-bit weights:
	// Active weights = 70B * 0.5 bytes = 35 GB
	// At 256 GB/s bus ceiling: 256 / 35 ~= 7.31 tok/s
	report70B := CalculateStreamingRoofline(target, 70_000_000_000, 4.0, 1)
	if !report70B.IsMemoryBandwidthBound {
		t.Errorf("expected 70B single-user decode to be strictly memory-bandwidth bound")
	}
	expectedTokSec70B := 256.0 * 1e9 / float64(report70B.ActiveWeightBytes)
	if math.Abs(report70B.TheoreticalDecodeTokSec-expectedTokSec70B) > 0.05 {
		t.Errorf("expected ~%.2f tok/s for 70B, got %.2f", expectedTokSec70B, report70B.TheoreticalDecodeTokSec)
	}
	if report70B.TheoreticalDecodeTokSec >= 10.0 {
		t.Errorf("70B decode (%.2f tok/s) should not exceed 10 tok/s on 256 GB/s bus", report70B.TheoreticalDecodeTokSec)
	}

	// 2. Model 8B with 4-bit weights:
	// Active weights = 8B * 0.5 bytes = 4 GB
	// At 256 GB/s: 256 / 4 = 64 tok/s
	report8B := CalculateStreamingRoofline(target, 8_000_000_000, 4.0, 1)
	if report8B.TheoreticalDecodeTokSec < 60.0 || report8B.TheoreticalDecodeTokSec > 68.0 {
		t.Errorf("expected ~64 tok/s for 8B, got %.2f", report8B.TheoreticalDecodeTokSec)
	}

	// 3. Model 32B with 4-bit weights:
	// Active weights = 32B * 0.5 bytes = 16 GB
	// At 256 GB/s: 256 / 16 = 16 tok/s
	report32B := CalculateStreamingRoofline(target, 32_000_000_000, 4.0, 1)
	if report32B.TheoreticalDecodeTokSec < 15.0 || report32B.TheoreticalDecodeTokSec > 17.0 {
		t.Errorf("expected ~16 tok/s for 32B, got %.2f", report32B.TheoreticalDecodeTokSec)
	}

	// 4. Test MeasureStreamingRate normal case: 200 GB/s over 5ms (1 GB transferred)
	achieved, sat, err := MeasureStreamingRate(1_000_000_000, 5*time.Millisecond) // 200 GB/s
	if err != nil {
		t.Fatalf("unexpected error on valid streaming rate: %v", err)
	}
	if math.Abs(achieved-200.0) > 1.0 {
		t.Errorf("expected ~200 GB/s, got %.2f", achieved)
	}
	if math.Abs(sat-78.125) > 1.0 {
		t.Errorf("expected ~78.1%% saturation, got %.2f", sat)
	}

	// 5. Test MeasureStreamingRate exceeding 256 GB/s ceiling (300 GB/s)
	_, _, err = MeasureStreamingRate(1_500_000_000, 5*time.Millisecond) // 300 GB/s
	if !errors.Is(err, ErrBusSaturation) {
		t.Errorf("expected ErrBusSaturation when exceeding 256 GB/s ceiling, got %v", err)
	}
}
