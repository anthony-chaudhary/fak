//go:build arm64

package compute

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// gemm_test.go — parity verification and microbenchmarks for ARM64 NEON register-blocked GEMM.
// Validates issue #12242:
//   1. Bit-exact integer parity for INT8 GEMM against reference math.
//   2. Max numeric error <= 1e-4 for FP16 GEMM against float math.
//   3. At least 2.5x speedup over un-tiled Go loop on Apple Silicon.
//   4. Clean CPU feature detection and fallback to generic Go.

func TestARM64RegisterBlockedGEMMParity(t *testing.T) {
	if !HasARM64DotProd() {
		t.Skip("skipping ARM64 NEON SDOT tests: FEAT_DotProd not available")
	}

	t.Run("Int8Parity", func(t *testing.T) {
		testCases := []struct {
			name    string
			M, N, K int
		}{
			{name: "Tile4x4x16", M: 4, N: 4, K: 16},
			{name: "Tile16x16x64", M: 16, N: 16, K: 64},
			{name: "Square64x64x128", M: 64, N: 64, K: 128},
			{name: "Prefill128x128x256", M: 128, N: 128, K: 256},
			{name: "Prefill256x256x512", M: 256, N: 256, K: 512},
			{name: "Odd1x1x1", M: 1, N: 1, K: 1},
			{name: "Odd5x7x19", M: 5, N: 7, K: 19},
			{name: "Odd13x17x35", M: 13, N: 17, K: 35},
			{name: "Odd33x41x65", M: 33, N: 41, K: 65},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				rng := rand.New(rand.NewSource(12345 + int64(tc.M*tc.N+tc.K)))
				M, N, K := tc.M, tc.N, tc.K

				A := make([]int8, M*K)
				for i := range A {
					A[i] = int8(rng.Intn(255) - 128)
				}

				// 1. Standard layout: B is [K, N]
				Bstd := make([]int8, K*N)
				for i := range Bstd {
					Bstd[i] = int8(rng.Intn(255) - 128)
				}
				Cgot := make([]int32, M*N)
				Cwant := make([]int32, M*N)

				GEMMInt8(A, Bstd, Cgot, M, N, K)
				ReferenceGEMMInt8(A, Bstd, Cwant, M, N, K)

				for i := 0; i < M*N; i++ {
					if Cgot[i] != Cwant[i] {
						t.Fatalf("Standard GEMMInt8 mismatch at index %d: got %d, want %d", i, Cgot[i], Cwant[i])
					}
				}

				// 2. Transposed layout (prefill weight): B is [N, K]
				Btn := make([]int8, N*K)
				for i := range Btn {
					Btn[i] = int8(rng.Intn(255) - 128)
				}
				CgotTN := make([]int32, M*N)
				CwantTN := make([]int32, M*N)

				GEMMInt8TN(A, Btn, CgotTN, M, N, K)
				ReferenceGEMMInt8TN(A, Btn, CwantTN, M, N, K)

				for i := 0; i < M*N; i++ {
					if CgotTN[i] != CwantTN[i] {
						t.Fatalf("GEMMInt8TN mismatch at index %d: got %d, want %d", i, CgotTN[i], CwantTN[i])
					}
				}

				// 3. PrefillGEMMInt8 helper check
				Cprefill := make([]int32, M*N)
				PrefillGEMMInt8(A, Btn, Cprefill, M, N, K)
				for i := 0; i < M*N; i++ {
					if Cprefill[i] != CwantTN[i] {
						t.Fatalf("PrefillGEMMInt8 mismatch at index %d: got %d, want %d", i, Cprefill[i], CwantTN[i])
					}
				}
			})
		}
	})

	t.Run("FP16Parity", func(t *testing.T) {
		if !HasARM64FP16() {
			t.Skip("skipping ARM64 NEON FP16 tests: FEAT_FP16 not available")
		}

		testCases := []struct {
			name    string
			M, N, K int
		}{
			{name: "Tile4x4x8", M: 4, N: 4, K: 8},
			{name: "Tile16x16x32", M: 16, N: 16, K: 32},
			{name: "Square64x64x64", M: 64, N: 64, K: 64},
			{name: "Prefill128x128x128", M: 128, N: 128, K: 128},
			{name: "Odd1x1x1", M: 1, N: 1, K: 1},
			{name: "Odd5x7x13", M: 5, N: 7, K: 13},
			{name: "Odd13x17x27", M: 13, N: 17, K: 27},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				rng := rand.New(rand.NewSource(54321 + int64(tc.M*tc.N+tc.K)))
				M, N, K := tc.M, tc.N, tc.K

				scale := float32(1.0 / math.Sqrt(float64(tc.K)))
				A := make([]uint16, M*K)
				for i := range A {
					val := (float32(rng.Intn(200)-100) / 1000.0) * scale
					A[i] = Float32ToFloat16Bits(val)
				}

				// Standard layout: B is [K, N]
				Bstd := make([]uint16, K*N)
				for i := range Bstd {
					val := (float32(rng.Intn(200)-100) / 1000.0) * scale
					Bstd[i] = Float32ToFloat16Bits(val)
				}
				Cgot := make([]float32, M*N)
				Cwant := make([]float32, M*N)

				GEMMFP16(A, Bstd, Cgot, M, N, K)
				ReferenceGEMMFP16(A, Bstd, Cwant, M, N, K)

				var maxErrStd float32
				for i := 0; i < M*N; i++ {
					diff := float32(math.Abs(float64(Cgot[i] - Cwant[i])))
					if diff > maxErrStd {
						maxErrStd = diff
					}
					if diff > 1e-4 {
						t.Fatalf("GEMMFP16 error exceeded 1e-4 at index %d: got %f, want %f, diff %e",
							i, Cgot[i], Cwant[i], diff)
					}
				}

				// Transposed layout (prefill weight): B is [N, K]
				Btn := make([]uint16, N*K)
				for i := range Btn {
					val := (float32(rng.Intn(200)-100) / 1000.0) * scale
					Btn[i] = Float32ToFloat16Bits(val)
				}
				CgotTN := make([]float32, M*N)
				CwantTN := make([]float32, M*N)

				GEMMFP16TN(A, Btn, CgotTN, M, N, K)
				ReferenceGEMMFP16TN(A, Btn, CwantTN, M, N, K)

				var maxErrTN float32
				for i := 0; i < M*N; i++ {
					diff := float32(math.Abs(float64(CgotTN[i] - CwantTN[i])))
					if diff > maxErrTN {
						maxErrTN = diff
					}
					if diff > 1e-4 {
						t.Fatalf("GEMMFP16TN error exceeded 1e-4 at index %d: got %f, want %f, diff %e",
							i, CgotTN[i], CwantTN[i], diff)
					}
				}
			})
		}
	})

	t.Run("SpeedupBenchmark", func(t *testing.T) {
		M, N, K := 128, 128, 512
		rng := rand.New(rand.NewSource(99999))
		A := make([]int8, M*K)
		B := make([]int8, N*K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 128)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 128)
		}

		// Warm up
		Cref := make([]int32, M*N)
		Cneon := make([]int32, M*N)
		ReferenceGEMMInt8TN(A, B, Cref, M, N, K)
		GEMMInt8TN(A, B, Cneon, M, N, K)

		iters := 15

		// Measure un-tiled Go reference
		t0 := time.Now()
		for it := 0; it < iters; it++ {
			ReferenceGEMMInt8TN(A, B, Cref, M, N, K)
		}
		durRef := time.Since(t0)

		// Measure register-blocked NEON tile
		t1 := time.Now()
		for it := 0; it < iters; it++ {
			GEMMInt8TN(A, B, Cneon, M, N, K)
		}
		durNeon := time.Since(t1)

		speedup := float64(durRef) / float64(durNeon)
		t.Logf("Un-tiled Go reference: %v (%v/op)", durRef, durRef/time.Duration(iters))
		t.Logf("ARM64 NEON tile:       %v (%v/op)", durNeon, durNeon/time.Duration(iters))
		t.Logf("Throughput speedup:    %.2fx (gate: >= 2.5x)", speedup)

		if speedup < 2.5 {
			t.Fatalf("speedup %.2fx did not meet the >= 2.5x acceptance threshold", speedup)
		}
	})

	t.Run("CPUFeatureDetectionAndFallback", func(t *testing.T) {
		if !HasARM64DotProd() {
			t.Fatal("HasARM64DotProd() expected true on Apple Silicon")
		}
		if !HasARM64FP16() {
			t.Fatal("HasARM64FP16() expected true on Apple Silicon")
		}

		// Test fallback path with FAK_GEMM_FORCE_GENERIC=1
		os.Setenv("FAK_GEMM_FORCE_GENERIC", "1")
		defer os.Unsetenv("FAK_GEMM_FORCE_GENERIC")

		if HasARM64DotProd() {
			t.Fatal("HasARM64DotProd() expected false when FAK_GEMM_FORCE_GENERIC=1")
		}
		if HasARM64FP16() {
			t.Fatal("HasARM64FP16() expected false when FAK_GEMM_FORCE_GENERIC=1")
		}

		// Confirm fallback executes cleanly and correctly
		M, N, K := 8, 8, 16
		A := make([]int8, M*K)
		B := make([]int8, N*K)
		for i := range A {
			A[i] = int8(i % 127)
		}
		for i := range B {
			B[i] = int8(i % 127)
		}
		Cfallback := make([]int32, M*N)
		Cref := make([]int32, M*N)

		GEMMInt8TN(A, B, Cfallback, M, N, K)
		ReferenceGEMMInt8TN(A, B, Cref, M, N, K)

		for i := range Cfallback {
			if Cfallback[i] != Cref[i] {
				t.Fatalf("Fallback GEMMInt8TN mismatch at index %d", i)
			}
		}
	})
}

// Microbenchmarks for single-stream prefill

func BenchmarkARM64RegisterBlockedGEMMInt8_NEON(b *testing.B) {
	M, N, K := 128, 128, 512
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C := make([]int32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GEMMInt8TN(A, B, C, M, N, K)
	}
}

func BenchmarkARM64RegisterBlockedGEMMInt8_Scalar(b *testing.B) {
	M, N, K := 128, 128, 512
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C := make([]int32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ReferenceGEMMInt8TN(A, B, C, M, N, K)
	}
}

func BenchmarkARM64RegisterBlockedGEMMFP16_NEON(b *testing.B) {
	M, N, K := 128, 128, 512
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	C := make([]float32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GEMMFP16TN(A, B, C, M, N, K)
	}
}

func BenchmarkARM64RegisterBlockedGEMMFP16_Scalar(b *testing.B) {
	M, N, K := 128, 128, 512
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	C := make([]float32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ReferenceGEMMFP16TN(A, B, C, M, N, K)
	}
}
