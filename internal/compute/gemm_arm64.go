//go:build arm64

package compute

import (
	"encoding/binary"
	"os"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/compute/arm64tile"
)

func gemmTile4x4Int8NEON(a0, a1, a2, a3, b0, b1, b2, b3 *int8, c *int32, ldc, k int) {
	arm64tile.GemmTile4x4Int8NEON(a0, a1, a2, a3, b0, b1, b2, b3, c, ldc, k)
}

func gemmTile4x4FP16NEON(a0, a1, a2, a3, b0, b1, b2, b3 *uint16, c *float32, ldc, k int) {
	arm64tile.GemmTile4x4FP16NEON(a0, a1, a2, a3, b0, b1, b2, b3, c, ldc, k)
}

// gemm_arm64.go — ARM64 NEON register-blocked microkernels and dispatch for int8 and fp16 GEMM.
// Addresses issue #12242: closes the prefill throughput gap on Apple Silicon (M-series)
// vs llama.cpp via 4x4 register accumulator blocking and interleaved NEON loads.
//
// Hardware capability detection:
//   - FEAT_DotProd (SDOT): unconditionally supported on all Apple Silicon (M1+) and ARMv8.4+.
//     Detected via AT_HWCAP on Linux; falls back to scalar/un-tiled Go loop when missing.
//   - FEAT_FP16 (Half-precision arithmetic): unconditionally supported on Apple Silicon (M1+)
//     and ARMv8.2+. Detected via AT_HWCAP on Linux; falls back to generic Go when missing.

// Functions implemented in internal/compute/arm64tile

var (
	hasARM64DotProdCached = detectARM64DotProd()
	hasARM64FP16Cached    = detectARM64FP16()
)

// HasARM64DotProd reports whether the host CPU supports ARM64 NEON SDOT (FEAT_DotProd).
func HasARM64DotProd() bool {
	if os.Getenv("FAK_GEMM_FORCE_GENERIC") == "1" {
		return false
	}
	if env := os.Getenv("FAK_ARM_DOTPROD"); env != "" {
		return env == "1" || env == "true"
	}
	return hasARM64DotProdCached
}

// HasARM64FP16 reports whether the host CPU supports ARM64 NEON half-precision float arithmetic (FEAT_FP16).
func HasARM64FP16() bool {
	if os.Getenv("FAK_GEMM_FORCE_GENERIC") == "1" {
		return false
	}
	if env := os.Getenv("FAK_ARM_FP16"); env != "" {
		return env == "1" || env == "true"
	}
	return hasARM64FP16Cached
}

func detectARM64DotProd() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		return true // All Apple Silicon (M1+) has FEAT_DotProd
	case "linux", "android":
		return readLinuxAuxvBit(16, 1<<20) // AT_HWCAP: HWCAP_ASIMDDP = 1<<20
	default:
		return false
	}
}

func detectARM64FP16() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		return true // All Apple Silicon (M1+) has FEAT_FP16
	case "linux", "android":
		// AT_HWCAP: HWCAP_FPHP (1<<9) or HWCAP_ASIMDHP (1<<10)
		return readLinuxAuxvBit(16, (1<<9)|(1<<10))
	default:
		return false
	}
}

func readLinuxAuxvBit(entry uint64, mask uint64) bool {
	b, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	for i := 0; i+16 <= len(b); i += 16 {
		typ := binary.LittleEndian.Uint64(b[i:])
		val := binary.LittleEndian.Uint64(b[i+8:])
		if typ == entry {
			return val&mask != 0
		}
		if typ == 0 {
			break
		}
	}
	return false
}

// GEMMInt8 computes standard matrix multiplication C = A * B:
//
//	A is [M, K] row-major
//	B is [K, N] row-major
//	C is [M, N] row-major
//
// Uses 4x4 register-blocked NEON SDOT microkernel when available; falls back to generic Go otherwise.
func GEMMInt8(A, B []int8, C []int32, M, N, K int) {
	if len(A) < M*K || len(B) < K*N || len(C) < M*N {
		panic("compute: GEMMInt8 buffer size mismatch")
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return
	}
	if !HasARM64DotProd() {
		ReferenceGEMMInt8(A, B, C, M, N, K)
		return
	}

	Mmain := M &^ 3
	Nmain := N &^ 3
	Kmain := K &^ 15

	// Pack 4-column panels of B into a local 4*K row buffer for streaming NEON loads
	bPack := make([]int8, 4*K)

	for j0 := 0; j0 < Nmain; j0 += 4 {
		for k := 0; k < K; k++ {
			bRow := k * N
			bPack[0*K+k] = B[bRow+j0+0]
			bPack[1*K+k] = B[bRow+j0+1]
			bPack[2*K+k] = B[bRow+j0+2]
			bPack[3*K+k] = B[bRow+j0+3]
		}

		for i := 0; i < Mmain; i += 4 {
			gemmTile4x4Int8NEON(
				&A[(i+0)*K], &A[(i+1)*K], &A[(i+2)*K], &A[(i+3)*K],
				&bPack[0*K], &bPack[1*K], &bPack[2*K], &bPack[3*K],
				&C[i*N+j0], N, Kmain,
			)
			if Kmain < K {
				for remK := Kmain; remK < K; remK++ {
					for r := 0; r < 4; r++ {
						aRow := int32(A[(i+r)*K+remK])
						for c := 0; c < 4; c++ {
							C[(i+r)*N+(j0+c)] += aRow * int32(bPack[c*K+remK])
						}
					}
				}
			}
		}
	}

	// Remainder columns (j in [Nmain, N))
	if Nmain < N {
		for i := 0; i < M; i++ {
			for j := Nmain; j < N; j++ {
				var sum int32
				for k := 0; k < K; k++ {
					sum += int32(A[i*K+k]) * int32(B[k*N+j])
				}
				C[i*N+j] = sum
			}
		}
	}

	// Remainder rows (i in [Mmain, M)) for processed columns [0, Nmain)
	if Mmain < M {
		for i := Mmain; i < M; i++ {
			for j := 0; j < Nmain; j++ {
				var sum int32
				for k := 0; k < K; k++ {
					sum += int32(A[i*K+k]) * int32(B[k*N+j])
				}
				C[i*N+j] = sum
			}
		}
	}
}

// GEMMInt8TN computes matrix multiplication with transposed weight C = A * B^T:
//
//	A is [M, K] row-major (e.g. activations X [P, in])
//	B is [N, K] row-major (e.g. weights W [out, in])
//	C is [M, N] row-major (e.g. Y [P, out])
//
// Both A and B have contiguous rows of length K, requiring zero repacking.
func GEMMInt8TN(A, B []int8, C []int32, M, N, K int) {
	if len(A) < M*K || len(B) < N*K || len(C) < M*N {
		panic("compute: GEMMInt8TN buffer size mismatch")
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return
	}
	if !HasARM64DotProd() {
		ReferenceGEMMInt8TN(A, B, C, M, N, K)
		return
	}

	Mmain := M &^ 3
	Nmain := N &^ 3
	Kmain := K &^ 15

	for i := 0; i < Mmain; i += 4 {
		for j := 0; j < Nmain; j += 4 {
			gemmTile4x4Int8NEON(
				&A[(i+0)*K], &A[(i+1)*K], &A[(i+2)*K], &A[(i+3)*K],
				&B[(j+0)*K], &B[(j+1)*K], &B[(j+2)*K], &B[(j+3)*K],
				&C[i*N+j], N, Kmain,
			)
			if Kmain < K {
				for remK := Kmain; remK < K; remK++ {
					for r := 0; r < 4; r++ {
						aRow := int32(A[(i+r)*K+remK])
						for c := 0; c < 4; c++ {
							C[(i+r)*N+(j+c)] += aRow * int32(B[(j+c)*K+remK])
						}
					}
				}
			}
		}
	}

	if Nmain < N {
		for i := 0; i < M; i++ {
			for j := Nmain; j < N; j++ {
				var sum int32
				for k := 0; k < K; k++ {
					sum += int32(A[i*K+k]) * int32(B[j*K+k])
				}
				C[i*N+j] = sum
			}
		}
	}

	if Mmain < M {
		for i := Mmain; i < M; i++ {
			for j := 0; j < Nmain; j++ {
				var sum int32
				for k := 0; k < K; k++ {
					sum += int32(A[i*K+k]) * int32(B[j*K+k])
				}
				C[i*N+j] = sum
			}
		}
	}
}

// PrefillGEMMInt8 computes prefill matrix multiplication Y = X * W^T:
//
//	X: [P, in] activations
//	W: [out, in] weights
//	Y: [P, out] outputs
func PrefillGEMMInt8(X, W []int8, Y []int32, P, out, in int) {
	GEMMInt8TN(X, W, Y, P, out, in)
}

// GEMMFP16 computes standard matrix multiplication C = A * B in fp16 half-precision:
//
//	A is [M, K] uint16 (IEEE 754 float16 bits)
//	B is [K, N] uint16 (IEEE 754 float16 bits)
//	C is [M, N] float32 output
func GEMMFP16(A, B []uint16, C []float32, M, N, K int) {
	if len(A) < M*K || len(B) < K*N || len(C) < M*N {
		panic("compute: GEMMFP16 buffer size mismatch")
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return
	}
	if !HasARM64FP16() {
		ReferenceGEMMFP16(A, B, C, M, N, K)
		return
	}

	Mmain := M &^ 3
	Nmain := N &^ 3
	Kmain := K &^ 7

	bPack := make([]uint16, 4*K)

	for j0 := 0; j0 < Nmain; j0 += 4 {
		for k := 0; k < K; k++ {
			bRow := k * N
			bPack[0*K+k] = B[bRow+j0+0]
			bPack[1*K+k] = B[bRow+j0+1]
			bPack[2*K+k] = B[bRow+j0+2]
			bPack[3*K+k] = B[bRow+j0+3]
		}

		for i := 0; i < Mmain; i += 4 {
			gemmTile4x4FP16NEON(
				&A[(i+0)*K], &A[(i+1)*K], &A[(i+2)*K], &A[(i+3)*K],
				&bPack[0*K], &bPack[1*K], &bPack[2*K], &bPack[3*K],
				&C[i*N+j0], N, Kmain,
			)
			if Kmain < K {
				for remK := Kmain; remK < K; remK++ {
					for r := 0; r < 4; r++ {
						aRow := Float16BitsToFloat32(A[(i+r)*K+remK])
						for c := 0; c < 4; c++ {
							C[(i+r)*N+(j0+c)] += aRow * Float16BitsToFloat32(bPack[c*K+remK])
						}
					}
				}
			}
		}
	}

	if Nmain < N {
		for i := 0; i < M; i++ {
			for j := Nmain; j < N; j++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += Float16BitsToFloat32(A[i*K+k]) * Float16BitsToFloat32(B[k*N+j])
				}
				C[i*N+j] = sum
			}
		}
	}

	if Mmain < M {
		for i := Mmain; i < M; i++ {
			for j := 0; j < Nmain; j++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += Float16BitsToFloat32(A[i*K+k]) * Float16BitsToFloat32(B[k*N+j])
				}
				C[i*N+j] = sum
			}
		}
	}
}

// GEMMFP16TN computes matrix multiplication with transposed weight C = A * B^T in fp16:
//
//	A is [M, K] uint16 (IEEE 754 float16 bits)
//	B is [N, K] uint16 (IEEE 754 float16 bits)
//	C is [M, N] float32 output
func GEMMFP16TN(A, B []uint16, C []float32, M, N, K int) {
	if len(A) < M*K || len(B) < N*K || len(C) < M*N {
		panic("compute: GEMMFP16TN buffer size mismatch")
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return
	}
	if !HasARM64FP16() {
		ReferenceGEMMFP16TN(A, B, C, M, N, K)
		return
	}

	Mmain := M &^ 3
	Nmain := N &^ 3
	Kmain := K &^ 7

	for i := 0; i < Mmain; i += 4 {
		for j := 0; j < Nmain; j += 4 {
			gemmTile4x4FP16NEON(
				&A[(i+0)*K], &A[(i+1)*K], &A[(i+2)*K], &A[(i+3)*K],
				&B[(j+0)*K], &B[(j+1)*K], &B[(j+2)*K], &B[(j+3)*K],
				&C[i*N+j], N, Kmain,
			)
			if Kmain < K {
				for remK := Kmain; remK < K; remK++ {
					for r := 0; r < 4; r++ {
						aRow := Float16BitsToFloat32(A[(i+r)*K+remK])
						for c := 0; c < 4; c++ {
							C[(i+r)*N+(j+c)] += aRow * Float16BitsToFloat32(B[(j+c)*K+remK])
						}
					}
				}
			}
		}
	}

	if Nmain < N {
		for i := 0; i < M; i++ {
			for j := Nmain; j < N; j++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += Float16BitsToFloat32(A[i*K+k]) * Float16BitsToFloat32(B[j*K+k])
				}
				C[i*N+j] = sum
			}
		}
	}

	if Mmain < M {
		for i := Mmain; i < M; i++ {
			for j := 0; j < Nmain; j++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += Float16BitsToFloat32(A[i*K+k]) * Float16BitsToFloat32(B[j*K+k])
				}
				C[i*N+j] = sum
			}
		}
	}
}

// GEMMFP16F32 computes matrix multiplication C = A * B in fp16 using float32 slices:
// Converts inputs to fp16 halfwords and runs the NEON microkernel.
func GEMMFP16F32(A, B []float32, C []float32, M, N, K int) {
	aF16 := make([]uint16, M*K)
	for i, v := range A[:M*K] {
		aF16[i] = Float32ToFloat16Bits(v)
	}
	bF16 := make([]uint16, K*N)
	for i, v := range B[:K*N] {
		bF16[i] = Float32ToFloat16Bits(v)
	}
	GEMMFP16(aF16, bF16, C, M, N, K)
}

// PrefillGEMMFP16 computes prefill matrix multiplication Y = X * W^T:
//
//	X: [P, in] activations
//	W: [out, in] weights
//	Returns Y: [P, out]
func PrefillGEMMFP16(X, W []float32, out, in, P int) []float32 {
	xF16 := make([]uint16, P*in)
	for i, v := range X[:P*in] {
		xF16[i] = Float32ToFloat16Bits(v)
	}
	wF16 := make([]uint16, out*in)
	for i, v := range W[:out*in] {
		wF16[i] = Float32ToFloat16Bits(v)
	}
	Y := make([]float32, P*out)
	GEMMFP16TN(xF16, wF16, Y, P, out, in)
	return Y
}

// Reference scalar implementations for parity verification:

// ReferenceGEMMInt8 computes exact C = A * B using un-tiled scalar arithmetic.
func ReferenceGEMMInt8(A, B []int8, C []int32, M, N, K int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(A[i*K+k]) * int32(B[k*N+j])
			}
			C[i*N+j] = sum
		}
	}
}

// ReferenceGEMMInt8TN computes exact C = A * B^T using un-tiled scalar arithmetic.
func ReferenceGEMMInt8TN(A, B []int8, C []int32, M, N, K int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(A[i*K+k]) * int32(B[j*K+k])
			}
			C[i*N+j] = sum
		}
	}
}

// ReferenceGEMMFP16 computes C = A * B using un-tiled float arithmetic.
func ReferenceGEMMFP16(A, B []uint16, C []float32, M, N, K int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += Float16BitsToFloat32(A[i*K+k]) * Float16BitsToFloat32(B[k*N+j])
			}
			C[i*N+j] = sum
		}
	}
}

// ReferenceGEMMFP16TN computes C = A * B^T using un-tiled float arithmetic.
func ReferenceGEMMFP16TN(A, B []uint16, C []float32, M, N, K int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += Float16BitsToFloat32(A[i*K+k]) * Float16BitsToFloat32(B[j*K+k])
			}
			C[i*N+j] = sum
		}
	}
}
