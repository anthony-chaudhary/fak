//go:build amd64

package model

import (
	"math"
	"testing"
)

// TestQdot8KernelsMatchScalar pins BOTH amd64 SIMD kernels (AVX2 and AVX-512BW) to the
// scalar reference bit-for-bit, directly — independent of which tier qtier resolved to
// for this run. Skips a kernel the hardware can't run. This is the asm correctness anchor:
// integer block sums are associative with no overflow, so every kernel computes the same
// int32 isum and the identical per-block float combine, hence Float32bits equality.
func TestQdot8KernelsMatchScalar(t *testing.T) {
	dims := []int{32, 64, 192, 576, 1536}
	check := func(name string, fn func(qw, qx *int8, dw, dx *float32, nblk int) float32) {
		for _, in := range dims {
			for trial := 0; trial < 16; trial++ {
				w := mkVec(in, uint64(in*15485863+trial*2654435761+1))
				x := mkVec(in, uint64(in*40503+trial*2246822519+7))
				qt := quantizeQ8(w, 1, in)
				qv := quantizeVecQ8(x)
				got := fn(&qt.q[0], &qv.q[0], &qt.d[0], &qv.d[0], qt.nblk)
				want := qdot8scalar(qt.q, qt.d, qv, qt.nblk)
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("%s in=%d trial=%d: %v != scalar %v (not bit-identical)", name, in, trial, got, want)
				}
			}
		}
	}
	if hasAVX2 := detectAVX2(); hasAVX2 {
		check("avx2", qdot8asm)
	} else {
		t.Log("AVX2 not available — skipping avx2 kernel pin")
	}
	if hasAVX512 := detectAVX512(); hasAVX512 {
		check("avx512", qdot8asm512)
	} else {
		t.Log("AVX-512 not available — skipping avx512 kernel pin")
	}
	t.Logf("active qtier=%d (0=scalar 1=avx2 2=avx512)", qtier)
}
