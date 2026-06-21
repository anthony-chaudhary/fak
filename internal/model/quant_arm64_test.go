//go:build arm64

package model

import (
	"math"
	"testing"
)

// TestQdot8NEONMatchesScalar pins the arm64 NEON SDOT kernel to the scalar reference
// bit-for-bit (Float32bits equality) — the asm correctness anchor for the quantized decode
// path on Apple Silicon, the arm64 twin of TestQdot8KernelsMatchScalar. Integer block sums
// are associative with no overflow, so SDOT computes the same int32 isum, and the per-block
// float combine matches qdot8scalar's order exactly. Skips on an arm64 part without
// FEAT_DotProd (where the dispatcher uses scalar and the kernel must not be invoked).
func TestQdot8NEONMatchesScalar(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON kernel inactive, scalar path only")
	}
	dims := []int{32, 64, 192, 576, 1536}
	for _, in := range dims {
		for trial := 0; trial < 16; trial++ {
			w := mkVec(in, uint64(in*15485863+trial*2654435761+1))
			x := mkVec(in, uint64(in*40503+trial*2246822519+7))
			qt := quantizeQ8(w, 1, in)
			qv := quantizeVecQ8(x)
			got := qdot8asm(&qt.q[0], &qv.q[0], &qt.d[0], &qv.d[0], qt.nblk)
			want := qdot8scalar(qt.q, qt.d, qv, qt.nblk)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("in=%d trial=%d: %v (%08x) != scalar %v (%08x) — not bit-identical",
					in, trial, got, math.Float32bits(got), want, math.Float32bits(want))
			}
		}
	}
	t.Logf("NEON SDOT bit-identical to scalar across in=%v (neonDot=%v)", dims, neonDot)
}
