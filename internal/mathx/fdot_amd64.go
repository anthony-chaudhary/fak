//go:build amd64

package mathx

import (
	"math"
	"os"
)

// SSE2 is part of the Go amd64 baseline, not an optional AVX/OSXSAVE feature.
// The new common primitive remains opt-in under an existing explicit accelerated
// kernel pin until physical qualification: unset/scalar/unknown keep the old loop.
// No per-call environment read or second optional-ISA probe is introduced.
func init() {
	switch os.Getenv("FAK_QKERNEL") {
	case "avx2", "avx512":
		fDotArch = fDotSSE2
	}
}

// fDotLanesSSE2 writes the eight independent accumulators, without folding them.
// It reads exactly groups*8 values. The Go adapter handles final-NaN fallback
// so the hot vector loop needs no operand scan or per-vector finite checks.
//
//go:noescape
func fDotLanesSSE2(r, x *float32, groups int, lanes *float32) bool

func fDotSSE2(r, x []float32) (float32, bool) {
	n := len(r)
	if n < 8 || len(x) < n {
		return 0, false
	}
	var lanes [8]float32
	if !fDotLanesSSE2(&r[0], &x[0], n/8, &lanes[0]) {
		return 0, false
	}
	// Keep both the fixed fold and the post-fold scalar tail exactly as FDot.
	s := ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
	for i := n &^ 7; i < n; i++ {
		s += r[i] * x[i]
	}
	// Every NaN input/intermediate propagates through this multiply/add-only
	// tree. Recompute only a final NaN with the unchanged scalar expression,
	// preserving its operand/payload semantics without scanning the inputs.
	// Non-NaN infinities and signed zeros retain the same ordered arithmetic.
	if math.Float32bits(s)&0x7fffffff > 0x7f800000 {
		return 0, false
	}
	return s, true
}
