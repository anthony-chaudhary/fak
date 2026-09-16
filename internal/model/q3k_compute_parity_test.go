package model

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestQ3KComputeAndModelDequantParity is the fak#13149 reference-vector parity witness:
// the compute-package Q3_K host dequant (which the R5 streamed routed-expert tier stages a
// Q3_K slab with) must decode byte-for-byte identically to the model-package DequantQ3K
// oracle it was ported from. compute cannot import model, so this test lives in model and
// compares the two exported decoders directly.
//
// The fixture is nonzero and multi-block: 4 independent 110-byte super-blocks, a modest
// f16 super-scale and nonconstant low codes / high-mask / sub-scales, so a degenerate
// all-zero decoder cannot pass.
func TestQ3KComputeAndModelDequantParity(t *testing.T) {
	const blocks = 4
	const n = blocks * 256
	raw := make([]byte, blocks*110)
	for i := range raw {
		raw[i] = byte((i*37 + 11) & 0xff)
	}
	// Keep d finite and nonzero so the scale is exercised without inf/nan long tails.
	for b := 0; b < blocks; b++ {
		binary.LittleEndian.PutUint16(raw[b*110+108:], 0x3000)
	}

	want := make([]float32, n)
	DequantQ3K(want, raw)

	got := make([]float32, n)
	compute.DequantQ3K(got, raw)

	nonzero := 0
	for i := range want {
		if want[i] != 0 && !math.IsNaN(float64(want[i])) {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("fixture decoded to all-zero; parity test would be vacuous")
	}

	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("compute Q3_K dequant differs from model.DequantQ3K at weight %d: compute=%v model=%v", i, got[i], want[i])
		}
	}

	// The model-side resident tensor must carry the same raw bytes the tier stages.
	qt := quantizeKQuantFromRaw(append([]byte(nil), raw...), 1, n, kindQ3K)
	if qt == nil || !bytes.Equal(qt.raw, raw) {
		t.Fatal("quantizeKQuantFromRaw did not preserve the Q3_K raw bytes")
	}
	if qt.kind.blockBytes() != 110 || qt.kind.blockWeights() != 256 {
		t.Fatalf("Q3_K geometry drift: blockBytes=%d blockWeights=%d", qt.kind.blockBytes(), qt.kind.blockWeights())
	}
}
