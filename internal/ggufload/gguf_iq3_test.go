package ggufload

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestDequantIQ3XXSGolden pins dequantIQ3XXS to a hand-computed anchor: one 256-element super-block
// with d=1.0, all grid-index bytes 0 (grid[0]=0x04040404 → magnitudes 4,4,4,4) and all scale/sign
// bytes 0 (scale nibble 0 → db = 1.0*(0.5+0)*0.5 = 0.25; sign selector 0 → ksignsIQ2XS[0]=0, all
// positive). Every output = 0.25*4 = 1.0. This catches the two easy ports-wrong: the *0.5 (not *0.25)
// scale factor and the uint32-grid→4-little-endian-bytes reinterpret.
func TestDequantIQ3XXSGolden(t *testing.T) {
	raw := make([]byte, blockIQ3XXSBytes)
	binary.LittleEndian.PutUint16(raw[0:], 0x3C00) // f16 1.0
	// grid-index bytes [2:66) and scale/sign bytes [66:98) all zero already.
	out := make([]float32, qkK)
	dequantIQ3XXS(out, raw)
	for i, v := range out {
		if math.Abs(float64(v)-1.0) > 1e-6 {
			t.Fatalf("out[%d] = %v, want 1.0 (grid0 mag 4 × db 0.25)", i, v)
		}
	}
}

// TestDequantIQ3XXSSignAndScale exercises a non-trivial scale nibble and a sign selector so the
// db formula and the sign-mask indexing are both checked, not just the all-zero degenerate case.
func TestDequantIQ3XXSSignAndScale(t *testing.T) {
	raw := make([]byte, blockIQ3XXSBytes)
	binary.LittleEndian.PutUint16(raw[0:], 0x3C00) // d = 1.0
	// sub-block 0: aux32 with scale nibble = 1 (top 4 bits) and sign selector for l=0 = 1 (bit0).
	// db = 1.0*(0.5+1)*0.5 = 0.75. selector 1 → ksignsIQ2XS[1]=129=0b10000001 → bit0 set (output 0
	// of the pair negated), bit7 set (parity, affects output 7 of the l-group = grid2[3]).
	var aux uint32 = (uint32(1) << 28) | 1
	binary.LittleEndian.PutUint32(raw[2+qkK/4:], aux)
	out := make([]float32, qkK)
	dequantIQ3XXS(out, raw)
	// grid index 0 → 0x04040404 → all magnitudes 4. db=0.75 → |val|=3.0.
	// l=0: output 0 negated (bit0), output 7 negated (bit7); outputs 1..6 positive.
	if math.Abs(float64(out[0])-(-3.0)) > 1e-6 {
		t.Fatalf("out[0] = %v, want -3.0 (sign bit0)", out[0])
	}
	for j := 1; j <= 6; j++ {
		if math.Abs(float64(out[j])-3.0) > 1e-6 {
			t.Fatalf("out[%d] = %v, want 3.0", j, out[j])
		}
	}
	if math.Abs(float64(out[7])-(-3.0)) > 1e-6 {
		t.Fatalf("out[7] = %v, want -3.0 (sign bit7)", out[7])
	}
	// other sub-blocks (aux32=0) → db=0.25, all positive → 1.0
	if math.Abs(float64(out[32])-1.0) > 1e-6 {
		t.Fatalf("out[32] = %v, want 1.0 (sub-block 1, db 0.25)", out[32])
	}
}
