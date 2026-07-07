//go:build amd64

package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestIQ3XXSDequantAsmMatchesScalar pins the AVX2 IQ3_XXS decode-dequant kernel
// (iq3xxsDequantSuperBlockArch) to the scalar reference (iq3xxsDequantSuperBlock), bit-for-bit,
// across many super-blocks whose qs/sign/scale bytes sweep all 256 grid entries, all 128 sign
// selectors, and all 16 db exponents. The AVX2 path flips signs by XORing the IEEE sign bit
// while the scalar path multiplies by ±1 — identical for finite floats — so any deviation is a
// real grid-index, sign-lane, or byte-packing bug in the asm. Skips on a CPU without AVX2.
func TestIQ3XXSDequantAsmMatchesScalar(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("AVX2 not available — iq3xxs asm inactive")
	}
	const nblk = 64 // sweeps grid/sign/scale coverage many times over
	raw := make([]byte, nblk*iq3xxsBlockBytes)
	lcgBytes(raw, 0xC0FFEE1234567890) // varied qs/sas bytes exercise every grid entry + selector
	for b := 0; b < nblk; b++ {
		binary.LittleEndian.PutUint16(raw[b*iq3xxsBlockBytes:], f16One) // d=1.0 (finite, clean db range)
	}

	scalar := make([]float32, qkK)
	arch := make([]float32, qkK)
	for b := 0; b < nblk; b++ {
		blk := raw[b*iq3xxsBlockBytes : (b+1)*iq3xxsBlockBytes]
		iq3xxsDequantSuperBlock(scalar, blk)
		for i := range arch {
			arch[i] = 0
		}
		if !iq3xxsDequantSuperBlockArch(arch, blk) {
			t.Fatalf("block %d: iq3xxsDequantSuperBlockArch returned false with AVX2 tier=%d", b, qtier)
		}
		for i := range scalar {
			if math.Float32bits(arch[i]) != math.Float32bits(scalar[i]) {
				t.Fatalf("block %d lane %d: asm=%v (%#08x) scalar=%v (%#08x)",
					b, i, arch[i], math.Float32bits(arch[i]), scalar[i], math.Float32bits(scalar[i]))
			}
		}
	}
	t.Logf("iq3xxs AVX2 dequant bit-identical to scalar across %d super-blocks x %d lanes (tier=%d)", nblk, qkK, qtier)
}
