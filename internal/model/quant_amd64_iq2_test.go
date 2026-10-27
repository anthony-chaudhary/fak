//go:build amd64

package model

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// fak-test:runtime fast est=150ms
func TestIQ2AVX2FamilyMatchesScalarAndOrdinaryPrefill(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved CPU tier does not admit AVX2")
	}
	original := iq2DequantSuperBlockArch
	if original == nil {
		t.Fatal("amd64 IQ2 dequant hook is not registered")
	}
	t.Cleanup(func() { iq2DequantSuperBlockArch = original })
	rng := rand.New(rand.NewSource(11221))
	for _, tc := range []struct {
		kind   kQuantKind
		scalar func([]float32, []byte)
	}{{kindIQ2XXS, dequantIQ2XXSScalar}, {kindIQ2XS, dequantIQ2XSScalar}, {kindIQ2S, dequantIQ2SScalar}} {
		for _, half := range []uint16{0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff} {
			blk := make([]byte, tc.kind.blockBytes())
			rng.Read(blk)
			binary.LittleEndian.PutUint16(blk, half)
			before := bytes.Clone(blk)
			var storage [qkK + 2]float32
			storage[0], storage[qkK+1] = 12345, -6789
			got := storage[1 : qkK+1]
			want := make([]float32, qkK)
			tc.scalar(want, blk)
			if !iq2DequantSuperBlockAVX2(got, blk, tc.kind) {
				t.Fatalf("%s half%04x unexpectedly declined", tc.kind, half)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%s half%04x lane%d=%08x want%08x", tc.kind, half, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			if storage[0] != 12345 || storage[qkK+1] != -6789 || !bytes.Equal(blk, before) {
				t.Fatal("SIMD block changed input or escaped destination")
			}
		}
		const out, in, P = 5, 512, 3
		raw := make([]byte, out*(in/qkK)*tc.kind.blockBytes())
		rng.Read(raw)
		for off := 0; off < len(raw); off += tc.kind.blockBytes() {
			binary.LittleEndian.PutUint16(raw[off:], 0x3400)
		}
		qt := quantizeKQuantFromRaw(raw, out, in, tc.kind)
		X := mkVec(P*in, uint64(tc.kind)+3300)
		beforeX, beforeRaw := slices.Clone(X), bytes.Clone(raw)
		want, got := make([]float32, P*out), make([]float32, P*out)
		iq2DequantSuperBlockArch = nil // exact pre-change ordinary prefill reference
		kQuantMatRowsIntoBatch(qt, X, P, want)
		iq2DequantSuperBlockArch = original
		kQuantMatRowsIntoBatch(qt, X, P, got)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s prefill cell%d=%g want%g", tc.kind, i, got[i], want[i])
			}
		}
		if !slices.Equal(X, beforeX) || !bytes.Equal(raw, beforeRaw) {
			t.Fatal("ordinary prefill changed input ownership")
		}
		// Nonfinite scales keep the scalar decoder's exact NaN/Inf behavior.
		for _, half := range []uint16{0x7c00, 0xfc00, 0x7e00, 0xfe01} {
			blk := raw[:tc.kind.blockBytes()]
			binary.LittleEndian.PutUint16(blk, half)
			dst := make([]float32, qkK)
			for i := range dst {
				dst[i] = 123
			}
			if iq2DequantSuperBlockAVX2(dst, blk, tc.kind) {
				t.Fatal("nonfinite scale must decline")
			}
			for _, v := range dst {
				if v != 123 {
					t.Fatal("declined block changed destination")
				}
			}
		}
	}
}
