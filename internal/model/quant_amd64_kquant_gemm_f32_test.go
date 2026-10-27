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

func assertDecodedPairBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s cell %d = %08x want %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime fast est=300ms
func TestKQuantF32PairsRetainAllDecodersAndScalarTails(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved tier does not admit AVX2")
	}
	oldHook, oldMode, oldForce := kQuantF32BlockPairArch, qgemmMode, kQuantSDOTForce
	t.Cleanup(func() { kQuantF32BlockPairArch, qgemmMode, kQuantSDOTForce = oldHook, oldMode, oldForce })
	qgemmMode, kQuantSDOTForce = qgemmModeTile, -1
	if oldHook == nil || oldHook() == nil {
		t.Fatal("eligible amd64 pair kernel not registered")
	}
	const out, in = 9, 512
	kinds := []kQuantKind{kindQ2K, kindQ3K, kindQ5K, kindQ6K, kindIQ3S, kindIQ3XXS, kindIQ4XS,
		kindIQ2XXS, kindIQ2XS, kindIQ2S, kindIQ1S, kindIQ1M, kindQ8_0, kindQ4_0, kindIQ4NL}
	for _, kind := range kinds {
		nblk, bb := in/kind.blockWeights(), kind.blockBytes()
		raw := make([]byte, out*nblk*bb)
		lcgBytes(raw, uint64(kind)+7007)
		pinResidentQuantScales(raw, out, nblk, kind)
		for off := 0; off < len(raw); off += bb {
			blk := raw[off : off+bb]
			if kind == kindQ3K {
				binary.LittleEndian.PutUint16(blk[bb-2:], 0x3400)
			}
			if kind == kindIQ1M {
				// Its f16 scale is split across four scale-word high nibbles.
				for j := 0; j < 4; j++ {
					word := binary.LittleEndian.Uint16(blk[48+2*j:]) & 0x0fff
					word |= ((uint16(0x3400) >> uint(4*j)) & 0xf) << 12
					binary.LittleEndian.PutUint16(blk[48+2*j:], word)
				}
			}
		}
		qt := quantizeKQuantFromRaw(raw, out, in, kind)
		beforeRaw := bytes.Clone(raw)
		for _, P := range []int{1, 2, 3, 19} {
			x := make([]float32, P*in)
			for i := range x {
				x[i] = float32(i%41-20) / 37
			}
			x[0] = float32(math.Copysign(0, -1))
			if P > 1 {
				clear(x[in : 2*in])
			}
			beforeX := slices.Clone(x)
			want := make([]float32, P*out)
			storage := make([]float32, P*out+2)
			storage[0], storage[len(storage)-1] = 12345, -6789
			got := storage[1 : len(storage)-1]
			for i := range got {
				got[i] = float32(math.NaN())
			}
			kQuantF32BlockPairArch = nil
			kQuantMatRowsIntoBatch(qt, x, P, want)
			calls := 0
			kQuantF32BlockPairArch = func() kQuantF32BlockPair { calls++; return oldHook() }
			kQuantMatRowsIntoBatch(qt, x, P, got)
			expectedCalls := 0
			if P > 1 && kind.blockWeights() == qkK {
				expectedCalls = 1
			}
			if calls != expectedCalls {
				t.Fatalf("%s P=%d selector calls=%d want%d", kind, P, calls, expectedCalls)
			}
			assertDecodedPairBits(t, kind.String(), got, want)
			if storage[0] != 12345 || storage[len(storage)-1] != -6789 || !bytes.Equal(raw, beforeRaw) {
				t.Fatal("pair route changed weight or output ownership")
			}
			assertDecodedPairBits(t, "activation", x, beforeX)
		}
	}
}

// The shared implementation also supplies the original Q4_K range hook; exercise
// its previously deferred odd-token, subset-row, multi-block and refusal edges.
func TestQ4KF32PairsRangeOwnershipAndRefusals(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved tier does not admit AVX2")
	}
	oldHook, oldMode, oldTier := q4kGemmF32RangeArch, qgemmMode, qtier
	t.Cleanup(func() { q4kGemmF32RangeArch, qgemmMode, qtier = oldHook, oldMode, oldTier })
	qgemmMode = qgemmModeTile
	const out, in = 7, 768
	raw := make([]byte, out*(in/qkK)*q4kBlockBytes)
	rng := rand.New(rand.NewSource(7007))
	for off := 0; off < len(raw); off += q4kBlockBytes {
		randQ4KBlockBounded(rng, raw[off:off+q4kBlockBytes], 2, 10)
	}
	qt := quantizeQ4KFromRaw(raw, out, in)
	beforeRaw := bytes.Clone(raw)
	for _, P := range []int{2, 3, 19} {
		x := mkVec(P*in, uint64(P)+7007)
		beforeX := slices.Clone(x)
		want, got := make([]float32, P*out), make([]float32, P*out)
		for i := range got {
			got[i], want[i] = 12345, 12345
		}
		q4kGemmF32RangeArch = nil
		q4kGemmRange(qt, x, P, want, 1, out-1)
		if !q4kGemmF32RangeAVX2(qt, x, P, got, 1, out-1) {
			t.Fatal("eligible range declined")
		}
		assertDecodedPairBits(t, "Q4_K row subset", got, want)
		assertDecodedPairBits(t, "Q4_K input", x, beforeX)
		if !bytes.Equal(raw, beforeRaw) {
			t.Fatal("weight input mutated")
		}
	}
	x, y := make([]float32, 3*in), make([]float32, 3*out)
	for i := range y {
		y[i] = 17
	}
	decline := func(label string, P int, xx, yy []float32, lo, hi int) {
		t.Helper()
		if q4kGemmF32RangeAVX2(qt, xx, P, yy, lo, hi) {
			t.Fatalf("accepted %s", label)
		}
		for _, v := range y {
			if v != 17 {
				t.Fatalf("%s decline wrote output", label)
			}
		}
	}
	decline("singleton", 1, x, y, 0, out)
	decline("short X", 3, x[:len(x)-1], y, 0, out)
	decline("short Y", 3, x, y[:len(y)-1], 0, out)
	decline("row bounds", 3, x, y, -1, out)
	qgemmMode = qgemmModeLegacy
	decline("legacy", 3, x, y, 0, out)
	if kQuantF32BlockPairAVX2() != nil {
		t.Fatal("K-quant legacy selector admitted")
	}
	qgemmMode = qgemmModeAccel
	decline("accelerate", 3, x, y, 0, out)
	qgemmMode = qgemmModeTile
	qtier = tierScalar
	decline("scalar tier", 3, x, y, 0, out)
	if kQuantF32BlockPairAVX2() != nil {
		t.Fatal("K-quant scalar selector admitted")
	}
}
