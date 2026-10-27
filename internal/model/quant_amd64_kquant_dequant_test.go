//go:build amd64

package model

import (
	"bytes"
	"encoding/binary"
	"math"
	"sync/atomic"
	"testing"
)

func assertKDequantBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal(label + ": length differs")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] bits%08x want%08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime fast est=60ms
func TestKQuantDequantAVX2MatchesScalarFamily(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	for _, tc := range []struct {
		kind   kQuantKind
		arch   func([]float32, []byte) bool
		scalar func([]float32, []byte)
	}{
		{kindQ5K, q5kDequantSuperBlockAVX2, q5kDequantSuperBlock}, {kindQ6K, q6kDequantSuperBlockAVX2, q6kDequantSuperBlock},
	} {
		for seed := 0; seed < 64; seed++ {
			raw := make([]byte, tc.kind.blockBytes())
			lcgBytes(raw, uint64(seed)+20261027)
			for _, scale := range []uint16{0, 0x8000, 1, 0x8001, 0x0400, 0x3400, 0xb400, 0x7bff, 0xfbff} {
				if tc.kind == kindQ5K {
					binary.LittleEndian.PutUint16(raw, scale)
					binary.LittleEndian.PutUint16(raw[2:], scale^0x8000)
				} else {
					binary.LittleEndian.PutUint16(raw[len(raw)-2:], scale)
				}
				before := bytes.Clone(raw)
				storage := make([]float32, qkK+2)
				storage[0], storage[len(storage)-1] = 17, -19
				got := storage[1 : len(storage)-1 : len(storage)-1]
				want := make([]float32, qkK)
				tc.scalar(want, raw)
				if !tc.arch(got, raw) {
					t.Fatal("finite well-formed block declined")
				}
				assertKDequantBits(t, tc.kind.String(), got, want)
				if storage[0] != 17 || storage[len(storage)-1] != -19 || !bytes.Equal(raw, before) {
					t.Fatal("decoder changed borrowed input/canary")
				}
			}
		}
	}
}

// fak-test:runtime fast est=20ms
func TestKQuantDequantAVX2ExceptionalAndShapeDeclines(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	oldTier := qtier
	t.Cleanup(func() { qtier = oldTier })
	for _, kind := range []kQuantKind{kindQ5K, kindQ6K} {
		arch, scalar := q5kDequantSuperBlockAVX2, q5kDequantSuperBlock
		offsets := []int{0, 2}
		if kind == kindQ6K {
			arch, scalar = q6kDequantSuperBlockAVX2, q6kDequantSuperBlock
			offsets = []int{q6kBlockBytes - 2}
		}
		raw := make([]byte, kind.blockBytes())
		lcgBytes(raw, 2917)
		for _, off := range offsets {
			binary.LittleEndian.PutUint16(raw[off:], 0x3400)
		}
		original := bytes.Clone(raw)
		dst := make([]float32, qkK)
		decline := func(label string, out []float32, block []byte) {
			t.Helper()
			for i := range dst {
				dst[i] = 123
			}
			if arch(out, block) {
				t.Fatal(label + " unexpectedly admitted")
			}
			for _, v := range dst {
				if v != 123 {
					t.Fatal(label + " wrote before decline")
				}
			}
		}
		decline("short output", dst[:qkK-1], raw)
		decline("short block", dst, raw[:len(raw)-1])
		qtier = tierScalar
		decline("scalar pin", dst, raw)
		qtier = oldTier
		for _, off := range offsets {
			for _, bits := range []uint16{0x7c00, 0xfc00, 0x7e35, 0x7c01, 0xfe67} {
				copy(raw, original)
				binary.LittleEndian.PutUint16(raw[off:], bits)
				decline("exceptional scale", dst, raw)
				want := make([]float32, qkK)
				scalar(want, raw)
				kQuantDequantSuperBlock(dst, raw, kind)
				assertKDequantBits(t, "exceptional scalar fallback", dst, want)
			}
		}
	}
}

// fak-test:runtime fast est=80ms
func TestKQuantDequantAVX2OrdinaryPrefillReuse(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	old5, old6, oldForce, oldMode := q5kDequantSuperBlockArch, q6kDequantSuperBlockArch, kQuantSDOTForce, qgemmMode
	t.Cleanup(func() {
		q5kDequantSuperBlockArch, q6kDequantSuperBlockArch, kQuantSDOTForce, qgemmMode = old5, old6, oldForce, oldMode
	})
	kQuantSDOTForce, qgemmMode = -1, qgemmModeTile
	const out, in = 7, 768
	for _, kind := range []kQuantKind{kindQ5K, kindQ6K} {
		nblk := in / qkK
		raw := make([]byte, out*nblk*kind.blockBytes())
		lcgBytes(raw, uint64(kind)+9923)
		pinResidentQuantScales(raw, out, nblk, kind)
		qt := quantizeKQuantFromRaw(raw, out, in, kind)
		rawBefore := bytes.Clone(raw)
		for _, P := range []int{1, 2, 3, 19} {
			x := mkVec(P*in, uint64(P)+19)
			xBefore := append([]float32(nil), x...)
			want, got := make([]float32, P*out), make([]float32, P*out)
			q5kDequantSuperBlockArch, q6kDequantSuperBlockArch = nil, nil
			kQuantMatRowsIntoBatch(qt, x, P, want)
			var calls atomic.Int64
			q5kDequantSuperBlockArch = func(dst []float32, blk []byte) bool { calls.Add(1); return old5(dst, blk) }
			q6kDequantSuperBlockArch = func(dst []float32, blk []byte) bool { calls.Add(1); return old6(dst, blk) }
			kQuantMatRowsIntoBatch(qt, x, P, got)
			if calls.Load() != int64(out*nblk) {
				t.Fatalf("decoded%d blocks want%d independent of P", calls.Load(), out*nblk)
			}
			assertKDequantBits(t, "ordinary prefill", got, want)
			assertKDequantBits(t, "borrowed activation", x, xBefore)
			if !bytes.Equal(raw, rawBefore) {
				t.Fatal("resident bytes changed")
			}
		}
	}
}

// fak-test:runtime fast est=60ms
func TestQ4KFloatDequantAVX2Companion(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	raw := make([]byte, q4kBlockBytes)
	for seed := 0; seed < 64; seed++ {
		lcgBytes(raw, uint64(seed)+1193)
		for _, scale := range []uint16{0, 0x8000, 1, 0x8001, 0x0400, 0x3400, 0xb400, 0x7bff, 0xfbff} {
			binary.LittleEndian.PutUint16(raw, scale)
			binary.LittleEndian.PutUint16(raw[2:], scale^0x8000)
			before := bytes.Clone(raw)
			storage := make([]float32, qkK+2)
			storage[0], storage[len(storage)-1] = 17, -19
			got, want := storage[1:len(storage)-1:len(storage)-1], make([]float32, qkK)
			q4kDequantSuperBlockScalar(want, raw)
			if !q4kDequantSuperBlockAVX2(got, raw) {
				t.Fatal("finite Q4_K declined")
			}
			assertKDequantBits(t, "Q4_K block", got, want)
			if storage[0] != 17 || storage[len(storage)-1] != -19 || !bytes.Equal(raw, before) {
				t.Fatal("Q4_K ownership changed")
			}
		}
	}
	dst := make([]float32, qkK)
	decline := func(out []float32, block []byte) {
		t.Helper()
		for i := range dst {
			dst[i] = 123
		}
		if q4kDequantSuperBlockAVX2(out, block) {
			t.Fatal("invalid Q4_K admitted")
		}
		for _, v := range dst {
			if v != 123 {
				t.Fatal("Q4_K decline wrote output")
			}
		}
	}
	decline(dst[:qkK-1], raw)
	decline(dst, raw[:len(raw)-1])
	for _, off := range []int{0, 2} {
		for _, bits := range []uint16{0x7c00, 0xfc00, 0x7e35, 0x7c01, 0xfe67} {
			binary.LittleEndian.PutUint16(raw, 0x3400)
			binary.LittleEndian.PutUint16(raw[2:], 0x3400)
			binary.LittleEndian.PutUint16(raw[off:], bits)
			decline(dst, raw)
			want := make([]float32, qkK)
			q4kDequantSuperBlockScalar(want, raw)
			q4kDequantSuperBlock(dst, raw)
			assertKDequantBits(t, "Q4_K exceptional fallback", dst, want)
		}
	}
	oldTier := qtier
	t.Cleanup(func() { qtier = oldTier })
	binary.LittleEndian.PutUint16(raw, 0x3400)
	binary.LittleEndian.PutUint16(raw[2:], 0x3400)
	qtier = tierScalar
	decline(dst, raw)
	qtier = oldTier
}

// fak-test:runtime fast est=80ms
func TestQ4KFloatDequantAVX2OrdinaryPrefillReuse(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved ISA does not admit AVX2")
	}
	oldHook, oldForce, oldMode := q4kDequantSuperBlockArch, q4kSDOTForce, qgemmMode
	t.Cleanup(func() { q4kDequantSuperBlockArch, q4kSDOTForce, qgemmMode = oldHook, oldForce, oldMode })
	q4kSDOTForce, qgemmMode = -1, qgemmModeTile
	const out, in = 7, 768
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	lcgBytes(raw, 97221)
	for off := 0; off < len(raw); off += q4kBlockBytes {
		binary.LittleEndian.PutUint16(raw[off:], 0x3400)
		binary.LittleEndian.PutUint16(raw[off+2:], 0x3000)
	}
	qt := quantizeQ4KFromRaw(raw, out, in)
	before := bytes.Clone(raw)
	for _, P := range []int{1, 2, 3, 19} {
		x := mkVec(P*in, uint64(P)+51)
		xBefore := append([]float32(nil), x...)
		q4kDequantSuperBlockArch = nil
		want := q4kGemm(qt, x, P)
		var calls atomic.Int64
		q4kDequantSuperBlockArch = func(dst []float32, blk []byte) bool { calls.Add(1); return oldHook(dst, blk) }
		got := q4kGemm(qt, x, P)
		if calls.Load() != int64(out*nblk) {
			t.Fatalf("Q4_K decoded%d blocks want%d independent ofP", calls.Load(), out*nblk)
		}
		assertKDequantBits(t, "Q4_K ordinary prefill", got, want)
		assertKDequantBits(t, "Q4_K borrowed activation", x, xBefore)
		if !bytes.Equal(raw, before) {
			t.Fatal("Q4_K resident bytes changed")
		}
	}
}
