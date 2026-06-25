package ggufload

// dequant_reuse_test.go — the #440 page-churn fix: dequantF32Into reuses one f32 arena
// across the GGUF->Q8 quant-on-load tensors instead of allocating a throwaway elems*4
// buffer per tensor. These are pure, file-free unit tests over the dequant helpers: a
// reused buffer must produce values bit-identical to a fresh allocation (no stale tail
// leaks), must actually reuse the backing array when capacity allows, and must fall back
// to a fresh allocation when the next tensor is larger than the arena.

import (
	"encoding/binary"
	"math"
	"testing"
)

// f32TensorFixture builds an F32 TensorInfo plus its little-endian raw payload from values.
func f32TensorFixture(name string, vals []float32) (TensorInfo, []byte) {
	raw := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return TensorInfo{Name: name, Dims: []uint64{uint64(len(vals))}, Type: TensorF32}, raw
}

func assertF32BitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len=%d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]=%v (bits %#x), want %v (bits %#x)", label, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestDequantF32IntoReusesArenaAndNeverLeaksStaleData(t *testing.T) {
	largeVals := make([]float32, 64)
	for i := range largeVals {
		largeVals[i] = float32(i)*1.5 + 100 // distinct from smallVals so a leak would show
	}
	smallVals := make([]float32, 16)
	for i := range smallVals {
		smallVals[i] = float32(i) * -2.25
	}
	largeInfo, largeRaw := f32TensorFixture("large", largeVals)
	smallInfo, smallRaw := f32TensorFixture("small", smallVals)

	// Shrink-reuse: dequant the large tensor fresh, then dequant the smaller tensor into
	// that arena. The smaller result must be correct AND must reuse the large arena.
	bufLarge, err := dequantF32Into(nil, largeInfo, largeRaw)
	if err != nil {
		t.Fatalf("dequant large: %v", err)
	}
	assertF32BitsEqual(t, "fresh-large", bufLarge, largeVals)

	bufSmall, err := dequantF32Into(bufLarge, smallInfo, smallRaw)
	if err != nil {
		t.Fatalf("dequant small into large arena: %v", err)
	}
	assertF32BitsEqual(t, "reused-small", bufSmall, smallVals)
	if len(bufSmall) != len(smallVals) {
		t.Fatalf("reused-small len=%d, want %d (stale tail not trimmed)", len(bufSmall), len(smallVals))
	}
	// Reuse proof: the small dequant wrote through the large arena, so the large arena's
	// element 0 now holds the small value, not the large one.
	if math.Float32bits(bufLarge[0]) != math.Float32bits(smallVals[0]) {
		t.Fatalf("expected arena reuse: bufLarge[0]=%v, want overwritten with %v", bufLarge[0], smallVals[0])
	}

	// Grow path: a buffer too small for the next tensor must allocate fresh and leave the
	// original arena untouched.
	bufSmall2, err := dequantF32Into(nil, smallInfo, smallRaw)
	if err != nil {
		t.Fatalf("dequant small fresh: %v", err)
	}
	bufLarge2, err := dequantF32Into(bufSmall2, largeInfo, largeRaw)
	if err != nil {
		t.Fatalf("dequant large into small arena: %v", err)
	}
	assertF32BitsEqual(t, "grown-large", bufLarge2, largeVals)
	if math.Float32bits(bufSmall2[0]) != math.Float32bits(smallVals[0]) {
		t.Fatalf("grow path must not write into the smaller arena: bufSmall2[0]=%v changed", bufSmall2[0])
	}
}

func TestDequantF32WrapperMatchesIntoNil(t *testing.T) {
	vals := []float32{1.25, -2.5, 0, 7.5, -0.125, 1024}
	info, raw := f32TensorFixture("w", vals)
	wrapper, err := dequantF32(info, raw)
	if err != nil {
		t.Fatalf("dequantF32: %v", err)
	}
	into, err := dequantF32Into(nil, info, raw)
	if err != nil {
		t.Fatalf("dequantF32Into(nil): %v", err)
	}
	assertF32BitsEqual(t, "wrapper-vs-into", wrapper, into)
	assertF32BitsEqual(t, "wrapper-vs-source", wrapper, vals)
}

func TestReuseF32CapacityRule(t *testing.T) {
	base := make([]float32, 8, 32) // len 8, cap 32
	if got := reuseF32(base, 16); &got[0] != &base[0] || len(got) != 16 {
		t.Fatalf("reuseF32 should reuse the backing array for n<=cap (len=%d)", len(got))
	}
	if got := reuseF32(base, 64); &got[0] == &base[0] || len(got) != 64 {
		t.Fatalf("reuseF32 should allocate fresh for n>cap (len=%d)", len(got))
	}
	if got := reuseF32(nil, 4); got == nil || len(got) != 4 {
		t.Fatalf("reuseF32(nil) should allocate length-n slice (len=%d)", len(got))
	}
}
