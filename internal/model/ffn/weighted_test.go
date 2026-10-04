package ffn

import (
	"math"
	"testing"
)

// TestAddScaledContract pins the projection-free accumulation leaf added for
// fak#13458: ordered, gate-weighted accumulation into a caller-owned dst, with
// equal-length validation BEFORE any mutation and a zero-allocation success
// path. Every named consumer (the standard MoE routed loop and both V4.1 expert
// accumulation branches) relies on exactly these invariants, so this is the
// independent leaf witness.
func TestAddScaledContract(t *testing.T) {
	t.Run("ordered bit-exact vs naive loop", func(t *testing.T) {
		dst := []float32{0.25, -1.5, 3.0, -0.0, 7.5, 12.0}
		src := []float32{2.0, -4.0, 0.5, -0.0, 1.25, -3.0}
		const w = float32(-1.75)
		want := append([]float32(nil), dst...)
		for i := range want {
			want[i] += w * src[i]
		}
		if err := AddScaled(dst, src, w); err != nil {
			t.Fatalf("AddScaled: %v", err)
		}
		for i := range want {
			if math.Float32bits(dst[i]) != math.Float32bits(want[i]) {
				t.Fatalf("dst[%d] bits=%#x want=%#x", i, math.Float32bits(dst[i]), math.Float32bits(want[i]))
			}
		}
	})

	t.Run("signed zero and single-weight identity", func(t *testing.T) {
		dst := make([]float32, 3)
		src := []float32{1.5, -2.5, 0.0}
		if err := AddScaled(dst, src, 1); err != nil {
			t.Fatalf("AddScaled: %v", err)
		}
		for i := range src {
			if math.Float32bits(dst[i]) != math.Float32bits(src[i]) {
				t.Fatalf("w=1 dst[%d] bits=%#x want=%#x", i, math.Float32bits(dst[i]), math.Float32bits(src[i]))
			}
		}
	})

	t.Run("length mismatch refused before mutation", func(t *testing.T) {
		dst := []float32{1, 2, 3}
		src := []float32{4, 5}
		dstCopy := append([]float32(nil), dst...)
		srcCopy := append([]float32(nil), src...)
		if err := AddScaled(dst, src, 2); err == nil {
			t.Fatal("AddScaled with mismatched lengths returned nil, want error")
		}
		for i := range dstCopy {
			if math.Float32bits(dst[i]) != math.Float32bits(dstCopy[i]) {
				t.Fatalf("dst mutated on refusal: [%d] bits=%#x want=%#x", i, math.Float32bits(dst[i]), math.Float32bits(dstCopy[i]))
			}
		}
		for i := range srcCopy {
			if math.Float32bits(src[i]) != math.Float32bits(srcCopy[i]) {
				t.Fatalf("src mutated on refusal: [%d] bits=%#x want=%#x", i, math.Float32bits(src[i]), math.Float32bits(srcCopy[i]))
			}
		}
	})

	t.Run("equal empty is a no-op", func(t *testing.T) {
		if err := AddScaled(nil, nil, 3); err != nil {
			t.Fatalf("AddScaled(empty, empty) = %v, want nil", err)
		}
		if err := AddScaled([]float32{}, []float32{}, -1); err != nil {
			t.Fatalf("AddScaled([], []) = %v, want nil", err)
		}
	})

	t.Run("src read-only and zero allocations", func(t *testing.T) {
		const n = 512
		dst := make([]float32, n)
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i%13) - 6
			dst[i] = float32(i%7) - 3
		}
		srcCopy := append([]float32(nil), src...)
		allocs := testing.AllocsPerRun(50, func() {
			if err := AddScaled(dst, src, 0.5); err != nil {
				t.Fatalf("AddScaled: %v", err)
			}
		})
		if allocs != 0 {
			t.Fatalf("AddScaled allocations = %v, want 0", allocs)
		}
		for i := range srcCopy {
			if math.Float32bits(src[i]) != math.Float32bits(srcCopy[i]) {
				t.Fatalf("src mutated: [%d] bits=%#x want=%#x", i, math.Float32bits(src[i]), math.Float32bits(srcCopy[i]))
			}
		}
	})
}
