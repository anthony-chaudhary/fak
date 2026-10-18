package norm

import (
	"math"
	"testing"
)

// Frozen pre-change ordinary RMSNorm arithmetic, independent of the new helper.
//
//go:noinline
func oldSerialNormIntoReference(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// fak-test:runtime fast est=20ms lane=default
func TestSerialRMSNormIntoParityAndAllocation(t *testing.T) {
	bits := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s lengths %d != %d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] bits %08x != %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	for _, width := range []int{0, 1, 3, 32, 257, 1024} {
		x, w := make([]float32, width), make([]float32, width)
		for i := range x {
			x[i], w[i] = float32(i%17-8)/7, float32(i%11-5)/3
		}
		for _, eps := range []float32{0, 1e-5} {
			for _, size := range []int{0, width / 2, width, width + 2} {
				got, want := make([]float32, size), make([]float32, size)
				for i := range got {
					got[i], want[i] = 19, 19
				}
				copy(want, oldSerialNormIntoReference(x, w, eps))
				SerialRMSNormInto(got, x, w, eps)
				bits("width/destination", got, want)
			}
		}
		if width > 0 {
			dst := make([]float32, width)
			before := testing.AllocsPerRun(20, func() { copy(dst, oldSerialNormIntoReference(x, w, 1e-5)) })
			after := testing.AllocsPerRun(20, func() { SerialRMSNormInto(dst, x, w, 1e-5) })
			if before < 1 || after != 0 {
				t.Fatalf("width=%d allocation premise/delta: before=%g after=%g", width, before, after)
			}
		}
	}
	for _, special := range []float32{float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		x, w := []float32{special, 1, -2}, []float32{1, -1, 0.5}
		got := make([]float32, len(x))
		SerialRMSNormInto(got, x, w, 1e-5)
		bits("nonfinite/signed-zero", got, oldSerialNormIntoReference(x, w, 1e-5))
	}
	// Both overlap directions and exact aliases preserve allocating-then-copy
	// semantics for x and for w; compare the entire backing slice, including tails.
	for _, weightsOverlap := range []bool{false, true} {
		for _, offset := range []int{-1, 0, 1} {
			got := []float32{1, 2, 3, 4, 5, 6}
			want := append([]float32(nil), got...)
			other := []float32{0.5, -1, 2, 0.25}
			gx, gw, wx, ww := got[1:5], other, want[1:5], other
			if weightsOverlap {
				gx, gw, wx, ww = other, got[1:5], other, want[1:5]
			}
			copy(want[1+offset:5+offset], oldSerialNormIntoReference(wx, ww, 1e-5))
			SerialRMSNormInto(got[1+offset:5+offset], gx, gw, 1e-5)
			bits("overlap", got, want)
		}
	}
	// Preserve source-evaluation failure before copy, even for a short destination.
	for _, dstLen := range []int{0, 2, 4} {
		dst := []float32{9, 9, 9, 9}
		panicked := false
		func() {
			defer func() { panicked = recover() != nil }()
			SerialRMSNormInto(dst[:dstLen], []float32{1, 2, 3, 4}, []float32{1}, 1e-5)
		}()
		if !panicked {
			t.Fatal("short weights did not panic")
		}
		bits("panic preserves destination", dst, []float32{9, 9, 9, 9})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestSerialRMSNormIntoDisjointParityAndAllocation(t *testing.T) {
	bits := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s lengths %d != %d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] bits %08x != %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	for _, width := range []int{0, 1, 3, 32, 257, 1024} {
		x, w := make([]float32, width), make([]float32, width)
		for i := range x {
			x[i], w[i] = float32(i%17-8)/7, float32(i%11-5)/3
		}
		for _, eps := range []float32{0, 1e-5} {
			for _, size := range []int{0, width / 2, width, width + 2} {
				got, want := make([]float32, size), make([]float32, size)
				for i := range got {
					got[i], want[i] = 19, 19
				}
				copy(want, oldSerialNormIntoReference(x, w, eps))
				SerialRMSNormIntoDisjoint(got, x, w, eps)
				bits("width/destination", got, want)
			}
		}
		if width > 0 {
			dst := make([]float32, width)
			before := testing.AllocsPerRun(20, func() { copy(dst, oldSerialNormIntoReference(x, w, 1e-5)) })
			after := testing.AllocsPerRun(20, func() { SerialRMSNormIntoDisjoint(dst, x, w, 1e-5) })
			if before < 1 || after != 0 {
				t.Fatalf("width=%d allocation premise/delta: before=%g after=%g", width, before, after)
			}
		}
	}
	for _, special := range []float32{float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		x, w := []float32{special, 1, -2}, []float32{1, -1, 0.5}
		got := make([]float32, len(x))
		SerialRMSNormIntoDisjoint(got, x, w, 1e-5)
		bits("nonfinite/signed-zero", got, oldSerialNormIntoReference(x, w, 1e-5))
	}
	// The two read-only inputs may share storage; only dst must be disjoint.
	backing := []float32{1, -2, 3, 0.5, 2}
	before := append([]float32(nil), backing...)
	dst := make([]float32, 4)
	SerialRMSNormIntoDisjoint(dst, backing[:4], backing[1:], 1e-5)
	bits("overlapping inputs", dst, oldSerialNormIntoReference(before[:4], before[1:], 1e-5))
	bits("inputs unchanged", backing, before)
	// Preserve source-evaluation failure before copy, even for a short destination.
	for _, dstLen := range []int{0, 2, 4} {
		dst := []float32{9, 9, 9, 9}
		panicked := false
		func() {
			defer func() { panicked = recover() != nil }()
			SerialRMSNormIntoDisjoint(dst[:dstLen], []float32{1, 2, 3, 4}, []float32{1}, 1e-5)
		}()
		if !panicked {
			t.Fatal("short weights did not panic")
		}
		bits("panic preserves destination", dst, []float32{9, 9, 9, 9})
	}
}

// Explicitly selected microbenchmark; no hardware/model throughput claim.
func BenchmarkSerialRMSNormInto(b *testing.B) {
	for _, width := range []int{128, 1024} {
		name := "width128"
		if width == 1024 {
			name = "width1024"
		}
		x, w, dst := make([]float32, width), make([]float32, width), make([]float32, width)
		for i := range x {
			x[i], w[i] = float32(i%17-8)/7, float32(i%11-5)/3
		}
		b.Run(name+"/old-copy", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				copy(dst, oldSerialNormIntoReference(x, w, 1e-5))
			}
		})
		b.Run(name+"/into", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				SerialRMSNormInto(dst, x, w, 1e-5)
			}
		})
		b.Run(name+"/disjoint", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				SerialRMSNormIntoDisjoint(dst, x, w, 1e-5)
			}
		})
	}
}
