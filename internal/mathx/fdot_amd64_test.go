//go:build amd64

package mathx

import (
	"math"
	"os"
	"testing"
)

func TestFDotSSE2ExistingPinSelection(t *testing.T) {
	pin := os.Getenv("FAK_QKERNEL")
	if (fDotArch != nil) != (pin == "avx2" || pin == "avx512") {
		t.Fatal("new route must remain behind the existing explicit accelerated pin")
	}
}

func TestFDotSSE2ExactTreeTailAndOwnership(t *testing.T) {
	old := fDotArch
	t.Cleanup(func() { fDotArch = old })
	fDotArch = fDotSSE2
	for _, n := range []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 128, 255, 256, 257, 1024} {
		r, x := make([]float32, n+2), make([]float32, n+2)
		r[0], r[n+1], x[0], x[n+1] = 12345, -6789, 2468, -1357
		rr, xx := r[1:n+1], x[1:n+1]
		for pattern := 0; pattern < 5; pattern++ {
			for i := range rr {
				switch pattern {
				case 0:
					rr[i], xx[i] = float32(i%29-14)/17, float32(i%37-18)/31
				case 1:
					rr[i], xx[i] = float32(math.Copysign(0, float64(i%2*2-1))), float32(i%3-1)
				case 2:
					rr[i], xx[i] = math.Float32frombits(uint32(i%31+1)), float32(i%7-3)
				case 3:
					rr[i], xx[i] = math.MaxFloat32, float32(i%3-1)*2
				case 4:
					rr[i], xx[i] = float32(i%5-2)*1e20, float32(i%7-3)*1e-20
				}
			}
			beforeR, beforeX := append([]float32(nil), rr...), append([]float32(nil), xx...)
			want := fDotScalar(rr, xx)
			got := FDot(rr, xx)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("n=%d pattern=%d got=%08x want=%08x", n, pattern, math.Float32bits(got), math.Float32bits(want))
			}
			for i := range rr {
				if math.Float32bits(rr[i]) != math.Float32bits(beforeR[i]) || math.Float32bits(xx[i]) != math.Float32bits(beforeX[i]) {
					t.Fatal("input changed")
				}
			}
			if r[0] != 12345 || r[n+1] != -6789 || x[0] != 2468 || x[n+1] != -1357 {
				t.Fatal("input boundary changed")
			}
		}
	}
	r, x := make([]float32, 16), make([]float32, 16)
	for i := range r {
		r[i], x[i] = float32(i-7)/9, float32(i%7-3)/5
	}
	var lanes [10]float32
	lanes[0], lanes[9] = 12345, -6789
	if !fDotLanesSSE2(&r[0], &x[0], 2, &lanes[1]) {
		t.Fatal("finite bulk declined")
	}
	var want [8]float32
	for i := range r {
		want[i&7] += r[i] * x[i]
	}
	for i := range want {
		if math.Float32bits(lanes[i+1]) != math.Float32bits(want[i]) {
			t.Fatal("lane order changed")
		}
	}
	if lanes[0] != 12345 || lanes[9] != -6789 {
		t.Fatal("native output overrun")
	}
}

func TestFDotSSE2ExceptionalAndShortInputsKeepScalar(t *testing.T) {
	old := fDotArch
	t.Cleanup(func() { fDotArch = old })
	fDotArch = fDotSSE2
	for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fc12345, 0xffc23456, 0x7fa12345, 0xffa12345} {
		for _, position := range []int{0, 7, 8, 16} {
			r, x := make([]float32, 17), make([]float32, 17)
			for i := range r {
				r[i], x[i] = 1, 1
			}
			r[position] = math.Float32frombits(bits)
			want := fDotScalar(r, x)
			if value, ok := fDotSSE2(r, x); math.IsNaN(float64(want)) {
				if ok {
					t.Fatal("final NaN bypassed scalar payload route")
				}
			} else if !ok || math.Float32bits(value) != math.Float32bits(want) {
				t.Fatal("non-NaN infinity changed")
			}
			got := FDot(r, x)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatal("exceptional payload changed")
			}
			// Check the second operand as well; both-NaN payload precedence
			// is delegated to the unchanged scalar expression.
			x[position] = math.Float32frombits(0x7fc76543)
			got, want = FDot(r, x), fDotScalar(r, x)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatal("two-NaN payload changed")
			}
		}
	}
	didPanic := func(fn func() float32) (panicked bool) {
		defer func() { panicked = recover() != nil }()
		fn()
		return
	}
	for _, n := range []int{1, 7, 8, 9, 16} {
		r, x := make([]float32, n), make([]float32, n-1)
		if _, ok := fDotSSE2(r, x); ok {
			t.Fatal("short second input entered native path")
		}
		if !didPanic(func() float32 { return FDot(r, x) }) || !didPanic(func() float32 { return fDotScalar(r, x) }) {
			t.Fatal("short input panic contract changed")
		}
	}
	if got := FDot(nil, nil); math.Float32bits(got) != 0 {
		t.Fatal("empty dot changed positive zero")
	}
}
