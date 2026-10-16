package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41RoPEBF16OwnedPairIdentityAndCallbacks(t *testing.T) {
	for _, selected := range []bool{false, true} {
		q, k := []float32{math.Float32frombits(0x80000000), 1}, []float32{-2, math.Float32frombits(0x00010000)}
		calls := 0
		var publish v41BF16BoundaryFunc
		if selected {
			publish = func(_ int, _ string, _ []float32) ([]float32, error) { calls++; panic("identity called publisher") }
		}
		a, b, e := v41RoPEBF16Pair(0, v41TailRoPEInputSite, q, k, publish)
		if e != nil || calls != 0 {
			t.Fatal(e)
		}
		for i := range q {
			if math.Float32bits(a[i]) != math.Float32bits(q[i]) {
				t.Fatal("Q bits")
			}
		}
		for i := range k {
			if math.Float32bits(b[i]) != math.Float32bits(k[i]) {
				t.Fatal("K bits")
			}
		}
		a[0] = 9
		b[0] = 8
		if q[0] == 9 || k[0] == 8 || cap(a) != len(a) {
			t.Fatal("caller storage aliased")
		}
	}
	for _, index := range []int{0, 3} {
		q, k := []float32{1, 2}, []float32{3, 4}
		if index == 0 {
			q[0] = math.Float32frombits(0x3f800001)
		} else {
			k[1] = math.Float32frombits(0x40800001)
		}
		calls := 0
		sentinel := errors.New("selected refusal")
		pub := func(_ int, site string, v []float32) ([]float32, error) {
			calls++
			if site != v41TailRoPEOutputSite || len(v) != 4 {
				t.Fatal("callback contract")
			}
			v[0] = 99
			return nil, sentinel
		}
		a, b, e := v41RoPEBF16Pair(2, v41TailRoPEOutputSite, q, k, pub)
		if !errors.Is(e, sentinel) || a != nil || b != nil || calls != 1 || q[0] == 99 {
			t.Fatal("selected error/caller isolation")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41BF16OrdinaryCallerStillCopies(t *testing.T) {
	in := []float32{1, 2}
	out, e := v41PublishBF16(0, "x", in, nil, func([]float32) ([]float32, error) { panic("identity host call") })
	if e != nil || &out[0] == &in[0] {
		t.Fatal("ordinary caller lost defensive copy")
	}
}
