// Tests for the pure kernel helpers in q8kernel: the f32 dot product, the
// weight-only int8xf32 dot product, and the median-ms reducer. These are all
// deterministic, allocation-free numeric functions with no external resource,
// so the expected values below are computed by hand and the tests fail if the
// arithmetic or the loop boundaries (8-wide body vs. scalar tail) regress.
package main

import (
	"math"
	"testing"
	"time"
)

func approxEq(a, b, eps float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

func TestFdot(t *testing.T) {
	tests := []struct {
		name string
		r    []float32
		x    []float32
		want float32
	}{
		{"empty", []float32{}, []float32{}, 0},
		// pure scalar tail (len < 8): 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
		{"tail_only", []float32{1, 2, 3}, []float32{4, 5, 6}, 32},
		// exactly one 8-wide body, no tail: dot([1..8],[1..8]) = sum k^2 = 204
		{"one_block", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, 204},
		// 8-wide body + scalar tail: dot([1..9],ones) = 1+2+...+9 = 45
		{"block_plus_tail", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1, 1}, 45},
		// negatives: 2*-1 + -2*3 = -2 + -6 = -8
		{"signed", []float32{2, -2}, []float32{-1, 3}, -8},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := fdot(tc.r, tc.x)
			if !approxEq(got, tc.want, 1e-4) {
				t.Fatalf("fdot(%v,%v) = %v, want %v", tc.r, tc.x, got, tc.want)
			}
		})
	}
}

func TestQdotWO(t *testing.T) {
	tests := []struct {
		name string
		q    []int8
		x    []float32
		want float32
	}{
		{"empty", []int8{}, []float32{}, 0},
		// scalar tail only: 1*0.5 + 2*0.5 + 3*0.5 = 3.0
		{"tail_only", []int8{1, 2, 3}, []float32{0.5, 0.5, 0.5}, 3},
		// one 8-wide body: int8(1..8) dot ones = 36
		{"one_block", []int8{1, 2, 3, 4, 5, 6, 7, 8},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1}, 36},
		// body + tail with a negative weight: (1+2+...+8) + (-2)*3 = 36 - 6 = 30
		{"block_plus_tail", []int8{1, 2, 3, 4, 5, 6, 7, 8, -2},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1, 3}, 30},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := qdotWO(tc.q, tc.x)
			if !approxEq(got, tc.want, 1e-4) {
				t.Fatalf("qdotWO(%v,%v) = %v, want %v", tc.q, tc.x, got, tc.want)
			}
		})
	}
}

func TestMedMS(t *testing.T) {
	ms := func(f float64) time.Duration { return time.Duration(f * float64(time.Millisecond)) }

	tests := []struct {
		name string
		in   []time.Duration
		want float64
	}{
		// single element
		{"single", []time.Duration{ms(5)}, 5},
		// odd count, unsorted: sorted [1,2,3], len/2=1 -> 2
		{"odd_unsorted", []time.Duration{ms(3), ms(1), ms(2)}, 2},
		// even count: len=4, len/2=2 -> upper-middle element of sorted [1,2,3,4] -> 3
		{"even_upper_middle", []time.Duration{ms(4), ms(2), ms(1), ms(3)}, 3},
		// duplicates
		{"dupes", []time.Duration{ms(2), ms(2), ms(2)}, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := medMS(tc.in)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Fatalf("medMS(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// medMS must not mutate its input (it copies before sorting).
func TestMedMSNoMutate(t *testing.T) {
	in := []time.Duration{3, 1, 2}
	_ = medMS(in)
	want := []time.Duration{3, 1, 2}
	for i := range in {
		if in[i] != want[i] {
			t.Fatalf("medMS mutated input: got %v, want %v", in, want)
		}
	}
}
