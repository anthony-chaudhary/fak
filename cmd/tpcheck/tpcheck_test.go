// Package main tests for cmd/tpcheck's pure, deterministic helpers.
//
// These cover the resource-free leaf functions in the command: CSV rank
// parsing (parseInts), the argmax used by the wired-forward argmax-agreement
// check (argmaxIdx), the bit-exact float comparison (bitExact), and the
// max-absolute-difference metric (maxAbs). No GPU, model file, or network is
// touched — every expected value is computed by hand.
package main

import (
	"math"
	"testing"
)

func TestParseInts(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []int
		wantErr bool
	}{
		{name: "basic", in: "1,2,4,8", want: []int{1, 2, 4, 8}},
		{name: "single", in: "7", want: []int{7}},
		{name: "whitespace trimmed", in: " 3 , 5 ", want: []int{3, 5}},
		{name: "empty fields skipped", in: "1,,2,", want: []int{1, 2}},
		{name: "empty string yields empty slice", in: "", want: []int{}},
		{name: "only commas and spaces", in: " , , ", want: []int{}},
		{name: "negative numbers", in: "-2,0,3", want: []int{-2, 0, 3}},
		{name: "non-numeric errors", in: "1,x,3", wantErr: true},
		{name: "float-like errors", in: "1.5", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseInts(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseInts(%q): expected error, got nil (out=%v)", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseInts(%q): unexpected error: %v", tc.in, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("parseInts(%q) = %v, want %v (len %d != %d)", tc.in, got, tc.want, len(got), len(tc.want))
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("parseInts(%q)[%d] = %d, want %d (full %v)", tc.in, i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

func TestArgmaxIdx(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{name: "ascending", in: []float32{1, 2, 3}, want: 2},
		{name: "descending", in: []float32{3, 2, 1}, want: 0},
		{name: "max in middle", in: []float32{0, 5, 1}, want: 1},
		{name: "first on ties", in: []float32{1, 3, 3, 2}, want: 1},
		{name: "all equal returns first", in: []float32{2, 2, 2}, want: 0},
		{name: "single element", in: []float32{-9}, want: 0},
		{name: "negatives", in: []float32{-3, -1, -2}, want: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := argmaxIdx(tc.in); got != tc.want {
				t.Fatalf("argmaxIdx(%v) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

func TestBitExact(t *testing.T) {
	if !bitExact([]float32{1, 2, 3}, []float32{1, 2, 3}) {
		t.Fatalf("bitExact: identical slices should be exact")
	}
	if bitExact([]float32{1, 2, 3}, []float32{1, 2, 3.0001}) {
		t.Fatalf("bitExact: differing element should not be exact")
	}
	if bitExact([]float32{1, 2}, []float32{1, 2, 3}) {
		t.Fatalf("bitExact: length mismatch should not be exact")
	}
	if !bitExact([]float32{}, []float32{}) {
		t.Fatalf("bitExact: two empty slices should be exact")
	}
	// +0.0 and -0.0 are numerically equal but have distinct bit patterns,
	// so bitExact (which compares Float32bits) must report them as NOT exact.
	negZero := float32(math.Copysign(0, -1))
	posZero := float32(0)
	if negZero != posZero {
		t.Fatalf("test precondition: -0.0 should compare == +0.0")
	}
	if bitExact([]float32{negZero}, []float32{posZero}) {
		t.Fatalf("bitExact: -0.0 vs +0.0 have different bits and must not be exact")
	}
}

func TestMaxAbs(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{name: "identical", a: []float32{1, 2, 3}, b: []float32{1, 2, 3}, want: 0},
		{name: "max at end", a: []float32{1, 2, 3}, b: []float32{1, 2, 0}, want: 3},
		{name: "max at start", a: []float32{5, 2, 3}, b: []float32{0, 2, 3}, want: 5},
		{name: "negative diff taken as abs", a: []float32{1, 2}, b: []float32{4, 2}, want: 3},
		{name: "empty slices", a: []float32{}, b: []float32{}, want: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := maxAbs(tc.a, tc.b)
			if math.Abs(got-tc.want) > 1e-6 {
				t.Fatalf("maxAbs(%v, %v) = %g, want %g", tc.a, tc.b, got, tc.want)
			}
		})
	}
}
