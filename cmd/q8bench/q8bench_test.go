// Package main tests for q8bench's pure, deterministic helpers.
//
// These cover the resource-free numeric helpers in main.go — the fixed-seed
// LCG id generator, argmax, and the min/median duration reducers. They need no
// model file, GPU, or network: every expected value is derived directly from
// the documented recurrence / arithmetic and verified against the real code.
package main

import (
	"testing"
	"time"
)

func TestLcgIDs(t *testing.T) {
	// The recurrence is fixed-seed (state0 = 2463534242) and deterministic, so
	// its output is reproducible bit-for-bit. The first five raw masked states
	// are 1266642227, 1626945776, 857116265, 1848955118, 171551119; taken mod
	// 100 they yield the sequence below.
	tests := []struct {
		name  string
		n     int
		vocab int
		want  []int
	}{
		{"n0 empty", 0, 100, []int{}},
		{"first five mod100", 5, 100, []int{27, 76, 65, 18, 19}},
		{"vocab1 all zero", 3, 1, []int{0, 0, 0}},
		{"vocab10", 8, 10, []int{7, 6, 5, 8, 9, 0, 1, 6}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lcgIDs(tt.n, tt.vocab)
			if len(got) != tt.n {
				t.Fatalf("len = %d, want %d", len(got), tt.n)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("len mismatch: got %v want %v", got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("ids[%d] = %d, want %d (full got %v)", i, got[i], tt.want[i], got)
				}
			}
		})
	}

	// Determinism: two independent calls with the same args agree exactly.
	a := lcgIDs(16, 257)
	b := lcgIDs(16, 257)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at %d: %d != %d", i, a[i], b[i])
		}
	}

	// Every id must be a valid index into [0,vocab).
	const vocab = 50
	for i, id := range lcgIDs(64, vocab) {
		if id < 0 || id >= vocab {
			t.Errorf("ids[%d] = %d out of range [0,%d)", i, id, vocab)
		}
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want int
	}{
		{"single", []float32{7}, 0},
		{"max at start", []float32{9, 1, 2, 3}, 0},
		{"max at end", []float32{1, 2, 3, 9}, 3},
		{"max in middle", []float32{1, 8, 2}, 1},
		{"tie returns first", []float32{1, 3, 3, 2}, 1},
		{"all negative", []float32{-5, -2, -9}, 1},
		{"all equal returns first", []float32{4, 4, 4}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.v); got != tt.want {
				t.Errorf("argmax(%v) = %d, want %d", tt.v, got, tt.want)
			}
		})
	}
}

func TestMinMS(t *testing.T) {
	tests := []struct {
		name string
		ds   []time.Duration
		want float64
	}{
		{"single", []time.Duration{5 * time.Millisecond}, 5},
		{"min first", []time.Duration{1 * time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond}, 1},
		{"min last", []time.Duration{3 * time.Millisecond, 2 * time.Millisecond, 1 * time.Millisecond}, 1},
		{"sub-millisecond", []time.Duration{1500 * time.Microsecond, 2 * time.Millisecond}, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := minMS(tt.ds); got != tt.want {
				t.Errorf("minMS(%v) = %v, want %v", tt.ds, got, tt.want)
			}
		})
	}
}

func TestMedianMS(t *testing.T) {
	// medianMS sorts a copy and returns element at index len/2 (the upper of the
	// two middle elements for even-length inputs). It must not mutate its input.
	tests := []struct {
		name string
		ds   []time.Duration
		want float64
	}{
		{"single", []time.Duration{5 * time.Millisecond}, 5},
		{"odd unsorted", []time.Duration{3 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond}, 2},
		{"even upper-middle", []time.Duration{4 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond}, 3},
		{"fractional", []time.Duration{1500 * time.Microsecond}, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := medianMS(tt.ds); got != tt.want {
				t.Errorf("medianMS(%v) = %v, want %v", tt.ds, got, tt.want)
			}
		})
	}

	// medianMS must leave its argument unsorted (it sorts a copy).
	in := []time.Duration{3 * time.Millisecond, 1 * time.Millisecond, 2 * time.Millisecond}
	_ = medianMS(in)
	if in[0] != 3*time.Millisecond || in[1] != 1*time.Millisecond || in[2] != 2*time.Millisecond {
		t.Errorf("medianMS mutated its input: %v", in)
	}
}
