// Package main tests for the pure slice utilities used by the deletioncert
// demonstrator. These functions are deterministic and resource-free: no model,
// no journal, no I/O — so they are unit-testable in isolation with hand-computed
// expected values.
package main

import "testing"

// TestArgmax pins argmax's exact contract: it scans with a strict '>' from an
// initial best of -1e30 at index 0, so it returns the FIRST index of the maximum
// value and returns 0 for an empty slice.
func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{"single", []float32{0}, 0},
		{"ascending", []float32{1, 3, 2}, 1},
		{"max at end", []float32{1, 2, 9}, 2},
		{"first of ties", []float32{5, 5, 1}, 0}, // strict '>' keeps the earliest max
		{"all negative", []float32{-2, -1, -3}, 1},
		{"single negative below init is still picked", []float32{-5}, 0},
		{"empty returns zero", []float32{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.in); got != tt.want {
				t.Fatalf("argmax(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

// TestMaxAbsIntDelta checks the deletion-property witness helper: equal-length
// slices yield the maximum element-wise absolute difference; a length mismatch
// yields the 1<<30 sentinel.
func TestMaxAbsIntDelta(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want int
	}{
		{"identical is zero", []int{3, 3}, []int{3, 3}, 0},
		{"max in second slot", []int{1, 5}, []int{4, 1}, 4}, // |1-4|=3, |5-1|=4
		{"negative diff abs", []int{0}, []int{7}, 7},
		{"sign both directions", []int{-3, 2}, []int{3, -2}, 6}, // |-3-3|=6, |2-(-2)|=4
		{"length mismatch sentinel", []int{1, 2, 3}, []int{1, 2}, 1 << 30},
		{"both empty is zero", []int{}, []int{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := maxAbsIntDelta(tt.a, tt.b); got != tt.want {
				t.Fatalf("maxAbsIntDelta(%v, %v) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

// TestEqualInts verifies element-wise equality with a length guard.
func TestEqualInts(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want bool
	}{
		{"equal", []int{1, 2, 3}, []int{1, 2, 3}, true},
		{"differ in middle", []int{1, 9, 3}, []int{1, 2, 3}, false},
		{"length mismatch", []int{1, 2}, []int{1, 2, 3}, false},
		{"both empty", []int{}, []int{}, true},
		{"empty vs nonempty", []int{}, []int{1}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := equalInts(tt.a, tt.b); got != tt.want {
				t.Fatalf("equalInts(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

// TestConcat checks the variadic flatten preserves order across all parts.
func TestConcat(t *testing.T) {
	got := concat([]int{3, 17, 5}, []int{41, 2}, []int{23, 11})
	want := []int{3, 17, 5, 41, 2, 23, 11}
	if !equalInts(got, want) {
		t.Fatalf("concat = %v, want %v", got, want)
	}

	// An empty middle part contributes nothing and order is kept.
	got2 := concat([]int{1}, []int{}, []int{2, 3})
	want2 := []int{1, 2, 3}
	if !equalInts(got2, want2) {
		t.Fatalf("concat with empty middle = %v, want %v", got2, want2)
	}

	// No parts yields an empty (length-0) result.
	if got3 := concat(); len(got3) != 0 {
		t.Fatalf("concat() = %v, want empty", got3)
	}
}

func TestDeletionCertificateSelfcheckRuns(t *testing.T) {
	if err := run(""); err != nil {
		t.Fatalf("deletioncert selfcheck failed: %v", err)
	}
}
