// Tests for the pure, deterministic argmax helper in cmd/diagtok.
//
// argmax returns the index of the first maximal element (ties resolve to the
// lowest index because the comparison is strict `>`), and returns 0 for an
// empty slice. These cases pin that contract.
package main

import (
	"math"
	"testing"
)

func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{"single element", []float32{42}, 0},
		{"max in middle", []float32{0.1, 0.9, 0.3}, 1},
		{"max at end", []float32{1, 2, 3}, 2},
		{"max at start", []float32{3, 2, 1}, 0},
		{"all negative", []float32{-1.5, -0.5, -2.5}, 1},
		{"all equal ties to first", []float32{1, 1, 1}, 0},
		{"duplicate max ties to first", []float32{3, 7, 7, 2}, 1},
		{"empty returns zero", []float32{}, 0},
		{"mixed signs", []float32{-5, -2, -9, 4, 0}, 3},
		{"negative infinity treated as below sentinel", []float32{float32(math.Inf(-1))}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.in); got != tt.want {
				t.Errorf("argmax(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}
