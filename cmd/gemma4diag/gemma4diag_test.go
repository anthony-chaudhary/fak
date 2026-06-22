// Unit tests for the pure helpers in the gemma4diag command.
//
// argmax returns the index of the maximum float32 in a slice, scanning
// left-to-right with a strict greater-than comparison so the FIRST maximum
// wins on ties. These table-driven cases pin that contract: ordering,
// tie-breaking, negatives, and single-element input.
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
		{"max at start", []float32{9, 1, 2, 3}, 0},
		{"max in middle", []float32{1, 2, 9, 3}, 2},
		{"max at end", []float32{1, 2, 3, 9}, 3},
		{"first max wins on tie", []float32{5, 5, 5}, 0},
		{"first of two equal maxima", []float32{1, 7, 4, 7, 2}, 1},
		{"all negative", []float32{-3, -1, -2}, 1},
		{"mixed sign", []float32{-10, 0, -0.5, 0.25}, 3},
		{"fractional values", []float32{0.1, 0.3, 0.2}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.in); got != tt.want {
				t.Errorf("argmax(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

// Even when every element sits at the floor used to seed the search, the
// strict comparison must leave the result at the first index rather than
// advancing on an equal value.
func TestArgmaxFloorValues(t *testing.T) {
	in := []float32{-math.MaxFloat32, -math.MaxFloat32}
	if got := argmax(in); got != 0 {
		t.Errorf("argmax(all -MaxFloat32) = %d, want 0", got)
	}
}
