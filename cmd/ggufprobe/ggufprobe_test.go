// Tests for the pure metadata-rendering and helper functions in ggufprobe's
// main package. summarize collapses long metadata slices to a count + head and
// truncates over-long scalar renderings; maxInt is the floor-1 guard used by
// the dump path. Both are deterministic and need no GGUF file, GPU, or I/O.
package main

import (
	"fmt"
	"strings"
	"testing"
)

func TestSummarize(t *testing.T) {
	// A scalar string rendering longer than 80 runes: summarize must keep the
	// first 77 bytes and append "..." for a total length of exactly 80.
	long := strings.Repeat("x", 100)
	wantLong := long[:77] + "..."

	tests := []struct {
		name string
		in   any
		want string
	}{
		{
			name: "short string slice rendered verbatim",
			in:   []string{"a", "b"},
			want: `["a" "b"]`,
		},
		{
			name: "string slice at boundary (len 6) rendered verbatim",
			in:   []string{"a", "b", "c", "d", "e", "f"},
			want: `["a" "b" "c" "d" "e" "f"]`,
		},
		{
			name: "long string slice collapsed to count + first 4",
			in:   []string{"a", "b", "c", "d", "e", "f", "g"},
			want: `[]string len=7  e.g. ["a" "b" "c" "d"]`,
		},
		{
			name: "short int32 slice rendered verbatim",
			in:   []int32{1, 2, 3},
			want: "[1 2 3]",
		},
		{
			name: "int32 slice at boundary (len 8) rendered verbatim",
			in:   []int32{1, 2, 3, 4, 5, 6, 7, 8},
			want: "[1 2 3 4 5 6 7 8]",
		},
		{
			name: "long int32 slice collapsed to count + first 6",
			in:   []int32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want: "[]int32 len=9  e.g. [1 2 3 4 5 6]",
		},
		{
			name: "short float32 slice rendered verbatim",
			in:   []float32{1.5, 2.5},
			want: "[1.5 2.5]",
		},
		{
			name: "long float32 slice collapsed to count only",
			in:   []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want: "[]float32 len=9",
		},
		{
			name: "scalar int via default branch",
			in:   42,
			want: "42",
		},
		{
			name: "short scalar string untouched",
			in:   "hello",
			want: "hello",
		},
		{
			name: "over-long scalar string truncated to 80",
			in:   long,
			want: wantLong,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(tt.in)
			if got != tt.want {
				t.Errorf("summarize(%v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}

	// Reinforce the truncation invariant: the collapsed form is exactly 80
	// bytes (77 kept + 3 dots).
	if got := summarize(long); len(got) != 80 {
		t.Errorf("summarize(long) length = %d, want 80", len(got))
	}
	if got := summarize(long); !strings.HasSuffix(got, "...") {
		t.Errorf("summarize(long) = %q, want trailing ...", got)
	}
}

func TestMaxInt(t *testing.T) {
	tests := []struct {
		a, b, want int
	}{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{-5, -3, -3},
		{7, 7, 7},
		{-1, 2, 2},
		{100, 99, 100},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("max(%d,%d)", tt.a, tt.b), func(t *testing.T) {
			if got := maxInt(tt.a, tt.b); got != tt.want {
				t.Errorf("maxInt(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
		})
	}
}
