// Tests for the pure grid-construction helpers in fleetbench: buildGrid (explicit
// passthrough, generated "full" range with the max<1 floor, and the "log"
// saturation ladder with its filter-by-max and force-include-max branches) and
// parseInts (whitespace trimming and empty-segment skipping). These functions take
// only strings/ints and return []int, so the tests need no external resources.
package main

import (
	"reflect"
	"testing"
)

func TestBuildGrid(t *testing.T) {
	tests := []struct {
		name     string
		explicit string
		max      int
		kind     string
		want     []int
	}{
		{
			name:     "explicit overrides max and kind",
			explicit: "1, 4 ,7",
			max:      50,
			kind:     "log",
			want:     []int{1, 4, 7},
		},
		{
			name: "full small range",
			max:  3,
			kind: "full",
			want: []int{1, 2, 3},
		},
		{
			name: "full clamps max below 1 up to 1",
			max:  0,
			kind: "full",
			want: []int{1},
		},
		{
			name: "log ladder ends exactly on a base value",
			max:  10,
			kind: "log",
			// base values <= 10, last is 10 == max so no append.
			want: []int{1, 2, 3, 4, 5, 6, 8, 10},
		},
		{
			name: "log ladder appends max when not a base value",
			max:  7,
			kind: "log",
			// base values <= 7 are 1..6; last (6) != 7 so 7 is appended.
			want: []int{1, 2, 3, 4, 5, 6, 7},
		},
		{
			name: "log clamps max below 1 up to 1",
			max:  0,
			kind: "log",
			want: []int{1},
		},
		{
			name: "log is case insensitive on kind",
			max:  7,
			kind: "LOG",
			want: []int{1, 2, 3, 4, 5, 6, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildGrid(tt.explicit, tt.max, tt.kind)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("buildGrid(%q, %d, %q) = %v, want %v",
					tt.explicit, tt.max, tt.kind, got, tt.want)
			}
		})
	}
}

func TestParseInts(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []int
	}{
		{
			name: "trims surrounding whitespace",
			in:   " 1, 2 , 3 ",
			want: []int{1, 2, 3},
		},
		{
			name: "skips empty segments",
			in:   "1,,2",
			want: []int{1, 2},
		},
		{
			name: "single value",
			in:   "42",
			want: []int{42},
		},
		{
			name: "negative values pass through Atoi",
			in:   "-1, 0, 5",
			want: []int{-1, 0, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseInts(tt.in)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseInts(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
