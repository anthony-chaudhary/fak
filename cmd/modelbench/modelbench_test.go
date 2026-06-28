// Unit tests for the pure helpers in modelbench's main package: the CSV
// size parser, the prompt-length clamp, and the greedy-token argmax. All three
// are deterministic and resource-free (no model file, GPU, or network), so the
// expected values below are computed by hand from the functions' actual logic.
package main

import (
	"math"
	"reflect"
	"testing"
	"time"
)

func TestParsePositiveInts(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []int
		wantErr bool
	}{
		{name: "default sizes", in: "16,64,256", want: []int{16, 64, 256}},
		{name: "single", in: "8", want: []int{8}},
		{name: "trims whitespace", in: " 1 , 2 ", want: []int{1, 2}},
		{name: "skips empty fields", in: "3,,4", want: []int{3, 4}},
		{name: "trailing comma", in: "5,", want: []int{5}},
		{name: "empty string", in: "", wantErr: true},
		{name: "only separators", in: " , ", wantErr: true},
		{name: "zero rejected", in: "0", wantErr: true},
		{name: "negative rejected", in: "-5", wantErr: true},
		{name: "non-numeric rejected", in: "abc", wantErr: true},
		{name: "mixed valid then invalid", in: "4,nope", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePositiveInts(tt.in)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parsePositiveInts(%q) = %v, want error", tt.in, got)
				}
				if got != nil {
					t.Fatalf("parsePositiveInts(%q) returned %v on error, want nil slice", tt.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parsePositiveInts(%q) unexpected error: %v", tt.in, err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parsePositiveInts(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestCapPositive(t *testing.T) {
	tests := []struct {
		name string
		n    int
		cap  int
		want int
	}{
		{name: "no cap passes through", n: 10, cap: 0, want: 10},
		{name: "over cap is clamped", n: 100, cap: 50, want: 50},
		{name: "exactly at cap unchanged", n: 50, cap: 50, want: 50},
		{name: "under cap unchanged", n: 3, cap: 5, want: 3},
		{name: "zero with no cap floored to one", n: 0, cap: 0, want: 1},
		{name: "negative floored to one", n: -3, cap: 0, want: 1},
		{name: "zero under positive cap floored to one", n: 0, cap: 5, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := capPositive(tt.n, tt.cap); got != tt.want {
				t.Fatalf("capPositive(%d, %d) = %d, want %d", tt.n, tt.cap, got, tt.want)
			}
		})
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{name: "middle max", in: []float32{0.1, 0.9, 0.3}, want: 1},
		{name: "first on ties", in: []float32{5, 5, 5}, want: 0},
		{name: "all negative", in: []float32{-3, -1, -2}, want: 1},
		{name: "single element", in: []float32{42}, want: 0},
		{name: "last is max", in: []float32{1, 2, 3, 4}, want: 3},
		{name: "empty slice", in: []float32{}, want: 0},
		{name: "below default floor", in: []float32{-math.MaxFloat32 / 2}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.in); got != tt.want {
				t.Fatalf("argmax(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

func TestSmokeOutcome(t *testing.T) {
	const dl = 90 * time.Second
	tests := []struct {
		name     string
		done     bool
		elapsed  time.Duration
		deadline time.Duration
		want     string
	}{
		{name: "finished under deadline", done: true, elapsed: 5 * time.Second, deadline: dl, want: smokeStatusLoaded},
		{name: "not finished (deadline fired)", done: false, elapsed: dl, deadline: dl, want: smokeStatusTimeout},
		{name: "finished but over deadline", done: true, elapsed: 2 * dl, deadline: dl, want: smokeStatusTimeout},
		{name: "finished, no deadline set", done: true, elapsed: time.Hour, deadline: 0, want: smokeStatusLoaded},
		{name: "exactly at deadline counts as loaded", done: true, elapsed: dl, deadline: dl, want: smokeStatusLoaded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := smokeOutcome(tt.done, tt.elapsed, tt.deadline); got != tt.want {
				t.Fatalf("smokeOutcome(%v, %v, %v) = %s, want %s", tt.done, tt.elapsed, tt.deadline, got, tt.want)
			}
		})
	}
}

func TestAllFinite(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want bool
	}{
		{name: "all finite", in: []float32{-1, 0, 3.5, 1e9}, want: true},
		{name: "empty is not a valid forward result", in: []float32{}, want: false},
		{name: "contains NaN", in: []float32{1, float32(math.NaN()), 3}, want: false},
		{name: "contains +Inf", in: []float32{1, float32(math.Inf(1)), 3}, want: false},
		{name: "contains -Inf", in: []float32{float32(math.Inf(-1)), 2}, want: false},
		{name: "single finite", in: []float32{42}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := allFinite(tt.in); got != tt.want {
				t.Fatalf("allFinite(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
