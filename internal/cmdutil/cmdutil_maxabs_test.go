package cmdutil

import (
	"math"
	"testing"
)

func TestMaxAbsDiffF32RefusesNaN(t *testing.T) {
	nan := float32(math.NaN())
	cases := []struct {
		name string
		a, b []float32
	}{
		{"nan-in-a", []float32{nan}, []float32{1}},
		{"nan-in-b", []float32{1}, []float32{nan}},
		{"nan-in-both", []float32{nan}, []float32{nan}},
		{"nan-behind-large-finite", []float32{0, nan, 0}, []float32{100, 0, 0}},
	}
	for _, tc := range cases {
		if got := MaxAbsDiffF32(tc.a, tc.b); !math.IsInf(got, 1) {
			t.Errorf("%s: MaxAbsDiffF32 = %v, want +Inf", tc.name, got)
		}
	}
}

func TestMaxAbsDiffF32RefusesLengthMismatch(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
	}{
		{"longer-a", []float32{1, 2, 100}, []float32{1, 2}},
		{"longer-b", []float32{1, 2}, []float32{1, 2, 100}},
		{"nil-a", nil, []float32{5}},
		{"nil-b", []float32{5}, nil},
	}
	for _, tc := range cases {
		if got := MaxAbsDiffF32(tc.a, tc.b); !math.IsInf(got, 1) {
			t.Errorf("%s: MaxAbsDiffF32 = %v, want +Inf", tc.name, got)
		}
	}
	if got := MaxAbsDiffF32(nil, []float32{}); got != 0 {
		t.Errorf("nil vs empty: MaxAbsDiffF32 = %v, want 0", got)
	}
}

func TestMaxAbsDiffF32InfinityContract(t *testing.T) {
	inf := float32(math.Inf(1))
	cases := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"pos-inf-equal", []float32{inf}, []float32{inf}, 0},
		{"neg-inf-equal", []float32{-inf}, []float32{-inf}, 0},
		{"pos-inf-vs-finite", []float32{inf}, []float32{1}, math.Inf(1)},
		{"neg-inf-vs-pos-inf", []float32{-inf}, []float32{inf}, math.Inf(1)},
		{"finite", []float32{1, -4, 3}, []float32{1, 2, 2.5}, 6},
		{"finite-half", []float32{1, 2}, []float32{1.5, 2}, 0.5},
	}
	for _, tc := range cases {
		if got := MaxAbsDiffF32(tc.a, tc.b); got != tc.want {
			t.Errorf("%s: MaxAbsDiffF32 = %v, want %v", tc.name, got, tc.want)
		}
	}
}
