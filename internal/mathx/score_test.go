package mathx

import "testing"

func TestClampScore(t *testing.T) {
	for _, tc := range []struct {
		in   float64
		want int
	}{{-1, 0}, {2.5, 2}, {3.5, 4}, {99.6, 100}, {101, 100}} {
		if got := ClampScore(tc.in); got != tc.want {
			t.Errorf("ClampScore(%g)=%d, want %d", tc.in, got, tc.want)
		}
	}
}
