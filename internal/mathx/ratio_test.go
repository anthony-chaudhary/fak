package mathx

import "testing"

func TestAgainstOracle(t *testing.T) {
	for _, tc := range []struct {
		a, o int
		want float64
	}{{0, 0, 1}, {1, 0, 0}, {3, 4, .75}} {
		if got := AgainstOracle(tc.a, tc.o); got != tc.want {
			t.Errorf("AgainstOracle=%g, want %g", got, tc.want)
		}
	}
}
