package cacheobs

import "testing"

// fak-test:runtime fast est=5ms lane=default
func TestRegimeForTokensBucketsByCliffThresholds(t *testing.T) {
	cases := []struct {
		cached, uncached int
		want             string
	}{
		{0, 0, RegimeUnknown},
		{-3, 0, RegimeUnknown},
		{0, 100, RegimeCold},
		{9, 91, RegimeCold},
		{10, 90, RegimePartial},
		{89, 11, RegimePartial},
		{90, 10, RegimeFrozen},
		{65, 0, RegimeFrozen},
	}
	for _, c := range cases {
		if got := RegimeForTokens(c.cached, c.uncached); got != c.want {
			t.Errorf("RegimeForTokens(%d, %d) = %q, want %q", c.cached, c.uncached, got, c.want)
		}
	}
}
