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
		{90, 910, RegimeCold},
		{100, 900, RegimePartial},
		{890, 110, RegimePartial},
		{90, 10, RegimeFrozen},
		{65, 0, RegimeFrozen},
		// Header-only reuse on a short cold prompt (Halo witness: 18 cached / 45 uncached).
		{18, 45, RegimeCold},
		{MinPartialReuseTokens - 1, 100, RegimeCold},
		{MinPartialReuseTokens, 100, RegimePartial},
		// A tiny prompt served wholly from cache stays frozen.
		{30, 2, RegimeFrozen},
	}
	for _, c := range cases {
		if got := RegimeForTokens(c.cached, c.uncached); got != c.want {
			t.Errorf("RegimeForTokens(%d, %d) = %q, want %q", c.cached, c.uncached, got, c.want)
		}
	}
}
