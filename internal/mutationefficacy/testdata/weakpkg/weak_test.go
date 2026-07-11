package weakpkg

import "testing"

// TestIsPositive is DELIBERATELY WEAK: it checks one positive input and never the boundary (0)
// or a negative, so it cannot tell `n > 0` from `n >= 0`. The mutation-efficacy probe exists to
// surface exactly this blind spot as a surviving mutant.
func TestIsPositive(t *testing.T) {
	if !IsPositive(5) {
		t.Fatal("IsPositive(5) should be true")
	}
}
