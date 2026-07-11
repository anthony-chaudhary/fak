package strongpkg

import "testing"

// TestIsPositive is STRONG: it pins the boundary (0 is not positive) and a negative, so the
// `>` -> `>=` mutant (which would make 0 positive) fails the suite and is killed. This is the
// control that proves the probe reports 0 survivors when the test actually catches the change.
func TestIsPositive(t *testing.T) {
	if !IsPositive(1) {
		t.Fatal("1 must be positive")
	}
	if IsPositive(0) {
		t.Fatal("0 must not be positive")
	}
	if IsPositive(-1) {
		t.Fatal("-1 must not be positive")
	}
}
