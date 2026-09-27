package safecommit

import (
	"testing"
	"time"
)

// TestPostValidationTimeoutCoversHooksAndStaysBelowStaleAge pins both ends of the commit
// budget. The default must exceed the measured 2-3 minute pre-commit hook set on the
// Windows fleet host (the 30s default killed every commit there mid-hook), and no
// configured budget may reach the age at which peers' reapers treat a frozen index.lock
// as abandoned — a commit still inside its hooks would lose its live lock.
func TestPostValidationTimeoutCoversHooksAndStaysBelowStaleAge(t *testing.T) {
	const measuredHookSet = 3 * time.Minute
	if DefaultPostValidationTimeout <= measuredHookSet {
		t.Fatalf("DefaultPostValidationTimeout %s does not cover the measured %s hook set", DefaultPostValidationTimeout, measuredHookSet)
	}
	if MaxPostValidationTimeout >= DefaultStaleIndexLockAge {
		t.Fatalf("MaxPostValidationTimeout %s reaches the %s stale-index age", MaxPostValidationTimeout, DefaultStaleIndexLockAge)
	}
	if DefaultPostValidationTimeout > MaxPostValidationTimeout {
		t.Fatalf("the default %s exceeds the cap %s", DefaultPostValidationTimeout, MaxPostValidationTimeout)
	}
	for _, tc := range []struct{ in, want time.Duration }{
		{0, 0},
		{7 * time.Minute, 7 * time.Minute},
		{MaxPostValidationTimeout, MaxPostValidationTimeout},
		{time.Hour, MaxPostValidationTimeout},
	} {
		if got := clampPostValidationTimeout(tc.in); got != tc.want {
			t.Errorf("clampPostValidationTimeout(%s) = %s, want %s", tc.in, got, tc.want)
		}
	}
}
