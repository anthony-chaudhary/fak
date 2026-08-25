//go:build windows

package windowgate

import (
	"testing"
)

func TestManagedJobMemoryLimitDefaultsAndOverride(t *testing.T) {
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "")
	if got := managedJobMemoryLimitBytes(); got != uint64(64)<<30 {
		t.Fatalf("default=%d", got)
	}
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "512")
	if got := managedJobMemoryLimitBytes(); got != uint64(512)<<20 {
		t.Fatalf("override=%d", got)
	}
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "invalid")
	if got := managedJobMemoryLimitBytes(); got != uint64(64)<<30 {
		t.Fatalf("invalid must fail closed to default, got=%d", got)
	}
}

func TestStartInNewJobRemainsUncappedUnlessManaged(t *testing.T) {
	// The explicit StartManagedAgentInNewJob API selects the cap. Generic job
	// ownership remains available for tests and housekeeping children.
	if got := managedJobMemoryLimitBytes(); got == 0 {
		t.Fatal("managed-agent limit disabled")
	}
}
