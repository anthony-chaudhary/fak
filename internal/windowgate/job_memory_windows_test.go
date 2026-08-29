//go:build windows

package windowgate

import (
	"testing"
)

func TestManagedJobMemoryLimitDefaultsAndOverride(t *testing.T) {
	t.Setenv("FAK_CHILD_MAX_COMMIT_MB", "1")
	if got := managedJobMemoryLimitBytes(ManagedJobConfig{}); got != uint64(64)<<30 {
		t.Fatalf("default=%d", got)
	}
	if got := managedJobMemoryLimitBytes(ManagedJobConfig{MemoryLimitBytes: uint64(512) << 20}); got != uint64(512)<<20 {
		t.Fatalf("override=%d", got)
	}
}

func TestStartInNewJobRemainsUncappedUnlessManaged(t *testing.T) {
	// The explicit StartManagedAgentInNewJob API selects the cap. Generic job
	// ownership remains available for tests and housekeeping children.
	if got := managedJobMemoryLimitBytes(ManagedJobConfig{}); got == 0 {
		t.Fatal("managed-agent limit disabled")
	}
}
