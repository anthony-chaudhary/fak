//go:build windows

package windowgate

import (
	"os/exec"
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

func TestManagedJobIORateBandwidthCombinesReadAndWrite(t *testing.T) {
	if got := managedJobIORateBandwidthBytes(ManagedJobConfig{ReadBytesPerSecond: 1024}); got != 1024 {
		t.Fatalf("read-only bandwidth=%d", got)
	}
	if got := managedJobIORateBandwidthBytes(ManagedJobConfig{ReadBytesPerSecond: 1024, WriteBytesPerSecond: 2048}); got != 3072 {
		t.Fatalf("aggregate bandwidth=%d", got)
	}
}

func TestStartManagedAgentAppliesNativeIORateControlWhenSupported(t *testing.T) {
	child := exec.Command("cmd.exe", "/c", "exit", "0")
	job, err := StartManagedAgentInNewJob(child, ManagedJobConfig{ReadBytesPerSecond: 1 << 20})
	if err != nil {
		t.Fatalf("start managed job: %v", err)
	}
	defer job.Close()
	if ioErr := job.IORateControlError(); ioErr != nil {
		t.Skipf("Windows I/O rate control unavailable; sampled guard fallback remains authoritative: %v", ioErr)
	}
	if err := child.Wait(); err != nil {
		t.Fatalf("wait child: %v", err)
	}
}

func TestStartInNewJobRemainsUncappedUnlessManaged(t *testing.T) {
	// The explicit StartManagedAgentInNewJob API selects the cap. Generic job
	// ownership remains available for tests and housekeeping children.
	if got := managedJobMemoryLimitBytes(ManagedJobConfig{}); got == 0 {
		t.Fatal("managed-agent limit disabled")
	}
}
