package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestGuardResourceBudgetStopsOutOfControlDiskReader(t *testing.T) {
	oldSessions := serveSessions
	serveSessions = session.NewTable()
	t.Cleanup(func() { serveSessions = oldSessions })

	const trace = "resource-budget-disk-witness"
	budget := session.ResourceBudget{ReadBytesPerSecond: 1 << 20}
	serveSessions.SetResourceBudget(trace, budget)

	var reads uint64
	collectMemory := func(int) (procguard.MemorySnapshot, bool, string) {
		return procguard.MemorySnapshot{RootPID: 42, TreeBytes: 1, Processes: []procguard.MemoryProcess{{PID: 42}}}, true, ""
	}
	collectResource := func(int) (procguard.ResourceSnapshot, bool, string) {
		reads += 1 << 30
		return procguard.ResourceSnapshot{RootPID: 42, ReadBytes: reads, WriteBytes: 0, HaveIO: true, HaveProcessCount: true, ProcessCount: 1}, true, ""
	}

	events := startGuardChildResourceMonitorWithCollectors(42, trace, "test-agent", guardResourcePolicy{
		PollInterval:   10 * time.Millisecond,
		ResourceBudget: budget,
	}, collectMemory, collectResource)

	select {
	case event := <-events:
		if event.Kind != guardChildResourceLimit || !event.Resource.Stop {
			t.Fatalf("event = %+v, want resource stop", event)
		}
		if event.Reason == "" || event.Resource.ResourceAxis != session.ResourceAxisReadRate {
			t.Fatalf("reason/axis = %q/%q", event.Reason, event.Resource.ResourceAxis)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("resource budget did not stop the disk reader")
	}
}
