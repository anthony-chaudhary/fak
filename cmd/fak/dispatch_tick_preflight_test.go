package main

import (
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"testing"
)

func TestDispatchHostResourcesUsesSharedProcessThreads(t *testing.T) {
	a, b := 3, 7
	got := dispatchPreflightHostResourcesFromProcesses(dispatchtick.ProcGuardInput{Processes: []dispatchtick.ProcInfo{{Threads: &a}, {Threads: &b}, {}}})
	if got.TotalThreads == nil || *got.TotalThreads != 10 {
		t.Fatalf("resources=%+v", got)
	}
}
