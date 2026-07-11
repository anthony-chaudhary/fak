package main

import (
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"testing"
	"time"
)

func TestDispatchHostResourcesUsesSharedProcessThreads(t *testing.T) {
	a, b := 3, 7
	got := dispatchPreflightHostResourcesFromProcesses(dispatchtick.ProcGuardInput{Processes: []dispatchtick.ProcInfo{{Threads: &a}, {Threads: &b}, {}}})
	if got.TotalThreads == nil || *got.TotalThreads != 10 {
		t.Fatalf("resources=%+v", got)
	}
}

func TestDispatchPreflightKernelCachesProbe(t *testing.T) {
	old := dispatchRunExternalJSON
	defer func() { dispatchRunExternalJSON = old }()
	calls := 0
	dispatchRunExternalJSON = func(string, time.Duration, string, ...string) (map[string]any, error) {
		calls++
		return map[string]any{"alive": 2, "target": 3, "verdict": "OK"}, nil
	}
	dispatchKernelCache.Lock()
	dispatchKernelCache.at = time.Time{}
	dispatchKernelCache.root = ""
	dispatchKernelCache.Unlock()
	root := t.TempDir()
	a := dispatchPreflightKernel(root)
	b := dispatchPreflightKernel(root)
	if calls != 1 || a.Alive == nil || b.Target == nil {
		t.Fatalf("calls=%d a=%+v b=%+v", calls, a, b)
	}
}
