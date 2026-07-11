package main

import (
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"io"
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

func TestDispatchPreflightTimedSubBuckets(t *testing.T) {
	oldProc, oldKernel, oldResources, oldWorkers := dispatchProbeProcesses, dispatchRunExternalJSON, dispatchProbeHostResources, dispatchProbeWorkerCount
	defer func() {
		dispatchProbeProcesses = oldProc
		dispatchRunExternalJSON = oldKernel
		dispatchProbeHostResources = oldResources
		dispatchProbeWorkerCount = oldWorkers
	}()
	dispatchProbeProcesses = func() dispatchtick.ProcGuardInput { return dispatchtick.ProcGuardInput{} }
	dispatchRunExternalJSON = func(string, time.Duration, string, ...string) (map[string]any, error) {
		return map[string]any{"alive": 1, "target": 1, "verdict": "OK"}, nil
	}
	dispatchProbeHostResources = func() dispatchtick.HostResources { return dispatchtick.HostResources{} }
	dispatchProbeWorkerCount = func(string, string) int { return 0 }
	dispatchKernelCache.Lock()
	dispatchKernelCache.at = time.Time{}
	dispatchKernelCache.root = ""
	dispatchKernelCache.Unlock()
	_, tm, err := dispatchPreflightTimed(t.TempDir(), io.Discard, 1, "engineering", "codex")
	if err != nil {
		t.Fatal(err)
	}
	if tm["process_snapshot"] <= 0 || tm["kernel_probe"] <= 0 {
		t.Fatalf("timings=%v", tm)
	}
}
