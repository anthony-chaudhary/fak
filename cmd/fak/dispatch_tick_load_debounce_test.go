package main

import (
	"io"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// The #3376 witnesses, driven through the PRODUCTION seam rather than the primitive.
//
// internal/loaddebounce already proves the dedup and the coalescing window on the
// Coalescer/Publisher in isolation. What those tests cannot say is whether the dispatch
// tick actually CONSUMES a published load: before the wiring, dispatchPreflightTimed put
// the raw per-tick worker-count probe straight into PreflightInput.OSWorkerProcs, so every
// sample -- including a single-tick blip -- moved the cap arithmetic. These tests drive the
// real dispatchPreflightTimed and read the load off its payload, so they fail if the probe
// is ever re-attached directly to admission.

// dispatchLoadDebounceWindow is the coalescing window these tests configure through the
// operator knob. The clock is advanced in multiples of it so the intent (inside the window
// vs past it) reads off the call site instead of a bare duration.
const dispatchLoadDebounceWindow = 50 * time.Millisecond

// dispatchLoadDebounceClock is a hand-cranked monotonic clock: the coalescing window is
// measured on injected time, so these tests never sleep and cannot flake on a loaded host.
type dispatchLoadDebounceClock struct{ at time.Time }

func (c *dispatchLoadDebounceClock) now() time.Time          { return c.at }
func (c *dispatchLoadDebounceClock) advance(d time.Duration) { c.at = c.at.Add(d) }

// stubDispatchLoadDebounceTick silences every expensive/host-dependent probe the preflight
// pulls (process table, kernel, tree build, the slow-cadence probes, the codex process
// scans) so a tick is pure arithmetic, points the worker-count probe at *sample so a test
// scripts the load stream, installs a fresh signal at cold start, and returns the clock the
// coalescing window is measured on. Everything is restored on cleanup -- the signal is
// process-global and other cmd/fak tests tick the same preflight.
func stubDispatchLoadDebounceTick(t *testing.T, sample *int) *dispatchLoadDebounceClock {
	t.Helper()
	t.Setenv(dispatchWorkerLoadDebounceEnv, strconv.Itoa(int(dispatchLoadDebounceWindow/time.Millisecond)))

	oldProcesses, oldKernel, oldResources := dispatchProbeProcesses, dispatchRunExternalJSON, dispatchProbeHostResources
	oldWorkers, oldTree := dispatchProbeWorkerCount, dispatchTreeBuildCommand
	oldHooklat, oldRate, oldFresh := dispatchProbeHooklat, dispatchProbeRateLimit, dispatchProbeFreshSeat
	oldCodexRows, oldWorkerRows := dispatchProbeCodexProcessRows, dispatchProbeWorkerProcessRows
	oldNow, oldSignal := dispatchWorkerLoadNow, dispatchWorkerLoad
	t.Cleanup(func() {
		dispatchProbeProcesses, dispatchRunExternalJSON, dispatchProbeHostResources = oldProcesses, oldKernel, oldResources
		dispatchProbeWorkerCount, dispatchTreeBuildCommand = oldWorkers, oldTree
		dispatchProbeHooklat, dispatchProbeRateLimit, dispatchProbeFreshSeat = oldHooklat, oldRate, oldFresh
		dispatchProbeCodexProcessRows, dispatchProbeWorkerProcessRows = oldCodexRows, oldWorkerRows
		dispatchWorkerLoadNow, dispatchWorkerLoad = oldNow, oldSignal
	})

	dispatchProbeProcesses = func() dispatchtick.ProcGuardInput { return dispatchtick.ProcGuardInput{} }
	dispatchRunExternalJSON = func(string, time.Duration, string, ...string) (map[string]any, error) {
		return map[string]any{"alive": 0, "target": 0, "verdict": "OK"}, nil
	}
	dispatchProbeHostResources = func() dispatchtick.HostResources { return dispatchtick.HostResources{} }
	dispatchProbeWorkerCount = func(string, string) int { return *sample }
	dispatchTreeBuildCommand = func(string) (string, error) { return "", nil }
	dispatchProbeHooklat = func(string) dispatchtick.GateCheck { return dispatchtick.GateCheck{} }
	dispatchProbeRateLimit = func(string, string) dispatchtick.RateLimitCheck { return dispatchtick.RateLimitCheck{} }
	dispatchProbeFreshSeat = func(string, string) int { return 0 }
	dispatchProbeCodexProcessRows = func() ([]dispatchCodexProcessRow, error) { return nil, nil }
	dispatchProbeWorkerProcessRows = func() ([]dispatchCodexProcessRow, error) { return nil, nil }

	clk := &dispatchLoadDebounceClock{at: time.Unix(0, 0).UTC()}
	dispatchWorkerLoadNow = clk.now
	dispatchWorkerLoad = &dispatchWorkerLoadSignal{}

	dispatchKernelCache.Lock()
	dispatchKernelCache.at, dispatchKernelCache.root = time.Time{}, ""
	dispatchKernelCache.Unlock()
	return clk
}

// dispatchLoadDebounceTick runs one real preflight tick and returns its payload.
func dispatchLoadDebounceTick(t *testing.T, root string) map[string]any {
	t.Helper()
	out, _, err := dispatchPreflightTimed(root, io.Discard, 8, "engineering", "codex")
	if err != nil {
		t.Fatalf("preflight tick: %v", err)
	}
	return out
}

// Witness #1 through the production seam: N identical consecutive samples publish ONCE.
// The probe still runs every tick (samples == ticks), but only the first observation ever
// reaches admission -- a steady fleet holds one load value across arbitrarily many ticks.
func TestDispatchPreflightPublishesWorkerLoadOnlyOnChange(t *testing.T) {
	sample := 3
	clk := stubDispatchLoadDebounceTick(t, &sample)
	root := t.TempDir()

	const ticks = 6
	for i := 0; i < ticks; i++ {
		out := dispatchLoadDebounceTick(t, root)
		if got := out["os_worker_procs"]; got != 3 {
			t.Fatalf("tick %d: os_worker_procs=%v, want the published load 3", i, got)
		}
		if raw, ok := out["os_worker_procs_sampled"]; ok {
			t.Fatalf("tick %d: an unchanged sample surfaced a lag readout %v", i, raw)
		}
		clk.advance(10 * dispatchLoadDebounceWindow) // well past it: nothing can be left pending
	}

	if got := dispatchWorkerLoad.samples; got != ticks {
		t.Fatalf("probe ran %d times over %d ticks, want one sample per tick", got, ticks)
	}
	if got := dispatchWorkerLoad.publishes; got != 1 {
		t.Fatalf("%d identical samples published %d times to admission, want exactly 1", ticks, got)
	}
}

// Witness #2 through the production seam: a burst of CHANGING samples inside one window
// publishes only the LATEST. The intermediate loads never reach the cap arithmetic; the
// raw sample is still visible on the payload so the lag is legible to an operator.
func TestDispatchPreflightCoalescesWorkerLoadBurstToLatest(t *testing.T) {
	sample := 2
	clk := stubDispatchLoadDebounceTick(t, &sample)
	root := t.TempDir()

	// Cold start: with nothing published there is nothing to debounce against, so the
	// first observed load is primed straight through to admission.
	if got := dispatchLoadDebounceTick(t, root)["os_worker_procs"]; got != 2 {
		t.Fatalf("cold start: os_worker_procs=%v, want the first observed load 2", got)
	}

	// Three distinct loads, each observed inside the previous one's window: every one
	// resets the deadline, so admission keeps consuming the value that was already
	// standing and none of the burst members lands.
	for _, v := range []int{5, 9, 4} {
		sample = v
		clk.advance(dispatchLoadDebounceWindow / 10)
		out := dispatchLoadDebounceTick(t, root)
		if got := out["os_worker_procs"]; got != 2 {
			t.Fatalf("mid-burst sample %d: os_worker_procs=%v, want the standing published load 2", v, got)
		}
		if got, ok := out["os_worker_procs_sampled"]; !ok || got != v {
			t.Fatalf("mid-burst sample %d: lag readout=%v (present=%v), want the raw sample", v, got, ok)
		}
	}
	if got := dispatchWorkerLoad.publishes; got != 1 {
		t.Fatalf("mid-burst publishes=%d, want 1 (only the cold-start prime)", got)
	}

	// The burst settles: the window finally elapses with the LAST value still standing.
	clk.advance(4 * dispatchLoadDebounceWindow)
	out := dispatchLoadDebounceTick(t, root)
	if got := out["os_worker_procs"]; got != 4 {
		t.Fatalf("settled: os_worker_procs=%v, want the LATEST burst value 4", got)
	}
	if raw, ok := out["os_worker_procs_sampled"]; ok {
		t.Fatalf("settled: lag readout %v still attached after the value published", raw)
	}
	if got := dispatchWorkerLoad.publishes; got != 2 {
		t.Fatalf("a burst of 3 changes published %d times, want 2 (the cold-start prime plus one settled value)", got)
	}
	if got := dispatchWorkerLoad.published; got != 4 {
		t.Fatalf("published load=%d, want 4", got)
	}
}
