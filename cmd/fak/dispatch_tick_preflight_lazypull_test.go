package main

import (
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// resetLazyPullState clears the process-lifetime dual-cadence state so each test
// starts from a never-run consumer registry (mirroring the kernel-cache resets in
// dispatch_tick_preflight_test.go).
func resetLazyPullState() {
	dispatchLazyPullState.Lock()
	dispatchLazyPullState.root = ""
	dispatchLazyPullState.product = ""
	dispatchLazyPullState.lastRun = nil
	dispatchLazyPullState.probes = dispatchSlowProbes{}
	dispatchLazyPullState.Unlock()
}

// stubSlowProbes replaces the three expensive probe seams with counting fakes whose
// returned observations still abstain in every fold (zero-pressure gate, idle rate
// budget, no-signal ceiling), so a wired preflight verdict is untouched.
func stubSlowProbes(t *testing.T) (gateCalls, rateCalls, seatCalls *int) {
	t.Helper()
	oldGate, oldRate, oldSeat := dispatchProbeHooklat, dispatchProbeRateLimit, dispatchProbeFreshSeat
	t.Cleanup(func() {
		dispatchProbeHooklat, dispatchProbeRateLimit, dispatchProbeFreshSeat = oldGate, oldRate, oldSeat
	})
	g, r, s := 0, 0, 0
	dispatchProbeHooklat = func(string) dispatchtick.GateCheck {
		g++
		return dispatchtick.GateCheck{HookBudgetMS: 123}
	}
	dispatchProbeRateLimit = func(string, string) dispatchtick.RateLimitCheck {
		r++
		return dispatchtick.RateLimitCheck{Window: time.Minute, Threshold: 7}
	}
	dispatchProbeFreshSeat = func(string, string) int {
		s++
		return 0
	}
	return &g, &r, &s
}

// The #3371 witness: with the dual cadence armed, an expensive probe is gathered on
// the base tick where its never-run consumer is due, SKIPPED on a tick where no due
// consumer needs it (the last pulled observation is served, no re-gather), and
// gathered again once the consumer's interval elapses.
func TestDispatchLazyPullSkipsProbeWhenNoDueConsumerNeedsIt(t *testing.T) {
	t.Setenv(dispatchLazyCadenceEnv, "1m")
	gate, rate, seat := stubSlowProbes(t)
	resetLazyPullState()
	t.Cleanup(resetLazyPullState)

	base := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	first, skipped := dispatchGatherSlowProbes("ws", "claude", base)
	if *gate != 1 || *rate != 1 || *seat != 1 {
		t.Fatalf("first tick gathers = %d/%d/%d, want 1/1/1 (never-run consumers are due)", *gate, *rate, *seat)
	}
	if len(skipped) != 0 {
		t.Fatalf("first tick skipped = %v, want none", skipped)
	}

	// 10s into a 1m cadence: no consumer is due, so no due consumer needs any probe
	// -- every expensive gather is skipped and the last pulled observations served.
	second, skipped := dispatchGatherSlowProbes("ws", "claude", base.Add(10*time.Second))
	if *gate != 1 || *rate != 1 || *seat != 1 {
		t.Fatalf("undue tick re-gathered (%d/%d/%d), want the expensive probes skipped", *gate, *rate, *seat)
	}
	if !reflect.DeepEqual(second, first) {
		t.Fatalf("undue tick observations = %+v, want the cached %+v served unchanged", second, first)
	}
	want := []string{dispatchProbePathHooklat, dispatchProbePathRateExits, dispatchProbePathFreshSeat}
	if !reflect.DeepEqual(skipped, want) {
		t.Fatalf("skipped = %v, want %v", skipped, want)
	}

	// Past the interval the consumers are due again -> the probes are gathered.
	if _, skipped = dispatchGatherSlowProbes("ws", "claude", base.Add(61*time.Second)); len(skipped) != 0 {
		t.Fatalf("due tick skipped %v, want none", skipped)
	}
	if *gate != 2 || *rate != 2 || *seat != 2 {
		t.Fatalf("due tick gathers = %d/%d/%d, want 2/2/2", *gate, *rate, *seat)
	}
}

// Unarmed cadence (the default): every consumer is due every base tick, so every
// probe is gathered every tick -- byte-identical to the pre-#3371 behavior. A scope
// change (another workspace/backend) must also drop the cache and re-gather.
func TestDispatchLazyPullUnarmedGathersEveryTick(t *testing.T) {
	t.Setenv(dispatchLazyCadenceEnv, "")
	gate, rate, seat := stubSlowProbes(t)
	resetLazyPullState()
	t.Cleanup(resetLazyPullState)
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 3; i++ {
		if _, skipped := dispatchGatherSlowProbes("ws", "claude", at.Add(time.Duration(i)*time.Millisecond)); len(skipped) != 0 {
			t.Fatalf("tick %d skipped %v with the due-filter unarmed", i, skipped)
		}
	}
	if *gate != 3 || *rate != 3 || *seat != 3 {
		t.Fatalf("gathers = %d/%d/%d, want 3/3/3 (every base tick)", *gate, *rate, *seat)
	}
	t.Setenv(dispatchLazyCadenceEnv, "1m")
	if _, skipped := dispatchGatherSlowProbes("ws", "claude", at); len(skipped) != 3 {
		t.Fatalf("armed in-cadence tick skipped %v, want all three probes skipped", skipped)
	}
	if _, skipped := dispatchGatherSlowProbes("other-ws", "claude", at); len(skipped) != 0 {
		t.Fatalf("scope change skipped %v, want a full re-gather (never serve another scope's cache)", skipped)
	}
	if *seat != 4 {
		t.Fatalf("seat gathers = %d, want 4 (three unarmed ticks + the scope-change re-gather)", *seat)
	}
}

// The need declaration matches a probe by exact path or by nesting under it (the
// exact-or-prefix rule); a sibling sharing only a string prefix must not match, and
// a probe declared by no due consumer is not needed.
func TestLazyPullProbeNeededExactOrNestedPrefix(t *testing.T) {
	due := []lazyPullConsumer{{Name: "c", Needs: []string{"seat.fresh_ceiling", "gate.hooklat.p99"}}}
	for probe, want := range map[string]bool{
		"seat.fresh_ceiling":      true,  // exact path
		"gate.hooklat":            true,  // need nests under the probe path
		"seat.fresh":              false, // string prefix but not a path prefix
		"rate_limit.worker_exits": false, // declared by no due consumer
	} {
		if got := lazyPullProbeNeeded(due, probe); got != want {
			t.Fatalf("lazyPullProbeNeeded(%q) = %v, want %v", probe, got, want)
		}
	}
	if lazyPullProbeNeeded(nil, "seat.fresh_ceiling") {
		t.Fatal("a probe cannot be needed on a tick with no due consumer")
	}
}

// End-to-end wiring: dispatchPreflightTimed consults the due-filter -- the second
// in-cadence tick serves the cached observations (one gather total) and names the
// skipped probes under lazy_pull, while the first tick's payload carries no
// lazy_pull key (byte-identical common payload).
func TestDispatchPreflightTimedLazyPullWired(t *testing.T) {
	t.Setenv(dispatchLazyCadenceEnv, "1m")
	gate, _, _ := stubSlowProbes(t)
	oldProc, oldKernel, oldResources, oldWorkers := dispatchProbeProcesses, dispatchRunExternalJSON, dispatchProbeHostResources, dispatchProbeWorkerCount
	oldCodexRows := dispatchProbeCodexProcessRows
	defer func() {
		dispatchProbeProcesses = oldProc
		dispatchRunExternalJSON = oldKernel
		dispatchProbeHostResources = oldResources
		dispatchProbeWorkerCount = oldWorkers
		dispatchProbeCodexProcessRows = oldCodexRows
	}()
	dispatchProbeProcesses = func() dispatchtick.ProcGuardInput { return dispatchtick.ProcGuardInput{} }
	dispatchRunExternalJSON = func(string, time.Duration, string, ...string) (map[string]any, error) {
		return map[string]any{"alive": 1, "target": 1, "verdict": "OK"}, nil
	}
	dispatchProbeHostResources = func() dispatchtick.HostResources { return dispatchtick.HostResources{} }
	dispatchProbeWorkerCount = func(string, string) int { return 0 }
	dispatchProbeCodexProcessRows = func() ([]dispatchCodexProcessRow, error) { return nil, nil }
	dispatchKernelCache.Lock()
	dispatchKernelCache.at = time.Time{}
	dispatchKernelCache.root = ""
	dispatchKernelCache.Unlock()
	resetLazyPullState()
	t.Cleanup(resetLazyPullState)

	root := t.TempDir()
	first, _, err := dispatchPreflightTimed(root, io.Discard, 1, "engineering", "codex")
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := first["lazy_pull"]; ok {
		t.Fatalf("first tick payload carries lazy_pull %v, want it only when a gather was skipped", got)
	}
	second, _, err := dispatchPreflightTimed(root, io.Discard, 1, "engineering", "codex")
	if err != nil {
		t.Fatal(err)
	}
	if *gate != 1 {
		t.Fatalf("gate probe gathered %d times across two in-cadence ticks, want 1", *gate)
	}
	lp, ok := second["lazy_pull"].(map[string]any)
	if !ok {
		t.Fatalf("second tick lazy_pull = %#v, want the skipped-probe note", second["lazy_pull"])
	}
	skipped, _ := lp["skipped"].([]string)
	if len(skipped) != 3 {
		t.Fatalf("lazy_pull.skipped = %v, want all three expensive probes", lp["skipped"])
	}
}
