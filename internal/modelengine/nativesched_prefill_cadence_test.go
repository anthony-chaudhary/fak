package modelengine

import (
	"fmt"
	"reflect"
	"testing"
)

// TestNativeSchedulerPrefillCadence pins the FAK_NATIVE_PREFILL_INTERVAL port of
// vLLM's local prefill cadence (vllm/config/scheduler.py:203-208,
// vllm/v1/core/sched/scheduler.py:389-406,748-769,1562-1577 @ae065918640a,
// Apache-2.0, ADAPT): prefill is held to every Nth engine step so decode runs in
// longer uninterrupted stretches, with a capacity escape for a backed-up
// waiting queue. These are exact step-order and counter contracts on the
// synthetic fixture — no GPU or throughput claim.
func TestNativeSchedulerPrefillCadence(t *testing.T) {
	const budget = nativeQwenPrefillMinChunkTokens
	shortPrompt := nativeSchedulerQwenPrompt(budget)

	// newFixture arms the chunked-prefill fixture through the production knob
	// seam. It returns the scheduler plus a pointer to the live event log: the
	// observer closure appends into the same backing array the caller reads.
	newFixture := func(t *testing.T, prompts map[string][]int, maxRunning int) (*NativeScheduler, *[]nativeSchedulerEvent) {
		t.Helper()
		clearServeKnobEnv(t)
		m := nativeSchedulerPrefillModel(t)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(prompts))
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		if maxRunning > 0 {
			s.SetMaxRunning(maxRunning)
		}
		applyNativeServeKnobs(s, 0)
		nativeSchedulerBeginManualDrain(t, s)
		t.Cleanup(func() { nativeSchedulerEndManualDrain(s) })
		events := make([]nativeSchedulerEvent, 0, 8)
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			events = append(events, event)
		}
		return s, &events
	}

	// driveWindow drives n iterations, asserting each one emitted exactly one
	// uninterrupted decode token for the resident lane.
	driveWindow := func(t *testing.T, s *NativeScheduler, decodeReq *schedLane, from, to int) {
		t.Helper()
		for iteration := from; iteration <= to; iteration++ {
			nativeSchedulerDriveIteration(t, s)
			if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
				t.Fatalf("iteration %d decode tokens = %v, want exactly one uninterrupted token", iteration, got)
			}
		}
	}

	cadenceCounters := func(t *testing.T, s *NativeScheduler) (executed, last int, bound bool) {
		t.Helper()
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.executedSteps, s.lastPrefillStep, s.prefillCapacityBound
	}

	laneCursor := func(t *testing.T, ln *schedLane) (cursor int, state schedLaneState) {
		t.Helper()
		ln.sched.mu.Lock()
		defer ln.sched.mu.Unlock()
		return ln.promptCursor, ln.state
	}

	waitingDepth := func(t *testing.T, s *NativeScheduler) int {
		t.Helper()
		s.mu.Lock()
		defer s.mu.Unlock()
		return len(s.waiting)
	}

	// settle cancels every live lane and drives the scheduler to idle.
	settle := func(t *testing.T, s *NativeScheduler, lanes ...*schedLane) {
		t.Helper()
		for _, ln := range lanes {
			ln.Cancel()
		}
		for {
			_, idle, _ := nativeSchedulerDriveIteration(t, s)
			if idle {
				return
			}
		}
	}

	t.Run("interval_two_alternates_decode_and_prefill", func(t *testing.T) {
		t.Setenv(envNativePrefillInterval, "2")
		s, events := newFixture(t, map[string][]int{
			"decode": shortPrompt,
			"long":   nativeSchedulerQwenPrompt(2*budget + 3),
		}, 0)

		decodeReq := nativeSchedulerAdmitLane(t, s, "decode")
		nativeSchedulerDriveIteration(t, s) // iteration 1: decode lane's own final chunk
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 1 decode tokens = %v, want one token", got)
		}
		longReq := nativeSchedulerAdmitLane(t, s, "long")

		driveWindow(t, s, decodeReq, 2, 5)

		// One waiting long prompt + one resident decode lane: the window
		// alternates decode-only / prefill-carrying across four iterations.
		assertPrefillStepOrder(t, events, []prefillStepWitness{
			{iteration: 1, tool: "decode", start: 0, length: budget},
			{iteration: 3, tool: "long", start: 0, length: budget},
			{iteration: 5, tool: "long", start: budget, length: budget},
		})
		if cursor, state := laneCursor(t, longReq); cursor != 2*budget || state != schedLanePrefilling {
			t.Fatalf("long lane after window = cursor %d state %d, want %d PREFILLING", cursor, state, 2*budget)
		}
		if executed, last, bound := cadenceCounters(t, s); executed != 5 || last != 5 || bound {
			t.Fatalf("cadence counters = executed %d last %d bound %t, want 5 5 false", executed, last, bound)
		}
		settle(t, s, decodeReq, longReq)
	})

	t.Run("capacity_escape_admits_prefill_early", func(t *testing.T) {
		t.Setenv(envNativePrefillInterval, "2")
		s, events := newFixture(t, map[string][]int{
			"decode": shortPrompt,
			"a":      nativeSchedulerQwenPrompt(2*budget + 3),
			"b":      nativeSchedulerQwenPrompt(2*budget + 5),
			"c":      nativeSchedulerQwenPrompt(2*budget + 7),
		}, 2)

		decodeReq := nativeSchedulerAdmitLane(t, s, "decode")
		nativeSchedulerDriveIteration(t, s) // iteration 1: decode lane's own final chunk
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 1 decode tokens = %v, want one token", got)
		}
		aReq := nativeSchedulerAdmitLane(t, s, "a")
		bReq := nativeSchedulerAdmitLane(t, s, "b")
		cReq := nativeSchedulerAdmitLane(t, s, "c")

		// Iteration 2 holds: the cadence distance is 1 < 2 and the queue depth
		// is recorded only on a release step.
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 2 decode tokens = %v, want one held-step token", got)
		}
		if _, _, bound := cadenceCounters(t, s); bound {
			t.Fatal("capacity bound armed before the first release step")
		}
		if got := waitingDepth(t, s); got != 2 {
			t.Fatalf("waiting depth after promotion = %d, want 2 (a promoted, b+c queued)", got)
		}

		// Iteration 3 releases: a's first chunk runs and the release step records
		// the escape — the waiting queue (2) is deeper than one interval's drain.
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 3 decode tokens = %v, want one token", got)
		}
		if cursor, _ := laneCursor(t, aReq); cursor != budget {
			t.Fatalf("iteration 3 a cursor = %d, want %d", cursor, budget)
		}
		if _, _, bound := cadenceCounters(t, s); !bound {
			t.Fatal("release step did not record the capacity-bound escape")
		}

		// Iteration 4 would hold under pure cadence ((3+1-3) < 2), but the
		// backed-up engine admits prefill early instead of starving the queue.
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 4 decode tokens = %v, want one token", got)
		}
		if cursor, _ := laneCursor(t, aReq); cursor != 2*budget {
			t.Fatalf("iteration 4 a cursor = %d, want %d (early admission)", cursor, 2*budget)
		}

		// Iteration 5 lands a's final chunk; b and c remain queued behind it.
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 5 decode tokens = %v, want one token", got)
		}
		if cursor, state := laneCursor(t, aReq); cursor != 2*budget+3 || state != schedLaneDecode {
			t.Fatalf("iteration 5 a = cursor %d state %d, want %d DECODE", cursor, state, 2*budget+3)
		}
		if got := waitingDepth(t, s); got != 2 {
			t.Fatalf("waiting depth after window = %d, want 2 (b+c never starved out of the queue)", got)
		}

		assertPrefillStepOrder(t, events, []prefillStepWitness{
			{iteration: 1, tool: "decode", start: 0, length: budget},
			{iteration: 3, tool: "a", start: 0, length: budget},
			{iteration: 4, tool: "a", start: budget, length: budget},
			{iteration: 5, tool: "a", start: 2 * budget, length: 3},
		})
		if executed, last, bound := cadenceCounters(t, s); executed != 5 || last != 5 || !bound {
			t.Fatalf("cadence counters = executed %d last %d bound %t, want 5 5 true", executed, last, bound)
		}
		settle(t, s, decodeReq, aReq, bReq, cReq)
	})

	t.Run("interval_one_default_keeps_mixed_step_order", func(t *testing.T) {
		t.Setenv(envNativePrefillInterval, "1")
		s, events := newFixture(t, map[string][]int{
			"decode": shortPrompt,
			"long":   nativeSchedulerQwenPrompt(2*budget + 3),
		}, 0)

		decodeReq := nativeSchedulerAdmitLane(t, s, "decode")
		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(decodeReq); len(got) != 1 {
			t.Fatalf("iteration 1 decode tokens = %v, want one token", got)
		}
		longReq := nativeSchedulerAdmitLane(t, s, "long")
		driveWindow(t, s, decodeReq, 2, 4)

		// The default interval keeps today's baseline: every step may carry
		// prefill, so the mixed-step order is unchanged.
		assertPrefillStepOrder(t, events, []prefillStepWitness{
			{iteration: 1, tool: "decode", start: 0, length: budget},
			{iteration: 2, tool: "long", start: 0, length: budget},
			{iteration: 3, tool: "long", start: budget, length: budget},
			{iteration: 4, tool: "long", start: 2 * budget, length: 3},
		})
		if cursor, state := laneCursor(t, longReq); cursor != 2*budget+3 || state != schedLaneDecode {
			t.Fatalf("long lane after window = cursor %d state %d, want %d DECODE", cursor, state, 2*budget+3)
		}
		if executed, last, bound := cadenceCounters(t, s); executed != 4 || last != 4 || bound {
			t.Fatalf("cadence counters = executed %d last %d bound %t, want 4 4 false", executed, last, bound)
		}
		settle(t, s, decodeReq, longReq)
	})

	t.Run("knob_reaches_production_scheduler", func(t *testing.T) {
		clearServeKnobEnv(t)
		read := func() (interval, last int) {
			s := New().nativeScheduler()
			s.mu.Lock()
			defer s.mu.Unlock()
			return s.prefillInterval, s.lastPrefillStep
		}
		if interval, last := read(); interval != 0 || last != 0 {
			t.Fatalf("default cadence = interval %d last %d, want 0/0 (off)", interval, last)
		}
		for value, want := range map[string]int{"2": 2, "8": 8, "bogus": 0, "0": 0, "-3": 0} {
			t.Setenv(envNativePrefillInterval, value)
			interval, last := read()
			if interval != want || last != -want {
				t.Fatalf("%s=%s cadence = interval %d last %d, want %d/-%d", envNativePrefillInterval, value, interval, last, want, want)
			}
		}
	})
}

type prefillStepWitness struct {
	iteration uint64
	tool      string
	start     int
	length    int
}

// assertPrefillStepOrder pins the exact ordered set of prefill-carrying steps:
// one nativeSchedulerEventPrefill per listed (iteration, lane, chunk) and none
// on any other iteration.
func assertPrefillStepOrder(t *testing.T, events *[]nativeSchedulerEvent, want []prefillStepWitness) {
	t.Helper()
	key := func(iteration uint64, tool string, start, length int) string {
		return fmt.Sprintf("%d:%s:%d:%d", iteration, tool, start, length)
	}
	got := make([]string, 0, len(want))
	for _, ev := range *events {
		if ev.Kind != nativeSchedulerEventPrefill || ev.Lane == nil {
			continue
		}
		got = append(got, key(ev.Iteration, ev.Lane.tool, ev.ChunkStart, ev.ChunkLen))
	}
	wantKeys := make([]string, 0, len(want))
	for _, w := range want {
		wantKeys = append(wantKeys, key(w.iteration, w.tool, w.start, w.length))
	}
	if !reflect.DeepEqual(got, wantKeys) {
		t.Fatalf("prefill step order = %v, want %v", got, wantKeys)
	}
}
