package modelengine

import (
	"testing"
)

// fak-test:runtime medium est=6s lane=default

// TestNativeSchedulerLongPrefillCapClamp pins the cross-request fairness clamp
// (vLLM long_prefill_token_threshold,
// vllm/v1/core/sched/scheduler.py:728-742@ae065918640a198a56f6f17eb45e251c3229fe41,
// Apache-2.0): while other requests are queued, a lone long prefill advances
// at most the armed threshold prompt tokens per scheduler step; empty queues
// leave the chunk untouched; and the clamp composes with the p0-prefill-budget
// resumable chunking - the next step resumes at the clamped cursor with no
// token dropped or replayed.
func TestNativeSchedulerLongPrefillCapClamp(t *testing.T) {
	const (
		promptLen    = 8192 // the acceptance criterion's lone 8k-prompt prefill
		nominal      = 2048 // armed per-iteration chunk ceiling
		threshold    = 1024 // the cross-request fairness cap
		waitingLanes = 3
	)

	t.Run("lone_long_prefill_clamped_to_threshold_while_others_wait", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		prompts := map[string][]int{
			"long":  nativeSchedulerQwenPrompt(promptLen),
			"wait1": nativeSchedulerQwenPrompt(32),
			"wait2": nativeSchedulerQwenPrompt(32),
			"wait3": nativeSchedulerQwenPrompt(32),
		}
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(prompts))
		if err := s.SetQwenPrefillMaxTokensPerIteration(nominal); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration(%d): %v", nominal, err)
		}
		if err := s.SetLongPrefillTokenThreshold(threshold); err != nil {
			t.Fatalf("SetLongPrefillTokenThreshold(%d): %v", threshold, err)
		}
		s.SetMaxRunning(1) // hold the three waiters out of the running set
		nativeSchedulerBeginManualDrain(t, s)

		longReq := nativeSchedulerAdmitLane(t, s, "long")
		if longReq.state != schedLanePrefilling || longReq.prefillChunkTokens != nominal {
			t.Fatalf("armed admission = state %d nominal %d, want PREFILLING/%d",
				longReq.state, longReq.prefillChunkTokens, nominal)
		}
		for _, tool := range []string{"wait1", "wait2", "wait3"} {
			nativeSchedulerAdmitLane(t, s, tool)
		}

		var prefills []nativeSchedulerEvent
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventPrefill && event.Lane == longReq {
				prefills = append(prefills, event)
			}
		}
		for i := 0; longReq.state != schedLaneDecode && i < 64; i++ {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(longReq)
		}
		if longReq.state != schedLaneDecode {
			t.Fatalf("lane state = %d after resume loop, want DECODE", longReq.state)
		}
		if longReq.promptCursor != promptLen {
			t.Fatalf("promptCursor = %d, want prompt end %d (truncated prefill)",
				longReq.promptCursor, promptLen)
		}

		// Closed contract: 8192 / 1024 = exactly 8 clamped chunks, contiguous
		// and strictly advancing - at most the threshold per step, none
		// dropped or replayed.
		wantChunks := promptLen / threshold
		if len(prefills) != wantChunks {
			t.Fatalf("prompt %d under cap %d split into %d chunks, want %d; events=%+v",
				promptLen, threshold, len(prefills), wantChunks, prefills)
		}
		cursor := 0
		for i, ev := range prefills {
			if ev.ChunkStart != cursor || ev.ChunkLen != threshold {
				t.Fatalf("chunk %d = start %d len %d, want start %d len %d",
					i, ev.ChunkStart, ev.ChunkLen, cursor, threshold)
			}
			if i > 0 && ev.Iteration <= prefills[i-1].Iteration {
				t.Fatalf("chunk %d iteration %d not strictly after chunk %d iteration %d",
					i, ev.Iteration, i-1, prefills[i-1].Iteration)
			}
			cursor += ev.ChunkLen
		}
		if cursor != promptLen {
			t.Fatalf("chunks consumed %d tokens, want prompt length %d", cursor, promptLen)
		}
		// The clamp is per-step: the admission-time nominal survives the resume.
		if got := longReq.prefillChunkTokens; got != nominal {
			t.Fatalf("post-resume nominal chunk = %d, want restored %d", got, nominal)
		}

		longReq.Cancel()
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("empty_queues_leave_chunk_untouched", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		prompt := nativeSchedulerQwenPrompt(promptLen)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"solo": prompt}))
		if err := s.SetQwenPrefillMaxTokensPerIteration(nominal); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration(%d): %v", nominal, err)
		}
		if err := s.SetLongPrefillTokenThreshold(threshold); err != nil {
			t.Fatalf("SetLongPrefillTokenThreshold(%d): %v", threshold, err)
		}
		nativeSchedulerBeginManualDrain(t, s)
		req := nativeSchedulerAdmitLane(t, s, "solo")

		var prefills []nativeSchedulerEvent
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventPrefill && event.Lane == req {
				prefills = append(prefills, event)
			}
		}
		for i := 0; req.state != schedLaneDecode && i < 64; i++ {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(req)
		}
		if req.state != schedLaneDecode {
			t.Fatalf("lane state = %d after resume loop, want DECODE", req.state)
		}

		// With nobody waiting or preempted the cap must stay inert: the full
		// nominal chunk runs, 8192 / 2048 = exactly 4 chunks.
		wantChunks := promptLen / nominal
		if len(prefills) != wantChunks {
			t.Fatalf("uncontended prompt %d split into %d chunks, want %d; events=%+v",
				promptLen, len(prefills), wantChunks, prefills)
		}
		cursor := 0
		for i, ev := range prefills {
			if ev.ChunkStart != cursor || ev.ChunkLen != nominal {
				t.Fatalf("chunk %d = start %d len %d, want start %d len %d (cap leaked into empty queues)",
					i, ev.ChunkStart, ev.ChunkLen, cursor, nominal)
			}
			cursor += ev.ChunkLen
		}
		if cursor != promptLen {
			t.Fatalf("chunks consumed %d tokens, want prompt length %d", cursor, promptLen)
		}

		req.Cancel()
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("trigger_and_setter_contract", func(t *testing.T) {
		s := newNativeScheduler(nativeSchedulerPrefillModel(t), nil)
		// Disabled by default.
		s.mu.Lock()
		if got := s.longPrefillStepCapLocked(); got != 0 {
			t.Fatalf("default cap = %d, want disabled 0", got)
		}
		s.mu.Unlock()
		// A sub-minimum threshold is refused and leaves the cap disabled.
		if err := s.SetLongPrefillTokenThreshold(nativeQwenPrefillMinChunkTokens - 1); err == nil {
			t.Fatal("sub-minimum threshold accepted, want refusal")
		}
		// Negative normalizes to disabled.
		if err := s.SetLongPrefillTokenThreshold(-1); err != nil {
			t.Fatalf("SetLongPrefillTokenThreshold(-1): %v", err)
		}
		s.mu.Lock()
		if got := s.longPrefillTokenThreshold; got != 0 {
			t.Fatalf("threshold after negative set = %d, want disabled 0", got)
		}
		s.mu.Unlock()
		// An armed threshold reports itself only while others are queued.
		if err := s.SetLongPrefillTokenThreshold(threshold); err != nil {
			t.Fatalf("SetLongPrefillTokenThreshold(%d): %v", threshold, err)
		}
		s.mu.Lock()
		if got := s.longPrefillStepCapLocked(); got != 0 {
			t.Fatalf("cap with empty queues = %d, want 0", got)
		}
		s.waiting = append(s.waiting, &schedLane{})
		if got := s.longPrefillStepCapLocked(); got != threshold {
			t.Fatalf("cap with one waiting lane = %d, want %d", got, threshold)
		}
		s.waiting = s.waiting[:0]
		s.preempted = append(s.preempted, &schedLane{})
		if got := s.longPrefillStepCapLocked(); got != threshold {
			t.Fatalf("cap with one preempted lane = %d, want %d", got, threshold)
		}
		s.preempted = s.preempted[:0]
		s.mu.Unlock()
	})
}
