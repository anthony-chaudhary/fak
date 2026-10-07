package modelengine

import (
	"testing"
)

// TestNativeSchedulerPrefillBudgetResumes is the named resume witness for the
// bounded prefill-token budget (ADAPT of mini-sglang's token_budget + resumable
// chunked_req, python/minisgl/scheduler/prefill.py:33-151@9a91cfa, MIT; fak#8395).
//
// It drives the PRODUCTION seam — Engine.nativeScheduler(), which arms
// SetQwenPrefillMaxTokensPerIteration on the serving admission path — so deleting
// that arming turns this test red. A prompt strictly longer than the armed budget
// must be consumed across scheduler iterations as contiguous chunks (no truncated
// prefix, no re-prefill of consumed tokens), yield to an already-decoding lane
// between chunks, and reach DECODE with the cursor at the prompt end. The refusal
// branch (a positive ceiling below the resident minimum) stays intact.
func TestNativeSchedulerPrefillBudgetResumes(t *testing.T) {
	t.Run("armed_serving_path_resumes_to_decode", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		e := New()
		e.Preload(m)
		s := e.nativeScheduler()
		if s.qwenPrefillTokens != nativeServingPrefillTokensPerIteration {
			t.Fatalf("Engine.nativeScheduler armed qwenPrefillTokens = %d, want %d (SetQwenPrefillMaxTokensPerIteration on the serving path)",
				s.qwenPrefillTokens, nativeServingPrefillTokensPerIteration)
		}
		budget := s.qwenPrefillTokens
		longPrompt := nativeSchedulerQwenPrompt(2*budget + 7) // three chunks, last one partial
		s.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
			"decode": nativeSchedulerQwenPrompt(nativeQwenPrefillMinChunkTokens - 1), // below the resident minimum: synchronous
			"long":   longPrompt,
		})
		nativeSchedulerBeginManualDrain(t, s)
		defer nativeSchedulerEndManualDrain(s)

		var events []nativeSchedulerEvent
		s.observeNativeEvent = func(ev nativeSchedulerEvent) { events = append(events, ev) }

		decodeReq := nativeSchedulerAdmitLane(t, s, "decode")
		if decodeReq.state != schedLaneDecode {
			t.Fatalf("short admission state = %d, want synchronous DECODE", decodeReq.state)
		}
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(decodeReq)

		events = nil
		longReq := nativeSchedulerAdmitLane(t, s, "long")
		if longReq.state != schedLanePrefilling || longReq.promptCursor != 0 {
			t.Fatalf("over-budget admission = state %d cursor %d, want PREFILLING/0", longReq.state, longReq.promptCursor)
		}
		for i := 0; longReq.state != schedLaneDecode && i < 16; i++ {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(decodeReq)
			nativeSchedulerDrainAvailable(longReq)
		}
		if longReq.state != schedLaneDecode || longReq.terminal {
			t.Fatalf("long lane state = %d terminal = %t, want live DECODE after resumed prefill", longReq.state, longReq.terminal)
		}
		if longReq.promptCursor != len(longPrompt) || longReq.promptLen != len(longPrompt) {
			t.Fatalf("long lane cursor/accounted = %d/%d, want prompt end %d", longReq.promptCursor, longReq.promptLen, len(longPrompt))
		}

		var idx []int
		cursor := 0
		for i, ev := range events {
			if ev.Kind != nativeSchedulerEventPrefill || ev.Lane != longReq {
				continue
			}
			if ev.ChunkStart != cursor {
				t.Fatalf("chunk %d starts at %d, want resumed cursor %d (no gap, no re-prefill)", len(idx), ev.ChunkStart, cursor)
			}
			if ev.ChunkLen <= 0 || ev.ChunkLen > budget {
				t.Fatalf("chunk %d length %d outside (0, budget=%d]", len(idx), ev.ChunkLen, budget)
			}
			if len(idx) > 0 && ev.Iteration <= events[idx[len(idx)-1]].Iteration {
				t.Fatalf("chunk %d did not run in a later scheduler iteration", len(idx))
			}
			cursor += ev.ChunkLen
			idx = append(idx, i)
		}
		if len(idx) != 3 || cursor != len(longPrompt) {
			t.Fatalf("prefill consumed %d tokens in %d chunks, want %d tokens in 3; events=%+v", cursor, len(idx), len(longPrompt), events)
		}
		for i := 0; i+1 < len(idx); i++ {
			if !nativeSchedulerHasDecodeBetween(events, idx[i], idx[i+1], decodeReq) {
				t.Fatalf("prefill did not yield to the decoding lane between chunks %d and %d", i, i+1)
			}
		}
		decodeReq.Cancel()
		longReq.Cancel()
		for !decodeReq.terminal || !longReq.terminal {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(decodeReq)
			nativeSchedulerDrainAvailable(longReq)
		}
	})

	t.Run("subminimum_ceiling_still_refused", func(t *testing.T) {
		s := NewNativeScheduler(nativeSchedulerPrefillModel(t))
		if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens - 1); err == nil {
			t.Fatal("sub-minimum ceiling accepted, want refusal")
		}
		if s.qwenPrefillTokens != 0 {
			t.Fatalf("refused ceiling left qwenPrefillTokens = %d, want 0 (disabled)", s.qwenPrefillTokens)
		}
	})
}
