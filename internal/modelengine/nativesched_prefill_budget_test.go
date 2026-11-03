package modelengine

import (
	"errors"
	"reflect"
	"testing"
)

// TestNativeSchedulerPrefillBudgetResumes pins the ported bounded prefill-token
// budget (mini-sglang PrefillAdder.token_budget@9a91cfa): one shared step budget
// consumed across prefill requests, an oversized request taking only what fits
// and resuming into later steps, exhaustion refusing with a sentinel, and the
// serving path arming the ceiling so a prompt longer than it resumes to
// completion without truncation or re-prefill.
func TestNativeSchedulerPrefillBudgetResumes(t *testing.T) {
	t.Run("step_budget_consumes_across_requests", func(t *testing.T) {
		var b nativePrefillStepBudget
		b.reset(1, 32)

		// Oversized request: takes what fits, stays resumable.
		chunk, resumed, err := b.take(48)
		if err != nil {
			t.Fatalf("take(48) on fresh 32 budget: %v", err)
		}
		if chunk != 32 || !resumed {
			t.Fatalf("take(48) = chunk %d resumed %t, want 32/true", chunk, resumed)
		}
		if got, want := b.remaining(), 0; got != want {
			t.Fatalf("remaining after full draw = %d, want %d", got, want)
		}
		if got, want := b.spent(), 32; got != want {
			t.Fatalf("spent after full draw = %d, want %d", got, want)
		}

		// An exhausted step refuses further draws with the sentinel and spends
		// nothing on the refusal.
		for _, remain := range []int{16, 1, 0} {
			if _, _, err := b.take(remain); !errors.Is(err, errNativePrefillBudgetExhausted) {
				t.Fatalf("exhausted take(%d) err = %v, want errNativePrefillBudgetExhausted", remain, err)
			}
		}
		if got, want := b.spent(), 32; got != want {
			t.Fatalf("refusals changed spent to %d, want %d", got, want)
		}

		// A fresh step re-arms and funds exact-fit requests without resuming.
		b.reset(2, 32)
		for i, remain := range []int{16, 16} {
			chunk, resumed, err = b.take(remain)
			if err != nil || chunk != remain || resumed {
				t.Fatalf("step 2 take(%d) #%d = %d/%t/%v, want %d/false/nil", remain, i, chunk, resumed, err, remain)
			}
		}
		if _, _, err := b.take(1); !errors.Is(err, errNativePrefillBudgetExhausted) {
			t.Fatalf("take past the re-armed budget err = %v, want exhaustion sentinel", err)
		}

		// Partial fit below the ask is funded and marked resumable.
		b.reset(3, 32)
		chunk, resumed, err = b.take(40)
		if err != nil || chunk != 32 || !resumed {
			t.Fatalf("partial take(40) = %d/%t/%v, want 32/true/nil", chunk, resumed, err)
		}
	})

	t.Run("long_prompt_resumes_to_completion_under_serving_arming", func(t *testing.T) {
		m := nativeSchedulerPrefillModel(t)
		e := New()
		e.Preload(m)
		s := e.nativeScheduler()
		// Production arming witness: deleting the serving-path arming in
		// Engine.nativeScheduler turns this red.
		if s.qwenPrefillTokens < nativeQwenPrefillMinChunkTokens {
			t.Fatalf("serving path armed qwenPrefillTokens=%d, want >= %d",
				s.qwenPrefillTokens, nativeQwenPrefillMinChunkTokens)
		}
		capital := s.qwenPrefillTokens
		prompt := nativeSchedulerQwenPrompt(capital + capital/2) // strictly longer than the ceiling
		s.prepare = nativeSchedulerPrefillPrepare(map[string][]int{"long": prompt})
		nativeSchedulerBeginManualDrain(t, s)

		var prefills []nativeSchedulerEvent
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventPrefill && event.Lane != nil && event.Lane.tool == "long" {
				prefills = append(prefills, event)
			}
		}

		req := nativeSchedulerAdmitLane(t, s, "long")
		if req.state != schedLanePrefilling || req.promptCursor != 0 || len(req.logits) != 0 {
			t.Fatalf("armed admission = state %d cursor %d logits %d, want PREFILLING/0/0",
				req.state, req.promptCursor, len(req.logits))
		}

		got := make([]int, 0, genTokens)
		for i := 0; req.state != schedLaneDecode && i < 10; i++ {
			nativeSchedulerDriveIteration(t, s)
			got = append(got, nativeSchedulerDrainAvailable(req)...)
		}
		if req.state != schedLaneDecode {
			t.Fatalf("lane state = %d after resume loop, want DECODE", req.state)
		}
		if req.promptCursor != len(prompt) {
			t.Fatalf("promptCursor = %d, want prompt end %d (truncated prefill)", req.promptCursor, len(prompt))
		}

		// Resume contract: contiguous, strictly-advancing chunks across strictly
		// increasing iterations - no gap, no overlap, no re-prefill.
		if len(prefills) != 2 {
			t.Fatalf("prompt %d under armed ceiling %d split into %d chunks, want 2; events=%+v",
				len(prompt), capital, len(prefills), prefills)
		}
		cursor := 0
		for i, ev := range prefills {
			if ev.ChunkStart != cursor {
				t.Fatalf("chunk %d starts at %d, want contiguous cursor %d (re-prefill or gap)", i, ev.ChunkStart, cursor)
			}
			if ev.ChunkLen <= 0 || ev.ChunkLen > capital {
				t.Fatalf("chunk %d length = %d, want (0, ceiling=%d]", i, ev.ChunkLen, capital)
			}
			if i > 0 && ev.Iteration <= prefills[i-1].Iteration {
				t.Fatalf("chunk %d iteration %d not strictly after chunk %d iteration %d",
					i, ev.Iteration, i-1, prefills[i-1].Iteration)
			}
			cursor += ev.ChunkLen
		}
		if cursor != len(prompt) {
			t.Fatalf("chunks consumed %d tokens, want prompt length %d", cursor, len(prompt))
		}
		if prefills[0].ChunkLen != capital || prefills[1].ChunkLen != len(prompt)-capital {
			t.Fatalf("chunk lengths = %d/%d, want %d/%d",
				prefills[0].ChunkLen, prefills[1].ChunkLen, capital, len(prompt)-capital)
		}

		// Step-budget accounting witness: the final step spent exactly its
		// chunk from the live ceiling, proving the advance path consults the
		// shared step budget rather than the raw field.
		s.mu.Lock()
		budgetStep := s.qwenPrefillCap.budget.step
		budgetSpent := s.qwenPrefillCap.budget.spent()
		budgetTotal := s.qwenPrefillCap.budget.total
		s.mu.Unlock()
		if budgetTotal != capital {
			t.Fatalf("final step budget total = %d, want armed ceiling %d", budgetTotal, capital)
		}
		if budgetStep != uint64(len(prefills)) {
			t.Fatalf("final step budget step = %d, want last prefill iteration %d", budgetStep, len(prefills))
		}
		if budgetSpent != prefills[len(prefills)-1].ChunkLen {
			t.Fatalf("final step budget spent = %d, want last chunk %d", budgetSpent, prefills[len(prefills)-1].ChunkLen)
		}

		// Generated-token parity with the synchronous-prefill oracle.
		want := nativeSchedulerSynchronousTokens(m, prompt)
		for len(got) < genTokens && !req.terminal {
			nativeSchedulerDriveIteration(t, s)
			got = append(got, nativeSchedulerDrainAvailable(req)...)
		}
		if len(got) < genTokens {
			t.Fatalf("emitted %d/%d tokens (terminal=%t state=%d)", len(got), genTokens, req.terminal, req.state)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("chunked tokens = %v, synchronous oracle = %v", got, want)
		}

		req.Cancel()
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("midflight_ceiling_shrink_follows_live_step_budget", func(t *testing.T) {
		// Admit under a 32-token ceiling, then shrink the live ceiling to the
		// resident minimum before any chunk runs: every step must fund from the
		// shrunken step budget instead of the admission-time nominal chunk.
		m := nativeSchedulerPrefillModel(t)
		prompt := nativeSchedulerQwenPrompt(48)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"shrink": prompt}))
		if err := s.SetQwenPrefillMaxTokensPerIteration(32); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration(32): %v", err)
		}
		nativeSchedulerBeginManualDrain(t, s)
		req := nativeSchedulerAdmitLane(t, s, "shrink")
		if req.state != schedLanePrefilling || req.prefillChunkTokens != 32 {
			t.Fatalf("admission = state %d nominal chunk %d, want PREFILLING/32",
				req.state, req.prefillChunkTokens)
		}
		if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
			t.Fatalf("shrink to %d: %v", nativeQwenPrefillMinChunkTokens, err)
		}

		var prefills []nativeSchedulerEvent
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventPrefill && event.Lane == req {
				prefills = append(prefills, event)
			}
		}
		for i := 0; req.state != schedLaneDecode && i < 10; i++ {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(req)
		}
		if req.state != schedLaneDecode {
			t.Fatalf("shrunk-ceiling lane state = %d, want DECODE", req.state)
		}
		if len(prefills) != 3 {
			t.Fatalf("shrunk ceiling split the prompt into %d chunks, want 3 x %d; events=%+v",
				len(prefills), nativeQwenPrefillMinChunkTokens, prefills)
		}
		for i, ev := range prefills {
			if ev.ChunkStart != i*nativeQwenPrefillMinChunkTokens || ev.ChunkLen != nativeQwenPrefillMinChunkTokens {
				t.Fatalf("chunk %d = start %d len %d, want start %d len %d",
					i, ev.ChunkStart, ev.ChunkLen, i*nativeQwenPrefillMinChunkTokens, nativeQwenPrefillMinChunkTokens)
			}
		}

		req.Cancel()
		nativeSchedulerEndManualDrain(s)
	})

	t.Run("subthreshold_refusal_stays_intact", func(t *testing.T) {
		s := newNativeScheduler(nativeSchedulerPrefillModel(t), nil)
		if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens - 1); err == nil {
			t.Fatal("subthreshold ceiling accepted, want refusal")
		}
		if s.qwenPrefillTokens != 0 {
			t.Fatalf("subthreshold refusal left qwenPrefillTokens=%d, want disabled 0", s.qwenPrefillTokens)
		}
	})
}
