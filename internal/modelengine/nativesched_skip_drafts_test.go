package modelengine

// fak-test:runtime medium est=5s lane=default

import (
	"testing"
)

// TestNativeSchedulerSkipUnusedDrafts pins the vLLM skip_drafts/_drafts_unused
// adaptation (vllm/v1/core/sched/scheduler.py:311-315, 483-525, 712, 1528-1530,
// 1693 @ae065918640a198a56f6f17eb45e251c3229fe41, Apache-2.0): NativeScheduler
// computes a per-step skipDrafts flag from lane state — the prefill-chunk
// cursor versus the prompt length, and terminal-token detection at
// publication — and reports it so the MTP self-speculation path proposes K=0
// drafts on steps whose drafts are SURE to be unused. Stop tokens and strings
// cannot be known in advance, so only those sure grounds earn skip=true.
// Software witness only: no GPU or throughput claim.
func TestNativeSchedulerSkipUnusedDrafts(t *testing.T) {
	const budget = nativeQwenPrefillMinChunkTokens

	harness := func(t *testing.T, prompts map[string][]int) (*NativeScheduler, *[]SkipDraftsDecision) {
		t.Helper()
		m := nativeSchedulerPrefillModel(t)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(prompts))
		if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
			t.Fatalf("SetQwenPrefillMaxTokensPerIteration(%d): %v", budget, err)
		}
		nativeSchedulerBeginManualDrain(t, s)
		t.Cleanup(func() { nativeSchedulerEndManualDrain(s) })
		decisions := []SkipDraftsDecision{}
		s.SetSkipDraftsObserver(func(d SkipDraftsDecision) {
			decisions = append(decisions, d)
		})
		return s, &decisions
	}

	t.Run("non_final_prefill_chunk_reports_skip_and_stages_nothing", func(t *testing.T) {
		long := nativeSchedulerQwenPrompt(2*budget + 3)
		s, decisions := harness(t, map[string][]int{"long": long})
		ln := nativeSchedulerAdmitLane(t, s, "long")
		if ln.state != schedLanePrefilling {
			t.Fatalf("admitted lane state = %d, want PREFILLING", ln.state)
		}

		nativeSchedulerDriveIteration(t, s)
		if got := nativeSchedulerDrainAvailable(ln); len(got) != 0 {
			t.Fatalf("non-final prefill chunk published tokens %v, want none", got)
		}
		if ln.state != schedLanePrefilling || ln.promptCursor != budget {
			t.Fatalf("after first chunk: state %d cursor %d, want PREFILLING/%d", ln.state, ln.promptCursor, budget)
		}
		if len(*decisions) != 1 {
			t.Fatalf("decisions after first chunk = %d, want exactly one", len(*decisions))
		}
		got := (*decisions)[0]
		if !got.Skip || got.Reason != SkipDraftsPrefillChunk {
			t.Fatalf("first-chunk verdict = %+v, want skip=true reason=%s", got, SkipDraftsPrefillChunk)
		}
		if r := s.SkipDraftsReceipt(); r.Steps != 1 || r.Skipped != 1 || r.PrefillChunks != 1 || r.TerminalTokens != 0 {
			t.Fatalf("receipt after first chunk = %+v, want {Steps:1 Skipped:1 PrefillChunks:1 TerminalTokens:0}", r)
		}
	})

	t.Run("final_output_token_reports_skip", func(t *testing.T) {
		long := nativeSchedulerQwenPrompt(2*budget + 3)
		s, _ := harness(t, map[string][]int{"long": long})
		ln := nativeSchedulerAdmitLane(t, s, "long")

		type stamped struct {
			d       SkipDraftsDecision
			emitted int
		}
		var stampedDecisions []stamped
		s.SetSkipDraftsObserver(func(d SkipDraftsDecision) {
			s.mu.Lock()
			emitted := ln.emitted
			s.mu.Unlock()
			stampedDecisions = append(stampedDecisions, stamped{d, emitted})
		})

		out := []int{}
		for i := 0; i < 200 && !ln.terminal; i++ {
			nativeSchedulerDriveIteration(t, s)
			out = append(out, nativeSchedulerDrainAvailable(ln)...)
		}
		if !ln.terminal {
			t.Fatal("lane did not reach terminal within 200 iterations")
		}
		out = append(out, nativeSchedulerDrainAvailable(ln)...)
		if len(out) != genTokens {
			t.Fatalf("published %d tokens, want the fixed %d", len(out), genTokens)
		}
		firstTerminal := -1
		for i, st := range stampedDecisions {
			if st.d.Reason == SkipDraftsTerminalToken {
				firstTerminal = i
				break
			}
		}
		if firstTerminal < 0 {
			t.Fatalf("no terminal-token verdict recorded: %+v", stampedDecisions)
		}
		st := stampedDecisions[firstTerminal]
		if !st.d.Skip {
			t.Fatalf("terminal-token verdict = %+v, want skip=true", st.d)
		}
		if st.emitted != genTokens-1 {
			t.Fatalf("terminal-token verdict minted with emitted=%d, want %d (the final token just staged)", st.emitted, genTokens-1)
		}
		if firstTerminal != len(stampedDecisions)-1 {
			t.Fatalf("verdicts continued past the final-output-token step: %+v", stampedDecisions[firstTerminal:])
		}
		for i, st := range stampedDecisions[:firstTerminal] {
			if st.d.Skip && st.d.Reason != SkipDraftsPrefillChunk {
				t.Fatalf("verdict %d claims a skip without a pinned reason: %+v", i, st.d)
			}
		}
		if r := s.SkipDraftsReceipt(); r.TerminalTokens != 1 || r.PrefillChunks != 2 {
			t.Fatalf("receipt = %+v, want TerminalTokens:1 PrefillChunks:2", r)
		}
	})

	t.Run("ordinary_decode_step_with_headroom_keeps_drafts", func(t *testing.T) {
		long := nativeSchedulerQwenPrompt(2*budget + 3)
		mid := nativeSchedulerQwenPrompt(2 * budget)
		s, decisions := harness(t, map[string][]int{"long": long, "mid": mid})
		a := nativeSchedulerAdmitLane(t, s, "long")
		b := nativeSchedulerAdmitLane(t, s, "mid")

		for i := 0; i < 200 && !(a.terminal && b.terminal); i++ {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(a)
			nativeSchedulerDrainAvailable(b)
		}
		if !a.terminal || !b.terminal {
			t.Fatal("lanes did not both reach terminal within 200 iterations")
		}
		ds := *decisions
		if len(ds) < 2 {
			t.Fatalf("recorded %d decisions, want at least a prefill and a decode verdict", len(ds))
		}
		last := ds[len(ds)-1]
		if !last.Skip || last.Reason != SkipDraftsTerminalToken {
			t.Fatalf("final verdict = %+v, want skip=true reason=%s", last, SkipDraftsTerminalToken)
		}
		ordinaryDecodeSeen := false
		for i, d := range ds[:len(ds)-1] {
			if d.Reason == SkipDraftsTerminalToken {
				t.Fatalf("verdict %d claims terminal certainty before the final step: %+v", i, d)
			}
			if d.Skip && d.Reason != SkipDraftsPrefillChunk {
				t.Fatalf("verdict %d skipped drafts without a pinned reason: %+v", i, d)
			}
			if !d.Skip && d.Reason == "" {
				ordinaryDecodeSeen = true
			}
		}
		if !ordinaryDecodeSeen {
			t.Fatalf("no ordinary decode verdict (skip=false, no reason) observed: %+v", ds)
		}
	})
}
