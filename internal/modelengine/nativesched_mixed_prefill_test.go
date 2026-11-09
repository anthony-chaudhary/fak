//go:build fak_vllm_packed_prefill

package modelengine

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=3s lane=default
func TestNativeMixedPackedPrefill(t *testing.T) {
	t.Run("owned_shared_forward_and_serial_parity", func(t *testing.T) {
		e, s, hot, first := mixedPackedFixture(t, 1)
		cold := mixedPackedAdmit(t, e, "cold")
		shared := false
		s.beforeModelExecute = func(_ nativeSchedulerEventKind, _ *schedLane) {
			if hot[0].inflightSess != nil && cold.inflightSess != nil {
				shared = true
			}
		}
		before := s.PackedPrefillReceipt()
		rec := enginestep.New(32)
		s.recorder = rec
		var decisions []SkipDraftsDecision
		s.SetSkipDraftsObserver(func(d SkipDraftsDecision) { decisions = append(decisions, d) })
		draftBefore := s.SkipDraftsReceipt()
		nativeSchedulerDriveIteration(t, s)
		if !shared {
			t.Fatal("shared owned forward=false")
		}
		if len(decisions) != 1 || decisions[0].Skip || s.SkipDraftsReceipt().Steps-draftBefore.Steps != 1 {
			t.Fatalf("one mixed call emitted contradictory draft verdicts: %+v", decisions)
		}
		gotHot := append(first[0], nativeSchedulerDrainAvailable(hot[0])...)
		if len(gotHot) != 2 || cold.promptCursor != 31 || hot[0].sess.Cache.Len() != 5 {
			t.Fatalf("duplicate/missing advance: hot=%v cold_cursor=%d hot_kv=%d", gotHot, cold.promptCursor, hot[0].sess.Cache.Len())
		}
		stats := s.PackedPrefillReceipt()
		if stats.TokenRows-before.TokenRows != 31 || stats.Sequences-before.Sequences != 1 || stats.MaxTokenRows != 32 ||
			mixedPackedCount(stats, "DecodeRows") != 1 || mixedPackedCount(stats, "MixedCalls") != 1 {
			t.Fatalf("prompt/decode accounting: %+v before=%+v", stats, before)
		}
		snap := rec.Snapshot(32, "")
		if got := snap.Decode[enginestep.PathBatched]; got.Steps != 1 || got.Tokens != 1 {
			t.Fatalf("mixed decode telemetry missing: %+v", got)
		}
		if got := snap.Decode[enginestep.PathSerial]; got.Steps != 0 {
			t.Fatalf("mixed decode replayed: %+v", got)
		}
		reference := s.m.NewSession()
		defer reference.Close()
		logits := reference.Prefill(packedServingPrompt(3, 7))
		for range 2 {
			logits = reference.Step(argmax(logits))
		}
		if !reflect.DeepEqual(hot[0].logits, logits) || !reflect.DeepEqual(hot[0].sess.Cache, reference.Cache) || hot[0].sess.SequencePosition() != reference.SequencePosition() {
			t.Fatal("mixed decode logits/cache/position differs from serial")
		}
		gotCold := nativeSchedulerDrainAvailable(cold)
		for step := 0; step < genTokens+8 && (!hot[0].terminal || !cold.terminal); step++ {
			nativeSchedulerDriveIteration(t, s)
			gotHot = append(gotHot, nativeSchedulerDrainAvailable(hot[0])...)
			gotCold = append(gotCold, nativeSchedulerDrainAvailable(cold)...)
		}
		for i, ln := range []*schedLane{hot[0], cold} {
			if !ln.terminal || !ln.Reclaimed() {
				t.Fatalf("owner %d not retired", i)
			}
			if _, err := ln.Result(); err != nil {
				t.Fatal(err)
			}
		}
		if !reflect.DeepEqual(gotHot, nativeSchedulerSynchronousTokens(s.m, packedServingPrompt(3, 7))) ||
			!reflect.DeepEqual(gotCold, nativeSchedulerSynchronousTokens(s.m, packedServingPrompt(37, 19))) {
			t.Fatal("served token stream differs from serial")
		}
	})
	t.Run("declined_prompt_preserves_healthy_decoder", func(t *testing.T) {
		e, s, hot, first := mixedPackedFixture(t, 1)
		cold := mixedPackedAdmit(t, e, "cold")
		cold.prompt[0] = s.m.Cfg.VocabSize
		nativeSchedulerDriveIteration(t, s)
		out := append(first[0], nativeSchedulerDrainAvailable(hot[0])...)
		if !cold.terminal || hot[0].terminal || len(out) != 2 || hot[0].emitted != 2 || hot[0].sess.Cache.Len() != 5 {
			t.Fatalf("decline damaged hot owner: cold_terminal=%v hot_terminal=%v emitted=%d out=%v", cold.terminal, hot[0].terminal, hot[0].emitted, out)
		}
		if got := mixedPackedCount(s.PackedPrefillReceipt(), "MixedCalls"); got != 0 {
			t.Fatalf("declined panel counted: %d", got)
		}
		want := nativeSchedulerSynchronousTokens(s.m, packedServingPrompt(3, 7))
		if !reflect.DeepEqual(out, want[:2]) {
			t.Fatalf("decline changed tokens: %v != %v", out, want[:2])
		}
	})
	t.Run("hot_swap_keeps_legacy_forward", func(t *testing.T) {
		e, s, hot, _ := mixedPackedFixture(t, 1)
		cold := mixedPackedAdmit(t, e, "cold")
		calls := 0
		impl := DefaultForwardImpl()
		impl.Step = func(sess *model.Session, id int) []float32 { calls++; return sess.Step(id) }
		if _, err := e.ForwardSlot().Install(ForwardDelta{Base: 1, Impl: impl}); err != nil {
			t.Fatal(err)
		}
		nativeSchedulerDriveIteration(t, s)
		if calls != 1 || cold.promptCursor != 32 || len(nativeSchedulerDrainAvailable(hot[0])) != 1 || mixedPackedCount(s.PackedPrefillReceipt(), "MixedCalls") != 0 {
			t.Fatalf("hot-swap bypass: calls=%d cold=%d receipt=%+v", calls, cold.promptCursor, s.PackedPrefillReceipt())
		}
	})
	t.Run("overflow_decoders_advance_once", func(t *testing.T) {
		e, s, hot, _ := mixedPackedFixture(t, 32)
		cold := mixedPackedAdmit(t, e, "cold")
		nativeSchedulerDriveIteration(t, s)
		if cold.promptCursor != 1 {
			t.Fatalf("prefill progress=%d, want one reserved row", cold.promptCursor)
		}
		for i, ln := range hot {
			if got := nativeSchedulerDrainAvailable(ln); len(got) != 1 || ln.sess.Cache.Len() != 5 {
				t.Fatalf("lane %d advanced incorrectly: out=%v cache=%d", i, got, ln.sess.Cache.Len())
			}
		}
		stats := s.PackedPrefillReceipt()
		if mixedPackedCount(stats, "DecodeRows") != 31 || stats.MaxTokenRows != 32 {
			t.Fatalf("overflow panel: %+v", stats)
		}
	})
	t.Run("one_generation_including_overflow", func(t *testing.T) {
		e, s, hot, _ := mixedPackedFixture(t, 32)
		mixedPackedAdmit(t, e, "cold")
		installed, newCalls := false, 0
		s.beforeModelExecute = func(_ nativeSchedulerEventKind, _ *schedLane) {
			if installed {
				return
			}
			installed = true
			impl := DefaultForwardImpl()
			impl.Step = func(sess *model.Session, id int) []float32 { newCalls++; return sess.Step(id) }
			impl.StepBatch = func(bs *model.BatchSession, ids []int) [][]float32 { newCalls += len(ids); return bs.StepBatch(ids) }
			if _, err := e.ForwardSlot().Install(ForwardDelta{Base: 1, Impl: impl}); err != nil {
				t.Fatal(err)
			}
		}
		nativeSchedulerDriveIteration(t, s)
		if newCalls != 0 {
			t.Fatalf("mixed iteration crossed generations: %d new calls", newCalls)
		}
		for _, ln := range hot {
			nativeSchedulerDrainAvailable(ln)
		}
		nativeSchedulerDriveIteration(t, s)
		if newCalls == 0 {
			t.Fatal("next iteration did not adopt new generation")
		}
	})
	t.Run("full_outbox_keeps_legacy_forward_order", func(t *testing.T) {
		e, s, hot, _ := mixedPackedFixture(t, 2)
		cold := mixedPackedAdmit(t, e, "cold")
		hot[0].tokens = make(chan abi.EngineToken, 1)
		hot[0].tokens <- abi.EngineToken{ID: -1}
		// The blocked owner's regular decode callback must run before publication
		// can block; free the existing outbox slot only at that boundary.
		legacy := false
		s.beforeModelExecute = func(kind nativeSchedulerEventKind, ln *schedLane) {
			if kind == nativeSchedulerEventDecode && ln == hot[0] {
				legacy = cold.inflightSess == nil
				<-hot[0].tokens
			}
		}
		nativeSchedulerDriveIteration(t, s)
		if !legacy || cold.promptCursor != 31 {
			t.Fatalf("full outbox joined mixed panel: legacy=%v cursor=%d", legacy, cold.promptCursor)
		}
		for _, ln := range hot {
			if len(nativeSchedulerDrainAvailable(ln)) != 1 {
				t.Fatal("decode advance lost")
			}
		}
	})
	t.Run("cold_cancel_keeps_decode_cadence", func(t *testing.T) {
		e, s, hot, _ := mixedPackedFixture(t, 1)
		cold := mixedPackedAdmit(t, e, "cold")
		before := s.executedSteps
		s.beforeModelExecute = func(_ nativeSchedulerEventKind, _ *schedLane) { cold.Cancel() }
		nativeSchedulerDriveIteration(t, s)
		if len(nativeSchedulerDrainAvailable(hot[0])) != 1 || hot[0].terminal || !cold.terminal || s.executedSteps != before+1 {
			t.Fatalf("mixed execution lost cadence: before=%d after=%d hot_terminal=%v cold_terminal=%v", before, s.executedSteps, hot[0].terminal, cold.terminal)
		}
	})
	for _, action := range []string{"cancel", "close", "panic"} {
		t.Run(action+"_retires_all_borrows", func(t *testing.T) {
			e, s, hot, _ := mixedPackedFixture(t, 1)
			cold := mixedPackedAdmit(t, e, "cold")
			shared := false
			s.beforeModelExecute = func(_ nativeSchedulerEventKind, _ *schedLane) {
				if shared {
					return
				}
				shared = hot[0].inflightSess != nil && cold.inflightSess != nil
				if !shared {
					t.Fatal("owners were not borrowed together")
				}
				switch action {
				case "cancel":
					hot[0].Cancel()
					cold.Cancel()
				case "close":
					s.Close()
				case "panic":
					panic("mixed forward witness")
				}
			}
			var raised any
			func() {
				s.executor.Lock()
				defer s.executor.Unlock()
				defer func() { raised = recover() }()
				s.runIteration(true)
			}()
			if (action == "panic") != (raised != nil) {
				t.Fatalf("panic=%v for %s", raised, action)
			}
			s.beforeModelExecute = nil
			for i, ln := range []*schedLane{hot[0], cold} {
				if !ln.terminal || !ln.Reclaimed() || ln.inflightSess != nil {
					t.Fatalf("owner %d retained after %s", i, action)
				}
				if out := nativeSchedulerDrainAvailable(ln); len(out) != 0 {
					t.Fatalf("published after %s: %v", action, out)
				}
			}
		})
	}
}

func mixedPackedCount(stats PackedPrefillReceipt, field string) uint64 {
	v := reflect.ValueOf(stats).FieldByName(field)
	if !v.IsValid() {
		return 0
	}
	return v.Uint()
}

func mixedPackedAdmit(t *testing.T, e *Engine, name string) *schedLane {
	t.Helper()
	req, err := e.Admit(context.Background(), inlineCall(name, "{}"))
	if err != nil {
		t.Fatal(err)
	}
	return req.(*schedLane)
}

func mixedPackedFixture(t *testing.T, count int) (*Engine, *NativeScheduler, []*schedLane, [][]int) {
	t.Helper()
	e := New()
	e.Preload(packedServingModel())
	s := e.nativeScheduler()
	if !s.packedPrefillEnabled {
		t.Fatal("ordinary Engine did not select packed qualification")
	}
	s.SetMaxRunning(count + 1)
	s.prepare = func(_ context.Context, call *abi.ToolCall, _ *model.Model) schedPrepare {
		if call.Tool == "cold" {
			return schedPrepare{prompt: packedServingPrompt(37, 19)}
		}
		return schedPrepare{prompt: packedServingPrompt(3, 7)}
	}
	nativeSchedulerBeginManualDrain(t, s)
	t.Cleanup(func() { s.beforeModelExecute = nil; nativeSchedulerEndManualDrain(s) })
	// The normal unselected admission prepares hot lanes; re-arm only the
	// existing qualification for the late prompt, without mutating cache state.
	s.packedPrefillEnabled = false
	hot := make([]*schedLane, count)
	for i := range hot {
		hot[i] = mixedPackedAdmit(t, e, fmt.Sprintf("hot-%d", i))
	}
	s.packedPrefillEnabled = true
	nativeSchedulerDriveIteration(t, s)
	first := make([][]int, count)
	for i, ln := range hot {
		first[i] = nativeSchedulerDrainAvailable(ln)
		if len(first[i]) != 1 {
			t.Fatalf("hot fixture lane %d produced %v", i, first[i])
		}
	}
	return e, s, hot, first
}
