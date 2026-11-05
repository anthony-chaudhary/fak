package modelengine

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestNativeSchedulerWaitingQueueGatesAndPreservesOutput is the issue-#36 acceptance-#2
// witness: the continuous-batching scheduler carries an explicit WAITING/RUNNING queue
// whose running set is recomputed each iteration. Driving the SAME admissions twice —
// uncapped (every lane co-batches) and capped at maxRunning=1 (lanes wait, then run
// strictly serially as slots free) — proves two acceptance bars at once:
//
//   - the waiting queue actually GATES the running set: the loop's own peak running-set
//     high-water mark is the cap (1 when capped, == #admitted when uncapped), so a
//     FINISHED slot is what lets the next waiting lane in — the admit/evict move; and
//   - a surviving sequence's output is BIT-IDENTICAL regardless of how many lanes ran
//     concurrently (acceptance #3): the StepBatch f32 guarantee carried through the
//     queue, so the queue changes WHEN a lane runs, never WHAT it decodes.
func TestNativeSchedulerWaitingQueueGatesAndPreservesOutput(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO"}`),
		inlineCall("get_user_details", `{"id":1}`),
		inlineCall("list_all_airports", `{"region":"EU"}`),
	}

	// Uncapped: every admitted lane is promoted at once and co-batched in one step.
	refOut, refPeak := drainAllLanes(t, m, calls, 0)
	if refPeak != len(calls) {
		t.Fatalf("uncapped peak running = %d, want %d (all lanes co-batched)", refPeak, len(calls))
	}

	// Capped at 1: the waiting queue holds the other lanes; they run only as slots free.
	gatedOut, gatedPeak := drainAllLanes(t, m, calls, 1)
	if gatedPeak != 1 {
		t.Fatalf("maxRunning=1 peak running = %d, want 1 (waiting queue did not gate)", gatedPeak)
	}

	for i := range calls {
		if len(gatedOut[i]) == 0 {
			t.Fatalf("call %d produced no tokens", i)
		}
		if !reflect.DeepEqual(gatedOut[i], refOut[i]) {
			t.Fatalf("call %d: gated output %v != uncapped output %v (queue changed the decode)",
				i, gatedOut[i], refOut[i])
		}
	}
}

// drainAllLanes admits every call into a fresh scheduler capped at maxRunning, drains
// each lane's token stream concurrently, and returns the per-lane token ids plus the
// loop's peak running-set size. With an unbuffered per-lane channel, draining all lanes
// concurrently is what lets a capped run make progress: a running lane finishes only as
// its consumer receives, freeing a slot for the next waiting lane.
func drainAllLanes(t *testing.T, m *model.Model, calls []*abi.ToolCall, maxRunning int) ([][]int, int) {
	t.Helper()
	s := NewNativeScheduler(m)
	s.SetMaxRunning(maxRunning)
	defer s.Close()

	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := s.Admit(context.Background(), c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
	}

	out := make([][]int, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for tok := range r.Tokens() {
				out[i] = append(out[i], tok.ID)
			}
		}(i, r)
	}
	wg.Wait()
	return out, s.MaxObservedRunning()
}

func TestNativeSchedulerPromotionRespectsTokenBudget(t *testing.T) {
	decode := func(seq int64) *schedLane {
		return &schedLane{ctx: context.Background(), state: schedLaneDecode, seqNo: seq}
	}
	prefilling := func(seq int64, chunk int) *schedLane {
		return &schedLane{
			ctx: context.Background(), state: schedLanePrefilling, seqNo: seq,
			prompt: make([]int, 64), prefillChunkTokens: chunk,
		}
	}
	seqs := func(lanes []*schedLane) []int64 {
		out := make([]int64, len(lanes))
		for i, ln := range lanes {
			out[i] = ln.seqNo
		}
		return out
	}

	for _, tc := range []struct {
		name        string
		budget      int
		wantRunning []int64
		wantWaiting []int64
	}{
		{name: "budget8", budget: 8, wantRunning: []int64{1, 2, 3, 5}, wantWaiting: []int64{4}},
		{name: "disabled", budget: 0, wantRunning: []int64{1, 2, 3, 4, 5}, wantWaiting: []int64{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newNativeScheduler(nil, nil)
			s.SetMaxTokensPerStep(tc.budget)
			s.mu.Lock()
			s.lanes = []*schedLane{decode(1), decode(2), decode(3)}
			s.waiting = []*schedLane{prefilling(4, 6), prefilling(5, 4)}
			s.promoteWaitingLocked(s.effectiveMaxRunningLocked())
			running, waiting := seqs(s.lanes), seqs(s.waiting)
			s.mu.Unlock()
			if !reflect.DeepEqual(running, tc.wantRunning) || !reflect.DeepEqual(waiting, tc.wantWaiting) {
				t.Fatalf("running=%v waiting=%v, want running=%v waiting=%v", running, waiting, tc.wantRunning, tc.wantWaiting)
			}
		})
	}
}

// TestNativeSchedulerBatchSequenceCheck pins the ported llama.cpp batch
// sequence-membership validation (ggml-org/llama.cpp llama_batch_allocr::init,
// src/llama-batch.cpp:475 @ 25603ef89168a570ea4f34803ce36531c05fd357, MIT,
// PR #30386): a proposed batch whose sequence ids fall outside the scheduler's
// active set is refused with a typed error before any slot or queue state
// moves, while known sequences pass in their given order.
// fak-test:runtime fast est=50ms
func TestNativeSchedulerBatchSequenceCheck(t *testing.T) {
	lane := func(seq int64, state schedLaneState) *schedLane {
		return &schedLane{ctx: context.Background(), state: state, seqNo: seq}
	}
	type batchCheckSnap struct {
		counter int64
		lanes   []int64
		waiting []int64
		preempt []int64
		closed  bool
	}
	snapshot := func(s *NativeScheduler) batchCheckSnap {
		s.mu.Lock()
		defer s.mu.Unlock()
		snap := batchCheckSnap{counter: s.seqNo, closed: s.closed,
			lanes: make([]int64, 0, len(s.lanes)), waiting: make([]int64, 0, len(s.waiting)),
			preempt: make([]int64, 0, len(s.preempted))}
		for _, ln := range s.lanes {
			snap.lanes = append(snap.lanes, ln.seqNo)
		}
		for _, ln := range s.waiting {
			snap.waiting = append(snap.waiting, ln.seqNo)
		}
		for _, ln := range s.preempted {
			snap.preempt = append(snap.preempt, ln.seqNo)
		}
		return snap
	}

	t.Run("unknown_sequence_refused_without_state_change", func(t *testing.T) {
		s := newNativeScheduler(nil, nil)
		s.mu.Lock()
		s.lanes = []*schedLane{lane(1, schedLaneDecode), lane(2, schedLaneDecode)}
		s.waiting = []*schedLane{lane(3, schedLanePrefilling)}
		s.preempted = []*schedLane{lane(4, schedLaneDecode)}
		s.mu.Unlock()

		before := snapshot(s)
		err := s.ValidateBatchSequences([]int64{1, 3, 99})
		if !errors.Is(err, ErrNativeBatchSequenceNotActive) {
			t.Fatalf("ValidateBatchSequences = %v, want ErrNativeBatchSequenceNotActive", err)
		}
		var unknown *BatchSequenceError
		if !errors.As(err, &unknown) || unknown.SeqNo != 99 {
			t.Fatalf("err = %v, want *BatchSequenceError with SeqNo 99", err)
		}
		if after := snapshot(s); !reflect.DeepEqual(after, before) {
			t.Fatalf("refusal changed scheduler state: %+v -> %+v", before, after)
		}
	})

	t.Run("known_sequences_admit_in_order", func(t *testing.T) {
		s := newNativeScheduler(nil, nil)
		s.mu.Lock()
		s.lanes = []*schedLane{lane(1, schedLaneDecode), lane(2, schedLaneDecode)}
		s.waiting = []*schedLane{lane(3, schedLanePrefilling)}
		s.mu.Unlock()

		// Known ids pass in the caller's order, duplicates included.
		if err := s.ValidateBatchSequences([]int64{3, 1, 3, 2}); err != nil {
			t.Fatalf("ValidateBatchSequences(known) = %v, want nil", err)
		}
		// The first unknown in input order is the one reported.
		var unknown *BatchSequenceError
		if err := s.ValidateBatchSequences([]int64{2, 77, 88}); !errors.As(err, &unknown) || unknown.SeqNo != 77 {
			t.Fatalf("err = %v, want first unknown in input order (77)", err)
		}
	})

	t.Run("admitted_lanes_are_active_foreign_id_is_not", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		s := NewNativeScheduler(m)
		// Hold the executor so no lane is promoted out of the waiting queue
		// while the membership assertions read stable admission identities.
		s.executor.Lock()
		var unlock sync.Once
		release := func() { unlock.Do(s.executor.Unlock) }
		t.Cleanup(func() {
			s.Close()
			release()
			wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
			defer stop()
			if err := s.CloseAndWait(wait); err != nil {
				t.Errorf("batch sequence fixture did not stop: %v", err)
			}
		})

		calls := []*abi.ToolCall{
			inlineCall("search_flights", `{"from":"SFO"}`),
			inlineCall("get_user_details", `{"id":1}`),
		}
		reqs := make([]abi.EngineRequest, len(calls))
		for i, c := range calls {
			r, err := s.Admit(context.Background(), c)
			if err != nil {
				t.Fatalf("Admit %d: %v", i, err)
			}
			reqs[i] = r
		}

		s.mu.Lock()
		nos := make([]int64, 0, len(s.waiting)+len(s.lanes))
		for _, ln := range s.waiting {
			nos = append(nos, ln.seqNo)
		}
		for _, ln := range s.lanes {
			nos = append(nos, ln.seqNo)
		}
		s.mu.Unlock()
		if len(nos) != len(calls) {
			t.Fatalf("admitted %d live lanes, want %d", len(nos), len(calls))
		}

		if err := s.ValidateBatchSequences(nos); err != nil {
			t.Fatalf("ValidateBatchSequences(admitted %v) = %v, want nil", nos, err)
		}
		foreign := append(append([]int64{}, nos...), -1)
		if err := s.ValidateBatchSequences(foreign); !errors.Is(err, ErrNativeBatchSequenceNotActive) {
			t.Fatalf("ValidateBatchSequences(foreign %v) = %v, want ErrNativeBatchSequenceNotActive", foreign, err)
		}

		release()
		var wg sync.WaitGroup
		for _, r := range reqs {
			wg.Add(1)
			go func(r abi.EngineRequest) {
				defer wg.Done()
				for range r.Tokens() {
				}
			}(r)
		}
		wg.Wait()
	})
}
