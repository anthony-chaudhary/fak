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
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak-test:runtime fast est=1ms
func TestNativeSchedulerLaneStepTokenCost(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name                  string
		state                 schedLaneState
		prompt, cursor, chunk int
		materialized          bool
		want                  int
	}{
		{"decode", schedLaneDecode, 8, 0, 4, true, 1},
		{"prefill chunk", schedLanePrefilling, 8, 2, 4, true, 4},
		{"prefill tail", schedLanePrefilling, 8, 6, 4, true, 2},
		{"unchunked remainder", schedLanePrefilling, 8, 3, 0, true, 5},
		{"exhausted prompt", schedLanePrefilling, 8, 8, 4, true, 1},
		{"empty prompt", schedLanePrefilling, 0, 0, 0, false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ln := &schedLane{state: tc.state, prompt: make([]int, tc.prompt), promptCursor: tc.cursor, prefillChunkTokens: tc.chunk}
			if tc.materialized {
				ln.sess = &model.Session{}
			}
			if got := laneStepTokens(ln); got != tc.want {
				t.Fatalf("laneStepTokens = %d, want %d", got, tc.want)
			}
		})
	}
	// No model or run loop is required to inspect the setter's budget contract.
	s := &NativeScheduler{}
	for _, disabled := range []int{0, -1} {
		s.SetMaxTokensPerStep(8)
		s.SetMaxTokensPerStep(disabled)
		s.mu.Lock()
		got := s.maxTokensPerStep
		s.mu.Unlock()
		if got > 0 {
			t.Fatalf("SetMaxTokensPerStep(%d) left budget enabled at %d", disabled, got)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestNativeSchedulerPromotionSkipNotBreakLiveness(t *testing.T) {
	t.Parallel()
	lane := func(seq int64, chunk int) *schedLane {
		ln := &schedLane{ctx: context.Background(), state: schedLaneDecode, seqNo: seq, sess: &model.Session{}}
		if chunk > 0 {
			ln.state = schedLanePrefilling
			ln.prompt = make([]int, chunk)
			ln.prefillChunkTokens = chunk
		}
		return ln
	}
	ids := func(lanes []*schedLane) []int64 {
		out := make([]int64, len(lanes))
		for i, ln := range lanes {
			out[i] = ln.seqNo
		}
		return out
	}
	for _, tc := range []struct {
		name                 string
		budget, maxRun       int
		running, waiting     []*schedLane
		wantRun, wantWaiting []int64
	}{
		{
			name: "skip oversized and charge promoted cost", budget: 8,
			running: []*schedLane{lane(1, 0), lane(2, 0), lane(3, 0)},
			waiting: []*schedLane{lane(4, 6), lane(5, 4), lane(6, 2)},
			wantRun: []int64{1, 2, 3, 5}, wantWaiting: []int64{4, 6},
		},
		{
			name: "empty set admits first oversized only", budget: 2,
			waiting: []*schedLane{lane(1, 6), lane(2, 0)},
			wantRun: []int64{1}, wantWaiting: []int64{2},
		},
		{
			name: "disabled budget retains lane cap and FIFO", budget: 0, maxRun: 4,
			running: []*schedLane{lane(1, 0), lane(2, 0), lane(3, 0)},
			waiting: []*schedLane{lane(4, 6), lane(5, 4)},
			wantRun: []int64{1, 2, 3, 4}, wantWaiting: []int64{5},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Inert lanes isolate promotion accounting: no model, KV capacity
			// policy, device allocation, or scheduler goroutine is involved.
			s := &NativeScheduler{maxTokensPerStep: tc.budget, lanes: tc.running, waiting: tc.waiting}
			s.mu.Lock()
			s.promoteWaitingLocked(tc.maxRun)
			running, waiting := ids(s.lanes), ids(s.waiting)
			s.mu.Unlock()
			if !reflect.DeepEqual(running, tc.wantRun) || !reflect.DeepEqual(waiting, tc.wantWaiting) {
				t.Fatalf("running=%v waiting=%v, want running=%v waiting=%v", running, waiting, tc.wantRun, tc.wantWaiting)
			}
		})
	}
}

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

// TestNativeSchedulerSlotLoraInvalidation pins the llama.cpp slot-cache LoRA
// invalidation adaptation (llama.cpp PR #3325 family: changing the global LoRA
// configuration clears per-slot prompt caches keyed on the old config). The
// native scheduler's per-slot prompt cache is the recorded prefix hit a lane
// would install at its next prefill iteration, so the pinned contract is:
// a genuine global adapter-config change turns a former hit into a cold
// prefill (record gone, boundary lease released), an unchanged configuration
// still hits, and the generation counter bumps exactly once per genuine
// change. Adapter loading is untouched.
//
// Estimate only; this test has not been timed.
// fak-test:runtime fast est=100ms lane=default
func TestNativeSchedulerSlotLoraInvalidation(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(48)
	const cached = 32

	// seedTree builds a radix tree holding prompt[:cached] backed by real KV
	// and returns the boundary lease a recorded hit would carry.
	seedTree := func(t *testing.T) (*radixkv.Tree, *radixkv.Node, int) {
		t.Helper()
		ctl := m.NewSession()
		ctl.Quant = true
		ctl.Q4K = true
		ctl.PrefillNoLogits(prompt[:cached])
		tree := radixkv.New(0)
		root, _ := tree.Lookup(prompt[:cached])
		tree.Done(tree.InsertWithLogits(root, prompt[:cached], ctl.Cache, nil))
		ctl.Close()
		boundary, matched := tree.Lookup(prompt)
		if matched != cached || boundary == nil {
			t.Fatalf("seeded lookup matched=%d boundary=%v, want %d/non-nil", matched, boundary, cached)
		}
		return tree, boundary, matched
	}

	// coldScheduler pairs a bare scheduler with one prefilling lane whose
	// session is empty, mirroring the apply-clamps harness.
	coldScheduler := func(t *testing.T) (*NativeScheduler, *schedLane, *model.Session) {
		t.Helper()
		s := &NativeScheduler{m: m}
		sess := m.NewSession()
		sess.Quant = true
		sess.Q4K = true
		t.Cleanup(sess.Close)
		ln := &schedLane{sess: sess, prompt: prompt, state: schedLanePrefilling, prefillChunkTokens: nativeQwenPrefillMinChunkTokens}
		s.getOrCreatePrefixState()
		s.lanes = []*schedLane{ln}
		return s, ln, sess
	}

	t.Run("generation_bumps_only_on_genuine_change", func(t *testing.T) {
		s := NewNativeScheduler(m)
		defer s.Close()
		if got := s.LoRAGeneration(); got != 0 {
			t.Fatalf("initial LoRA generation = %d, want 0", got)
		}
		s.SetLoRAAdapters([]string{"math-lora"})
		if got := s.LoRAGeneration(); got != 1 {
			t.Fatalf("generation after first config = %d, want 1", got)
		}
		s.SetLoRAAdapters([]string{"math-lora"})
		if got := s.LoRAGeneration(); got != 1 {
			t.Fatalf("identical re-application bumped generation to %d, want 1", got)
		}
		s.SetLoRAAdapters([]string{"math-lora", "code-lora"})
		if got := s.LoRAGeneration(); got != 2 {
			t.Fatalf("generation after changed config = %d, want 2", got)
		}
		s.SetLoRAAdapters(nil)
		if got := s.LoRAGeneration(); got != 3 {
			t.Fatalf("generation after clearing config = %d, want 3", got)
		}
		s.SetLoRAAdapters([]string{})
		if got := s.LoRAGeneration(); got != 3 {
			t.Fatalf("empty list is not a distinct config, generation = %d, want 3", got)
		}
	})

	t.Run("config_change_turns_former_hit_into_cold_prefill", func(t *testing.T) {
		tree, boundary, matched := seedTree(t)
		s, ln, sess := coldScheduler(t)
		s.SetLoRAAdapters([]string{"math-lora"})
		s.recordPrefixLookup(sess, prompt, matched, matched, boundary, tree)
		if s.getPrefixHitInfo(ln) == nil {
			t.Fatal("hit not recorded under the first config")
		}
		if got := tree.Stats().ProtectedTokens; got != cached {
			t.Fatalf("protected=%d before the change, want %d (boundary lease live)", got, cached)
		}

		s.SetLoRAAdapters([]string{"code-lora"})
		if s.getPrefixHitInfo(ln) != nil {
			t.Fatal("config change left the recorded hit in place")
		}
		if got := tree.Stats().ProtectedTokens; got != 0 {
			t.Fatalf("stale boundary lease survived the change: protected=%d, want 0", got)
		}
		s.applyPrefixHitsLocked()
		if ln.promptCursor != 0 || ln.promptLen != 0 || ln.state != schedLanePrefilling {
			t.Fatalf("cursor=%d promptLen=%d state=%d, want cold prefill 0/0/PREFILLING",
				ln.promptCursor, ln.promptLen, ln.state)
		}
		if ln.sess.Cache != nil && ln.sess.Cache.Len() != 0 {
			t.Fatalf("cold prefill consumed %d cached KV tokens", ln.sess.Cache.Len())
		}
	})

	t.Run("unchanged_config_still_hits", func(t *testing.T) {
		tree, boundary, matched := seedTree(t)
		s, ln, sess := coldScheduler(t)
		s.SetLoRAAdapters([]string{"math-lora"})
		s.recordPrefixLookup(sess, prompt, matched, matched, boundary, tree)
		s.SetLoRAAdapters([]string{"math-lora"})
		if s.getPrefixHitInfo(ln) == nil {
			t.Fatal("identical re-application dropped the recorded hit")
		}
		s.applyPrefixHitsLocked()
		if ln.promptCursor != cached || ln.promptLen != cached {
			t.Fatalf("cursor=%d promptLen=%d, want both %d (hit applied)", ln.promptCursor, ln.promptLen, cached)
		}
		if got := ln.sess.Cache.Len(); got != cached {
			t.Fatalf("installed KV len = %d, want %d", got, cached)
		}
		s.clearPrefixLookup(sess)
	})

	t.Run("aliased_lane_and_holder_hits_release_lease_once", func(t *testing.T) {
		tree, boundary, matched := seedTree(t)
		s, ln, sess := coldScheduler(t)
		s.recordPrefixLookup(sess, prompt, matched, matched, boundary, tree)
		s.getOrCreatePrefixState().laneLookups[ln] = s.getPrefixHitInfo(ln)
		if s.getPrefixHitInfo(ln) == nil {
			t.Fatal("aliased hit not recorded")
		}
		s.SetLoRAAdapters([]string{"code-lora"})
		if s.getPrefixHitInfo(ln) != nil {
			t.Fatal("config change left the aliased hit in place")
		}
		if got := tree.Stats().ProtectedTokens; got != 0 {
			t.Fatalf("aliased lease released more than once or retained: protected=%d, want 0", got)
		}
	})
}
