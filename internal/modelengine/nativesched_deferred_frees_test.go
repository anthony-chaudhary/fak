package modelengine

// fak-test:runtime fast est=1s

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeSchedulerDeferredBlockFrees(t *testing.T) {
	t.Run("unobservable_before_step_publication_and_visible_after", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		s := NewNativeScheduler(m)
		var err error
		s.elastic, err = newNativeElasticLedger(8, 4)
		if err != nil {
			t.Fatalf("newNativeElasticLedger: %v", err)
		}
		s.SetDeferredBlockFrees(true)
		nativeSchedulerBeginManualDrain(t, s)
		t.Cleanup(func() { nativeSchedulerEndManualDrain(s) })

		// Two 12-token prompts -> 3 blocks each (6 allocated, 2 free of 8).
		s.prepare = func(_ context.Context, call *abi.ToolCall, _ *model.Model) schedPrepare {
			return schedPrepare{prompt: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}
		}
		laneA := nativeSchedulerAdmitLane(t, s, "lane-a")
		laneB := nativeSchedulerAdmitLane(t, s, "lane-b")

		beforeSnap := s.elastic.snapshot()
		if beforeSnap.Allocator.AllocatedBlocks != 6 || beforeSnap.Allocator.FreeBlocks != 2 {
			t.Fatalf("initial allocator = %+v, want 6 allocated / 2 free", beforeSnap.Allocator)
		}

		// Cancel laneA during step N's beforeModelExecute hook; stepDecodeCohort
		// retires laneA before advancing laneB, and laneB reaches its decode
		// event hook before step N's publication fence completes. Inspect the
		// allocator there — AFTER laneA reclaimed, BEFORE step N publishes.
		var observedDuringStep struct {
			checked         bool
			laneATerminal   bool
			processedSeq    uint64
			pendingEntries  int
			pendingBlocks   int
			allocatedBlocks int
			cachedReleases  int64
		}
		canceledA := false
		s.beforeModelExecute = func(kind nativeSchedulerEventKind, ln *schedLane) {
			if kind == nativeSchedulerEventDecode && ln == laneA && !canceledA {
				canceledA = true
				laneA.Cancel()
			}
		}
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventDecode && event.Lane == laneB && canceledA && !observedDuringStep.checked {
				observedDuringStep.checked = true
				s.mu.Lock()
				observedDuringStep.laneATerminal = laneA.terminal
				s.mu.Unlock()
				df := s.DeferredFreeStats()
				es := s.elastic.snapshot()
				observedDuringStep.processedSeq = df.ProcessedStepSeq
				observedDuringStep.pendingEntries = df.PendingEntries
				observedDuringStep.pendingBlocks = df.PendingBlocks
				observedDuringStep.allocatedBlocks = es.Allocator.AllocatedBlocks
				observedDuringStep.cachedReleases = es.CachedReleases
			}
		}

		nativeSchedulerDriveIteration(t, s)

		if !observedDuringStep.checked {
			t.Fatal("did not observe mid-step allocator state on laneB")
		}
		if !observedDuringStep.laneATerminal {
			t.Fatal("laneA did not finish on cancel before laneB forward")
		}
		if observedDuringStep.processedSeq != 0 {
			t.Fatalf("mid-step processedStepSeq = %d, want 0", observedDuringStep.processedSeq)
		}
		if observedDuringStep.pendingEntries != 1 || observedDuringStep.pendingBlocks != 3 {
			t.Fatalf("mid-step pending = (%d entries, %d blocks), want (1, 3)",
				observedDuringStep.pendingEntries, observedDuringStep.pendingBlocks)
		}
		// Crucial invariant: freed blocks are NOT observable to the allocator
		// until step N's publication completes.
		if observedDuringStep.allocatedBlocks != 6 || observedDuringStep.cachedReleases != 0 {
			t.Fatalf("mid-step allocator leaked freed blocks early: allocated=%d cachedReleases=%d, want 6/0",
				observedDuringStep.allocatedBlocks, observedDuringStep.cachedReleases)
		}

		// After step N's publication completes, processedStepSeq advances to 1
		// and laneA's 3 fenced blocks become observable to the allocator.
		dfAfter := s.DeferredFreeStats()
		esAfter := s.elastic.snapshot()
		if dfAfter.ProcessedStepSeq != 1 || dfAfter.PendingEntries != 0 || dfAfter.DrainedBlocks != 3 {
			t.Fatalf("post-step deferred free stats = %+v, want processed=1 pending=0 drained=3", dfAfter)
		}
		if esAfter.Allocator.AllocatedBlocks != 3 || esAfter.Allocator.FreeBlocks != 5 || esAfter.CachedReleases != 3 {
			t.Fatalf("post-step elastic stats = %+v, want allocated=3 free=5 cachedReleases=3", esAfter)
		}
		if err := s.elastic.alloc.VerifyInvariant(); err != nil {
			t.Fatalf("VerifyInvariant: %v", err)
		}
		_ = laneB
	})

	t.Run("reverse_drain_order_tail_blocks_evict_first", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		s := NewNativeScheduler(m)
		var err error
		s.elastic, err = newNativeElasticLedger(3, 4)
		if err != nil {
			t.Fatalf("newNativeElasticLedger: %v", err)
		}
		s.SetDeferredBlockFrees(true)

		// Allocate all 3 blocks [0, 1, 2] to one lane.
		ln := &schedLane{sched: s}
		s.elastic.laneStarted(ln, 12)
		s.elastic.mu.Lock()
		laneBlocks := append([]int(nil), s.elastic.lanes[ln].blocks...)
		s.elastic.mu.Unlock()
		if !reflect.DeepEqual(laneBlocks, []int{0, 1, 2}) {
			t.Fatalf("allocated lane blocks = %v, want [0 1 2]", laneBlocks)
		}

		// Fence the lane's blocks on step 1 and release upon step 1 publication.
		stepSeq := s.beginStepSeq()
		s.mu.Lock()
		s.finishLaneElasticBlocksLocked(ln)
		s.mu.Unlock()

		// Before publication completes, allocator still holds all 3 blocks.
		if got := s.elastic.alloc.Stats().AllocatedBlocks; got != 3 {
			t.Fatalf("before publication AllocatedBlocks = %d, want 3", got)
		}

		s.completeStepPublication(stepSeq)

		// Re-allocating from the cached FIFO must hand back tail block 2 first,
		// then 1, then head block 0 (because drain pushed them in reverse).
		if err := s.elastic.alloc.RegisterSequence("reuser", 0); err != nil {
			t.Fatalf("RegisterSequence: %v", err)
		}
		var reallocated []int
		for range 3 {
			blk, err := s.elastic.alloc.AllocBlock("reuser")
			if err != nil {
				t.Fatalf("AllocBlock: %v", err)
			}
			reallocated = append(reallocated, blk)
		}
		if !reflect.DeepEqual(reallocated, []int{2, 1, 0}) {
			t.Fatalf("reallocated cached order = %v, want reverse [2 1 0] (tail blocks evict first)", reallocated)
		}
		if err := s.elastic.alloc.VerifyInvariant(); err != nil {
			t.Fatalf("VerifyInvariant: %v", err)
		}
	})

	t.Run("close_flushes_pending_fences_without_leaking_blocks", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		s := NewNativeScheduler(m)
		var err error
		s.elastic, err = newNativeElasticLedger(6, 4)
		if err != nil {
			t.Fatalf("newNativeElasticLedger: %v", err)
		}
		s.SetDeferredBlockFrees(true)

		// Arm an in-flight step 1 and also fence an entry at step 5 so
		// neither is normally drainable yet when Close() is called.
		_ = s.beginStepSeq()
		ln := &schedLane{sched: s}
		s.elastic.laneStarted(ln, 8) // blocks [0, 1]

		var releasedOrder []int
		s.mu.Lock()
		s.finishLaneElasticBlocksLocked(ln)
		s.enqueueDeferredFreeLocked(5, []int{10, 11, 12}, func(blk int) {
			releasedOrder = append(releasedOrder, blk)
		}, nil)
		s.mu.Unlock()

		if st := s.DeferredFreeStats(); st.PendingEntries != 2 || st.PendingBlocks != 5 {
			t.Fatalf("before Close pending = %+v, want 2 entries / 5 blocks", st)
		}
		if got := s.elastic.alloc.Stats().AllocatedBlocks; got != 2 {
			t.Fatalf("before Close AllocatedBlocks = %d, want 2", got)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := s.CloseAndWait(ctx); err != nil {
			t.Fatalf("CloseAndWait: %v", err)
		}

		st := s.DeferredFreeStats()
		es := s.elastic.snapshot()
		if st.PendingEntries != 0 || st.PendingBlocks != 0 || st.FlushedOnClose < 2 {
			t.Fatalf("after Close deferred stats = %+v, want 0 pending and FlushedOnClose>=2", st)
		}
		if es.Allocator.AllocatedBlocks != 0 || es.CachedReleases != 2 {
			t.Fatalf("after Close elastic stats = %+v, want 0 allocated / 2 cached releases", es)
		}
		if !reflect.DeepEqual(releasedOrder, []int{12, 11, 10}) {
			t.Fatalf("Close flush releasedOrder = %v, want reverse [12 11 10]", releasedOrder)
		}
		if err := s.elastic.alloc.VerifyInvariant(); err != nil {
			t.Fatalf("VerifyInvariant: %v", err)
		}
	})
}
