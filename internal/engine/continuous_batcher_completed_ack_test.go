package engine

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=5ms lane=default
func TestBatchingEngineAcknowledgesCompletedOwnership(t *testing.T) {
	t.Parallel()
	newEngine := func(t *testing.T) *BatchingEngine {
		t.Helper()
		cfg := DefaultBatchingEngineConfig()
		cfg.Batcher.MaxSlots = 2
		cb, err := NewContinuousBatcher(cfg.Batcher)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		// Delay the real scheduler's start for deterministic lifecycle boundaries.
		// Admission, stepping, pumping and shutdown below are production methods.
		e := &BatchingEngine{
			cfg: cfg, cb: cb, requests: make(map[string]*batcherRequest),
			ctx: ctx, cancel: cancel, done: make(chan struct{}),
			wake: make(chan struct{}, 1),
		}
		t.Cleanup(func() { cancel(); _ = cb.Close() })
		return e
	}
	admit := func(t *testing.T, e *BatchingEngine, id string) (*batcherRequest, *Slot) {
		t.Helper()
		args, err := json.Marshal(batcherAdmitArgs{
			SessionID: id, PromptTokens: []int{41, 42}, TargetTokens: 1,
		})
		if err != nil {
			t.Fatal(err)
		}
		handle, err := e.Admit(context.Background(), &abi.ToolCall{
			Tool: "test", Args: abi.Ref{Kind: abi.RefInline, Inline: args},
		})
		if err != nil {
			t.Fatal(err)
		}
		r := handle.(*batcherRequest)
		t.Cleanup(r.Cancel)
		slot, ok := e.cb.GetSlot(id)
		if !ok {
			t.Fatal("admitted request has no slot")
		}
		return r, slot
	}
	assertReleased := func(t *testing.T, e *BatchingEngine, slot *Slot) {
		t.Helper()
		if e.outstanding() != 0 {
			t.Fatal("driver retained the terminal request")
		}
		e.cb.mu.Lock()
		n := len(e.cb.completedMap)
		e.cb.mu.Unlock()
		if n != 0 {
			t.Fatalf("driver retained %d completed slots after terminal consumption", n)
		}
		// Dropping an index reference must not erase an externally held Slot.
		if slot.Request == nil || len(slot.PromptTokens) != 2 || slot.PromptTokens[0] != 41 {
			t.Fatal("acknowledgement mutated the externally held slot payload")
		}
		select {
		case <-slot.Done():
		default:
			t.Fatal("slot ownership was released before producer completion")
		}
	}

	t.Run("normal", func(t *testing.T) {
		e := newEngine(t)
		r, slot := admit(t, e, "normal")
		if _, err := e.cb.StepPhase(context.Background()); err != nil {
			t.Fatal(err)
		}
		if got, ok := e.cb.GetSlot("normal"); !ok || got != slot {
			t.Fatal("completed slot disappeared before the driver consumed it")
		}
		e.pumpAll()
		if !r.finished() {
			t.Fatal("driver did not publish completion")
		}
		if result, err := r.Result(); err != nil || result == nil {
			t.Fatalf("normal Result = %v, %v", result, err)
		}
		if tok, open := <-r.Tokens(); !open || len(slot.GeneratedTokens) != 1 || tok.ID != slot.GeneratedTokens[0] {
			t.Fatal("acknowledgement lost the final generated token")
		}
		assertReleased(t, e, slot)
	})
	t.Run("cancel", func(t *testing.T) {
		e := newEngine(t)
		r, slot := admit(t, e, "cancel")
		r.Cancel()
		e.pumpAll()
		if !r.finished() {
			t.Fatal("driver did not publish cancellation")
		}
		if result, err := r.Result(); result != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled Result = %v, %v", result, err)
		}
		assertReleased(t, e, slot)
	})
	t.Run("close_after_finish_all", func(t *testing.T) {
		e := newEngine(t)
		r, slot := admit(t, e, "close")
		// Close makes run finish handles first; cb.Close retires active slots
		// afterwards. A pre-Close acknowledgement alone misses those entries.
		e.cancel()
		go e.run()
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		if result, err := r.Result(); result != nil || !errors.Is(err, ErrBatcherClosed) {
			t.Fatalf("closed Result = %v, %v", result, err)
		}
		assertReleased(t, e, slot)
	})
	t.Run("stale_acknowledgement", func(t *testing.T) {
		e := newEngine(t)
		old, oldSlot := admit(t, e, "reused")
		if err := e.cb.Cancel("reused"); err != nil {
			t.Fatal(err)
		}
		// A live-ID reservation requires the old handle to drain before reuse.
		// Replay its acknowledgement below only after a new submission exists.
		e.pumpAll()
		if !old.finished() {
			t.Fatal("old handle must finish before the ID is reused")
		}
		current, currentSlot := admit(t, e, "reused")
		if oldSlot.SubmissionSeq == currentSlot.SubmissionSeq {
			t.Fatal("test requires distinct accepted submissions")
		}
		if err := e.cb.Cancel("reused"); err != nil {
			t.Fatal(err)
		}
		e.removeRequest(old)
		if e.requests["reused"] != current {
			t.Fatal("stale acknowledgement removed the newer request handle")
		}
		if got, ok := e.cb.GetSlot("reused"); !ok || got != currentSlot {
			t.Fatal("stale acknowledgement removed the newer completed slot")
		}
		e.cancelRequest(current)
		assertReleased(t, e, currentSlot)
	})
	t.Run("standalone_history", func(t *testing.T) {
		e := newEngine(t)
		// Use the batcher directly, without an owning driver handle.
		id, err := e.cb.Submit(&SubagentRequest{SessionID: "standalone", TargetTokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		slot, _ := e.cb.GetSlot(id)
		if err := e.cb.Cancel(id); err != nil {
			t.Fatal(err)
		}
		if err := e.cb.Close(); err != nil {
			t.Fatal(err)
		}
		if got, ok := e.cb.GetSlot(id); !ok || got != slot {
			t.Fatal("standalone completed-slot lookup contract changed")
		}
	})
}
