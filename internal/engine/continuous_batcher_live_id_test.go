package engine

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=5ms lane=default
func TestBatchingEngineReservesLiveSessionIDUntilDrain(t *testing.T) {
	t.Parallel()
	newEngine := func(t *testing.T) *BatchingEngine {
		t.Helper()
		cfg := DefaultBatchingEngineConfig()
		cfg.Batcher.MaxSlots = 1
		cfg.Batcher.PrefillBudget = 0
		cb, err := NewContinuousBatcher(cfg.Batcher)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		// Keep the scheduler paused at exact ownership boundaries. Admission,
		// stepping, pumping and cancellation still use the production methods.
		e := &BatchingEngine{
			cfg: cfg, cb: cb, requests: make(map[string]*batcherRequest),
			ctx: ctx, cancel: cancel, done: make(chan struct{}),
			wake: make(chan struct{}, 1),
		}
		t.Cleanup(func() { cancel(); e.finishAll(ErrBatcherClosed); _ = cb.Close() })
		return e
	}
	call := func(id string) *abi.ToolCall {
		args, err := json.Marshal(batcherAdmitArgs{
			SessionID: id, PromptTokens: []int{41, 42}, TargetTokens: 1,
		})
		if err != nil {
			t.Fatal(err)
		}
		return &abi.ToolCall{Tool: "test", Args: abi.Ref{Kind: abi.RefInline, Inline: args}}
	}
	admit := func(t *testing.T, e *BatchingEngine, id string) *batcherRequest {
		t.Helper()
		h, err := e.Admit(context.Background(), call(id))
		if err != nil {
			t.Fatal(err)
		}
		r := h.(*batcherRequest)
		t.Cleanup(r.Cancel)
		return r
	}
	step := func(t *testing.T, e *BatchingEngine) {
		t.Helper()
		if _, err := e.cb.StepPhase(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	assertResult := func(t *testing.T, r *batcherRequest) {
		t.Helper()
		if !r.finished() {
			t.Fatal("driver orphaned a request handle")
		}
		if result, err := r.Result(); err != nil || result == nil {
			t.Fatalf("Result = %v, %v", result, err)
		}
		if len(r.gen) != 1 {
			t.Fatalf("generated token count = %d, want 1", len(r.gen))
		}
		if tok, open := <-r.Tokens(); !open || tok.ID != r.gen[0] {
			t.Fatal("terminal drain lost the generated token")
		}
		if _, open := <-r.Tokens(); open {
			t.Fatal("terminal token channel remains open")
		}
	}

	t.Run("retired_then_queued_reuse", func(t *testing.T) {
		e := newEngine(t)
		old := admit(t, e, "reused")
		step(t, e) // Producer retires the slot before the driver pumps it.
		oldSlot, ok := e.cb.GetSlot("reused")
		if !ok || oldSlot.State != SlotStateFinished || old.finished() {
			t.Fatal("test requires retired producer and undrained live handle")
		}
		if h, err := e.Admit(context.Background(), call("reused")); h != nil || !errors.Is(err, ErrSessionAlreadyExists) {
			t.Fatalf("undrained duplicate Admit = %v, %v", h, err)
		}
		if got, ok := e.cb.GetSlot("reused"); !ok || got != oldSlot || e.requests["reused"] != old {
			t.Fatal("rejected duplicate replaced the old slot or handle")
		}
		e.pumpAll()
		assertResult(t, old)
		if e.outstanding() != 0 {
			t.Fatal("terminal drain did not release the ID")
		}

		blocker := admit(t, e, "blocker")
		current := admit(t, e, "reused")
		if e.cb.WaitingQueueLength() != 1 {
			t.Fatal("reuse must wait behind the occupied slot")
		}
		if _, ok := e.cb.GetSlot("reused"); ok {
			t.Fatal("queued replacement still resolves to the old completed slot")
		}
		old.Cancel() // A late caller cancellation owns only the old context.
		e.pumpAll()
		if current.finished() || e.requests["reused"] != current {
			t.Fatal("queued replacement finished before its own producer ran")
		}
		blocker.Cancel()
		e.pumpAll()
		if !blocker.finished() {
			t.Fatal("driver did not complete cancellation")
		}
		if _, err := blocker.Result(); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled Result error = %v", err)
		}
		step(t, e)
		e.pumpAll()
		assertResult(t, current)
		if e.outstanding() != 0 {
			t.Fatal("replacement handle was not released")
		}
		e.cancel()
		if h, err := e.Admit(context.Background(), call("reused")); h != nil || !errors.Is(err, ErrBatcherClosed) {
			t.Fatalf("closed Admit = %v, %v", h, err)
		}
	})

	t.Run("generated_id_skips_undrained_explicit_id", func(t *testing.T) {
		e := newEngine(t)
		old := admit(t, e, "subagent-1")
		step(t, e)
		current := admit(t, e, "")
		if current.sessionID != "subagent-2" || e.requests[old.sessionID] != old || e.outstanding() != 2 {
			t.Fatal("canonical automatic ID allocation replaced the live explicit handle")
		}
		e.pumpAll()
		assertResult(t, old)
		step(t, e)
		e.pumpAll()
		assertResult(t, current)
		if e.outstanding() != 0 {
			t.Fatal("generated request handles were not released")
		}
	})
}
