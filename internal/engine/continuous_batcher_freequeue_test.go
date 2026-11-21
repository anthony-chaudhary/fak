package engine

import (
	"context"
	"reflect"
	"testing"
)

// TestContinuousBatcher_FreeQueue proves the M2 free-queue batching discipline:
// the free-queue is the LIVE set of empty slots, derived on each call, with NO
// steady-state batch-band hold. A freed slot is immediately available.
func TestContinuousBatcher_FreeQueue(t *testing.T) {
	t.Parallel()

	newBatcher := func(t *testing.T, maxSlots int) *ContinuousBatcher {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = maxSlots
		cfg.Model = nil // pure-scheduler mode: deterministic, no model forward pass
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })
		return cb
	}

	// submit fills slots 0..n-1 with active-decode requests of the given target.
	submit := func(t *testing.T, cb *ContinuousBatcher, session string, target int) {
		t.Helper()
		if _, err := cb.Submit(&SubagentRequest{
			SessionID:    session,
			TargetTokens: target,
		}); err != nil {
			t.Fatalf("Submit %s failed: %v", session, err)
		}
	}

	t.Run("AscendingEmptyIndices_KnownLayout", func(t *testing.T) {
		cb := newBatcher(t, 4)

		// Occupy slots 0,1,2; leave slot 3 empty.
		submit(t, cb, "A", 1)   // slot 0, finishes after one step
		submit(t, cb, "B", 100) // slot 1, stays active
		submit(t, cb, "C", 1)   // slot 2, finishes after one step

		// Before any step: only slot 3 is empty.
		if got := cb.FreeQueue(); !reflect.DeepEqual(got, []int{3}) {
			t.Fatalf("FreeQueue before step = %v, want [3]", got)
		}

		// One decode step: A (slot 0) and C (slot 2) reach their target and are
		// retired to Empty; B (slot 1) stays active.
		if _, err := cb.Step(context.Background()); err != nil {
			t.Fatalf("Step failed: %v", err)
		}

		// The free-queue is the live set of empty slots: 0, 2, 3 — ascending and
		// non-contiguous. This is the known layout the scheduler admits from.
		if got := cb.FreeQueue(); !reflect.DeepEqual(got, []int{0, 2, 3}) {
			t.Fatalf("FreeQueue after step = %v, want [0 2 3]", got)
		}
	})

	t.Run("FreedSlotImmediatelyAvailable_NoBandHold", func(t *testing.T) {
		cb := newBatcher(t, 4)

		submit(t, cb, "A", 1)
		submit(t, cb, "B", 100)
		submit(t, cb, "C", 1)

		// The two short requests occupy slots 0 and 2; their slots are NOT yet free.
		before := cb.FreeQueue()
		if reflect.DeepEqual(before, []int{0, 2, 3}) {
			t.Fatalf("FreeQueue before step = %v; freed slots must not appear early", before)
		}

		// Step retires A and C. Their slots return to Empty.
		if _, err := cb.Step(context.Background()); err != nil {
			t.Fatalf("Step failed: %v", err)
		}

		// NO steady-state batch-band hold: the freed slots 0 and 2 are immediately
		// present in the very next FreeQueue() call, with no extra batch-completion
		// event required.
		after := cb.FreeQueue()
		if !reflect.DeepEqual(after, []int{0, 2, 3}) {
			t.Fatalf("FreeQueue after step = %v, want [0 2 3] (freed slots immediately available)", after)
		}
	})

	t.Run("AllOccupied_ReturnsEmptyNonNil", func(t *testing.T) {
		cb := newBatcher(t, 4)

		// Fill every slot with a long-running request.
		for _, s := range []string{"A", "B", "C", "D"} {
			submit(t, cb, s, 100)
		}
		if got := cb.ActiveSlotCount(); got != 4 {
			t.Fatalf("ActiveSlotCount = %d, want 4", got)
		}

		fq := cb.FreeQueue()
		if fq == nil {
			t.Fatalf("FreeQueue returned nil when all slots occupied; want empty non-nil slice")
		}
		if len(fq) != 0 {
			t.Fatalf("FreeQueue = %v when all slots occupied, want empty", fq)
		}
	})
}
