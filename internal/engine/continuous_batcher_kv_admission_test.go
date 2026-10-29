package engine

import (
	"context"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
//
// TestKVAdmissionPrefillingCounted pins the finite-KV admission invariant for
// chunked-prefill slots: a slot held in SlotStatePrefilling counts its full
// KVCacheBytes reservation against MaxKVCacheBytes, so two overlapping prefill
// reservations whose sum exceeds the cap cannot both hold live reservations.
// Each request below reserves 6,000 bytes ((5 prompt + 1 target) * 1,000
// B/token); the cap under test is 10,000 bytes.
func TestKVAdmissionPrefillingCounted(t *testing.T) {
	t.Parallel()

	req := func(sessionID string) *SubagentRequest {
		return &SubagentRequest{
			SessionID:       sessionID,
			PromptTokens:    []int{1, 2, 3, 4, 5},
			TargetTokens:    1,
			KVBytesPerToken: 1_000,
			ChunkedPrefill:  true,
		}
	}

	newBatcher := func(t *testing.T, maxKV int64) *ContinuousBatcher {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cfg.PrefillBudget = 2
		cfg.MaxKVCacheBytes = maxKV
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })
		return cb
	}

	t.Run("OverlappingReservationsQueueSecond", func(t *testing.T) {
		t.Parallel()
		cb := newBatcher(t, 10_000) // cap < 6,000 + 6,000

		if _, err := cb.Submit(req("first")); err != nil {
			t.Fatalf("Submit(first) failed: %v", err)
		}
		first, ok := cb.GetSlot("first")
		if !ok || first.State != SlotStatePrefilling {
			t.Fatalf("first state = %v, want %s", ok, SlotStatePrefilling)
		}
		if got := cb.CurrentKVCacheBytes(); got != 6_000 {
			t.Fatalf("CurrentKVCacheBytes after first submit = %d, want 6000", got)
		}

		if _, err := cb.Submit(req("second")); err != nil {
			t.Fatalf("Submit(second) failed: %v", err)
		}
		if _, ok := cb.GetSlot("second"); ok {
			t.Fatal("second was admitted while first prefilling reservation plus its own exceed the cap")
		}
		if got := len(cb.WaitingQueue()); got != 1 {
			t.Fatalf("waiting requests = %d, want 1", got)
		}
		if got := cb.CurrentKVCacheBytes(); got != 6_000 {
			t.Fatalf("CurrentKVCacheBytes with second queued = %d, want 6000 (no overcommit)", got)
		}
	})

	t.Run("ZeroCapDefaultAdmitsBoth", func(t *testing.T) {
		t.Parallel()
		cb := newBatcher(t, 0) // package default: unconstrained

		if _, err := cb.Submit(req("first")); err != nil {
			t.Fatalf("Submit(first) failed: %v", err)
		}
		if _, err := cb.Submit(req("second")); err != nil {
			t.Fatalf("Submit(second) failed: %v", err)
		}
		for _, id := range []string{"first", "second"} {
			slot, ok := cb.GetSlot(id)
			if !ok || slot.State != SlotStatePrefilling {
				t.Fatalf("%s state = %v, want %s", id, ok, SlotStatePrefilling)
			}
		}
		if got := len(cb.WaitingQueue()); got != 0 {
			t.Fatalf("waiting requests = %d, want 0", got)
		}
		if got := cb.CurrentKVCacheBytes(); got != 12_000 {
			t.Fatalf("CurrentKVCacheBytes = %d, want 12000", got)
		}
	})

	t.Run("FinishReleasesAndPromotesWaiting", func(t *testing.T) {
		t.Parallel()
		cb := newBatcher(t, 10_000)
		ctx := context.Background()

		if _, err := cb.Submit(req("first")); err != nil {
			t.Fatalf("Submit(first) failed: %v", err)
		}
		if _, err := cb.Submit(req("second")); err != nil {
			t.Fatalf("Submit(second) failed: %v", err)
		}

		// Drive the batcher until the first request finishes naturally; its
		// retirement must free the reservation and promote the waiter.
		var first *Slot
		for i := 0; i < 50; i++ {
			if s, ok := cb.GetSlot("first"); ok && s.State == SlotStateFinished {
				first = s
				break
			}
			if _, err := cb.StepPhase(ctx); err != nil {
				t.Fatalf("StepPhase failed at iteration %d: %v", i, err)
			}
		}
		if first == nil || first.State != SlotStateFinished {
			state := "absent"
			if first != nil {
				state = string(first.State)
			}
			t.Fatalf("first state = %s, want %s within the bounded step loop", state, SlotStateFinished)
		}
		second, ok := cb.GetSlot("second")
		if !ok {
			t.Fatal("second was not promoted after first finished and released its reservation")
		}
		if second.State != SlotStatePrefilling {
			t.Fatalf("second state = %s, want %s", second.State, SlotStatePrefilling)
		}
		if got := len(cb.WaitingQueue()); got != 0 {
			t.Fatalf("waiting requests after promotion = %d, want 0", got)
		}
		if got := cb.CurrentKVCacheBytes(); got != 6_000 {
			t.Fatalf("CurrentKVCacheBytes after promotion = %d, want 6000", got)
		}
	})

	t.Run("CancelReleasesAndPromotesWaiting", func(t *testing.T) {
		t.Parallel()
		cb := newBatcher(t, 10_000)

		if _, err := cb.Submit(req("first")); err != nil {
			t.Fatalf("Submit(first) failed: %v", err)
		}
		if _, err := cb.Submit(req("second")); err != nil {
			t.Fatalf("Submit(second) failed: %v", err)
		}
		if err := cb.Cancel("first"); err != nil {
			t.Fatalf("Cancel(first) failed: %v", err)
		}
		second, ok := cb.GetSlot("second")
		if !ok {
			t.Fatal("second was not promoted after cancellation released its predecessor's reservation")
		}
		if second.State != SlotStatePrefilling {
			t.Fatalf("second state = %s, want %s", second.State, SlotStatePrefilling)
		}
		if got := len(cb.WaitingQueue()); got != 0 {
			t.Fatalf("waiting requests after cancellation = %d, want 0", got)
		}
	})
}
