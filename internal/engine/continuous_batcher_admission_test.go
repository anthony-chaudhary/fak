package engine

import "testing"

func TestContinuousBatcher_PrefillReservesKV(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 2
	cfg.PrefillBudget = 1
	cfg.MaxKVCacheBytes = 100

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	t.Cleanup(func() { _ = cb.Close() })

	request := func(sessionID string) *SubagentRequest {
		return &SubagentRequest{
			SessionID:       sessionID,
			PromptTokens:    []int{1, 2},
			TargetTokens:    1,
			KVBytesPerToken: 20,
			ChunkedPrefill:  true,
		}
	}

	if _, err := cb.Submit(request("first")); err != nil {
		t.Fatalf("Submit(first) failed: %v", err)
	}
	if _, err := cb.Submit(request("second")); err != nil {
		t.Fatalf("Submit(second) failed: %v", err)
	}

	if _, ok := cb.GetSlot("second"); ok {
		t.Fatal("second request was admitted while first prefilling request reserved 60% of KV capacity")
	}
	if got := len(cb.WaitingQueue()); got != 1 {
		t.Fatalf("waiting requests = %d, want 1", got)
	}

	if err := cb.Cancel("first"); err != nil {
		t.Fatalf("Cancel(first) failed: %v", err)
	}
	second, ok := cb.GetSlot("second")
	if !ok {
		t.Fatal("second request was not promoted after cancellation released reserved KV capacity")
	}
	if second.State != SlotStatePrefilling {
		t.Fatalf("second state = %s, want %s", second.State, SlotStatePrefilling)
	}
	if got := len(cb.WaitingQueue()); got != 0 {
		t.Fatalf("waiting requests after cancellation = %d, want 0", got)
	}
}
