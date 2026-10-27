package engine

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestContinuousBatcherQueueRetirementClearsBackingSlots(t *testing.T) {
	t.Parallel()
	for _, promote := range []bool{false, true} {
		name := "cancel"
		if promote {
			name = "promote"
		}
		t.Run(name, func(t *testing.T) {
			cfg := DefaultContinuousBatcherConfig()
			cfg.MaxSlots = 1
			cfg.KVBytesPerToken = 1
			cfg.MaxKVCacheBytes = 1
			if promote {
				cfg.MaxKVCacheBytes = 0
			}
			cb, err := NewContinuousBatcher(cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = cb.Close() })
			requests := []*SubagentRequest{
				{SessionID: "first", PromptTokens: []int{1}, TargetTokens: 1},
				{SessionID: "second", PromptTokens: []int{2}, TargetTokens: 1},
				{SessionID: "third", PromptTokens: []int{3}, TargetTokens: 1},
			}
			for _, req := range requests {
				if _, err := cb.Submit(req); err != nil {
					t.Fatal(err)
				}
			}
			assertQueue := func(want ...*SubagentRequest) {
				t.Helper()
				got := cb.WaitingQueue()
				if len(got) != len(want) {
					t.Fatalf("queue length = %d, want %d", len(got), len(want))
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("queue[%d] changed FIFO request identity", i)
					}
				}
			}
			cancel := func(id string) {
				t.Helper()
				if err := cb.Cancel(id); err != nil {
					t.Fatal(err)
				}
			}
			// Keep a view of the actual queue allocation, not a copied snapshot.
			// No scheduler goroutine runs: Submit and Cancel are the real owners.
			// This checks retired references, not GC timing or process RSS.
			backing := cb.waitingQueue
			if promote {
				assertQueue(requests[1], requests[2])
				for i := 0; i < 2; i++ {
					cancel(requests[i].SessionID)
					if backing[i] != nil {
						t.Fatal("promoted request remains referenced by retired queue head")
					}
					slot, ok := cb.GetSlot(requests[i+1].SessionID)
					if !ok || slot.Request != requests[i+1] || slot.State != SlotStateActiveDecode {
						t.Fatal("promotion lost active request ownership")
					}
					assertQueue(requests[i+2:]...)
				}
				return
			}
			assertQueue(requests...)
			cancel("second")
			assertQueue(requests[0], requests[2])
			if backing[2] != nil {
				t.Fatal("retired cancellation queue tail still holds a request")
			}
			cancel("third")
			assertQueue(requests[0])
			if backing[1] != nil {
				t.Fatal("canceled tail request remains referenced")
			}
			cancel("first")
			assertQueue()
			if backing[0] != nil {
				t.Fatal("canceled sole request remains referenced")
			}
		})
	}
}
