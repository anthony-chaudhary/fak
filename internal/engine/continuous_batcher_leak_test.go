package engine

import (
	"context"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=25ms lane=default
func TestContinuousBatcherCompletedMapBounded(t *testing.T) {
	t.Parallel()

	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 32
	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher: %v", err)
	}
	t.Cleanup(func() { _ = cb.Close() })

	const total = defaultMaxCompletedSlots + 104
	ctx := context.Background()

	for start := 0; start < total; start += cfg.MaxSlots {
		end := start + cfg.MaxSlots
		if end > total {
			end = total
		}
		for i := start; i < end; i++ {
			id := fmt.Sprintf("req-%d", i)
			if _, err := cb.Submit(&SubagentRequest{
				SessionID:    id,
				PromptTokens: []int{1, 2},
				TargetTokens: 1,
			}); err != nil {
				t.Fatalf("Submit(%s): %v", id, err)
			}
		}
		if _, err := cb.Step(ctx); err != nil {
			t.Fatalf("Step(%d): %v", start, err)
		}
	}

	cb.mu.Lock()
	gotLen := len(cb.completedMap)
	cb.mu.Unlock()
	if gotLen != defaultMaxCompletedSlots {
		t.Fatalf("len(completedMap) = %d, want %d", gotLen, defaultMaxCompletedSlots)
	}

	evictedCount := total - defaultMaxCompletedSlots
	for i := 0; i < evictedCount; i++ {
		id := fmt.Sprintf("req-%d", i)
		if _, ok := cb.GetSlot(id); ok {
			t.Fatalf("expected oldest completed slot %s to be evicted", id)
		}
	}
	for i := evictedCount; i < total; i++ {
		id := fmt.Sprintf("req-%d", i)
		if _, ok := cb.GetSlot(id); !ok {
			t.Fatalf("expected recent completed slot %s to remain retained", id)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestContinuousBatcherCompletedMapBoundedOnCancelAndReAdmit(t *testing.T) {
	t.Parallel()

	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 4
	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher: %v", err)
	}
	t.Cleanup(func() { _ = cb.Close() })

	cb.mu.Lock()
	cb.maxCompletedSlots = 4
	cb.mu.Unlock()

	ctx := context.Background()
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("c-%d", i)
		if _, err := cb.Submit(&SubagentRequest{
			SessionID:    id,
			PromptTokens: []int{1},
			TargetTokens: 2,
		}); err != nil {
			t.Fatalf("Submit(%s): %v", id, err)
		}
		if i%2 == 0 {
			if err := cb.Cancel(id); err != nil {
				t.Fatalf("Cancel(%s): %v", id, err)
			}
		} else {
			if _, err := cb.Step(ctx); err != nil {
				t.Fatalf("Step 1(%s): %v", id, err)
			}
			if _, err := cb.Step(ctx); err != nil {
				t.Fatalf("Step 2(%s): %v", id, err)
			}
		}
	}

	cb.mu.Lock()
	gotLen := len(cb.completedMap)
	orderLen := len(cb.completedOrder)
	cb.mu.Unlock()
	if gotLen != 4 {
		t.Fatalf("len(completedMap) = %d, want 4", gotLen)
	}
	if orderLen > 8 {
		t.Fatalf("completedOrder grew unbounded: %d", orderLen)
	}
	if _, ok := cb.GetSlot("c-0"); ok {
		t.Fatalf("expected c-0 to be evicted")
	}
	if _, ok := cb.GetSlot("c-1"); ok {
		t.Fatalf("expected c-1 to be evicted")
	}
	for i := 2; i < 6; i++ {
		id := fmt.Sprintf("c-%d", i)
		if _, ok := cb.GetSlot(id); !ok {
			t.Fatalf("expected %s to remain retained", id)
		}
	}
}
