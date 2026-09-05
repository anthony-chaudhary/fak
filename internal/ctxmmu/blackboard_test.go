package ctxmmu

import (
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestBlackboard_BasicPublishSubscribeLookup(t *testing.T) {
	bb := NewBlackboard()

	payload := []byte("subagent test payload")
	ref := &abi.Ref{
		Kind:   abi.RefInline,
		Inline: payload,
		Len:    int64(len(payload)),
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}

	meta := map[string]string{"subagent": "worker-1", "task": "analysis"}
	id, err := bb.Publish("subagent:analysis", ref, 1, meta)
	if err != nil {
		t.Fatalf("Publish failed: %v", err)
	}
	if id == "" {
		t.Fatal("Publish returned empty id")
	}

	// Lookup by ID
	entry, ok := bb.Lookup(id)
	if !ok || entry == nil {
		t.Fatalf("Lookup(%q) failed", id)
	}
	if entry.Topic != "subagent:analysis" {
		t.Errorf("expected topic %q, got %q", "subagent:analysis", entry.Topic)
	}
	if entry.Epoch != 1 {
		t.Errorf("expected epoch 1, got %d", entry.Epoch)
	}
	if string(entry.Ref.Inline) != string(payload) {
		t.Errorf("expected payload %q, got %q", payload, entry.Ref.Inline)
	}
	if entry.Metadata["subagent"] != "worker-1" {
		t.Errorf("expected metadata subagent=worker-1, got %q", entry.Metadata["subagent"])
	}

	// Subscribe by topic
	entries := bb.Subscribe("subagent:analysis")
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry from Subscribe, got %d", len(entries))
	}
	if entries[0].ID != id {
		t.Errorf("expected entry ID %s, got %s", id, entries[0].ID)
	}

	// Non-existent lookups
	if _, found := bb.Lookup("non-existent"); found {
		t.Error("expected Lookup on non-existent id to return false")
	}
	if entries := bb.Subscribe("non-existent-topic"); len(entries) != 0 {
		t.Errorf("expected empty subscribe for non-existent topic, got %d", len(entries))
	}

	// Invalid inputs
	if _, err := bb.Publish("", ref, 1, nil); err == nil {
		t.Error("expected error publishing with empty topic")
	}
	if _, err := bb.Publish("topic", nil, 1, nil); err == nil {
		t.Error("expected error publishing with nil ref")
	}
}

func TestBlackboard_ReferenceCounting(t *testing.T) {
	bb := NewBlackboard()

	payload := []byte("ref-count payload")
	ref := &abi.Ref{
		Kind:   abi.RefInline,
		Inline: payload,
		Len:    int64(len(payload)),
	}

	id, err := bb.Publish("topic:ref", ref, 1, nil)
	if err != nil {
		t.Fatalf("Publish failed: %v", err)
	}

	stats := bb.Stats()
	if stats.ActiveRefs != 1 {
		t.Fatalf("expected ActiveRefs=1, got %d", stats.ActiveRefs)
	}
	if stats.EntryCount != 1 {
		t.Fatalf("expected EntryCount=1, got %d", stats.EntryCount)
	}

	// Retain increments refcount
	if err := bb.Retain(id); err != nil {
		t.Fatalf("Retain failed: %v", err)
	}
	stats = bb.Stats()
	if stats.ActiveRefs != 2 {
		t.Fatalf("expected ActiveRefs=2 after Retain, got %d", stats.ActiveRefs)
	}

	entry, ok := bb.Lookup(id)
	if !ok || entry.RefCount != 2 {
		t.Fatalf("expected entry refcount 2, got %d", entry.RefCount)
	}

	// Release decrements refcount
	if err := bb.Release(id); err != nil {
		t.Fatalf("Release failed: %v", err)
	}
	stats = bb.Stats()
	if stats.ActiveRefs != 1 {
		t.Fatalf("expected ActiveRefs=1 after first Release, got %d", stats.ActiveRefs)
	}

	// Entry still exists
	if _, ok := bb.Lookup(id); !ok {
		t.Fatal("entry should still exist when RefCount == 1")
	}

	// Second Release drops refcount to 0, reaping the entry
	if err := bb.Release(id); err != nil {
		t.Fatalf("second Release failed: %v", err)
	}
	stats = bb.Stats()
	if stats.ActiveRefs != 0 {
		t.Fatalf("expected ActiveRefs=0, got %d", stats.ActiveRefs)
	}
	if stats.EntryCount != 0 {
		t.Fatalf("expected EntryCount=0, got %d", stats.EntryCount)
	}
	if stats.MemoryBytes != 0 {
		t.Fatalf("expected MemoryBytes=0, got %d", stats.MemoryBytes)
	}

	// Entry is gone
	if _, ok := bb.Lookup(id); ok {
		t.Fatal("entry should be reaped when RefCount == 0")
	}

	// Additional Release / Retain on reaped entry must fail
	if err := bb.Release(id); err == nil {
		t.Fatal("expected error on Release of already-reaped entry")
	}
	if err := bb.Retain(id); err == nil {
		t.Fatal("expected error on Retain of already-reaped entry")
	}
}

func TestBlackboard_EpochInvalidation(t *testing.T) {
	bb := NewBlackboard()

	ref := &abi.Ref{Kind: abi.RefInline, Inline: []byte("epoch data"), Len: 10}

	// Publish 3 entries in epoch 1, 2 in epoch 2, 1 in epoch 3
	var epoch1IDs []string
	for i := 0; i < 3; i++ {
		id, err := bb.Publish("topic:epoch", ref, 1, nil)
		if err != nil {
			t.Fatalf("Publish epoch 1 entry %d failed: %v", i, err)
		}
		epoch1IDs = append(epoch1IDs, id)
	}
	for i := 0; i < 2; i++ {
		_, err := bb.Publish("topic:epoch", ref, 2, nil)
		if err != nil {
			t.Fatalf("Publish epoch 2 entry %d failed: %v", i, err)
		}
	}
	_, err := bb.Publish("topic:epoch", ref, 3, nil)
	if err != nil {
		t.Fatalf("Publish epoch 3 entry failed: %v", err)
	}

	stats := bb.Stats()
	if stats.EntryCount != 6 {
		t.Fatalf("expected 6 entries, got %d", stats.EntryCount)
	}
	if stats.ActiveRefs != 6 {
		t.Fatalf("expected 6 active refs, got %d", stats.ActiveRefs)
	}

	// Invalidate epoch 1
	reaped := bb.InvalidateEpoch(1)
	if reaped != 3 {
		t.Fatalf("expected 3 entries reaped for epoch 1, got %d", reaped)
	}

	stats = bb.Stats()
	if stats.EntryCount != 3 {
		t.Fatalf("expected 3 remaining entries, got %d", stats.EntryCount)
	}
	if stats.ActiveRefs != 3 {
		t.Fatalf("expected 3 active refs, got %d", stats.ActiveRefs)
	}

	// Verify epoch 1 entries are gone
	for _, id := range epoch1IDs {
		if _, ok := bb.Lookup(id); ok {
			t.Fatalf("entry %s from epoch 1 should have been reaped", id)
		}
	}

	// Invalidate epoch 2
	reaped = bb.InvalidateEpoch(2)
	if reaped != 2 {
		t.Fatalf("expected 2 entries reaped for epoch 2, got %d", reaped)
	}
	if bb.Stats().EntryCount != 1 {
		t.Fatalf("expected 1 remaining entry, got %d", bb.Stats().EntryCount)
	}

	// Invalidate non-existent epoch
	reaped = bb.InvalidateEpoch(999)
	if reaped != 0 {
		t.Fatalf("expected 0 entries reaped for non-existent epoch, got %d", reaped)
	}
}

func TestBlackboard_Stats(t *testing.T) {
	bb := NewBlackboard()

	r1 := &abi.Ref{Kind: abi.RefInline, Inline: []byte("12345"), Len: 5}
	r2 := &abi.Ref{Kind: abi.RefInline, Inline: []byte("1234567890"), Len: 10}

	id1, _ := bb.Publish("topic:stats", r1, 1, nil)
	id2, _ := bb.Publish("topic:stats", r2, 1, nil)

	stats := bb.Stats()
	if stats.EntryCount != 2 {
		t.Errorf("expected EntryCount=2, got %d", stats.EntryCount)
	}
	if stats.MemoryBytes != 15 {
		t.Errorf("expected MemoryBytes=15, got %d", stats.MemoryBytes)
	}
	if stats.ActiveRefs != 2 {
		t.Errorf("expected ActiveRefs=2, got %d", stats.ActiveRefs)
	}

	// Read via Lookup -> +1 zeroCopyReads
	_, _ = bb.Lookup(id1)
	_, _ = bb.Lookup(id2)
	if bb.Stats().ZeroCopyReads != 2 {
		t.Errorf("expected ZeroCopyReads=2 after 2 Lookups, got %d", bb.Stats().ZeroCopyReads)
	}

	// Read via Subscribe -> +2 zeroCopyReads (2 entries returned)
	_ = bb.Subscribe("topic:stats")
	if bb.Stats().ZeroCopyReads != 4 {
		t.Errorf("expected ZeroCopyReads=4 after Subscribe, got %d", bb.Stats().ZeroCopyReads)
	}
}

func TestBlackboard_Concurrency(t *testing.T) {
	bb := NewBlackboard()
	const numGoroutines = 30
	const numOpsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(workerID int) {
			defer wg.Done()
			topic := fmt.Sprintf("topic:worker-%d", workerID%5)
			payload := []byte(fmt.Sprintf("worker-%d-data", workerID))
			ref := &abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))}

			for i := 0; i < numOpsPerGoroutine; i++ {
				epoch := uint64(workerID % 3)
				id, err := bb.Publish(topic, ref, epoch, map[string]string{"worker": fmt.Sprintf("%d", workerID)})
				if err != nil {
					t.Errorf("concurrent publish error: %v", err)
					return
				}

				if err := bb.Retain(id); err != nil {
					t.Errorf("concurrent retain error: %v", err)
					return
				}

				entries := bb.Subscribe(topic)
				if len(entries) == 0 {
					t.Errorf("concurrent subscribe expected entries for topic %s", topic)
					return
				}

				if _, ok := bb.Lookup(id); !ok {
					t.Errorf("concurrent lookup failed for %s", id)
					return
				}

				_ = bb.Release(id)
				_ = bb.Release(id)
			}
		}(g)
	}

	wg.Wait()
}
