package macrostate

import (
	"testing"
	"time"
)

// Invariant: Macrostate event stores must support deterministic replay and atomic state compaction.
// Guard: Apply refuses corrupted events or writes to retired stores.

func TestMacroStateLifecycle(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()
	s := &Store{}
	receipt, err := s.Apply(Event{
		Schema:     Schema,
		ID:         "ev-1",
		At:         now,
		Kind:       Promote,
		Key:        "test-key",
		Value:      "test-val",
		Provenance: "operator:test",
	})
	if err != nil || !receipt.Applied {
		t.Fatalf("failed applying lifecycle event: %v (receipt: %+v)", err, receipt)
	}

	compacted := s.Compact(now)
	if compacted["test-key"] != "test-val" {
		t.Fatalf("expected compacted state to contain test-key, got: %v", compacted)
	}
}
