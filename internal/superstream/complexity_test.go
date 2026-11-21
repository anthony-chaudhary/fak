package superstream

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// TestDecideStepCopiesQueueOnce pins the complexity contract that makes many
// workstreams navigable: one decision over a queue of Q items copies that queue
// at most once.
//
// The measurement is allocation bytes (runtime.MemStats.TotalAlloc delta), not
// wall time, so it is deterministic and cheap on every host and Go version.
// Before 2026-11-21 a step copied the queue twice on the normal path
// (DecideStep plus EvaluateContextSafety) and thrice at a terminal boundary
// (BuildCarryoverSeed re-normalized an already-normalized spec); this test fails
// on that shape at ~2x the per-item bound.
func TestDecideStepCopiesQueueOnce(t *testing.T) {
	const items = 2000
	const runs = 20

	queue := make([]WorkItem, items)
	for i := range queue {
		queue[i] = WorkItem{
			ID:     fmt.Sprintf("item-%d", i),
			Lane:   fmt.Sprintf("lane-%d", i),
			Status: ItemPending,
		}
	}
	spec := StreamSpec{
		ID:              "copy-budget",
		Intent:          "one-normalized-copy",
		MaxTurnsTotal:   1 << 20,
		MaxTurnsPerItem: 8,
		// Complexity gate: deliberately beyond the default MaxItems cap so
		// the per-step cost of a huge queue is still measurable.
		MaxItems: 1 << 20,
		Queue:    queue,
	}
	state := StreamState{StreamID: spec.ID, Queue: queue}
	tax := laneadmit.Taxonomy{Loaded: true}

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < runs; i++ {
		if dec := DecideStep(spec, state, "copy-budget-holder", nil, tax); dec.Action != ActionAcquireLease {
			t.Fatalf("unexpected action: %s", dec.Action)
		}
	}
	runtime.ReadMemStats(&after)

	itemSize := int64(reflect.TypeOf(WorkItem{}).Size())
	bytesPerOp := int64(after.TotalAlloc-before.TotalAlloc) / runs
	// 1.5 copies: one is the contract, the headroom absorbs allocator rounding
	// and the fixed per-call overhead. Two copies (~2x) must fail.
	bound := itemSize * 3 / 2 * items
	if bytesPerOp > bound {
		t.Fatalf("DecideStep copies the queue more than once: %d B/op > %d B/op bound (%d B/item; one WorkItem is %d B)",
			bytesPerOp, bound, bytesPerOp/items, itemSize)
	}
}
