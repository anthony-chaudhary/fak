package compute

import (
	"reflect"
	"testing"
)

// ---- Exact batch resource-hazard lowering (#12218) ------------------------------
//
// The batch recorder fences every recorded dispatch against its predecessor with one
// global compute->compute memory barrier because it has no per-dispatch resource-hazard
// input (vulkan_shim.cpp recordComputeBarrier). These tests bind the exact RAW/WAR/WAW
// lowering that lets two adjacent dispatches whose ranges are disjoint skip that
// barrier, and prove the fail-closed contract: incomplete metadata never yields a
// partial plan.

func TestVulkanBatchResourceHazards(t *testing.T) {
	t.Run("disjoint_ranges_elide_barrier", func(t *testing.T) {
		// Two dispatches touching non-overlapping ranges of the SAME buffer, adjacent in
		// the shim op sequence (ordinals 0 then 1): no dependency, so the barrier between
		// them is provably elidable.
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
			}},
		})
		if !plan.Exact {
			t.Fatalf("plan not exact: %s", plan.Reason)
		}
		if plan.Elided != 1 || plan.Synchronized != 0 {
			t.Fatalf("elided=%d synchronized=%d, want 1/0", plan.Elided, plan.Synchronized)
		}
		if plan.Edges[0].NeedsSync || plan.Edges[0].Kind != VulkanHazardNone {
			t.Fatalf("edge=%+v, want no-sync none", plan.Edges[0])
		}
		if plan.BarrierFloor() != VulkanHazardMinBarriers(2) {
			t.Fatalf("BarrierFloor=%d, want %d", plan.BarrierFloor(), VulkanHazardMinBarriers(2))
		}
	})

	t.Run("different_buffers_elide_barrier", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 101, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
		})
		if !plan.Exact || plan.Elided != 1 {
			t.Fatalf("exact=%v elided=%d reason=%s", plan.Exact, plan.Elided, plan.Reason)
		}
	})

	t.Run("interleaved_undeclared_op_fails_closed", func(t *testing.T) {
		// Two disjoint matmuls whose shim ordinals are NOT adjacent: an undeclared op sits
		// between them. Even though the two declared ranges are disjoint, the shim fences
		// the successor against the interleaved op, so the verdict must stay sync.
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 5, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
			}},
		})
		if plan.Elided != 0 || plan.Synchronized != 1 {
			t.Fatalf("elided=%d sync=%d, want 0/1 (interleaved op must keep the fence)", plan.Elided, plan.Synchronized)
		}
		if plan.Edges[0].NeedsSync || plan.Edges[0].Kind != VulkanHazardUnknown {
			// NeedsSync is set; Kind must be Unknown, not a fabricated benign class.
			if !plan.Edges[0].NeedsSync {
				t.Fatalf("interleaved edge must need sync: %+v", plan.Edges[0])
			}
		}
	})

	t.Run("overlapping_raw_keeps_barrier", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 128, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 64, Size: 128, Mode: VulkanAccessRead},
			}},
		})
		if !plan.Exact || plan.Synchronized != 1 || plan.Elided != 0 {
			t.Fatalf("exact=%v sync=%d elided=%d", plan.Exact, plan.Synchronized, plan.Elided)
		}
		if plan.Edges[0].Kind != VulkanHazardReadAfterWrite || !plan.Edges[0].NeedsSync {
			t.Fatalf("edge=%+v, want RAW needsSync", plan.Edges[0])
		}
	})

	t.Run("war_and_waw_classified", func(t *testing.T) {
		war := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 7, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 7, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
		})
		if war.Edges[0].Kind != VulkanHazardWriteAfterRead || !war.Edges[0].NeedsSync {
			t.Fatalf("WAR edge=%+v", war.Edges[0])
		}
		waw := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 8, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 8, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
		})
		if waw.Edges[0].Kind != VulkanHazardWriteAfterWrite || !waw.Edges[0].NeedsSync {
			t.Fatalf("WAW edge=%+v", waw.Edges[0])
		}
	})

	t.Run("read_after_read_is_benign", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 9, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 9, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
		})
		if plan.Edges[0].Kind != VulkanHazardReadAfterRead || plan.Edges[0].NeedsSync {
			t.Fatalf("RAR edge=%+v, want benign", plan.Edges[0])
		}
		if plan.Elided != 1 {
			t.Fatalf("elided=%d, want 1", plan.Elided)
		}
	})

	t.Run("undeclared_dispatch_fails_closed", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, Declared: false},
		})
		if plan.Exact {
			t.Fatal("plan claimed exact over an undeclared dispatch")
		}
		if plan.Reason == "" {
			t.Fatal("non-exact plan must state a reason")
		}
		if len(plan.Edges) != 0 {
			t.Fatalf("non-exact plan must not emit partial edges, got %d", len(plan.Edges))
		}
	})
}

// TestVulkanBatchHazardLoweringArmed pins the default-off, typo-safe toggle.
func TestVulkanBatchHazardLoweringArmed(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"", false},
		{"0", false},
		{"off", false},
		{"coarse", false},
		{"exact", true},
		{" EXACT ", true},
		{"1", true},
		{"on", true},
		{"true", true},
	}
	for _, tc := range cases {
		env := func(string) string { return tc.val }
		if got := VulkanHazardLoweringArmed(env); got != tc.want {
			t.Fatalf("VulkanHazardLoweringArmed(%q)=%v, want %v", tc.val, got, tc.want)
		}
	}
	if VulkanHazardLoweringArmed(nil) {
		t.Fatal("nil env must not arm the path")
	}
}

// TestVulkanBatchHazardPlanDeterministic proves two identical inputs lower to byte-identical
// edge order, so a receipt over the plan is comparable across runs.
func TestVulkanBatchHazardPlanDeterministic(t *testing.T) {
	dispatches := []VulkanDispatchAccess{
		{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
			{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
		}},
		{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
			{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessRead},
		}},
		{NodeID: 2, ShimOrdinal: 2, Declared: true, Accesses: []VulkanBufferAccess{
			{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessWrite},
		}},
	}
	a := LowerVulkanBatchHazards(dispatches)
	b := LowerVulkanBatchHazards(dispatches)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("plans differ:\n%+v\n%+v", a, b)
	}
	if a.Synchronized != 1 || a.Elided != 1 {
		t.Fatalf("sync=%d elided=%d, want 1/1 (edges: %+v)", a.Synchronized, a.Elided, a.Edges)
	}
}

// TestVulkanBatchHazardStatsDeltas pins the receipt arithmetic and its non-negative clamp.
func TestVulkanBatchHazardStatsDeltas(t *testing.T) {
	stats := VulkanBatchHazardStats{
		Windows:        3,
		ExactWindows:   2,
		CoarseFallback: 1,
		BarriersCoarse: 30,
		BarriersExact:  18,
	}
	avoided, fellBack := stats.Deltas()
	if avoided != 12 || fellBack != 1 {
		t.Fatalf("avoided=%d fellBack=%d, want 12/1", avoided, fellBack)
	}
	// A pathological exact>coarse must clamp to zero, never a negative savings claim.
	neg := VulkanBatchHazardStats{BarriersCoarse: 5, BarriersExact: 9}
	if avoided, _ := neg.Deltas(); avoided != 0 {
		t.Fatalf("avoided=%d, want 0 (clamped)", avoided)
	}
	if got := VulkanHazardMinBarriers(0); got != 0 {
		t.Fatalf("VulkanHazardMinBarriers(0)=%d, want 0", got)
	}
	if got := VulkanHazardMinBarriers(5); got != 4 {
		t.Fatalf("VulkanHazardMinBarriers(5)=%d, want 4", got)
	}
}

// TestVulkanBatchHazardSortStable pins the receipt ordering as byte-stable.
func TestVulkanBatchHazardSortStable(t *testing.T) {
	plan := VulkanHazardPlan{Exact: true, Edges: []VulkanHazardEdge{
		{Prev: 2, Next: 3, BufferID: 9, Kind: VulkanHazardReadAfterWrite, NeedsSync: true},
		{Prev: 0, Next: 1, BufferID: 5, Kind: VulkanHazardNone},
		{Prev: 0, Next: 1, BufferID: 4, Kind: VulkanHazardNone},
	}}
	got := plan.SortEdgesByDependency()
	want := []int64{4, 5, 9}
	for i, e := range got {
		if e.BufferID != want[i] {
			t.Fatalf("edge %d buffer=%d, want %d (order %+v)", i, e.BufferID, want[i], got)
		}
	}
}

// TestVulkanBatchWindowReachableFromRecorder binds the production wiring: a window
// opened by the same begin/record/lower calls vulkan.go BeginBatch/recordBatchDispatch/
// FlushBatch make lowers to a plan and accumulates a bounded receipt. This is the
// invokability witness the presence≠invokability rule requires — the ledger is not
// merely declared, it is exercised through its production entrypoints.
func TestVulkanBatchWindowReachableFromRecorder(t *testing.T) {
	var ledger vulkanBatchHazardLedger
	var stats vulkanBatchHazardStats

	// Window 1: two disjoint dispatches on one buffer => one barrier elided.
	ledger.begin()
	ledger.record(VulkanDispatchAccess{NodeID: 1, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
	}})
	ledger.record(VulkanDispatchAccess{NodeID: 2, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
	}})
	plan, window := ledger.lower(true)
	if !plan.Exact || plan.Elided != 1 {
		t.Fatalf("window1 exact=%v elided=%d reason=%s", plan.Exact, plan.Elided, plan.Reason)
	}
	stats.observe(window)

	// Window 2: an undeclared dispatch forces the coarse fallback for the whole window.
	ledger.begin()
	ledger.record(VulkanDispatchAccess{NodeID: 1, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 7, Size: 0, Mode: VulkanAccessWrite},
	}})
	ledger.record(VulkanDispatchAccess{NodeID: 2, ShimOrdinal: 1})
	plan2, window2 := ledger.lower(true)
	if plan2.Exact {
		t.Fatal("window2 claimed exact over an undeclared dispatch")
	}
	if window2.CoarseFallback != 1 || window2.BarriersExact != window2.BarriersCoarse {
		t.Fatalf("window2 fallback=%d exact=%d coarse=%d", window2.CoarseFallback, window2.BarriersExact, window2.BarriersCoarse)
	}
	stats.observe(window2)

	// A record outside any window must be silently dropped, never accumulate.
	ledger.record(VulkanDispatchAccess{NodeID: 9, ShimOrdinal: 0, Declared: true})
	if n := len(ledger.dispatches); n != 0 {
		t.Fatalf("closed-window record leaked %d declaration(s)", n)
	}

	snap := stats.snapshot()
	if snap.Windows != 2 || snap.ExactWindows != 1 || snap.CoarseFallback != 1 {
		t.Fatalf("snapshot=%+v, want 2/1/1", snap)
	}
	avoided, fellBack := snap.Deltas()
	if avoided != 1 || fellBack != 1 {
		t.Fatalf("avoided=%d fellBack=%d, want 1/1", avoided, fellBack)
	}
}

// TestVulkanBatchWindowUnarmedKeepsCoarseFloor proves the default-off path reports the
// coarse barrier count so a receipt never claims an elision the device did not perform.
func TestVulkanBatchWindowUnarmedKeepsCoarseFloor(t *testing.T) {
	var ledger vulkanBatchHazardLedger
	ledger.begin()
	ledger.record(VulkanDispatchAccess{NodeID: 1, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
	}})
	ledger.record(VulkanDispatchAccess{NodeID: 2, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
	}})
	plan, window := ledger.lower(false)
	if !plan.Exact {
		t.Fatalf("plan not exact: %s", plan.Reason)
	}
	if window.BarriersExact != window.BarriersCoarse {
		t.Fatalf("unarmed window exact=%d coarse=%d, want equal", window.BarriersExact, window.BarriersCoarse)
	}
	if avoided, _ := window.Deltas(); avoided != 0 {
		t.Fatalf("unarmed window claimed avoided=%d, want 0", avoided)
	}
}
