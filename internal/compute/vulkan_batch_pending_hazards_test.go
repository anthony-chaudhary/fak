package compute

import "testing"

// These are planned barriers only. The default-off native activation and its
// recording-time verdict delivery are separate contracts, not witnessed here.
// fak-test:runtime fast est=20ms lane=default
func TestVulkanBatchPendingHazards(t *testing.T) {
	t.Parallel()
	a := func(id int64, mode VulkanAccessMode) VulkanBufferAccess {
		return VulkanBufferAccess{BufferID: id, Size: 4, Mode: mode}
	}
	r, w, rw := VulkanAccessRead, VulkanAccessWrite, VulkanAccessReadWrite
	for _, tc := range []struct {
		name string
		ops  [][]VulkanBufferAccess
		want []bool
		last VulkanHazardKind
	}{
		{"nonadjacent RAW", [][]VulkanBufferAccess{{a(1, w)}, {a(2, w)}, {a(1, r)}}, []bool{false, true}, VulkanHazardReadAfterWrite},
		{"nonadjacent WAR", [][]VulkanBufferAccess{{a(1, r)}, {a(2, w)}, {a(1, w)}}, []bool{false, true}, VulkanHazardWriteAfterRead},
		{"nonadjacent WAW", [][]VulkanBufferAccess{{a(1, w)}, {a(2, w)}, {a(1, w)}}, []bool{false, true}, VulkanHazardWriteAfterWrite},
		{"nonadjacent read to read-write", [][]VulkanBufferAccess{{a(1, r)}, {a(2, w)}, {a(1, rw)}}, []bool{false, true}, VulkanHazardWriteAfterRead},
		{"adjacent read to read-write", [][]VulkanBufferAccess{{a(1, r)}, {a(1, rw)}}, []bool{true}, VulkanHazardWriteAfterRead},
		{"disjoint buffers", [][]VulkanBufferAccess{{a(1, w)}, {a(2, w)}, {a(3, r)}}, []bool{false, false}, VulkanHazardNone},
		{"multiple outstanding dispatches", [][]VulkanBufferAccess{{a(1, w)}, {a(2, w)}, {a(3, r)}, {a(1, r)}}, []bool{false, false, true}, VulkanHazardReadAfterWrite},
		{"multiple accesses in producer", [][]VulkanBufferAccess{{a(1, w), a(2, r)}, {a(3, w)}, {a(2, w)}}, []bool{false, true}, VulkanHazardWriteAfterRead},
		{"barrier retires earlier write", [][]VulkanBufferAccess{{a(1, w)}, {a(1, r)}, {a(2, r)}, {a(1, r)}}, []bool{true, false, false}, VulkanHazardReadAfterRead},
		{"barrier retains successor write", [][]VulkanBufferAccess{{a(1, r)}, {a(1, w)}, {a(2, r)}, {a(1, r)}}, []bool{true, false, true}, VulkanHazardReadAfterWrite},
		{"disjoint range retains earlier range", [][]VulkanBufferAccess{{a(1, w)}, {{BufferID: 1, Offset: 4, Size: 4, Mode: w}}, {a(1, r)}}, []bool{false, true}, VulkanHazardReadAfterWrite},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dispatches := make([]VulkanDispatchAccess, len(tc.ops))
			for i, accesses := range tc.ops {
				dispatches[i] = VulkanDispatchAccess{NodeID: i, ShimOrdinal: i, Declared: true, Accesses: accesses}
			}
			plan := LowerVulkanBatchHazards(dispatches)
			if !plan.Exact || len(plan.Edges) != len(tc.want) {
				t.Fatalf("unexpected plan framing: %+v", plan)
			}
			syncs := 0
			for i, want := range tc.want {
				if want {
					syncs++
				}
				edge := plan.Edges[i]
				if edge.NeedsSync != want || edge.Prev != i || edge.Next != i+1 || edge.NextShimOrdinal != i+1 {
					t.Errorf("edge %d=%+v, want sync=%t at its original ordinal", i, edge, want)
				}
			}
			if plan.Synchronized != syncs || plan.Elided != len(tc.want)-syncs || plan.Edges[len(tc.want)-1].Kind != tc.last {
				t.Fatalf("counts or final classification changed: %+v", plan)
			}
		})
	}

	// Independent finite-state oracle: two single-word resources, each carrying
	// pending read/write bits. Exhaust all 6^4 four-dispatch mode/resource chains.
	// A global barrier retires both resources before adding its successor.
	for sequence := 0; sequence < 6*6*6*6; sequence++ {
		n := sequence
		dispatches := make([]VulkanDispatchAccess, 4)
		resources, modes := [4]int{}, [4]uint8{}
		for i := range dispatches {
			choice := n % 6
			n /= 6
			resources[i], modes[i] = choice/3, uint8(choice%3+1)
			dispatches[i] = VulkanDispatchAccess{NodeID: i, ShimOrdinal: i, Declared: true, Accesses: []VulkanBufferAccess{a(int64(resources[i]+1), VulkanAccessMode(modes[i]))}}
		}
		plan := LowerVulkanBatchHazards(dispatches)
		if !plan.Exact || len(plan.Edges) != 3 {
			t.Fatalf("sequence %d framing: %+v", sequence, plan)
		}
		var pending [2]uint8
		pending[resources[0]] = modes[0]
		for i := 1; i < len(dispatches); i++ {
			prior, next := pending[resources[i]], modes[i]
			want := prior != 0 && (prior&2 != 0 || next&2 != 0)
			if plan.Edges[i-1].NeedsSync != want {
				t.Fatalf("sequence %d boundary %d sync=%t want %t pending=%v next=%d/%d", sequence, i, plan.Edges[i-1].NeedsSync, want, pending, resources[i], next)
			}
			if want {
				pending = [2]uint8{}
			}
			pending[resources[i]] |= next
		}
	}
}

// The bounded planner never publishes a partially proved window. These fixtures
// use existing APIs, including the ledger that supplies native fallback stats.
// fak-test:runtime fast est=50ms lane=default
func TestVulkanBatchPendingHazardBudget(t *testing.T) {
	t.Parallel()
	reads := func(first int64, n int) []VulkanBufferAccess {
		out := make([]VulkanBufferAccess, n)
		for i := range out {
			out[i] = VulkanBufferAccess{BufferID: first + int64(i), Size: 4, Mode: VulkanAccessRead}
		}
		return out
	}
	window := func(accesses ...[]VulkanBufferAccess) []VulkanDispatchAccess {
		out := make([]VulkanDispatchAccess, len(accesses))
		for i := range out {
			out[i] = VulkanDispatchAccess{NodeID: i, ShimOrdinal: i, Declared: true, Accesses: accesses[i]}
		}
		return out
	}
	// One pending-dispatch visit plus 255*257 comparisons consumes exactly 65536.
	boundary := window(reads(1, 255), reads(1000, 257))
	plan := LowerVulkanBatchHazards(boundary)
	if !plan.Exact || plan.Elided != 1 || len(plan.Edges) != 1 {
		t.Fatalf("exact budget boundary refused: %+v", plan)
	}
	// Exhaustion occurs after the first edge has already been computed.
	over := append(append([]VulkanDispatchAccess(nil), boundary...), VulkanDispatchAccess{
		NodeID: 2, ShimOrdinal: 2, Declared: true, Accesses: reads(2000, 1),
	})
	// A synchronization costs two units, then a fresh pending search needs the
	// entire budget. Resetting the allowance at that barrier would miss exhaustion.
	barrier := window([]VulkanBufferAccess{{BufferID: 1, Size: 4, Mode: VulkanAccessWrite}}, reads(1, 255), reads(1000, 257))
	// Empty producers still cost a visit. Empty successors themselves need only
	// the linear dispatch/edge traversal, avoiding an uncounted quadratic scan.
	sparse := make([]VulkanDispatchAccess, 65538)
	for i := range sparse {
		sparse[i] = VulkanDispatchAccess{NodeID: i, ShimOrdinal: i, Declared: true}
	}
	sparse[len(sparse)-1].Accesses = reads(1, 1)
	for _, tc := range []struct {
		name string
		ops  []VulkanDispatchAccess
	}{{"after produced edge", over}, {"barrier does not reset budget", barrier}, {"empty producer visits", sparse}} {
		t.Run(tc.name, func(t *testing.T) {
			for _, armed := range []bool{false, true} {
				ledger := vulkanBatchHazardLedger{dispatches: tc.ops, recording: true}
				plan, stats := ledger.lower(armed)
				if plan.Exact || plan.Reason != "batch hazard search budget exhausted" || len(plan.Edges) != 0 || plan.Synchronized != 0 || plan.Elided != 0 {
					t.Fatalf("exhausted plan (armed=%t): exact=%t reason=%q edges=%d sync=%d elided=%d", armed, plan.Exact, plan.Reason, len(plan.Edges), plan.Synchronized, plan.Elided)
				}
				if stats.Windows != 1 || stats.CoarseFallback != 1 || stats.ExactWindows != 0 || stats.BarriersExact != len(tc.ops)-1 || stats.BarriersCoarse != len(tc.ops)-1 {
					t.Fatalf("exhausted window did not retain coarse receipt (armed=%t): %+v", armed, stats)
				}
			}
		})
	}
}
