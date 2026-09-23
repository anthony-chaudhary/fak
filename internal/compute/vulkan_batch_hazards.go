package compute

import (
	"fmt"
	"sort"
	"strings"
)

// vulkan_batch_hazards.go lowers declared graph resource accesses into exact
// synchronization2-style buffer hazards, so the batched recorder can elide the
// coarse global compute->compute barrier between two adjacent dispatches whose read
// and write ranges are disjoint (fak#12218).
//
// The existing batch recorder (vulkan_shim.cpp recordComputeBarrier) fences EVERY
// recorded op against its predecessor with one full memory barrier because it has no
// per-dispatch resource-hazard input. This file supplies that input as pure data so
// the decision is testable on any host without a GPU, and is byte-for-byte
// deterministic. Like vulkan_plan.go and vulkan_graph.go, it is always compiled.
//
// Fail-closed contract: callers MUST hand this lowering complete access metadata. If
// any dispatch in the window is missing a declaration, or the aggregation overflows
// the representable range, HazardPlan.Exact is false and the caller must emit the
// conservative global barrier instead. An exact plan is never inferred from partial
// input.

// VulkanAccessMode classifies one dispatch's access to a buffer range.
type VulkanAccessMode uint8

const (
	// VulkanAccessNone means the dispatch does not touch the range (no hazard).
	VulkanAccessNone VulkanAccessMode = iota
	// VulkanAccessRead is a shader read (Uniform or Storage read).
	VulkanAccessRead
	// VulkanAccessWrite is a shader write.
	VulkanAccessWrite
	// VulkanAccessReadWrite is a read-modify-write (append, in-place accumulate).
	VulkanAccessReadWrite
)

// VulkanBufferAccess declares one dispatch's access to a byte range of a buffer.
// BufferID is the same opaque handle used by VulkanGraphNode (BufferID/Inputs/Outputs).
// Size == 0 means "entire buffer" (VK_WHOLE_SIZE), which conflicts with any access on
// the same buffer.
type VulkanBufferAccess struct {
	BufferID int64
	Offset   int64
	Size     int64
	Mode     VulkanAccessMode
}

// VulkanDispatchAccess is the complete resource-access declaration for one recorded
// dispatch in a batch window. Declared must be true for the access set to be trusted;
// a dispatch that cannot state its exact ranges must be marked undeclared so the
// lowering fails closed. ShimOrdinal is the recorder's own op index for this dispatch
// (0 when unset); it binds a lowered verdict to the exact recorded op it describes.
type VulkanDispatchAccess struct {
	NodeID      int
	ShimOrdinal int
	Declared    bool
	Accesses    []VulkanBufferAccess
}

// VulkanHazardKind names the exact dependency between two dispatches.
type VulkanHazardKind string

const (
	// VulkanHazardNone means the two dispatches are fully disjoint and need no barrier.
	VulkanHazardNone VulkanHazardKind = "none"
	// VulkanHazardReadAfterWrite is a RAW dependency (producer wrote, consumer reads).
	VulkanHazardReadAfterWrite VulkanHazardKind = "RAW"
	// VulkanHazardWriteAfterRead is a WAR dependency (producer read, consumer writes).
	VulkanHazardWriteAfterRead VulkanHazardKind = "WAR"
	// VulkanHazardWriteAfterWrite is a WAW dependency (both write the same range).
	VulkanHazardWriteAfterWrite VulkanHazardKind = "WAW"
	// VulkanHazardReadAfterRead is a benign read sharing (no barrier required).
	VulkanHazardReadAfterRead VulkanHazardKind = "RAR"
	// VulkanHazardUnknown is returned when either side is undeclared; fail closed.
	VulkanHazardUnknown VulkanHazardKind = "unknown"
)

// VulkanHazardEdge records the exact dependency between a producer and its successor.
type VulkanHazardEdge struct {
	Prev      int
	Next      int
	Kind      VulkanHazardKind
	BufferID  int64
	NeedsSync bool
	// NextShimOrdinal is the successor dispatch's recorder op index, so a device-side
	// consumer can bind this edge's verdict to the exact recorded op it fences.
	NextShimOrdinal int
}

// VulkanHazardPlan is the deterministic lowering result for one batch window.
type VulkanHazardPlan struct {
	// Exact is true only when every dispatch declared complete access metadata and the
	// plan was fully computed. When false the caller must keep the coarse global
	// barrier for the whole window (fail closed).
	Exact bool
	// Reason explains a non-exact plan (empty when Exact).
	Reason string
	// Edges holds one entry per adjacent (prev,next) pair, in recording order.
	Edges []VulkanHazardEdge
	// Synchronized is the number of adjacent pairs that require a barrier.
	Synchronized int
	// Elided is the number of adjacent pairs that are provably disjoint.
	Elided int
}

// BarrierFloor is the whole-window barrier count the coarse path would emit: one per
// adjacent pair. The exact plan can only reduce it.
func (p VulkanHazardPlan) BarrierFloor() int {
	return len(p.Edges)
}

// VulkanHazardMinBarriers returns the number of barriers the coarse recorder would emit
// for n recorded ops (one fence before every op after the first).
func VulkanHazardMinBarriers(n int) int {
	if n <= 1 {
		return 0
	}
	return n - 1
}

// accessModesConflict reports whether a producer access `prev` and a successor access
// `next` describe a dependency requiring synchronization, and which kind. Disjoint
// ranges on the same buffer never conflict; different buffers never conflict.
func accessModesConflict(prev, next VulkanBufferAccess) (VulkanHazardKind, bool) {
	if prev.BufferID != next.BufferID {
		return VulkanHazardNone, false
	}
	if !accessRangesOverlap(prev, next) {
		return VulkanHazardNone, false
	}
	switch {
	case prev.Mode == VulkanAccessRead && next.Mode == VulkanAccessRead:
		return VulkanHazardReadAfterRead, false
	case prev.Mode == VulkanAccessWrite && next.Mode == VulkanAccessRead:
		return VulkanHazardReadAfterWrite, true
	case prev.Mode == VulkanAccessWrite && next.Mode == VulkanAccessWrite:
		return VulkanHazardWriteAfterWrite, true
	case prev.Mode == VulkanAccessWrite && next.Mode == VulkanAccessReadWrite:
		return VulkanHazardReadAfterWrite, true
	case prev.Mode == VulkanAccessReadWrite && next.Mode == VulkanAccessRead:
		return VulkanHazardReadAfterWrite, true
	case prev.Mode == VulkanAccessReadWrite && next.Mode == VulkanAccessWrite:
		return VulkanHazardWriteAfterWrite, true
	case prev.Mode == VulkanAccessReadWrite && next.Mode == VulkanAccessReadWrite:
		return VulkanHazardWriteAfterWrite, true
	case prev.Mode == VulkanAccessRead && next.Mode == VulkanAccessWrite:
		return VulkanHazardWriteAfterRead, true
	default:
		return VulkanHazardNone, false
	}
}

// accessRangesOverlap reports whether two declared accesses on the same buffer overlap.
// A Size of 0 means the entire buffer and therefore overlaps anything on that buffer.
func accessRangesOverlap(a, b VulkanBufferAccess) bool {
	if a.Size == 0 || b.Size == 0 {
		return true
	}
	if a.Size < 0 || b.Size < 0 || a.Offset < 0 || b.Offset < 0 {
		// Negative geometry is not a claim of disjointness; fail closed.
		return true
	}
	aEnd := a.Offset + a.Size
	bEnd := b.Offset + b.Size
	return a.Offset < bEnd && b.Offset < aEnd
}

// LowerVulkanBatchHazards computes the exact hazard plan for a batch window in
// recording order. It returns Exact=false (never a partial plan) when any dispatch is
// undeclared, so a caller can fail closed to the coarse barrier.
func LowerVulkanBatchHazards(dispatches []VulkanDispatchAccess) VulkanHazardPlan {
	for _, d := range dispatches {
		if !d.Declared {
			return VulkanHazardPlan{
				Exact:  false,
				Reason: fmt.Sprintf("dispatch node %d has no complete access declaration", d.NodeID),
			}
		}
	}
	plan := VulkanHazardPlan{Exact: true}
	for i := 1; i < len(dispatches); i++ {
		prev, next := dispatches[i-1], dispatches[i]
		edge := VulkanHazardEdge{Prev: prev.NodeID, Next: next.NodeID, Kind: VulkanHazardNone, NextShimOrdinal: next.ShimOrdinal}
		for _, pa := range prev.Accesses {
			for _, na := range next.Accesses {
				kind, needs := accessModesConflict(pa, na)
				if kind == VulkanHazardNone {
					continue
				}
				edge.Kind = kind
				edge.BufferID = pa.BufferID
				edge.NeedsSync = edge.NeedsSync || needs
			}
		}
		// The shim fences each recorded op against its IMMEDIATE predecessor. A verdict may
		// only license an elision when these two declarations are adjacent in the shim's own
		// op sequence; an interleaved undeclared dispatch (RMSNorm/RoPE/attention/…) would
		// otherwise be silently skipped. Fail closed to sync when adjacency is unproven.
		if !edge.NeedsSync && prev.ShimOrdinal+1 != next.ShimOrdinal {
			edge.NeedsSync = true
			edge.Kind = VulkanHazardUnknown
		}
		if edge.NeedsSync {
			plan.Synchronized++
		} else {
			plan.Elided++
		}
		plan.Edges = append(plan.Edges, edge)
	}
	return plan
}

// VulkanHazardLoweringArmed reports whether the operator has opted into exact
// synchronization2 buffer-hazard lowering. Default-off: unset leaves the coarse global
// barrier in place (P3 preserved). A nonempty value other than "exact"/"1"/"on"
// explicitly disables it, so a typo cannot silently arm the path.
func VulkanHazardLoweringArmed(env func(string) string) bool {
	if env == nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(env("FAK_VULKAN_BATCH_HAZARDS"))) {
	case "exact", "1", "on", "true":
		return true
	default:
		return false
	}
}

// VulkanBatchHazardStats is a bounded, order-independent summary for receipts.
type VulkanBatchHazardStats struct {
	Windows        int
	ExactWindows   int
	CoarseFallback int
	BarriersCoarse int
	BarriersExact  int
}

// Deltas returns (barriers avoided, windows that failed closed to coarse).
func (s VulkanBatchHazardStats) Deltas() (avoided, fellBack int) {
	avoided = s.BarriersCoarse - s.BarriersExact
	if avoided < 0 {
		avoided = 0
	}
	return avoided, s.CoarseFallback
}

// SortEdgesByDependency returns the plan edges ordered by (Prev, Next, BufferID) so a
// receipt over a plan is byte-stable regardless of traversal order.
func (p VulkanHazardPlan) SortEdgesByDependency() []VulkanHazardEdge {
	out := make([]VulkanHazardEdge, len(p.Edges))
	copy(out, p.Edges)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Prev != out[j].Prev {
			return out[i].Prev < out[j].Prev
		}
		if out[i].Next != out[j].Next {
			return out[i].Next < out[j].Next
		}
		return out[i].BufferID < out[j].BufferID
	})
	return out
}
