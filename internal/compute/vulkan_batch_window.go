package compute

import (
	"math"
	"sync"
)

// vulkan_batch_window.go owns the live batch-window hazard ledger consulted by the
// cgo Vulkan dispatcher. It is always compiled (no cgo, no GPU) so the recording and
// lowering decisions are unit-testable on any host.
//
// Reachability: vulkan.go's BeginBatch/FlushBatch bracket a batch window and
// recordBatchDispatch (called from the MatMul/BatchedMatMul/MatMulAddInPlace seams)
// installs each dispatch's verdict before native recording. Only the already declared
// prefix may authorize an elision; an unknown suffix cannot undo a safe earlier cut.
// Invalid metadata, missing ordinals and exhausted budgets retain the coarse barrier.
// FlushBatch collects a receipt, never changes already recorded barriers. This is the production
// consumer of the pure lowering in vulkan_batch_hazards.go (#12218).

// vulkanBatchHazardLedger accumulates declared accesses for one batch window.
type vulkanBatchHazardLedger struct {
	mu         sync.Mutex
	dispatches []VulkanDispatchAccess
	recording  bool
	exhausted  bool
}

// begin opens a window and clears the previous declaration set.
func (l *vulkanBatchHazardLedger) begin() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.dispatches = l.dispatches[:0]
	l.recording = true
	l.exhausted = false
}

// recordVerdict bounds prefix lowering and retains every earlier access until the
// lowering proves a global barrier. Adjacent-only comparisons lose W(A), W(B), R(A).
// Returning true is always safe, including outside a window or after admission fails.
func (l *vulkanBatchHazardLedger) recordVerdict(d VulkanDispatchAccess) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.recording || l.exhausted {
		return true
	}
	const maxDeclarations = 256
	if len(l.dispatches) >= maxDeclarations {
		l.exhausted = true
		l.dispatches = append(l.dispatches, VulkanDispatchAccess{NodeID: d.NodeID})
		return true
	}
	if len(d.Accesses) > 16 {
		// Refuse before scanning or cloning caller-controlled storage. Keep the
		// undeclared sentinel so later prefixes cannot forget this unknown work.
		d.Declared = false
		d.Accesses = nil
	}
	if d.ShimOrdinal < 0 || len(d.Accesses) == 0 {
		d.Declared = false
	}
	for _, a := range d.Accesses {
		if a.BufferID == 0 || a.Offset < 0 || a.Size < 0 || a.Offset > math.MaxInt64-a.Size ||
			a.Mode < VulkanAccessRead || a.Mode > VulkanAccessReadWrite {
			d.Declared = false
		}
	}
	if len(l.dispatches) > 0 && d.ShimOrdinal <= l.dispatches[len(l.dispatches)-1].ShimOrdinal {
		// A skipped native dispatch must not leave a declaration usable as a producer.
		d.Declared = false
	}
	d.Accesses = append([]VulkanBufferAccess(nil), d.Accesses...)
	l.dispatches = append(l.dispatches, d)
	plan := LowerVulkanBatchHazards(l.dispatches)
	return !plan.Exact || len(plan.Edges) == 0 || plan.Edges[len(plan.Edges)-1].NeedsSync
}

// record appends one dispatch declaration to the open window. A record outside an open
// window is dropped: the coarse path never depended on it, so this stays fail-closed.
func (l *vulkanBatchHazardLedger) record(d VulkanDispatchAccess) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.recording {
		return
	}
	l.dispatches = append(l.dispatches, d)
}

// close ends the window and returns a copy of the declarations for lowering.
func (l *vulkanBatchHazardLedger) close() []VulkanDispatchAccess {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.recording = false
	out := make([]VulkanDispatchAccess, len(l.dispatches))
	copy(out, l.dispatches)
	l.dispatches = l.dispatches[:0]
	return out
}

// lower folds the window into a plan plus the receipt-facing stats delta.
func (l *vulkanBatchHazardLedger) lower(armed bool) (VulkanHazardPlan, VulkanBatchHazardStats) {
	dispatches := l.close()
	stats := VulkanBatchHazardStats{}
	if len(dispatches) == 0 {
		return VulkanHazardPlan{Exact: true}, stats
	}
	stats.Windows = 1
	plan := LowerVulkanBatchHazards(dispatches)
	stats.BarriersCoarse = VulkanHazardMinBarriers(len(dispatches))
	if plan.Exact {
		stats.ExactWindows = 1
		stats.BarriersExact = plan.Synchronized
		if !armed {
			// Not armed: report the coarse floor as the effective barrier count so a
			// receipt never claims an elision the device did not perform.
			stats.BarriersExact = stats.BarriersCoarse
		}
	} else {
		stats.CoarseFallback = 1
		stats.BarriersExact = stats.BarriersCoarse
	}
	return plan, stats
}

// vulkanBatchHazardStats is the process-wide accumulator surfaced to receipts.
type vulkanBatchHazardStats struct {
	mu    sync.Mutex
	stats VulkanBatchHazardStats
}

func (s *vulkanBatchHazardStats) observe(window VulkanBatchHazardStats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats.Windows += window.Windows
	s.stats.ExactWindows += window.ExactWindows
	s.stats.CoarseFallback += window.CoarseFallback
	s.stats.BarriersCoarse += window.BarriersCoarse
	s.stats.BarriersExact += window.BarriersExact
}

func (s *vulkanBatchHazardStats) snapshot() VulkanBatchHazardStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stats
}

var (
	vulkanBatchHazardLedgerOnce sync.Once
	vulkanBatchLedgerGlobal     vulkanBatchHazardLedger
	vulkanBatchStatsGlobal      vulkanBatchHazardStats
)

func vulkanBatchLedger() *vulkanBatchHazardLedger {
	vulkanBatchHazardLedgerOnce.Do(func() {})
	return &vulkanBatchLedgerGlobal
}

// VulkanBatchHazardStatsSnapshot returns the accumulated exact-barrier lowering receipt
// for this process. It is bounded and order-independent, so it is safe to embed in a
// benchmark or serve receipt.
func VulkanBatchHazardStatsSnapshot() VulkanBatchHazardStats {
	return vulkanBatchStatsGlobal.snapshot()
}
