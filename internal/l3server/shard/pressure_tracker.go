package shard

import (
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
)

// PressureSnapshot is an immutable snapshot of per-class pressure counters.
// Published via atomic.Pointer for lock-free external reads.
type PressureSnapshot struct {
	ClassSizes     []uint64  // size of each class
	Evictions      []int64   // per-class eviction count in window
	AllocOps       []int64   // per-class Alloc() attempts in window
	AllocFails     []int64   // per-class Alloc() failures in window
	PromotionsInto []int64   // promotions INTO this class (from smaller classes)
	PromotionsFrom []int64   // failed allocs FROM this class (promoted elsewhere)
	WindowStart    time.Time // when this window began
	WindowDuration time.Duration
}

// classPressureTracker accumulates per-class eviction and allocation counters
// within a vacuum evaluation window. Written only by the shard goroutine;
// external readers use the atomic snapshot pointer.
type classPressureTracker struct {
	classEvictions      []int64 // per-class eviction count in window
	classAllocOps       []int64 // per-class Alloc() attempts in window
	classAllocFails     []int64 // per-class Alloc() failures in window
	classPromotionsInto []int64 // promotions INTO this class (from smaller)
	classPromotionsFrom []int64 // failed allocs FROM this class (promoted elsewhere)
	numClasses          int
	windowStart         time.Time
	snap                atomic.Pointer[PressureSnapshot] // for external readers
}

// newClassPressureTracker creates a tracker for the given number of classes.
func newClassPressureTracker(numClasses int) *classPressureTracker {
	t := &classPressureTracker{
		classEvictions:      make([]int64, numClasses),
		classAllocOps:       make([]int64, numClasses),
		classAllocFails:     make([]int64, numClasses),
		classPromotionsInto: make([]int64, numClasses),
		classPromotionsFrom: make([]int64, numClasses),
		numClasses:          numClasses,
		windowStart:         time.Now(),
	}
	return t
}

// recordEviction increments the eviction counter for classIdx.
// Called from shard goroutine only (no synchronization needed).
func (t *classPressureTracker) recordEviction(classIdx int) {
	if classIdx >= 0 && classIdx < t.numClasses {
		t.classEvictions[classIdx]++
	}
}

// recordAllocOp increments the allocation attempt counter for classIdx.
func (t *classPressureTracker) recordAllocOp(classIdx int) {
	if classIdx >= 0 && classIdx < t.numClasses {
		t.classAllocOps[classIdx]++
	}
}

// recordAllocFailure increments the allocation failure counter for classIdx.
func (t *classPressureTracker) recordAllocFailure(classIdx int) {
	if classIdx >= 0 && classIdx < t.numClasses {
		t.classAllocFails[classIdx]++
	}
}

// recordPromotion records that srcClass overflowed and the allocation was promoted to dstClass.
func (t *classPressureTracker) recordPromotion(srcClass, dstClass int) {
	if srcClass >= 0 && srcClass < t.numClasses {
		t.classPromotionsFrom[srcClass]++
	}
	if dstClass >= 0 && dstClass < t.numClasses {
		t.classPromotionsInto[dstClass]++
	}
}

// snapshot builds and publishes a PressureSnapshot from current counters
// and the provided class utilizations. Returns the snapshot.
func (t *classPressureTracker) snapshot(a alloc.Allocator) *PressureSnapshot {
	now := time.Now()
	snap := &PressureSnapshot{
		ClassSizes:     make([]uint64, t.numClasses),
		Evictions:      make([]int64, t.numClasses),
		AllocOps:       make([]int64, t.numClasses),
		AllocFails:     make([]int64, t.numClasses),
		PromotionsInto: make([]int64, t.numClasses),
		PromotionsFrom: make([]int64, t.numClasses),
		WindowStart:    t.windowStart,
		WindowDuration: now.Sub(t.windowStart),
	}
	for i := 0; i < t.numClasses; i++ {
		if i < a.NumClasses() {
			snap.ClassSizes[i] = a.ClassSize(i)
		}
		snap.Evictions[i] = t.classEvictions[i]
		snap.AllocOps[i] = t.classAllocOps[i]
		snap.AllocFails[i] = t.classAllocFails[i]
		snap.PromotionsInto[i] = t.classPromotionsInto[i]
		snap.PromotionsFrom[i] = t.classPromotionsFrom[i]
	}
	t.snap.Store(snap)
	return snap
}

// reset zeroes all counters and starts a new window.
// Called when the allocator is rebuilt or flushed.
func (t *classPressureTracker) reset(numClasses int) {
	t.numClasses = numClasses
	t.classEvictions = make([]int64, numClasses)
	t.classAllocOps = make([]int64, numClasses)
	t.classAllocFails = make([]int64, numClasses)
	t.classPromotionsInto = make([]int64, numClasses)
	t.classPromotionsFrom = make([]int64, numClasses)
	t.windowStart = time.Now()
}

// loadSnapshot returns the most recently published snapshot, or nil.
func (t *classPressureTracker) loadSnapshot() *PressureSnapshot {
	return t.snap.Load()
}
