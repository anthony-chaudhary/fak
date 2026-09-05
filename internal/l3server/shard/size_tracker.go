package shard

import (
	"fmt"
	"log"
	"sort"
	"sync/atomic"
)

// valueSizeTracker records value sizes from SET operations and detects the
// dominant value size. LLM workloads typically have 1-3 distinct value sizes,
// so a simple map is sufficient.
//
// Internal fields are written only by the shard goroutine (single-writer).
// External goroutines read the cached snapshot via atomic pointer.
type valueSizeTracker struct {
	counts          map[uint64]int64 // exact value_size -> count
	totalSets       int64
	warmupTarget    int64 // 0 = disabled
	detected        bool
	optimalSize     uint64  // detected dominant value size
	slotUtilization float64 // slot utilization with current slab classes (1.0 = perfect fit)
	shardID         int
	justDetected    bool // set in detect(), cleared after rebuild
	frozen          bool // set after successful auto-rebuild, prevents further recording

	verbose          bool // emit per-shard log lines
	onWarmupComplete func(shardID int, dominantSize uint64, freqPercent float64)

	cachedSnap atomic.Pointer[DetectionSnapshot] // safe for concurrent reads
}

// SizeEntry is used for sorting sizes by frequency.
type SizeEntry struct {
	Size    uint64
	Count   int64
	Percent float64
}

// DetectionSnapshot holds the result of size detection for Stats API exposure.
type DetectionSnapshot struct {
	Status                 string      `json:"status"` // "disabled", "warming_up", "detected"
	WarmupProgress         string      `json:"warmup_progress"`
	DominantValueSize      uint64      `json:"dominant_value_size"`
	DominantFreqPercent    float64     `json:"dominant_frequency_percent"`
	CurrentSlotUtilization float64     `json:"current_slot_utilization"`
	RecommendedPageBytes   uint64      `json:"recommended_model_page_bytes"`
	TopSizes               []SizeEntry `json:"top_sizes"`
}

func newValueSizeTracker(warmupTarget int64, shardID int, verbose bool, onComplete func(int, uint64, float64)) *valueSizeTracker {
	t := &valueSizeTracker{
		counts:           make(map[uint64]int64),
		warmupTarget:     warmupTarget,
		shardID:          shardID,
		verbose:          verbose,
		onWarmupComplete: onComplete,
	}
	t.updateCachedSnapshot()
	return t
}

// record adds a value size observation. Called on every SET.
func (t *valueSizeTracker) record(valueSize uint64) {
	if t.warmupTarget <= 0 || t.frozen {
		return // disabled or already rebuilt
	}
	t.counts[valueSize]++
	t.totalSets++

	if !t.detected && t.totalSets >= t.warmupTarget {
		t.detect()
	}
	t.updateCachedSnapshot()
}

// recordBatch records multiple value sizes in one pass. Triggers detection at
// most once and updates the cached snapshot once at the end.
func (t *valueSizeTracker) recordBatch(sizes []uint64) {
	if t.warmupTarget <= 0 || t.frozen || len(sizes) == 0 {
		return
	}
	for _, sz := range sizes {
		t.counts[sz]++
	}
	t.totalSets += int64(len(sizes))

	if !t.detected && t.totalSets >= t.warmupTarget {
		t.detect()
	}
	t.updateCachedSnapshot()
}

// detect finds the dominant value size.
func (t *valueSizeTracker) detect() {
	if len(t.counts) == 0 {
		return
	}

	top := t.topSizes(3)
	if len(top) == 0 {
		return
	}

	dominant := top[0]
	t.optimalSize = dominant.Size
	t.detected = true
	t.justDetected = true

	if t.verbose {
		log.Printf("[l3server] shard %d: auto-detect: dominant value size %d bytes (%.1f%% of SETs)",
			t.shardID, dominant.Size, dominant.Percent)
	}
	if t.onWarmupComplete != nil {
		t.onWarmupComplete(t.shardID, dominant.Size, dominant.Percent)
	}
}

// setSlotUtilization is called after detect() with the utilization computed by the slab allocator.
func (t *valueSizeTracker) setSlotUtilization(util float64) {
	t.slotUtilization = util
	if t.verbose {
		if util < 0.98 {
			log.Printf("[l3server] shard %d: auto-detect: slot utilization %.1f%% for dominant size %d â€” recommend model_page_bytes=%d",
				t.shardID, util*100, t.optimalSize, t.optimalSize)
		} else {
			log.Printf("[l3server] shard %d: auto-detect: slot utilization %.1f%% â€” good fit",
				t.shardID, util*100)
		}
	}
	t.updateCachedSnapshot()
}

// topSizes returns the top-N sizes by frequency.
func (t *valueSizeTracker) topSizes(n int) []SizeEntry {
	entries := make([]SizeEntry, 0, len(t.counts))
	for size, count := range t.counts {
		entries = append(entries, SizeEntry{
			Size:    size,
			Count:   count,
			Percent: float64(count) / float64(t.totalSets) * 100,
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Count > entries[j].Count
	})
	if len(entries) > n {
		entries = entries[:n]
	}
	return entries
}

// reset clears all state so detection can re-run after an auto-tune.
func (t *valueSizeTracker) reset() {
	t.counts = make(map[uint64]int64)
	t.totalSets = 0
	t.detected = false
	t.optimalSize = 0
	t.slotUtilization = 0
	t.justDetected = false
	t.frozen = false
	t.updateCachedSnapshot()
}

// snapshot returns detection state for the Stats API.
// Safe to call from any goroutine â€” reads from atomic cache.
func (t *valueSizeTracker) snapshot() DetectionSnapshot {
	if p := t.cachedSnap.Load(); p != nil {
		return *p
	}
	return DetectionSnapshot{Status: "disabled"}
}

// updateCachedSnapshot rebuilds the snapshot from internal state and stores it atomically.
// Must be called from the shard goroutine (the sole writer).
func (t *valueSizeTracker) updateCachedSnapshot() {
	var snap DetectionSnapshot

	if t.warmupTarget <= 0 {
		snap.Status = "disabled"
		t.cachedSnap.Store(&snap)
		return
	}

	snap.WarmupProgress = formatProgress(t.totalSets, t.warmupTarget)
	snap.TopSizes = t.topSizes(5)

	switch {
	case t.frozen:
		snap.Status = "rebuilt"
		snap.DominantValueSize = t.optimalSize
		snap.CurrentSlotUtilization = t.slotUtilization * 100
		snap.RecommendedPageBytes = t.optimalSize
		if len(snap.TopSizes) > 0 {
			snap.DominantFreqPercent = snap.TopSizes[0].Percent
		}
	case t.detected:
		snap.Status = "detected"
		snap.DominantValueSize = t.optimalSize
		snap.CurrentSlotUtilization = t.slotUtilization * 100
		snap.RecommendedPageBytes = t.optimalSize
		if len(snap.TopSizes) > 0 {
			snap.DominantFreqPercent = snap.TopSizes[0].Percent
		}
	default:
		snap.Status = "warming_up"
	}

	t.cachedSnap.Store(&snap)
}

func formatProgress(current, target int64) string {
	if current > target {
		current = target
	}
	return fmt.Sprintf("%d/%d", current, target)
}
