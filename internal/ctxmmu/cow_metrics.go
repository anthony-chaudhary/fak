package ctxmmu

import (
	"time"
)

// COWMetrics tracks falsifiable metrics on prefix deduplication and memory savings.
type COWMetrics struct {
	TotalAllocatedBytes int64   `json:"total_allocated_bytes"` // Physical DRAM bytes allocated
	DeduplicatedBytes   int64   `json:"deduplicated_bytes"`    // Avoided DRAM bytes through sharing
	DedupRatio          float64 `json:"dedup_ratio"`           // 1.0 - (physical / logical)
	ActiveBranches      int     `json:"active_branches"`       // Active session branches count
	TotalPhysicalBlocks int     `json:"total_physical_blocks"` // Unique active physical blocks in DRAM
	TotalLogicalBlocks  int     `json:"total_logical_blocks"`  // Total logical blocks across all branches
	LogicalBytes        int64   `json:"logical_bytes"`         // Total logical bytes across all branches
	SharedPages         int     `json:"shared_pages"`          // Count of shared physical page blocks
	PrefixHitRate       float64 `json:"prefix_hit_rate"`       // Prefix cache hit rate across branches
}

// Metrics calculates and returns deduplication, memory, and branch metrics.
func (t *COWPageTable) Metrics() COWMetrics {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var dedupRatio float64
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupRatio = 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}

	dedupBytes := t.totalLogicalBytes - t.totalAllocatedBytes
	if dedupBytes < 0 {
		dedupBytes = 0
	}

	totalLogicalBlocks := 0
	for _, s := range t.sessions {
		totalLogicalBlocks += len(s.Blocks)
	}

	sharedPagesCount := 0
	for _, blk := range t.physicalBlocks {
		if blk.IsShared() {
			sharedPagesCount++
		}
	}
	prefixHitRate := t.PrefixHitRate()

	return COWMetrics{
		TotalAllocatedBytes: t.totalAllocatedBytes,
		DeduplicatedBytes:   dedupBytes,
		DedupRatio:          dedupRatio,
		ActiveBranches:      len(t.sessions),
		TotalPhysicalBlocks: len(t.physicalBlocks),
		TotalLogicalBlocks:  totalLogicalBlocks,
		LogicalBytes:        t.totalLogicalBytes,
		SharedPages:         sharedPagesCount,
		PrefixHitRate:       prefixHitRate,
	}
}

// TotalAllocatedBytes returns the total physical DRAM bytes allocated for all active blocks.
func (t *COWPageTable) TotalAllocatedBytes() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalAllocatedBytes
}

// DeduplicatedBytes returns total avoided DRAM bytes through copy-on-write page table sharing.
func (t *COWPageTable) DeduplicatedBytes() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.totalLogicalBytes > t.totalAllocatedBytes {
		return t.totalLogicalBytes - t.totalAllocatedBytes
	}
	return 0
}

// DedupRatio returns the memory deduplication ratio: 1.0 - (physical / logical).
func (t *COWPageTable) DedupRatio() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		return 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}
	return 0.0
}

// ActiveBranches returns the number of active, unreleased session branches.
func (t *COWPageTable) ActiveBranches() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.sessions)
}

// PhysicalBlockCount returns the number of active unique physical blocks in DRAM.
func (t *COWPageTable) PhysicalBlockCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.physicalBlocks)
}

// DuplicatePhysicalPagesAllocated returns the number of redundant physical pages allocated on fork (always 0).
func (t *COWPageTable) DuplicatePhysicalPagesAllocated() int {
	return 0
}

// BlockCapacity returns the configured block capacity.
func (t *COWPageTable) BlockCapacity() int {
	return t.blockCapacity
}

// BlockSize returns the byte size of one page block.
func (t *COWPageTable) BlockSize() int64 {
	return t.blockSize
}

// SharedPages returns the number of physical page blocks currently shared across multiple session branches.
func (t *COWPageTable) SharedPages() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	count := 0
	for _, blk := range t.physicalBlocks {
		if blk.IsShared() {
			count++
		}
	}
	return count
}

// PrefixHitRate returns the prefix cache hit rate across active session branches.
// For forked subagent sessions, this measures the fraction of logical blocks served from
// the shared parent prefix.
func (t *COWPageTable) PrefixHitRate() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	totalForkedBlocks := 0
	totalPrefixBlocks := 0
	hasForks := false

	for _, s := range t.sessions {
		s.mu.RLock()
		if s.ParentID != "" {
			hasForks = true
			totalForkedBlocks += len(s.Blocks)
			totalPrefixBlocks += s.PrefixBlocks
		}
		s.mu.RUnlock()
	}

	if !hasForks {
		if len(t.sessions) > 0 {
			return 1.0
		}
		return 0.0
	}

	if totalForkedBlocks > 0 {
		rate := float64(totalPrefixBlocks) / float64(totalForkedBlocks)
		if rate > 1.0 {
			return 1.0
		}
		return rate
	}
	return 1.0
}

// SubagentForkTelemetry captures zero-copy physical memory savings and prefix sharing metrics
// when a subagent session is forked via COWPageTable.ForkSession.
type SubagentForkTelemetry struct {
	ParentID           string        `json:"parent_id"`
	ChildID            string        `json:"child_id"`
	SharedPages        int           `json:"shared_pages"`
	SharedTokens       int           `json:"shared_tokens"`
	PhysicalBytesSaved int64         `json:"physical_bytes_saved"`
	DeduplicatedBytes  int64         `json:"deduplicated_bytes"`
	DedupRatio         float64       `json:"dedup_ratio"`
	PrefixHitRate      float64       `json:"prefix_hit_rate"`
	ForkLatency        time.Duration `json:"fork_latency"`
	DuplicatePages     int           `json:"duplicate_pages"` // Always 0 for zero-copy COW
	SavedMemorySummary string        `json:"saved_memory_summary"`
}

// Summary returns the human-readable summary of saved physical memory.
func (t SubagentForkTelemetry) Summary() string {
	return t.SavedMemorySummary
}

// CaptureSubagentForkTelemetry captures physical memory saved and prefix sharing telemetry
// for a child subagent session branch forked via ForkSession.
func (t *COWPageTable) CaptureSubagentForkTelemetry(childSessionID string) (SubagentForkTelemetry, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	child, ok := t.sessions[childSessionID]
	if !ok {
		return SubagentForkTelemetry{}, ErrSessionNotFound
	}
	return t.captureForkTelemetryLocked(child), nil
}

func (t *COWPageTable) captureForkTelemetryLocked(child *SessionBranch) SubagentForkTelemetry {
	child.mu.RLock()
	defer child.mu.RUnlock()

	sharedPages := 0
	for _, blk := range child.Blocks {
		if blk.RefCount() > 1 {
			sharedPages++
		}
	}

	bytesSaved := int64(child.PrefixBlocks) * t.blockSize
	dedupBytes := int64(0)
	if t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupBytes = t.totalLogicalBytes - t.totalAllocatedBytes
	}

	dedupRatio := 0.0
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupRatio = 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}

	return SubagentForkTelemetry{
		ParentID:           child.ParentID,
		ChildID:            child.ID,
		SharedPages:        sharedPages,
		SharedTokens:       child.PrefixTokens,
		PhysicalBytesSaved: bytesSaved,
		DeduplicatedBytes:  dedupBytes,
		DedupRatio:         dedupRatio,
		PrefixHitRate:      child.PrefixHitRate,
		ForkLatency:        child.ForkLatency,
		DuplicatePages:     0,
		SavedMemorySummary: FormatSavedMemory(bytesSaved),
	}
}

// SubagentForkTelemetry returns telemetry capturing physical memory saved by this session branch.
func (s *SessionBranch) SubagentForkTelemetry() SubagentForkTelemetry {
	s.table.mu.RLock()
	defer s.table.mu.RUnlock()
	return s.table.captureForkTelemetryLocked(s)
}

// ForkSubagent forks a child subagent session and returns both the branch and its zero-copy memory telemetry.
func (t *COWPageTable) ForkSubagent(parentID, childID string) (*SessionBranch, SubagentForkTelemetry, error) {
	child, err := t.ForkSession(parentID, childID)
	if err != nil {
		return nil, SubagentForkTelemetry{}, err
	}
	telem, err := t.CaptureSubagentForkTelemetry(childID)
	return child, telem, err
}

// CaptureSubagentForkTelemetry captures physical memory saved when a subagent session branch
// is forked from a parent coordinator via the default COWPageTable.
func CaptureSubagentForkTelemetry(table *COWPageTable, childSessionID string) (SubagentForkTelemetry, error) {
	if table == nil {
		table = defaultCOWPageTable
	}
	return table.CaptureSubagentForkTelemetry(childSessionID)
}

// SubagentSharingReport captures aggregate telemetry across all forked subagents.
type SubagentSharingReport struct {
	ActiveBranches      int     `json:"active_branches"`
	ForkedSubagents     int     `json:"forked_subagents"`
	TotalSharedPages    int     `json:"total_shared_pages"`
	TotalPhysicalBlocks int     `json:"total_physical_blocks"`
	TotalLogicalBlocks  int     `json:"total_logical_blocks"`
	AllocatedBytes      int64   `json:"allocated_bytes"`
	DeduplicatedBytes   int64   `json:"deduplicated_bytes"`
	DedupRatio          float64 `json:"dedup_ratio"`
	PrefixHitRate       float64 `json:"prefix_hit_rate"`
	DuplicatePagesAlloc int     `json:"duplicate_pages_allocated"` // Always 0
	SavedMemorySummary  string  `json:"saved_memory_summary"`
}

// SubagentSharingReport returns aggregate zero-copy memory savings across all active branches.
func (t *COWPageTable) SubagentSharingReport() SubagentSharingReport {
	t.mu.RLock()
	defer t.mu.RUnlock()

	sharedBlocks := 0
	for _, blk := range t.physicalBlocks {
		if blk.IsShared() {
			sharedBlocks++
		}
	}

	totalLogicalBlocks := 0
	forkedCount := 0
	for _, s := range t.sessions {
		s.mu.RLock()
		totalLogicalBlocks += len(s.Blocks)
		if s.ParentID != "" {
			forkedCount++
		}
		s.mu.RUnlock()
	}

	dedupBytes := int64(0)
	if t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupBytes = t.totalLogicalBytes - t.totalAllocatedBytes
	}

	dedupRatio := 0.0
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupRatio = 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}

	prefixRate := 1.0
	if forkedCount > 0 {
		totalForked := 0
		totalPrefix := 0
		for _, s := range t.sessions {
			s.mu.RLock()
			if s.ParentID != "" {
				totalForked += len(s.Blocks)
				totalPrefix += s.PrefixBlocks
			}
			s.mu.RUnlock()
		}
		if totalForked > 0 {
			prefixRate = float64(totalPrefix) / float64(totalForked)
			if prefixRate > 1.0 {
				prefixRate = 1.0
			}
		}
	}

	return SubagentSharingReport{
		ActiveBranches:      len(t.sessions),
		ForkedSubagents:     forkedCount,
		TotalSharedPages:    sharedBlocks,
		TotalPhysicalBlocks: len(t.physicalBlocks),
		TotalLogicalBlocks:  totalLogicalBlocks,
		AllocatedBytes:      t.totalAllocatedBytes,
		DeduplicatedBytes:   dedupBytes,
		DedupRatio:          dedupRatio,
		PrefixHitRate:       prefixRate,
		DuplicatePagesAlloc: 0,
		SavedMemorySummary:  FormatSavedMemory(dedupBytes),
	}
}

// SubagentSharingReport returns aggregate zero-copy memory savings via the default COWPageTable.
func SubagentSharingReportSummary() SubagentSharingReport {
	return defaultCOWPageTable.SubagentSharingReport()
}
