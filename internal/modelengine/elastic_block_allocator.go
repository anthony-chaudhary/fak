package modelengine

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// DrainStateMachineState represents the lifecycle phase of the elastic block allocator.
type DrainStateMachineState int

const (
	// StateStable indicates the allocator is operating normally at target capacity.
	StateStable DrainStateMachineState = iota
	// StateDraining indicates the allocator is progressively shrinking toward target capacity.
	StateDraining
	// StateExpanding indicates the allocator is expanding blocks up to a new target capacity.
	StateExpanding
)

func (s DrainStateMachineState) String() string {
	switch s {
	case StateStable:
		return "STABLE"
	case StateDraining:
		return "DRAINING"
	case StateExpanding:
		return "EXPANDING"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", int(s))
	}
}

var (
	// ErrInvalidTargetBlocks is returned when a non-positive or invalid target block count is requested.
	ErrInvalidTargetBlocks = errors.New("modelengine: target block count must be positive")
	// ErrAllocDrainRefused is returned when an allocation cannot be fulfilled because the allocator is draining.
	ErrAllocDrainRefused = errors.New("modelengine: block allocation refused during progressive drain")
	// ErrAllocCapacityExceeded is returned when all current capacity is exhausted.
	ErrAllocCapacityExceeded = errors.New("modelengine: block allocation capacity exceeded")
	// ErrSequenceNotFound is returned when an operation references an unregistered sequence.
	ErrSequenceNotFound = errors.New("modelengine: sequence not found")
	// ErrBlockNotAllocated is returned when freeing a block that is not currently marked allocated.
	ErrBlockNotAllocated = errors.New("modelengine: block is not currently allocated")
)

// LRUPrefixReclaimCallback is invoked during progressive drain to reclaim prefix cache entries
// or sequence prefixes to recover blocks. Returns the number of blocks freed.
type LRUPrefixReclaimCallback func(neededBlocks int) (freedBlocks int, err error)

// VictimSequenceSelector is invoked during progressive drain to select candidate sequences
// for preemption or eviction when unallocated free blocks and prefix cache evictions are insufficient.
// It returns the sequence ID to preempt/evict.
type VictimSequenceSelector func(activeSequences []ActiveSequenceInfo, neededBlocks int) (victimID string, err error)

// ActiveSequenceInfo carries snapshot metadata for candidate victim sequence selection.
type ActiveSequenceInfo struct {
	SequenceID   string
	Allocated    int
	LastAccessAt time.Time
	Priority     int
	Tokens       int
}

// ElasticBlockStats is a thread-safe snapshot of the elastic block allocator metrics.
type ElasticBlockStats struct {
	State              DrainStateMachineState
	CurrentTotalBlocks int
	TargetBlocks       int
	AllocatedBlocks    int
	FreeBlocks         int
	DrainingBlocks     int
	ActiveSequences    int
	TotalExpansions    int64
	TotalDrains        int64
	ReclaimedBlocks    int64
	PreemptedSequences int64
}

// ElasticBlockAllocatorConfig configures the elastic allocator.
type ElasticBlockAllocatorConfig struct {
	InitialBlocks   int
	ModelCfg        model.Config
	BlockTokens     int
	PrefixReclaimer LRUPrefixReclaimCallback
	VictimSelector  VictimSequenceSelector
}

type sequenceState struct {
	id           string
	blocks       map[int]bool
	lastAccessAt time.Time
	priority     int
	tokens       int
}

// ElasticBlockAllocator implements an elastic PagedAttention KV-cache block allocator
// with a progressive drain state machine (Tier 2).
// It maintains the core invariant:
//
//	allocated + free == current_total
//
// across expansion, progressive drain, preemption, and steady-state allocations.
type ElasticBlockAllocator struct {
	mu sync.Mutex

	state        DrainStateMachineState
	currentTotal int
	targetBlocks int
	blockTokens  int

	pool *model.PagedKVPool

	// Track free physical block IDs in this elastic allocator.
	freeBlocks map[int]bool
	// Track allocated physical block IDs to ensure ownership and invariant verification.
	allocatedBlocks map[int]bool

	// Sequence tracking for victim selection and per-sequence block accounting.
	sequences map[string]*sequenceState

	// Pluggable eviction/victim callbacks.
	prefixReclaimer LRUPrefixReclaimCallback
	victimSelector  VictimSequenceSelector

	// Metrics
	totalExpansions    int64
	totalDrains        int64
	reclaimedBlocks    int64
	preemptedSequences int64
}

// NewElasticBlockAllocator creates a new allocator initialized to initialBlocks.
func NewElasticBlockAllocator(cfg ElasticBlockAllocatorConfig) (*ElasticBlockAllocator, error) {
	if cfg.InitialBlocks <= 0 {
		return nil, ErrInvalidTargetBlocks
	}
	bt := cfg.BlockTokens
	if bt <= 0 {
		bt = 16
	}

	pool := model.NewPagedKVPoolWithRaw(cfg.ModelCfg, bt)
	// Pre-populate pool to initialBlocks
	freeMap := make(map[int]bool, cfg.InitialBlocks)
	for i := 0; i < cfg.InitialBlocks; i++ {
		id := pool.Alloc()
		freeMap[id] = true
	}

	allocator := &ElasticBlockAllocator{
		state:           StateStable,
		currentTotal:    cfg.InitialBlocks,
		targetBlocks:    cfg.InitialBlocks,
		blockTokens:     bt,
		pool:            pool,
		freeBlocks:      freeMap,
		allocatedBlocks: make(map[int]bool, cfg.InitialBlocks),
		sequences:       make(map[string]*sequenceState),
		prefixReclaimer: cfg.PrefixReclaimer,
		victimSelector:  cfg.VictimSelector,
	}

	return allocator, nil
}

// State returns the current drain state machine state.
func (a *ElasticBlockAllocator) State() DrainStateMachineState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state
}

// Stats returns a thread-safe snapshot of the allocator's operational statistics.
func (a *ElasticBlockAllocator) Stats() ElasticBlockStats {
	a.mu.Lock()
	defer a.mu.Unlock()

	draining := 0
	if a.state == StateDraining && a.currentTotal > a.targetBlocks {
		draining = a.currentTotal - a.targetBlocks
	}

	return ElasticBlockStats{
		State:              a.state,
		CurrentTotalBlocks: a.currentTotal,
		TargetBlocks:       a.targetBlocks,
		AllocatedBlocks:    len(a.allocatedBlocks),
		FreeBlocks:         len(a.freeBlocks),
		DrainingBlocks:     draining,
		ActiveSequences:    len(a.sequences),
		TotalExpansions:    a.totalExpansions,
		TotalDrains:        a.totalDrains,
		ReclaimedBlocks:    a.reclaimedBlocks,
		PreemptedSequences: a.preemptedSequences,
	}
}

// VerifyInvariant checks that:
//
//	len(allocatedBlocks) + len(freeBlocks) == currentTotal
//
// and that no block is in both allocated and free sets. Returns nil if valid.
func (a *ElasticBlockAllocator) VerifyInvariant() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.verifyInvariantLocked()
}

func (a *ElasticBlockAllocator) verifyInvariantLocked() error {
	allocCount := len(a.allocatedBlocks)
	freeCount := len(a.freeBlocks)
	if allocCount+freeCount != a.currentTotal {
		return fmt.Errorf("modelengine: invariant violation: allocated (%d) + free (%d) != current_total (%d)",
			allocCount, freeCount, a.currentTotal)
	}

	for id := range a.allocatedBlocks {
		if a.freeBlocks[id] {
			return fmt.Errorf("modelengine: invariant violation: block %d is in both allocated and free sets", id)
		}
	}
	return nil
}

// RegisterSequence registers a sequence for allocation tracking and victim selection.
func (a *ElasticBlockAllocator) RegisterSequence(seqID string, priority int) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if _, exists := a.sequences[seqID]; exists {
		return nil
	}
	a.sequences[seqID] = &sequenceState{
		id:           seqID,
		blocks:       make(map[int]bool),
		lastAccessAt: time.Now(),
		priority:     priority,
	}
	return nil
}

// UnregisterSequence removes sequence tracking and automatically frees all its owned blocks.
func (a *ElasticBlockAllocator) UnregisterSequence(seqID string) ([]int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	seq, exists := a.sequences[seqID]
	if !exists {
		return nil, ErrSequenceNotFound
	}

	freed := make([]int, 0, len(seq.blocks))
	for blkID := range seq.blocks {
		freed = append(freed, blkID)
		delete(a.allocatedBlocks, blkID)
		a.releaseBlockLocked(blkID)
	}
	delete(a.sequences, seqID)

	a.checkDrainCompletionLocked()
	return freed, a.verifyInvariantLocked()
}

// AllocBlock allocates a block to a sequence.
// In StateDraining, new allocations are strictly checked against available capacity.
func (a *ElasticBlockAllocator) AllocBlock(seqID string) (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	seq, exists := a.sequences[seqID]
	if !exists {
		return -1, ErrSequenceNotFound
	}

	// In draining mode, if we are at or below target after accounting for allocated blocks,
	// or no free blocks exist, refuse allocation or fail.
	if len(a.freeBlocks) == 0 {
		return -1, ErrAllocCapacityExceeded
	}

	// Pick a free block (deterministically lowest block ID)
	var blkID int = -1
	for id := range a.freeBlocks {
		if blkID == -1 || id < blkID {
			blkID = id
		}
	}
	delete(a.freeBlocks, blkID)
	a.allocatedBlocks[blkID] = true
	seq.blocks[blkID] = true
	seq.lastAccessAt = time.Now()

	if err := a.verifyInvariantLocked(); err != nil {
		return -1, err
	}
	return blkID, nil
}

// FreeBlock returns an allocated block from a sequence back to the pool.
// If the allocator is draining, freed blocks may be reclaimed to progress toward the target watermark.
func (a *ElasticBlockAllocator) FreeBlock(seqID string, blkID int) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	seq, exists := a.sequences[seqID]
	if !exists {
		return ErrSequenceNotFound
	}
	if !seq.blocks[blkID] || !a.allocatedBlocks[blkID] {
		return ErrBlockNotAllocated
	}

	delete(seq.blocks, blkID)
	delete(a.allocatedBlocks, blkID)

	a.releaseBlockLocked(blkID)
	a.checkDrainCompletionLocked()

	return a.verifyInvariantLocked()
}

func (a *ElasticBlockAllocator) releaseBlockLocked(blkID int) {
	if a.state == StateDraining && a.currentTotal > a.targetBlocks {
		// Progressive drain: retire the block completely from pool
		a.pool.Release(blkID)
		a.currentTotal--
	} else {
		// Recycle into free set
		a.freeBlocks[blkID] = true
	}
}

// Expand increases the block pool capacity to newTargetBlocks.
// Transitions through StateExpanding -> StateStable atomically or mutex-protected.
func (a *ElasticBlockAllocator) Expand(newTargetBlocks int) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if newTargetBlocks <= a.currentTotal {
		return fmt.Errorf("modelengine: expansion target %d must be greater than current total %d",
			newTargetBlocks, a.currentTotal)
	}

	a.state = StateExpanding
	a.targetBlocks = newTargetBlocks
	a.totalExpansions++

	delta := newTargetBlocks - a.currentTotal
	for i := 0; i < delta; i++ {
		id := a.pool.Alloc()
		a.freeBlocks[id] = true
		a.currentTotal++
	}

	a.state = StateStable
	return a.verifyInvariantLocked()
}

// RequestDrain initiates a progressive drain toward targetWatermark.
// If targetWatermark >= currentTotal, this is an error or no-op.
// Returns immediately after setting StateDraining and draining any currently free blocks.
func (a *ElasticBlockAllocator) RequestDrain(targetWatermark int) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if targetWatermark <= 0 {
		return ErrInvalidTargetBlocks
	}
	if targetWatermark >= a.currentTotal {
		// Nothing to drain; already at or below target
		a.targetBlocks = targetWatermark
		a.state = StateStable
		return nil
	}

	a.state = StateDraining
	a.targetBlocks = targetWatermark
	a.totalDrains++

	// Immediately reclaim available free blocks up to the delta
	a.reclaimFreeBlocksLocked()
	a.checkDrainCompletionLocked()

	return a.verifyInvariantLocked()
}

func (a *ElasticBlockAllocator) reclaimFreeBlocksLocked() {
	if a.state != StateDraining {
		return
	}
	needed := a.currentTotal - a.targetBlocks
	if needed <= 0 {
		return
	}

	freeIDs := make([]int, 0, len(a.freeBlocks))
	for id := range a.freeBlocks {
		freeIDs = append(freeIDs, id)
	}
	sort.Ints(freeIDs)

	for _, id := range freeIDs {
		if needed <= 0 {
			break
		}
		delete(a.freeBlocks, id)
		a.pool.Release(id)
		a.currentTotal--
		needed--
	}
}

func (a *ElasticBlockAllocator) checkDrainCompletionLocked() {
	if a.state == StateDraining && a.currentTotal <= a.targetBlocks {
		a.currentTotal = a.targetBlocks
		a.state = StateStable
	}
}

// ProgressDrain executes one step of progressive drain:
// 1. Reclaims any newly freed blocks toward targetWatermark.
// 2. If still over target, calls LRUPrefixReclaimCallback if configured.
// 3. If still over target, calls VictimSequenceSelector (or default LRU sequence selection) to preempt a sequence.
// Returns true when the target watermark has been reached (StateStable), false if more draining is needed.
func (a *ElasticBlockAllocator) ProgressDrain() (bool, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.state != StateDraining {
		return true, nil
	}

	// 1. Reclaim free blocks
	a.reclaimFreeBlocksLocked()
	if a.currentTotal <= a.targetBlocks {
		a.checkDrainCompletionLocked()
		return true, a.verifyInvariantLocked()
	}

	needed := a.currentTotal - a.targetBlocks

	// 2. Prefix cache reclaim
	if a.prefixReclaimer != nil && needed > 0 {
		// Temporarily unlock if reclaim callback needs to call into allocator,
		// but typically callback cleans outside or unregisters sequences.
		// For safe invocation, call callback while holding or not holding lock.
		// Here prefixReclaimer takes neededBlocks and returns freedBlocks.
		freed, err := a.prefixReclaimer(needed)
		if err != nil {
			return false, err
		}
		if freed > 0 {
			a.reclaimedBlocks += int64(freed)
			a.reclaimFreeBlocksLocked()
			if a.currentTotal <= a.targetBlocks {
				a.checkDrainCompletionLocked()
				return true, a.verifyInvariantLocked()
			}
			needed = a.currentTotal - a.targetBlocks
		}
	}

	// 3. Victim sequence selection
	if needed > 0 && len(a.sequences) > 0 {
		victimID, err := a.selectVictimLocked(needed)
		if err != nil {
			return false, err
		}
		if victimID != "" {
			seq, exists := a.sequences[victimID]
			if exists {
				// Preempt victim sequence: release all its blocks
				for blkID := range seq.blocks {
					delete(a.allocatedBlocks, blkID)
					a.releaseBlockLocked(blkID)
				}
				delete(a.sequences, victimID)
				a.preemptedSequences++
				a.reclaimFreeBlocksLocked()
			}
		}
	}

	a.checkDrainCompletionLocked()
	isComplete := (a.state == StateStable)
	return isComplete, a.verifyInvariantLocked()
}

func (a *ElasticBlockAllocator) selectVictimLocked(needed int) (string, error) {
	if len(a.sequences) == 0 {
		return "", nil
	}

	// Build active candidate list
	candidates := make([]ActiveSequenceInfo, 0, len(a.sequences))
	for id, seq := range a.sequences {
		candidates = append(candidates, ActiveSequenceInfo{
			SequenceID:   id,
			Allocated:    len(seq.blocks),
			LastAccessAt: seq.lastAccessAt,
			Priority:     seq.priority,
			Tokens:       seq.tokens,
		})
	}

	if a.victimSelector != nil {
		return a.victimSelector(candidates, needed)
	}

	// Default fallback: LRU selection (oldest LastAccessAt, tie-break on SequenceID)
	var oldestID string
	var oldestTime time.Time
	for _, cand := range candidates {
		if oldestID == "" || cand.LastAccessAt.Before(oldestTime) || (cand.LastAccessAt.Equal(oldestTime) && cand.SequenceID < oldestID) {
			oldestID = cand.SequenceID
			oldestTime = cand.LastAccessAt
		}
	}
	return oldestID, nil
}

// TouchSequence updates the last-access timestamp for LRU ordering.
func (a *ElasticBlockAllocator) TouchSequence(seqID string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	seq, exists := a.sequences[seqID]
	if !exists {
		return ErrSequenceNotFound
	}
	seq.lastAccessAt = time.Now()
	return nil
}

// SequenceAllocatedBlocks returns the count of blocks currently allocated to seqID.
func (a *ElasticBlockAllocator) SequenceAllocatedBlocks(seqID string) (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	seq, exists := a.sequences[seqID]
	if !exists {
		return 0, ErrSequenceNotFound
	}
	return len(seq.blocks), nil
}
