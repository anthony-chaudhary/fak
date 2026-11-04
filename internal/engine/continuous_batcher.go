package engine

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ContinuousBatcher manages dynamic iteration-level continuous batching for subagent turn loops.
type ContinuousBatcher struct {
	mu           sync.Mutex
	cfg          ContinuousBatcherConfig
	slots        []*Slot
	sessionMap   map[string]*Slot
	completedMap map[string]*Slot
	waitingQueue []*SubagentRequest
	iteration    uint64
	totalTokens  int64
	seqCounter   uint64
	uidSeq       uint64
	closed       bool

	// prefixCache is the extend-only GDN recurrent boundary cache. It is non-nil
	// only for a hybrid model with reuse enabled (and not explicitly disabled).
	prefixCache *RecurrentPrefixCache

	// pendingPrefixHits / pendingPrefixReuseTokens are admissions observed since
	// the last emitted step. They are incremented once at slot admission (when a
	// cached recurrent boundary is reused) and captured-and-reset by the next
	// step, so a reused slot resident for N steps is reported exactly once.
	pendingPrefixHits        int
	pendingPrefixReuseTokens int
	decodeSinceRefill        int // decode steps since the last prefill step
}

// NewContinuousBatcher constructs a scheduler with the specified configuration.
func NewContinuousBatcher(cfg ...ContinuousBatcherConfig) (*ContinuousBatcher, error) {
	c := DefaultContinuousBatcherConfig()
	if len(cfg) > 0 {
		c = cfg[0]
		if c.MaxSlots == 0 {
			c.MaxSlots = DefaultMaxSlots
		}
		if c.MemoryBandwidthGBs <= 0 {
			c.MemoryBandwidthGBs = StrixHaloDefaultBandwidthGBs
		}
		if c.ComputePeakTFLOPs <= 0 {
			c.ComputePeakTFLOPs = StrixHaloComputePeakTFLOPs
		}
		if c.ModelWeightsBytes <= 0 {
			c.ModelWeightsBytes = Qwen38_14B_WeightsBytes
		}
		if c.ModelParams <= 0 {
			c.ModelParams = Qwen38_14B_Params
		}
		if c.SingleAgentTokPerSec <= 0 {
			c.SingleAgentTokPerSec = Qwen38_14B_SingleAgentTokSec
		}
		if c.FixedOverheadSec <= 0 {
			c.FixedOverheadSec = 0.0025
		}
		if c.KVBytesPerToken <= 0 {
			c.KVBytesPerToken = 1024
		}
		if c.MaxKVCacheBytes < 0 {
			c.MaxKVCacheBytes = 0
		}
	}

	if c.MaxSlots < MinSlots || c.MaxSlots > MaxSlots {
		return nil, ErrInvalidSlots
	}

	slots := make([]*Slot, c.MaxSlots)
	for i := 0; i < c.MaxSlots; i++ {
		slots[i] = &Slot{
			Index: i,
			State: SlotStateEmpty,
		}
	}

	cb := &ContinuousBatcher{
		cfg:          c,
		slots:        slots,
		sessionMap:   make(map[string]*Slot),
		completedMap: make(map[string]*Slot),
		waitingQueue: make([]*SubagentRequest, 0),
	}
	if c.Model != nil && c.Model.Cfg.IsHybrid() && !c.DisableRecurrentPrefixReuse {
		cb.prefixCache = NewRecurrentPrefixCache(c.RecurrentPrefixCapacity)
	}
	return cb, nil
}

// YieldSlot transitions a slot to SlotStateYieldedIO when a subagent executes tools,
// freeing compute iteration time without evicting its KV cache.
func (cb *ContinuousBatcher) YieldSlot(sessionID string) error {
	if sessionID == "" {
		return errors.New("continuous_batcher: empty session ID")
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return ErrBatcherClosed
	}

	slot, ok := cb.sessionMap[sessionID]
	if !ok {
		return ErrSessionNotFound
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	if slot.State != SlotStateActiveDecode {
		return ErrSlotNotActive
	}

	slot.State = SlotStateYieldedIO
	slot.YieldCount++
	slot.EvictionDuration = 0
	slot.BytesSwapped = 0
	slot.KVCacheStationary = true

	return nil
}

// ResumeSlot resumes a yielded subagent immediately when tool execution completes.
func (cb *ContinuousBatcher) ResumeSlot(sessionID string) error {
	if sessionID == "" {
		return errors.New("continuous_batcher: empty session ID")
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return ErrBatcherClosed
	}

	slot, ok := cb.sessionMap[sessionID]
	if !ok {
		return ErrSessionNotFound
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	if slot.State != SlotStateYieldedIO {
		return ErrSlotNotYielded
	}

	slot.State = SlotStateActiveDecode
	slot.ResumeCount++
	slot.ReprefillTokens = 0

	return nil
}

// ActiveSlotCount returns count of slots currently in SlotStateActiveDecode.
func (cb *ContinuousBatcher) ActiveSlotCount() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	count := 0
	for _, s := range cb.slots {
		s.mu.Lock()
		if s.State == SlotStateActiveDecode {
			count++
		}
		s.mu.Unlock()
	}
	return count
}

// YieldedSlotCount returns count of slots currently in SlotStateYieldedIO.
func (cb *ContinuousBatcher) YieldedSlotCount() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	count := 0
	for _, s := range cb.slots {
		s.mu.Lock()
		if s.State == SlotStateYieldedIO {
			count++
		}
		s.mu.Unlock()
	}
	return count
}

// FinishedSlotCount returns count of slots currently in SlotStateFinished.
func (cb *ContinuousBatcher) FinishedSlotCount() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	count := 0
	for _, s := range cb.slots {
		s.mu.Lock()
		if s.State == SlotStateFinished {
			count++
		}
		s.mu.Unlock()
	}
	return count
}

// EmptySlotCount returns count of slots currently in SlotStateEmpty.
func (cb *ContinuousBatcher) EmptySlotCount() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	count := 0
	for _, s := range cb.slots {
		s.mu.Lock()
		if s.State == SlotStateEmpty {
			count++
		}
		s.mu.Unlock()
	}
	return count
}

// TotalTokensGenerated returns cumulative count of decode tokens produced.
func (cb *ContinuousBatcher) TotalTokensGenerated() int64 {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.totalTokens
}

// Iteration returns the current batch iteration step index.
func (cb *ContinuousBatcher) Iteration() uint64 {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.iteration
}

// GetSlot retrieves slot status for a session ID.
func (cb *ContinuousBatcher) GetSlot(sessionID string) (*Slot, bool) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if slot, ok := cb.sessionMap[sessionID]; ok {
		return slot, true
	}
	if completed, ok := cb.completedMap[sessionID]; ok {
		return completed, true
	}
	return nil, false
}

// Slots returns a snapshot of the slots slice.
func (cb *ContinuousBatcher) Slots() []*Slot {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	res := make([]*Slot, len(cb.slots))
	for i, s := range cb.slots {
		res[i] = s.snapshot()
	}
	return res
}

// Cancel cancels a subagent session and frees its slot.
func (cb *ContinuousBatcher) Cancel(sessionID string) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return ErrBatcherClosed
	}

	slot, ok := cb.sessionMap[sessionID]
	if !ok {
		// Check waiting queue
		for i, q := range cb.waitingQueue {
			if q.SessionID == sessionID {
				copy(cb.waitingQueue[i:], cb.waitingQueue[i+1:])
				last := len(cb.waitingQueue) - 1
				cb.waitingQueue[last] = nil
				cb.waitingQueue = cb.waitingQueue[:last]
				return nil
			}
		}
		return ErrSessionNotFound
	}

	slot.mu.Lock()
	slot.State = SlotStateFinished
	slot.err = context.Canceled
	select {
	case <-slot.doneCh:
	default:
		close(slot.doneCh)
	}
	close(slot.tokenCh)
	slot.closeSessionLocked()
	slot.mu.Unlock()

	cb.completedMap[sessionID] = slot
	delete(cb.sessionMap, sessionID)

	idx := slot.Index
	cb.slots[idx] = &Slot{
		Index: idx,
		State: SlotStateEmpty,
	}
	cb.tryPromoteWaitingLocked()

	return nil
}

// Close closes the batcher and releases all scheduler-owned work.
func (cb *ContinuousBatcher) Close() error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return nil
	}
	cb.closed = true
	cb.waitingQueue = nil

	for i, slot := range cb.slots {
		slot.mu.Lock()
		if slot.State != SlotStateEmpty {
			sessionID := slot.SessionID
			slot.State = SlotStateFinished
			slot.err = ErrBatcherClosed
			slot.CompletedAt = time.Now()
			select {
			case <-slot.doneCh:
			default:
				close(slot.doneCh)
				close(slot.tokenCh)
			}
			slot.closeSessionLocked()
			slot.mu.Unlock()

			cb.completedMap[sessionID] = slot
			delete(cb.sessionMap, sessionID)
			cb.slots[i] = &Slot{Index: i, State: SlotStateEmpty}
			continue
		}
		slot.mu.Unlock()
	}
	for _, slot := range cb.completedMap {
		slot.mu.Lock()
		slot.closeSessionLocked()
		slot.mu.Unlock()
	}

	if cb.prefixCache != nil {
		cb.prefixCache.Clear()
	}

	return nil
}
