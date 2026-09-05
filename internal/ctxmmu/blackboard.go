package ctxmmu

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BlackboardEntry represents an immutable memory slice published to the blackboard.
type BlackboardEntry struct {
	ID        string            `json:"id"`
	Topic     string            `json:"topic"`
	Ref       *abi.Ref          `json:"ref"`
	Epoch     uint64            `json:"epoch"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	RefCount  int32             `json:"ref_count"`
	CreatedAt time.Time         `json:"created_at"`
}

// BlackboardStats captures diagnostic counters for the in-process blackboard MMU.
type BlackboardStats struct {
	EntryCount    int   `json:"entry_count"`
	MemoryBytes   int64 `json:"memory_bytes"`
	ZeroCopyReads int64 `json:"zero_copy_reads"`
	ActiveRefs    int64 `json:"active_refs"`
}

// Blackboard provides an in-process, thread-safe memory blackboard with topic-keyed
// immutable memory slices (*abi.Ref) and reference counting.
type Blackboard struct {
	mu            sync.RWMutex
	entries       map[string]*BlackboardEntry
	topics        map[string][]string // topic -> list of entry IDs
	nextID        uint64
	zeroCopyReads int64
	memoryBytes   int64
	activeRefs    int64
}

// NewBlackboard constructs a new in-process thread-safe Blackboard.
func NewBlackboard() *Blackboard {
	return &Blackboard{
		entries: make(map[string]*BlackboardEntry),
		topics:  make(map[string][]string),
	}
}

// Publish registers a topic-keyed immutable memory slice (*abi.Ref) into the blackboard.
// The entry is initialized with a reference count of 1.
func (b *Blackboard) Publish(topic string, ref *abi.Ref, epoch uint64, metadata map[string]string) (string, error) {
	if topic == "" {
		return "", errors.New("ctxmmu: blackboard publish requires non-empty topic")
	}
	if ref == nil {
		return "", errors.New("ctxmmu: blackboard publish requires non-nil ref")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	b.nextID++
	id := fmt.Sprintf("bb-%d", b.nextID)

	refCopy := *ref
	if refCopy.Len <= 0 && len(refCopy.Inline) > 0 {
		refCopy.Len = int64(len(refCopy.Inline))
	}

	var metaCopy map[string]string
	if len(metadata) > 0 {
		metaCopy = make(map[string]string, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
	}

	memBytes := refCopy.Len
	if memBytes <= 0 && len(refCopy.Inline) > 0 {
		memBytes = int64(len(refCopy.Inline))
	}

	entry := &BlackboardEntry{
		ID:        id,
		Topic:     topic,
		Ref:       &refCopy,
		Epoch:     epoch,
		Metadata:  metaCopy,
		RefCount:  1,
		CreatedAt: time.Now().UTC(),
	}

	b.entries[id] = entry
	b.topics[topic] = append(b.topics[topic], id)
	b.memoryBytes += memBytes
	b.activeRefs++

	abi.PinResolved(refCopy)

	return id, nil
}

// Subscribe returns all active blackboard entries published under topic.
// Each returned entry counts toward zero-copy read diagnostics.
func (b *Blackboard) Subscribe(topic string) []*BlackboardEntry {
	b.mu.RLock()
	defer b.mu.RUnlock()

	ids := b.topics[topic]
	if len(ids) == 0 {
		return nil
	}

	res := make([]*BlackboardEntry, 0, len(ids))
	for _, id := range ids {
		if e, ok := b.entries[id]; ok {
			res = append(res, e)
		}
	}
	if len(res) > 0 {
		atomic.AddInt64(&b.zeroCopyReads, int64(len(res)))
	}
	return res
}

// Lookup retrieves a single blackboard entry by its unique ID.
// Successful retrieval increments zero-copy read diagnostics.
func (b *Blackboard) Lookup(id string) (*BlackboardEntry, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	e, ok := b.entries[id]
	if ok {
		atomic.AddInt64(&b.zeroCopyReads, 1)
		return e, true
	}
	return nil, false
}

// Retain increments the reference count of the entry identified by id.
func (b *Blackboard) Retain(id string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	e, ok := b.entries[id]
	if !ok {
		return fmt.Errorf("ctxmmu: blackboard retain failed: entry %s not found", id)
	}
	e.RefCount++
	b.activeRefs++
	return nil
}

// Release decrements the reference count of the entry identified by id.
// If the reference count drops to zero, the entry is reaped from the blackboard.
func (b *Blackboard) Release(id string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	e, ok := b.entries[id]
	if !ok {
		return fmt.Errorf("ctxmmu: blackboard release failed: entry %s not found", id)
	}
	if e.RefCount <= 0 {
		return fmt.Errorf("ctxmmu: blackboard release failed: entry %s already released", id)
	}

	e.RefCount--
	b.activeRefs--
	if e.RefCount == 0 {
		b.deleteEntryLocked(e)
	}
	return nil
}

// InvalidateEpoch reaps all entries matching epoch (e.g. when parent turns close or leases expire).
// Returns the count of reaped entries.
func (b *Blackboard) InvalidateEpoch(epoch uint64) int {
	b.mu.Lock()
	defer b.mu.Unlock()

	var toReap []*BlackboardEntry
	for _, e := range b.entries {
		if e.Epoch == epoch {
			toReap = append(toReap, e)
		}
	}
	for _, e := range toReap {
		b.activeRefs -= int64(e.RefCount)
		b.deleteEntryLocked(e)
	}
	if b.activeRefs < 0 {
		b.activeRefs = 0
	}
	return len(toReap)
}

// Stats returns diagnostic counters reporting entry count, memory bytes, zero-copy reads, and active refs.
func (b *Blackboard) Stats() BlackboardStats {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return BlackboardStats{
		EntryCount:    len(b.entries),
		MemoryBytes:   b.memoryBytes,
		ZeroCopyReads: atomic.LoadInt64(&b.zeroCopyReads),
		ActiveRefs:    b.activeRefs,
	}
}

// deleteEntryLocked removes an entry and adjusts internal structures and stats. Caller must hold b.mu.
func (b *Blackboard) deleteEntryLocked(e *BlackboardEntry) {
	delete(b.entries, e.ID)
	memBytes := e.Ref.Len
	if memBytes <= 0 && len(e.Ref.Inline) > 0 {
		memBytes = int64(len(e.Ref.Inline))
	}
	b.memoryBytes -= memBytes
	if b.memoryBytes < 0 {
		b.memoryBytes = 0
	}

	if ids, ok := b.topics[e.Topic]; ok {
		filtered := ids[:0]
		for _, id := range ids {
			if id != e.ID {
				filtered = append(filtered, id)
			}
		}
		if len(filtered) == 0 {
			delete(b.topics, e.Topic)
		} else {
			b.topics[e.Topic] = filtered
		}
	}

	if e.Ref != nil {
		abi.UnpinResolved(*e.Ref)
	}
}
