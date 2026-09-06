package compute

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// dram_ring.go — Host DRAM write-back staging ring buffer with lazy flush for offload write coalescing
// Parent issue: #10964 (SSD endurance architecture and 5-year lifespan budget for high-volume local caching on AMD Strix)
// Tracking issue: #11072.

const (
	// DefaultDRAMStagingCapacity is the default memory capacity limit (2 GiB).
	DefaultDRAMStagingCapacity int64 = 2 * 1024 * 1024 * 1024

	// DefaultDRAMStagingCooldown is the default dwell timer cooldown (60 seconds, within 60-120s window).
	DefaultDRAMStagingCooldown = 60 * time.Second

	// MinDRAMStagingCooldown is the minimum recommended cooldown for write coalescing (60s).
	MinDRAMStagingCooldown = 60 * time.Second

	// MaxDRAMStagingCooldown is the maximum recommended cooldown before write-back (120s).
	MaxDRAMStagingCooldown = 120 * time.Second
)

// Common errors returned by DRAMStagingRing operations.
var (
	ErrEmptyPageID         = errors.New("compute: empty pageID")
	ErrPageExceedsCapacity = errors.New("compute: page size exceeds staging ring capacity")
	ErrNilStorageSink      = errors.New("compute: storage sink callback is nil")
)

// StorageSink is a callback function that writes dirty pages to persistent storage (e.g. NVMe SSD).
type StorageSink func(pageID string, data []byte) error

// DRAMStagingRingConfig configures a DRAMStagingRing.
type DRAMStagingRingConfig struct {
	// CapacityBytes is the maximum memory capacity limit in bytes (default: 2 GiB).
	CapacityBytes int64

	// Cooldown is the dwell time cooldown before a dirty page is eligible for flush (default: 60s).
	Cooldown time.Duration

	// StorageSink is the storage callback invoked when flushing dirty pages.
	StorageSink StorageSink

	// TimeSource provides the current time (defaults to time.Now). Used for deterministic testing.
	TimeSource func() time.Time
}

type dramPageEntry struct {
	id         string
	data       []byte
	dirty      bool
	modifiedAt time.Time
	prev       *dramPageEntry
	next       *dramPageEntry
}

// DRAMStagingRing implements a host DRAM write-back staging ring buffer with lazy flush
// for offload write coalescing. It absorbs high-frequency KV-cache and agent turn updates
// in host memory, coalescing repeated writes and invalidations within a 60-120s dwell window
// to prevent premature SSD wearout.
type DRAMStagingRing struct {
	mu            sync.RWMutex
	capacityBytes int64
	cooldown      time.Duration
	sink          StorageSink
	timeSource    func() time.Time

	usedBytes  int64
	dirtyBytes int64
	dirtyCount int

	pages map[string]*dramPageEntry
	head  *dramPageEntry
	tail  *dramPageEntry
}

// NewDRAMStagingRing constructs a DRAMStagingRing with the given configuration.
func NewDRAMStagingRing(cfg DRAMStagingRingConfig) *DRAMStagingRing {
	capBytes := cfg.CapacityBytes
	if capBytes <= 0 {
		capBytes = DefaultDRAMStagingCapacity
	}
	cd := cfg.Cooldown
	if cd <= 0 {
		cd = DefaultDRAMStagingCooldown
	}

	head := &dramPageEntry{}
	tail := &dramPageEntry{}
	head.next = tail
	tail.prev = head

	return &DRAMStagingRing{
		capacityBytes: capBytes,
		cooldown:      cd,
		sink:          cfg.StorageSink,
		timeSource:    cfg.TimeSource,
		pages:         make(map[string]*dramPageEntry),
		head:          head,
		tail:          tail,
	}
}

func (r *DRAMStagingRing) initLocked() {
	if r.pages == nil {
		r.pages = make(map[string]*dramPageEntry)
		r.head = &dramPageEntry{}
		r.tail = &dramPageEntry{}
		r.head.next = r.tail
		r.tail.prev = r.head
	}
	if r.capacityBytes <= 0 {
		r.capacityBytes = DefaultDRAMStagingCapacity
	}
	if r.cooldown <= 0 {
		r.cooldown = DefaultDRAMStagingCooldown
	}
}

func (r *DRAMStagingRing) now() time.Time {
	if r.timeSource != nil {
		return r.timeSource()
	}
	return time.Now()
}

func (r *DRAMStagingRing) insertTailLocked(entry *dramPageEntry) {
	entry.prev = r.tail.prev
	entry.next = r.tail
	r.tail.prev.next = entry
	r.tail.prev = entry
}

func (r *DRAMStagingRing) removeEntryLocked(entry *dramPageEntry) {
	if entry.prev != nil && entry.next != nil {
		entry.prev.next = entry.next
		entry.next.prev = entry.prev
		entry.prev = nil
		entry.next = nil
	}
}

func (r *DRAMStagingRing) moveToTailLocked(entry *dramPageEntry) {
	r.removeEntryLocked(entry)
	r.insertTailLocked(entry)
}

// reclaimSpaceLocked reclaims memory to accommodate neededBytes.
// Clean pages are evicted first with zero storage writes.
// If further space is needed, dirty pages are flushed to the sink before eviction.
func (r *DRAMStagingRing) reclaimSpaceLocked(neededBytes int64, skip *dramPageEntry) error {
	if (r.usedBytes + neededBytes) <= r.capacityBytes {
		return nil
	}

	// Phase 1: Evict clean resident pages starting from oldest (head.next).
	curr := r.head.next
	for curr != r.tail && (r.usedBytes+neededBytes) > r.capacityBytes {
		next := curr.next
		if curr != skip && !curr.dirty {
			r.removeEntryLocked(curr)
			delete(r.pages, curr.id)
			r.usedBytes -= int64(len(curr.data))
		}
		curr = next
	}

	// Phase 2: If still exceeding capacity, write-back and evict dirty pages.
	if (r.usedBytes + neededBytes) > r.capacityBytes {
		curr = r.head.next
		for curr != r.tail && (r.usedBytes+neededBytes) > r.capacityBytes {
			next := curr.next
			if curr != skip && curr.dirty {
				if r.sink != nil {
					if err := r.sink(curr.id, curr.data); err != nil {
						return fmt.Errorf("compute: evict dirty page %q sink failed: %w", curr.id, err)
					}
				}
				r.removeEntryLocked(curr)
				delete(r.pages, curr.id)
				r.usedBytes -= int64(len(curr.data))
				r.dirtyBytes -= int64(len(curr.data))
				r.dirtyCount--
			}
			curr = next
		}
	}

	if (r.usedBytes + neededBytes) > r.capacityBytes {
		return ErrPageExceedsCapacity
	}
	return nil
}

// Put inserts or updates a page in the DRAM staging ring buffer.
// If dirty is true, the dwell timer resets and dirty state is updated.
// Memory capacity is enforced; clean pages are evicted first, followed by dirty page write-backs.
func (r *DRAMStagingRing) Put(pageID string, data []byte, dirty bool) error {
	if pageID == "" {
		return ErrEmptyPageID
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.initLocked()

	pageSize := int64(len(data))
	if pageSize > r.capacityBytes {
		return ErrPageExceedsCapacity
	}

	now := r.now()

	if existing, exists := r.pages[pageID]; exists {
		delta := pageSize - int64(len(existing.data))
		if delta > 0 && (r.usedBytes+delta) > r.capacityBytes {
			if err := r.reclaimSpaceLocked(delta, existing); err != nil {
				return err
			}
		}

		oldDirty := existing.dirty
		oldSize := int64(len(existing.data))

		if dirty {
			if !oldDirty {
				r.dirtyBytes += pageSize
				r.dirtyCount++
			} else {
				r.dirtyBytes += delta
			}
			existing.modifiedAt = now
			existing.dirty = true
		} else {
			if oldDirty {
				r.dirtyBytes -= oldSize
				r.dirtyCount--
			}
			existing.dirty = false
		}

		cp := make([]byte, len(data))
		copy(cp, data)
		existing.data = cp
		r.usedBytes += delta

		r.moveToTailLocked(existing)
		return nil
	}

	// New page insertion
	if (r.usedBytes + pageSize) > r.capacityBytes {
		if err := r.reclaimSpaceLocked(pageSize, nil); err != nil {
			return err
		}
	}

	cp := make([]byte, len(data))
	copy(cp, data)
	entry := &dramPageEntry{
		id:         pageID,
		data:       cp,
		dirty:      dirty,
		modifiedAt: now,
	}

	r.pages[pageID] = entry
	r.insertTailLocked(entry)
	r.usedBytes += pageSize

	if dirty {
		r.dirtyBytes += pageSize
		r.dirtyCount++
	}

	return nil
}

// Get retrieves a defensive copy of the page data from the staging ring buffer.
func (r *DRAMStagingRing) Get(pageID string) ([]byte, bool) {
	if pageID == "" {
		return nil, false
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.pages == nil {
		return nil, false
	}

	entry, exists := r.pages[pageID]
	if !exists {
		return nil, false
	}

	cp := make([]byte, len(entry.data))
	copy(cp, entry.data)
	return cp, true
}

// Drop invalidates and removes pageID from the staging ring buffer immediately,
// canceling any pending flushes for the page.
func (r *DRAMStagingRing) Drop(pageID string) {
	if pageID == "" {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.initLocked()

	entry, exists := r.pages[pageID]
	if !exists {
		return
	}

	r.removeEntryLocked(entry)
	delete(r.pages, pageID)
	r.usedBytes -= int64(len(entry.data))
	if entry.dirty {
		r.dirtyBytes -= int64(len(entry.data))
		r.dirtyCount--
	}
}

// FlushExpired inspects dirty pages and flushes all whose dwell time has reached
// or exceeded the configured cooldown window (relative to now) to the storage sink callback.
// Successfully flushed pages remain cached in DRAM as clean pages.
func (r *DRAMStagingRing) FlushExpired(now time.Time) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.initLocked()

	if now.IsZero() {
		now = r.now()
	}

	var expired []*dramPageEntry
	for curr := r.head.next; curr != r.tail; curr = curr.next {
		if curr.dirty && now.Sub(curr.modifiedAt) >= r.cooldown {
			expired = append(expired, curr)
		}
	}

	if len(expired) == 0 {
		return 0, nil
	}

	if r.sink == nil {
		return 0, ErrNilStorageSink
	}

	flushed := 0
	for _, page := range expired {
		if p, ok := r.pages[page.id]; ok && p.dirty {
			if err := r.sink(p.id, p.data); err != nil {
				return flushed, fmt.Errorf("compute: flush page %q failed: %w", p.id, err)
			}
			p.dirty = false
			r.dirtyBytes -= int64(len(p.data))
			r.dirtyCount--
			flushed++
		}
	}

	return flushed, nil
}

// Capacity returns the maximum memory capacity limit in bytes.
func (r *DRAMStagingRing) Capacity() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.capacityBytes <= 0 {
		return DefaultDRAMStagingCapacity
	}
	return r.capacityBytes
}

// Cooldown returns the dwell timer cooldown duration.
func (r *DRAMStagingRing) Cooldown() time.Duration {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.cooldown <= 0 {
		return DefaultDRAMStagingCooldown
	}
	return r.cooldown
}

// UsedBytes returns the current total resident memory in bytes.
func (r *DRAMStagingRing) UsedBytes() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.usedBytes
}

// DirtyBytes returns the memory consumed by dirty pages awaiting flush in bytes.
func (r *DRAMStagingRing) DirtyBytes() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.dirtyBytes
}

// DirtyCount returns the number of dirty pages currently in the ring.
func (r *DRAMStagingRing) DirtyCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.dirtyCount
}

// Len returns the total number of resident pages in the staging ring.
func (r *DRAMStagingRing) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.pages)
}

// SetStorageSink updates the storage sink callback.
func (r *DRAMStagingRing) SetStorageSink(sink StorageSink) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sink = sink
}
