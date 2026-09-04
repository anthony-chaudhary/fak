package lease

import "time"

// Table manages leases and pins for a single shard.
// Accessed only by the shard goroutine — zero synchronization needed.
type Table struct {
	leases        map[uint64]int64 // keyHash → expiry (Unix ms)
	pins          map[uint64]bool  // keyHash → pinned
	maxDurationMs int64
}

// NewTable creates a new lease table with maximum lease duration.
func NewTable(maxDurationMs int64) *Table {
	return &Table{
		leases:        make(map[uint64]int64),
		pins:          make(map[uint64]bool),
		maxDurationMs: maxDurationMs,
	}
}

// Grant creates or extends a lease for a key hash.
func (t *Table) Grant(keyHash uint64, durationMs int64) {
	if durationMs > t.maxDurationMs {
		durationMs = t.maxDurationMs
	}
	if durationMs <= 0 {
		durationMs = 5000 // default 5s
	}
	expiry := time.Now().UnixMilli() + durationMs
	// Only extend, never shorten an existing lease
	if existing, ok := t.leases[keyHash]; ok && existing > expiry {
		return
	}
	t.leases[keyHash] = expiry
}

// IsProtected returns true if the key has an active lease or is pinned.
func (t *Table) IsProtected(keyHash uint64) bool {
	if t.pins[keyHash] {
		return true
	}
	expiry, ok := t.leases[keyHash]
	if !ok {
		return false
	}
	if time.Now().UnixMilli() > expiry {
		delete(t.leases, keyHash)
		return false
	}
	return true
}

// Pin permanently protects a key from eviction (until Unpin).
func (t *Table) Pin(keyHash uint64) {
	t.pins[keyHash] = true
}

// Unpin removes pin protection.
func (t *Table) Unpin(keyHash uint64) {
	delete(t.pins, keyHash)
}

// Cleanup removes expired leases. Called periodically by the shard.
func (t *Table) Cleanup() {
	now := time.Now().UnixMilli()
	for k, expiry := range t.leases {
		if now > expiry {
			delete(t.leases, k)
		}
	}
}

// LeaseCount returns the number of active leases.
func (t *Table) LeaseCount() int {
	return len(t.leases)
}

// PinCount returns the number of pinned keys.
func (t *Table) PinCount() int {
	return len(t.pins)
}
