package index

import (
	"encoding/binary"
	"math/bits"
	"sync/atomic"
)

const (
	CtrlEmpty     = 0x80
	CtrlTombstone = 0xFE
	GroupSize     = 16
	SlotSize      = 48

	// BytesPerEntry is the total memory per Swiss table entry: one 48-byte slot
	// plus one control byte.  Used for index memory estimation in config validation
	// and startup logging — do NOT hardcode "49" elsewhere.
	BytesPerEntry = SlotSize + 1 // 49

	// MaxLoadNumerator / MaxLoadDenominator define the Swiss table load factor
	// (7/8 = 87.5%).  The table triggers growth when count reaches
	// capacity * MaxLoadNumerator / MaxLoadDenominator.
	MaxLoadNumerator   = 7
	MaxLoadDenominator = 8

	// Slot field offsets within the 48-byte slot
	offKeyHash       = 0
	offKeyLen        = 8
	offKeyOffset     = 10
	offValueOffset   = 18
	offValueLen      = 26
	offTTL           = 30
	offRefCount      = 38
	offFlags         = 42
	offValueClassIdx = 44 // 1 byte: actual slab class index
	offKeyClassIdx   = 45 // 1 byte: actual slab class index
	// bytes 46-47 unused

	FlagPinned      = 1 << 0
	FlagLeased      = 1 << 1
	FlagMigrated    = 1 << 2 // entry has been migrated to new allocator during ZeroLatencyBalance
	FlagPromoted    = 1 << 3 // value was promoted to a larger slab class
	FlagHasClassIdx = 1 << 4 // stored class indices are valid (new entries set this)
)

// Entry represents a decoded Swiss table slot.
type Entry struct {
	KeyHash       uint64
	KeyLen        uint16
	KeyOffset     uint64
	ValueOffset   uint64
	ValueLen      uint32
	TTL           int64 // Unix timestamp in ms; 0 = no expiry
	RefCount      uint32
	Flags         uint16
	ValueClassIdx uint8 // actual slab class index (valid when FlagHasClassIdx set)
	KeyClassIdx   uint8 // actual slab class index (valid when FlagHasClassIdx set)
}

// Table is an off-heap Swiss table stored in flat byte slices.
// All data is in ctrl[] and slots[] — no Go heap pointers.
type Table struct {
	ctrl        []byte // len = capacity + GroupSize (for overflow probing)
	slots       []byte // len = capacity * SlotSize
	capacity    uint64
	count       uint64
	growAt      uint64 // 7/8 load factor threshold
	tombstones  uint64 // number of tombstone slots; compacted when > capacity/4
	maxCapacity uint64 // L2: hard cap on entries; 0 = unlimited
}

// SetMaxCapacity sets the hard cap on the number of entries. 0 = unlimited.
func (t *Table) SetMaxCapacity(max uint64) { t.maxCapacity = max }

// IsFull returns true if the table has reached its max capacity.
// When true, callers should evict before inserting.
func (t *Table) IsFull() bool {
	return t.maxCapacity > 0 && t.count >= t.maxCapacity
}

// NewTable creates a Swiss table with the given capacity (must be power of 2).
func NewTable(capacity uint64) *Table {
	if capacity < GroupSize {
		capacity = GroupSize
	}
	// Round up to power of 2
	capacity = nextPow2(capacity)

	ctrl := make([]byte, capacity+GroupSize)
	for i := range ctrl {
		ctrl[i] = CtrlEmpty
	}
	slots := make([]byte, capacity*SlotSize)

	return &Table{
		ctrl:     ctrl,
		slots:    slots,
		capacity: capacity,
		count:    0,
		growAt:   capacity * MaxLoadNumerator / MaxLoadDenominator,
	}
}

func nextPow2(v uint64) uint64 {
	if v == 0 {
		return 1
	}
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	v++
	return v
}

// Lookup finds an entry by key hash and returns (entry, slotIndex, found).
// Caller must verify actual key bytes match (hash collision possible).
func (t *Table) Lookup(keyHash uint64, keyLen uint16) (Entry, uint64, bool) {
	h2 := H2(keyHash)
	pos := H1(keyHash) & (t.capacity - 1)

	for {
		// Probe group of 16 ctrl bytes
		matches := t.matchGroup(pos, h2)
		for matches != 0 {
			bit := uint64(bits.TrailingZeros16(matches))
			idx := (pos + bit) & (t.capacity - 1)
			e := t.readSlot(idx)
			if e.KeyHash == keyHash && e.KeyLen == keyLen {
				return e, idx, true
			}
			matches &= matches - 1 // clear lowest bit
		}

		// Check if any empty in group → key definitely not present
		empties := t.matchEmptyGroup(pos)
		if empties != 0 {
			return Entry{}, 0, false
		}

		// Move to next group
		pos = (pos + GroupSize) & (t.capacity - 1)
	}
}

// Insert inserts or updates an entry. Returns (slotIndex, grew).
// If the key already exists (same keyHash + keyLen), it updates the value fields.
func (t *Table) Insert(keyHash uint64, e Entry) (uint64, bool) {
	// Check for existing entry
	_, existIdx, found := t.Lookup(keyHash, e.KeyLen)
	if found {
		t.writeSlot(existIdx, e)
		return existIdx, false
	}

	// Need to grow?
	grew := false
	if t.count >= t.growAt {
		t.grow()
		grew = true
	} else if t.tombstones > t.capacity/4 {
		t.compact()
	}

	// Find insertion position
	h2 := H2(keyHash)
	pos := H1(keyHash) & (t.capacity - 1)

	for {
		empties := t.matchEmptyOrTombstoneGroup(pos)
		if empties != 0 {
			bit := uint64(bits.TrailingZeros16(empties))
			idx := (pos + bit) & (t.capacity - 1)
			t.ctrl[idx] = h2
			// Mirror to overflow region if needed
			if idx < GroupSize {
				t.ctrl[t.capacity+idx] = h2
			}
			e.KeyHash = keyHash
			t.writeSlot(idx, e)
			t.count++
			return idx, grew
		}
		pos = (pos + GroupSize) & (t.capacity - 1)
	}
}

// Delete marks a slot as tombstone.
func (t *Table) Delete(keyHash uint64, keyLen uint16) bool {
	_, idx, found := t.Lookup(keyHash, keyLen)
	if !found {
		return false
	}
	t.ctrl[idx] = CtrlTombstone
	if idx < GroupSize {
		t.ctrl[t.capacity+idx] = CtrlTombstone
	}
	t.count--
	t.tombstones++
	return true
}

// ReadSlotAt reads the entry at a specific slot index.
func (t *Table) ReadSlotAt(idx uint64) Entry {
	return t.readSlot(idx)
}

// WriteSlotAt writes an entry at a specific slot index (for updating fields).
func (t *Table) WriteSlotAt(idx uint64, e Entry) {
	t.writeSlot(idx, e)
}

// Count returns the number of occupied slots.
func (t *Table) Count() uint64 {
	return atomic.LoadUint64(&t.count)
}

// Capacity returns the table capacity.
func (t *Table) Capacity() uint64 {
	return t.capacity
}

func (t *Table) matchGroup(pos uint64, h2 uint8) uint16 {
	var mask uint16
	for i := uint64(0); i < GroupSize; i++ {
		idx := (pos + i) & (t.capacity - 1)
		if t.ctrl[idx] == h2 {
			mask |= 1 << i
		}
	}
	return mask
}

func (t *Table) matchEmptyGroup(pos uint64) uint16 {
	var mask uint16
	for i := uint64(0); i < GroupSize; i++ {
		idx := (pos + i) & (t.capacity - 1)
		if t.ctrl[idx] == CtrlEmpty {
			mask |= 1 << i
		}
	}
	return mask
}

func (t *Table) matchEmptyOrTombstoneGroup(pos uint64) uint16 {
	var mask uint16
	for i := uint64(0); i < GroupSize; i++ {
		idx := (pos + i) & (t.capacity - 1)
		c := t.ctrl[idx]
		if c == CtrlEmpty || c == CtrlTombstone {
			mask |= 1 << i
		}
	}
	return mask
}

func (t *Table) readSlot(idx uint64) Entry {
	base := idx * SlotSize
	s := t.slots[base : base+SlotSize]
	return Entry{
		KeyHash:       binary.LittleEndian.Uint64(s[offKeyHash:]),
		KeyLen:        binary.LittleEndian.Uint16(s[offKeyLen:]),
		KeyOffset:     binary.LittleEndian.Uint64(s[offKeyOffset:]),
		ValueOffset:   binary.LittleEndian.Uint64(s[offValueOffset:]),
		ValueLen:      binary.LittleEndian.Uint32(s[offValueLen:]),
		TTL:           int64(binary.LittleEndian.Uint64(s[offTTL:])),
		RefCount:      binary.LittleEndian.Uint32(s[offRefCount:]),
		Flags:         binary.LittleEndian.Uint16(s[offFlags:]),
		ValueClassIdx: s[offValueClassIdx],
		KeyClassIdx:   s[offKeyClassIdx],
	}
}

func (t *Table) writeSlot(idx uint64, e Entry) {
	base := idx * SlotSize
	s := t.slots[base : base+SlotSize]
	binary.LittleEndian.PutUint64(s[offKeyHash:], e.KeyHash)
	binary.LittleEndian.PutUint16(s[offKeyLen:], e.KeyLen)
	binary.LittleEndian.PutUint64(s[offKeyOffset:], e.KeyOffset)
	binary.LittleEndian.PutUint64(s[offValueOffset:], e.ValueOffset)
	binary.LittleEndian.PutUint32(s[offValueLen:], e.ValueLen)
	binary.LittleEndian.PutUint64(s[offTTL:], uint64(e.TTL))
	binary.LittleEndian.PutUint32(s[offRefCount:], e.RefCount)
	binary.LittleEndian.PutUint16(s[offFlags:], e.Flags)
	s[offValueClassIdx] = e.ValueClassIdx
	s[offKeyClassIdx] = e.KeyClassIdx
}

func (t *Table) grow() {
	newCap := t.capacity * 2
	newTable := NewTable(newCap)

	for i := uint64(0); i < t.capacity; i++ {
		c := t.ctrl[i]
		if c != CtrlEmpty && c != CtrlTombstone {
			e := t.readSlot(i)
			newTable.Insert(e.KeyHash, e)
		}
	}

	t.ctrl = newTable.ctrl
	t.slots = newTable.slots
	t.capacity = newTable.capacity
	t.growAt = newTable.growAt
	t.tombstones = 0
	// count stays the same
}

// compact rehashes at the same capacity to reclaim tombstone slots.
func (t *Table) compact() {
	newTable := NewTable(t.capacity)

	for i := uint64(0); i < t.capacity; i++ {
		c := t.ctrl[i]
		if c != CtrlEmpty && c != CtrlTombstone {
			e := t.readSlot(i)
			newTable.Insert(e.KeyHash, e)
		}
	}

	t.ctrl = newTable.ctrl
	t.slots = newTable.slots
	t.tombstones = 0
	// capacity, growAt, count stay the same
}

// Tombstones returns the number of tombstone slots.
func (t *Table) Tombstones() uint64 {
	return t.tombstones
}

// Iter calls fn for each occupied slot. If fn returns false, iteration stops.
func (t *Table) Iter(fn func(idx uint64, e Entry) bool) {
	for i := uint64(0); i < t.capacity; i++ {
		c := t.ctrl[i]
		if c != CtrlEmpty && c != CtrlTombstone {
			if !fn(i, t.readSlot(i)) {
				return
			}
		}
	}
}

// IterFrom calls fn for each occupied slot starting from startIdx (inclusive).
// If fn returns false, iteration stops and returns the index where it stopped.
// Returns (nextIdx, done): nextIdx is the slot after the last visited, done is
// true if the full table was scanned from startIdx to capacity.
func (t *Table) IterFrom(startIdx uint64, fn func(idx uint64, e Entry) bool) (uint64, bool) {
	for i := startIdx; i < t.capacity; i++ {
		c := t.ctrl[i]
		if c != CtrlEmpty && c != CtrlTombstone {
			if !fn(i, t.readSlot(i)) {
				return i + 1, false
			}
		}
	}
	return t.capacity, true
}

// ClearFlagAll clears the given flag mask from all occupied slots.
func (t *Table) ClearFlagAll(mask uint16) {
	invMask := ^mask
	for i := uint64(0); i < t.capacity; i++ {
		c := t.ctrl[i]
		if c != CtrlEmpty && c != CtrlTombstone {
			e := t.readSlot(i)
			if e.Flags&mask != 0 {
				e.Flags &= invMask
				t.writeSlot(i, e)
			}
		}
	}
}
