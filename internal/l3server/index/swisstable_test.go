package index

import (
	"fmt"
	"testing"
)

func TestSwissTableBasicOps(t *testing.T) {
	tbl := NewTable(64)
	key := []byte("test-key-1")
	h := KeyHash(key)
	e := Entry{
		KeyHash:     h,
		KeyLen:      uint16(len(key)),
		KeyOffset:   100,
		ValueOffset: 200,
		ValueLen:    42,
	}
	idx, grew := tbl.Insert(h, e)
	if grew {
		t.Error("unexpected grow on first insert")
	}
	_ = idx

	found, _, ok := tbl.Lookup(h, uint16(len(key)))
	if !ok {
		t.Fatal("expected to find key")
	}
	if found.KeyHash != h {
		t.Errorf("hash mismatch: got %x, want %x", found.KeyHash, h)
	}
	if found.ValueLen != 42 {
		t.Errorf("value len mismatch: got %d, want 42", found.ValueLen)
	}
	if tbl.Count() != 1 {
		t.Errorf("count: got %d, want 1", tbl.Count())
	}

	deleted := tbl.Delete(h, uint16(len(key)))
	if !deleted {
		t.Error("expected delete to return true")
	}
	if tbl.Count() != 0 {
		t.Errorf("count after delete: got %d, want 0", tbl.Count())
	}
	_, _, ok = tbl.Lookup(h, uint16(len(key)))
	if ok {
		t.Error("found key after delete")
	}
}

func TestSwissTableGrow(t *testing.T) {
	tbl := NewTable(16)
	for i := 0; i < 20; i++ {
		key := []byte(fmt.Sprintf("key-%04d", i))
		h := KeyHash(key)
		tbl.Insert(h, Entry{KeyHash: h, KeyLen: uint16(len(key)), ValueLen: uint32(i)})
	}
	if tbl.Count() != 20 {
		t.Errorf("count: got %d, want 20", tbl.Count())
	}
	for i := 0; i < 20; i++ {
		key := []byte(fmt.Sprintf("key-%04d", i))
		h := KeyHash(key)
		found, _, ok := tbl.Lookup(h, uint16(len(key)))
		if !ok {
			t.Errorf("lost key %d after grow", i)
			continue
		}
		if found.ValueLen != uint32(i) {
			t.Errorf("key %d: value len got %d, want %d", i, found.ValueLen, i)
		}
	}
}

func TestSwissTableTombstoneCompaction(t *testing.T) {
	// Insert 1000 keys, delete 900, verify tombstone compaction works
	tbl := NewTable(2048) // big enough to avoid grow
	const insertCount = 1000
	const deleteCount = 900

	for i := 0; i < insertCount; i++ {
		key := []byte(fmt.Sprintf("tomb-key-%04d", i))
		h := KeyHash(key)
		tbl.Insert(h, Entry{KeyHash: h, KeyLen: uint16(len(key)), ValueLen: uint32(i)})
	}
	if tbl.Count() != insertCount {
		t.Fatalf("count after insert: got %d, want %d", tbl.Count(), insertCount)
	}

	// Delete 900 entries, creating tombstones
	for i := 0; i < deleteCount; i++ {
		key := []byte(fmt.Sprintf("tomb-key-%04d", i))
		h := KeyHash(key)
		tbl.Delete(h, uint16(len(key)))
	}
	if tbl.Tombstones() != deleteCount {
		t.Fatalf("tombstones: got %d, want %d", tbl.Tombstones(), deleteCount)
	}
	if tbl.Count() != insertCount-deleteCount {
		t.Fatalf("count after deletes: got %d, want %d", tbl.Count(), insertCount-deleteCount)
	}

	// Insert a new key — should trigger compaction (tombstones > capacity/4)
	triggerKey := []byte("trigger-compact")
	triggerHash := KeyHash(triggerKey)
	tbl.Insert(triggerHash, Entry{KeyHash: triggerHash, KeyLen: uint16(len(triggerKey)), ValueLen: 999})

	if tbl.Tombstones() != 0 {
		t.Errorf("tombstones after compaction: got %d, want 0", tbl.Tombstones())
	}

	// Verify surviving keys are still accessible
	for i := deleteCount; i < insertCount; i++ {
		key := []byte(fmt.Sprintf("tomb-key-%04d", i))
		h := KeyHash(key)
		found, _, ok := tbl.Lookup(h, uint16(len(key)))
		if !ok {
			t.Errorf("lost key %d after compaction", i)
			continue
		}
		if found.ValueLen != uint32(i) {
			t.Errorf("key %d: value len got %d, want %d", i, found.ValueLen, i)
		}
	}

	// Verify trigger key is accessible
	found, _, ok := tbl.Lookup(triggerHash, uint16(len(triggerKey)))
	if !ok {
		t.Fatal("trigger key not found after compaction")
	}
	if found.ValueLen != 999 {
		t.Errorf("trigger key value len: got %d, want 999", found.ValueLen)
	}
}

func TestSwissTableUpdate(t *testing.T) {
	tbl := NewTable(64)
	key := []byte("update-key")
	h := KeyHash(key)
	tbl.Insert(h, Entry{KeyHash: h, KeyLen: uint16(len(key)), ValueLen: 10})
	tbl.Insert(h, Entry{KeyHash: h, KeyLen: uint16(len(key)), ValueLen: 20})
	if tbl.Count() != 1 {
		t.Errorf("count after update: got %d, want 1", tbl.Count())
	}
	found, _, ok := tbl.Lookup(h, uint16(len(key)))
	if !ok {
		t.Fatal("key not found after update")
	}
	if found.ValueLen != 20 {
		t.Errorf("value len after update: got %d, want 20", found.ValueLen)
	}
}

func TestSwissTableMaxCapacity(t *testing.T) {
	tbl := NewTable(64)
	tbl.SetMaxCapacity(5)

	// Insert 5 keys — should succeed
	for i := 0; i < 5; i++ {
		key := []byte(fmt.Sprintf("cap-key-%d", i))
		h := KeyHash(key)
		tbl.Insert(h, Entry{KeyHash: h, KeyLen: uint16(len(key)), ValueLen: uint32(i)})
	}
	if tbl.Count() != 5 {
		t.Fatalf("count: got %d, want 5", tbl.Count())
	}

	// IsFull should return true
	if !tbl.IsFull() {
		t.Error("expected IsFull=true at capacity")
	}

	// Update an existing key — should succeed (not a new entry)
	key0 := []byte("cap-key-0")
	h0 := KeyHash(key0)
	tbl.Insert(h0, Entry{KeyHash: h0, KeyLen: uint16(len(key0)), ValueLen: 999})
	if tbl.Count() != 5 {
		t.Fatalf("count after update: got %d, want 5", tbl.Count())
	}

	// Delete one — IsFull should become false
	tbl.Delete(h0, uint16(len(key0)))
	if tbl.IsFull() {
		t.Error("expected IsFull=false after delete")
	}
	if tbl.Count() != 4 {
		t.Fatalf("count after delete: got %d, want 4", tbl.Count())
	}

	// SetMaxCapacity(0) = unlimited
	tbl.SetMaxCapacity(0)
	if tbl.IsFull() {
		t.Error("expected IsFull=false with unlimited capacity")
	}
}
