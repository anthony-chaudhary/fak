package ctxmmu

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestCOWPageTable_CandidatePoolClearsEntireBuffer(t *testing.T) {
	table := NewCOWPageTable()
	buffer := make([]byte, table.blockSize)
	for i := range buffer {
		buffer[i] = 0xa5
	}
	dirty := &PageBlock{
		ID:         7,
		Capacity:   table.blockCapacity,
		Buffer:     buffer,
		Tokens:     []int{123, 456},
		TokenCount: 2,
		immutable:  true,
	}
	dirty.refCount.Store(0)
	// The pool is unused. Supplying its first object through New avoids relying
	// on sync.Pool retaining a Put across a GC or a scheduler migration.
	table.blockPool.New = func() any { return dirty }

	got := table.allocBlockLockFree()
	if got != dirty || &got.Buffer[0] != &buffer[0] {
		t.Fatal("candidate allocation must reuse the supplied block and buffer")
	}
	if got.ID != 7 || got.Capacity != table.blockCapacity || len(got.Buffer) != int(table.blockSize) {
		t.Fatal("candidate allocation changed block identity or capacity")
	}
	if got.RefCount() != 1 || got.TokenCount != 0 || len(got.Tokens) != 0 || got.immutable {
		t.Fatalf("block state was not reset: refs=%d count=%d tokens=%v immutable=%v",
			got.RefCount(), got.TokenCount, got.Tokens, got.immutable)
	}
	for i, b := range got.Buffer {
		if b != 0 {
			t.Fatalf("stale byte at offset %d: %#x", i, b)
		}
	}
}
