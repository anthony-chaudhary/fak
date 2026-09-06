package l3kv

import (
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
)

// TestChunkStoreAlignedAppends proves that multiple sequential Put() calls
// pack into fixed-size aligned chunk files with exact offset recovery on Get().
func TestChunkStoreAlignedAppends(t *testing.T) {
	dir := t.TempDir()
	chunkSize := uint64(64 * 1024) // 64 KiB
	alignment := uint64(4096)      // 4 KiB

	store, err := NewChunkStore(ChunkStoreConfig{
		Dir:        dir,
		ChunkSize:  chunkSize,
		Alignment:  alignment,
		PadToChunk: true,
	})
	if err != nil {
		t.Fatalf("NewChunkStore: %v", err)
	}
	defer store.Close()

	ctx := context.Background()

	// Define 4 payloads that fit within 64 KiB:
	// 8 KiB + 12 KiB + 16 KiB + 4 KiB = 40 KiB <= 64 KiB
	sizes := []int{8192, 12288, 16384, 4096}
	payloads := make(map[string][]byte)
	var expectedOffset uint64

	for i, size := range sizes {
		digest := fmt.Sprintf("span-%04d", i)
		data := make([]byte, size)
		if _, err := rand.Read(data); err != nil {
			t.Fatalf("rand.Read: %v", err)
		}
		payloads[digest] = data

		if err := store.Put(ctx, digest, data); err != nil {
			t.Fatalf("Put(%s): %v", digest, err)
		}

		// Verify location in index manifest immediately
		loc, ok := store.Location(digest)
		if !ok {
			t.Fatalf("Location(%s) not found after Put", digest)
		}
		if loc.ChunkID != 0 {
			t.Errorf("expected ChunkID 0, got %d", loc.ChunkID)
		}
		if loc.Offset != expectedOffset {
			t.Errorf("expected Offset %d, got %d", expectedOffset, loc.Offset)
		}
		if loc.Length != uint64(size) {
			t.Errorf("expected Length %d, got %d", size, loc.Length)
		}

		// Test reading back from active in-memory buffer before flush
		got, err := store.Get(ctx, digest)
		if err != nil {
			t.Fatalf("Get(%s) from active buffer: %v", digest, err)
		}
		if !bytes.Equal(got, data) {
			t.Fatalf("Get(%s) active buffer mismatch", digest)
		}

		expectedOffset += uint64(size)
	}

	// Flush the active chunk to disk
	if err := store.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	// Verify chunk file exists and is fixed-size (64 KiB)
	chunkFile := store.ChunkPath(0)
	info, err := os.Stat(chunkFile)
	if err != nil {
		t.Fatalf("stat chunk file: %v", err)
	}
	if uint64(info.Size()) != chunkSize {
		t.Fatalf("expected chunk file size %d, got %d", chunkSize, info.Size())
	}
	if uint64(info.Size())%alignment != 0 {
		t.Fatalf("chunk file size %d is not aligned to %d bytes", info.Size(), alignment)
	}

	// Verify alignment methods
	if err := store.VerifyAlignment(0); err != nil {
		t.Fatalf("VerifyAlignment(0): %v", err)
	}
	if err := store.VerifyAllAlignments(); err != nil {
		t.Fatalf("VerifyAllAlignments: %v", err)
	}

	// Direct raw disk verification: prove exact offset recovery from the chunk file
	rawChunk, err := os.ReadFile(chunkFile)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", chunkFile, err)
	}

	for digest, data := range payloads {
		loc, ok := store.Location(digest)
		if !ok {
			t.Fatalf("Location(%s) missing after sync", digest)
		}

		// Slice directly from the raw chunk file bytes
		recovered := rawChunk[loc.Offset : loc.Offset+loc.Length]
		if !bytes.Equal(recovered, data) {
			t.Fatalf("raw disk chunk byte mismatch for %s at offset %d len %d", digest, loc.Offset, loc.Length)
		}

		// Verify Get() reads from committed disk file
		got, err := store.Get(ctx, digest)
		if err != nil {
			t.Fatalf("Get(%s) from disk: %v", digest, err)
		}
		if !bytes.Equal(got, data) {
			t.Fatalf("Get(%s) disk read mismatch", digest)
		}
	}
}

// TestChunkStoreIndexRecoveryAndBoundaryWrapping verifies:
// 1. Boundary wrapping when payloads exceed chunk size across multiple chunks.
// 2. Exact index recovery and persistence across store instances.
func TestChunkStoreIndexRecoveryAndBoundaryWrapping(t *testing.T) {
	dir := t.TempDir()
	chunkSize := uint64(32 * 1024) // 32 KiB
	alignment := uint64(4096)      // 4 KiB

	cfg := ChunkStoreConfig{
		Dir:        dir,
		ChunkSize:  chunkSize,
		Alignment:  alignment,
		PadToChunk: true,
	}

	store1, err := NewChunkStore(cfg)
	if err != nil {
		t.Fatalf("NewChunkStore: %v", err)
	}

	ctx := context.Background()

	// Span 0: 20 KiB -> fits in Chunk 0 (remaining: 12 KiB)
	// Span 1: 20 KiB -> doesn't fit in Chunk 0 (20 > 12) -> triggers boundary wrap to Chunk 1!
	// Span 2: 10 KiB -> fits in Chunk 1 (20 + 10 = 30 KiB, remaining: 2 KiB)
	// Span 3: 16 KiB -> doesn't fit in Chunk 1 (16 > 2) -> triggers boundary wrap to Chunk 2!
	testSpans := []struct {
		key      string
		size     int
		expected ChunkExpected
	}{
		{"span-0", 20 * 1024, ChunkExpected{chunkID: 0, offset: 0}},
		{"span-1", 20 * 1024, ChunkExpected{chunkID: 1, offset: 0}},
		{"span-2", 10 * 1024, ChunkExpected{chunkID: 1, offset: 20 * 1024}},
		{"span-3", 16 * 1024, ChunkExpected{chunkID: 2, offset: 0}},
	}

	payloads := make(map[string][]byte)
	for _, ts := range testSpans {
		data := make([]byte, ts.size)
		if _, err := rand.Read(data); err != nil {
			t.Fatalf("rand.Read: %v", err)
		}
		payloads[ts.key] = data

		if err := store1.Put(ctx, ts.key, data); err != nil {
			t.Fatalf("Put(%s): %v", ts.key, err)
		}

		loc, ok := store1.Location(ts.key)
		if !ok {
			t.Fatalf("Location(%s) not found", ts.key)
		}
		if loc.ChunkID != ts.expected.chunkID {
			t.Errorf("[%s] expected ChunkID %d, got %d", ts.key, ts.expected.chunkID, loc.ChunkID)
		}
		if loc.Offset != ts.expected.offset {
			t.Errorf("[%s] expected Offset %d, got %d", ts.key, ts.expected.offset, loc.Offset)
		}
		if loc.Length != uint64(ts.size) {
			t.Errorf("[%s] expected Length %d, got %d", ts.key, ts.size, loc.Length)
		}
	}

	// Close store1 to ensure all pending chunks and index manifest are committed to disk
	if err := store1.Close(); err != nil {
		t.Fatalf("store1.Close(): %v", err)
	}

	// Verify on-disk chunk files: chunks 0, 1, and 2 must all exist and be exactly 32 KiB
	for chunkID := uint64(0); chunkID <= 2; chunkID++ {
		p := store1.ChunkPath(chunkID)
		info, err := os.Stat(p)
		if err != nil {
			t.Fatalf("stat chunk %d (%s): %v", chunkID, p, err)
		}
		if uint64(info.Size()) != chunkSize {
			t.Errorf("chunk %d size expected %d, got %d", chunkID, chunkSize, info.Size())
		}
		if uint64(info.Size())%alignment != 0 {
			t.Errorf("chunk %d size %d not aligned to %d", chunkID, info.Size(), alignment)
		}
	}

	// --- Index Recovery: open a fresh store2 instance on the same directory ---
	store2, err := NewChunkStore(cfg)
	if err != nil {
		t.Fatalf("reopen NewChunkStore: %v", err)
	}
	defer store2.Close()

	// Verify all recovered index locations and payloads
	index := store2.Index()
	if len(index) != len(testSpans) {
		t.Fatalf("recovered index has %d entries, expected %d", len(index), len(testSpans))
	}

	for _, ts := range testSpans {
		loc, ok := index[ts.key]
		if !ok {
			t.Fatalf("recovered index missing %s", ts.key)
		}
		if loc.ChunkID != ts.expected.chunkID {
			t.Errorf("recovered [%s] ChunkID: expected %d, got %d", ts.key, ts.expected.chunkID, loc.ChunkID)
		}
		if loc.Offset != ts.expected.offset {
			t.Errorf("recovered [%s] Offset: expected %d, got %d", ts.key, ts.expected.offset, loc.Offset)
		}

		got, err := store2.Get(ctx, ts.key)
		if err != nil {
			t.Fatalf("store2.Get(%s): %v", ts.key, err)
		}
		if !bytes.Equal(got, payloads[ts.key]) {
			t.Fatalf("store2.Get(%s) payload mismatch", ts.key)
		}
	}

	// Append a new span in store2 to prove active chunk starts at the next chunkID (3)
	newKey := "span-new-after-recovery"
	newData := []byte("new payload after recovery")
	if err := store2.Put(ctx, newKey, newData); err != nil {
		t.Fatalf("store2.Put(%s): %v", newKey, err)
	}

	locNew, ok := store2.Location(newKey)
	if !ok {
		t.Fatalf("Location(%s) not found in store2", newKey)
	}
	if locNew.ChunkID != 3 {
		t.Errorf("expected new chunkID 3, got %d", locNew.ChunkID)
	}
	if locNew.Offset != 0 {
		t.Errorf("expected offset 0, got %d", locNew.Offset)
	}

	gotNew, err := store2.Get(ctx, newKey)
	if err != nil {
		t.Fatalf("store2.Get(%s): %v", newKey, err)
	}
	if !bytes.Equal(gotNew, newData) {
		t.Fatalf("store2.Get(%s) payload mismatch", newKey)
	}

	// Verify RecoverIndex() method explicitly reloads index
	if err := store2.RecoverIndex(); err != nil {
		t.Fatalf("RecoverIndex: %v", err)
	}
}

type ChunkExpected struct {
	chunkID uint64
	offset  uint64
}

// TestChunkStoreAdapter verifies compatibility with the l3kv.Store interface.
func TestChunkStoreAdapter(t *testing.T) {
	dir := t.TempDir()
	cs, err := NewDefaultChunkStore(dir)
	if err != nil {
		t.Fatalf("NewDefaultChunkStore: %v", err)
	}
	defer cs.Close()

	var store Store = cs.AsStore()
	ctx := context.Background()

	key := "test-store-adapter-key"
	val := []byte("payload-through-store-interface")

	if err := store.Put(ctx, key, val); err != nil {
		t.Fatalf("store.Put: %v", err)
	}

	got, found, err := store.Get(ctx, key)
	if err != nil {
		t.Fatalf("store.Get: %v", err)
	}
	if !found {
		t.Fatalf("expected found=true, got false")
	}
	if !bytes.Equal(got, val) {
		t.Fatalf("got %q, want %q", got, val)
	}

	// Clean miss test
	missPayload, foundMiss, err := store.Get(ctx, "non-existent-key")
	if err != nil {
		t.Fatalf("store.Get miss returned error: %v", err)
	}
	if foundMiss {
		t.Fatalf("expected found=false for miss, got true")
	}
	if missPayload != nil {
		t.Fatalf("expected nil payload for miss, got %v", missPayload)
	}
}

// TestChunkStoreValidationAndErrors verifies error paths.
func TestChunkStoreValidationAndErrors(t *testing.T) {
	dir := t.TempDir()
	chunkSize := uint64(8192)
	alignment := uint64(4096)

	// Unaligned ChunkSize validation
	_, err := NewChunkStore(ChunkStoreConfig{
		Dir:       dir,
		ChunkSize: 5000,
		Alignment: alignment,
	})
	if err == nil {
		t.Fatal("expected error for unaligned ChunkSize, got nil")
	}

	store, err := NewChunkStore(ChunkStoreConfig{
		Dir:       dir,
		ChunkSize: chunkSize,
		Alignment: alignment,
	})
	if err != nil {
		t.Fatalf("NewChunkStore: %v", err)
	}
	defer store.Close()

	ctx := context.Background()

	// Empty digest
	if err := store.Put(ctx, "", []byte("test")); err == nil {
		t.Fatal("expected error on empty digest Put, got nil")
	}
	if _, err := store.Get(ctx, ""); err == nil {
		t.Fatal("expected error on empty digest Get, got nil")
	}

	// Payload exceeds chunk size
	huge := make([]byte, chunkSize+10)
	if err := store.Put(ctx, "huge", huge); !errors.Is(err, ErrPayloadTooLarge) {
		t.Fatalf("expected ErrPayloadTooLarge, got %v", err)
	}

	// Key not found
	if _, err := store.Get(ctx, "does-not-exist"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}

	// Closed store behavior
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := store.Put(ctx, "k", []byte("v")); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed on Put, got %v", err)
	}
	if _, err := store.Get(ctx, "k"); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed on Get, got %v", err)
	}
}

func TestChunkStoreConcurrentAccess(t *testing.T) {
	dir := t.TempDir()
	store, err := NewChunkStore(ChunkStoreConfig{
		Dir:        dir,
		ChunkSize:  64 * 1024,
		Alignment:  4096,
		PadToChunk: false,
	})
	if err != nil {
		t.Fatalf("NewChunkStore: %v", err)
	}
	defer store.Close()

	ctx := context.Background()
	const goroutines = 8
	const opsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < opsPerGoroutine; i++ {
				key := fmt.Sprintf("worker-%d-span-%d", workerID, i)
				data := []byte(fmt.Sprintf("payload-data-%d-%d", workerID, i))

				if err := store.Put(ctx, key, data); err != nil {
					t.Errorf("worker %d Put failed: %v", workerID, err)
					return
				}

				got, err := store.Get(ctx, key)
				if err != nil {
					t.Errorf("worker %d Get failed: %v", workerID, err)
					return
				}
				if !bytes.Equal(got, data) {
					t.Errorf("worker %d Get mismatch: got %q, want %q", workerID, got, data)
					return
				}

				_, _ = store.Location(key)
				_ = store.Index()
			}
		}(g)
	}

	wg.Wait()
}
