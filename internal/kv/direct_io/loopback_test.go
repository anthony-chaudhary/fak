package directio_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	directio "github.com/anthony-chaudhary/fak/internal/kv/direct_io"
)

// TestAlignedBuffer verifies that AlignedBuffer returns a memory address strictly aligned to 4096 bytes.
func TestAlignedBuffer(t *testing.T) {
	sizes := []int{1, 64, 512, 1024, 4096, 8192, 16384}
	for _, sz := range sizes {
		buf := directio.AlignedBuffer(sz)
		if len(buf) < sz {
			t.Fatalf("AlignedBuffer(%d) length = %d; want at least %d", sz, len(buf), sz)
		}
		addr := uintptr(unsafe.Pointer(&buf[0]))
		if addr%uintptr(directio.BlockSize) != 0 {
			t.Fatalf("AlignedBuffer(%d) address %x is not 4096-byte aligned (remainder %d)", sz, addr, addr%uintptr(directio.BlockSize))
		}
	}
}

// TestLoopbackKVStoreColdWriteAndDirectIORead satisfies requirement 3:
// Unit test demonstrating cold write and direct-I/O read of KV token blocks with byte-exact verification.
func TestLoopbackKVStoreColdWriteAndDirectIORead(t *testing.T) {
	tmpDir := t.TempDir()
	storePath := filepath.Join(tmpDir, "loopback_test.kv")

	store, err := directio.NewLoopbackKVStore(storePath, 64)
	if err != nil {
		t.Fatalf("failed to create loopback store: %v", err)
	}
	defer store.Close()

	// 1. Allocate block with KV cache metadata
	meta := directio.BlockMetadata{
		SessionID:   "session-metal-rocm-001",
		Turn:        1,
		TokenOffset: 0,
		NumTokens:   128,
		Layer:       0,
	}
	blockID, err := store.AllocateBlock(meta)
	if err != nil {
		t.Fatalf("allocate block: %v", err)
	}
	if blockID != 0 {
		t.Fatalf("first block id = %d; want 0", blockID)
	}

	// 2. Generate pseudo-random KV tensor token bytes (exact 4096-byte block)
	kvTokenBytes := make([]byte, directio.BlockSize)
	for i := 0; i < len(kvTokenBytes); i++ {
		kvTokenBytes[i] = byte((i*37 + 13) % 256)
	}

	// 3. Cold write the block to disk
	if err := store.WriteBlockWithMeta(blockID, meta, kvTokenBytes); err != nil {
		t.Fatalf("cold write block %d: %v", blockID, err)
	}

	// 4. Direct-I/O read via ReadBlock with byte-exact verification
	readBytes, err := store.ReadBlock(blockID)
	if err != nil {
		t.Fatalf("direct-io read block %d: %v", blockID, err)
	}
	if len(readBytes) != len(kvTokenBytes) {
		t.Fatalf("read length = %d; want %d", len(readBytes), len(kvTokenBytes))
	}
	if !bytes.Equal(readBytes, kvTokenBytes) {
		t.Fatalf("byte mismatch between cold write and direct-io read on block %d", blockID)
	}

	// 5. Direct-I/O read via ReadBlockDirect into an aligned destination buffer
	directDst := directio.AlignedBuffer(directio.BlockSize)
	n, err := store.ReadBlockDirect(blockID, directDst)
	if err != nil {
		t.Fatalf("ReadBlockDirect: %v", err)
	}
	if n != directio.BlockSize {
		t.Fatalf("ReadBlockDirect read %d bytes; want %d", n, directio.BlockSize)
	}
	if !bytes.Equal(directDst[:directio.BlockSize], kvTokenBytes) {
		t.Fatalf("ReadBlockDirect byte mismatch against written KV token block")
	}

	// 6. Verify metadata was correctly recorded
	retrievedMeta, ok := store.GetMetadata(blockID)
	if !ok {
		t.Fatalf("failed to retrieve metadata for block %d", blockID)
	}
	if retrievedMeta.SessionID != meta.SessionID {
		t.Errorf("session ID = %q, want %q", retrievedMeta.SessionID, meta.SessionID)
	}
	if retrievedMeta.Turn != meta.Turn {
		t.Errorf("turn = %d, want %d", retrievedMeta.Turn, meta.Turn)
	}
	if retrievedMeta.NumTokens != meta.NumTokens {
		t.Errorf("num tokens = %d, want %d", retrievedMeta.NumTokens, meta.NumTokens)
	}
	if retrievedMeta.BytesUsed != directio.BlockSize {
		t.Errorf("bytes used = %d, want %d", retrievedMeta.BytesUsed, directio.BlockSize)
	}
	if retrievedMeta.Checksum == 0 {
		t.Errorf("expected non-zero checksum for written block")
	}

	// 7. Verify partial page write (e.g. 1536 bytes) with byte-exact verification
	partialBytes := make([]byte, 1536)
	_, _ = rand.Read(partialBytes)
	blockID2, err := store.AllocateBlock(directio.BlockMetadata{
		SessionID:   "session-metal-rocm-001",
		Turn:        2,
		TokenOffset: 128,
		NumTokens:   48,
	})
	if err != nil {
		t.Fatalf("allocate block 2: %v", err)
	}
	if err := store.WriteBlock(blockID2, partialBytes); err != nil {
		t.Fatalf("write partial block: %v", err)
	}
	readPartial, err := store.ReadBlock(blockID2)
	if err != nil {
		t.Fatalf("read partial block: %v", err)
	}
	if len(readPartial) != len(partialBytes) {
		t.Fatalf("read partial length = %d; want %d", len(readPartial), len(partialBytes))
	}
	if !bytes.Equal(readPartial, partialBytes) {
		t.Fatalf("partial block byte mismatch")
	}
}

// TestMultiTurnResumeSubSecond verifies requirement 1:
// Provide sub-second multi-turn resume serialization and deserialization.
func TestMultiTurnResumeSubSecond(t *testing.T) {
	tmpDir := t.TempDir()
	storePath := filepath.Join(tmpDir, "multiturn_resume.kv")

	store, err := directio.NewLoopbackKVStore(storePath, 256)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	sessionID := "session-multiturn-agent-42"
	numTurns := 8
	blocksPerTurn := 4

	// Populate multiple turns of KV blocks
	tokenOffset := 0
	for turn := 1; turn <= numTurns; turn++ {
		blocks, err := store.AllocateBlocks(blocksPerTurn, sessionID, turn)
		if err != nil {
			t.Fatalf("turn %d allocate: %v", turn, err)
		}
		for i, bid := range blocks {
			slabData := make([]byte, directio.BlockSize)
			slabData[0] = byte(turn)
			slabData[1] = byte(i)
			meta := directio.BlockMetadata{
				SessionID:   sessionID,
				Turn:        turn,
				TokenOffset: tokenOffset,
				NumTokens:   64,
				Layer:       i,
			}
			if err := store.WriteBlockWithMeta(bid, meta, slabData); err != nil {
				t.Fatalf("turn %d write block %d: %v", turn, bid, err)
			}
			tokenOffset += 64
		}
	}

	// Sub-second serialization benchmark and verification
	startSer := time.Now()
	resumeBytes, err := store.SerializeResume(sessionID)
	serDuration := time.Since(startSer)
	if err != nil {
		t.Fatalf("SerializeResume: %v", err)
	}
	if serDuration >= 1*time.Second {
		t.Fatalf("serialization took %v; want sub-second (< 1s)", serDuration)
	}
	t.Logf("multi-turn resume serialization took %v for %d blocks", serDuration, numTurns*blocksPerTurn)

	// Create a second store simulating cold process restart
	resumeStorePath := filepath.Join(tmpDir, "resume_target.kv")
	targetStore, err := directio.NewLoopbackKVStore(resumeStorePath, 256)
	if err != nil {
		t.Fatalf("open target store: %v", err)
	}
	defer targetStore.Close()

	// Sub-second deserialization benchmark and verification
	startDeser := time.Now()
	manifest, err := targetStore.DeserializeResume(resumeBytes)
	deserDuration := time.Since(startDeser)
	if err != nil {
		t.Fatalf("DeserializeResume: %v", err)
	}
	if deserDuration >= 1*time.Second {
		t.Fatalf("deserialization took %v; want sub-second (< 1s)", deserDuration)
	}
	t.Logf("multi-turn resume deserialization took %v", deserDuration)

	if manifest.SessionID != sessionID {
		t.Errorf("manifest sessionID = %q, want %q", manifest.SessionID, sessionID)
	}
	if manifest.Turn != numTurns {
		t.Errorf("manifest turn = %d, want %d", manifest.Turn, numTurns)
	}
	expectedBlocks := numTurns * blocksPerTurn
	if len(manifest.Blocks) != expectedBlocks {
		t.Errorf("manifest blocks count = %d, want %d", len(manifest.Blocks), expectedBlocks)
	}

	// Save and load resume manifest to file
	resumeFilePath := filepath.Join(tmpDir, "manifest.json")
	if err := store.SaveResumeToFile(sessionID, resumeFilePath); err != nil {
		t.Fatalf("SaveResumeToFile: %v", err)
	}
	loadedManifest, err := targetStore.LoadResumeFromFile(resumeFilePath)
	if err != nil {
		t.Fatalf("LoadResumeFromFile: %v", err)
	}
	if loadedManifest.TotalTokens != manifest.TotalTokens {
		t.Errorf("loaded manifest tokens = %d, want %d", loadedManifest.TotalTokens, manifest.TotalTokens)
	}
}

// TestLoopbackKVStoreEvictAndPrune verifies block freeing and eviction mechanisms.
func TestLoopbackKVStoreEvictAndPrune(t *testing.T) {
	tmpDir := t.TempDir()
	storePath := filepath.Join(tmpDir, "evict_test.kv")

	store, err := directio.NewLoopbackKVStore(storePath, 16)
	if err != nil {
		t.Fatalf("create store: %v", err)
	}
	defer store.Close()

	for i := 0; i < 8; i++ {
		bid, err := store.AllocateBlock(directio.BlockMetadata{
			SessionID: "sess-1",
			Turn:      1,
		})
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		data := make([]byte, 1024)
		data[0] = byte(i)
		_ = store.WriteBlock(bid, data)
	}

	if store.AllocatedCount() != 8 {
		t.Fatalf("allocated count = %d, want 8", store.AllocatedCount())
	}

	// Evict 3 oldest blocks
	freed, err := store.EvictOldest(3)
	if err != nil {
		t.Fatalf("EvictOldest: %v", err)
	}
	if freed != 3 {
		t.Fatalf("freed = %d, want 3", freed)
	}
	if store.AllocatedCount() != 5 {
		t.Fatalf("remaining count = %d, want 5", store.AllocatedCount())
	}

	// Verify KVEvictor interface implementation
	ctx := context.Background()
	freedKVEvict, err := store.EvictKV(ctx)
	if err != nil {
		t.Fatalf("EvictKV: %v", err)
	}
	if freedKVEvict != 5 {
		t.Fatalf("EvictKV freed %d blocks; want 5", freedKVEvict)
	}
	if store.AllocatedCount() != 0 {
		t.Fatalf("all blocks should be freed, remaining = %d", store.AllocatedCount())
	}

	// FreeSessionBlocks
	bid, _ := store.AllocateBlock(directio.BlockMetadata{SessionID: "target-sess"})
	_ = store.WriteBlock(bid, make([]byte, 100))
	freedSess, err := store.FreeSessionBlocks("target-sess")
	if err != nil {
		t.Fatalf("FreeSessionBlocks: %v", err)
	}
	if freedSess != 1 {
		t.Fatalf("FreeSessionBlocks freed = %d, want 1", freedSess)
	}
}
