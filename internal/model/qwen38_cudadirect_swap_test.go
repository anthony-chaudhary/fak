package model

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestQwen38CUDADirectSwapRoundTrip(t *testing.T) {
	slabCfg := compute.CUDADirectStorageConfig{
		NodeID:      0,
		BlockSize:   64 * 1024,
		TotalBlocks: 512,
	}
	slab, err := compute.NewCUDADirectStorageMemorySlab(slabCfg)
	if err != nil {
		t.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}
	hmm, err := compute.NewHierarchicalMemoryManager(compute.HierarchicalMemoryConfig{}, slab)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	swapper, err := NewQwen38CUDADirectSwapper(ModelArchQwen38_27B, hmm, slab, 16)
	if err != nil {
		t.Fatalf("NewQwen38CUDADirectSwapper failed: %v", err)
	}

	// 4 KV pages, 1024 bytes each, with distinct byte patterns
	kvPages := make([][]byte, 4)
	for i := range kvPages {
		kvPages[i] = make([]byte, 1024)
		for j := range kvPages[i] {
			kvPages[i][j] = byte((i*17 + j*31) & 0xFF)
		}
	}

	gdnConv := make([]byte, 512)
	for i := range gdnConv {
		gdnConv[i] = byte((i*7 + 0xAA) & 0xFF)
	}

	gdnRecurrent := make([]byte, 1024)
	for i := range gdnRecurrent {
		gdnRecurrent[i] = byte((i*13 + 0xBB) & 0xFF)
	}

	sessionID := "session-roundtrip-cudadirect-1"
	tokenCount := 64
	desc, err := swapper.SwapOut(sessionID, tokenCount, kvPages, gdnConv, gdnRecurrent)
	if err != nil {
		t.Fatalf("SwapOut failed: %v", err)
	}

	if desc.Magic != Qwen38CUDADirectSwapMagic {
		t.Errorf("Magic mismatch: got %q, want %q", desc.Magic, Qwen38CUDADirectSwapMagic)
	}
	if desc.SessionID != sessionID {
		t.Errorf("SessionID mismatch: got %q, want %q", desc.SessionID, sessionID)
	}
	if desc.TokenCount != tokenCount {
		t.Errorf("TokenCount mismatch: got %d, want %d", desc.TokenCount, tokenCount)
	}
	if len(desc.KVBlocks) != len(kvPages) {
		t.Fatalf("KVBlocks count mismatch: got %d, want %d", len(desc.KVBlocks), len(kvPages))
	}
	if desc.GDNConvBytes != uint64(len(gdnConv)) {
		t.Errorf("GDNConvBytes mismatch: got %d, want %d", desc.GDNConvBytes, len(gdnConv))
	}
	if desc.GDNRecurrentBytes != uint64(len(gdnRecurrent)) {
		t.Errorf("GDNRecurrentBytes mismatch: got %d, want %d", desc.GDNRecurrentBytes, len(gdnRecurrent))
	}

	gotKVPages, gotConv, gotRec, err := swapper.SwapIn(desc)
	if err != nil {
		t.Fatalf("SwapIn failed: %v", err)
	}

	if len(gotKVPages) != len(kvPages) {
		t.Fatalf("SwapIn KV page count mismatch: got %d, want %d", len(gotKVPages), len(kvPages))
	}
	for i := range kvPages {
		if !bytes.Equal(kvPages[i], gotKVPages[i]) {
			t.Fatalf("KV page %d payload mismatch: bit-exact check failed", i)
		}
	}
	if !bytes.Equal(gdnConv, gotConv) {
		t.Fatalf("GDN conv payload mismatch: bit-exact check failed")
	}
	if !bytes.Equal(gdnRecurrent, gotRec) {
		t.Fatalf("GDN recurrent payload mismatch: bit-exact check failed")
	}
}

func TestQwen38CUDADirectZeroCopyAssertion(t *testing.T) {
	coord, err := NewBlackwellModelCoordinator(ModelArchQwen38_27B, nil, nil)
	if err != nil {
		t.Fatalf("NewBlackwellModelCoordinator failed: %v", err)
	}

	kvPages := [][]byte{
		[]byte("zero-copy-block-0-payload-verification"),
		[]byte("zero-copy-block-1-payload-verification"),
	}
	gdnConv := []byte("gdn-conv-state")
	gdnRecurrent := []byte("gdn-recurrent-state")

	desc, err := coord.SwapOut("zero-copy-session", 32, kvPages, gdnConv, gdnRecurrent)
	if err != nil {
		t.Fatalf("SwapOut failed: %v", err)
	}

	// Zero-copy assertion: StagingCopyCount must strictly be 0
	if staging := desc.StagingCopyCount(); staging != 0 {
		t.Fatalf("StagingCopyCount assertion violated: got %d, want 0", staging)
	}

	_, _, _, err = coord.SwapIn(desc)
	if err != nil {
		t.Fatalf("SwapIn failed: %v", err)
	}

	stats := coord.Stats()
	if stats.ZeroCopyAssertions < 2 {
		t.Errorf("expected at least 2 zero-copy assertions (swap-out and swap-in), got %d", stats.ZeroCopyAssertions)
	}
	if stats.StagingCopies != 0 {
		t.Errorf("stats.StagingCopies must be 0, got %d", stats.StagingCopies)
	}
	if stats.BytesMoved != desc.TotalBytes()*2 {
		t.Errorf("bytes moved mismatch: got %d, want %d", stats.BytesMoved, desc.TotalBytes()*2)
	}
}

func TestQwen38ContextRestoration32K(t *testing.T) {
	// 32K context with 16 tokens/block = 2048 blocks
	tokenCount := 32768
	blockTokens := 16
	numBlocks := tokenCount / blockTokens // 2048

	slabCfg := compute.CUDADirectStorageConfig{
		NodeID:        0,
		BlockSize:     64 * 1024,
		TotalBlocks:   numBlocks + 64, // 2112 blocks
		QueueCapacity: 8192,
	}
	slab, err := compute.NewCUDADirectStorageMemorySlab(slabCfg)
	if err != nil {
		t.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}
	hmm, err := compute.NewHierarchicalMemoryManager(compute.HierarchicalMemoryConfig{}, slab)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	swapper, err := NewQwen38CUDADirectSwapper(ModelArchQwen38_27B, hmm, slab, blockTokens)
	if err != nil {
		t.Fatalf("NewQwen38CUDADirectSwapper failed: %v", err)
	}

	kvPages := make([][]byte, numBlocks)
	for i := range kvPages {
		page := make([]byte, 128)
		binary.LittleEndian.PutUint64(page[0:8], uint64(i))
		kvPages[i] = page
	}
	gdnConv := []byte("gdn-conv-state-32k")
	gdnRecurrent := []byte("gdn-recurrent-state-32k")

	sessionID := "session-32k-context"
	desc, err := swapper.SwapOut(sessionID, tokenCount, kvPages, gdnConv, gdnRecurrent)
	if err != nil {
		t.Fatalf("SwapOut for 32K failed: %v", err)
	}
	if len(desc.KVBlocks) != numBlocks {
		t.Fatalf("expected %d KVBlocks, got %d", numBlocks, len(desc.KVBlocks))
	}

	restoredDesc, duration, err := swapper.RestoreContext32K(sessionID)
	if err != nil {
		t.Fatalf("RestoreContext32K failed: %v", err)
	}

	if duration >= Qwen38Max32KRestorationDuration {
		t.Fatalf("32K context restoration exceeded target envelope (<450ms): %v", duration)
	}
	if restoredDesc == nil || restoredDesc.TokenCount != tokenCount {
		t.Fatalf("restored descriptor token count mismatch: got %v", restoredDesc)
	}
	if restoredDesc.StagingCopyCount() != 0 {
		t.Fatalf("restored descriptor zero-copy violation")
	}

	t.Logf("32K context restoration duration: %v (< 450ms target passed)", duration)
}

func TestQwen38FlashNextMoESlotStreaming(t *testing.T) {
	coord, err := NewBlackwellModelCoordinator(ModelArchQwen38FlashNext, nil, nil)
	if err != nil {
		t.Fatalf("NewBlackwellModelCoordinator failed: %v", err)
	}

	swapper := coord.Swapper()
	// Configure with 8 slots for testing cache hits and evictions
	swapper.ConfigureMoESlots(8, 1024*1024)

	// Stream first batch of 4 experts
	layer0 := 0
	expertsBatch1 := []int{2, 5, 8, 11}
	if err := coord.StreamMoEExperts(layer0, expertsBatch1); err != nil {
		t.Fatalf("StreamMoEExperts batch 1 failed: %v", err)
	}

	stats1 := coord.Stats()
	if stats1.ExpertPrefetches != 4 {
		t.Errorf("expected 4 expert prefetches, got %d", stats1.ExpertPrefetches)
	}
	if stats1.StreamMemopWaits != 4 {
		t.Errorf("expected 4 cuStreamWaitValue64 memop calls, got %d", stats1.StreamMemopWaits)
	}
	if stats1.ExpertCacheHits != 0 {
		t.Errorf("expected 0 cache hits on initial stream, got %d", stats1.ExpertCacheHits)
	}

	// Stream again with same experts (should be all cache hits)
	if err := coord.StreamMoEExperts(layer0, expertsBatch1); err != nil {
		t.Fatalf("StreamMoEExperts batch 1 repeat failed: %v", err)
	}

	stats2 := coord.Stats()
	if stats2.ExpertCacheHits != 4 {
		t.Errorf("expected 4 expert cache hits, got %d", stats2.ExpertCacheHits)
	}
	if stats2.ExpertPrefetches != 4 {
		t.Errorf("expert prefetches should still be 4, got %d", stats2.ExpertPrefetches)
	}

	// Stream mixed batch: 2 hits (2, 5) and 2 new (14, 17)
	mixedBatch := []int{2, 5, 14, 17}
	if err := coord.StreamMoEExperts(layer0, mixedBatch); err != nil {
		t.Fatalf("StreamMoEExperts mixed batch failed: %v", err)
	}

	stats3 := coord.Stats()
	if stats3.ExpertCacheHits != 6 { // 4 + 2 hits
		t.Errorf("expected 6 cumulative cache hits, got %d", stats3.ExpertCacheHits)
	}
	if stats3.ExpertPrefetches != 6 { // 4 + 2 prefetches
		t.Errorf("expected 6 cumulative prefetches, got %d", stats3.ExpertPrefetches)
	}

	// Stream more experts to trigger slot eviction across the 8 capacity slots
	evictionBatch := []int{20, 21, 22, 23, 24, 25}
	if err := coord.StreamMoEExperts(layer0, evictionBatch); err != nil {
		t.Fatalf("StreamMoEExperts eviction batch failed: %v", err)
	}

	stats4 := coord.Stats()
	if stats4.ExpertPrefetches != 12 { // 6 + 6
		t.Errorf("expected 12 cumulative prefetches, got %d", stats4.ExpertPrefetches)
	}
}

func TestQwen38UVAEmbeddingOffload(t *testing.T) {
	coord, err := NewBlackwellModelCoordinator(ModelArchQwen38_27B, nil, nil)
	if err != nil {
		t.Fatalf("NewBlackwellModelCoordinator failed: %v", err)
	}

	emb := coord.Embedding()
	if emb == nil {
		t.Fatalf("expected non-nil VocabParallelEmbedding")
	}

	// 1. Verify embedding resides in Tier 1 Host DRAM
	if emb.Tier != compute.Tier1HostDRAM {
		t.Errorf("expected Tier1HostDRAM (%v), got %v", compute.Tier1HostDRAM, emb.Tier)
	}

	// 2. Verify HostPinned flag is set for UVA direct GPU addressing
	if !emb.HostPinned {
		t.Errorf("expected HostPinned=true for UVA access")
	}

	// 3. Verify size is 2.37 GB nominal BF16 (152064 * 8192 * 2 = 2,491,482,112 bytes)
	expectedSize := uint64(152064) * 8192 * 2
	if emb.SizeBytes != expectedSize {
		t.Errorf("embedding SizeBytes mismatch: got %d, want %d", emb.SizeBytes, expectedSize)
	}

	// 4. Verify VRAM (Tier 0) usage does not consume memory for the embedding table
	vram := coord.VRAMUsageBytes()
	host := coord.HostRAMUsageBytes()

	// Pinned weights (~15GB) are in VRAM
	if vram < 15*1024*1024*1024 {
		t.Errorf("expected VRAM usage >= 15GB for weights, got %d", vram)
	}

	// Host RAM contains the 2.37GB embedding table
	if host < emb.SizeBytes {
		t.Errorf("expected Host RAM usage >= embedding size (%d), got %d", emb.SizeBytes, host)
	}

	// 5. Test UVA host gather / lookup operations
	testTokenID := 42
	testVector := make([]float32, emb.HiddenDim)
	for i := range testVector {
		testVector[i] = float32(i) * 0.05
	}
	if err := emb.SetRow(testTokenID, testVector); err != nil {
		t.Fatalf("SetRow failed: %v", err)
	}

	lookedUp, err := emb.Lookup([]int{testTokenID})
	if err != nil {
		t.Fatalf("Lookup failed: %v", err)
	}
	if len(lookedUp) != 1 || len(lookedUp[0]) != emb.HiddenDim {
		t.Fatalf("unexpected lookup vector dimensions: got %d x %d", len(lookedUp), len(lookedUp[0]))
	}
	for i := 0; i < 10; i++ {
		if lookedUp[0][i] != testVector[i] {
			t.Fatalf("lookedUp[%d] mismatch: got %f, want %f", i, lookedUp[0][i], testVector[i])
		}
	}

	// 6. Test invalid token ID boundary check
	if _, err := emb.Lookup([]int{-1}); err == nil {
		t.Errorf("expected error for negative token ID, got nil")
	}
	if _, err := emb.Lookup([]int{emb.VocabSize}); err == nil {
		t.Errorf("expected error for token ID >= VocabSize, got nil")
	}
}

func TestQwen38CUDADirectDescriptorTotalBytes(t *testing.T) {
	desc := &Qwen38CUDADirectDescriptor{
		KVBlocks: []Qwen38NVMeBlockMapping{
			{SizeBytes: 1024},
			{SizeBytes: 2048},
		},
		GDNConvBytes:      512,
		GDNRecurrentBytes: 1024,
		PLETableBytes:     4096,
	}

	expectedTotal := uint64(1024 + 2048 + 512 + 1024 + 4096)
	if got := desc.TotalBytes(); got != expectedTotal {
		t.Errorf("TotalBytes mismatch: got %d, want %d", got, expectedTotal)
	}

	var nilDesc *Qwen38CUDADirectDescriptor
	if got := nilDesc.TotalBytes(); got != 0 {
		t.Errorf("nil TotalBytes should be 0, got %d", got)
	}
}

func TestQwen38CUDADirectErrors(t *testing.T) {
	swapper, err := NewQwen38CUDADirectSwapper(ModelArchQwen38_27B, nil, nil)
	if err != nil {
		t.Fatalf("NewQwen38CUDADirectSwapper failed: %v", err)
	}

	// Empty session
	if _, err := swapper.SwapOut("", 16, nil, nil, nil); err == nil {
		t.Errorf("expected error for empty sessionID, got nil")
	}

	// Negative tokens
	if _, err := swapper.SwapOut("s1", -1, nil, nil, nil); err == nil {
		t.Errorf("expected error for negative token count, got nil")
	}

	// Bad magic on SwapIn
	badDesc := &Qwen38CUDADirectDescriptor{Magic: "INVALID"}
	if _, _, _, err := swapper.SwapIn(badDesc); err == nil {
		t.Errorf("expected error for invalid magic, got nil")
	}

	// Nil descriptor on SwapIn
	if _, _, _, err := swapper.SwapIn(nil); err == nil {
		t.Errorf("expected error for nil descriptor, got nil")
	}

	// Unknown session on RestoreContext32K
	if _, _, err := swapper.RestoreContext32K("non-existent-session"); err == nil {
		t.Errorf("expected error for unknown session, got nil")
	}

	// Unsupported arch
	if _, err := NewBlackwellModelCoordinator("unsupported-arch", nil, nil); err == nil {
		t.Errorf("expected error for unsupported arch, got nil")
	}
}

// Aliases matching TestQwen38CUDA filter pattern
func TestQwen38CUDAContextRestoration32K(t *testing.T) {
	TestQwen38ContextRestoration32K(t)
}

func TestQwen38CUDAFlashNextMoESlotStreaming(t *testing.T) {
	TestQwen38FlashNextMoESlotStreaming(t)
}

func TestQwen38CUDAUVAEmbeddingOffload(t *testing.T) {
	TestQwen38UVAEmbeddingOffload(t)
}
