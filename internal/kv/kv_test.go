package kv_test

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/kv"
)

func TestConfig_ValidationAndDefaults(t *testing.T) {
	cfg := kv.DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("DefaultConfig failed validation: %v", err)
	}
	if cfg.PageSize != kv.DefaultPageSize {
		t.Errorf("expected page size %d, got %d", kv.DefaultPageSize, cfg.PageSize)
	}
	if cfg.EvictionPolicy != kv.EvictionPolicyLRU {
		t.Errorf("expected default policy %s, got %s", kv.EvictionPolicyLRU, cfg.EvictionPolicy)
	}

	// Invalid PageSize
	badCfg := cfg
	badCfg.PageSize = 0
	if err := badCfg.Validate(); !errors.Is(err, kv.ErrInvalidConfig) {
		t.Errorf("expected ErrInvalidConfig for non-positive page size, got %v", err)
	}

	// Invalid EvictionPolicy
	badPolicyCfg := cfg
	badPolicyCfg.EvictionPolicy = "random"
	if err := badPolicyCfg.Validate(); !errors.Is(err, kv.ErrInvalidConfig) {
		t.Errorf("expected ErrInvalidConfig for unknown policy, got %v", err)
	}

	// DirectIO without backing file
	badDirectCfg := cfg
	badDirectCfg.DirectIO = true
	badDirectCfg.BackingFile = ""
	if err := badDirectCfg.Validate(); !errors.Is(err, kv.ErrInvalidConfig) {
		t.Errorf("expected ErrInvalidConfig for DirectIO without backing file, got %v", err)
	}
}

func TestCacheKey_ValidationAndOverlaps(t *testing.T) {
	validKey := kv.CacheKey{
		SessionID:   "sess-alpha",
		Turn:        1,
		Layer:       0,
		TokenOffset: 128,
		NumTokens:   64,
		Tag:         "prompt",
	}
	if err := validKey.Validate(); err != nil {
		t.Fatalf("valid key failed: %v", err)
	}
	if !validKey.MatchesSession("sess-alpha") {
		t.Errorf("MatchesSession failed for sess-alpha")
	}
	if validKey.MatchesSession("sess-beta") {
		t.Errorf("MatchesSession should be false for sess-beta")
	}

	strKey := validKey.String()
	if strKey == "" {
		t.Errorf("expected non-empty key string representation")
	}

	// Invalid keys
	emptySess := validKey
	emptySess.SessionID = "   "
	if err := emptySess.Validate(); !errors.Is(err, kv.ErrInvalidKey) {
		t.Errorf("expected ErrInvalidKey for empty session, got %v", err)
	}

	negativeTurn := validKey
	negativeTurn.Turn = -1
	if err := negativeTurn.Validate(); !errors.Is(err, kv.ErrInvalidKey) {
		t.Errorf("expected ErrInvalidKey for negative turn, got %v", err)
	}

	// Overlaps test
	overlappingKey := kv.CacheKey{
		SessionID:   "sess-alpha",
		Turn:        1,
		Layer:       0,
		TokenOffset: 160,
		NumTokens:   64,
	}
	if !validKey.Overlaps(overlappingKey) {
		t.Errorf("expected keys [128:192) and [160:224) to overlap")
	}

	disjointKey := kv.CacheKey{
		SessionID:   "sess-alpha",
		Turn:        1,
		Layer:       0,
		TokenOffset: 200,
		NumTokens:   32,
	}
	if validKey.Overlaps(disjointKey) {
		t.Errorf("keys [128:192) and [200:232) should not overlap")
	}

	diffLayerKey := overlappingKey
	diffLayerKey.Layer = 1
	if validKey.Overlaps(diffLayerKey) {
		t.Errorf("keys on different layers should not overlap")
	}
}

func TestPage_OperationsAndClone(t *testing.T) {
	key := kv.CacheKey{SessionID: "sess-1", Turn: 0, Layer: 0, TokenOffset: 0, NumTokens: 16}
	page := &kv.Page{
		ID:        1,
		Key:       key,
		Data:      make([]byte, 128),
		BytesUsed: 32,
	}
	for i := 0; i < 32; i++ {
		page.Data[i] = byte(i)
	}

	if page.IsPinned() {
		t.Errorf("page should initially be unpinned")
	}
	page.Pin()
	if !page.IsPinned() {
		t.Errorf("page should be pinned after Pin()")
	}
	page.Unpin()
	if page.IsPinned() {
		t.Errorf("page should be unpinned after Unpin()")
	}

	bytes := page.Bytes()
	if len(bytes) != 32 {
		t.Fatalf("expected 32 bytes, got %d", len(bytes))
	}

	cloned := page.Clone()
	if cloned.ID != page.ID {
		t.Errorf("cloned ID mismatch")
	}
	// Mutating cloned buffer should not affect original
	cloned.Data[0] = 0xFF
	if page.Data[0] == 0xFF {
		t.Errorf("mutation in cloned page mutated original page data")
	}
}

func TestStore_AllocatePageAndBatch(t *testing.T) {
	cfg := kv.Config{
		PageSize:       1024,
		MaxPages:       10,
		EvictionPolicy: kv.EvictionPolicyLRU,
	}
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatalf("New store failed: %v", err)
	}
	defer store.Close()

	key := kv.CacheKey{SessionID: "s1", Turn: 0, Layer: 0, TokenOffset: 0, NumTokens: 32}
	p, err := store.AllocatePage(key)
	if err != nil {
		t.Fatalf("AllocatePage failed: %v", err)
	}
	if p.ID != 0 {
		t.Errorf("expected first page ID 0, got %d", p.ID)
	}

	// Idempotent allocation of same key returns existing page
	p2, err := store.AllocatePage(key)
	if err != nil {
		t.Fatalf("AllocatePage duplicate key failed: %v", err)
	}
	if p2.ID != p.ID {
		t.Errorf("expected page ID %d for same key, got %d", p.ID, p2.ID)
	}

	// Batch allocation
	batchKeys := []kv.CacheKey{
		{SessionID: "s1", Turn: 1, Layer: 0, TokenOffset: 32, NumTokens: 32},
		{SessionID: "s1", Turn: 1, Layer: 1, TokenOffset: 32, NumTokens: 32},
		{SessionID: "s1", Turn: 1, Layer: 2, TokenOffset: 32, NumTokens: 32},
	}
	pages, err := store.AllocateBatch(batchKeys)
	if err != nil {
		t.Fatalf("AllocateBatch failed: %v", err)
	}
	if len(pages) != 3 {
		t.Fatalf("expected 3 allocated pages, got %d", len(pages))
	}

	stats := store.Stats()
	if stats.AllocatedPages != 4 {
		t.Errorf("expected 4 allocated pages in stats, got %d", stats.AllocatedPages)
	}
}

func TestStore_PutAndGet(t *testing.T) {
	cfg := kv.DefaultConfig()
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatalf("New store failed: %v", err)
	}
	defer store.Close()

	key := kv.CacheKey{SessionID: "s-put", Turn: 0, Layer: 0, TokenOffset: 0, NumTokens: 64}
	payload := []byte("transformer KV tensor activation state bytes")

	p, err := store.Put(key, payload)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	if !bytes.Equal(p.Bytes(), payload) {
		t.Errorf("page bytes do not match payload")
	}

	// Get by key
	got, err := store.Get(key)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if !bytes.Equal(got.Bytes(), payload) {
		t.Errorf("Get returned mismatched payload")
	}

	// Get by ID
	gotByID, err := store.GetByID(p.ID)
	if err != nil {
		t.Fatalf("GetByID failed: %v", err)
	}
	if !bytes.Equal(gotByID.Bytes(), payload) {
		t.Errorf("GetByID returned mismatched payload")
	}

	// Missing key
	missingKey := kv.CacheKey{SessionID: "s-missing", Turn: 0, Layer: 0}
	if _, err := store.Get(missingKey); !errors.Is(err, kv.ErrKeyNotFound) {
		t.Errorf("expected ErrKeyNotFound for missing key, got %v", err)
	}

	// Missing ID
	if _, err := store.GetByID(9999); !errors.Is(err, kv.ErrPageNotFound) {
		t.Errorf("expected ErrPageNotFound for missing page ID, got %v", err)
	}

	// Oversized payload
	oversized := make([]byte, cfg.PageSize+1)
	if _, err := store.Put(key, oversized); !errors.Is(err, kv.ErrDataTooLarge) {
		t.Errorf("expected ErrDataTooLarge, got %v", err)
	}
}

func TestStore_CapacityAndEviction(t *testing.T) {
	// Capacity of 3 pages with LRU eviction
	cfg := kv.Config{
		PageSize:       512,
		MaxPages:       3,
		EvictionPolicy: kv.EvictionPolicyLRU,
	}
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatalf("New store: %v", err)
	}
	defer store.Close()

	keys := []kv.CacheKey{
		{SessionID: "s", Turn: 0, Layer: 0, TokenOffset: 0, NumTokens: 10},
		{SessionID: "s", Turn: 0, Layer: 1, TokenOffset: 0, NumTokens: 10},
		{SessionID: "s", Turn: 0, Layer: 2, TokenOffset: 0, NumTokens: 10},
	}

	for _, k := range keys {
		if _, err := store.Put(k, []byte("data")); err != nil {
			t.Fatalf("Put %v: %v", k, err)
		}
		time.Sleep(2 * time.Millisecond)
	}

	// Touch keys[0] so keys[1] becomes the least recently used
	if _, err := store.Get(keys[0]); err != nil {
		t.Fatalf("Get keys[0]: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// Inserting a 4th key should evict keys[1]
	key4 := kv.CacheKey{SessionID: "s", Turn: 0, Layer: 3, TokenOffset: 0, NumTokens: 10}
	if _, err := store.Put(key4, []byte("data4")); err != nil {
		t.Fatalf("Put key4: %v", err)
	}

	// keys[1] should now be gone
	if _, err := store.Get(keys[1]); !errors.Is(err, kv.ErrKeyNotFound) {
		t.Errorf("expected keys[1] to be evicted, got err = %v", err)
	}

	// keys[0] and keys[2] and key4 should exist
	if _, err := store.Get(keys[0]); err != nil {
		t.Errorf("keys[0] should still exist: %v", err)
	}
	if _, err := store.Get(key4); err != nil {
		t.Errorf("key4 should exist: %v", err)
	}

	stats := store.Stats()
	if stats.Evictions != 1 {
		t.Errorf("expected 1 eviction, got %d", stats.Evictions)
	}
}

func TestStore_FIFOEviction(t *testing.T) {
	cfg := kv.Config{
		PageSize:       512,
		MaxPages:       2,
		EvictionPolicy: kv.EvictionPolicyFIFO,
	}
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	key1 := kv.CacheKey{SessionID: "s", Turn: 0, Layer: 0}
	key2 := kv.CacheKey{SessionID: "s", Turn: 0, Layer: 1}
	key3 := kv.CacheKey{SessionID: "s", Turn: 0, Layer: 2}

	_, _ = store.Put(key1, []byte("1"))
	time.Sleep(2 * time.Millisecond)
	_, _ = store.Put(key2, []byte("2"))
	time.Sleep(2 * time.Millisecond)

	// In FIFO, even if key1 is frequently accessed, it was allocated first and will be evicted first
	_, _ = store.Get(key1)
	_, _ = store.Get(key1)

	_, err = store.Put(key3, []byte("3"))
	if err != nil {
		t.Fatalf("Put key3: %v", err)
	}

	if _, err := store.Get(key1); !errors.Is(err, kv.ErrKeyNotFound) {
		t.Errorf("expected key1 to be evicted under FIFO, got %v", err)
	}
	if _, err := store.Get(key2); err != nil {
		t.Errorf("expected key2 to survive under FIFO: %v", err)
	}
}

func TestStore_PinAndUnpinProtection(t *testing.T) {
	cfg := kv.Config{
		PageSize: 512,
		MaxPages: 1,
	}
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer store.Close()

	key1 := kv.CacheKey{SessionID: "s", Turn: 0, Layer: 0}
	p1, err := store.Put(key1, []byte("data1"))
	if err != nil {
		t.Fatalf("Put key1: %v", err)
	}

	// Pin page 1
	if err := store.Pin(p1.ID); err != nil {
		t.Fatalf("Pin failed: %v", err)
	}
	stats := store.Stats()
	if stats.PinnedPages != 1 {
		t.Errorf("expected 1 pinned page, got %d", stats.PinnedPages)
	}

	// Now attempt to put key2 when store is full and only page is pinned
	key2 := kv.CacheKey{SessionID: "s", Turn: 0, Layer: 1}
	_, err = store.Put(key2, []byte("data2"))
	if !errors.Is(err, kv.ErrCapacityExceeded) {
		t.Fatalf("expected ErrCapacityExceeded when all pages are pinned, got %v", err)
	}

	// Unpin page 1 and try again
	if err := store.Unpin(p1.ID); err != nil {
		t.Fatalf("Unpin failed: %v", err)
	}
	if _, err := store.Put(key2, []byte("data2")); err != nil {
		t.Fatalf("Put key2 after unpin should succeed: %v", err)
	}
}

func TestStore_PruneIdleAndEvictSession(t *testing.T) {
	store, err := kv.DefaultStore()
	if err != nil {
		t.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	sessA := "session-A"
	sessB := "session-B"

	for i := 0; i < 3; i++ {
		key := kv.CacheKey{SessionID: sessA, Turn: i, Layer: 0, TokenOffset: i * 32, NumTokens: 32}
		_, _ = store.Put(key, []byte("a"))
	}
	for i := 0; i < 2; i++ {
		key := kv.CacheKey{SessionID: sessB, Turn: i, Layer: 0, TokenOffset: i * 32, NumTokens: 32}
		_, _ = store.Put(key, []byte("b"))
	}

	if store.Stats().AllocatedPages != 5 {
		t.Fatalf("expected 5 pages, got %d", store.Stats().AllocatedPages)
	}

	// Evict session B
	evicted, err := store.EvictSession(sessB)
	if err != nil {
		t.Fatalf("EvictSession failed: %v", err)
	}
	if evicted != 2 {
		t.Errorf("expected 2 evicted pages for sessB, got %d", evicted)
	}
	if store.Stats().AllocatedPages != 3 {
		t.Errorf("expected 3 remaining pages, got %d", store.Stats().AllocatedPages)
	}

	// Prune idle with duration should evict everything unpinned older than 1ms
	time.Sleep(5 * time.Millisecond)
	pruned, err := store.PruneIdle(1 * time.Millisecond)
	if err != nil {
		t.Fatalf("PruneIdle failed: %v", err)
	}
	if pruned != 3 {
		t.Errorf("expected 3 pruned pages, got %d", pruned)
	}
	if store.Stats().AllocatedPages != 0 {
		t.Errorf("expected 0 remaining pages, got %d", store.Stats().AllocatedPages)
	}
}

func TestStore_LookupAndLookupSession(t *testing.T) {
	store, err := kv.DefaultStore()
	if err != nil {
		t.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	sess := "session-search"
	key1 := kv.CacheKey{SessionID: sess, Turn: 2, Layer: 0, TokenOffset: 64, NumTokens: 32}
	key2 := kv.CacheKey{SessionID: sess, Turn: 1, Layer: 0, TokenOffset: 0, NumTokens: 64}

	_, _ = store.Put(key1, []byte("k1"))
	_, _ = store.Put(key2, []byte("k2"))

	// Non-mutating lookup
	p, ok := store.Lookup(key1)
	if !ok || p == nil {
		t.Fatalf("Lookup failed")
	}
	if !bytes.Equal(p.Bytes(), []byte("k1")) {
		t.Errorf("Lookup returned wrong data")
	}

	// LookupSession should return pages sorted by Turn ascending
	sessPages := store.LookupSession(sess)
	if len(sessPages) != 2 {
		t.Fatalf("expected 2 pages for session, got %d", len(sessPages))
	}
	if sessPages[0].Key.Turn != 1 || sessPages[1].Key.Turn != 2 {
		t.Errorf("expected pages sorted by turn: turn %d, then turn %d", sessPages[0].Key.Turn, sessPages[1].Key.Turn)
	}
}

func TestStore_LookupRange(t *testing.T) {
	store, err := kv.DefaultStore()
	if err != nil {
		t.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	sess := "session-range"
	// Create pages spanning [0:64), [64:128), [128:192)
	k0 := kv.CacheKey{SessionID: sess, Turn: 1, Layer: 0, TokenOffset: 0, NumTokens: 64}
	k1 := kv.CacheKey{SessionID: sess, Turn: 1, Layer: 0, TokenOffset: 64, NumTokens: 64}
	k2 := kv.CacheKey{SessionID: sess, Turn: 1, Layer: 0, TokenOffset: 128, NumTokens: 64}

	_, _ = store.Put(k0, []byte("p0"))
	_, _ = store.Put(k1, []byte("p1"))
	_, _ = store.Put(k2, []byte("p2"))

	// Query range [50:100) -> overlaps k0 and k1
	res := store.LookupRange(sess, 1, 50, 100)
	if len(res) != 2 {
		t.Fatalf("expected 2 overlapping pages for [50, 100), got %d", len(res))
	}
	if res[0].Key.TokenOffset != 0 || res[1].Key.TokenOffset != 64 {
		t.Errorf("unexpected offsets in range lookup: %d, %d", res[0].Key.TokenOffset, res[1].Key.TokenOffset)
	}

	// Query exact token position 150 -> overlaps k2
	resSingle := store.LookupRange(sess, 1, 150, 150)
	if len(resSingle) != 1 {
		t.Fatalf("expected 1 page for token 150, got %d", len(resSingle))
	}
	if resSingle[0].Key.TokenOffset != 128 {
		t.Errorf("expected offset 128, got %d", resSingle[0].Key.TokenOffset)
	}
}

func TestStore_StatsAndReset(t *testing.T) {
	store, err := kv.DefaultStore()
	if err != nil {
		t.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	key := kv.CacheKey{SessionID: "s-stat", Turn: 0, Layer: 0}
	_, _ = store.Put(key, []byte("test"))
	_, _ = store.Get(key)
	_, _ = store.Get(kv.CacheKey{SessionID: "missing", Turn: 0})

	st := store.Stats()
	if st.Puts != 1 {
		t.Errorf("expected 1 put, got %d", st.Puts)
	}
	if st.Gets != 2 {
		t.Errorf("expected 2 gets, got %d", st.Gets)
	}
	if st.Hits != 1 {
		t.Errorf("expected 1 hit, got %d", st.Hits)
	}
	if st.Misses != 1 {
		t.Errorf("expected 1 miss, got %d", st.Misses)
	}
	if st.HitRatio() != 0.5 {
		t.Errorf("expected hit ratio 0.5, got %f", st.HitRatio())
	}

	store.ResetStats()
	rst := store.Stats()
	if rst.Puts != 0 || rst.Gets != 0 || rst.Hits != 0 || rst.Misses != 0 {
		t.Errorf("expected reset metrics to be 0")
	}
	// Allocated pages should not be reset
	if rst.AllocatedPages != 1 {
		t.Errorf("expected allocated pages to remain 1")
	}
}

func TestStore_DirectIOBacking(t *testing.T) {
	tmpDir := t.TempDir()
	backingFile := filepath.Join(tmpDir, "store_direct.kv")

	cfg := kv.Config{
		PageSize:    kv.DefaultPageSize,
		MaxPages:    16,
		DirectIO:    true,
		BackingFile: backingFile,
	}

	store, err := kv.New(cfg)
	if err != nil {
		t.Fatalf("failed to open DirectIO store: %v", err)
	}

	key := kv.CacheKey{
		SessionID:   "sess-direct",
		Turn:        1,
		Layer:       0,
		TokenOffset: 0,
		NumTokens:   64,
	}
	testData := make([]byte, 1024)
	for i := 0; i < len(testData); i++ {
		testData[i] = byte((i * 17) % 256)
	}

	page, err := store.Put(key, testData)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	if page.DirectIOBlockID < 0 {
		t.Errorf("expected valid direct IO block ID, got %d", page.DirectIOBlockID)
	}

	got, err := store.Get(key)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if !bytes.Equal(got.Bytes(), testData) {
		t.Fatalf("direct-IO persisted payload mismatch")
	}

	if err := store.Close(); err != nil {
		t.Fatalf("store.Close failed: %v", err)
	}
}

func TestStore_Concurrency(t *testing.T) {
	store, err := kv.DefaultStore()
	if err != nil {
		t.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	const numWorkers = 8
	const opsPerWorker = 50
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				key := kv.CacheKey{
					SessionID:   fmt.Sprintf("worker-%d", workerID),
					Turn:        i % 4,
					Layer:       i % 2,
					TokenOffset: i * 16,
					NumTokens:   16,
				}
				data := []byte(fmt.Sprintf("payload-%d-%d", workerID, i))
				_, err := store.Put(key, data)
				if err != nil {
					t.Errorf("concurrent Put failed: %v", err)
					return
				}

				got, err := store.Get(key)
				if err != nil {
					t.Errorf("concurrent Get failed: %v", err)
					return
				}
				if !bytes.Equal(got.Bytes(), data) {
					t.Errorf("concurrent data mismatch")
					return
				}

				if i%10 == 0 {
					_ = store.LookupRange(key.SessionID, key.Turn, 0, 1000)
				}
			}
		}()
	}

	wg.Wait()
	st := store.Stats()
	if st.Puts != uint64(numWorkers*opsPerWorker) {
		t.Errorf("expected %d puts, got %d", numWorkers*opsPerWorker, st.Puts)
	}
}

func TestStore_Close(t *testing.T) {
	store, err := kv.DefaultStore()
	if err != nil {
		t.Fatalf("DefaultStore: %v", err)
	}

	if err := store.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("subsequent Close should be idempotent, got %v", err)
	}

	key := kv.CacheKey{SessionID: "s", Turn: 0}
	if _, err := store.Put(key, []byte("test")); !errors.Is(err, kv.ErrStoreClosed) {
		t.Errorf("expected ErrStoreClosed on Put, got %v", err)
	}
	if _, err := store.Get(key); !errors.Is(err, kv.ErrStoreClosed) {
		t.Errorf("expected ErrStoreClosed on Get, got %v", err)
	}
	if _, err := store.AllocatePage(key); !errors.Is(err, kv.ErrStoreClosed) {
		t.Errorf("expected ErrStoreClosed on AllocatePage, got %v", err)
	}
}

func BenchmarkStore_PutGet(b *testing.B) {
	store, err := kv.DefaultStore()
	if err != nil {
		b.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	data := make([]byte, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := kv.CacheKey{
			SessionID:   "bench-sess",
			Turn:        i % 8,
			Layer:       i % 4,
			TokenOffset: (i % 64) * 16,
			NumTokens:   16,
		}
		if _, err := store.Put(key, data); err != nil {
			b.Fatalf("Put: %v", err)
		}
		if _, err := store.Get(key); err != nil {
			b.Fatalf("Get: %v", err)
		}
	}
}

func BenchmarkStore_LookupRange(b *testing.B) {
	store, err := kv.DefaultStore()
	if err != nil {
		b.Fatalf("DefaultStore: %v", err)
	}
	defer store.Close()

	data := make([]byte, 256)
	sess := "bench-range"
	for i := 0; i < 100; i++ {
		key := kv.CacheKey{
			SessionID:   sess,
			Turn:        0,
			Layer:       0,
			TokenOffset: i * 32,
			NumTokens:   32,
		}
		_, _ = store.Put(key, data)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results := store.LookupRange(sess, 0, 500, 1500)
		if len(results) == 0 {
			b.Fatalf("expected non-empty range result")
		}
	}
}

func BenchmarkStore_AllocateEvict(b *testing.B) {
	cfg := kv.Config{
		PageSize:       512,
		MaxPages:       64,
		EvictionPolicy: kv.EvictionPolicyLRU,
	}
	store, err := kv.New(cfg)
	if err != nil {
		b.Fatalf("New store: %v", err)
	}
	defer store.Close()

	payload := make([]byte, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := kv.CacheKey{
			SessionID:   "bench-evict",
			Turn:        i % 16,
			Layer:       i % 8,
			TokenOffset: i * 8,
			NumTokens:   8,
		}
		if _, err := store.Put(key, payload); err != nil {
			b.Fatalf("Put: %v", err)
		}
	}
}
