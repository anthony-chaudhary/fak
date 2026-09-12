package kv_test

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kv"
)

func auditBatchKey(offset int) kv.CacheKey {
	return kv.CacheKey{SessionID: "audit", Turn: 0, Layer: 0, TokenOffset: offset, NumTokens: 1}
}

func TestAuditAllocateBatchFailurePreservesResidentPages(t *testing.T) {
	cfg := kv.DefaultConfig()
	cfg.MaxPages = 2
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for i := 0; i < 2; i++ {
		if _, err := store.Put(auditBatchKey(i), []byte{byte(i + 1)}); err != nil {
			t.Fatal(err)
		}
	}
	_, err = store.AllocateBatch([]kv.CacheKey{auditBatchKey(10), auditBatchKey(11), auditBatchKey(12)})
	if !errors.Is(err, kv.ErrCapacityExceeded) {
		t.Fatalf("AllocateBatch err=%v, want ErrCapacityExceeded", err)
	}
	for i := 0; i < 2; i++ {
		if _, err := store.Get(auditBatchKey(i)); err != nil {
			t.Fatalf("resident page %d was lost after rejected atomic batch: %v", i, err)
		}
	}
}

func TestAuditAllocateBatchPinnedInsufficiencyPreservesEvictablePage(t *testing.T) {
	cfg := kv.DefaultConfig()
	cfg.MaxPages = 3
	store, err := kv.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for i := 0; i < 3; i++ {
		if _, err := store.Put(auditBatchKey(i), []byte{byte(i + 1)}); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 2; i++ {
		page, err := store.Get(auditBatchKey(i))
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Pin(page.ID); err != nil {
			t.Fatal(err)
		}
	}
	_, err = store.AllocateBatch([]kv.CacheKey{auditBatchKey(20), auditBatchKey(21)})
	if !errors.Is(err, kv.ErrCapacityExceeded) {
		t.Fatalf("AllocateBatch err=%v, want ErrCapacityExceeded", err)
	}
	if _, err := store.Get(auditBatchKey(2)); err != nil {
		t.Fatalf("evictable resident page 2 was partially evicted by a rejected batch: %v", err)
	}
	for i := 0; i < 2; i++ {
		if _, err := store.Get(auditBatchKey(i)); err != nil {
			t.Fatalf("pinned resident page %d was lost after rejected batch: %v", i, err)
		}
	}
}
