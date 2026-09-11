package kv_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kv"
)

func auditIdentityKey() kv.CacheKey {
	return kv.CacheKey{SessionID: "audit-identity", Turn: 0, Layer: 0, TokenOffset: 0, NumTokens: 1}
}

func TestAuditAllocateBatchPreservesKeyIdentity(t *testing.T) {
	newStore := func(t *testing.T) *kv.KVStore {
		t.Helper()
		cfg := kv.DefaultConfig()
		cfg.MaxPages = 8
		s, err := kv.New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Close() })
		return s
	}

	t.Run("existing key", func(t *testing.T) {
		s := newStore(t)
		key := auditIdentityKey()
		first, err := s.AllocatePage(key)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := s.AllocateBatch([]kv.CacheKey{key})
		if err != nil || len(batch) != 1 {
			t.Fatalf("AllocateBatch: pages=%v err=%v", batch, err)
		}
		if batch[0].ID != first.ID || s.Stats().AllocatedPages != 1 {
			t.Errorf("existing key allocated new identity: first=%d batch=%d allocated=%d", first.ID, batch[0].ID, s.Stats().AllocatedPages)
		}
		if err := s.FreePage(batch[0].ID); err != nil {
			t.Fatal(err)
		}
		if pages := s.ListPages(); len(pages) != 0 {
			_, indexed := s.Lookup(key)
			t.Errorf("free left %d resident page(s) with key indexed=%v", len(pages), indexed)
		}
	})

	t.Run("duplicate keys", func(t *testing.T) {
		s := newStore(t)
		key := auditIdentityKey()
		batch, err := s.AllocateBatch([]kv.CacheKey{key, key})
		if err != nil || len(batch) != 2 {
			t.Fatalf("AllocateBatch: pages=%v err=%v", batch, err)
		}
		if batch[0].ID != batch[1].ID || s.Stats().AllocatedPages != 1 {
			t.Errorf("duplicate keys allocated distinct identities: ids=(%d,%d) allocated=%d", batch[0].ID, batch[1].ID, s.Stats().AllocatedPages)
		}
	})
}
