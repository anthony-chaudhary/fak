package kv_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kv"
)

func TestAuditPutPagePreservesKeyIdentity(t *testing.T) {
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

	key := kv.CacheKey{SessionID: "audit-putpage", Turn: 0, Layer: 0, TokenOffset: 0, NumTokens: 1}

	s := newStore(t)

	pageA := &kv.Page{ID: 100, Key: key, Data: []byte{0xA1}, BytesUsed: 1, NumTokens: 1}
	if err := s.PutPage(pageA); err != nil {
		t.Fatalf("PutPage(pageA): %v", err)
	}

	pageB := &kv.Page{ID: 200, Key: key, Data: []byte{0xB2}, BytesUsed: 1, NumTokens: 1}
	if err := s.PutPage(pageB); err != nil {
		t.Fatalf("PutPage(pageB): %v", err)
	}

	pages := s.ListPages()
	if len(pages) != 1 {
		t.Errorf("resident pages=%d, want 1 after PutPage over the same key (orphaned=%d)", len(pages), len(pages)-1)
	}

	got, ok := s.Lookup(key)
	if !ok {
		t.Fatalf("Lookup(key) not found after PutPage over the same key; resident pages=%d", len(pages))
	}
	if got.ID != pageB.ID {
		t.Errorf("Lookup(key).ID=%d, want newest %d", got.ID, pageB.ID)
	}

	if err := s.FreePage(got.ID); err != nil {
		t.Fatalf("FreePage(%d): %v", got.ID, err)
	}
	if remaining := s.ListPages(); len(remaining) != 0 {
		_, indexed := s.Lookup(key)
		t.Errorf("free left %d resident page(s) with key indexed=%v (key-unreachable orphan)", len(remaining), indexed)
	}
}
