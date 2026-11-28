package l3kv

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"
)

func TestDiskStoreDirectReadProbeFallsBackToBuffered(t *testing.T) {
	dir := t.TempDir()
	store, err := NewDiskStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Startup probe must succeed (either direct=true or clean buffered fallback direct=false).
	if _, err := store.ProbeDirectRead(); err != nil {
		t.Fatalf("ProbeDirectRead failed: %v", err)
	}

	key := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	payload := []byte("kv-restore-payload-non-4k-aligned")
	if err := store.Put(context.Background(), key, payload); err != nil {
		t.Fatal(err)
	}

	// Simulate O_DIRECT refusal on open -> must fall back to buffered read and return exact bytes.
	got, usedDirect, err := readRecordWithBounce(store.path(key), func(p string) (*os.File, bool, error) {
		// Force fallback path by returning buffered file with direct=false as OpenDirectOrBuffered does on EINVAL.
		f, err := os.Open(p)
		return f, false, err
	})
	if err != nil {
		t.Fatalf("readRecordWithBounce fallback failed: %v", err)
	}
	if usedDirect {
		t.Fatal("expected usedDirect=false on buffered fallback")
	}
	if len(got) < recordHeaderLen || !bytes.Equal(got[recordHeaderLen:], payload) {
		t.Fatalf("fallback payload mismatch: got=%d bytes", len(got))
	}

	// Simulate direct open that fails ReadAt (e.g. EINVAL on non-aligned EOF) -> must fall back to buffered.
	badDirect := func(p string) (*os.File, bool, error) {
		// Return a closed *os.File with direct=true so ReadAt fails and triggers os.ReadFile fallback.
		f, err := os.Open(p)
		if err != nil {
			return nil, false, err
		}
		_ = f.Close()
		return f, true, nil
	}
	got2, usedDirect2, err := readRecordWithBounce(store.path(key), badDirect)
	if err != nil && !errors.Is(err, os.ErrClosed) {
		t.Fatalf("unexpected error on direct-read fallback: %v", err)
	}
	if usedDirect2 {
		t.Fatal("expected usedDirect=false after ReadAt fallback")
	}
	if !bytes.Equal(got2[recordHeaderLen:], payload) {
		t.Fatalf("direct-read fallback payload mismatch")
	}
}
