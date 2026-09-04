package shard

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
)

// submitRestore runs an OpRestore with the given entries and returns the
// shard-reported loaded count.
func submitRestore(t *testing.T, s *Shard, entries []snapshot.KVEntry) int {
	t.Helper()
	res := s.Submit(ShardOp{
		Type:           OpRestore,
		RestoreEntries: entries,
		Result:         make(chan OpResult, 1),
	})
	if res.Err != nil {
		t.Fatalf("restore submit failed: %v", res.Err)
	}
	return res.Loaded
}

// readValue reads the value bytes for a restored entry directly from the
// allocator, so a test can prove which Value won an overwrite.
func readValue(t *testing.T, s *Shard, e index.Entry) []byte {
	t.Helper()
	a := s.allocPtr.Load().a
	ci := int(e.ValueClassIdx)
	raw := a.Read(alloc.Allocation{
		ClassIdx: ci,
		Offset:   e.ValueOffset,
		Size:     a.ClassSize(ci),
	})
	if uint32(len(raw)) < e.ValueLen {
		t.Fatalf("read returned %d bytes, want >= %d", len(raw), e.ValueLen)
	}
	out := make([]byte, e.ValueLen)
	copy(out, raw[:e.ValueLen])
	return out
}

// TestHandleRestore_UpdatesMetrics covers C1c: restored entries feed
// Sets/BytesIn/Key+ValueBytesIn so Prometheus is not blind post-restore.
func TestHandleRestore_UpdatesMetrics(t *testing.T) {
	s := newTestShard(t, 0, 0, 1000, true)
	defer s.Allocator().Close()
	s.Start()
	defer func() { s.Stop(); <-s.Done() }()

	entries := []snapshot.KVEntry{
		{Key: []byte("alpha"), Value: []byte("0123456789")},
		{Key: []byte("beta"), Value: []byte("abcdef")},
		{Key: []byte("gamma"), Value: []byte("xyz")},
	}
	loaded := submitRestore(t, s, entries)
	if loaded != len(entries) {
		t.Fatalf("loaded=%d, want %d", loaded, len(entries))
	}

	var wantKey, wantVal int64
	for _, e := range entries {
		wantKey += int64(len(e.Key))
		wantVal += int64(len(e.Value))
	}
	if got := s.metrics.Sets(); got != int64(len(entries)) {
		t.Errorf("Sets=%d, want %d", got, len(entries))
	}
	if got := s.metrics.KeyBytesIn(); got != wantKey {
		t.Errorf("KeyBytesIn=%d, want %d", got, wantKey)
	}
	if got := s.metrics.ValueBytesIn(); got != wantVal {
		t.Errorf("ValueBytesIn=%d, want %d", got, wantVal)
	}
	if got := s.metrics.BytesIn(); got != wantKey+wantVal {
		t.Errorf("BytesIn=%d, want %d", got, wantKey+wantVal)
	}
}

// TestHandleRestore_FeedsSizeTracker covers the auto-tune half of C1c: value
// sizes from restored entries reach sizeTracker, so detection can fire on
// post-restore traffic alone.
func TestHandleRestore_FeedsSizeTracker(t *testing.T) {
	// WarmupOps=4 â†’ detection fires once sizeTracker has seen 4 values.
	s := newTestShard(t, 0, 0, 4, true)
	defer s.Allocator().Close()
	s.Start()
	defer func() { s.Stop(); <-s.Done() }()

	const valSize = 524288 // 512KB
	val := make([]byte, valSize)
	entries := make([]snapshot.KVEntry, 4)
	for i := range entries {
		entries[i] = snapshot.KVEntry{
			Key:   []byte{'k', byte(i)},
			Value: val,
		}
	}
	submitRestore(t, s, entries)

	if !s.sizeTracker.detected {
		t.Fatal("expected sizeTracker.detected=true after restore â€” handleRestore is not feeding sizeTracker.record")
	}
	if s.sizeTracker.optimalSize != valSize {
		t.Errorf("optimalSize=%d, want %d", s.sizeTracker.optimalSize, valSize)
	}
}

// TestHandleRestore_DuplicateKey covers C1b: restoring the same key twice
// leaves exactly one index entry holding the second value. The pre-fix code
// double-Admitted the eviction tracker and leaked the first allocation.
func TestHandleRestore_DuplicateKey(t *testing.T) {
	s := newTestShard(t, 0, 0, 1000, true)
	defer s.Allocator().Close()
	s.Start()
	defer func() { s.Stop(); <-s.Done() }()

	key := []byte("dup")
	first := snapshot.KVEntry{Key: key, Value: []byte("first-value")}
	second := snapshot.KVEntry{Key: key, Value: []byte("second-value-longer")}

	submitRestore(t, s, []snapshot.KVEntry{first})
	submitRestore(t, s, []snapshot.KVEntry{second})

	if c := s.idx.Count(); c != 1 {
		t.Fatalf("index Count=%d after duplicate restore, want 1", c)
	}

	keyHash := index.KeyHash(key)
	e, _, ok := s.idx.Lookup(keyHash, uint16(len(key)))
	if !ok {
		t.Fatal("expected key to be present after duplicate restore")
	}
	if int(e.ValueLen) != len(second.Value) {
		t.Fatalf("ValueLen=%d, want %d (second restore did not overwrite)",
			e.ValueLen, len(second.Value))
	}
	got := readValue(t, s, e)
	if string(got) != string(second.Value) {
		t.Errorf("value=%q, want %q", got, second.Value)
	}
}
