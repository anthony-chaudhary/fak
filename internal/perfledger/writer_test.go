package perfledger

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=50ms lane=default
func TestWriterOfferCloseWritesReadableRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "perf.jsonl")
	w := OpenWriter(path, 0)
	if w.Path() != path {
		t.Fatalf("path = %q", w.Path())
	}
	const n = 5
	for i := 0; i < n; i++ {
		if !w.Offer(NewRecord(time.UnixMilli(int64(i+1)), "stop", LocalityUnknown, 10, 2, 0, time.Second, 0)) {
			t.Fatalf("offer %d dropped", i)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if w.Written() != n || w.Dropped() != 0 {
		t.Fatalf("written/dropped = %d/%d, want %d/0", w.Written(), w.Dropped(), n)
	}
	recs, truncated := ReadTail(path)
	if truncated || len(recs) != n {
		t.Fatalf("read back %d rows (truncated=%v), want %d", len(recs), truncated, n)
	}
	for i, r := range recs {
		if r.UnixMS != int64(i+1) {
			t.Fatalf("row %d unix_ms = %d", i, r.UnixMS)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestWriterOfferAfterCloseCountsDropped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "perf.jsonl")
	w := OpenWriter(path, 0)
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if w.Offer(NewRecord(time.Now(), "stop", "", 1, 1, 0, time.Second, 0)) {
		t.Fatal("offer after close reported accepted")
	}
	if w.Dropped() != 1 || w.Written() != 0 {
		t.Fatalf("dropped/written = %d/%d, want 1/0", w.Dropped(), w.Written())
	}
	if err := w.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestWriterNilIsInert(t *testing.T) {
	var w *Writer
	if w.Offer(Record{}) || w.Dropped() != 0 || w.Written() != 0 || w.Close() != nil || w.Path() != "" {
		t.Fatal("nil writer not inert")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestWriterRotatesPastMaxBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "perf.jsonl")
	old := bytes.Repeat([]byte("x"), 300)
	old = append(old, '\n')
	if err := os.WriteFile(path, old, 0o644); err != nil {
		t.Fatal(err)
	}
	w := OpenWriter(path, 256)
	w.Offer(NewRecord(time.UnixMilli(42), "stop", "", 1, 1, 0, time.Second, 0))
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	sealed, err := os.ReadFile(path + ".1")
	if err != nil {
		t.Fatalf("rotated generation missing: %v", err)
	}
	if !bytes.Equal(sealed, old) {
		t.Fatalf("sealed generation = %d bytes, want the prior %d", len(sealed), len(old))
	}
	recs, _ := ReadTail(path)
	if len(recs) != 1 || recs[0].UnixMS != 42 {
		t.Fatalf("active file rows = %+v, want one row unix_ms=42", recs)
	}
}
