package ggufload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=100ms
func TestMmapReadOnlyRangeBoundsAndOwnership(t *testing.T) {
	r := &mmapReaderAt{data: []byte("prefix-payload-tail")}
	type readOnlyRange interface {
		ReadOnlyRange(int64, int64) ([]byte, error)
	}
	ranges, ok := any(r).(readOnlyRange)
	if !ok {
		t.Fatal("mapped range capability missing")
	}
	view, err := ranges.ReadOnlyRange(7, 7)
	if err != nil || !bytes.Equal(view, []byte("payload")) || &view[0] != &r.data[7] || cap(view) != len(view) {
		t.Fatalf("borrowed range=%q cap=%d err=%v", view, cap(view), err)
	}
	owned := make([]byte, len(view))
	if _, err := r.ReadAt(owned, 7); err != nil || !bytes.Equal(owned, view) || &owned[0] == &view[0] {
		t.Fatalf("ReadAt did not retain its owned-copy contract: %v", err)
	}
	for _, bounds := range [][2]int64{{-1, 1}, {0, -1}, {20, 0}, {18, 2}, {1, math.MaxInt64}, {math.MaxInt64, 1}} {
		if got, err := ranges.ReadOnlyRange(bounds[0], bounds[1]); err == nil || got != nil {
			t.Fatalf("invalid bounds %v returned %d bytes, err=%v", bounds, len(got), err)
		}
	}
	if end, err := ranges.ReadOnlyRange(int64(len(r.data)), 0); err != nil || len(end) != 0 || cap(end) != 0 {
		t.Fatalf("empty end range len=%d cap=%d err=%v", len(end), cap(end), err)
	}
	if got, err := any((*mmapReaderAt)(nil)).(readOnlyRange).ReadOnlyRange(0, 0); err == nil || got != nil {
		t.Fatal("nil map was accepted")
	}
	if allocs := testing.AllocsPerRun(100, func() { view, err = ranges.ReadOnlyRange(7, 7) }); allocs != 0 || err != nil {
		t.Fatalf("borrow allocated %g times, err=%v", allocs, err)
	}
}

// fak-test:runtime fast est=100ms
func TestMmapReadOnlyRangeUsesActualRetainedMapping(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "shard.bin")
	want := []byte("prefix-payload-tail")
	if err := os.WriteFile(path, want, 0o600); err != nil {
		t.Fatal(err)
	}
	r, _, closer, data, err := openShardReader(path, true)
	if err != nil {
		t.Fatal(err)
	}
	defer closer.Close()
	mapped, ok := r.(interface {
		ReadOnlyRange(int64, int64) ([]byte, error)
	})
	if !ok {
		t.Skip("platform uses the existing unmapped ReaderAt fallback")
	}
	view, err := mapped.ReadOnlyRange(7, 7)
	if err != nil || !bytes.Equal(view, want[7:14]) || &view[0] != &data[7] || cap(view) != 7 {
		t.Fatalf("actual mapping not borrowed exactly: %q err=%v", view, err)
	}
	owned := make([]byte, 7)
	if _, err := r.ReadAt(owned, 7); err != nil || !bytes.Equal(owned, view) || &owned[0] == &view[0] {
		t.Fatalf("actual mapping ReadAt copy: %v", err)
	}
}
