//go:build linux

package ggufload

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

// fak-test:runtime fast est=100ms lane=default
func TestMmapReadAheadRangeBoundsAndLinuxAdvice(t *testing.T) {
	t.Parallel()
	page := unix.Getpagesize()
	want := bytes.Repeat([]byte{0x69}, 2*page+17)
	path := filepath.Join(t.TempDir(), "mapped-experts.bin")
	if err := os.WriteFile(path, want, 0o600); err != nil {
		t.Fatal(err)
	}
	r, _, closer, data, err := openShardReader(path, true)
	if err != nil {
		t.Fatal(err)
	}
	defer closer.Close()
	m, ok := r.(*mmapReaderAt)
	if !ok {
		t.Fatal("Linux file did not retain its mapping")
	}
	calls := 0
	for _, span := range [][2]int64{{int64(page + 7), 11}, {int64(2 * page), 17}, {int64(len(want) - 1), 1}} {
		start, end := span[0]-span[0]%int64(page), span[0]+span[1]
		if err := m.readAheadRange(span[0], span[1], func(got []byte, advice int) error {
			calls++
			if advice != unix.MADV_WILLNEED || &got[0] != &data[start] || len(got) != int(end-start) || cap(got) != len(got) {
				t.Fatalf("advice=%d len=%d cap=%d, want exact page-aligned [%d,%d)", advice, len(got), cap(got), start, end)
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		// This is an actual Linux syscall acceptance/permission witness, not
		// an assertion that the backing store performed or overlapped I/O.
		if err := m.ReadAheadRange(span[0], span[1]); err != nil {
			t.Fatalf("Linux MADV_WILLNEED refused: %v", err)
		}
	}
	never := func([]byte, int) error { t.Fatal("invalid or empty range reached syscall"); return nil }
	for _, span := range [][2]int64{{-1, 1}, {0, -1}, {int64(len(want)), 1}, {math.MaxInt64, 1}, {1, math.MaxInt64}} {
		if err := m.readAheadRange(span[0], span[1], never); err == nil {
			t.Fatalf("accepted invalid bounds %v", span)
		}
	}
	if err := (*mmapReaderAt)(nil).readAheadRange(0, 0, never); err == nil {
		t.Fatal("accepted nil mapping")
	}
	if err := m.readAheadRange(int64(len(want)), 0, never); err != nil {
		t.Fatal(err)
	}
	sentinel := errors.New("advice denied")
	if err := m.readAheadRange(1, 1, func([]byte, int) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatalf("advice error was hidden: %v", err)
	}
	if calls != 3 || !bytes.Equal(data, want) {
		t.Fatal("advice altered mapped source or issued extra ranges")
	}
}
