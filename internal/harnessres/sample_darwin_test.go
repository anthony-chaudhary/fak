//go:build darwin && cgo

package harnessres

import (
	"os"
	"syscall"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestDarwinSelfRSSReaderTracksCurrentResidentPages(t *testing.T) {
	reader := DarwinSelfRSSReader()
	if reader == nil {
		t.Fatal("Darwin+cgo must expose a current resident-RSS reader")
	}
	before, ok := reader()
	if !ok || before == 0 {
		t.Fatalf("initial current RSS = %d (ok=%v), want a positive live reading", before, ok)
	}

	const mappingBytes = 32 << 20
	mapping, err := syscall.Mmap(-1, 0, mappingBytes, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatalf("mmap resident-RSS witness: %v", err)
	}
	t.Cleanup(func() {
		if mapping != nil {
			if err := syscall.Munmap(mapping); err != nil {
				t.Errorf("release resident-RSS witness: %v", err)
			}
		}
	})
	// Reservation alone is virtual memory. Write every page so the mapping is
	// resident; the generous threshold tolerates unrelated process bookkeeping.
	for offset := 0; offset < len(mapping); offset += os.Getpagesize() {
		mapping[offset] = byte(offset/os.Getpagesize()%251 + 1)
	}
	during, ok := reader()
	if !ok || during < before || during-before < mappingBytes/2 {
		t.Fatalf("touched %d bytes: current RSS before=%d during=%d (ok=%v), want growth >= %d", mappingBytes, before, during, ok, mappingBytes/2)
	}

	if err := syscall.Munmap(mapping); err != nil {
		t.Fatalf("munmap resident-RSS witness: %v", err)
	}
	mapping = nil
	after, ok := reader()
	t.Logf("current resident RSS bytes: before=%d touched=%d reclaimed=%d", before, during, after)
	// A high-water RSS counter cannot drop after munmap. Current resident_size
	// must reflect the reclaimed pages without depending on Go's heap scavenger.
	if !ok || after == 0 || after > during || during-after < mappingBytes/2 {
		t.Fatalf("released %d bytes: current RSS during=%d after=%d (ok=%v), want decline >= %d", mappingBytes, during, after, ok, mappingBytes/2)
	}
}
