//go:build darwin && cgo

package main

import (
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"testing"
)

// fak-test:runtime slow est=15s
// vmmap independently reads the kernel's current physical footprint. Keep a
// bounded anonymous allocation alive across the observations, then unmap it to
// distinguish current footprint from a lifetime high-water mark.
func TestPlatformCurrentRSSMatchesPhysicalFootprint(t *testing.T) {
	if _, err := exec.LookPath("vmmap"); err != nil {
		t.Fatal("physical witness requires native vmmap: ", err)
	}
	observe := func(label string) uint64 {
		t.Helper()
		before := platformCurrentRSS()
		out, err := exec.Command("vmmap", "-summary", strconv.Itoa(os.Getpid())).CombinedOutput()
		if err != nil {
			t.Fatalf("%s vmmap: %v", label, err)
		}
		want, err := parseModelCanaryFootprint(out)
		if err != nil || want <= 0 {
			t.Fatalf("%s physical footprint: %d, %v", label, want, err)
		}
		after := platformCurrentRSS()
		lo, hi := before, after
		if lo > hi {
			lo, hi = hi, lo
		}
		// vmmap rounds its human-readable result; the subprocess and runtime
		// can dirty a few pages between reads. Two MiB covers this bounded drift.
		const tolerance = uint64(2 << 20)
		if before == 0 || after == 0 || uint64(want)+tolerance < lo || uint64(want) > hi+tolerance {
			t.Fatalf("%s current guard=%d..%d vmmap physical footprint=%d (tolerance=%d)", label, lo, hi, want, tolerance)
		}
		t.Logf("%s guard=%d..%d physical_footprint=%d", label, lo, hi, want)
		return after
	}
	baseline := observe("baseline")
	pages, err := syscall.Mmap(-1, 0, 16<<20, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if pages != nil {
			_ = syscall.Munmap(pages)
		}
	}()
	for i := 0; i < len(pages); i += os.Getpagesize() {
		pages[i] = 1
	}
	allocated := observe("allocated")
	if allocated < baseline+(12<<20) {
		t.Fatalf("16 MiB dirty allocation was not accounted: baseline=%d allocated=%d", baseline, allocated)
	}
	if err := syscall.Munmap(pages); err != nil {
		t.Fatal(err)
	}
	pages = nil
	released := observe("released")
	if released+(12<<20) > allocated {
		t.Fatalf("current footprint did not recover after unmapping: allocated=%d released=%d", allocated, released)
	}
}
