//go:build !linux

package alloc

import (
	"fmt"
	"testing"
)

// The non-Linux fallback caps each region's heap backing (region_default.go)
// so unit tests cannot exhaust host commit (#11311). Every slot and offset
// computation derives from Region.Size(), so the reported size must never
// exceed the bytes actually backing the region (#13518).

// overCapRegion is 1.5x the 64 MiB non-Linux dev ceiling.
const overCapRegion = 96 << 20

// mustNotPanic runs fn and converts a panic into a test failure so a red run
// reports the defect instead of aborting the whole package.
func mustNotPanic(t *testing.T, what string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s panicked: %v", what, fmt.Sprint(r))
		}
	}()
	fn()
}

func TestRegionSizeMatchesBackingOnNonLinux(t *testing.T) {
	r, err := NewRegion(overCapRegion, 0)
	if err != nil {
		t.Fatalf("NewRegion: %v", err)
	}
	defer r.Close()

	if size, backing := r.Size(), uint64(len(r.Data())); size != backing {
		t.Fatalf("Region.Size()=%d but backing len=%d: slot math would slice past the backing array", size, backing)
	}
}

func TestBitmapAllocatorStaysInsideCappedRegion(t *testing.T) {
	const slot = 32 << 20
	r, err := NewRegion(overCapRegion, 0)
	if err != nil {
		t.Fatalf("NewRegion: %v", err)
	}
	defer r.Close()

	ba := NewBitmapAllocator(r, slot)
	for {
		off, ok := ba.Alloc()
		if !ok {
			break
		}
		var data []byte
		mustNotPanic(t, fmt.Sprintf("SlotData(%d)", off), func() { data = ba.SlotData(off) })
		if uint64(len(data)) != slot {
			t.Fatalf("SlotData(%d) len=%d, want %d", off, len(data), slot)
		}
		data[len(data)-1] = 0xAB // touch the last byte of every slot
	}
	if got := ba.NumSlots() * slot; got > uint64(len(r.Data())) {
		t.Fatalf("NumSlots*slot=%d exceeds backing len %d", got, len(r.Data()))
	}
}

func TestBitmapAllocatorSlotLargerThanCappedRegion(t *testing.T) {
	r, err := NewRegion(256<<20, 0)
	if err != nil {
		t.Fatalf("NewRegion: %v", err)
	}
	defer r.Close()

	// A slot class bigger than the whole capped backing yields no usable
	// slots; it must fail allocation cleanly instead of panicking.
	var ba *BitmapAllocator
	mustNotPanic(t, "NewBitmapAllocator", func() { ba = NewBitmapAllocator(r, 128<<20) })
	var (
		off uint64
		ok  bool
	)
	mustNotPanic(t, "Alloc", func() { off, ok = ba.Alloc() })
	if ok {
		mustNotPanic(t, fmt.Sprintf("SlotData(%d)", off), func() { _ = ba.SlotData(off) })
	}
}

func TestSlabAllocatorLargeClassStaysInsideCappedRegion(t *testing.T) {
	const big = 32 << 20
	// With 64 MiB total, the 32 MiB default class gets the sz*4 floor
	// (128 MiB logical), which exceeds the capped backing.
	sa, err := NewSlabAllocator(SlabConfig{MaxMemoryBytes: 64 << 20})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	payload := make([]byte, big)
	payload[len(payload)-1] = 0x5A
	for i := 0; ; i++ {
		a, err := sa.Alloc(big)
		if err != nil {
			break // class exhausted: the honest outcome once real capacity is used
		}
		mustNotPanic(t, fmt.Sprintf("Write #%d at offset %d", i, a.Offset), func() { sa.Write(a, payload) })
		var got []byte
		mustNotPanic(t, fmt.Sprintf("Read #%d at offset %d", i, a.Offset), func() { got = sa.Read(a) })
		if got[len(got)-1] != 0x5A {
			t.Fatalf("Read #%d lost the last byte", i)
		}
	}
}

func TestOffsetAllocatorStaysInsideCappedRegion(t *testing.T) {
	oa, err := NewOffsetAllocator(OffsetAllocatorConfig{MaxMemoryBytes: overCapRegion, MaxAllocations: 16})
	if err != nil {
		t.Fatalf("NewOffsetAllocator: %v", err)
	}
	defer oa.Close()

	backing := uint64(len(oa.region.Data()))
	if size := oa.ClassSize(0); size > backing {
		t.Fatalf("OffsetAllocator reports %d bytes of capacity but backing len is %d", size, backing)
	}

	const want = 80 << 20 // fits the logical size but not the capped backing
	a, err := oa.Alloc(want)
	if err != nil {
		return // refused: correct, the capped region cannot hold it
	}
	var got []byte
	mustNotPanic(t, "Read", func() { got = oa.Read(a) })
	if uint64(len(got)) < want {
		t.Fatalf("Alloc(%d) succeeded but Read returned only %d bytes", want, len(got))
	}
}
