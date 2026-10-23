package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1s
func TestDirectSlabRejectsUnrepresentableBacking(t *testing.T) {
	maxInt := int64(int(^uint(0) >> 1))
	for _, tc := range []struct {
		name      string
		capacity  int64
		alignment int
	}{
		{"max_int", maxInt, 64},
		{"alignment_headroom", maxInt - 63, 64},
		{"default_alignment", math.MaxInt64, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			slab, err := NewDirectSlabAllocator(DirectSlabConfig{TotalBytes: tc.capacity, Alignment: tc.alignment})
			if slab != nil {
				defer slab.Close()
			}
			if err == nil || slab != nil {
				t.Fatalf("unrepresentable backing accepted: slab=%v err=%v", slab, err)
			}
		})
	}
	for _, alignment := range []int{1, 64} {
		name := "byte_aligned"
		if alignment == 64 {
			name = "cache_aligned"
		}
		t.Run(name, func(t *testing.T) {
			const capacity = 1024
			slab, err := NewDirectSlabAllocator(DirectSlabConfig{TotalBytes: capacity, Alignment: alignment})
			if err != nil {
				t.Fatal(err)
			}
			defer slab.Close()
			allocation, err := slab.Allocate(capacity)
			if err != nil {
				t.Fatal(err)
			}
			if len(allocation.Data) != capacity || slab.Available() != 0 || slab.BaseAddress()%uintptr(alignment) != 0 {
				t.Fatal("valid backing geometry changed")
			}
			if err := allocation.Free(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestDirectSlabOversizedRequestPreservesActiveRegion(t *testing.T) {
	const capacity int64 = 1024
	const retained int64 = 64
	for _, tc := range []struct {
		name string
		size int64
	}{
		{"max_int64", math.MaxInt64},
		{"alignment_overflow", math.MaxInt64 - 31},
		{"remaining_capacity", capacity - retained + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			slab, err := NewDirectSlabAllocator(DirectSlabConfig{TotalBytes: capacity, Alignment: int(retained), DirectInterconnect: true})
			if err != nil {
				t.Fatal(err)
			}
			defer slab.Close()
			// Software owner fixture, not a native registration witness.
			bindTestDirectSlabRegistration(t, slab)
			live, err := slab.Allocate(retained)
			if err != nil {
				t.Fatal(err)
			}
			live.Data[0] = 0x5a
			before, err := RegisterDirectSlabRegion(slab, live)
			if err != nil {
				t.Fatal(err)
			}
			for attempt := 0; attempt < 2; attempt++ {
				allocation, err := slab.Allocate(tc.size)
				if allocation != nil || err == nil {
					t.Fatalf("oversized allocation=%v err=%v", allocation, err)
				}
				if slab.Used() != retained || slab.Available() != capacity-retained || live.Data[0] != 0x5a {
					t.Fatal("refusal changed active allocation")
				}
				after, err := RegisterDirectSlabRegion(slab, live)
				if err != nil {
					t.Fatal(err)
				}
				if after.IOVA != before.IOVA || after.Length != before.Length || after.LKey != before.LKey {
					t.Fatal("refusal changed registered region")
				}
			}
			if err := live.Free(); err != nil {
				t.Fatal(err)
			}
			whole, err := slab.Allocate(capacity)
			if err != nil {
				t.Fatalf("refusal corrupted reusable free list: %v", err)
			}
			if whole.Offset != 0 || whole.Size != capacity {
				t.Fatal("whole slab no longer reusable")
			}
			if err := whole.Free(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
