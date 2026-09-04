package alloc

import (
	"testing"
)

func newTestSlabAllocator(t *testing.T, memBytes uint64) *SlabAllocator {
	t.Helper()
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: memBytes,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	return sa
}

func TestAllocWithPromotion_BestFitAvailable(t *testing.T) {
	sa := newTestSlabAllocator(t, 64*1024*1024)
	defer sa.Close()

	a, promoted, err := sa.AllocWithPromotion(100, -1)
	if err != nil {
		t.Fatalf("AllocWithPromotion: %v", err)
	}
	if promoted {
		t.Error("expected no promotion when best-fit has space")
	}
	if a.Size < 100 {
		t.Errorf("alloc size %d < requested 100", a.Size)
	}
	if a.Size != 128 {
		t.Errorf("expected class size 128, got %d", a.Size)
	}
}

func TestAllocWithPromotion_BestFitFull_PromotesUp(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes:  1024 * 1024,
		MaxKeysPerClass: 4,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	var allocs []Allocation
	for {
		a, err := sa.Alloc(60)
		if err != nil {
			break
		}
		allocs = append(allocs, a)
	}
	if len(allocs) == 0 {
		t.Fatal("should have allocated at least one slot")
	}

	a, promoted, err := sa.AllocWithPromotion(60, -1)
	if err != nil {
		t.Fatalf("AllocWithPromotion should promote up: %v", err)
	}
	if !promoted {
		t.Error("expected promoted=true when best-fit is full")
	}
	if a.Size < 60 {
		t.Errorf("promoted alloc size %d < requested 60", a.Size)
	}
	if a.Size <= 64 {
		t.Errorf("expected class size > 64, got %d", a.Size)
	}

	for _, al := range allocs {
		sa.Free(al)
	}
	sa.Free(a)
}

func TestAllocWithPromotion_RespectsMaxClass(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes:  1024 * 1024,
		MaxKeysPerClass: 4,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	for {
		_, err := sa.Alloc(60)
		if err != nil {
			break
		}
	}

	bestFit := sa.findClass(60)

	_, _, err = sa.AllocWithPromotion(60, bestFit)
	if err == nil {
		t.Error("expected error when maxClassIdx=bestFit and best-fit is full")
	}
}

func TestAllocWithPromotion_AllExhausted(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes:  256 * 1024,
		MaxKeysPerClass: 2,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	for ci := 0; ci < sa.NumClasses(); ci++ {
		sz := sa.ClassSize(ci)
		for {
			_, err := sa.Alloc(sz)
			if err != nil {
				break
			}
		}
	}

	_, _, err = sa.AllocWithPromotion(60, -1)
	if err == nil {
		t.Error("expected error when all classes are exhausted")
	}
}

func TestAllocWithPromotion_Uncapped(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes:  1024 * 1024,
		MaxKeysPerClass: 4,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	for {
		_, err := sa.Alloc(60)
		if err != nil {
			break
		}
	}
	for {
		_, err := sa.Alloc(100)
		if err != nil {
			break
		}
	}

	a, promoted, err := sa.AllocWithPromotion(60, -1)
	if err != nil {
		t.Fatalf("expected uncapped promotion to succeed: %v", err)
	}
	if !promoted {
		t.Error("expected promoted=true")
	}
	if a.Size < 256 {
		t.Errorf("expected class size >= 256, got %d", a.Size)
	}
	sa.Free(a)
}

func TestFreeAfterPromotion(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes:  1024 * 1024,
		MaxKeysPerClass: 4,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	for {
		_, err := sa.Alloc(60)
		if err != nil {
			break
		}
	}

	a, promoted, err := sa.AllocWithPromotion(60, -1)
	if err != nil || !promoted {
		t.Fatalf("promotion should succeed: err=%v, promoted=%v", err, promoted)
	}

	data := []byte("test data for promoted allocation")
	sa.Write(a, data)
	readBack := sa.Read(a)
	if string(readBack[:len(data)]) != string(data) {
		t.Error("data mismatch after write to promoted slot")
	}

	allocBefore := sa.AllocatedBytes()
	sa.Free(a)
	allocAfter := sa.AllocatedBytes()
	if allocAfter >= allocBefore {
		t.Errorf("allocated bytes didn't decrease after free: before=%d, after=%d", allocBefore, allocAfter)
	}
}

func TestOffsetAllocatorPromotion(t *testing.T) {
	oa, err := NewOffsetAllocator(OffsetAllocatorConfig{
		MaxMemoryBytes: 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("NewOffsetAllocator: %v", err)
	}
	defer oa.Close()

	a, promoted, err := oa.AllocWithPromotion(100, -1)
	if err != nil {
		t.Fatalf("AllocWithPromotion: %v", err)
	}
	if promoted {
		t.Error("offset allocator should never report promoted=true")
	}
	if a.Size < 100 {
		t.Errorf("alloc size %d < requested 100", a.Size)
	}
	oa.Free(a)
}
