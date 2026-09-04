package alloc

import (
	"fmt"
	"math/bits"
	"math/rand"
	"testing"
)

// --- Bin encoding tests ---

func TestBinEncodingSmallValues(t *testing.T) {
	for i := uint64(0); i < mantissaValue; i++ {
		up := uintToFloatRoundUp(i)
		down := uintToFloatRoundDown(i)
		if up != uint32(i) {
			t.Errorf("uintToFloatRoundUp(%d) = %d, want %d", i, up, i)
		}
		if down != uint32(i) {
			t.Errorf("uintToFloatRoundDown(%d) = %d, want %d", i, down, i)
		}
		bs := binSize(uint32(i))
		if bs != i {
			t.Errorf("binSize(%d) = %d, want %d", i, bs, i)
		}
	}
}

func TestBinEncodingRoundTrip(t *testing.T) {
	for bin := uint32(0); bin < numBins; bin++ {
		bs := binSize(bin)
		if bs == 0 && bin == 0 {
			continue
		}
		got := uintToFloatRoundDown(bs)
		if got != bin {
			t.Errorf("roundDown(binSize(%d)=%d) = %d, want %d", bin, bs, got, bin)
		}
		got = uintToFloatRoundUp(bs)
		if got != bin {
			t.Errorf("roundUp(binSize(%d)=%d) = %d, want %d", bin, bs, got, bin)
		}
	}
}

func TestBinEncodingMonotonic(t *testing.T) {
	var prev uint64
	for bin := uint32(1); bin < numBins; bin++ {
		bs := binSize(bin)
		if bs < prev {
			t.Fatalf("binSize(%d)=%d < binSize(%d)=%d — not monotonic", bin, bs, bin-1, prev)
		}
		prev = bs
	}
}

func TestBinRoundUpProperty(t *testing.T) {
	sizes := []uint64{1, 7, 8, 9, 15, 16, 17, 100, 255, 256, 1000, 1023, 1024, 1025,
		4096, 65536, 1 << 20, 1 << 30}
	for _, size := range sizes {
		bin := uintToFloatRoundUp(size)
		bs := binSize(bin)
		if bs < size {
			t.Errorf("binSize(roundUp(%d)) = %d < %d — property violated", size, bs, size)
		}
	}

	hugeBin := uintToFloatRoundUp(1 << 37)
	if hugeBin != numBins-1 {
		t.Errorf("roundUp(1<<37) = %d, want %d (saturated)", hugeBin, numBins-1)
	}
}

func TestBinRoundDownProperty(t *testing.T) {
	sizes := []uint64{1, 7, 8, 9, 15, 16, 17, 100, 255, 256, 1000, 1023, 1024, 1025,
		4096, 65536, 1 << 20, 1 << 30}
	for _, size := range sizes {
		bin := uintToFloatRoundDown(size)
		bs := binSize(bin)
		if bs > size {
			t.Errorf("binSize(roundDown(%d)) = %d > %d — property violated", size, bs, size)
		}
	}
}

func TestBinMaxWaste(t *testing.T) {
	for exp := uint(4); exp <= 30; exp++ {
		size := uint64(1<<exp) + 1
		bin := uintToFloatRoundUp(size)
		bs := binSize(bin)
		waste := float64(bs-size) / float64(bs)
		if waste > 0.125+0.001 {
			t.Errorf("waste for size %d: bin=%d binSize=%d waste=%.4f > 12.5%%",
				size, bin, bs, waste)
		}
	}
}

func newTestAllocator(t *testing.T, totalSize uint64, maxAllocs uint32) *OffsetAllocator {
	t.Helper()
	oa, err := NewOffsetAllocator(OffsetAllocatorConfig{
		MaxMemoryBytes: totalSize,
		UseHugePages:   false,
		MaxAllocations: maxAllocs,
	})
	if err != nil {
		t.Fatalf("NewOffsetAllocator failed: %v", err)
	}
	t.Cleanup(func() { oa.Close() })
	return oa
}

func TestOffsetAllocBasic(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)

	data := []byte("hello, offset allocator!")
	a, err := oa.Alloc(uint64(len(data)))
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	if a.ClassIdx != 0 {
		t.Errorf("ClassIdx = %d, want 0", a.ClassIdx)
	}
	if a.Size < uint64(len(data)) {
		t.Errorf("Size = %d < len(data) = %d", a.Size, len(data))
	}
	if a.Offset == 0 {
		t.Error("first alloc returned offset 0 — should be reserved")
	}

	oa.Write(a, data)
	got := oa.Read(a)
	if string(got[:len(data)]) != string(data) {
		t.Errorf("Read mismatch: %q != %q", got[:len(data)], data)
	}

	allocated := oa.AllocatedBytes()
	if allocated <= 0 {
		t.Errorf("AllocatedBytes = %d, want > 0", allocated)
	}

	oa.Free(a)
	if oa.AllocatedBytes() != 0 {
		t.Errorf("AllocatedBytes after free = %d, want 0", oa.AllocatedBytes())
	}
}

func TestOffsetAllocNeverReturnsZero(t *testing.T) {
	oa := newTestAllocator(t, 1<<16, 1024)

	for i := 0; i < 100; i++ {
		a, err := oa.Alloc(256)
		if err != nil {
			break
		}
		if a.Offset == 0 {
			t.Fatalf("alloc %d returned offset 0", i)
		}
		oa.Free(a)
	}
}

func TestOffsetAllocZeroSize(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)
	_, err := oa.Alloc(0)
	if err == nil {
		t.Fatal("expected error for 0-byte allocation")
	}
}

func TestOffsetAllocMultiple(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)

	var allocs []Allocation
	for i := 0; i < 10; i++ {
		size := uint64(100 + i*50)
		a, err := oa.Alloc(size)
		if err != nil {
			t.Fatalf("Alloc(%d) #%d: %v", size, i, err)
		}
		payload := []byte(fmt.Sprintf("block-%d", i))
		oa.Write(a, payload)
		allocs = append(allocs, a)
	}

	for i, a := range allocs {
		expected := fmt.Sprintf("block-%d", i)
		got := oa.Read(a)
		if string(got[:len(expected)]) != expected {
			t.Errorf("block %d: got %q, want %q", i, got[:len(expected)], expected)
		}
	}

	for _, a := range allocs {
		oa.Free(a)
	}
	if oa.AllocatedBytes() != 0 {
		t.Errorf("AllocatedBytes after freeing all = %d", oa.AllocatedBytes())
	}
}

func TestCoalescingRightNeighbor(t *testing.T) {
	oa := newTestAllocator(t, 4096, 256)

	a, _ := oa.Alloc(1024)
	b, _ := oa.Alloc(1024)

	oa.Free(a)
	oa.Free(b)

	big, err := oa.Alloc(3072)
	if err != nil {
		t.Fatalf("Alloc after coalescing: %v", err)
	}
	oa.Free(big)
}

func TestCoalescingLeftNeighbor(t *testing.T) {
	oa := newTestAllocator(t, 4096, 256)

	a, _ := oa.Alloc(1024)
	b, _ := oa.Alloc(1024)

	oa.Free(b)
	oa.Free(a)

	big, err := oa.Alloc(3072)
	if err != nil {
		t.Fatalf("Alloc after coalescing: %v", err)
	}
	oa.Free(big)
}

func TestCoalescingThreeWay(t *testing.T) {
	oa := newTestAllocator(t, 8192, 256)

	a, _ := oa.Alloc(1024)
	b, _ := oa.Alloc(1024)
	c, _ := oa.Alloc(1024)

	oa.Free(a)
	oa.Free(c)
	oa.Free(b)

	if oa.AllocatedBytes() != 0 {
		t.Fatalf("AllocatedBytes = %d after freeing all", oa.AllocatedBytes())
	}

	big, err := oa.Alloc(6144)
	if err != nil {
		t.Fatalf("Alloc after three-way coalesce: %v", err)
	}
	oa.Free(big)
}

func TestFragmentationRecovery(t *testing.T) {
	oa := newTestAllocator(t, 1<<16, 1024)

	var allocs []Allocation
	for i := 0; i < 32; i++ {
		a, err := oa.Alloc(512)
		if err != nil {
			t.Fatalf("Alloc(%d): %v", i, err)
		}
		allocs = append(allocs, a)
	}

	for i := 0; i < len(allocs); i += 2 {
		oa.Free(allocs[i])
		allocs[i] = Allocation{}
	}

	for i := 1; i < len(allocs); i += 2 {
		oa.Free(allocs[i])
	}

	if oa.AllocatedBytes() != 0 {
		t.Fatalf("AllocatedBytes = %d after freeing all", oa.AllocatedBytes())
	}

	big, err := oa.Alloc(1 << 15)
	if err != nil {
		t.Fatalf("Alloc(32KB) after defrag: %v", err)
	}
	oa.Free(big)
}

func TestExhaustion(t *testing.T) {
	oa := newTestAllocator(t, 4096, 256)

	var allocs []Allocation
	for {
		a, err := oa.Alloc(512)
		if err != nil {
			break
		}
		allocs = append(allocs, a)
	}

	if len(allocs) == 0 {
		t.Fatal("expected at least one allocation before exhaustion")
	}

	_, err := oa.Alloc(512)
	if err == nil {
		t.Fatal("expected error when allocator is exhausted")
	}

	oa.Free(allocs[0])
	a, err := oa.Alloc(512)
	if err != nil {
		t.Fatalf("Alloc after free: %v", err)
	}
	oa.Free(a)
}

func TestSlotUtilization(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)

	for bin := uint32(1); bin < numBins; bin++ {
		bs := binSize(bin)
		if bs == 0 {
			continue
		}
		u := oa.SlotUtilization(bs)
		if u != 1.0 {
			t.Errorf("SlotUtilization(binSize(%d)=%d) = %f, want 1.0", bin, bs, u)
		}
	}

	u := oa.SlotUtilization(1025)
	if u < 0.874 || u >= 1.0 {
		t.Errorf("SlotUtilization(1025) = %f, want [0.875, 1.0)", u)
	}

	if u := oa.SlotUtilization(0); u != 0 {
		t.Errorf("SlotUtilization(0) = %f, want 0", u)
	}
}

func TestMetricsCompat(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)

	if oa.NumClasses() != 1 {
		t.Errorf("NumClasses() = %d, want 1", oa.NumClasses())
	}
	if oa.ClassSize(0) != 1<<20 {
		t.Errorf("ClassSize(0) = %d, want %d", oa.ClassSize(0), 1<<20)
	}
	if oa.FindClass(100) != 0 {
		t.Errorf("FindClass(100) = %d, want 0", oa.FindClass(100))
	}
	if oa.FindClass(1<<21) != -1 {
		t.Errorf("FindClass(2MB) = %d, want -1", oa.FindClass(1<<21))
	}

	slots, cs := oa.ModelClassCapacity(4096)
	if slots == 0 || cs == 0 {
		t.Errorf("ModelClassCapacity(4096) = (%d, %d), want non-zero", slots, cs)
	}
	if cs < 4096 {
		t.Errorf("ModelClassCapacity classSize %d < 4096", cs)
	}

	regions := oa.Regions()
	if len(regions) != 1 {
		t.Fatalf("Regions() len = %d, want 1", len(regions))
	}
	if regions[0].ClassIdx != 0 {
		t.Errorf("Region ClassIdx = %d, want 0", regions[0].ClassIdx)
	}

	_, _, regular := oa.HugepageSummary()
	if regular != 1 {
		t.Errorf("HugepageSummary regular = %d, want 1", regular)
	}

	oa.Alloc(1000)
	oa.Alloc(2000)
	utils := oa.ClassUtilizations()
	if len(utils) != 1 {
		t.Fatalf("ClassUtilizations() len = %d, want 1", len(utils))
	}
	if utils[0].AllocCount != 2 {
		t.Errorf("AllocCount = %d, want 2", utils[0].AllocCount)
	}
	if utils[0].UsedSlots == 0 {
		t.Error("UsedSlots = 0, want > 0")
	}

	oa.ResetCounters()
	utils = oa.ClassUtilizations()
	if utils[0].AllocCount != 0 {
		t.Errorf("AllocCount after reset = %d, want 0", utils[0].AllocCount)
	}
}

func TestDoubleFreeIsNoop(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)

	a, _ := oa.Alloc(1024)
	oa.Free(a)
	before := oa.AllocatedBytes()
	oa.Free(a)
	if oa.AllocatedBytes() != before {
		t.Errorf("double free changed allocated bytes: %d -> %d", before, oa.AllocatedBytes())
	}
}

func TestAllocatorInterface(t *testing.T) {
	a, err := NewAllocator("offset", SlabConfig{}, OffsetAllocatorConfig{
		MaxMemoryBytes: 1 << 20,
		MaxAllocations: 1024,
	})
	if err != nil {
		t.Fatalf("NewAllocator(offset): %v", err)
	}
	defer a.Close()

	alloc, err := a.Alloc(512)
	if err != nil {
		t.Fatalf("Alloc via interface: %v", err)
	}
	a.Free(alloc)
}

func BenchmarkOffsetAlloc(b *testing.B) {
	oa, err := NewOffsetAllocator(OffsetAllocatorConfig{
		MaxMemoryBytes: 1 << 30,
		UseHugePages:   false,
		MaxAllocations: 1 << 20,
	})
	if err != nil {
		b.Fatalf("NewOffsetAllocator: %v", err)
	}
	defer oa.Close()

	b.Run("Alloc+Free/4KB", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			a, err := oa.Alloc(4096)
			if err != nil {
				b.Fatalf("Alloc: %v", err)
			}
			oa.Free(a)
		}
	})

	b.Run("Alloc+Free/variable", func(b *testing.B) {
		rng := rand.New(rand.NewSource(42))
		for i := 0; i < b.N; i++ {
			size := uint64(rng.Intn(1<<16-64) + 64)
			a, err := oa.Alloc(size)
			if err != nil {
				b.Fatalf("Alloc(%d): %v", size, err)
			}
			oa.Free(a)
		}
	})

	b.Run("BinEncoding", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sz := uint64(i+1) & 0xFFFFFFF
			bin := uintToFloatRoundUp(sz)
			_ = binSize(bin)
		}
	})
}

func BenchmarkOffsetAllocChurn(b *testing.B) {
	oa, err := NewOffsetAllocator(OffsetAllocatorConfig{
		MaxMemoryBytes: 1 << 30,
		UseHugePages:   false,
		MaxAllocations: 1 << 20,
	})
	if err != nil {
		b.Fatalf("NewOffsetAllocator: %v", err)
	}
	defer oa.Close()

	rng := rand.New(rand.NewSource(99))
	allocs := make([]Allocation, 1000)
	for i := range allocs {
		a, err := oa.Alloc(uint64(rng.Intn(8192) + 256))
		if err != nil {
			b.Fatalf("pre-fill: %v", err)
		}
		allocs[i] = a
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % len(allocs)
		oa.Free(allocs[idx])
		a, err := oa.Alloc(uint64(rng.Intn(8192) + 256))
		if err != nil {
			b.Fatalf("churn alloc: %v", err)
		}
		allocs[idx] = a
	}
}

func TestFindBinEmpty(t *testing.T) {
	oa := newTestAllocator(t, 1<<20, 1024)

	var allocs []Allocation
	for {
		a, err := oa.Alloc(1024)
		if err != nil {
			break
		}
		allocs = append(allocs, a)
	}

	minBin := uintToFloatRoundUp(1024)
	_, found := oa.findBin(minBin)
	if found {
		t.Error("findBin should return false for 1024-byte bin when allocator is full")
	}

	for _, a := range allocs {
		oa.Free(a)
	}
}

func TestBinCount(t *testing.T) {
	if numBins != 256 {
		t.Fatalf("numBins = %d, want 256", numBins)
	}
	if numTopBins != 32 {
		t.Fatalf("numTopBins = %d, want 32", numTopBins)
	}
	if mantissaBits != 3 {
		t.Fatalf("mantissaBits = %d, want 3", mantissaBits)
	}
	maxBinSize := binSize(numBins - 1)
	if bits.Len64(maxBinSize) < 30 {
		t.Errorf("highest bin covers only %d — expected at least 2^30", maxBinSize)
	}
}
