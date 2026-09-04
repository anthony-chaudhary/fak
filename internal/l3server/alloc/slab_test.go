package alloc

import (
	"strings"
	"testing"
)

func TestSlabAllocBasic(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 64 * 1024 * 1024,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	a, err := sa.Alloc(100)
	if err != nil {
		t.Fatalf("Alloc(100): %v", err)
	}
	if a.Size < 100 {
		t.Errorf("alloc size %d < requested 100", a.Size)
	}

	data := []byte("hello, cama!")
	sa.Write(a, data)
	readBack := sa.Read(a)
	if string(readBack[:len(data)]) != string(data) {
		t.Errorf("read back mismatch: got %q", string(readBack[:len(data)]))
	}

	sa.Free(a)
}

func TestSlabSlotUtilizationLlama70B(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 64 * 1024 * 1024,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	util := sa.SlotUtilization(5242880)
	t.Logf("SlotUtilization for 5242880 bytes (Llama 70B page): %.2f%%", util*100)
	if util < 0.95 {
		t.Errorf("slot utilization for Llama 70B page: %.2f%%, expected >= 95%%", util*100)
	}

	util640k := sa.SlotUtilization(655360)
	t.Logf("SlotUtilization for 655360 bytes (Llama 70B per-layer): %.2f%%", util640k*100)
	if util640k < 0.95 {
		t.Errorf("slot utilization for per-layer page: %.2f%%, expected >= 95%%", util640k*100)
	}

	buddyUtil := float64(5242880) / float64(8388608)
	t.Logf("Buddy allocator slot utilization for same size: %.1f%%", buddyUtil*100)
}

func TestBitmapAllocator(t *testing.T) {
	region, err := NewRegion(4096, 0)
	if err != nil {
		t.Fatalf("NewRegion: %v", err)
	}
	defer region.Close()

	ba := NewBitmapAllocator(region, 64)
	if ba.NumSlots() != 64 {
		t.Errorf("num slots: got %d, want 64", ba.NumSlots())
	}
	if ba.FreeCount() != 63 {
		t.Errorf("free count: got %d, want 63", ba.FreeCount())
	}

	firstOff, ok := ba.Alloc()
	if !ok {
		t.Fatal("first alloc failed")
	}
	if firstOff == 0 {
		t.Error("first alloc returned offset 0 — slot 0 should be reserved")
	}
	if firstOff != 64 {
		t.Errorf("first alloc: got offset %d, want 64 (slotSize)", firstOff)
	}

	offsets := make([]uint64, 63)
	offsets[0] = firstOff
	for i := 1; i < 63; i++ {
		off, ok := ba.Alloc()
		if !ok {
			t.Fatalf("alloc %d failed", i)
		}
		if off == 0 {
			t.Errorf("alloc %d returned offset 0", i)
		}
		offsets[i] = off
	}

	_, ok = ba.Alloc()
	if ok {
		t.Error("expected alloc to fail when full")
	}

	ba.Free(offsets[0])
	if ba.FreeCount() != 1 {
		t.Errorf("free count after free: got %d, want 1", ba.FreeCount())
	}
	realloc, ok := ba.Alloc()
	if !ok {
		t.Error("expected alloc to succeed after free")
	}
	if realloc == 0 {
		t.Error("re-alloc returned offset 0")
	}
}

func TestBitmapAllocatorNeverReturnsZero(t *testing.T) {
	region, err := NewRegion(8192, 0)
	if err != nil {
		t.Fatalf("NewRegion: %v", err)
	}
	defer region.Close()

	ba := NewBitmapAllocator(region, 128)
	for i := 0; i < int(ba.NumSlots()); i++ {
		off, ok := ba.Alloc()
		if !ok {
			break
		}
		if off == 0 {
			t.Fatalf("alloc %d returned offset 0", i)
		}
	}
}

func TestSlabPerClassCounters(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 64 * 1024 * 1024,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	allocs := make([]Allocation, 3)
	for i := 0; i < 3; i++ {
		a, err := sa.Alloc(100)
		if err != nil {
			t.Fatalf("Alloc %d: %v", i, err)
		}
		allocs[i] = a
	}

	cls := &sa.classes[allocs[0].ClassIdx]
	if cls.AllocCount.Load() != 3 {
		t.Errorf("AllocCount: got %d, want 3", cls.AllocCount.Load())
	}
	if cls.TotalRequestBytes.Load() != 300 {
		t.Errorf("TotalRequestBytes: got %d, want 300", cls.TotalRequestBytes.Load())
	}
	if cls.FreeCount_.Load() != 0 {
		t.Errorf("FreeCount_: got %d, want 0", cls.FreeCount_.Load())
	}

	sa.Free(allocs[0])
	if cls.FreeCount_.Load() != 1 {
		t.Errorf("FreeCount_ after free: got %d, want 1", cls.FreeCount_.Load())
	}
}

func TestSlabClassUtilizations(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 64 * 1024 * 1024,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	a1, _ := sa.Alloc(100)
	a2, _ := sa.Alloc(100)
	sa.Alloc(5242880)
	sa.Free(a1)

	utils := sa.ClassUtilizations()
	if len(utils) == 0 {
		t.Fatal("expected non-empty utilizations")
	}

	cls := utils[a2.ClassIdx]
	if cls.AllocCount != 2 {
		t.Errorf("AllocCount: got %d, want 2", cls.AllocCount)
	}
	if cls.FreeCount != 1 {
		t.Errorf("FreeCount: got %d, want 1", cls.FreeCount)
	}
	if cls.UsedSlots != 2 {
		t.Errorf("UsedSlots: got %d, want 2 (1 real + 1 reserved)", cls.UsedSlots)
	}
	expectedAvgReq := 100.0
	if cls.AvgRequestBytes != expectedAvgReq {
		t.Errorf("AvgRequestBytes: got %.1f, want %.1f", cls.AvgRequestBytes, expectedAvgReq)
	}
	if cls.SlotUtilization >= 1.0 {
		t.Errorf("expected SlotUtilization < 1.0, got %.4f", cls.SlotUtilization)
	}
	if cls.SlotUtilization <= 0 {
		t.Errorf("expected SlotUtilization > 0, got %.4f", cls.SlotUtilization)
	}
}

func TestSlabResetCounters(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 64 * 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	a, _ := sa.Alloc(100)
	sa.Free(a)
	sa.ResetCounters()

	for i := range sa.classes {
		if sa.classes[i].AllocCount.Load() != 0 {
			t.Errorf("class %d AllocCount not zero after reset", i)
		}
		if sa.classes[i].FreeCount_.Load() != 0 {
			t.Errorf("class %d FreeCount_ not zero after reset", i)
		}
		if sa.classes[i].TotalRequestBytes.Load() != 0 {
			t.Errorf("class %d TotalRequestBytes not zero after reset", i)
		}
	}
}

func TestModelAwareWeights_Dominant(t *testing.T) {
	perShardMem := uint64(32) * 1024 * 1024 * 1024
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: perShardMem,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	modelSlots, modelClassSize := sa.ModelClassCapacity(5242880)
	if modelClassSize != 5242880 {
		t.Fatalf("expected class size 5242880, got %d", modelClassSize)
	}

	modelBytes := modelSlots * modelClassSize
	fraction := float64(modelBytes) / float64(perShardMem)
	t.Logf("model class: %d slots × %d bytes = %.1f GB (%.1f%% of %.0f GB)",
		modelSlots, modelClassSize, float64(modelBytes)/(1<<30), fraction*100, float64(perShardMem)/(1<<30))

	if fraction < 0.50 {
		t.Errorf("model class got only %.1f%% of memory, expected >= 50%%", fraction*100)
	}
}

func TestModelAwareWeights_NoModel(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 1024 * 1024 * 1024,
		ModelPageBytes: 0,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	var largeSlots []uint64
	for i := 0; i < sa.NumClasses(); i++ {
		cls := sa.ClassInfo(i)
		if cls.Size >= 524288 {
			largeSlots = append(largeSlots, cls.Allocator.NumSlots())
		}
	}

	if len(largeSlots) < 2 {
		t.Skip("fewer than 2 large classes")
	}

	var minSlots, maxSlots uint64
	minSlots = largeSlots[0]
	maxSlots = largeSlots[0]
	for _, s := range largeSlots[1:] {
		if s < minSlots {
			minSlots = s
		}
		if s > maxSlots {
			maxSlots = s
		}
	}
	for i, s := range largeSlots {
		if s == 0 {
			t.Errorf("large class %d has 0 slots", i)
		}
	}
	t.Logf("large class slots: min=%d, max=%d (%d classes)", minSlots, maxSlots, len(largeSlots))
}

func TestModelClassCapacity(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 256 * 1024 * 1024,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	slots, classSize := sa.ModelClassCapacity(5242880)
	if classSize != 5242880 {
		t.Errorf("expected class size 5242880, got %d", classSize)
	}
	if slots == 0 {
		t.Error("expected non-zero slots for model class")
	}

	slots, classSize = sa.ModelClassCapacity(999999999)
	if slots != 0 || classSize != 0 {
		t.Errorf("expected (0,0) for oversized value, got (%d, %d)", slots, classSize)
	}

	slots, classSize = sa.ModelClassCapacity(100)
	if classSize < 100 {
		t.Errorf("expected class size >= 100, got %d", classSize)
	}
	if slots == 0 {
		t.Error("expected non-zero slots for small class")
	}
}

func TestMemoryMap(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 32 * 1024 * 1024 * 1024,
		ModelPageBytes: 5242880,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	out := MemoryMap(sa, MemoryMapConfig{
		NumShards:      16,
		ModelPageBytes: 5242880,
	})
	t.Log("\n" + out)

	for _, want := range []string{
		"slab memory map",
		"16 shards",
		"<64K",
		"64K-256K",
		"5.00 MB",
		"** model page",
		"* derivative",
		"class",
		"slots/shard",
		"%mem",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q", want)
		}
	}

	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "5.00 MB") && !strings.HasPrefix(line, "**") {
			if !strings.Contains(line, "model page") {
				t.Errorf("model class line should start with **: %s", line)
			}
		}
	}
}

func TestMemoryMapNoModel(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 1024 * 1024 * 1024,
		ModelPageBytes: 0,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	out := MemoryMap(sa, MemoryMapConfig{
		NumShards:      4,
		ModelPageBytes: 0,
	})
	t.Log("\n" + out)

	if !strings.Contains(out, "slab memory map") {
		t.Error("output missing header")
	}
	if !strings.Contains(out, "4 shards") {
		t.Error("output missing shard count")
	}
	if strings.Contains(out, "** model page") {
		t.Error("no-model output should not contain model legend")
	}
	if strings.Contains(out, "* derivative") {
		t.Error("no-model output should not contain derivative legend")
	}
}

func TestModelAwareWeights_MismatchNeighbor(t *testing.T) {
	perShardMem := uint64(32) * 1024 * 1024 * 1024
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: perShardMem,
		ModelPageBytes: 2560000,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	slots2M, classSize2M := sa.ModelClassCapacity(2031616)
	if classSize2M != 2097152 {
		t.Fatalf("expected 2031616 to land in class 2097152, got %d", classSize2M)
	}
	bytes2M := slots2M * classSize2M
	frac2M := float64(bytes2M) / float64(perShardMem)

	if frac2M < 0.05 {
		t.Errorf("2097152 class (neighbor) got only %.1f%% of memory, expected >= 5%%", frac2M*100)
	}

	slotsModel, classSizeModel := sa.ModelClassCapacity(2560000)
	if classSizeModel != 2560000 {
		t.Fatalf("expected class size 2560000, got %d", classSizeModel)
	}
	bytesModel := slotsModel * classSizeModel
	fracModel := float64(bytesModel) / float64(perShardMem)

	if fracModel < 0.50 {
		t.Errorf("model class got only %.1f%% of memory, expected >= 50%%", fracModel*100)
	}
}

func TestModelAwareWeights_NoNeighborWithoutModel(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 1024 * 1024 * 1024,
		ModelPageBytes: 0,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	weights := sa.CurrentWeights()
	for sz, w := range weights {
		if sz >= 524288 && w > 8.0 {
			t.Errorf("class %d has weight %.1f > 8.0 with no model configured", sz, w)
		}
	}
}

func TestSlabDedicatedMode(t *testing.T) {
	mpb := uint64(5242880)
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 256 * 1024 * 1024,
		ModelPageBytes: mpb,
		Dedicated:      true,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	if sa.NumClasses() != 2 {
		t.Fatalf("expected 2 classes in dedicated mode, got %d", sa.NumClasses())
	}
	c0 := sa.ClassInfo(0)
	c1 := sa.ClassInfo(1)
	if c0.Size != 256 {
		t.Errorf("expected first class = 256 bytes, got %d", c0.Size)
	}
	if c1.Size != mpb {
		t.Errorf("expected second class = %d bytes, got %d", mpb, c1.Size)
	}

	modelSlots := c1.Allocator.NumSlots()
	keySlots := c0.Allocator.NumSlots()
	modelBytes := modelSlots * c1.Size
	keyBytes := keySlots * c0.Size
	totalBytes := modelBytes + keyBytes
	modelFrac := float64(modelBytes) / float64(totalBytes)

	if modelFrac < 0.90 {
		t.Errorf("model class got only %.1f%% of memory, expected >= 90%%", modelFrac*100)
	}

	keyAlloc, err := sa.Alloc(100)
	if err != nil {
		t.Fatalf("Alloc(100) for key: %v", err)
	}
	if keyAlloc.ClassIdx != 0 {
		t.Errorf("key alloc landed in class %d, expected 0 (256-byte)", keyAlloc.ClassIdx)
	}

	valAlloc, err := sa.Alloc(mpb)
	if err != nil {
		t.Fatalf("Alloc(%d) for value: %v", mpb, err)
	}
	if valAlloc.ClassIdx != 1 {
		t.Errorf("value alloc landed in class %d, expected 1 (model class)", valAlloc.ClassIdx)
	}

	sa.Free(keyAlloc)
	sa.Free(valAlloc)
}

func TestSlabDedicatedMode_NoModelPageBytes(t *testing.T) {
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 64 * 1024 * 1024,
		ModelPageBytes: 0,
		Dedicated:      true,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	if sa.NumClasses() <= 2 {
		t.Errorf("expected many classes when Dedicated=true but ModelPageBytes=0, got %d", sa.NumClasses())
	}
}

func TestSlabDedicatedMode_Weights(t *testing.T) {
	mpb := uint64(5242880)
	sa, err := NewSlabAllocator(SlabConfig{
		MaxMemoryBytes: 256 * 1024 * 1024,
		ModelPageBytes: mpb,
		Dedicated:      true,
	})
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer sa.Close()

	weights := sa.CurrentWeights()
	if len(weights) != 2 {
		t.Fatalf("expected 2 weight entries, got %d", len(weights))
	}
	if w, ok := weights[mpb]; !ok || w != 95.0 {
		t.Errorf("model class weight: got %.1f, want 95.0", w)
	}
	if w, ok := weights[256]; !ok || w != 5.0 {
		t.Errorf("key class weight: got %.1f, want 5.0", w)
	}
}
