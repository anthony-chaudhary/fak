package compute

import "testing"

func TestDeviceMemoryProfileDrivesScratchPolicy(t *testing.T) {
	halo, ok := LookupDeviceMemoryProfile("gfx1151")
	if !ok || !halo.CachePlanningAvailable() {
		t.Fatalf("gfx1151 profile unavailable: profile=%+v ok=%v", halo, ok)
	}
	if halo.CacheCapacityBytes != 32*1024*1024 || halo.StorageAlignmentBytes != 128 || halo.SubgroupWidth != 32 {
		t.Fatalf("gfx1151 memory facts = %+v, want 32 MiB cache, 128-byte alignment, subgroup 32", halo)
	}

	mi300, ok := LookupDeviceMemoryProfile("gfx942")
	if !ok {
		t.Fatal("gfx942 must remain a recognized device architecture")
	}
	if mi300.CacheCapacityKnown || mi300.StorageAlignmentKnown || !mi300.SubgroupWidthKnown || mi300.SubgroupWidth != 64 {
		t.Fatalf("gfx942 must preserve known Wave64 while leaving unmeasured memory facts unavailable: %+v", mi300)
	}
	unknown, ok := LookupDeviceMemoryProfile("unknown-vulkan")
	if ok || unknown.CachePlanningAvailable() {
		t.Fatalf("unknown Vulkan profile must fail closed: profile=%+v ok=%v", unknown, ok)
	}
	if tokens, available := DeviceCacheTileTokens(unknown, 8); available || tokens != 0 {
		t.Fatalf("unknown Vulkan tile policy = (%d, %v), want (0, false)", tokens, available)
	}
	if impostor, ok := LookupDeviceMemoryProfile("gfx11510"); ok || impostor.CachePlanningAvailable() {
		t.Fatalf("substring architecture impostor must not resolve as gfx1151: profile=%+v ok=%v", impostor, ok)
	}

	provided := DeviceMemoryProfile{
		Architecture:          "fixture-vulkan",
		CacheCapacityBytes:    4096,
		StorageAlignmentBytes: 256,
		SubgroupWidth:         64,
		CacheCapacityKnown:    true,
		StorageAlignmentKnown: true,
		SubgroupWidthKnown:    true,
	}
	if !provided.CachePlanningAvailable() {
		t.Fatalf("complete supplied backend profile rejected: %+v", provided)
	}
	if tokens, available := DeviceCacheTileTokens(provided, 8); !available || tokens != 512 {
		t.Fatalf("supplied profile tile policy = (%d, %v), want (512, true)", tokens, available)
	}
	haloArch, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 must remain a recognized ROCm architecture")
	}
	providedQSA := haloArch.TuneQSASparseGatherWithProfile(provided, 256, 2, 2)
	if providedQSA.RadixLDSBytes != 2048 || providedQSA.MaxWavesPerCU != 16 {
		t.Fatalf("QSA ignored supplied subgroup 64: radixLDS=%d maxWaves=%d", providedQSA.RadixLDSBytes, providedQSA.MaxWavesPerCU)
	}
	qsaBoundary := DeviceMemoryProfile{
		Architecture:          "gfx1151",
		CacheCapacityBytes:    5000,
		StorageAlignmentBytes: 1024,
		SubgroupWidth:         64,
		CacheCapacityKnown:    true,
		StorageAlignmentKnown: true,
		SubgroupWidthKnown:    true,
	}
	if cfg := haloArch.TuneQSASparseGatherWithProfile(qsaBoundary, 1, 1, 1); !cfg.CachePlanningAvailable || cfg.GatherScratchBytes != 6144 || cfg.FitsInInfinityCache {
		t.Fatalf("QSA ignored separate K/V alignment at cache boundary: %+v", cfg)
	}
	if tokens, available := DeviceCacheTileTokens(provided, 4096); available || tokens != 0 {
		t.Fatalf("subgroup larger than cache tile must fail closed, got (%d, %v)", tokens, available)
	}
	nearBoundary := provided
	nearBoundary.CacheCapacityBytes = 1000
	nearBoundary.SubgroupWidth = 1
	if tokens, available := DeviceCacheTileTokens(nearBoundary, 8); !available || tokens != 64 {
		t.Fatalf("separately aligned K/V boundary = (%d, %v), want (64, true)", tokens, available)
	}
	for _, bytesPerToken := range []int64{0, -1, int64(^uint64(0) >> 1)} {
		if tokens, available := DeviceCacheTileTokens(provided, bytesPerToken); available || tokens != 0 {
			t.Errorf("invalid/overflow tile input %d = (%d, %v), want (0, false)", bytesPerToken, tokens, available)
		}
	}
}
