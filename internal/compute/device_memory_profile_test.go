package compute

import "testing"

type memoryProfileResolverBackend struct {
	Backend
	name string
	tier string
}

func (b memoryProfileResolverBackend) Name() string { return b.name }
func (b memoryProfileResolverBackend) Tier() string { return b.tier }

type memoryProfileResolverProvider struct {
	memoryProfileResolverBackend
	profile DeviceMemoryProfile
	known   bool
}

func (b memoryProfileResolverProvider) DeviceMemoryProfile() (DeviceMemoryProfile, bool) {
	return b.profile, b.known
}

// fak-test:runtime fast est=1ms lane=default
func TestResolveDeviceMemoryProfile(t *testing.T) {
	t.Parallel()
	halo := DeviceMemoryProfile{
		Architecture:          "gfx1151",
		CacheCapacityBytes:    32 * 1024 * 1024,
		StorageAlignmentBytes: 128,
		SubgroupWidth:         32,
		CacheCapacityKnown:    true,
		StorageAlignmentKnown: true,
		SubgroupWidthKnown:    true,
	}
	provided := DeviceMemoryProfile{
		Architecture:          "selected-device",
		CacheCapacityBytes:    4096,
		StorageAlignmentBytes: 256,
		SubgroupWidth:         64,
		CacheCapacityKnown:    true,
		StorageAlignmentKnown: true,
		SubgroupWidthKnown:    true,
	}
	partial := DeviceMemoryProfile{Architecture: "selected-device", SubgroupWidth: 32, SubgroupWidthKnown: true}
	invalid := provided
	invalid.CacheCapacityBytes = 0
	mi300 := DeviceMemoryProfile{Architecture: "gfx942", SubgroupWidth: 64, SubgroupWidthKnown: true}
	vulkan := func(tier string) Backend {
		return memoryProfileResolverBackend{name: "VuLkAn", tier: tier}
	}
	provider := func(profile DeviceMemoryProfile, known bool) Backend {
		return memoryProfileResolverProvider{
			memoryProfileResolverBackend: memoryProfileResolverBackend{name: "vulkan", tier: "gfx1151"},
			profile:                      profile,
			known:                        known,
		}
	}
	for _, tc := range []struct {
		name         string
		backend      Backend
		architecture string
		want         DeviceMemoryProfile
		wantKnown    bool
		wantPlanning bool
	}{
		{
			name: "provider overrides conflicting tier and hint", backend: provider(provided, true), architecture: "gfx1151",
			want: provided, wantKnown: true, wantPlanning: true,
		},
		{
			name: "partial provider is not filled from fallback", backend: provider(partial, true), architecture: "gfx1151",
			want: partial, wantKnown: true,
		},
		{
			name: "provider refusal preserves partial facts", backend: provider(partial, false), architecture: "gfx1151",
			want: partial,
		},
		{
			name: "unknown provider does not use known fallback", backend: provider(DeviceMemoryProfile{}, false), architecture: "gfx1151",
		},
		{
			name: "provider refusal is not inferred from fields", backend: provider(provided, false), architecture: "gfx1151",
			want: provided, wantPlanning: true,
		},
		{
			name: "known flags cannot admit nonpositive cache", backend: provider(invalid, true), architecture: "gfx1151",
			want: invalid, wantKnown: true,
		},
		{
			name: "nil backend uses qualified architecture", architecture: " GFX1151:sramecc+ ",
			want: halo, wantKnown: true, wantPlanning: true,
		},
		{
			name: "nil backend leaves unknown architecture unavailable", architecture: "future-gpu",
			want: DeviceMemoryProfile{Architecture: "future-gpu"},
		},
		{
			name: "non Vulkan architecture stays subgroup only", backend: memoryProfileResolverBackend{name: "rocm"}, architecture: "gfx942",
			want: mi300, wantKnown: true,
		},
		{
			name: "unknown CUDA architecture does not borrow tier", backend: memoryProfileResolverBackend{name: "cuda", tier: "gfx1151"}, architecture: "sm120",
			want: DeviceMemoryProfile{Architecture: "sm120"},
		},
		{
			name: "Vulkan bounded tier overrides conflicting hint", backend: vulkan("RADV (GFX1151)"), architecture: "gfx942",
			want: halo, wantKnown: true, wantPlanning: true,
		},
		{
			name: "Vulkan partial tier overrides Halo alias", backend: vulkan("gfx942 Radeon 8060S"), architecture: "gfx1151",
			want: mi300, wantKnown: true,
		},
		{
			name: "unknown Vulkan tier does not borrow hint", backend: vulkan("unknown-device"), architecture: "gfx1151",
		},
		{
			name: "unknown GFX token blocks Halo alias", backend: vulkan("gfx11510 Radeon 8060S"), architecture: "gfx1151",
		},
		{
			name: "Radeon 8060S alias remains qualified", backend: vulkan("AMD Radeon 8060S"), architecture: "sm120",
			want: halo, wantKnown: true, wantPlanning: true,
		},
		{
			name: "Radeon 8050S alias remains qualified", backend: vulkan("AMD Radeon 8050S"), architecture: "sm120",
			want: halo, wantKnown: true, wantPlanning: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, known := ResolveDeviceMemoryProfile(tc.backend, tc.architecture)
			if got != tc.want || known != tc.wantKnown {
				t.Fatalf("ResolveDeviceMemoryProfile = (%+v, %v), want (%+v, %v)", got, known, tc.want, tc.wantKnown)
			}
			if got.CachePlanningAvailable() != tc.wantPlanning {
				t.Fatalf("CachePlanningAvailable = %v, want %v for %+v", got.CachePlanningAvailable(), tc.wantPlanning, got)
			}
		})
	}
}

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
