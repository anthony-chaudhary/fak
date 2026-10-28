package compute

// CUDA runtime attributes have signed 32-bit values. Widening to int64 retains
// errors/invalid values for checking; it does not convert VRAM or KiB to cache.
type cudaDeviceMemoryFacts struct {
	cacheBytes int64
	warpWidth  int64
	cacheOK    bool
	warpOK     bool
}

func cudaDeviceMemoryProfileFromFacts(architecture string, f cudaDeviceMemoryFacts) (DeviceMemoryProfile, bool) {
	p := DeviceMemoryProfile{Architecture: architecture}
	const maxAttribute = int64(1<<31 - 1)
	if f.cacheOK && f.cacheBytes > 0 && f.cacheBytes <= maxAttribute {
		p.CacheCapacityBytes = f.cacheBytes
		p.CacheCapacityKnown = true
	}
	if f.warpOK && f.warpWidth > 0 && f.warpWidth <= maxAttribute && uint64(f.warpWidth) <= uint64(^uint(0)>>1) {
		p.SubgroupWidth = int(f.warpWidth)
		p.SubgroupWidthKnown = true
	}
	// Neither cache capacity nor texture/pitch alignment proves a storage
	// alignment shared by every allocator route. Complete planning stays refused.
	return p, p.CacheCapacityKnown || p.SubgroupWidthKnown
}
