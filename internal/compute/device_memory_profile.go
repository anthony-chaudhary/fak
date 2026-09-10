package compute

import "strings"

// DeviceMemoryProfile contains only memory facts a backend has established for
// the selected device. A zero value is deliberately unusable for cache planning.
type DeviceMemoryProfile struct {
	Architecture          string `json:"architecture"`
	CacheCapacityBytes    int64  `json:"cache_capacity_bytes"`
	StorageAlignmentBytes int64  `json:"storage_alignment_bytes"`
	SubgroupWidth         int    `json:"subgroup_width"`
	CacheCapacityKnown    bool   `json:"cache_capacity_known"`
	StorageAlignmentKnown bool   `json:"storage_alignment_known"`
	SubgroupWidthKnown    bool   `json:"subgroup_width_known"`
}

// CachePlanningAvailable reports whether cache-bounded scratch planning has
// enough genuine device information to produce a decision.
func (p DeviceMemoryProfile) CachePlanningAvailable() bool {
	return p.CacheCapacityKnown && p.CacheCapacityBytes > 0 &&
		p.StorageAlignmentKnown && p.StorageAlignmentBytes > 0 &&
		p.SubgroupWidthKnown && p.SubgroupWidth > 0
}

// DeviceMemoryProfileProvider is an optional Backend capability. Native
// backends should return facts discovered for their selected physical device.
type DeviceMemoryProfileProvider interface {
	DeviceMemoryProfile() (DeviceMemoryProfile, bool)
}

// LookupDeviceMemoryProfile returns only architecture facts that are stable
// across every device carrying that architecture identifier. Cache capacity
// and allocation alignment remain unavailable where the identifier alone does
// not establish them.
func LookupDeviceMemoryProfile(architecture string) (DeviceMemoryProfile, bool) {
	arch := normalizeGFX(strings.TrimSpace(architecture))
	switch arch {
	case "gfx1151":
		return DeviceMemoryProfile{
			Architecture:          arch,
			CacheCapacityBytes:    32 * 1024 * 1024,
			StorageAlignmentBytes: 128,
			SubgroupWidth:         32,
			CacheCapacityKnown:    true,
			StorageAlignmentKnown: true,
			SubgroupWidthKnown:    true,
		}, true
	case "gfx906", "gfx908", "gfx90a", "gfx942":
		return DeviceMemoryProfile{Architecture: arch, SubgroupWidth: 64, SubgroupWidthKnown: true}, true
	case "gfx1010", "gfx1030", "gfx1032", "gfx1100", "gfx1102":
		return DeviceMemoryProfile{Architecture: arch, SubgroupWidth: 32, SubgroupWidthKnown: true}, true
	default:
		return DeviceMemoryProfile{Architecture: arch}, false
	}
}

func resolveDeviceMemoryProfile(be Backend, architecture string) (DeviceMemoryProfile, bool) {
	if provider, ok := be.(DeviceMemoryProfileProvider); ok {
		return provider.DeviceMemoryProfile()
	}
	if be != nil && strings.EqualFold(be.Name(), "vulkan") {
		return DeviceMemoryProfileForVulkanTier(be.Tier())
	}
	return LookupDeviceMemoryProfile(architecture)
}

// DeviceMemoryProfileForVulkanTier resolves the bounded architecture token or
// established Strix Halo device name reported by Vulkan enumeration.
func DeviceMemoryProfileForVulkanTier(tier string) (DeviceMemoryProfile, bool) {
	detected := strings.ToLower(tier)
	tokens := strings.FieldsFunc(detected, func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
	for _, arch := range KnownROCmArches() {
		for _, token := range tokens {
			if token == arch.GFX {
				return LookupDeviceMemoryProfile(arch.GFX)
			}
		}
	}
	for _, token := range tokens {
		if strings.HasPrefix(token, "gfx") {
			return DeviceMemoryProfile{}, false
		}
	}
	if strings.Contains(detected, "radeon 8060s") || strings.Contains(detected, "radeon 8050s") {
		return LookupDeviceMemoryProfile("gfx1151")
	}
	return DeviceMemoryProfile{}, false
}

// DeviceCacheTileTokens returns the largest subgroup-aligned K/V tile admitted
// by an explicit cache profile. bytesPerToken is the combined raw K+V size;
// each equally sized buffer is aligned separately before cache accounting.
func DeviceCacheTileTokens(profile DeviceMemoryProfile, bytesPerToken int64) (int, bool) {
	if !profile.CachePlanningAvailable() || bytesPerToken <= 0 || bytesPerToken%2 != 0 {
		return 0, false
	}
	perBufferPerToken := bytesPerToken / 2
	perBufferBudget := profile.CacheCapacityBytes / 2
	maxAlignedPerBuffer := (perBufferBudget / profile.StorageAlignmentBytes) * profile.StorageAlignmentBytes
	quotient := maxAlignedPerBuffer / perBufferPerToken
	maxInt := int64(^uint(0) >> 1)
	if quotient > maxInt {
		return 0, false
	}
	tokens := int(quotient)
	tokens = (tokens / profile.SubgroupWidth) * profile.SubgroupWidth
	if tokens < profile.SubgroupWidth {
		return 0, false
	}
	return tokens, true
}

func checkedPositiveInt64Product(values ...int64) (int64, bool) {
	product := int64(1)
	for _, value := range values {
		if value <= 0 || product > (int64(^uint64(0)>>1))/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}

func alignInt64(value, alignment int64) (int64, bool) {
	if value <= 0 || alignment <= 0 || value > (int64(^uint64(0)>>1))-(alignment-1) {
		return 0, false
	}
	return ((value + alignment - 1) / alignment) * alignment, true
}

func checkedPositiveIntProduct(values ...int) (int, bool) {
	product := 1
	maxInt := int(^uint(0) >> 1)
	for _, value := range values {
		if value <= 0 || product > maxInt/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}
